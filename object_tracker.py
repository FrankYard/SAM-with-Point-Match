import torch
from torchvision.transforms.functional import crop
import numpy as np
from collections import defaultdict
from itertools import chain

from model.build_model import build_sam, build_pips
from model.superpoint.superpoint_v1 import SuperPoint
from model.superglue import SuperGlue
from utils.segment_utils import preprocess_data, remove_overlap, sample_points, get_crop_box
from fusion_graph import FusionGraph

def get_neighbor(vertex_id,tri):
    # get neighbor vertexes of a vertex
    helper = tri.vertex_neighbor_vertices
    index_pointers = helper[0]
    indices = helper[1]
    result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id+1]]

    return result_ids
def get_adj(points, tri):
    adj = np.zeros((points.shape[0], points.shape[0]))

    for i in range(points.shape[0]):
        adj[i,get_neighbor(i,tri)] = 1

    return adj

class ObjectTracker:
    def __init__(self, configs) -> None:
        self.configs = configs

        if torch.cuda.is_available():
            self.device = torch.device("cuda") 
        else: 
            print('Using cpu for inferrence !')
            self.device = torch.device("cpu")

        ## others
        configs['num_gpu'] = [0]
        configs['public_model'] = 0
        # model
        with torch.no_grad():
            self.superpoint = SuperPoint(configs['model']['superpoint']).to(self.device).eval()
            if configs['mask_method'] in ['sam_with_gt']:
                self.mask_predictor = build_sam(configs)
            else:
                raise NotImplementedError

            if configs['track_method'] == 'superglue':
                self.superglue = SuperGlue(configs['model']['superglue']).to(self.device).eval()
                self.num_ref = configs['num_ref']
            elif configs['track_method'] == 'pips':
                self.pips = build_pips(configs)
                self.pips_step = 8 # length of img sequence at each inference of pips
                self.pips_head = 0 # index of the firt image frame that is not completely tracked 

        self.match_object_dict = {} # global data for object match

        self.max_describe_input = 9
        
        self.track_object_dict = {}
        self.track_frame_thr = configs['track_frame_thr']
        self.object_min_points = configs['object_min_points']
        self.prompt_points_num = configs['prompt_points_num']
        self.neg_points_num = configs['neg_points_num']
        self.iters = configs['iters']
        self.resample = configs['resample']

        self.crop_scale_factor = 2
        self.crop_normal = configs['data']['normal_size']
        self.box_padding = configs['data']['box_padding']
        self.frame_detections = []
        self.instacnes_2_classes = [] # len(self.instacnes_2_classes) == self.instance_count

        self.fusion_graphs = {}

        # self.points_yx_int = []
        self.match_buffer = []
        self.images = []
        self.frame_object_images = {}

        self.img_count = 0
        self.instance_count = 0

    def add_img(self, image_raw):
        self.images.append(image_raw)
        if len(self.images) > 10:
            self.images = self.images[-10:]

    def init_boxes(self, new_masks=None):
        """new_masks.shape == (num, 1, h, w)
        """
        instance_ids_for_match = []
        boxes = []
        for instance_id, data_dict in self.track_object_dict.items():
            frame_interval =  self.img_count - data_dict['frame_ids'][-1]
            if frame_interval < self.track_frame_thr:
                instance_ids_for_match.append(instance_id)
                past_boxes = data_dict['boxes'][-self.track_frame_thr:]
                past_boxes = np.stack(past_boxes, axis=0)
                past_sizes = data_dict['box_sizes'][-self.track_frame_thr:]
                past_sizes = np.stack(past_sizes, axis=0)
                box_center = (past_boxes[-1][:2] + past_boxes[-1][2:4]) // 2
                box_size = np.max(past_sizes, axis=0)
                box = np.concatenate([box_center, box_center + box_size // 2])
                boxes.append(box)
        if new_masks is not None:
            for mask in new_masks:
                inds = np.array(np.where(mask[0]))
                y1, x1 = np.amin(inds, axis=1)
                y2, x2 = np.amax(inds, axis=1)
                boxes.append(np.array([x1, y1, x2, y2]))
        return instance_ids_for_match, boxes

    def object_crop(self, image, boxes):
        """image: (c, h, w)
        """
        _, h, w = image.shape
        crop_toplefts, crop_scales = get_crop_box(boxes, self.crop_scale_factor, self.box_padding, self.crop_normal, (h, w))
        object_imgs = [crop(image, *crop_topleft, *crop_scale) 
                       for crop_topleft, crop_scale in zip(crop_toplefts, crop_scales)]
        return object_imgs, crop_toplefts, crop_scales
    
    def get_reference_frames(self, instance_ids_for_match, num_ref):
        reference_frames = []
        for instance_id in instance_ids_for_match:
            # len_buffer = len(self.track_object_dict[instance_id]['frame_ids'])
            # if len_buffer >= num_ref:
            #     reference_frames.append(list(range(len_buffer-num_ref, len_buffer)))
            # else:
            #     reference_frames.append(list(range(0, len_buffer)))
            frame_records = self.track_object_dict[instance_id]['frame_ids']
            reference_frames.append(frame_records[-num_ref : ])
        return reference_frames
    
    def match_object_points(self, superpoint_output, instance_ids_for_match, shape, reference_frames=None):
        if reference_frames is None:
            reference_frames = [[-1] for _ in instance_ids_for_match]
        matches_llist0, matches_llist1, conf_llist0, conf_llist1 = [], [], [], []
        for i, instance_id in enumerate(instance_ids_for_match):
            key_points_int, descriptors, scores = superpoint_output['keypoints'][i], superpoint_output['descriptors'][i], superpoint_output['scores'][i]
            object_data = self.track_object_dict[instance_id]
            frame_records : list = object_data['frame_ids']
            frames_sublist = reference_frames[i]

            matches_sublist0, matches_sublist1, conf_sublist0, conf_sublist1 = [], [], [], []
            for frame_id in frames_sublist:
                frame_index= frame_records.index(frame_id)
                data = {'keypoints0': object_data['croppd_points'][frame_index][None],
                        'keypoints1': key_points_int[None],
                        'descriptors0': object_data['descriptors'][frame_index][None],
                        'descriptors1': descriptors[None],
                        'scores0': object_data['scores'][frame_index][None],
                        'scores1': scores[None],
                        'image0_shape': shape,
                        'image1_shape': shape
                }
                pred = self.superglue(data)
                matches_sublist0.append(pred['matches0'][0].cpu().numpy())
                conf_sublist0.append(pred['matching_scores0'][0].cpu().numpy())
                matches_sublist1.append(pred['matches1'][0].cpu().numpy())
                conf_sublist1.append(pred['matching_scores1'][0].cpu().numpy())

            matches_llist0.append(matches_sublist0)
            conf_llist0.append(conf_sublist0)
            matches_llist1.append(matches_sublist1)
            conf_llist1.append(conf_sublist1)

        return matches_llist0, conf_llist0, matches_llist1, conf_llist1

    def update_track_data(self, detections, instance_ids_for_match, superpoint_output, restored_point_list,
                          crop_toplefts, crop_scales, new_masks=None):
        if detections is None:
            labels, masks = [], []
        else:
            labels, masks = detections[0]['labels'], detections[0]['masks']

        if new_masks is None:
            new_labels, new_masks = [], []
        else:
            new_labels = list(range(self.instance_count, self.instance_count + len(new_masks)))
            self.instance_count += len(new_masks)
            for label in new_labels:
                self.track_object_dict[label] = defaultdict(list)

        frame_object_data = defaultdict(list)
        instance_ids_all = instance_ids_for_match + new_labels
        mask_union = None
        for instance_id, mask in zip(chain(new_labels, labels), chain(new_masks, masks)):
            i = instance_ids_all.index(instance_id)
            restored_point = restored_point_list[i]
            cropped_keypoints, descriptors, scores = \
                superpoint_output['keypoints'][i], superpoint_output['descriptors'][i], superpoint_output['scores'][i]
            point_ids = torch.arange(len(cropped_keypoints))

            mask = mask[0] # shape: (1, h, w) -> (h, w)
            if mask_union is None:
                mask_union = mask
            else:
                overlap = mask_union & mask
                if overlap.sum() > 0:
                    mask ^= overlap
                    if mask.sum() == 0:
                        continue
                mask_union = mask_union | mask

            point_mask = mask[restored_point[:,0], restored_point[:,1]]
            point_keep = (point_mask == 1)
            restored_object_points = restored_point[point_keep]
            # cropped_object_points = cropped_keypoints[point_keep]
            object_sp_descs = descriptors[:, point_keep] # (c, num)
            object_sp_scores = scores[point_keep]
            object_point_ids = point_ids[point_keep]

            # neg_points = points_int[point_keep==False]
            neg_point_ids = point_ids[point_keep==False]

            inds = np.array(np.where(mask))
            y1, x1 = np.amin(inds, axis=1)
            y2, x2 = np.amax(inds, axis=1)
            box = np.array([x1, y1, x2, y2])
            # 'cropped_neg_pids''cropped_obj_pids''scores''descriptors''croppd_points''boxes''frame_ids'
            self.track_object_dict[instance_id]['frame_ids'].append(self.img_count)
            self.track_object_dict[instance_id]['det_index'].append(i)

            self.track_object_dict[instance_id]['croppd_points'].append(cropped_keypoints)
            self.track_object_dict[instance_id]['descriptors'].append(descriptors)
            self.track_object_dict[instance_id]['scores'].append(scores)

            self.track_object_dict[instance_id]['cropped_obj_pids'].append(object_point_ids)
            self.track_object_dict[instance_id]['cropped_neg_pids'].append(neg_point_ids)
            self.track_object_dict[instance_id]['boxes'].append(box)
            self.track_object_dict[instance_id]['box_sizes'].append(np.array([x2-x1, y2-y1]))

            self.track_object_dict[instance_id]['obj_points'].append(restored_object_points)

            self.track_object_dict[instance_id]['points'].append(restored_point)
            self.track_object_dict[instance_id]['point_masks'].append(point_mask)
            self.track_object_dict[instance_id]['masks'].append(mask)

            frame_object_data['image_id'] = self.img_count
            frame_object_data['image_raw'] = self.images[-1]
            frame_object_data['obj_points'].append(restored_object_points)
            frame_object_data['descs'].append(object_sp_descs)
            frame_object_data['neg_pids'].append(neg_point_ids)
            frame_object_data['labels'].append(instance_id)
            frame_object_data['masks'].append(mask)
            cy1, cx1 = crop_toplefts[i]
            cy2, cx2 = crop_toplefts[i] + crop_scales[i]
            frame_object_data['boxes'].append(np.array([cx1, cy1, cx2, cy2]))
            frame_object_data['scores'].append(object_sp_scores.mean())
        return frame_object_data

    def graph_fusion(self, instance_ids_for_match, reference_frames, points_int_list, image_raw,
                     matches_llist0, conf_llist0, matches_llist1, conf_llist1):
        self.mask_predictor.set_image(image_raw)
        instance_list = []
        mask_list = []
        inliner_rates = []

        tracking_key_points, tracking_neg_points = [], []
        for i, instance_id in enumerate(instance_ids_for_match):
            graph = FusionGraph(instance_id)
            ref_frames = reference_frames[i]
            obj_data = self.track_object_dict[instance_id]
            frame_records = obj_data['frame_ids']

            yx_points_iter = (obj_data['points'][frame_records.index(ref)] for ref in ref_frames)
            masks_iter = (obj_data['masks'][frame_records.index(ref)] for ref in ref_frames)
            graph.add_fixed_frames(ref_frames, yx_points_iter, masks_iter)
            graph.add_superglue_data(ref_frames, self.img_count, points_int_list[i],
                                     matches_llist0[i], conf_llist0[i],  matches_llist1[i], conf_llist1[i])
            graph.init_label(self.img_count)
            pts, labels = graph.get_prompt(self.img_count, self.prompt_points_num, 0)
            if len(pts) < self.object_min_points:
                print('no enough positive points')
                continue
            masks, scores, logits = self.mask_predictor.predict(pts, labels, multimask_output=False)
            inliner_rate = graph.compute_inliner_rate(self.img_count, masks[0])
            out_mask = masks[0]
            input_logits = logits[0:1]
            for iter in range(self.iters-1):
                # mask = masks[0]
                if self.resample == True:
                    pts, labels = graph.get_prompt(self.img_count, self.prompt_points_num, self.neg_points_num)
                masks, scores, logits = self.mask_predictor.predict(pts, labels, mask_input=input_logits, multimask_output=False)
                new_inliner_rate = graph.compute_inliner_rate(self.img_count, masks[0])
                if new_inliner_rate > inliner_rate:
                    inliner_rate = new_inliner_rate
                    out_mask = masks[0]
                    input_logits = logits[0:1]
            if inliner_rate < 0.95:
                print('inliner rate of {} too low'.format(inliner_rate))
                continue

            inliner_rates.append(inliner_rate)
            instance_list.append(instance_id)
            mask_list.append(out_mask[None, None, :, :])
        
        if len(mask_list) == 0:
            return None
        
        inliner_rates, mask_list, instance_list = remove_overlap(inliner_rates, mask_list, instance_list)
        masks_array = np.concatenate(mask_list)
        point_labels = np.arange(len(mask_list))
        instance_ids = np.array(instance_list)
        detections = [{'masks': masks_array, 'point_labels': point_labels, 'labels':  instance_ids}] 
        return detections

    def visual_matches(self, instance_ids_for_match, superpoint_out, matches_llist, conf_llist, delay=-1):
        import matplotlib.cm as cm
        from utils.superglue_utils import make_matching_plot_fast
        import cv2
        if len(self.frame_object_images) > 1 and len(matches_llist) > 0:
            for i in range(len(matches_llist)):
                instance_index = instance_ids_for_match[i]
                matches = matches_llist[i][-1]
                conf = conf_llist[i][-1]
                prev_frame_id = self.track_object_dict[instance_index]['frame_ids'][-1]
                prev_det_index = self.track_object_dict[instance_index]['det_index'][-1]
                kpts0 = self.track_object_dict[instance_index]['croppd_points'][-1].cpu().numpy()
                kpts1 = superpoint_out['keypoints'][i].cpu().numpy()

                keep = matches > -1
                mkpts0 = kpts0[keep]
                mkpts1 = kpts1[matches[keep]]
                mconf = conf[keep]
                color = cm.jet(mconf)
                text = [
                    'frame:{}-{}'.format(prev_frame_id, self.img_count),
                    'instance: {}'.format(instance_index),
                    'Matches: {}'.format(len(mkpts0)),
                ]

                # Display extra parameter info.
                k_thresh = self.superpoint.config['keypoint_threshold']
                m_thresh = self.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                ]

                image0 = self.frame_object_images[prev_frame_id][prev_det_index][0]
                image1 = self.frame_object_images[self.img_count][i][0]
                image0 = (image0 * 255).cpu().numpy().astype(int)
                image1 = (image1 * 255).cpu().numpy().astype(int)

                make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path=None, show_keypoints=True, margin=10,
                                opencv_display=True, opencv_title='match'+str(instance_index), small_text=small_text,
                                delay=delay if i == len(matches_llist)-1 else 1)


    @torch.no_grad()
    def inference_crop(self, image, new_masks=None, masks_classes=None):
        instance_ids_for_match, init_boxes = self.init_boxes(new_masks)

        if len(init_boxes) == 0:
            self.img_count += 1
            return
        
        image_raw = image
        self.add_img(image_raw)
        image = torch.from_numpy(image).type(torch.float32).to(self.device)
        image = image.permute(2,0,1)
        image /= 255

        cropped_imgs, crop_toplefts, crop_scales = self.object_crop(image, init_boxes)

        batch = {'image' : cropped_imgs}
        object_images, sizes = preprocess_data(batch, self.configs['data'])
        object_images_gray = object_images.mean(dim=1, keepdim=True)
        superpoint_out = self.superpoint({'image' : object_images_gray}) 

        reference_frames = self.get_reference_frames(instance_ids_for_match, num_ref=self.num_ref)
        img_shape = object_images_gray[:1].shape
        matches_llist0, conf_llist0,  matches_llist1, conf_llist1 = \
            self.match_object_points(superpoint_out, instance_ids_for_match, img_shape, reference_frames)

        self.frame_object_images[self.img_count] = object_images_gray
        # if self.img_count >= 0:
        #     self.visual_matches(instance_ids_for_match, superpoint_out, matches_llist0, conf_llist0)

        points_int_list, descriptors_list = self.restore_points(superpoint_out, sizes, crop_toplefts)

        detections = self.graph_fusion(instance_ids_for_match, reference_frames, points_int_list, image_raw,
                          matches_llist0, conf_llist0,  matches_llist1, conf_llist1)

        single_frame_segments = self.update_track_data(detections, instance_ids_for_match, superpoint_out, 
                                                    points_int_list, crop_toplefts, crop_scales, new_masks)
        if len(single_frame_segments['masks']) == 0:
            multi_frame_segments = None
        else:
            multi_frame_segments = [single_frame_segments]

        if masks_classes is not None:
            self.instacnes_2_classes += masks_classes.tolist()
            
        point_out = [{'points': torch.concat(points_int_list), 'point_descs': torch.concat(descriptors_list)}]
        self.img_count += 1
        return point_out, multi_frame_segments
        
    @torch.no_grad()
    def generate_mask(self, image, prompt_points, labels):
        assert len(prompt_points) > 0, "empty prompt_points!"
        self.mask_predictor.set_image(image)
        masks, scores, logits = self.mask_predictor.predict(prompt_points, labels, multimask_output=False)
        for i in range(4-1):
            masks, score, logits = self.mask_predictor.predict(prompt_points, labels, mask_input=logits, multimask_output=False)

        return masks[:, None]

    @torch.no_grad()
    def inference_pips(self, image, new_masks=None, masks_classes=None, pips_iters=6, H_=360, W_=640):
        image_raw = image
        h, w, _ = image.shape
        self.add_img(image_raw)

        prepared_instances = []
        waiting_instances = []
        tracking_instances = []
        pips_init_points_list = []
        for instance_id, data in self.track_object_dict.items():
            # if data['lost'] is True:
            #     continue
            # data['frame_ids'].append(self.img_count)
            # if len(data['frame_ids']) % (self.pips_step - 1) != 1:
            #     continue
            # if len(data['frame_ids']) - len(data['pips_points']) != (self.pips_step - 1):
            #     continue
            frame_diff = self.img_count - data['last_frame']
            if frame_diff > (self.pips_step - 1):
                # track lost
                continue
            elif frame_diff < (self.pips_step - 1):
                # self.pips_head >= data['first_frame'] and 
                if self.pips_head <= data['last_frame']:
                # this instance is tracked ahead of pips head
                    prepared_instances.append(instance_id)
                elif self.pips_head < data['first_frame']:
                    waiting_instances.append(instance_id)
                continue

            tracking_instances.append(instance_id)
            pips_init_points_list.append(data['pips_init_points'][-1])

        # if len(tracking_instances) == 0:
            # frame_object_data = None
        track_length = [0] * len(tracking_instances) # the number of successfully tracked new images of each object
        if len(tracking_instances) > 0:
            instacne_points_num = self.prompt_points_num * 2
            # frame_object_data = defaultdict(list)

            # init data for pips tracking
            image_raws =  self.images[-self.pips_step : ]
            pips_init_points = torch.cat(pips_init_points_list, dim=-2).to(self.device)
            images = [torch.from_numpy(image).type(torch.float32).to(self.device).permute(2,0,1) 
                for image in image_raws]
            images = torch.stack(images)
            rgbs_ = torch.nn.functional.interpolate(images, (H_, W_), mode='bilinear')
            pips_init_points_ = pips_init_points[None]
            rgbs = rgbs_[None]

            # do pips tracking
            preds, preds_anim, vis_e, stats = self.pips(pips_init_points_, rgbs, iters=pips_iters)
            p_logit = vis_e[0].to('cpu')
            pred = preds[-1][0].to('cpu')
            assert pred.shape == (self.pips_step, instacne_points_num * len(tracking_instances), 2)

            # record point data and predict masks
            mask_new_list = []
            unhidden = np.zeros((self.pips_step - 1, len(tracking_instances)), dtype=bool)
            for t in range(1, self.pips_step):
                self.mask_predictor.set_image(image_raws[t])

                mask_list = []
                for i, ins_id in enumerate(tracking_instances):
                    start = i * instacne_points_num
                    pred_t = pred[t, start : start + instacne_points_num]
                    p_logit_t = p_logit[t, start : start + instacne_points_num]
                    check_unhidden = torch.sigmoid(p_logit_t) > 0.9
                    labels = check_unhidden.numpy()
                    labels[self.prompt_points_num :] = 0

                    # labels = np.ones(self.prompt_points_num) ignore negative points

                    track_valid = (labels.sum() >= self.object_min_points)

                    if track_valid:
                        track_length[i] = t

                    # self.track_object_dict[ins_id]['pips_points'].append(pred_t)
                    # self.track_object_dict[ins_id]['pips_logits'].append(p_logit_t)

                        prompt_points = pred_t.numpy() * np.array([[w / W_, h / H_]])
                        mask_, score, logits = self.mask_predictor.predict(prompt_points, labels, multimask_output=False)
                        for j in range(self.iters - 1):
                            mask_, score, logits = self.mask_predictor.predict(prompt_points, labels, mask_input=logits, multimask_output=False)
                    else:
                        mask_ = np.zeros((1, h, w), dtype=bool)
                    mask_list.append(mask_[None]) # (1,1,h,w) 
                    unhidden[t - 1, i] = track_valid

                masks_array = np.concatenate(mask_list)
                mask_new_list.append(masks_array)

            
            for i, ins_id in enumerate(tracking_instances):
                num_new_frame = track_length[i]
                if num_new_frame >= 1:
                    # use masks of the last successful image to resample prompt points
                    reinit_mask = mask_new_list[num_new_frame-1][i, 0]
                    pips_init_points = self.sampele_pips_points(reinit_mask, h, w, H_, W_)
                    self.track_object_dict[ins_id]['pips_init_points'].append(pips_init_points)
                    self.track_object_dict[ins_id]['pips_init_masks'].append(reinit_mask)
                    self.track_object_dict[ins_id]['pips_points'] += \
                        [pred[n, i * instacne_points_num : (i+1) * instacne_points_num] for n in range(num_new_frame)]
                    # record predicted mask from new frames 
                    self.track_object_dict[ins_id]['masks'] += [mask_new_list[n][i, 0] for n in range(num_new_frame)]
                    self.track_object_dict[ins_id]['unhidden'] += unhidden[:num_new_frame, i].tolist()
                    self.track_object_dict[ins_id]['last_frame'] += num_new_frame

                # record data of persent image frame
                frame_mask = mask_new_list[-1][i, 0]
                # frame_object_data['labels'].append(ins_id)
                # frame_object_data['masks'].append(frame_mask)
                inds = np.array(np.where(frame_mask))
                if inds.size > 0:
                    y1, x1 = np.amin(inds, axis=1)
                    y2, x2 = np.amax(inds, axis=1)
                    box = np.array([x1, y1, x2, y2])
                    score = 1.
                else:
                    box = np.zeros(4, dtype=int)
                    score = 0
                # frame_object_data['boxes'].append(box)
                # frame_object_data['scores'].append(score)

        if new_masks is not None:
            for mask_ in new_masks:
                mask = mask_[0]

                object_data = {}
                object_data['first_frame'] = self.img_count
                object_data['last_frame'] = self.img_count
                pips_init_points = self.sampele_pips_points(mask, h, w, H_, W_)                

                object_data['pips_points'] = [pips_init_points]
                # object_data['pips_logits'] = [torch.full(pips_points.shape[:1], 100)]
                
                object_data['pips_init_points'] = [pips_init_points]
                object_data['pips_init_masks'] = [mask]
                object_data['masks'] = [mask]
                object_data['unhidden'] = [True]

                self.track_object_dict[self.instance_count] = object_data
                self.instance_count += 1

            self.instacnes_2_classes += masks_classes.tolist()
        
        valid_instances = [ins_id for ins_id, length in zip(tracking_instances, track_length)
                           if length > 0]
        if len(tracking_instances) > 0:
            if len(prepared_instances + valid_instances) > 0:
                multi_frame_segments = self.get_pips_detection(prepared_instances, valid_instances, point_factor=torch.tensor([[h / H_, w / W_]]))
            # if detections is not None:
                # points_output = [{'points': pred[t].flip(1) * torch.tensor([[h / H_, w / W_]])} for t in range(0, len(detections))]
                points_output = [{'points': frame_segments['points']} for frame_segments in multi_frame_segments]
            else:
                # previous objects are all lost, reset pips head 
                # if len(waiting_instances) > 0:
                #     self.pips_head = min([self.track_object_dict[ins_id]['first_frame'] for ins_id in waiting_instances])
                multi_frame_segments = None
                points_output = [{'points': pred[-1].flip(1) * torch.tensor([[h / H_, w / W_]])}]

            out = points_output, multi_frame_segments
        else:

            out = None

        self.img_count += 1
        return out

    def sampele_pips_points(self, mask, h, w, H_, W_):
        positive_init_points = sample_points(mask, self.prompt_points_num) # tensor shape: (self.prompt_points_num, [x, y])

        inds = np.array(np.where(mask))
        y1, x1 = np.amin(inds, axis=1)
        y2, x2 = np.amax(inds, axis=1)
        box = np.array([x1, y1, x2, y2])
        crop_toplefts, crop_scales = get_crop_box([box], self.crop_scale_factor, 
                                                  self.box_padding, self.crop_normal, (h, w))
        y, x = crop_toplefts[0]
        hc, wc = crop_scales[0]
        crop_mask = np.zeros_like(mask)
        crop_mask[y:y+hc, x:x+wc] = True
        crop_mask ^= mask
        negative_init_points = sample_points(crop_mask, self.prompt_points_num)

        init_points = torch.cat([positive_init_points, negative_init_points], dim=0)
        pips_init_points = init_points * torch.tensor([[W_ / w, H_ / h]]) 
        return pips_init_points

    def get_pips_detection(self, prepared_instances, valid_instances, point_factor=None):
        """track_length:"""
        detections = []
        for frame_id in range(self.pips_head, self.img_count):
            trackable = True
            boxes, masks, scores, labels, points = [], [], [], [], []
            for i, instance_id in enumerate(chain(prepared_instances,valid_instances)):
                object_data = self.track_object_dict[instance_id]
                if object_data['last_frame'] < frame_id:
                    trackable = False
                    break
                    
                first_frame = object_data['first_frame']
                if frame_id < first_frame:
                    continue
                mask = object_data['masks'][frame_id - first_frame]
                obj_points = object_data['pips_points'][frame_id - first_frame].flip(1) * point_factor

                inds = np.array(np.where(mask))
                if inds.size > 0:
                    y1, x1 = np.amin(inds, axis=1)
                    y2, x2 = np.amax(inds, axis=1)
                    box = np.array([x1, y1, x2, y2])
                    score = 1.
                else:
                    continue
                
                masks.append(mask)
                boxes.append(box)
                scores.append(score)
                labels.append(instance_id)
                points.append(obj_points)
            
            if not trackable:
                break
            if len(points) == 0:
                # objects are either occluded or not present yet
                continue
            points = torch.cat(points)
            labels = np.array(labels)
            detections.append({'masks': masks, 'boxes': boxes, 'labels': labels, 
                                'scores': scores, 'image_id': frame_id, 'points': points,
                                'image_raw': self.images[-1 + frame_id - self.img_count],
                                })
        if len(detections) == 0:
            print('no detections!')
        if frame_id == self.pips_head and trackable == False:
            raise RuntimeError
        else:
            self.pips_head = frame_id + 1 - int(not trackable)
        return detections
    
    def restore_points(self, superpoint_output, sizes, crop_toplefts=None):
        list_len = len(superpoint_output['keypoints'])
        if crop_toplefts is not None:
            assert len(crop_toplefts) == list_len
        original_sizes = sizes['original_sizes']
        unified_new_size = sizes['unified_new_size']
        points_int_list, descriptors_list = [], []
        for i in range(list_len):
            key_points = superpoint_output['keypoints'][i].clone()
            scales = np.array(original_sizes[i], dtype=float) / np.array(unified_new_size)
            scale_y, scale_x = scales.tolist()
            key_points[:, 0] *= scale_x
            key_points[:, 1] *= scale_y
            points_int = key_points[:, [1, 0]].cpu().long()  # (num, [x,y]) -> (num, [y ,x])
            if crop_toplefts is not None:
                points_int += crop_toplefts[i]
            descriptors = superpoint_output['descriptors'][i].permute(1, 0) # (C, num) -> (num, C)

            points_int_list.append(points_int)
            descriptors_list.append(descriptors)
        return points_int_list, descriptors_list

