import numpy as np
import PIL.Image as Image
import os
import glob
import argparse
import yaml
import torch
import cv2
from pycocotools import mask as mask_utils
import subprocess

from object_tracker import ObjectTracker
from utils.visualize import overlay_objects, overlay_points
from trackeval.datasets import KittiMOTS

GT_FOLDER = '/my_data_root/KITTIMOTS'
TRACKERS_FOLDER = '/my_data_root/KITTIMOTS'
TRACKER_SUB_FOLDER = 'trackrcnn_kitti_mots_result'
SEQMAP_FILE_VAL = 'config/val.seqmap'
SEQMAP_FILE_VAL_PED = 'config/val_ped.seqmap'
GT_LOC_FORMAT = '{gt_folder}/instances_txt/{seq}.txt'

SEQNUMS = ['0002', '0006', '0007', '0008', '0010', '0013', '0014' , '0016', '0018']
SEQDIR_BASE = '/my_data_root/KITTIMOTS/data_tracking_image_2/training/image_02/'

EVAL_SCRIPT_PATH = '/my_code_root/TrackEval/scripts/run_kitti_mots.py'

dataconfig = KittiMOTS.get_default_dataset_config()
dataconfig.update({
    'GT_FOLDER': GT_FOLDER,
    'TRACKERS_FOLDER': TRACKERS_FOLDER,
    'TRACKERS_TO_EVAL': [],
    # 'TRACKERS_TO_EVAL': ['trackrcnn'],
    'TRACKER_SUB_FOLDER': TRACKER_SUB_FOLDER,
    'SEQMAP_FILE': SEQMAP_FILE_VAL,
    'GT_LOC_FORMAT': GT_LOC_FORMAT
})
dataset = KittiMOTS(dataconfig)

def initialize_masks(dataset, seq, is_gt=False, tracker='trackrcnn'):
    """retrun masks of instances' first appearences, listed in frame sequence. 
    Shape: (h, w, num_new_instances)
    """
    id_key = 'gt_ids' if is_gt else 'tracker_ids'
    id_det = 'gt_dets' if is_gt else 'tracker_dets'
    id_class = 'gt_classes' if is_gt else 'tracker_classes'
    raw_data = dataset._load_raw_file(tracker=tracker, seq=seq, is_gt=is_gt)
    init_masks = []
    init_classes = []
    object_ids = []
    for i, oid_array in enumerate(raw_data[id_key]):
        selector = []
        for j, oid in enumerate(oid_array):
            if oid not in object_ids:
                object_ids.append(oid)
                selector.append(j)
        if len(selector) != 0:
            det = raw_data[id_det][i]
            mask = mask_utils.decode(det)
            mask = mask[:, :, selector]
            init_masks.append(mask)
            init_classes.append(raw_data[id_class][i][selector])
        else:
            init_masks.append(None)
            init_classes.append(None)
    return init_masks, init_classes
    
def get_mots(frame_id, frame_data, instances_2_classes, mask_union_dict=None):
    instance_labels = frame_data['labels']
    masks = frame_data['masks'] # list[np.ndarray]
    assert len(masks) == len(instance_labels)
    if isinstance(instance_labels, torch.Tensor):
        instance_labels = instance_labels.numpy()

    assert masks[0].dtype == bool
    h, w = masks[0].shape
    if isinstance(masks[0], torch.Tensor):
        masks = (m.numpy() for m in masks)

    if 'scores' in frame_data:
        scores = frame_data['scores']
        masks = (m for m, s in zip(masks, scores) if s > 0)
        instance_labels = (l for l, s in zip(instance_labels, scores) if s > 0)

    if mask_union_dict is None or frame_id not in mask_union_dict:
        mask_union = np.zeros((h, w), dtype=bool)
    else:
        mask_union = mask_union_dict[frame_id]

    mots_str = ''
    for instance_id, mask in zip(instance_labels, masks):
        class_id = instances_2_classes[instance_id]
        h, w = mask.shape

        overlap = mask_union & mask
        if overlap.sum() > 0:
            mask ^= overlap
            if mask.max() == 0:
                continue
        mask_union |= mask
        
        mask_det = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        mask_coding = mask_det['counts'].decode('utf-8')
        mots_str += '{} {} {} {} {} {}\n'.format(frame_id, instance_id, class_id, h, w, mask_coding)
    
    if mask_union_dict is not None:
        mask_union_dict[frame_id] = mask_union
    return mots_str

def run(mapper : ObjectTracker, seqdir : str, mode, gt_prompt, dir_suffix='', out_root=''):
    np.random.seed(0)
    torch.random.manual_seed(0)
    seq = os.path.basename(seqdir)
    init_masks, init_classes = initialize_masks(dataset, seq, is_gt=gt_prompt)
    imgnames = sorted(glob.glob(os.path.join(seqdir, "*.png")))
    out_str = ''
    if mode == 'pips':
        mask_union_dict = {}
    else:
        mask_union_dict = None
    for img_id, imgpath in enumerate(imgnames):
        init_masks_frame, init_classes_frame = init_masks[img_id], init_classes[img_id]
        if init_masks_frame is not None:
            # init_masks_frame: uint8 -> bool, (h, w, num_instances) -> (num_instances, 1, h, w,)
            init_masks_frame = np.transpose(init_masks_frame[None].astype(bool), [3,0,1,2])

        image = np.array(Image.open(imgpath))
        assert image.shape[2] == 3
        image = image[:, :, [2,1,0]]
        if mode == 'superglue':
            out = mapper.inference_crop(image, init_masks_frame, init_classes_frame)
        if mode == 'pips':
            out = mapper.inference_pips(image, init_masks_frame, init_classes_frame)

        if out is not None:
            points_output, frame_segments_list = out
            if frame_segments_list is not None:
                for data in frame_segments_list:
                    mots_str = get_mots(data['image_id'], data, mapper.instacnes_2_classes, mask_union_dict)
                    out_str += mots_str
                image_list = [det['image_raw'].copy() for det in frame_segments_list]
                image_list = overlay_objects(image_list, frame_segments_list, None)
            else:
                image_list = [image.copy()]

            image_list = overlay_points(image_list, points_output)
            for t, image in enumerate(image_list):
                cv2.imshow(f'segments_{t}', image_list[t])
            if img_id >= 2000000:
                cv2.waitKey(-1)
            else:
                cv2.waitKey(1)

    file_name = f'{seq}.txt'
    init_mask = 'gt' if gt_prompt else 'trcnn'
    dir_name = f'out_{mode}_{init_mask}{dir_suffix}'

    dir_path = os.path.join(out_root, dir_name)
    if not os.path.exists(dir_path):      
        os.makedirs(dir_path)
    full_path = os.path.join(dir_path, file_name)
    with open(full_path, 'w') as f:
        f.write(out_str)

    return dir_name

def test_tracker(configs, suffix, group_name):
    for seqnum in SEQNUMS:
        seqdir = SEQDIR_BASE + seqnum
        out_root = os.path.join(TRACKERS_FOLDER, group_name)
        tracker = ObjectTracker(configs)
        out_dir_name = run(tracker, seqdir, mode=configs['track_method'], gt_prompt=configs['gt_prompt'], 
            dir_suffix=suffix, out_root=out_root)
        del tracker

def evaluate(group_name):
    group_path = os.path.join(TRACKERS_FOLDER, group_name)
    dir_names = os.listdir(group_path)
    out_dir_names = [name for name in dir_names if name.startswith('out_')]
    for out_dirname in out_dir_names:
        eval_dirname = 'eval_' + out_dirname
        argslist = ['python', EVAL_SCRIPT_PATH, '--USE_PARALLEL', 'True', '--METRICS', 'JAndF', 
                    '--GT_FOLDER', GT_FOLDER,
                    '--TRACKERS_FOLDER', TRACKERS_FOLDER,
                    '--TRACKERS_TO_EVAL', group_name,
                    '--TRACKER_SUB_FOLDER', out_dirname,
                    '--OUTPUT_SUB_FOLDER', eval_dirname,
                    '--GT_LOC_FORMAT', '{gt_folder}/instances_txt/{seq}.txt']
        subprocess.run(argslist + ['--SEQMAP_FILE', SEQMAP_FILE_VAL, '--CLASSES_TO_EVAL', 'car'])
        subprocess.run(argslist + ['--SEQMAP_FILE', SEQMAP_FILE_VAL_PED, '--CLASSES_TO_EVAL', 'pedestrian'])

def ablation(default_configs):
    assert default_configs['track_method'] == 'superglue'
    assert default_configs['gt_prompt'] == True
    group_name = 'ablation'

    print('Test traking box expasion:') 
    configs = default_configs.copy()
    for pading in [50, 100, 150, 200, 250, 300]:
        configs['data']['box_padding'] = [pading, pading]
        suffix = f'p{pading}'
        test_tracker(configs, suffix, group_name)

    print('Test sliding window length:')
    configs = default_configs.copy()
    for track_frame_thr in [2, 3, 4, 5, 6, 7]:
        configs['track_frame_thr'] = track_frame_thr
        suffix = f'tft{track_frame_thr}'
        test_tracker(configs, suffix, group_name)

    print('Test the number of reference frames:')
    configs = default_configs.copy()
    for num_ref in [1,2,3]:
        configs['num_ref'] = num_ref
        suffix = f'ref{num_ref}'
        test_tracker(configs, suffix, group_name)

    print('Test RANSAC with varied number of positive sample points :')
    configs = default_configs.copy()
    for ppts in [1,2,3,4,5,6,7,8,10,12,14,16,32]:
        configs['prompt_points_num'] = ppts
        suffix = f'ppts{ppts}'
        test_tracker(configs, suffix, group_name)

    print('Test Positive Only with varied number of positive sample points :')
    configs = default_configs.copy()
    for ppts in [1,2,3,4,5,6,7,8,10,12,14,16,32]:
        configs['prompt_points_num'] = ppts
        configs['neg_points_num'] = 0
        suffix = f'ppts{ppts}negpts0'
        test_tracker(configs, suffix, group_name)

    print('Test Fixed Sample with varied number of positive sample points :')
    configs = default_configs.copy()
    for ppts in [1,2,3,4,5,6,7,8,10,12,14,16,32]:
        configs['prompt_points_num'] = ppts
        configs['resample'] = False
        suffix = f'ppts{ppts}rspFalse'
        test_tracker(configs, suffix, group_name)

    print('Test Non-iterative with varied number of positive sample points :')
    configs = default_configs.copy()
    for ppts in [1,2,3,4,5,6,7,8,10,12,14,16,32]:
        configs['prompt_points_num'] = ppts
        configs['resample'] = False
        configs['iters'] = 1
        suffix = f'ppts{ppts}iter1'
        test_tracker(configs, suffix, group_name)

    print('Evaluating:')
    evaluate(group_name)

def compare_with_pips(default_configs):
    assert default_configs['gt_prompt'] == True
    group_name = 'compare'

    print('Test point matching method:') 
    configs = default_configs.copy()
    configs['track_method'] = 'superglue'
    configs['track_frame_thr'] = 7
    suffix = ''
    test_tracker(configs, suffix, group_name)

    print('Test point tracking method:')
    configs = default_configs.copy()
    configs['track_method'] = 'pips'
    suffix = ''
    test_tracker(configs, suffix, group_name)

    print('Evaluating:')
    evaluate(group_name)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 
        default = "config/default.yaml"
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int,
        default = 1
    )
    args = parser.parse_args()

    config_file = args.config_file
    print("Load config:", config_file)
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['use_gpu'] = args.gpu

    ablation(configs)
    # compare_with_pips(configs)


