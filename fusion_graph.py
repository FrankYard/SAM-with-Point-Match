import numpy as np
from collections import defaultdict

class FusionGraph:
    def __init__(self, instance_id) -> None:
        self.instance_id = instance_id
        self.frame_ids = set()
        self.points = {}

        self.point_labels = {}

        self.fixed_frames = set()
        self.masks = {}

        self.edge_2_match_data = {}
        self.refs = defaultdict(set)

        self.predictor = None


    def add_fixed_frames(self, frame_ids, frame_yx_points_iter, masks_iter):
        for fid, pts, mask in zip(frame_ids, frame_yx_points_iter, masks_iter):
            self.frame_ids.add(fid)
            self.points[fid] = pts
            self.masks[fid] = mask

            self.fixed_frames.add(fid)
            self.point_labels[fid] = mask[pts[:,0], pts[:,1]].astype(np.uint8)

    def add_edges(self, edges, matches_iter, confs_iter):
        for edge, matches, confs in zip(edges, matches_iter, confs_iter):
            keep = matches > -1
            valid_matches = matches[keep]
            valid_confs = confs[keep]
            self.edge_2_match_data[edge] = (keep, valid_matches, valid_confs)
            self.refs[edge[1]].add(edge[0])

    def add_superglue_data(self, reference_frame_list, 
                           target_frame, frame_yx_points,
                           matches_llist0, conf_llist0,  
                           matches_llist1, conf_llist1):
        self.frame_ids.add(target_frame)
        self.points[target_frame] = frame_yx_points
        ref_tar_edges = ((ref, target_frame) for ref in reference_frame_list)
        tar_ref_edges = ((target_frame, ref) for ref in reference_frame_list)
        self.add_edges(ref_tar_edges, matches_llist0, conf_llist0)
        self.add_edges(tar_ref_edges, matches_llist1, conf_llist1)

    def init_label(self, frame_id):
        """all referenced frame_ids should have point labels
        """
        assert frame_id not in self.fixed_frames
        point_labels = np.full(len(self.points[frame_id]), -1, dtype=np.int8)
        point_scores = np.zeros(len(point_labels))
        for ref_id in self.refs[frame_id]:
            keep, valid_matches, valid_confs = self.edge_2_match_data[(ref_id, frame_id)]
            ref_labels = self.point_labels[ref_id]
            transferred_labels = ref_labels[keep]

            should_update = valid_confs > point_scores[valid_matches]
            point_scores[valid_matches] = np.where(should_update, valid_confs, point_scores[valid_matches])
            point_labels[valid_matches] = np.where(should_update, transferred_labels, point_labels[valid_matches])

        if frame_id not in self.point_labels:
            self.point_labels[frame_id] = point_labels
        else:
            updated = point_scores > 0
            self.point_labels[frame_id][updated] = point_labels[updated]

    def compute_inliner_rate(self, frame_id, mask):
        pts = self.points[frame_id]
        label = mask[pts[:,0], pts[:,1]]
        num_inliners = 0
        num_matches = 0
        for ref_id in self.refs[frame_id]:
            if ref_id >= frame_id:
                continue
            keep, valid_matches, valid_confs = self.edge_2_match_data[(ref_id, frame_id)]
            matched_ref_labels = self.point_labels[ref_id][keep]
            matched_frame_labels = label[valid_matches]
            inline_matches = (matched_ref_labels == matched_frame_labels).sum()
            num_inliners += inline_matches
            num_matches += len(matched_frame_labels)

        return num_inliners / num_matches

    def get_prompt(self, frmae_id, positive_sample_num, negative_sample_num):
        points = self.points[frmae_id]
        labels = self.point_labels[frmae_id]
        key_mask = labels == 1
        neg_mask = labels == 0
        pos_pts = points[key_mask][:, [1,0]].cpu().numpy() # y,x -> x,y
        neg_pts = points[neg_mask][:, [1,0]].cpu().numpy()

        if len(pos_pts) > positive_sample_num:
            filtered = np.random.permutation(len(pos_pts))[:positive_sample_num]
            pos_pts_filt = pos_pts[filtered]
        else:
            pos_pts_filt = pos_pts

        if len(neg_pts) > negative_sample_num:
            filtered = np.random.permutation(len(neg_pts))[:negative_sample_num]
            neg_pts_filt = neg_pts[filtered]
        else:
            neg_pts_filt = neg_pts

        pos_labels = np.ones(len(pos_pts_filt))
        neg_labels = np.zeros(len(neg_pts_filt))
        pts = np.concatenate([pos_pts_filt, neg_pts_filt])
        labels = np.concatenate([pos_labels, neg_labels])
        return pts, labels

