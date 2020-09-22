# -*- coding:utf-8 -*- #
# @Author  :wang dian

import pickle
import time
import torch
import math
import numpy as np
import math
import config as Cfg
# from shapely.geometry import Polygon
from spconv.spconv_utils import rotate_non_max_suppression_cpu


class Detection(object):
    def __init__(self):
        self.output_height = Cfg.HEIGHT
        self.output_width = Cfg.WIDTH
        self.num_class = Cfg.NUM_CLASS
        self.anchor_per_loc = self.num_class * 2
        self.anchors = self.get_anchors(Cfg.ANCHORS_PATH)
        self.batch_size = self.anchors.shape[0]
        self.post_center_range = Cfg.POST_CENTER_RANGE
        self._num_direction_bins = 2
        self._score_threshold = .3
        self._iou_threshold = .1

    def get_anchors(self, path):
        anchors = np.fromfile(path, dtype=np.float32)
        anchors = anchors.reshape(-1, self.output_height * self.output_width * self.anchor_per_loc, 7)
        return anchors

    def np_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def np_topk(self, matrix, n, axis=0):
        full_sort = np.argsort(-matrix, axis=axis)
        index = full_sort.take(np.arange(n), axis=axis)
        return matrix[index], index

    def limit_period(self, val, offset=0.5, period=np.pi):
        return val - np.floor(val / period + offset) * period

    def corners_nd(self, dims, origin=0.5):
        """generate relative box corners based on length per dim and
        origin point.

        Args:
            dims (float array, shape=[N, ndim]): array of length per dim
            origin (list or array or float): origin point relate to smallest point.

        Returns:
            float array, shape=[N, 2 ** ndim, ndim]: returned corners.
            point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
                (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
                where x0 < x1, y0 < y1, z0 < z1
        """
        ndim = int(dims.shape[1])
        corners_norm = np.stack(
            np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
            axis=1).astype(dims.dtype)
        if ndim == 2:
            # generate clockwise box corners
            corners_norm = corners_norm[[0, 1, 3, 2]]
        elif ndim == 3:
            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
        corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
            [1, 2 ** ndim, ndim])
        return corners

    def rotation_2d(self, points, angles):
        """rotation 2d points based on origin point clockwise when angle positive.
        Args:
            points (float array, shape=[N, point_size, 2]): points to be rotated.
            angles (float array, shape=[N]): rotation angle.
        Returns:
            float array: same shape as points
        """
        rot_sin = np.sin(angles)
        rot_cos = np.cos(angles)
        rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
        corners = np.einsum('aij,jka->aik', points, rot_mat_T)
        return corners

    def center_to_corner_box2d(self, centers, dims, angles=None, origin=0.5):
        """convert kitti locations, dimensions and angles to corners.
        format: center(xy), dims(xy), angles(clockwise when positive)

        Args:
            centers (float array, shape=[N, 2]): locations in kitti label file.
            dims (float array, shape=[N, 2]): dimensions in kitti label file.
            angles (float array, shape=[N]): rotation_y in kitti label file.

        Returns:
            [type]: [description]
        """
        # 'length' in kitti format is in x axis.
        # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
        # center in kitti format is [0.5, 1.0, 0.5] in xyz.
        corners = self.corners_nd(dims, origin=origin)
        # corners: [N, 4, 2]
        if angles is not None:
            corners = self.rotation_2d(corners, angles)
        corners += centers.reshape([-1, 1, 2])

        return corners

    def corner_to_standup_nd(self, boxes_corner):
        assert len(boxes_corner.shape) == 3
        standup_boxes = []
        standup_boxes.append(np.min(boxes_corner, axis=1))
        standup_boxes.append(np.max(boxes_corner, axis=1))
        return np.concatenate(standup_boxes, -1)

    def iou_jit(self, boxes, query_boxes, eps=1.0):
        """calculate box iou. note that jit version runs 2x faster than cython in
        my machine!
        Parameters
        ----------
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), dtype=boxes.dtype)
        for k in range(K):
            box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                        (query_boxes[k, 3] - query_boxes[k, 1] + eps))
            for n in range(N):
                iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                    boxes[n, 0], query_boxes[k, 0]) + eps)
                if iw > 0:
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                        boxes[n, 1], query_boxes[k, 1]) + eps)
                    if ih > 0:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0] + eps) *
                                (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                        overlaps[n, k] = iw * ih / ua
        return overlaps

    def rotate_nms(self, rbboxes, scores, pre_max_size=1000, post_max_size=100):
        if pre_max_size is not None:
            num_keeped_scores = scores.shape[0]
            pre_max_size = min(num_keeped_scores, pre_max_size)
            scores, indices = self.np_topk(scores, pre_max_size)
            rbboxes = rbboxes[indices]
        dets = np.concatenate([rbboxes, np.expand_dims(scores, -1)], axis=1)
        dets_np = dets
        if len(dets_np) == 0:
            keep = np.array([], dtype=np.int64)
        else:
            ret = np.array(self.rotate_nms_cc(dets_np, self._iou_threshold), dtype=np.int64)
            keep = ret[:post_max_size]
        if keep.shape[0] == 0:
            return np.zeros([0])
        if pre_max_size is not None:
            keep = np.array(keep)
            return indices[keep]
        else:
            return np.array(keep)

    def rotate_nms_cc(self, dets, thresh):
        scores = dets[:, 5]
        order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
        dets_corners = self.center_to_corner_box2d(dets[:, :2], dets[:, 2:4], dets[:, 4])

        dets_standup = self.corner_to_standup_nd(dets_corners)

        standup_iou = self.iou_jit(dets_standup, dets_standup, eps=0.0)
        # todo: rotate_non_max_suppression_cpu
        ret_iou = rotate_non_max_suppression_cpu(dets_corners, order, standup_iou, thresh)
        return ret_iou

    def second_box_decode(self, box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
        """box decode for VoxelNet in lidar
        Args:
            boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
            anchors ([N, 7] Tensor): anchor
        """
        box_ndim = anchors.shape[-1]
        cas, cts = [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
            if encode_angle_to_vector:
                xt, yt, zt, wt, lt, ht, rtx, rty, *cts = np.split(box_encodings, box_ndim, axis=-1)
            else:
                xt, yt, zt, wt, lt, ht, rt, *cts = np.split(box_encodings, box_ndim, axis=-1)
        else:
            xa, ya, za, wa, la, ha, ra = np.split(anchors, box_ndim, axis=-1)
            if encode_angle_to_vector:
                xt, yt, zt, wt, lt, ht, rtx, rty = np.split(box_encodings, box_ndim, axis=-1)
            else:
                xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, box_ndim, axis=-1)

        diagonal = np.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za  # -1
        if smooth_dim:  # FALSE
            lg = (lt + 1) * la
            wg = (wt + 1) * wa
            hg = (ht + 1) * ha
        else:
            lg = np.exp(lt) * la
            wg = np.exp(wt) * wa
            hg = np.exp(ht) * ha
        if encode_angle_to_vector:  # FALSE
            rax = np.cos(ra)
            ray = np.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = math.atan2(rgy, rgx)
        else:
            rg = rt + ra
        cgs = [t + a for t, a in zip(cts, cas)]
        decode = np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)
        return decode

    def process(self, box_preds, cls_preds, dir_cls_preds, format='NCHW'):
        if isinstance(box_preds, torch.FloatTensor):
            box_preds = box_preds.detach().numpy()
            cls_preds = cls_preds.detach().numpy()
            dir_cls_preds = dir_cls_preds.detach().numpy()
        if format.upper() == 'NCHW':
            box_preds = box_preds.reshape(1, self.anchor_per_loc, 7, self.output_height, self.output_width).\
                transpose(0, 1, 3, 4, 2)
            cls_preds = cls_preds.reshape(1, self.anchor_per_loc, self.num_class, self.output_height,
                                          self.output_width).transpose(0, 1, 3, 4, 2)

            dir_cls_preds = dir_cls_preds.reshape(1, self.anchor_per_loc, 2, self.output_height, self.output_width).\
                transpose(0, 1, 3, 4, 2)
        elif format.upper() == 'NHWC':
            box_preds = box_preds.reshape(1, self.output_height, self.output_width, self.anchor_per_loc, 7).\
                transpose(0, 3, 1, 2, 4)
            cls_preds = cls_preds.reshape(1, self.output_height, self.output_width, self.anchor_per_loc,
                                          self.num_class).transpose(0, 3, 1, 2, 4)

            dir_cls_preds = dir_cls_preds.reshape(1, self.output_height, self.output_width, self.anchor_per_loc, 2).\
                transpose(0, 3, 1, 2, 4)
        else:
            raise ValueError('data format is NCHW or NHWC')
        return box_preds, cls_preds, dir_cls_preds

    def generate_bbox(self, box_preds, cls_preds, dir_cls_preds):
        box_preds, cls_preds, dir_cls_preds = self.process(box_preds, cls_preds, dir_cls_preds)
        meta_list = [None] * self.batch_size
        batch_anchors = self.anchors.reshape(self.batch_size, -1, self.anchors.shape[-1])
        batch_anchors_mask = [None] * self.batch_size
        batch_box_preds = box_preds
        batch_cls_preds = cls_preds
        batch_box_preds = batch_box_preds.reshape(self.batch_size, -1, 7)
        num_class_with_bg = self.num_class
        # if not _encode_background_as_zeros:
        # if not True:
        #     num_class_with_bg = num_class + 1
        batch_cls_preds = batch_cls_preds.reshape(self.batch_size, -1, num_class_with_bg)
        batch_box_preds = self.second_box_decode(batch_box_preds, batch_anchors)
        # if _use_direction_classifier:

        batch_dir_preds = dir_cls_preds
        batch_dir_preds = batch_dir_preds.reshape(self.batch_size, -1, self._num_direction_bins)
        selected = []
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:  # none
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.astype(np.float32)
            cls_preds = cls_preds.astype(np.float32)
            # if _use_direction_classifier: # true
            if a_mask is not None:  # none
                dir_preds = dir_preds[a_mask]
            dir_labels = np.argmax(dir_preds, axis=-1)

            # this don't support softmax
            total_scores = self.np_sigmoid(cls_preds)

            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = np.zeros(
                    total_scores.shape[0],
                    dtype=np.int64)
            else:
                top_scores = np.max(
                    total_scores, axis=-1)
                top_labels = np.argmax(total_scores, axis=1)
            _nms_score_thresholds = self._score_threshold

            assert _nms_score_thresholds > 0.0
            top_scores_keep = top_scores >= _nms_score_thresholds  # remove boxes by score
            top_scores = top_scores[top_scores_keep]  # mask

            if top_scores.shape[0] != 0:  # the rest of boxes
                if _nms_score_thresholds > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]  # bev_nms 01346
                # the nms in 3d detection just remove overlap boxes.
                # ================rotate_nms(rbboxes, scores)=====================
                pre_max_size = 1000
                # post_max_size = 100
                num_keeped_scores = top_scores.shape[0]
                pre_max_size = min(num_keeped_scores, pre_max_size)
                scores, indices = self.np_topk(top_scores, pre_max_size)
                rbboxes = boxes_for_nms[indices]
                dets = np.concatenate([rbboxes, np.expand_dims(scores, -1)], axis=1)
                # dets_np = dets
                if len(dets) == 0:
                    dets_corners = np.array([], dtype=np.int64)
                else:
                    dets_corners = self.center_to_corner_box2d(dets[:, :2], dets[:, 2:4], dets[:, 4])

                return dets_corners
            else:
                return None

    def predict(self, box_preds, cls_preds, dir_cls_preds):
        box_preds, cls_preds, dir_cls_preds = self.process(box_preds, cls_preds, dir_cls_preds)
        meta_list = [None] * self.batch_size
        batch_anchors = self.anchors.reshape(self.batch_size, -1, self.anchors.shape[-1])
        batch_anchors_mask = [None] * self.batch_size
        batch_box_preds = box_preds
        batch_cls_preds = cls_preds
        batch_box_preds = batch_box_preds.reshape(self.batch_size, -1, 7)
        num_class_with_bg = self.num_class
        # if not _encode_background_as_zeros:
        # if not True:
        #     num_class_with_bg = num_class + 1
        batch_cls_preds = batch_cls_preds.reshape(self.batch_size, -1, num_class_with_bg)
        batch_box_preds = self.second_box_decode(batch_box_preds, batch_anchors)
        # if _use_direction_classifier:

        batch_dir_preds = dir_cls_preds
        batch_dir_preds = batch_dir_preds.reshape(self.batch_size, -1, self._num_direction_bins)

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:  # none
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.astype(np.float32)
            cls_preds = cls_preds.astype(np.float32)
            # if _use_direction_classifier: # true
            if a_mask is not None:  # none
                dir_preds = dir_preds[a_mask]
            dir_labels = np.argmax(dir_preds, axis=-1)

            # this don't support softmax
            total_scores = self.np_sigmoid(cls_preds)

            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = np.zeros(
                    total_scores.shape[0],
                    dtype=np.int64)
            else:
                top_scores = np.max(
                    total_scores, axis=-1)
                top_labels = np.argmax(total_scores, axis=1)
            _nms_score_thresholds = self._score_threshold

            assert _nms_score_thresholds > 0.0
            top_scores_keep = top_scores >= _nms_score_thresholds  # remove boxes by score
            top_scores = top_scores[top_scores_keep]  # mask
            if top_scores.shape[0] != 0:  # the rest of boxes
                if _nms_score_thresholds > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]  # bev_nms 01346
                # the nms in 3d detection just remove overlap boxes.
                selected = self.rotate_nms(boxes_for_nms, top_scores)
            else:
                selected = []

            selected_boxes = box_preds[selected]
            # if _use_direction_classifier:  # true
            selected_dir_labels = dir_labels[selected]  # selected dir
            selected_labels = top_labels[selected]  # selected class
            selected_scores = top_scores[selected]  # selected score

            # finally generate predictions.
            final_box_preds = np.array([])
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels

                box_preds.tofile("./final_out/final_box.bin")
                scores.tofile('./final_out/final_scores.bin')
                label_preds.tofile('./final_out/final_labels.bin')

                # if _use_direction_classifier:  # true
                if True:
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)  # =pi
                    _dir_offset = 0.0
                    _dir_limit_offset = 1.0
                    dir_rot = self.limit_period(box_preds[..., 6] - _dir_offset, _dir_limit_offset, period)

                    box_preds[..., 6] = dir_rot + _dir_offset + period * dir_labels  # dir_rot +0 or dir_rot+pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds

                # post_center_range = np.array([0.0000, -20.0000, -2.2000, 55.2000, 20.0000, 0.8000], np.float32)
                if self.post_center_range is not None:  # not none
                    mask = (final_box_preds[:, :3] >= self.post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= self.post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": final_labels[mask],
                        "metadata": meta,
                    }

                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": final_labels,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                if final_box_preds.shape[0] == 0:
                    print('*********************error***********************')
                predictions_dict = {
                    "box3d_lidar":
                        np.zeros([0, final_box_preds.shape[-1]],
                                 dtype=dtype, ),
                    "scores":
                        np.zeros([0], dtype=dtype),
                    "label_preds":
                        np.zeros([0], dtype=top_labels.dtype),
                    "metadata":
                        meta,
                }
            predictions_dicts.append(predictions_dict)
            # with open("result_box3d_lidar_scores_label_preds_metadata.pkl", 'wb') as f:
            #     pickle.dump(predictions_dicts, f)
        return predictions_dicts


if __name__ == '__main__':
    output_path = '../../detection/out/'
    box_preds = np.fromfile(output_path + 'output-output_box-int32-1_56_100_138-0000.bin', dtype=np.float32)
    cls_preds = np.fromfile(output_path + 'output-output_cls-int32-1_32_100_138-0000.bin', dtype=np.float32)
    dir_cls_preds = np.fromfile(output_path + 'output-output_dir_cls-int32-1_16_100_138-0000.bin', dtype=np.float32)

    detect = Detection()
    detect.predict(box_preds, cls_preds, dir_cls_preds)
    # dets_corners = detect.generate_bbox(box_preds, cls_preds, dir_cls_preds)
    # dets_corners.tofile('./final_out/dets_corners.bin')
    # print(dets_corners.shape)


