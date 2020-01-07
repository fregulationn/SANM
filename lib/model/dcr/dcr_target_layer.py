# --------------------------------------------------------
# Spatial Attention Network withFeature Mimicking
# Copyright (c) 2018 University of Illinois
# Licensed under The Apache-2.0 License [see LICENSE for details]
#
# Reorganized and modified Modified by Junjie Zhang
# --------------------------------------------------------

from model.dcr.dcr import sample_rois_fg_bg, sample_rcnn, sample_rois_random,sample_rois_fg, sample_rois_fg_only,sample_rcnn

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox.bbox_transform import bbox_pred, clip_boxes, bbox_overlaps
import pdb


class _DCRTargetLayer(nn.Module):
    def __init__(self, nclasses,class_agnostic):
        super(_DCRTargetLayer, self).__init__()
        self.class_agnostic = class_agnostic
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = cfg.TRAIN.BBOX_NORMALIZE_MEANS
        self.BBOX_NORMALIZE_STDS = cfg.TRAIN.BBOX_NORMALIZE_STDS
        self.BBOX_INSIDE_WEIGHTS = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

    def forward(self, rois, cls_prob, bbox_pred_tensor, im_info, gt_boxes):

        rois = rois.cpu().detach().numpy()
        cls_prob = cls_prob.cpu().detach().numpy()

        if self.class_agnostic:
            bbox_deltas = bbox_pred_tensor.cpu().detach().numpy()
        else:
            fg_cls_prob = cls_prob[:, 1:]
            fg_cls_idx = np.argmax(fg_cls_prob, axis=1).astype(np.int)
            batch_idx_array = np.arange(fg_cls_idx.shape[0], dtype=np.int)
            # bbox_deltas = in_data[2].asnumpy()[batch_idx_array, fg_cls_idx * 4 : (fg_cls_idx+1) * 4]
            # in_data2 = in_data[2].asnumpy()
            in_data2 = bbox_pred_tensor.cpu().detach().numpy()
            bbox_deltas = np.hstack((in_data2[batch_idx_array, fg_cls_idx * 4].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4+1].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4+2].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4+3].reshape(-1, 1)))
        
        im_info = im_info.cpu().detach().numpy()[0, :]
        gt_boxes = gt_boxes.cpu().detach().numpy()
        gt_zeros = np.where(gt_boxes[:,2]<0.01)
        if len(gt_zeros[0]) > 0:
            gt_boxes = gt_boxes[:gt_zeros[0][0]]


        # post processing
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            bbox_deltas = bbox_deltas * np.array(self.BBOX_NORMALIZE_STDS) + np.array(self.BBOX_NORMALIZE_MEANS)

        # proposals = bbox_pred(rois[:, 1:], bbox_deltas)
        # proposals = clip_boxes(proposals, im_info[:2])
        proposals = bbox_pred(rois[:, 1:], bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])


        # only support single batch
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        # reassign label
        gt_classes = gt_boxes[:, -1].astype(np.int)
        overlaps = np.zeros((blob.shape[0], self._num_classes ), dtype=np.float32)
        # n boxes and k gt_boxes => n * k overlap
        gt_overlaps = bbox_overlaps(blob[:, 1:].astype(np.float), gt_boxes[:, :-1].astype(np.float))
        # for each box in n boxes, select only maximum overlap (must be greater than zero),the dimensions between gt_overlaps and overlaps are different，roi max classes predict roi's class
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

        roi_max_classes = overlaps.argmax(axis=1)
        roi_max_overlaps = overlaps.max(axis=1)
        # assign bg labels
        roi_max_classes[np.where(roi_max_overlaps < cfg.TRAIN.FG_THRESH)] = 0
        assert (roi_max_classes[np.where(roi_max_overlaps < cfg.TRAIN.FG_THRESH)] == 0).all()

        if  cfg.DCR.SAMPLE_PER_IMAGE == -1:
            return blob,roi_max_classes
            # self.assign(out_data[0], req[0], blob)
            # self.assign(out_data[1], req[1], roi_max_classes)
        else:
            # Include ground-truth boxes in the set of candidate rois
            batch_inds = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
            all_rois = np.vstack((np.hstack((batch_inds, gt_boxes[:, :-1])), blob))

            # gt boxes
            pred_classes = gt_boxes[:, -1]
            pred_scores = np.ones_like(pred_classes)
            max_classes = pred_classes.copy()
            max_overlaps = np.ones_like(max_classes)
            # predicted boxes
            roi_pred_classes = cls_prob.argmax(axis=1)
            roi_pred_scores = cls_prob.max(axis=1)

            roi_rec = {}
            roi_rec['pred_classes'] = np.append(pred_classes, roi_pred_classes)
            roi_rec['scores'] = np.append(pred_scores, roi_pred_scores)
            roi_rec['max_classes'] = np.append(max_classes, roi_max_classes)
            roi_rec['max_overlaps'] = np.append(max_overlaps, roi_max_overlaps)

            if cfg.DCR.SAMPLE == 'SANM':
                keep_indexes, pad_indexes = sample_rois_fg_bg(roi_rec, cfg, cfg.DCR.SAMPLE_PER_IMAGE)
                # keep_indexes, pad_indexes = sample_rois_fg_only(roi_rec, cfg, cfg.DCR.SAMPLE_PER_IMAGE)
            elif cfg.DCR.SAMPLE == 'RANDOM':
                keep_indexes, pad_indexes = sample_rois_random(roi_rec, cfg, cfg.DCR.SAMPLE_PER_IMAGE)
            elif cfg.DCR.SAMPLE == 'RCNN':
                keep_indexes, pad_indexes = sample_rcnn(roi_rec, cfg, cfg.DCR.SAMPLE_PER_IMAGE)
            else:
                raise ValueError('Undefined sampling method: %s' % cfg.DCR.SAMPLE)

            resampled_blob = np.vstack((all_rois[keep_indexes, :], all_rois[pad_indexes, :]))
            # assign bg classes
            assert (roi_rec['max_classes'][np.where(roi_rec['max_overlaps'] < cfg.TRAIN.FG_THRESH)] == 0).all()
            resampled_label = np.append(roi_rec['max_classes'][keep_indexes], 0*np.ones(len(pad_indexes)))

            return resampled_blob,resampled_label

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass
       