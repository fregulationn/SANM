from __future__ import absolute_import
# --------------------------------------------------------
# Spatial Attention Network withFeature Mimicking
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified Modified by Junjie Zhang
# -------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from model.rpn.generate_anchors import generate_anchors
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from .bbox.bbox_transform import bbox_pred, clip_boxes, bbox_overlaps
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
import pdb

DEBUG = False


class _DCRProposalLayer(nn.Module):
    def __init__(self, class_agnostic):
        super(_DCRProposalLayer, self).__init__()
        self.class_agnostic = class_agnostic
        self._top = cfg.DCR.TOP

    def forward(self, rois, cls_prob, bbox_pred_tensor, im_info):
        num_keep_index = int(rois.shape[0] * self._top)


        rois = rois[0].cpu().detach().numpy()[:, 1:]
        bbox_deltas = bbox_pred_tensor.cpu().detach().numpy()[:, 4:8]
        im_info = im_info.cpu().detach().numpy()[0, :]
        cls_prob =  cls_prob.cpu().detach().numpy()[:, 1:]  # ignore bg

        
        # sort scores
        max_scores = np.amax(cls_prob, axis=1)
        # keep top scores
        keep_index = np.argsort(-max_scores)[:num_keep_index]

        proposals = bbox_pred(rois, bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return blob[keep_index, :], keep_index
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass