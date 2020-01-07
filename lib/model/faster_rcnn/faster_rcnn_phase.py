import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool
from model.dcr.dcr_layer import _DCRProposalLayer
from model.dcr.dcr_target_layer import _DCRTargetLayer
from model.faster_rcnn.losses import Attention_loss,CenterLoss

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        self.DCR_proposal = _DCRProposalLayer(self.class_agnostic)
        self.DCR_target_proposal = _DCRTargetLayer(self.n_classes, self.class_agnostic)
        # self.DCR_roi_pool = ROIPool((112, 112), 1.0/4.0)
        self.DCR_roi_pool = ROIAlign((56, 56), 1.0/4.0, 0)
        #self.DCR_roi_pool = ROIPool((56, 56), 1.0/4.0)
        # self.DCR_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        # self.DCR_roi_pool = ROIPool((28, 28), 1.0/8.0)
        
        
        if cfg.TRAIN.ATTENTION_MODEL:
            self.levelattentionLoss = Attention_loss()
        # if cfg.DCR.DCR_NET:
            # self.centerloss = CenterLoss(num_classes=2, feat_dim=2048, use_gpu=True)

    # def forward(self, im_data, im_info, gt_boxes, num_boxes, extract_gt_boxes):
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # import visdom
        # vis = visdom.Visdom()

        # feed image data to base model to obtain base feature map
        base_share_feat = self.RCNN_base_share(im_data)
        # vis.heatmap(base_share_feat[0][0])

        
        base_feat_before = self.RCNN_base_before(base_share_feat)
        # base_feat = self.RCNN_base(base_feat_before)
        base_feat = self.RCNN_base(base_feat_before)
       
        
         # attention network
        if cfg.TRAIN.ATTENTION_MODEL:
            attention = self.ATTENTION_model(base_feat)
            base_feat = base_feat *torch.exp(attention)
            # base_feat = base_feat * attention
            
            # # Plot heatmap
            
            # vis.heatmap(attention[0][0])

        # import visdom
        # vis = visdom.Visdom()
        # vis.heatmap(base_feat[0][0])
        

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # rois = torch.cat((extract_gt_boxes,rois), 1)
        # # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        keep_index = None
        pos_index = None

        if self.training:
            if cfg.DCR.DCR_NET:
                
                if self.training:
                    dcr_rois, dcr_label = self.DCR_target_proposal(rois.view(-1, 5),cls_prob, bbox_pred, im_info, gt_boxes.view(-1, 5))

                    dcr_label = torch.from_numpy(dcr_label).long().cuda().view(-1)
                else:
                    dcr_rois, keep_index = self.DCR_proposal(rois,cls_prob, bbox_pred, im_info)


                # im = cv2.imread(imdb.image_path_at(i))
                # im2show = np.copy(im)
                # cls_scores = scores[:,1]
                # cls_boxes = pred_boxes
                # cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # im2show = vis_detections(im2show, imdb.classes[1], cls_dets.cpu().numpy(), 0.3)
                # cv2.imwrite('result_socres.png', im2show)


                dcr_rois = torch.from_numpy(dcr_rois).cuda().view(batch_size,-1,5)
                res2_roi_pool = self.DCR_roi_pool(base_share_feat,dcr_rois.view(-1, 5))
                # res2_roi_pool = self.DCR_roi_pool(base_feat_before,dcr_rois.view(-1, 5))

                dcr_feat = self.RCNN_base_before(res2_roi_pool)
                dcr_feat = self.RCNN_base(dcr_feat)    

                dcr_feat = self._head_to_tail(dcr_feat)
                # dcr_feat = self.DCR_BRANCH(res2_roi_pool)
                # dcr_feat = dcr_feat.view(dcr_feat.size(0), -1)
                dcr_fc1 = self.RCNN_dcr_score(dcr_feat)
                dcr_prob = F.softmax(dcr_fc1, 1)

                if cfg.DCR.MIMICK:
                    
                    # pos_index = (dcr_label==1).nonzero().view(1,-1)[0]
                    # dcr_rois_pos = torch.index_select(dcr_rois,1, pos_index)

                    if cfg.POOLING_MODE == 'align':
                        dcr_mimick_feat = self.RCNN_roi_align(base_feat, dcr_rois.view(-1, 5))
                    elif cfg.POOLING_MODE == 'pool':
                        dcr_mimick_feat = self.RCNN_roi_pool(base_feat, dcr_rois.view(-1,5))
                    mimick_feat = self._head_to_tail(dcr_mimick_feat)
                    
                    target = torch.ones(mimick_feat.size(0),1).cuda()

    
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_dcr = torch.tensor(0)
        RCNN_loss_mask = torch.tensor(0)
        RCNN_loss_mimick = torch.tensor(0)


        if self.training:
            
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            if cfg.DCR.DCR_NET:
                RCNN_loss_dcr = F.cross_entropy(dcr_fc1, dcr_label)
                if cfg.DCR.MIMICK:
                    # dcr_feat_pos = torch.index_select(dcr_feat, 0, pos_index)
                    RCNN_loss_mimick = F.cosine_embedding_loss(dcr_feat, mimick_feat, target)
                    #RCNN_loss_mimick = F.mse_loss(dcr_feat, mimick_feat)
                    
            
            if cfg.TRAIN.ATTENTION_MODEL:
                RCNN_loss_mask = self.levelattentionLoss(im_data.size(), attention, gt_boxes)




        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        # feature_roi = pooled_feat.view(batch_size, rois.size(1), -1)

        dcr_prob = torch.zeros([0])
        if self.training:
            if cfg.DCR.DCR_NET:
                dcr_prob = dcr_prob.view(batch_size, dcr_rois.size(1), -1)
                

        return rois, cls_prob, bbox_pred, dcr_prob, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,RCNN_loss_dcr, RCNN_loss_mask, RCNN_loss_mimick, rois_label,keep_index

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_dcr_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

        if cfg.TRAIN.ATTENTION_MODEL:
            normal_init(self.ATTENTION_model.conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.ATTENTION_model.conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.ATTENTION_model.conv3, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.ATTENTION_model.conv4, 0, 0.01, cfg.TRAIN.TRUNCATED)
            self.ATTENTION_model.conv5.weight.data.fill_(0)
            self.ATTENTION_model.conv5.bias.data.fill_(0)
            

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
