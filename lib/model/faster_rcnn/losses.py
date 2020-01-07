import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_loss(nn.Module):

    def forward(self, img_batch_shape, attention_mask, bboxs):

        h, w = img_batch_shape[2], img_batch_shape[3]

        mask_losses = []

        batch_size = bboxs.shape[0]
        for j in range(batch_size):

            bbox_annotation = bboxs[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            cond1 = torch.le(bbox_annotation[:, 0], w)
            cond2 = torch.le(bbox_annotation[:, 1], h)
            cond3 = torch.le(bbox_annotation[:, 2], w)
            cond4 = torch.le(bbox_annotation[:, 3], h)
            cond = cond1 * cond2 * cond3 * cond4

            bbox_annotation = bbox_annotation[cond, :]

            if bbox_annotation.shape[0] == 0:
                mask_losses.append(torch.tensor(0).float().cuda())
                continue

            mask_loss = []
            attention_map = attention_mask[j, 0, :, :]
            label_bbox_annotation = bbox_annotation.clone()
            attention_h, attention_w = attention_map.shape

            if label_bbox_annotation.shape[0]:
                label_bbox_annotation[:, 0] *= attention_w / w
                label_bbox_annotation[:, 1] *= attention_h / h
                label_bbox_annotation[:, 2] *= attention_w / w
                label_bbox_annotation[:, 3] *= attention_h / h

            mask_gt = torch.zeros(attention_map.shape)
            mask_gt = mask_gt.cuda()

            for i in range(label_bbox_annotation.shape[0]):

                x1_margin = int(label_bbox_annotation[i, 0]) + 1 - label_bbox_annotation[i, 0]
                y1_margin = int(label_bbox_annotation[i, 1]) + 1 - label_bbox_annotation[i, 1]
                x2_margin = label_bbox_annotation[i, 2] - int(label_bbox_annotation[i, 2])
                y2_margin = label_bbox_annotation[i, 3] - int(label_bbox_annotation[i, 3])

                x1 = max(int(label_bbox_annotation[i, 0]), 0)
                y1 = max(int(label_bbox_annotation[i, 1]), 0)
                x2 = min(math.ceil(label_bbox_annotation[i, 2]) + 1, attention_w)
                y2 = min(math.ceil(label_bbox_annotation[i, 3]) + 1, attention_h)

                mask_gt[y1:y2, x1:x2] = 1

                mask_gt[y1:y2, x1] = mask_gt[y1:y2, x1] * x1_margin
                mask_gt[y1:y2, x2-1] = mask_gt[y1:y2, x2-1] * x2_margin
                mask_gt[y1, x1:x2] = mask_gt[y1, x1:x2] * y1_margin
                mask_gt[y2 -1, x1:x2] = mask_gt[y2-1, x1:x2] * y2_margin

            mask_gt = mask_gt[mask_gt >= 0]
            mask_predict = attention_map[attention_map >= 0]

            # if mask_predict.size()[0] ==0 or mask_gt.size()[0] ==0:
            #     print("stop")
            #     print(mask_gt)
            #     print(attention_map)
            #     print("predict {}".format(mask_predict))

            # print("how atttention is {}".format(attention_mask[j, 0, :, :]))
            
            mask_loss.append(F.binary_cross_entropy(mask_predict, mask_gt))
            mask_losses.append(torch.stack(mask_loss).mean())

        return torch.stack(mask_losses).mean(dim=0, keepdim=True)


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=2, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
