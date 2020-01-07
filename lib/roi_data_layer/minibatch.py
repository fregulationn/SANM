# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
import random
import scipy.sparse

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)

  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  # print("num iamges{}".format(num_images))

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    
    # im = Random_crop(roidb, im, i)
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

# def Random_crop(roidb, im, index):
#   image = im
#   annots = roidb[index]["boxes"]

#   if not annots.shape[0]:
#       return image
#   if random.choice([0, 1]):
#       return image
#   else:
#       rows, cols, cns = image.shape
#       flag = 0
#       while True:
#           flag += 1
#           if flag > 10:
#               return image

#           crop_ratio = random.uniform(0.5, 1)
#           rows_zero = int(rows * random.uniform(0, 1 - crop_ratio))
#           cols_zero = int(cols * random.uniform(0, 1 - crop_ratio))
#           crop_rows = int(rows * crop_ratio)
#           crop_cols = int(cols * crop_ratio)
#           '''
#           new_image = image[rows_zero:rows_zero+crop_rows, cols_zero:cols_zero+crop_cols, :]
#           new_image = cv2.resize(new_image, (cols, rows))
#           #new_image = skimage.transform.resize(new_image, (rows, cols))

#           new_annots = np.zeros((0, 5))
#           for i in range(annots.shape[0]):
#               x1 = max(annots[i, 0] - cols_zero, 0)
#               y1 = max(annots[i, 1] - rows_zero, 0)
#               x2 = min(annots[i, 2] - cols_zero, crop_cols)
#               y2 = min(annots[i, 3] - rows_zero, crop_rows)
#               label = annots[i, 4]
#               if x1 + 10 < x2 and y1 + 10 < y2:
#                   x1 /= crop_ratio
#                   y1 /= crop_ratio
#                   x2 /= crop_ratio
#                   y2 /= crop_ratio
#                   new_annots = np.append(new_annots, np.array([[x1, y1, x2, y2, label]]), axis=0)

#           if not new_annots.shape[0]:
#               continue
#           '''
#           new_image = np.zeros((rows , cols , cns))
#           new_image[rows_zero:rows_zero+crop_rows, cols_zero:cols_zero+crop_cols, :] = image[rows_zero:rows_zero+crop_rows, cols_zero:cols_zero+crop_cols, :]
#           im = new_image
          
#           # new_annots = np.zeros((0, 4))

#           NUM_CLASS = 2
#           new_annots = np.zeros((0, 4), dtype=np.uint16)
#           gt_classes = np.zeros((0), dtype=np.int32)
#           overlaps = np.zeros((0, NUM_CLASS), dtype=np.float32)
#           max_classes = np.zeros((0), dtype=np.int64)
#           max_overlaps = np.zeros((0), dtype=np.float32)
#           # seg_areas = np.zeros((0), dtype=np.float32)

#           for i in range(annots.shape[0]):
#               x1 = max(cols_zero, annots[i, 0])
#               y1 = max(rows_zero, annots[i, 1])
#               x2 = min(cols_zero+crop_cols, annots[i, 2])
#               y2 = min(rows_zero+crop_rows, annots[i, 3])
#               if x1+10 < x2 and y1+10 < y2:
#                   new_annots = np.append(new_annots, np.array([[x1,y1,x2,y2]]), axis=0)
#                   gt_classes = np.append(gt_classes,roidb[index]['gt_classes'][i])

#                   if roidb[index]['gt_overlaps'].data[i] <= 0:
#                     # Set overlap to -1 for all classes for crowd objects
#                     # so they will be excluded during training
#                     tmp_overlap = np.zeros((1, NUM_CLASS), dtype=np.float32)
#                     tmp_overlap[0,:] = -1.0
#                     overlaps = np.append(overlaps, tmp_overlap)
#                     # overlaps[ix, :] = -1.0
#                   else:
#                     tmp_overlap = np.zeros((1, NUM_CLASS), dtype=np.float32)
#                     tmp_overlap[0,gt_classes] = 1.0
#                     overlaps = np.append(overlaps, tmp_overlap)
                    
#                     # overlaps[ix, cls] = 1.0

#                   max_classes = np.append(max_classes, roidb[index]['max_classes'][i])
#                   max_overlaps = np.append(max_overlaps, roidb[index]['max_overlaps'][i])
          
#           if not new_annots.shape[0]:
#               continue
          
#           overlaps = scipy.sparse.csr_matrix(overlaps)
#           roidb[index]['boxes'] = new_annots
#           roidb[index]['gt_classes'] = gt_classes
#           roidb[index]['gt_overlaps'] = overlaps
#           roidb[index]['max_classes'] = max_classes
#           roidb[index]['max_overlaps'] = max_overlaps
#           roidb[index]['height'] = new_image.shape[0]
#           roidb[index]['width'] = new_image.shape[1]

#           return new_image