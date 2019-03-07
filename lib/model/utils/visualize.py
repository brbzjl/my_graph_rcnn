from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import cPickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
# from model.faster_rcnn.faster_rcnn_cascade import _fasterRCNN
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.co_nms.co_nms_wrapper import co_nms
# from model.fast_rcnn.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.network import save_net, load_net, vis_detections, vis_det_att, vis_relations, vis_gt_relations, eval_relations_recall
from model.repn.bbox_transform import combine_box_pairs

def vis_dets(img, im_info, rois, bbox_pred, obj_cls_prob, imdb):

	pdb.set_trace()
	im2show = img.data.permute(2, 3, 1, 0).squeeze().cpu().numpy()
	im2show += cfg.PIXEL_MEANS
	thresh = 0.01
	boxes = rois[:, :, 1:5]
	if cfg.TEST.BBOX_REG:
	  # Apply bounding-box regression deltas
	  box_deltas = bbox_pred
	  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
	  # Optionally normalize targets by a precomputed mean and stdev
	        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
	                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
	        box_deltas = box_deltas.view(1, -1, 4)
	  pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
	  pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

	obj_scores = obj_cls_prob.squeeze()
	pred_boxes = pred_boxes.squeeze()

	for j in xrange(1, len(imdb._classes)):
	  	inds = torch.nonzero(obj_scores[:,j] > thresh).view(-1)
	  	# if there is det
	  	if inds.numel() > 0:
	  		obj_cls_scores = obj_scores[:,j][inds]
	  		_, order = torch.sort(obj_cls_scores, 0, True)
	  		cls_boxes = pred_boxes[inds, :]
	  		cls_dets = torch.cat((cls_boxes, obj_cls_scores), 1)
	  		cls_dets = cls_dets[order]
	  		keep = nms(cls_dets, cfg.TEST.NMS)
	  		cls_dets = cls_dets[keep.view(-1).long()]
	  		im2show = vis_detections(im2show, imdb._classes[j], cls_dets.cpu().numpy(), 0.2)
	# save image to disk
	cv2.imwrite("detections.jpg", im2show)