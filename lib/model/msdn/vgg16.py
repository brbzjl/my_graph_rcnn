# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.msdn.msdn import _MSDN
import pdb
import copy
from model.utils.config import cfg

class vgg16(_MSDN):
  def __init__(self, obj_classes, att_classes, rel_classes, d_top_feat, resume=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    vgg = models.vgg16()
    if not resume:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    base_net = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    self.dout_base_model = 512
    self.d_top_feat = d_top_feat
    _MSDN.__init__(self, base_net, obj_classes, att_classes, rel_classes, self.dout_base_model, self.d_top_feat)

    self.RCNN_top = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    if self.d_top_feat != 4096:
      self.RCNN_top_fc = nn.Linear(4096, self.d_top_feat)

    # NOTE: check whether this is hard copy
    self.RCNN_top_att = copy.deepcopy(self.RCNN_top)
    if self.d_top_feat != 4096:
      self.RCNN_top_att_fc = nn.Linear(4096, self.d_top_feat)

    # NOTE: check whether this is hard copy
    self.RCNN_top_rel = copy.deepcopy(self.RCNN_top)
    if self.d_top_feat != 4096:
      self.RCNN_top_rel_fc = nn.Linear(4096, self.d_top_feat)

    self.RCNN_bbox_pred = nn.Linear(self.d_top_feat, 4)

    self.RCNN_obj_cls_score = nn.Linear(self.d_top_feat, self.n_obj_classes)

    self.RCNN_att_cls_score = nn.Linear(self.d_top_feat, self.n_att_classes) \
      if cfg.HAS_ATTRIBUTES else None

    # for relation classification, we concatenate features of two rois
    self.RCNN_rel_cls_score = nn.Linear(self.d_top_feat, self.n_rel_classes) \
      if cfg.HAS_RELATIONS else None

  def _init_modules(self):
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base_model[layer].parameters(): p.requires_grad = False

  def _head_to_tail(self, pool5):    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)
    if self.d_top_feat != 4096:
      fc7 = F.relu(self.RCNN_top_fc(fc7))
    return fc7

  def _head_to_tail_att(self, pool5):  
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top_att(pool5_flat)
    if self.d_top_feat != 4096:
      fc7 = F.relu(self.RCNN_top_att_fc(fc7))
    return fc7

  def _head_to_tail_rel(self, pool5):    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top_rel(pool5_flat)
    if self.d_top_feat != 4096:
      fc7 = F.relu(self.RCNN_top_rel_fc(fc7))
    return fc7