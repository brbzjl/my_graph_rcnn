# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
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
import h5py

import torchvision.transforms as transforms
import torch.nn.functional as F

import pdb

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader_ng import roibatchLoader # use roibatchloader without graph
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.co_nms.co_nms_wrapper import co_nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.network import save_net, load_net, vis_detections, vis_det_att, vis_relations, vis_gt_relations, eval_relations_recall
from model.repn.bbox_transform import combine_box_pairs
from model.graph_rcnn.vgg16 import vgg16
from model.graph_rcnn.resnet import resnet
from datasets.vg import vg
from model.utils.visualize import vis_dets

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--dataset', dest='dataset',    
                      help='evaluation dataset',
                      default='coco', type=str)
  parser.add_argument('--vg_dataset', dest='vg_dataset',
                      help='training dataset',
                      default='vg_bm', type=str)  
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='train', type=str)  
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="local",
                      nargs=argparse.REMAINDER)

  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple gpus or not',
                      action='store_true')
  parser.add_argument('--vis', dest='vis',
                      help='whether use multiple gpus or not',
                      action='store_true')

  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=4, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=6, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10000, type=int)

  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

MIN_BOXES = 36
MAX_BOXES = 36

MIN_REL_PAIRS = 64
MAX_REL_PAIRS = 64

args = parse_args()

print('Called with args:')
print(args)

vis = args.vis

is_oracle = False

imdb_vg_name = "minitrain"

if args.vg_dataset == "vg1":
    vg_version = "150-50-20"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
elif args.vg_dataset == "vg2":
    vg_version = "500-150-80"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
elif args.vg_dataset == "vg3":
    vg_version = "750-250-150"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
elif args.vg_dataset == "vg4":
    vg_version = "1750-700-450"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
elif args.vg_dataset == "vg5":
    vg_version = "1600-400-20"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']    
elif args.vg_dataset == "vg6":
    vg_version = "1600-400-400"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']        
elif args.vg_dataset == "vg_bm":
    vg_version = "150-50-50"
    vg_split = imdb_vg_name
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']

imdb_vg = vg(vg_version, vg_split)

def bbox_proposal_fast(obj_prob, att_prob, rois):

    batch_size = obj_prob.size(0)

    # get the top obj cls, excluded the background class.
    max_obj_prob, max_obj_clss = obj_prob[:, :, 1:].max(2)
    # get the top att cls, exclude the background class. 
    max_att_prob, max_att_clss = att_prob[:, :, 1:].max(2)
    # get the top rel cls, exlude the background class
    # max_rel_scores, max_rel_ind = rel_prob[:, :, 1:].max(2)

    # compute the final score, B x N
    obj_att_scores = max_obj_prob * max_att_prob

    # sort final scores
    obj_att_scores_sorted, order = torch.sort(obj_att_scores, 1, True)

    rois_pop = rois.new(batch_size, MIN_BOXES, rois.size(2)).zero_()     
    rois_pop_id = rois.new(batch_size, MIN_BOXES).long().zero_()

    # rel_pairs_pop = rel_pairs.new(batch_size, self.rel_num, rel_pairs.size(2))
    # rel_pairs_pop_id = rel_pairs.new(batch_size, self.rel_num)

    # pdb.set_trace()

    for i in range(batch_size):
        proposals_i = rois[i][order[i]][:, 1:]
        scores_i = obj_att_scores[i][order[i]].view(-1, 1)
        keep_idx_i = nms(torch.cat((proposals_i, scores_i), 1), 0.5)
        keep_idx_i = keep_idx_i.long().view(-1)            
        num_rois_pop = min(keep_idx_i.size(0), MIN_BOXES)
        rois_pop[i][:num_rois_pop] = rois[i][order[i][keep_idx_i[:num_rois_pop]]]
        rois_pop_id[i][:num_rois_pop] = order[i][keep_idx_i[:num_rois_pop]]

    return rois_pop, rois_pop_id

def bbox_proposal(obj_prob, att_prob, rois, conf_thresh=0.2, thresh=0.01):

    scores, clss = obj_prob[:,1:].max(1)
    clss  = clss.view(-1) + 1
    scores = scores.view(-1)

    max_conf = obj_prob.new(obj_prob.size(0)).zero_()
    max_index = obj_prob.new(obj_prob.size(0)).zero_().long()
    for obj_cls_ind in range(1, obj_prob.size(1)):            
        obj_cls_scores = obj_prob[:, obj_cls_ind]
        
        inds = torch.nonzero(obj_cls_scores > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            obj_cls_boxes_p = rois[inds]
            obj_cls_scores_p = obj_cls_scores[inds]

            _, order = torch.sort(obj_cls_scores_p, 0, True)
            cls_dets = torch.cat((obj_cls_boxes_p, obj_cls_scores_p), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, 0.5)
            keep = keep.view(-1).long()
            keep_idx = inds[order[keep]]
            tmp_idx = torch.nonzero(obj_cls_scores[keep_idx] > max_conf[keep_idx]).view(-1)
            if tmp_idx.numel() != 0:
                max_index[keep_idx[tmp_idx]] = obj_cls_ind
            
            max_conf[keep_idx] = torch.max(obj_cls_scores[keep_idx], max_conf[keep_idx])

    keep_boxes = (max_conf >= conf_thresh).nonzero().view(-1)
    num_boxes = keep_boxes.numel()
    if num_boxes < MIN_BOXES:
        _, order = torch.sort(max_conf, 0, True)
        keep_boxes = order[:MIN_BOXES]

        keep_clss = max_index[keep_boxes]
        all_clss = clss[keep_boxes]
        keep_clss[num_boxes:] = all_clss[num_boxes:]

        keep_scores = max_conf[keep_boxes]
        all_scores = scores[keep_boxes]
        keep_scores[num_boxes:] = all_scores[num_boxes:]

    elif num_boxes > MAX_BOXES:
        _, order = torch.sort(max_conf, 0, True)          
        keep_boxes = order[:MAX_BOXES]        

        keep_clss = max_index[keep_boxes]
        keep_scores = max_conf[keep_boxes]            

    else:
        keep_clss = max_index[keep_boxes]
        keep_scores = max_conf[keep_boxes]    

    return keep_boxes, keep_clss, keep_scores, num_boxes

def ext_feat_pred(model, im_data, im_info, gt_boxes, num_boxes):

    outs = model(im_data, im_info, gt_boxes, num_boxes, is_oracle)
    base_feat, rois, rel_pairs, bbox_pred, x_obj, x_att, x_rel, \
        obj_cls_prob, att_cls_prob, rel_cls_prob, \
        obj_cls_score, att_cls_score, rel_cls_score = outs

    ### extract and save global feature
    global_feat = model._head_to_tail(base_feat)
    np.save(os.path.join(dir_feat, str(img_id[0]) + '_glb'), global_feat.data.cpu().numpy())

    # return

    ### extract graph feature    
    # filter out the rois that 
    rois_pop_id, rois_pop_clss, rois_pop_scores, num_boxes = bbox_proposal(obj_cls_prob, att_cls_prob, rois)

    rois_pop = rois[:, rois_pop_id, :]
    x_obj_pop = x_obj[rois_pop_id]
    score_obj_pop = obj_cls_score[:, 1:][rois_pop_id]

    np.save(os.path.join(dir_feat, str(img_id[0]) + '_obj'), torch.cat((x_obj_pop, score_obj_pop), 1).cpu().numpy())

    x_att_pop = x_att[rois_pop_id]
    score_att_pop = att_cls_score[:, 1:][rois_pop_id]

    np.save(os.path.join(dir_feat, str(img_id[0]) + '_att'), torch.cat((x_att_pop, score_att_pop), 1).cpu().numpy())

    # get poped rel pairs according to rois_pop_id
    rois_pop_id_cpu = rois_pop_id.cpu()
    kept_rois = torch.zeros(rois[0].size(0))
    kept_rois_to_idx = torch.zeros(rois[0].size(0))

    kept_rois[rois_pop_id_cpu] = 1
    kept_rois_to_idx[rois_pop_id_cpu] = torch.arange(0, rois_pop_id_cpu.size(0))

    # find the top-N triplets
    rel_pairs = rel_pairs.cpu()
    sobj_inds = rel_pairs[0][:, 0]
    oobj_inds = rel_pairs[0][:, 1]
    rels_pop_id = (kept_rois[sobj_inds] + kept_rois[oobj_inds]).eq(2)

    # pdb.set_trace()

    if rels_pop_id.sum() > 0:
      rels_pop_id = rels_pop_id.nonzero().squeeze()
      rels_pop = rel_pairs[0][rels_pop_id]
      rels_pop[:, 0] = kept_rois_to_idx[rels_pop[:, 0]]
      rels_pop[:, 1] = kept_rois_to_idx[rels_pop[:, 1]]      
      x_rel_pop = x_rel.cpu()[rels_pop_id]
      rel_score_pop = rel_cls_score.cpu()[rels_pop_id]

      np.save(os.path.join(dir_feat, str(img_id[0]) + '_rel'), torch.cat((x_rel_pop, rel_score_pop), 1).numpy())

    np.savez(os.path.join(dir_meta, str(img_id[0])), rois_pop_clss=rois_pop_clss, rois_pop_score=rois_pop_scores, num_boxes=num_boxes, 
          num_obj_cls=obj_cls_prob.size(2)-1, num_att_cls=att_cls_prob.size(2)-1, num_rel_cls=rel_cls_prob.size(2)-1,
          gt_box=gt_box, ori_box=ori_box, info=info)

    if vis:
        vis_dets(im_data, im_info, rois_pop, bbox_pred[:, rois_pop_id, :], obj_cls_prob[:, rois_pop_id, :], imdb_vg)

def ext_feat_pred_hdf5(model, im_data, im_info, gt_boxes, num_boxes):
    # extract graph representations from image and save it into hdf5

    # pdb.set_trace()

    outs = model(im_data, im_info, gt_boxes, num_boxes, is_oracle)
    base_feat, rois, rel_pairs, bbox_pred, x_obj, x_att, x_rel, \
        obj_cls_prob, att_cls_prob, rel_cls_prob, \
        obj_cls_score, att_cls_score, rel_cls_score = outs
    ### extract and save global feature
    global_obj_feat = model._head_to_tail(base_feat)
    global_att_feat = model._head_to_tail_att(base_feat)
    # np.save(os.path.join(dir_feat, str(img_id[0]) + '_glb'), global_feat.data.cpu().numpy())

    # return

    ### extract graph feature    
    # filter out the rois that 
    # rois_pop_id, _, _, _ = bbox_proposal(obj_cls_prob[0], att_cls_prob[0], rois[0])
    _, rois_pop_id = bbox_proposal_fast(obj_cls_prob, att_cls_prob, rois)
    rois_pop_id = rois_pop_id.view(-1)

    rois_pop = rois[:, rois_pop_id, :]
    x_obj_pop = x_obj[rois_pop_id]
    score_obj_pop = obj_cls_score[:, 1:][rois_pop_id]

    # np.save(os.path.join(dir_feat, str(img_id[0]) + '_obj'), torch.cat((x_obj_pop, score_obj_pop), 1).cpu().numpy())
    obj_feat = torch.cat((x_obj_pop, rois_pop[0, :, 1:].contiguous()), 1).cpu().numpy()

    x_att_pop = x_att[rois_pop_id]
    score_att_pop = att_cls_score[:, 1:][rois_pop_id]

    # np.save(os.path.join(dir_feat, str(img_id[0]) + '_att'), torch.cat((x_att_pop, score_att_pop), 1).cpu().numpy())
    att_feat = x_att_pop.cpu().numpy()  

    # get poped rel pairs according to rois_pop_id
    rois_pop_id_cpu = rois_pop_id.cpu()
    kept_rois = torch.zeros(rois[0].size(0))
    kept_rois_to_idx = torch.zeros(rois[0].size(0))

    kept_rois[rois_pop_id_cpu] = 1
    kept_rois_to_idx[rois_pop_id_cpu] = torch.arange(0, rois_pop_id_cpu.size(0))

    # find the top-N triplets
    rel_pairs = rel_pairs.cpu()
    sobj_inds = rel_pairs[0][:, 0]
    oobj_inds = rel_pairs[0][:, 1]
    rels_pop_id = (kept_rois[sobj_inds] + kept_rois[oobj_inds]).eq(2)

    rel_feat = torch.zeros(MAX_REL_PAIRS, x_rel.size(1) + rel_pairs[0].size(1))
    # if rels_pop_id.sum() > 0:
    #   rels_pop_id = rels_pop_id.nonzero().squeeze()
    #   rels_pop = rel_pairs[0][rels_pop_id]
    #   rels_pop[:, 0] = kept_rois_to_idx[rels_pop[:, 0]]
    #   rels_pop[:, 1] = kept_rois_to_idx[rels_pop[:, 1]]      
    #   x_rel_pop = x_rel.cpu()[rels_pop_id]
    #   rel_score_pop = rel_cls_score.cpu()[rels_pop_id]

    #   all_feat = torch.cat((rels_pop.float(), x_rel_pop), 1)
    #   if all_feat.size(0) > MAX_REL_PAIRS:
    #     rel_feat = all_feat[:MAX_REL_PAIRS, :]
    #   elif all_feat.size(0) < MIN_REL_PAIRS:
    #     rel_feat[:all_feat.size(0), :] = all_feat


    # pdb.set_trace()
    # np.savez(os.path.join(dir_meta, str(img_id[0])), rois_pop_clss=rois_pop_clss, rois_pop_score=rois_pop_scores, num_boxes=num_boxes, 
    #       num_obj_cls=obj_cls_prob.size(2)-1, num_att_cls=att_cls_prob.size(2)-1, num_rel_cls=rel_cls_prob.size(2)-1,
    #       gt_box=gt_box, ori_box=ori_box, info=info)

    # pdb.set_trace()
    if vis:
        vis_dets(im_data, im_info, rois_pop, bbox_pred[:, rois_pop_id, :], obj_cls_prob[:, rois_pop_id, :], imdb_vg)

    return global_obj_feat.data.cpu().numpy(), global_att_feat.data.cpu().numpy(), obj_feat, att_feat, rel_feat.numpy()


def ext_feat_pred_hdf5_v2(model, im_data, im_info, gt_boxes, num_boxes):
    # extract graph representations from image and save it into hdf5

    base_feat = model.RCNN_base_model(im_data)    
    rois, rpn_loss_cls, rpn_loss_bbox = model.RCNN_rpn(base_feat, im_info.data, gt_boxes.data, num_boxes.data)

    valid = rois.sum(2).view(-1).nonzero().view(-1)
    rois = rois[:, valid, :]

    rois = Variable(rois)
    if cfg.POOLING_MODE == 'crop':
        grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        pooled_feat = model.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
        pooled_feat = model.RCNN_roi_align(base_feat, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
        pooled_feat = model.RCNN_roi_pool(base_feat, rois.view(-1,5))


    # feed pooled features to top model
    x_obj = model._head_to_tail(pooled_feat)  # (B x N) x D

    obj_cls_score = model.RCNN_obj_cls_score(x_obj)
    obj_cls_prob = F.softmax(obj_cls_score)
    bbox_pred = model.RCNN_bbox_pred(x_obj)

    # get attribute scores
    if cfg.SHARE_OBJ_ATT_FEATURE:
        x_att = x_obj
    else:
        x_att = model._head_to_tail_att(pooled_feat)  # (B x N) x D
    
    att_cls_score = model.RCNN_att_cls_score(x_att)
    att_cls_prob = F.softmax(att_cls_score)
    
    # filter rois first
    _, rois_pop_id = bbox_proposal_fast(obj_cls_prob.data.unsqueeze(0), att_cls_prob.data.unsqueeze(0), rois.data)
    rois_pop_id = rois_pop_id.view(-1)
    
    rois_pop = rois[:, rois_pop_id, :].data
    x_obj = x_obj[rois_pop_id]
    obj_cls_score = obj_cls_score[rois_pop_id]

    # get attribute features
    x_att = x_att[rois_pop_id]

    # propose relation between rois
    rel_feats = obj_cls_score.view(rois_pop.size(0), rois_pop.size(1), obj_cls_score.size(1))
    roi_rel_pairs, roi_pair_proposals, roi_rel_pairs_score, relpn_loss_cls = \
        model.RELPN_rpn(rois_pop, rel_feats, im_info.data, gt_boxes.data, num_boxes.data, False)

    valid = roi_rel_pairs.sum(2).view(-1).nonzero().view(-1)
    roi_rel_pairs = roi_rel_pairs[:, valid, :]
    roi_pair_proposals = roi_pair_proposals[:, valid, :]
    roi_rel_pairs_score = roi_rel_pairs_score[:, valid, :]

    size_per_batch = x_obj.size(0)
    roi_pair_proposals_v = roi_pair_proposals.view(-1, 2)
    ind_subject = roi_pair_proposals_v[:, 0]
    ind_object = roi_pair_proposals_v[:, 1]

    rois_pred = combine_box_pairs(roi_rel_pairs.view(-1, 9))
    rois_pred = Variable(rois_pred)

    # # do roi pooling based on predicted rois
    # pooled_pred_feat = self.RELPN_roi_pool(base_feat, rois_pred.view(-1,5))
    if cfg.POOLING_MODE == 'crop':
        grid_xy = _affine_grid_gen(rois_pred.view(-1, 5), base_feat.size()[2:], model.grid_size)
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        pooled_pred_feat = model.RELPN_roi_crop(base_feat, Variable(grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            pooled_pred_feat = F.max_pool2d(pooled_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
        pooled_pred_feat = model.RELPN_roi_align(base_feat, rois_pred.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
        pooled_pred_feat = model.RELPN_roi_pool(base_feat, rois_pred.view(-1,5))

    # # combine subject, object and relation feature tohether
    x_pred = model._head_to_tail_rel(pooled_pred_feat)

    ind_subject = roi_pair_proposals_v[:, 0]
    ind_object = roi_pair_proposals_v[:, 1]


    # pdb.set_trace()

    if cfg.GCN_ON_FEATS and cfg.GCN_LAYERS > 0:

        if cfg.GCN_HAS_ATTENTION:
            x_sobj = obj_cls_score[ind_subject]
            x_oobj = obj_cls_score[ind_object]
            attend_score = model.GRCNN_gcn_att1(x_sobj, x_oobj) # N_rel x 1
            attend_score = attend_score.view(1, x_pred.size(0))
        else:
            attend_score = Variable(x_pred.data.new(1, x_pred.size(0)).fill_(1))

        # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
        # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
        # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
        # NOTE exclude one roi feature from another.
        # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
        # cooccurence, such as bus are more frequently appear on the road.
        # assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"

        size_per_batch = int(x_obj.size(0))

        map_obj_att = torch.eye(x_obj.size(0)).type_as(x_obj.data)

        if cfg.MUTE_ATTRIBUTES:
            map_obj_att.zero_()
            x_att = x_att.detach()

        map_obj_att = Variable(map_obj_att)

        map_obj_obj = x_obj.data.new(x_obj.size(0), x_obj.size(0)).fill_(0.0)
        eye_mat = torch.eye(size_per_batch).type_as(x_obj.data)
        for i in range(batch_size):
            map_obj_obj[i * size_per_batch:(i + 1) * size_per_batch, i * size_per_batch:(i + 1) * size_per_batch].fill_(1.0)
            map_obj_obj[i * size_per_batch:(i + 1) * size_per_batch, i * size_per_batch:(i + 1) * size_per_batch] =\
                map_obj_obj[i * size_per_batch:(i + 1) * size_per_batch, i * size_per_batch:(i + 1) * size_per_batch]\
                - eye_mat

        map_obj_obj = Variable(map_obj_obj)

        map_sobj_rel = Variable(x_obj.data.new(x_obj.size(0), x_pred.size(0)).zero_())
        map_sobj_rel.scatter_(0, Variable(ind_subject.contiguous().view(1, x_pred.size(0))), attend_score)
        map_oobj_rel = Variable(x_obj.data.new(x_obj.size(0), x_pred.size(0)).zero_())
        map_oobj_rel.scatter_(0, Variable(ind_object.contiguous().view(1, x_pred.size(0))), attend_score)
        map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 2)

        if cfg.MUTE_RELATIONS:
            map_obj_rel.data.zero_()
            x_pred = x_pred.detach()

        # map_obj_rel = Variable(map_obj_rel)

        # x_obj = F.relu(self.fc4obj(x_obj))
        # x_att = F.relu(self.fc4att(x_att))
        # x_pred = F.relu(self.fc4rel(x_pred))

        for i in range(cfg.GCN_LAYERS):
            # pass graph representation to gcn
            x_obj, x_att, x_pred = model.GRCNN_gcn_feat(x_obj, x_att, x_pred, map_obj_att, map_obj_obj, map_obj_rel)

            # pdb.set_trace()
            # compute object classification loss
            obj_cls_score = model.RCNN_obj_cls_score(x_obj)
            obj_cls_prob = F.softmax(obj_cls_score)

            # compute attribute classification loss
            att_cls_score = model.RCNN_att_cls_score(x_att)
            att_cls_prob = F.softmax(att_cls_score)
            att_cls_log_prob = F.log_softmax(att_cls_score)

            # compute relation classifcation loss
            # x_sobj = x_obj[ind_subject]
            # x_oobj = x_obj[ind_object]
            rel_cls_score = model.RCNN_rel_cls_score(x_pred)
            rel_cls_prob = F.softmax(rel_cls_score)

    # pdb.set_trace()
    
    obj_feat = torch.cat((x_obj.data, rois_pop[0, :, 1:].contiguous()), 1).cpu()

    score_att_pop = att_cls_score[:, 1:]
    att_feat = x_att.data.cpu()

    rel_cls_prob[:, 0] = 0
    val, ind = rel_cls_prob.max(1)

    _, order_rel = torch.sort(val, 0, True)

    rel_feat = torch.zeros(MAX_REL_PAIRS, x_pred.size(1) + 2)
    rel_pop_id = order_rel[:MAX_REL_PAIRS].data

    all_feat = torch.cat((roi_pair_proposals_v[rel_pop_id].float().cpu(), x_pred[rel_pop_id].data.cpu()), 1)
    
    if all_feat.size(0) < MAX_REL_PAIRS:
      rel_feat[:all_feat.size(0), :] = all_feat
    else:
      rel_feat = all_feat

    # pdb.set_trace()

    if vis:
        vis_dets(im_data, im_info, rois_pop, bbox_pred.unsqueeze(0)[:, rois_pop_id, :].data, obj_cls_prob.unsqueeze(0).data, imdb_vg)

    global_obj_feat = model._head_to_tail(base_feat)
    global_att_feat = model._head_to_tail_att(base_feat)
    # pdb.set_trace()
    
    return global_obj_feat.data.cpu().numpy(), global_att_feat.data.cpu().numpy(), obj_feat.numpy(), att_feat.numpy(), rel_feat.numpy()

def ext_feat_pred_hdf5_batch(model, im_data, im_info, gt_boxes, num_boxes):
    # extract graph representations from image and save it into hdf5

    outs = model(im_data, im_info, gt_boxes, num_boxes)
    base_feat, rois, rel_pairs, bbox_pred, x_obj, x_att, x_rel, \
        obj_cls_prob, att_cls_prob, rel_cls_prob, \
        obj_cls_score, att_cls_score, rel_cls_score = outs

    batch_size = im_data.size(0)

    # pdb.set_trace()
    
    global_feat_batch = model._head_to_tail(base_feat)

    obj_feat_batch = []
    att_feat_batch = []
    rel_feat_batch = []

    offset = int(x_obj.size(0) / batch_size)
    
    x_rel = x_rel.cpu()
    rel_pairs = rel_pairs.cpu()

    for i in range(batch_size):
      ### extract and save global feature
      # np.save(os.path.join(dir_feat, str(img_id[0]) + '_glb'), global_feat.data.cpu().numpy())
      ### extract graph feature    
      # filter out the rois that 
      rois_pop_id, rois_pop_clss, rois_pop_scores, num_boxes = bbox_proposal(obj_cls_prob[i], att_cls_prob[i], rois[i])

      rois_pop = rois[:, rois_pop_id, :]
      x_obj_pop = x_obj[i * offset + rois_pop_id]

      score_obj_pop = obj_cls_score[:, 1:][rois_pop_id]

      # np.save(os.path.join(dir_feat, str(img_id[0]) + '_obj'), torch.cat((x_obj_pop, score_obj_pop), 1).cpu().numpy())
      # obj_feat = torch.cat((x_obj_pop, score_obj_pop), 1).cpu()
      obj_feat = torch.cat((x_obj_pop, rois_pop[0, :, 1:].contiguous()), 1).cpu()
      obj_feat_batch.append(obj_feat)

      x_att_pop = x_att[i * offset + rois_pop_id]
      score_att_pop = att_cls_score[:, 1:][rois_pop_id]

      # np.save(os.path.join(dir_feat, str(img_id[0]) + '_att'), torch.cat((x_att_pop, score_att_pop), 1).cpu().numpy())
      # att_feat = torch.cat((x_att_pop, score_att_pop), 1).cpu()
      att_feat = x_att_pop.cpu()
      att_feat_batch.append(att_feat)

      # get poped rel pairs according to rois_pop_id
      rois_pop_id_cpu = rois_pop_id.cpu()
      kept_rois = torch.zeros(rois[0].size(0))
      kept_rois_to_idx = torch.zeros(rois[0].size(0))

      kept_rois[rois_pop_id_cpu] = 1
      kept_rois_to_idx[rois_pop_id_cpu] = torch.arange(0, rois_pop_id_cpu.size(0))

      # find the top-N triplets
      sobj_inds = rel_pairs[i][:, 0] - i * offset
      oobj_inds = rel_pairs[i][:, 1] - i * offset
      rels_pop_id = (kept_rois[sobj_inds] + kept_rois[oobj_inds]).eq(2)

      rel_feat = torch.zeros(MAX_REL_PAIRS, x_rel.size(1) + rel_pairs[i].size(1))
      if rels_pop_id.sum() > 0:
        rels_pop_id = rels_pop_id.nonzero().squeeze()
        rels_pop = rel_pairs[i][rels_pop_id] - i * offset
        rels_pop[:, 0] = kept_rois_to_idx[rels_pop[:, 0]]
        rels_pop[:, 1] = kept_rois_to_idx[rels_pop[:, 1]]      
        x_rel_pop = x_rel[i * offset + rels_pop_id]
        rel_score_pop = rel_cls_score.cpu()[rels_pop_id]

        all_feat = torch.cat((rels_pop.float(), x_rel_pop), 1)
        if all_feat.size(0) > MAX_REL_PAIRS:
          rel_feat = all_feat[:MAX_REL_PAIRS, :]
        elif all_feat.size(0) < MIN_REL_PAIRS:
          rel_feat[:all_feat.size(0), :] = all_feat

      rel_feat_batch.append(rel_feat)

      if vis:
          vis_dets(im_data[i].unsqueeze(0), im_info[i].unsqueeze(0), rois_pop[i].unsqueeze(0), 
            bbox_pred[i, rois_pop_id, :].unsqueeze(0), obj_cls_prob[i, rois_pop_id, :].unsqueeze(0), imdb_vg)

    return global_feat_batch.data.cpu().numpy(), torch.stack(obj_feat_batch, 0).numpy(), torch.stack(att_feat_batch, 0).numpy(), torch.stack(rel_feat_batch, 0).numpy()

def ext_feat_gt(im_data, im_info, gt_boxes, num_boxes):

    outs = graphRCNN(im_data, im_info, gt_boxes, num_boxes, True)
    base_feat, rois, rel_pairs, bbox_pred, x_obj, x_att, x_rel, \
        obj_cls_prob, att_cls_prob, rel_cls_prob, \
        obj_cls_score, att_cls_score, rel_cls_score = outs

    ### extract and save global feature
    global_feat = graphRCNN._head_to_tail(base_feat)
    np.save(os.path.join(dir_feat, str(img_id[0]) + '_glb_gt'), global_feat.data.cpu().numpy())

    ### extract graph feature    
    # filter out the rois that 
    rois_pop_id = bbox_proposal_1(obj_cls_prob, att_cls_prob, rois)

    rois_pop = rois[0][rois_pop_id]
    x_obj_pop = x_obj[rois_pop_id]
    x_att_pop = x_att[rois_pop_id]
    np.save(os.path.join(dir_feat, str(img_id[0]) + '_obj_gt'), x_obj_pop.cpu().numpy())
    np.save(os.path.join(dir_feat, str(img_id[0]) + '_att_gt'), x_att_pop.cpu().numpy())

    score_obj_pop = obj_cls_score[:, 1:][rois_pop_id]
    score_att_pop = att_cls_score[:, 1:][rois_pop_id]
    np.save(os.path.join(dir_feat, str(img_id[0]) + '_obj_sc_gt'), score_obj_pop.cpu().numpy())
    np.save(os.path.join(dir_feat, str(img_id[0]) + '_att_sc_gt'), score_att_pop.cpu().numpy())

    # get poped rel pairs according to rois_pop_id
    rois_pop_id = rois_pop_id.cpu()
    kept_rois = torch.zeros(rois[0].size(0))
    kept_rois_to_idx = torch.zeros(rois[0].size(0))

    kept_rois[rois_pop_id] = 1
    kept_rois_to_idx[rois_pop_id] = torch.arange(0, rois_pop_id.size(0))

    # find the top-N triplets
    rel_pairs = rel_pairs.cpu()
    sobj_inds = rel_pairs[0][:, 0]
    oobj_inds = rel_pairs[0][:, 1]
    rels_pop_id = (kept_rois[sobj_inds] + kept_rois[oobj_inds]).eq(2)

    if rels_pop_id.sum() > 0:
      rels_pop_id = rels_pop_id.nonzero().squeeze()
      rels_pop = rel_pairs[0][rels_pop_id]
      rels_pop[:, 0] = kept_rois_to_idx[rels_pop[:, 0]]
      rels_pop[:, 1] = kept_rois_to_idx[rels_pop[:, 1]]      
      x_rel_pop = x_rel.cpu()[rels_pop_id]
      rel_score_pop = rel_cls_score.cpu()[rels_pop_id]

      np.save(os.path.join(dir_feat, str(img_id[0]) + '_rel_gt'), x_rel_pop.numpy())
      np.save(os.path.join(dir_feat, str(img_id[0]) + '_rel_sc_gt'), rel_score_pop.numpy())


    np.savez(os.path.join(dir_meta, str(img_id[0])), gt_box=gt_box, ori_box=ori_box, info=info)

    pdb.set_trace()
    
    if vis:
        vis_dets(im_data, im_info, rois, bbox_pred, obj_cls_prob, imdb_vg)


if __name__ == '__main__':

  if args.imdb_name == "train2014":
    imdb_name = args.dataset + "_2014_train"
  elif args.imdb_name == "val2014":
    imdb_name = args.dataset + "_2014_val"
  elif args.imdb_name == "test2014":
    imdb_name = args.dataset + "_2014_test"
  elif args.imdb_name == "test2015":
    imdb_name = args.dataset + "_2015_test"

  cfg.TRAIN.OBJECT_CLASSES = imdb_vg._classes
  cfg.TRAIN.ATTRIBUTE_CLASSES = imdb_vg._attributes
  cfg.TRAIN.RELATION_CLASSES = imdb_vg._relations

  cfg_file = "cfgs/{}.yml".format(args.net)
  if cfg_file is not None:
      cfg_from_file(cfg_file)
  if set_cfgs is not None:
      cfg_from_list(set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = False


  imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
  train_size = len(roidb)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the network here.
  if args.ngpu > 0:
    cfg.CUDA = True

  input_dir = "models/" + args.net


  load_name = os.path.join(input_dir, 'graph_rcnn_{}_a{}_r{}_{}_{}_{}.pth'.format(args.vg_dataset,
                      1 if cfg.HAS_ATTRIBUTES else 0, 1 if cfg.HAS_RELATIONS else 0,
                      args.checksession, args.checkepoch, args.checkpoint))

  print("loading checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)

  if 'anchor_scales' in checkpoint.keys():
    cfg.ANCHOR_SCALES = checkpoint['anchor_scales']
  if 'anchor_ratios' in checkpoint.keys():
    cfg.ANCHOR_RATIOS = checkpoint['anchor_ratios']
  if 'mute_attributes' in checkpoint.keys():
    cfg.MUTE_ATTRIBUTES = checkpoint['mute_attributes']
  if 'mute_relations' in checkpoint.keys():
    cfg.MUTE_RELATIONS = checkpoint['mute_relations']
  if 'gcn_layers' in checkpoint.keys():
    cfg.GCN_LAYERS = checkpoint['gcn_layers']
  if 'gcn_on_scores' in checkpoint.keys():
    cfg.GCN_ON_SCORES = checkpoint['gcn_on_scores']
  if 'gcn_on_feats' in checkpoint.keys():
    cfg.GCN_ON_FEATS = checkpoint['gcn_on_feats']
  if 'gcn_share_feat_params' in checkpoint.keys():
    cfg.GCN_SHARE_FEAT_PARAMS = checkpoint['gcn_share_feat_params']
  if 'gcn_low_rank_params' in checkpoint.keys():
    cfg.GCN_LOW_RANK_PARAMS = checkpoint['gcn_low_rank_params']
  if 'gcn_low_rank_dim' in checkpoint.keys():
    cfg.GCN_LOW_RANK_DIM = checkpoint['gcn_low_rank_dim']
  if 'gcn_has_attention' in checkpoint.keys():
    cfg.GCN_HAS_ATTENTION = checkpoint['gcn_has_attention']
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  if 'crop_with_max_pool' in checkpoint.keys():
    cfg.ROI_CROP_WITH_MAX_POOL = checkpoint['crop_with_max_pool']

  pprint.pprint(cfg)
  print('loaded checkpoint successfully!')

  # initilize the network here.
  if args.net == 'vgg16':
    graphRCNN = vgg16(imdb_vg._classes, imdb_vg._attributes, imdb_vg._relations, 4096)
  elif args.net == 'res50':
    graphRCNN = resnet(imdb_vg._classes, imdb_vg._attributes, imdb_vg._relations, 50)    
  elif args.net == 'res101':
    graphRCNN = resnet(imdb_vg._classes, imdb_vg._attributes, imdb_vg._relations, 101)
  elif args.net == 'res152':
    graphRCNN = resnet(imdb_vg._classes, imdb_vg._attributes, imdb_vg._relations, 152)
      
  graphRCNN.create_architecture(True)

  if args.mGPUs == True:
    graphRCNN = nn.DataParallel(graphRCNN)
    graphRCNN.load_state_dict(checkpoint['model'])
    graphRCNN = graphRCNN.module
  else:
    graphRCNN.load_state_dict(checkpoint['model'])      

  # pdb.set_trace()
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.ngpu > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.ngpu > 0:
    cfg.CUDA = True

  if args.ngpu > 0:
    graphRCNN.cuda()

  graphRCNN.eval()

  start = time.time()
  max_per_image = 300
  thresh = 0.01
  vis = args.vis

  save_name = 'graph_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                        imdb.num_classes, training=True, normalize = False)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  top_Ns = [50, 100]
  rel_cnt_all = 0
  rel_correct_cnt_all = torch.zeros(len(top_Ns)).int()

  is_oracle = False
  
  dir_ext = "data/coco/extract/" + "test_arch,{}_vg,{}_s,{}_e,{}_t,{}_p,{}_obj,{}_rel,{}". \
                                format(args.net, 600, args.vg_dataset, 
                                args.checksession, args.checkepoch, args.checkpoint, 
                                MIN_BOXES, MIN_REL_PAIRS)

  if not os.path.exists(dir_ext):
      os.makedirs(dir_ext)

#   dir_feat = os.path.join(dir_ext, "feat")
#   dir_meta = os.path.join(dir_ext, "meta")

#   if not os.path.exists(dir_feat):
#       os.makedirs(dir_feat)

#   if not os.path.exists(dir_meta):
#       os.makedirs(dir_meta)

  start = time.time()

  dim_feat = 2048
  num_rois = MAX_BOXES
  num_rels = MAX_REL_PAIRS

  hdf5_file = h5py.File(os.path.join(dir_ext, args.imdb_name + ".hdf5"), 'w')

  shape_glb_obj = (num_images, cfg.OBJECT_DIM)
  print('INFO: shape_glb_obj={}'.format(shape_glb_obj))
  hdf5_glb_obj = hdf5_file.create_dataset('glb_obj', shape_glb_obj, dtype='f')#, compression='gzip')

  shape_glb_att = (num_images, cfg.ATTRIBUTE_DIM)
  print('INFO: shape_glb_att={}'.format(shape_glb_att))
  hdf5_glb_att = hdf5_file.create_dataset('glb_att', shape_glb_att, dtype='f')#, compression='gzip')

  shape_obj = (num_images, num_rois, cfg.OBJECT_DIM + 4)
  print('INFO: shape_obj={}'.format(shape_obj))
  hdf5_obj = hdf5_file.create_dataset('obj', shape_obj, dtype='f')#, compression='gzip')

  shape_att = (num_images, num_rois, cfg.ATTRIBUTE_DIM)
  print('INFO: shape_att={}'.format(shape_att))
  hdf5_att = hdf5_file.create_dataset('att', shape_att, dtype='f')#, compression='gzip')

  shape_rel = (num_images, num_rels, cfg.RELATION_DIM + 2)
  print('INFO: shape_rel={}'.format(shape_rel))
  hdf5_rel = hdf5_file.create_dataset('rel', shape_rel, dtype='f')#, compression='gzip')

  file = open(os.path.join(dir_ext, args.imdb_name + ".txt"), 'w')

  ind = 0
  for i in range(int(num_images / args.batch_size)):

      data = data_iter.next()
      img, info, gt_box, num_box, img_id = data

      im_data.data.resize_(img.size()).copy_(img)
      im_info.data.resize_(info.size()).copy_(info)
      gt_boxes.data.resize_(gt_box.size()).copy_(gt_box)
      num_boxes.data.resize_(num_box.size()).copy_(num_box)

    #   if i % 100 == 0:
    #     end = time.time()
    #     print("image: [{}/{}] time: {}".format(i * args.batch_size, num_images, end - start))
    #     start = time.time()
    #   continue
      # extract feature from image

      # if not os.path.exists(os.path.join(dir_meta, str(img_id[0]) + '.npz')):
          # ext_feat_pred(graphRCNN, im_data, im_info, gt_boxes, num_boxes)

      # pdb.set_trace()

      batch_size = im_data.size(0)      

      if batch_size == 1:
        global_obj_feat, global_att_feat, obj_feat, att_feat, rel_feat = \
                ext_feat_pred_hdf5_v2(graphRCNN, im_data, im_info, gt_boxes, num_boxes)
      elif batch_size > 1:
        global_obj_feat, global_att_feat, obj_feat, att_feat, rel_feat = \
                ext_feat_pred_hdf5_batch(graphRCNN, im_data, im_info, gt_boxes, num_boxes)

    #   pdb.set_trace()

      # extract feature given gt object boxes
      # ext_feat_gt(im_data, im_info, gt_boxes, num_boxes)

      hdf5_glb_obj[ind:ind+batch_size]   = global_obj_feat
      hdf5_glb_att[ind:ind+batch_size]   = global_att_feat      
      hdf5_obj[ind:ind+batch_size]   = obj_feat
      hdf5_att[ind:ind+batch_size]   = att_feat
      hdf5_rel[ind:ind+batch_size]   = rel_feat

      ind += batch_size

      # pdb.set_trace()
      for j in range(batch_size):
        name = "COCO_" + args.imdb_name + ("_%012d" % img_id[j]) + ".jpg"
        file.write(name + '\n')

      if i % 100 == 0:
        end = time.time()
        print("image: [{}/{}] time: {}".format(i * batch_size, num_images, end - start))
        start = time.time()

  hdf5_file.close()
  file.close()
  print("feature extraction all done!")