# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch    ###

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
# import torch  ###
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

from model.graph_rcnn.vgg16 import vgg16
from model.graph_rcnn.resnet import resnet

import pdb

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='vg1', type=str)       ## 'pascal voc'
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='test', type=str)    ### 'train'
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="server",
                      nargs=argparse.REMAINDER)

  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple gpus or not',
                      action='store_true')
  parser.add_argument('--vis', dest='vis',
                      help='whether use multiple gpus or not',  ## todo: wrong?
                      action='store_false')      ###

  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=4, type=int)                  ###
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=74004, type=int)

  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)                  ## must be 1!!!!!

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
  # os.environ['CUDA_VISIBLE_DEVICES'] = '4'          ###
  torch.set_num_threads(8)  ###

  args = parse_args()

  # print('Called with args:')    ###
  # print(args)

  if args.dataset == "vg1":
      imdb_name = ["vg_150-50-20_" + s for s in args.imdb_name]
      imdb_name = "vg_150-50-20_" + args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
  elif args.dataset == "vg2":
      imdb_name = ["vg_500-150-80_" + s for s in args.imdb_name]
      imdb_name = "vg_500-150-80_" + args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
  elif args.dataset == "vg3":
      imdb_name = ["vg_750-250-150_" + s for s in args.imdb_name]
      imdb_name = "vg_750-250-150_" + args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
  elif args.dataset == "vg4":
      imdb_name = ["vg_1750-700-450_" + s for s in args.imdb_name]
      imdb_name = "vg_1750-700-450_" + args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
  elif args.dataset == "vg5":
      imdb_name = ["vg_1600-400-20_" + s for s in args.imdb_name]
      imdb_name = "vg_1600-400-20_" + args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']
  elif args.dataset == "vg_bm":
      imdb_name = ["vg_150-50-50_" + s for s in args.imdb_name]
      imdb_name = "vg_150-50-50_" + args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]']

  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True

  imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)

  train_size = len(roidb)
  cfg.TRAIN.OBJECT_CLASSES = imdb._classes
  cfg.TRAIN.ATTRIBUTE_CLASSES = imdb._attributes
  cfg.TRAIN.RELATION_CLASSES = imdb._relations
  imdb.competition_mode(on=True)        ## todo??

  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the network here.
  if args.ngpu > 0:
    cfg.CUDA = True


  if args.load_dir is "local":
    input_dir = "models/" + args.net
  elif args.load_dir is "server":
    # input_dir = "/srv/share/jyang375/models/" + args.net
    input_dir = "/home/yijinhui/Projects/VRD/master_thesis/data/models/" + args.net        ###

  else:
    raise Exception('Input directory is wrong for loading network')

  load_name = os.path.join(input_dir, 'graph_rcnn_{}_a{}_r{}_{}_{}_{}.pth'.format(args.dataset,
                      1 if cfg.HAS_ATTRIBUTES else 0, 1 if cfg.HAS_RELATIONS else 0,
                      args.checksession, args.checkepoch, args.checkpoint))

  print("loading checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)

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
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  if 'crop_with_max_pool' in checkpoint.keys():
    cfg.ROI_CROP_WITH_MAX_POOL = checkpoint['crop_with_max_pool']

  # pprint.pprint(cfg)    ###
  print('loaded checkpoint successfully!')

  # initilize the network here.
  if args.net == 'vgg16':
    graphRCNN = vgg16(imdb._classes, imdb._attributes, imdb._relations, 4096)
  elif args.net == 'res101':
    graphRCNN = resnet(imdb._classes, imdb._attributes, imdb._relations, 101)
  elif args.net == 'res50':
    graphRCNN = resnet(imdb._classes, imdb._attributes, imdb._relations, 50)
  elif args.net == 'res152':
    graphRCNN = resnet(imdb._classes, imdb._attributes, imdb._relations, 152)

  graphRCNN.create_architecture()
  # graphRCNN = nn.DataParallel(graphRCNN)                  ### for multi gpu
  graphRCNN.load_state_dict(checkpoint['model'])
  # graphRCNN = graphRCNN.module                            ### for multi gpu


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
  im_data = Variable(im_data, volatile=True)        ## todo: volatile?
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.ngpu > 0:
    cfg.CUDA = True

  if args.ngpu > 0:
    graphRCNN.cuda()

  graphRCNN.eval()      ## evaluation mode

  start = time.time()
  max_per_image = 300   ## todo:??
  # thresh = -np.inf
  thresh = 0.01
  vis = args.vis

  save_name = 'graph_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)] ## todo:??
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)  ## todo??

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                        imdb.num_classes, training=False, normalize = False)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=True)    ## todo:pin_memory? shuffle?

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  top_Ns = [50, 100, 10000]
  rel_cnt_all = 0
  rel_correct_cnt_all = torch.zeros(len(top_Ns)).int()

  use_gt_boxes = False
  # for i in range(num_images):
  for i in range(int(num_images)):           ###

      data = data_iter.next()
      im_data.data.resize_(data[0].size()).copy_(data[0])       ## todo: ??
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()

      outs = graphRCNN(im_data, im_info, gt_boxes, num_boxes, use_gt_boxes)

      if cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
        rois, rels, bbox_pred, obj_cls_prob, att_cls_prob, rel_cls_prob, rpn_loss, relpn_loss, rcnn_loss = outs
      elif cfg.HAS_ATTRIBUTES:
        rois, bbox_pred, obj_cls_prob, att_cls_prob, rpn_loss, rcnn_loss = outs
      else:
        rois, bbox_pred, obj_cls_prob, rpn_loss, rcnn_loss = outs

      # pdb.set_trace()
      # pdb.set_trace()
      scores = obj_cls_prob.data
      boxes = rois.data[:, :, 1:5]
      if cfg.TEST.BBOX_REG and not use_gt_boxes:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas = box_deltas.view(1, -1, 4)
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
        pred_boxes = boxes

      # evaluate realtion detection performance
      if cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
        # pdb.set_trace()
        gt_annotations = gt_boxes.data[0]
        object_rois = pred_boxes[0]

        object_scores = obj_cls_prob.data[0]
        relation_indice = rels.view(-1, 2)
        relation_scores = rel_cls_prob.data[0]
        rel_cnt, rel_corrent_cnt, gt_rel_rois, gt_rel_labels = \
            eval_relations_recall(gt_annotations, object_rois, object_scores,
                              relation_indice, relation_scores, top_Ns, use_gt_boxes)
        rel_cnt_all += rel_cnt
        rel_correct_cnt_all += rel_corrent_cnt

      sys.stdout.write('rel_recall: {:d}/top-50: {:d} top-100: {:d} top-1000: {:d} \r' \
          .format(rel_cnt_all, rel_correct_cnt_all[0], rel_correct_cnt_all[1], rel_correct_cnt_all[2]))
      continue

      pred_boxes /= data[1][0][2]

      obj_scores = obj_cls_prob.data.squeeze()
      att_scores = att_cls_prob.data.squeeze()
      att_scores[:, 0] = 0

      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()

      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
          im2show_rel = np.copy(im)
          # visualize gt relations
          # if gt_rel_rois is not None:
          #   gt_rel_rois /= data[1][0][2]
          #   for i in range(min(1, gt_rel_rois.size(0))):
          #     gt_rel_rois_i = gt_rel_rois[i]
          #     gt_rel_labels_i = gt_rel_labels[i]
          #     im2show = vis_gt_relations(im2show, gt_rel_rois_i.cpu().numpy(),
          #                                         gt_rel_labels_i.cpu().numpy(),
          #                                         imdb._classes, imdb._relations)

      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(obj_scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            obj_cls_scores = obj_scores[:,j][inds]
            att_cls_scores = att_scores[inds]
            _, order = torch.sort(obj_cls_scores, 0, True)
            cls_boxes = pred_boxes[inds, :]
            cls_dets = torch.cat((cls_boxes, obj_cls_scores), 1)
            cls_dets = cls_dets[order]
            cls_atts = att_cls_scores[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            cls_atts = cls_atts[keep.view(-1).long()]
            if vis:
              im2show = vis_det_att(im2show, imdb.classes[j], imdb._attributes,
                cls_dets.cpu().numpy(), cls_atts.cpu().numpy(), 0.5)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      # # visualize relations
      rel_scores = rel_cls_prob.data.squeeze()
      rels = rels.squeeze()
      for j in xrange(1, len(imdb._relations)):
          inds = torch.nonzero(rel_scores[:, j] > thresh).view(-1)
          if inds.numel() > 0:
            rels_val = rels[inds]
            rel_cls_scores = rel_scores[:,j][inds]
            _, order = torch.sort(rel_cls_scores, 0, True)
            proposals_subject = pred_boxes[rels_val[order][:, 0]]
            proposals_object = pred_boxes[rels_val[order][:, 1]]
            rel_cls_scores = rel_cls_scores[order]
            keep = co_nms(torch.cat((proposals_subject, proposals_object), 1), cfg.TEST.NMS)
            keep = keep.long().view(-1)
            rel_cls_scores = rel_cls_scores[keep]
            roi_rel_pairs = pred_boxes.new(keep.size(0), 9).zero_()
            roi_rel_pairs[:, 1:5] = proposals_subject[keep]
            roi_rel_pairs[:, 5:9] = proposals_object[keep]
            rois_rel = combine_box_pairs(roi_rel_pairs)
            if vis:
              im2show_rel = vis_relations(im2show_rel, imdb._relations[j], rel_cls_scores.cpu().numpy(),
                rois_rel[:, 1:].cpu().numpy(), 0.5)

          # pdb.set_trace()
          # keep_idx_i = co_nms(torch.cat((proposals_subject, proposals_object), 1), nms_thresh)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))

      sys.stdout.flush()

      if vis:
          cv2.imwrite("images/result_{}.jpg".format(i), im2show)
          cv2.imwrite("images/result_rel_{}.jpg".format(i), im2show_rel)
          # cv2.imshow('test', im2show)
          # cv2.imshow('test_rel', im2show_rel)
          # cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

  # print('Evaluating detections')
  # imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))

  for i in range(len(top_Ns)):
    print("rel recall@%d: %0.4f\r" % (top_Ns[i], float(rel_correct_cnt_all[i]) / float(rel_cnt_all)))



'''
2-1-37001   bs:4    gpu:1
test time: 3875.9729s-50: 4673 top-100: 5751 top-1000: 7043 
rel recall@50: 0.0520
rel recall@100: 0.0640
rel recall@10000: 0.0784
2-2-37001   bs:4    gpu:2!! T:  t:  mem:9197 + 4481

2-1-74004   bs:2    gpu:1     cpu:?     mem:6629  t:60s/100steps  T:46054s/epoch=12.8h
test time: 3768.1669s-50: 3313 top-100: 4038 top-1000: 4917 
rel recall@50: 0.0369
rel recall@100: 0.0449
rel recall@10000: 0.0547
2-2-74004   bs:2    gpu:1
test time: 3998.1214s-50: 3995 top-100: 4817 top-1000: 5772 
rel recall@50: 0.0445
rel recall@100: 0.0536
rel recall@10000: 0.0642
2-3-74004   bs:2    gpu:1
test time: 5776.3653s-50: 3881 top-100: 4653 top-1000: 5589 
rel recall@50: 0.0432
rel recall@100: 0.0518
rel recall@10000: 0.0622


3-1-18501   bs:8    gpu:4    cpu:?  mem:4500*4   t:128s/100steps   T:31190s/epoch=8.7h
test time: 4057.6388s-50: 5070 top-100: 6238 top-1000: 7827 
rel recall@50: 0.0564
rel recall@100: 0.0694
rel recall@10000: 0.0871

4-1-74004   bs:2    gpu:1   cpu:5   grad acc:4    mem: 6827   t:60s/100steps  T: 40613s/epoch=11.3h
test time: 4342.3485s-50: 4881 top-100: 5815 top-1000: 7191 
rel recall@50: 0.0543
rel recall@100: 0.0647
rel recall@10000: 0.0800

'''