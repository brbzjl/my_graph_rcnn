# --------------------------------------------------------
# Pytorch graph-rcnn
# Licensed under The MIT License [see LICENSE for details]
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils import network
from model.utils.network import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint

from model.graph_rcnn.vgg16 import vgg16
from model.graph_rcnn.resnet import resnet

import pdb

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='train', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="server",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--nworker', dest='nworker',
                      help='number of workers',
                      default=0, type=int)
  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--mGPUs', dest='use_mGPUs',
                      help='whether use multiple GPUs or not',
                      default=0, type=int)
# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr_base',
                      help='base learning rate',
                      default=0.01, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    num_data = train_size
    self.num_per_batch = int(num_data / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if num_data % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, num_data).long()
      self.leftover_flag = True
  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

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
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
  train_size = len(roidb)

  cfg.TRAIN.OBJECT_CLASSES = imdb._classes
  cfg.TRAIN.ATTRIBUTE_CLASSES = imdb._attributes
  cfg.TRAIN.RELATION_CLASSES = imdb._relations

  # clamp aspect ratio
  # ratio_list.clamp_(0.5, 2)
  # resort aspect ratio

  print('{:d} roidb entries'.format(len(roidb)))

  if args.save_dir is "local":
    output_dir = "models/" + args.net
  else:
    output_dir = "/srv/share/jyang375/models/" + args.net

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, normalize = False)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.nworker)

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
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.ngpu > 0:
    cfg.CUDA = True

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

  lr = cfg.TRAIN.LEARNING_RATE

  params = []
  for key, value in dict(graphRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.use_mGPUs > 0:
    graphRCNN = nn.DataParallel(graphRCNN)

  if args.resume:
    load_name = os.path.join(output_dir, 'graph_rcnn_{}_a{}_r{}_{}_{}_{}.pth'.format(args.dataset,
                                1 if cfg.HAS_ATTRIBUTES else 0, 1 if cfg.HAS_RELATIONS else 0,
                                args.checksession, args.checkepoch, args.checkpoint))

    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    graphRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']

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

    print("loaded checkpoint %s" % (load_name))

  if args.ngpu > 0:
    graphRCNN.cuda()

  for epoch in range(args.start_epoch, args.max_epochs):
    loss_temp = 0
    start = time.time()

    data_iter = iter(dataloader)

    for step in range(int(train_size / args.batch_size)):
      data = data_iter.next()
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      # if step % 100 == 0:
      #     print("epoch: %d, step: %d\n" % (epoch, step))

      # continue

    #   pdb.set_trace()

      graphRCNN.zero_grad()

      outs = graphRCNN(im_data, im_info, gt_boxes, num_boxes)

      if cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
          rois, bbox_pred, obj_cls_prob, att_cls_prob, rel_cls_prob, rpn_loss, relpn_loss, rcnn_loss = outs
      elif cfg.HAS_ATTRIBUTES:
          rois, bbox_pred, obj_cls_prob, att_cls_prob, rpn_loss, rcnn_loss = outs
      else:
          rois, bbox_pred, obj_cls_prob, rpn_loss, rcnn_loss = outs

      loss = rpn_loss.sum() + rcnn_loss.sum()

      # if cfg.HAS_RELATIONS:
      #   loss += relpn_loss.sum()
      loss /= rpn_loss.size(0)
      loss_temp += loss.data[0]

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          network.clip_gradient(graphRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= args.disp_interval

        if args.use_mGPUs > 0:
          print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
            % (args.session, epoch, step, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), rel_fg/bg_rel=(%d/%d), time cost: %f" % (0, 0, 0, 0, end-start))
          print("\t\t\trpn: %.4f, rcnn: %.4f, relpn: %.4f" %
            (rpn_loss.data.mean(), \
             rcnn_loss.data.mean(), \
             0 if not cfg.HAS_RELATIONS else relpn_loss.data.mean()))

          if args.use_tfboard:
            info = {
              'loss': loss_temp / args.disp_interval
            }
            for tag, value in info.items():
              logger.scalar_summary(tag, value, step)

        else:
          print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
            % (args.session, epoch, step, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (graphRCNN.fg_cnt, graphRCNN.bg_cnt, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_obj_cls: %.4f, rcnn_box: %.4f, rcnn_att_cls: %.4f, relpn_clss: %.4f, rcnn_rel_cls: %.4f" %
            (graphRCNN.RCNN_rpn.rpn_loss_cls.data[0], \
             graphRCNN.RCNN_rpn.rpn_loss_box.data[0], \
             graphRCNN.RCNN_loss_obj_cls.data[0], \
             graphRCNN.RCNN_loss_bbox.data[0], \
             0 if not cfg.HAS_ATTRIBUTES else graphRCNN.RCNN_loss_att_cls.data[0], \
             0 if not cfg.HAS_RELATIONS else relpn_loss.data[0],
             0 if not cfg.HAS_RELATIONS else graphRCNN.RCNN_loss_rel_cls.data[0]))

          if args.use_tfboard:
            info = {
              'loss': loss_temp / args.disp_interval,
              'loss_rpn_cls': graphRCNN.RCNN_rpn.rpn_loss_cls.data[0],
              'loss_rpn_box': graphRCNN.RCNN_rpn.rpn_loss_box.data[0],
              'loss_rcnn_obj_cls': graphRCNN.RCNN_loss_obj_cls.data[0],
              'loss_rcnn_att_cls': graphRCNN.RCNN_loss_att_cls.data[0],
              'loss_rcnn_box': graphRCNN.RCNN_loss_bbox.data[0]
            }
            for tag, value in info.items():
              logger.scalar_summary(tag, value, step)

        loss_temp = 0
        start = time.time()

    if epoch % args.lr_decay_step == 0:

        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    #   pdb.set_trace()
    save_name = os.path.join(output_dir, 'graph_rcnn_rpn_{}_a{}_r{}_{}_{}_{}.pth'.format(args.dataset,
                                                                                   1 if cfg.HAS_ATTRIBUTES else 0,
                                                                                   1 if cfg.HAS_RELATIONS else 0,
                                                                                   args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': graphRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'anchor_scales': cfg.ANCHOR_SCALES,
      'anchor_ratios': cfg.ANCHOR_RATIOS,
      'relpn_with_box_info': cfg.TRAIN.RELPN_WITH_BBOX_INFO,
      'mute_attributes': cfg.MUTE_ATTRIBUTES,
      'mute_relations': cfg.MUTE_RELATIONS,
      'gcn_layers': cfg.GCN_LAYERS,
      'gcn_on_scores': cfg.GCN_ON_SCORES,
      'gcn_on_feats': cfg.GCN_ON_FEATS,
      'gcn_share_feat_params': cfg.GCN_SHARE_FEAT_PARAMS,
      'gcn_low_rank_params': cfg.GCN_LOW_RANK_PARAMS,
      'gcn_low_rank_dim': cfg.GCN_LOW_RANK_DIM,
      'gcn_has_attention': cfg.GCN_HAS_ATTENTION,
      'pooling_mode': cfg.POOLING_MODE,
      'crop_with_max_pool': cfg.ROI_CROP_WITH_MAX_POOL,
    }, save_name)
    print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)
