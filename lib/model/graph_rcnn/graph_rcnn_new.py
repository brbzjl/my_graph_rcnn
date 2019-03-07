import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.rpn.proposal_target_layer import _ProposalTargetLayer

from model.repn.relpn import _RelPN
from model.repn.relpn_target_layer import _RelProposalTargetLayer
from model.repn.bbox_transform import combine_box_pairs
from model.graph_conv.graph_conv import _GraphConvolutionLayer as _GCN_1
from model.graph_conv.graph_conv_score import _GraphConvolutionLayer as _GCN_2

from model.utils import network
import time
import pdb
from model.utils.network import _smooth_l1_loss, _softmax_with_loss

class _graphRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, baseModels, obj_classes, att_classes, rel_classes, dout_base_model, pooled_feat_dim):

        super(_graphRCNN, self).__init__()
        self.obj_classes = obj_classes
        self.n_obj_classes = len(obj_classes)

        self.att_classes = att_classes
        self.n_att_classes = 0 if att_classes == None else len(att_classes)

        self.rel_classes = rel_classes
        self.n_rel_classes = 0 if rel_classes == None else len(rel_classes)

        # define base model
        self.RCNN_base_model = baseModels

        # define rpn
        self.RCNN_rpn = _RPN(dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_obj_classes, self.n_att_classes, self.n_rel_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        if cfg.HAS_RELATIONS:
            self.RELPN_rpn = _RelPN(pooled_feat_dim)
            self.RELPN_proposal_target = _RelProposalTargetLayer(self.n_rel_classes)
            self.RELPN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        reduced_pooled_feat_dim = 512
        
        self.fc4obj = nn.Linear(pooled_feat_dim, reduced_pooled_feat_dim)
        self.fc4att = nn.Linear(pooled_feat_dim, reduced_pooled_feat_dim)
        self.fc4rel = nn.Linear(pooled_feat_dim, reduced_pooled_feat_dim)

        self.RCNN_gcn_obj_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_obj_classes)
        self.RCNN_gcn_att_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_att_classes)
        self.RCNN_gcn_rel_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_rel_classes)

        if cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS and cfg.GCN_LAYERS > 0:
            if not cfg.GCN_ON_SCORES:
                self.GRCNN_gcn = _GCN_1(reduced_pooled_feat_dim)
            else:
                self.GRCNN_gcn = _GCN_2(self.n_obj_classes, self.n_att_classes, self.n_rel_classes)

        self.RCNN_loss_obj_cls = 0
        self.RCNN_loss_att_cls = 0
        self.RCNN_loss_rel_cls = 0
        self.RCNN_loss_bbox = 0

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
        normal_init(self.RCNN_obj_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        if self.RCNN_att_cls_score != None:
            normal_init(self.RCNN_att_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if self.RCNN_rel_cls_score != None:
            normal_init(self.RCNN_rel_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, use_gt_boxes = False):
        batch_size = im_data.size(0)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base_model(im_data)

        if not use_gt_boxes:
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info.data, gt_boxes.data, num_boxes.data)
        else:
            # otherwise use groundtruth box as the outputs of RCNN_rpn
            rois = gt_boxes
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rpn_loss = rpn_loss_cls + rpn_loss_bbox

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes.data)
            rois, rois_obj_label, rois_att_label, \
            rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_obj_label = Variable(rois_obj_label.view(-1))
            rois_att_label = Variable(rois_att_label.view(-1, self.n_att_classes))
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_obj_label = None
            rois_att_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        x_obj = self._head_to_tail(pooled_feat)  # (B x N) x D

        # compute object classification probability
        obj_cls_score = self.RCNN_obj_cls_score(x_obj)
        obj_cls_prob = F.softmax(obj_cls_score)
        bbox_pred = self.RCNN_bbox_pred(x_obj)

        if cfg.HAS_ATTRIBUTES:
            x_att = self._head_to_tail_att(pooled_feat)  # (B x N) x D
            att_cls_score = self.RCNN_att_cls_score(x_att)
            att_cls_prob = F.softmax(att_cls_score)
            att_cls_log_prob = F.log_softmax(att_cls_score)

        if cfg.HAS_RELATIONS:
            # feed base feature map tp RPN to obtain rois
            # x_view = x.view(rois.size(0), rois.size(1), x.size(1))
            rel_feats = obj_cls_score.view(rois.size(0), rois.size(1), obj_cls_score.size(1))
            roi_rel_pairs, roi_pair_proposals, relpn_loss_cls = \
                self.RELPN_rpn(rois.data, rel_feats, im_info.data, gt_boxes.data, num_boxes.data)

            relpn_loss = relpn_loss_cls

            roi_pair_proposals_v = roi_pair_proposals.view(-1, 2)
            ind_subject = roi_pair_proposals_v[:, 0]
            ind_object = roi_pair_proposals_v[:, 1]

            if self.training:

                roi_pair_data = self.RELPN_proposal_target(roi_rel_pairs, gt_boxes.data, num_boxes.data)

                # pdb.set_trace()

                roi_rel_pairs, rois_rel_label, roi_pair_keep = roi_pair_data
                rois_rel_label = Variable(rois_rel_label.view(-1))

                roi_pair_keep = roi_pair_keep + torch.arange(0, roi_pair_keep.size(0)).view(roi_pair_keep.size(0), 1).cuda() \
                                                * roi_pair_proposals_v.size(0) / batch_size
                roi_pair_keep = roi_pair_keep.view(-1).long()

                ind_subject = roi_pair_proposals_v[roi_pair_keep][:, 0]
                ind_object = roi_pair_proposals_v[roi_pair_keep][:, 1]


            rois_pred = combine_box_pairs(roi_rel_pairs.view(-1, 9))
            rois_pred = Variable(rois_pred)

            # # do roi pooling based on predicted rois
            pooled_pred_feat = self.RELPN_roi_pool(base_feat, rois_pred.view(-1,5))

            # # combine subject, object and relation feature tohether
            x_pred = self._head_to_tail_rel(pooled_pred_feat)

            # x_sobj = x_obj[ind_subject]
            # x_oobj = x_obj[ind_object]

            x_rel = x_pred #torch.cat((x_sobj, x_pred, x_oobj), 1)

            # compute object classification probability
            rel_cls_score = self.RCNN_rel_cls_score(x_rel)
            rel_cls_prob = F.softmax(rel_cls_score)

        if not cfg.GCN_ON_SCORES and cfg.GCN_LAYERS > 0 and cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
            # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
            # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
            # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
            # NOTE exclude one roi feature from another.
            # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
            # cooccurence, such as bus are more frequently appear on the road.
            assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"
            map_obj_att = torch.eye(x_obj.size(0)).type_as(x_obj.data)
            map_obj_obj = x_obj.data.new(x_obj.size(0), x_obj.size(0)).fill_(1.0) - torch.eye(x_obj.size(0)).type_as(x_obj.data)
            map_sobj_rel = x_obj.data.new(x_obj.size(0), x_rel.size(0)).zero_()
            map_sobj_rel.scatter_(0, ind_subject.contiguous().view(1, x_rel.size(0)), x_rel.data.new(1, x_pred.size(0)).fill_(1))
            map_oobj_rel = x_obj.data.new(x_obj.size(0), x_rel.size(0)).zero_()
            map_oobj_rel.scatter_(0, ind_object.contiguous().view(1, x_rel.size(0)), x_rel.data.new(1, x_pred.size(0)).fill_(1))
            map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 2)

            map_obj_att = Variable(map_obj_att)
            map_obj_obj = Variable(map_obj_obj)
            map_obj_rel = Variable(map_obj_rel)

            x_obj = F.relu(self.fc4obj(x_obj))
            x_att = F.relu(self.fc4att(x_att))
            x_pred = F.relu(self.fc4rel(x_pred))

            for i in range(cfg.GCN_LAYERS):
                # pass graph representation to gcn
                x_obj, x_att, x_pred = self.GRCNN_gcn(x_obj, x_att, x_pred, map_obj_att, map_obj_obj, map_obj_rel)

                # pdb.set_trace()
                # compute object classification loss
                obj_cls_score = self.RCNN_gcn_obj_cls_score(x_obj)
                obj_cls_prob = F.softmax(obj_cls_score)

                # compute attribute classification loss
                att_cls_score = self.RCNN_gcn_att_cls_score(x_att)
                att_cls_prob = F.softmax(att_cls_score)
                att_cls_log_prob = F.log_softmax(att_cls_score)

                # compute relation classifcation loss
                # x_sobj = x_obj[ind_subject]
                # x_oobj = x_obj[ind_object]
                x_rel = x_pred # torch.cat((x_sobj, x_pred, x_oobj), 1)
                rel_cls_score = self.RCNN_gcn_rel_cls_score(x_rel)
                rel_cls_prob = F.softmax(rel_cls_score)

        if cfg.GCN_ON_SCORES and cfg.GCN_LAYERS > 0 and cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
            # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
            # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
            # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
            # NOTE exclude one roi feature from another.
            # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
            # cooccurence, such as bus are more frequently appear on the road.

            assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"
            map_obj_att = torch.eye(x_obj.size(0)).type_as(x_obj.data)
            map_obj_obj = x_obj.data.new(x_obj.size(0), x_obj.size(0)).fill_(1.0) - torch.eye(x_obj.size(0)).type_as(x_obj.data)
            map_sobj_rel = x_obj.data.new(x_obj.size(0), x_rel.size(0)).zero_()
            map_sobj_rel.scatter_(0, ind_subject.contiguous().view(1, x_rel.size(0)), x_rel.data.new(1, x_pred.size(0)).fill_(1))
            map_oobj_rel = x_obj.data.new(x_obj.size(0), x_rel.size(0)).zero_()
            map_oobj_rel.scatter_(0, ind_object.contiguous().view(1, x_rel.size(0)), x_rel.data.new(1, x_pred.size(0)).fill_(1))
            map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 2)

            map_obj_att = Variable(map_obj_att)
            map_obj_obj = Variable(map_obj_obj)
            map_obj_rel = Variable(map_obj_rel)

            for i in range(cfg.GCN_LAYERS):
                # pass graph representation to gcn
                obj_cls_score, att_cls_score, rel_cls_score =\
                    self.GRCNN_gcn(obj_cls_score, att_cls_score, rel_cls_score, map_obj_obj, map_obj_att, map_obj_rel)

                # compute object classification loss
                obj_cls_prob = F.softmax(obj_cls_score)

                # compute attribute classification loss
                att_cls_prob = F.softmax(att_cls_score)
                att_cls_log_prob = F.log_softmax(att_cls_score)

                # compute relation classifcation loss
                rel_cls_prob = F.softmax(rel_cls_score)

        self.RCNN_loss_bbox = 0
        self.RCNN_loss_obj_cls = 0
        self.RCNN_loss_att_cls = 0
        self.RCNN_loss_rel_cls = 0

        if self.training:

            self.fg_cnt = torch.sum(rois_obj_label.data.ne(0))
            self.bg_cnt = rois_obj_label.data.numel() - self.fg_cnt
            self.RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            # object classification loss
            obj_label = rois_obj_label.long()
            self.RCNN_loss_obj_cls = F.cross_entropy(obj_cls_score, obj_label)

            # attribute classification loss
            if cfg.HAS_ATTRIBUTES:
                att_label = rois_att_label
                att_label = att_label[rois_obj_label.data.nonzero().squeeze()]
                # att_cls_score = att_cls_score[rois_obj_label.data.nonzero().squeeze()]
                # self.RCNN_loss_att_cls = F.multilabel_soft_margin_loss(att_cls_score, att_label)
                att_cls_log_prob = att_cls_log_prob[rois_obj_label.data.nonzero().squeeze()]
                self.RCNN_loss_att_cls = _softmax_with_loss(att_cls_log_prob, att_label)

            if cfg.HAS_RELATIONS:
                self.rel_fg_cnt = torch.sum(rois_rel_label.data.ne(0))
                self.rel_bg_cnt = rois_rel_label.data.numel() - self.rel_fg_cnt
                rel_label = rois_rel_label.long()
                self.RCNN_loss_rel_cls = F.cross_entropy(rel_cls_score, rel_label)

        rcnn_loss = self.RCNN_loss_bbox + self.RCNN_loss_obj_cls

        if cfg.HAS_ATTRIBUTES:
            rcnn_loss += 0.5 * self.RCNN_loss_att_cls

        if cfg.HAS_RELATIONS:
            rcnn_loss += 0.5 * self.RCNN_loss_rel_cls


        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        obj_cls_prob = obj_cls_prob.view(batch_size, rois.size(1), -1)
        att_cls_prob = None if not cfg.HAS_ATTRIBUTES else att_cls_prob.view(batch_size, rois.size(1), -1)
        rel_cls_prob = None if not cfg.HAS_RELATIONS else rel_cls_prob.view(batch_size, rois.size(1), -1)

        if cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
            if self.training:
                return rois, bbox_pred, obj_cls_prob, att_cls_prob, rel_cls_prob, rpn_loss, relpn_loss, rcnn_loss
            else:
                rel_pairs = roi_pair_proposals
                return rois, rel_pairs, bbox_pred, obj_cls_prob, att_cls_prob, rel_cls_prob, rpn_loss, relpn_loss, rcnn_loss
        elif cfg.HAS_ATTRIBUTES:
            return rois, bbox_pred, obj_cls_prob, att_cls_prob, rpn_loss, rcnn_loss
        else:
            return rois, bbox_pred, obj_cls_prob, rpn_loss, rcnn_loss
