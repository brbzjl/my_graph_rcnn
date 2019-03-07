import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import time
import pdb

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.rpn.proposal_target_layer_msdn import _ProposalTargetLayer_MSDN

from model.repn.relpn import _RelPN
from model.repn.relpn_target_layer import _RelProposalTargetLayer
from model.repn.bbox_transform import combine_box_pairs

from model.graph_conv.graph_conv_score import _GraphConvolutionLayer as _GCN_1
from model.graph_conv.graph_conv import _GraphConvolutionLayer as _GCN_2
from model.graph_conv.graph_conv_share import _GraphConvolutionLayer as _GCN_3
from model.graph_conv.graph_conv_lowrank import _GraphConvolutionLayer as _GCN_4

from model.graph_conv.graph_attention import _GraphAttentionLayer as _GCN_ATT

from model.utils import network
from model.utils.network import _smooth_l1_loss, _softmax_with_loss
from model.utils.net_utils import _affine_grid_gen

from .imp import _IMP

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
        	x = self.relu(x)
        return x

class _ISGG(nn.Module):
    """ faster RCNN """
    def __init__(self, baseModels, obj_classes, att_classes, rel_classes, dout_base_model, pooled_feat_dim):

        super(_ISGG, self).__init__()
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
        self.RCNN_proposal_target_msdn = _ProposalTargetLayer_MSDN(self.n_obj_classes, self.n_att_classes, self.n_rel_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        if cfg.HAS_RELATIONS:
            self.RELPN_rpn = _RelPN(pooled_feat_dim, self.n_obj_classes)
            self.RELPN_proposal_target = _RelProposalTargetLayer(self.n_rel_classes)


            self.RELPN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
            self.RELPN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
            self.RELPN_grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
            self.RELPN_roi_crop = _RoICrop()

        reduced_pooled_feat_dim = pooled_feat_dim

        # define mps
        nhidden = 512
        dropout = False
        gate_width = 1
        use_kernel_function = False

        self.imp = _IMP(nhidden, dropout,
                        gate_width=gate_width, use_kernel_function=use_kernel_function) # the hierarchical message passing structure
        network.weights_normal_init(self.imp, 0.01)


        # self.fc4obj = nn.Linear(pooled_feat_dim, reduced_pooled_feat_dim)
        # self.fc4att = nn.Linear(pooled_feat_dim, reduced_pooled_feat_dim)
        # self.fc4rel = nn.Linear(pooled_feat_dim, reduced_pooled_feat_dim)

        # self.RCNN_gcn_obj_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_obj_classes)
        # self.RCNN_gcn_att_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_att_classes)
        # self.RCNN_gcn_rel_cls_score = nn.Linear(reduced_pooled_feat_dim, self.n_rel_classes)

        if cfg.GCN_LAYERS > 0:
            if cfg.GCN_ON_SCORES:
                self.GRCNN_gcn_score = _GCN_1(self.n_obj_classes, self.n_att_classes, self.n_rel_classes)

            if cfg.GCN_ON_FEATS and not cfg.GCN_SHARE_FEAT_PARAMS:
                self.GRCNN_gcn_feat = _GCN_2(reduced_pooled_feat_dim)

            if cfg.GCN_ON_FEATS and cfg.GCN_SHARE_FEAT_PARAMS:
                self.GRCNN_gcn_feat = _GCN_3(reduced_pooled_feat_dim)

            if cfg.GCN_ON_FEATS and cfg.GCN_LOW_RANK_PARAMS:
                self.GRCNN_gcn_feat = _GCN_4(reduced_pooled_feat_dim)

        if cfg.GCN_HAS_ATTENTION:
            self.GRCNN_gcn_att1 = _GCN_ATT(self.n_obj_classes)
            self.GRCNN_gcn_att2 = _GCN_ATT(self.n_obj_classes)

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


    def _setup_connection(self, object_rois, graph_generation=False):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        roi_num = 64 # cfg.TEST.BBOX_NUM
        keep_inds = torch.arange(0, min(roi_num, object_rois.size(0))).long()
        roi_num = len(keep_inds)

        id_i, id_j = self._generate_pairs(keep_inds.numpy()) # Grouping the input object rois and remove the diagonal items
        id_i = torch.from_numpy(id_i).long().cuda()
        id_j = torch.from_numpy(id_j).long().cuda()

        rois = object_rois[keep_inds.cuda()]

        roi_pair_proposals = torch.stack((id_i, id_j), 1)
        roi_rel_pairs = object_rois.new(id_i.size(0), 9).zero_()
        roi_rel_pairs[:, :5].copy_(rois[id_i])
        roi_rel_pairs[:, 5:].copy_(rois[id_j][:, 1:5])
        return rois, roi_rel_pairs, roi_pair_proposals

    def _generate_pairs(self, ids):
        id_i, id_j = np.meshgrid(ids, ids, indexing='ij') # Grouping the input object rois
        id_i = id_i.reshape(-1)
        id_j = id_j.reshape(-1)
        # remove the diagonal items
        id_num = len(ids)
        diagonal_items = np.array(range(id_num))
        diagonal_items = diagonal_items * id_num + diagonal_items
        all_id = range(len(id_i))
        selected_id = np.setdiff1d(all_id, diagonal_items)
        id_i = id_i[selected_id]
        id_j = id_j[selected_id]

        return id_i, id_j

    def create_architecture(self, ext_feat=False):
        self._init_modules()
        self._init_weights()
        self.ext_feat = ext_feat

    def forward(self, im_data, im_info, gt_boxes, num_boxes, use_gt_boxes = False):
        batch_size = im_data.size(0)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base_model(im_data)

        if not use_gt_boxes:
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info.data, gt_boxes.data, num_boxes.data)
        else:
            # otherwise use groundtruth box as the outputs of RCNN_rpn
            rois = gt_boxes.data.clone()
            rois[0, :, 0] = 0
            rois[0, :, 1:] = gt_boxes.data[0, :, :4]
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        if not self.training:
            if batch_size == 1:
                valid = rois.sum(2).view(-1).nonzero().view(-1)
                rois = rois[:, valid, :]

        rpn_loss = rpn_loss_cls + rpn_loss_bbox

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes.data)
            # rois, rois_obj_label, rois_att_label, \
            # rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # rois_obj_label = Variable(rois_obj_label.view(-1))
            # rois_att_label = Variable(rois_att_label.view(-1, self.n_att_classes))
            # rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            # rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            # rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            roi_data_msdn = self.RCNN_proposal_target_msdn(rois, gt_boxes.data)

            rois, roi_rel_pairs, roi_pair_proposals, rois_obj_label, rois_att_label, rois_rel_label, \
            rois_target, rois_inside_ws, rois_outside_ws = roi_data_msdn
            rois_obj_label = Variable(rois_obj_label.view(-1))
            rois_att_label = Variable(rois_att_label.view(-1, self.n_att_classes))
            rois_rel_label = Variable(rois_rel_label.view(-1))
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            roi_pair_proposals = roi_pair_proposals.long()
            roi_pair_proposals_v = roi_pair_proposals.view(-1, 2)
            ind_subject = roi_pair_proposals_v[:, 0]
            ind_object = roi_pair_proposals_v[:, 1]
        else:

            rois_obj_label = None
            rois_att_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            rois_out = []
            roi_rel_pairs_out = []
            roi_pair_proposals_out = []
            for i in range(rois.size(0)):
                rois, roi_rel_pairs, roi_pair_proposals = self._setup_connection(rois[i])
                rois_out.append(rois)
                roi_rel_pairs_out.append(roi_rel_pairs)
                roi_pair_proposals_out.append(roi_pair_proposals)

            rois = torch.stack(rois_out, 0)
            roi_rel_pairs = torch.stack(roi_rel_pairs_out, 0)
            roi_pair_proposals = torch.stack(roi_pair_proposals_out, 0)

            roi_pair_proposals = roi_pair_proposals.long()
            roi_pair_proposals_v = roi_pair_proposals.view(-1, 2)
            ind_subject = roi_pair_proposals_v[:, 0]
            ind_object = roi_pair_proposals_v[:, 1]

        rois = Variable(rois)

        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
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
            # rel_feats = obj_cls_score.view(rois.size(0), rois.size(1), obj_cls_score.size(1))
            # roi_rel_pairs, roi_pair_proposals, roi_rel_pairs_score, relpn_loss_cls = \
            #     self.RELPN_rpn(rois.data, rel_feats, im_info.data, gt_boxes.data, num_boxes.data, use_gt_boxes)

            # relpn_loss = relpn_loss_cls

            # size_per_batch = x_obj.size(0) / batch_size

            # roi_pair_proposals = roi_pair_proposals + torch.arange(0, batch_size).view(batch_size, 1, 1).type_as(roi_pair_proposals)\
            #     * size_per_batch

            # roi_pair_proposals_v = roi_pair_proposals.view(-1, 2)
            # ind_subject = roi_pair_proposals_v[:, 0]
            # ind_object = roi_pair_proposals_v[:, 1]

            # if self.training:

            #     roi_pair_data = self.RELPN_proposal_target(roi_rel_pairs, gt_boxes.data, num_boxes.data)

            #     # pdb.set_trace()

            #     roi_rel_pairs, rois_rel_label, roi_pair_keep = roi_pair_data
            #     rois_rel_label = Variable(rois_rel_label.view(-1))

            #     roi_pair_keep = roi_pair_keep + torch.arange(0, roi_pair_keep.size(0)).view(roi_pair_keep.size(0), 1).cuda() \
            #                                     * roi_pair_proposals_v.size(0) / batch_size
            #     roi_pair_keep = roi_pair_keep.view(-1).long()

            #     ind_subject = roi_pair_proposals_v[roi_pair_keep][:, 0]
            #     ind_object = roi_pair_proposals_v[roi_pair_keep][:, 1]


            rois_pred = combine_box_pairs(roi_rel_pairs.view(-1, 9))
            rois_pred = Variable(rois_pred)

            # # do roi pooling based on predicted rois
            if cfg.POOLING_MODE == 'crop':
                grid_xy = _affine_grid_gen(rois_pred.view(-1, 5), base_feat.size()[2:], self.grid_size)
                grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
                pooled_pred_feat = self.RELPN_roi_crop(base_feat, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    pooled_pred_feat = F.max_pool2d(pooled_feat, 2, 2)
            elif cfg.POOLING_MODE == 'align':
                pooled_pred_feat = self.RELPN_roi_align(base_feat, rois_pred.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                pooled_pred_feat = self.RELPN_roi_pool(base_feat, rois_pred.view(-1,5))

            # # combine subject, object and relation feature tohether
            x_pred = self._head_to_tail_rel(pooled_pred_feat)

            x_rel = x_pred #torch.cat((x_sobj, x_pred, x_oobj), 1)

            # compute object classification probability
            rel_cls_score = self.RCNN_rel_cls_score(x_rel)
            rel_cls_prob = F.softmax(rel_cls_score)

        if cfg.GCN_ON_FEATS and cfg.GCN_LAYERS > 0:

            if cfg.GCN_HAS_ATTENTION:
                x_sobj = obj_cls_score[ind_subject]
                x_oobj = obj_cls_score[ind_object]
                attend_score = self.GRCNN_gcn_att1(x_sobj, x_oobj) # N_rel x 1
                attend_score = attend_score.view(1, x_pred.size(0))
            else:
                attend_score = Variable(x_rel.data.new(1, x_pred.size(0)).fill_(1))

            # compute the intiial maps, including map_obj_att, map_obj_obj and map_obj_rel
            # NOTE we have two ways to compute map among objects, one way is based on the overlaps among object rois.
            # NOTE the intution behind this is that rois with overlaps should share some common features, we need to
            # NOTE exclude one roi feature from another.
            # NOTE another way is based on the classfication scores. The intuition is that, objects have some common
            # cooccurence, such as bus are more frequently appear on the road.
            assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"

            size_per_batch = x_obj.size(0) / batch_size

            assert x_obj.size() == x_att.size(), "the numbers of object features and attribute features should be the same"
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

            map_sobj_rel = Variable(x_obj.data.new(x_obj.size(0), x_rel.size(0)).zero_())
            map_sobj_rel.scatter_(0, Variable(ind_subject.contiguous().view(1, x_rel.size(0))), attend_score)
            map_oobj_rel = Variable(x_obj.data.new(x_obj.size(0), x_rel.size(0)).zero_())
            map_oobj_rel.scatter_(0, Variable(ind_object.contiguous().view(1, x_rel.size(0))), attend_score)
            map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 1)

            if cfg.MUTE_RELATIONS:
                map_obj_rel.data.zero_()
                x_rel = x_rel.detach()

            mat_phrase = Variable(torch.stack((ind_subject, ind_object), 1))

            # map_obj_rel = Variable(map_obj_rel)

            # x_obj = F.relu(self.fc4obj(x_obj))
            # x_att = F.relu(self.fc4att(x_att))
            # x_pred = F.relu(self.fc4rel(x_pred))
            for i in range(cfg.GCN_LAYERS):
                # pass graph representation to gcn
                x_obj, x_rel = self.imp(x_obj, x_rel, map_obj_rel, mat_phrase)

            # pdb.set_trace()
            # compute object classification loss
            obj_cls_score = self.RCNN_obj_cls_score(x_obj)
            obj_cls_prob = F.softmax(obj_cls_score)

            # compute attribute classification loss
            att_cls_score = self.RCNN_att_cls_score(x_att)
            att_cls_prob = F.softmax(att_cls_score)
            att_cls_log_prob = F.log_softmax(att_cls_score)

            # compute relation classifcation loss
            # x_sobj = x_obj[ind_subject]
            # x_oobj = x_obj[ind_object]
            x_rel = x_pred # torch.cat((x_sobj, x_pred, x_oobj), 1)
            rel_cls_score = self.RCNN_rel_cls_score(x_rel)
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

                # ce_weights = rel_cls_score.data.new(rel_cls_score.size(1)).fill_(1)
                # ce_weights[0] = float(self.rel_bg_cnt) / (rois_rel_label.data.numel() + 1e-5)
                # ce_weights = ce_weights
                rel_label = rois_rel_label.long()
                self.RCNN_loss_rel_cls = F.cross_entropy(rel_cls_score, rel_label)

        rcnn_loss = self.RCNN_loss_bbox + self.RCNN_loss_obj_cls

        if cfg.HAS_ATTRIBUTES and not cfg.MUTE_ATTRIBUTES:
            rcnn_loss += cfg.WEIGHT_ATTRIBUTES * self.RCNN_loss_att_cls

        if cfg.HAS_RELATIONS and not cfg.MUTE_RELATIONS:
            rcnn_loss += cfg.WEIGHT_RELATIONS * self.RCNN_loss_rel_cls

        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        obj_cls_prob = obj_cls_prob.view(batch_size, rois.size(1), -1)
        att_cls_prob = None if not cfg.HAS_ATTRIBUTES else att_cls_prob.view(batch_size, rois.size(1), -1)
        rel_cls_prob = None if not cfg.HAS_RELATIONS else rel_cls_prob.view(batch_size, rel_cls_prob.size(0) / batch_size, -1)

        if self.ext_feat:
            rel_pairs = roi_pair_proposals
            return base_feat, rois.data, rel_pairs, bbox_pred.data, x_obj.data, x_att.data, x_rel.data, \
                    obj_cls_prob.data, att_cls_prob.data, rel_cls_prob.data, \
                    obj_cls_score.data, att_cls_score.data, rel_cls_score.data

        if cfg.HAS_ATTRIBUTES and cfg.HAS_RELATIONS:
            if self.training:
                return rois, bbox_pred, obj_cls_prob, att_cls_prob, rel_cls_prob, rpn_loss, rcnn_loss
            else:
                rel_pairs = roi_pair_proposals
                return rois, rel_pairs, bbox_pred, obj_cls_prob, att_cls_prob, rel_cls_prob, rpn_loss, rcnn_loss
        elif cfg.HAS_ATTRIBUTES:
            return rois, bbox_pred, obj_cls_prob, att_cls_prob, rpn_loss, rcnn_loss
        else:
            return rois, bbox_pred, obj_cls_prob, rpn_loss, rcnn_loss
