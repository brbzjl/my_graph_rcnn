# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from bbox_transform import bbox_transform, bbox_overlaps, bbox_overlaps_batch2, bbox_transform_batch2
import pdb

DEBUG = False

class _ProposalTargetLayer_MSDN(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses_obj, nclasses_att, nclasses_rel):
        super(_ProposalTargetLayer_MSDN, self).__init__()
        self._num_obj_classes = nclasses_obj
        self._num_att_classes = nclasses_att
        self._num_rel_classes = nclasses_rel

        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size(0), gt_boxes.size(1), all_rois.size(2)).zero_()
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]
        for i in range(gt_boxes.size(0)):
            gt_boxes_append[i, :, 0] = i

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        labels, labels_att, labels_rel, rois, roi_pairs, roi_pairs_proposal, bbox_targets, bbox_inside_weights = \
            self._sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, roi_pairs, roi_pairs_proposal, labels, labels_att, labels_rel, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """

        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch2(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps, all_rois_zero, gt_boxes_zero = bbox_overlaps_batch2(all_rois, gt_boxes[:, :, :5])

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,4].contiguous().view(-1).index(offset.view(-1))\
                                                            .view(batch_size, -1)

        fg_mask = max_overlaps >= cfg.TRAIN.FG_THRESH

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        if cfg.HAS_ATTRIBUTES:
            labels_att_batch = labels.new(batch_size, rois_per_image, self._num_att_classes).zero_()
            # labels_att_batch = labels.new(batch_size, rois_per_image, 16).fill_(-1)

        if cfg.HAS_RELATIONS:
            labels_rel_batch = labels.new(batch_size, rois_per_image, 1).zero_()
            roi_pairs_batch = all_rois.new(batch_size, rois_per_image, 9).zero_()
            roi_pairs_proposal = all_rois.new(batch_size, rois_per_image, 2).zero_()

        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                # rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).long().cuda()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).long().cuda()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).long().cuda()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).long().cuda()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                print("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                pdb.set_trace()

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            # print(fg_rois_per_this_image)

            if fg_rois_per_this_image < labels_batch[i].size(0):
                labels_batch[i][fg_rois_per_this_image:] = 0

            # copy attribute labels to labels_att_batch
            if cfg.HAS_ATTRIBUTES:
                # att_labels = gt_boxes[i][gt_assignment[i][fg_inds]][:, 5:21].int()
                # labels_att_batch[i][:fg_rois_per_this_image].copy_(att_labels)
                # replace_index = (labels_att_batch[i][:fg_rois_per_this_image, 1:] == 0)
                # labels_att_batch[i][:fg_rois_per_this_image, 1:][replace_index] = -1

                att_labels = gt_boxes[i][gt_assignment[i][fg_inds]][:, 5:21].int()
                # att_labels size: fg_rois_per_this_image x 16
                for j in range(att_labels.size(0)):
                    for k in range(att_labels.size(1)):
                        if k == 0 and att_labels[j][k] == 0:
                            labels_att_batch[i, j, 0] = 1                        
                        if att_labels[j][k] > 0:
                            labels_att_batch[i, j, att_labels[j][k]] = 1

            if cfg.HAS_RELATIONS:
                rel_per_image = rois_per_image
                if fg_inds.size(0) > 0:
                    gt_relationships = gt_boxes[i][:, 21:]
                    id_i, id_j = np.meshgrid(xrange(fg_inds.size(0)), xrange(fg_inds.size(0)), indexing='ij') # Grouping the input object rois
                    id_i = torch.from_numpy(id_i.reshape(-1)).type_as(fg_inds)
                    id_j = torch.from_numpy(id_j.reshape(-1)).type_as(fg_inds)
                    pair_labels = gt_relationships[gt_assignment[i][fg_inds[id_i]], gt_assignment[i][fg_inds[id_j]]]

                    if pair_labels.sum() > 0:
                        fg_id_rel = pair_labels.nonzero().view(-1)
                        rel_fg_num = fg_id_rel.size(0)
                        rel_fg_num = int(min(np.round(rel_per_image * cfg.TRAIN.FG_FRACTION), rel_fg_num))
                        bg_id_rel = (pair_labels == 0).nonzero().view(-1)
                    else:
                        rel_fg_num = 0
                    # print 'rel_fg_num'
                    # print rel_fg_num
                    if rel_fg_num > 0:
                        fg_id_rel = torch.from_numpy(npr.choice(fg_id_rel.cpu().numpy(), size=rel_fg_num, replace=False)).type_as(fg_inds)
                        rel_labels_fg = pair_labels[fg_id_rel]
                        sub_assignment_fg = id_i[fg_id_rel]
                        obj_assignment_fg = id_j[fg_id_rel]                        
                        sub_list_fg = fg_inds[sub_assignment_fg]
                        obj_list_fg = fg_inds[obj_assignment_fg]

                        labels_rel_batch[i][:rel_fg_num, 0].copy_(rel_labels_fg)

                        rel_bg_num = rel_per_image - rel_fg_num

                        # bg_id_rel = torch.from_numpy(npr.choice(bg_id_rel.cpu().numpy(), size=rel_bg_num, replace=False)).type_as(bg_inds)
                        # sub_assignment_bg = id_i[bg_id_rel]
                        # obj_assignment_bg = id_j[bg_id_rel] 
                        # sub_list_bg = keep_inds[sub_assignment_bg]
                        # obj_list_bg = keep_inds[obj_assignment_bg]
                        sub_assignment_bg = torch.from_numpy(npr.choice(xrange(keep_inds.size(0)), size=rel_bg_num, replace=True)).type_as(bg_inds)
                        obj_assignment_bg = torch.from_numpy(npr.choice(xrange(keep_inds.size(0)), size=rel_bg_num, replace=True)).type_as(bg_inds)
                        sub_list_bg = keep_inds[sub_assignment_bg]
                        obj_list_bg = keep_inds[obj_assignment_bg]

                        sub_assignment = torch.cat((sub_assignment_fg, sub_assignment_bg), 0)
                        obj_assignment = torch.cat((obj_assignment_fg, obj_assignment_bg), 0)

                        sub_list = torch.cat((sub_list_fg, sub_list_bg), 0)
                        obj_list = torch.cat((obj_list_fg, obj_list_bg), 0)
                    else:
                        rel_bg_num = rel_per_image
                        sub_assignment = torch.from_numpy(npr.choice(xrange(keep_inds.size(0)), size=rel_bg_num, replace=True)).type_as(keep_inds)
                        obj_assignment = torch.from_numpy(npr.choice(xrange(keep_inds.size(0)), size=rel_bg_num, replace=True)).type_as(keep_inds)
                        sub_list = keep_inds[sub_assignment]
                        obj_list = keep_inds[obj_assignment]

                    roi_pairs = roi_pairs_batch.new(rel_per_image, 9).zero_()
                    roi_pairs[:, 1:5].copy_(all_rois[i][sub_list][:, 1:5])
                    roi_pairs[:, 5:9].copy_(all_rois[i][obj_list][:, 1:5])                    
                    roi_pairs[:, 0] = i
                    roi_pairs_batch[i].copy_(roi_pairs)

                    roi_pairs_proposal[i].copy_(torch.stack((sub_assignment, obj_assignment), 1) + i * rois_per_image)

            rois_batch[i].copy_(all_rois[i][keep_inds])
            rois_batch[i,:,0] = i

            gt_rois_batch[i].copy_(gt_boxes[i][gt_assignment[i][keep_inds]][:, :5])

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch)

        return labels_batch, labels_att_batch, labels_rel_batch, rois_batch, roi_pairs_batch, roi_pairs_proposal, bbox_targets, bbox_inside_weights
