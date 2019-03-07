import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from vgg16 import vgg16
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import pdb
from model.rpn.bbox_transform import bbox_overlaps
from model.nms.nms_wrapper import nms
from model.utils.config import cfg

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

# visualize detection results
def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im

# visualize attribute results
def vis_det_att(im, class_name, att_class_names, dets, att_scores, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)

            max_score = np.max(att_scores[i])
            max_index = np.argmax(att_scores[i])
            if max_score > 0.1:
                cv2.putText(im, '%s: %.3f' % (att_class_names[max_index], att_scores[i][max_index]),
                    (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), thickness=1)

    return im

# visualize relation results
def vis_relations(im, class_name, scores, dets, thresh=0.8):
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = scores[i]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 255), thickness=1)
    return im

def vis_gt_relations(im, gt_rel_rois, gt_rel_labels, obj_class_names, rel_class_names):

    gt_rel_labels = gt_rel_labels.astype(int)
    # get union box
    union_box = np.zeros(4)
    union_box[0] = min(gt_rel_rois[0::4])
    union_box[1] = min(gt_rel_rois[1::4])
    union_box[2] = max(gt_rel_rois[2::4])
    union_box[3] = max(gt_rel_rois[3::4])

    gt_rel_rois = tuple(int(np.round(x)) for x in gt_rel_rois)
    union_box = tuple(int(np.round(x)) for x in union_box)

    # first draw subject bbox
    cv2.rectangle(im, gt_rel_rois[0:2], gt_rel_rois[2:4], (0, 204, 0), 1)
    cv2.putText(im, '%s: %.3f' % (obj_class_names[gt_rel_labels[0]], 1), (gt_rel_rois[0], gt_rel_rois[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 0, 255), thickness=1)

    # then draw object bbox
    cv2.rectangle(im, gt_rel_rois[4:6], gt_rel_rois[6:8], (0, 204, 0), 1)
    cv2.putText(im, '%s: %.3f' % (obj_class_names[gt_rel_labels[2]], 1), (gt_rel_rois[4], gt_rel_rois[5] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 0, 255), thickness=1)

    # then draw relation union bbox
    cv2.rectangle(im, union_box[0:2], union_box[2:4], (0, 0, 204), 2)
    cv2.putText(im, '%s: %.3f' % (rel_class_names[gt_rel_labels[1]], 1), (union_box[0], union_box[3] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 0, 255), thickness=1)

    return im


def nms_detections(obj_rois, obj_scores):

    obj_scores = obj_scores.unsqueeze(1)    ###
    # print(obj_rois.shape)     ###
    # print(obj_scores.shape)   ###

    cls_dets = torch.cat((obj_rois, obj_scores), 1)
    keep = nms(cls_dets, 0.9)
    return keep

def eval_relations_recall(gt_annot, obj_rois, obj_scores, rel_inds, rel_scores, top_Ns, use_gt_box = False):
    gt_obj_labels = gt_annot[:, 4].contiguous().view(-1, 1)
    gt_obj_rois = gt_annot[:, :4]
    gt_rels = gt_annot[:, 21:]

    gt_rels_ind = gt_rels.nonzero()

    if len(gt_rels_ind.size()) == 0:
        return 0, torch.zeros(len(top_Ns)).int(), None, None

    gt_rels_view = gt_rels.contiguous().view(-1)

    rel_cnt = gt_rels_ind.size(0)

    rel_correct_cnt = torch.zeros(len(top_Ns)).int()

    gt_pred_labels = gt_rels_view[gt_rels_view.nonzero().squeeze()].contiguous().view(-1, 1)

    gt_rel_rois = torch.cat((gt_obj_rois[gt_rels_ind[:, 0]], gt_obj_rois[gt_rels_ind[:, 1]]), 1)
    gt_rel_labels = torch.cat((gt_obj_labels[gt_rels_ind[:, 0]], gt_pred_labels, gt_obj_labels[gt_rels_ind[:, 1]]), 1)


    obj_scores[:, 0].zero_()
    max_obj_scores, max_obj_ind = torch.max(obj_scores, 1)

    # find the top-N triplets
    sobj_inds = rel_inds[:, 0]
    oobj_inds = rel_inds[:, 1]

    # pdb.set_trace()
    # perform nms on object rois

    use_nms = True
    if use_nms and not use_gt_box:
        _, order = torch.sort(max_obj_scores, 0, True)
        obj_scores_ordered = max_obj_scores[order]
        obj_rois_ordered = obj_rois[order]
        keep = nms_detections(obj_rois_ordered, obj_scores_ordered)

        notkeep_ind = order.clone().fill_(1)
        notkeep_ind[order[keep.squeeze().long()]] = 0

        notkeep_rels = notkeep_ind[sobj_inds].eq(1) | notkeep_ind[oobj_inds].eq(1)

    rel_scores[:, 0].zero_()
    max_rel_scores, max_rel_ind = torch.max(rel_scores, 1)

    rel_scores_final = max_rel_scores * max_obj_scores[sobj_inds] * max_obj_scores[oobj_inds]
    # rel_scores_final = max_rel_scores # * max_obj_scores[sobj_inds] * max_obj_scores[oobj_inds]

    if use_nms and not use_gt_box:
        rel_scores_final[notkeep_rels] = 0

    rel_rois_final = torch.cat((obj_rois[sobj_inds], obj_rois[oobj_inds]), 1)

    max_obj_ind = max_obj_ind.contiguous().view(-1, 1)
    max_rel_ind = max_rel_ind.contiguous().view(-1, 1)
    rel_annot_final = torch.cat((max_obj_ind[sobj_inds], max_rel_ind, max_obj_ind[oobj_inds]), 1)

    # compute overlaps between gt_sobj and pred_sobj
    overlap_sobjs = bbox_overlaps(rel_rois_final[:, :4].contiguous(), gt_rel_rois[:, :4].contiguous())
    # compute overlaps between gt_oobj and pred_oobj
    overlap_oobjs = bbox_overlaps(rel_rois_final[:, 4:].contiguous(), gt_rel_rois[:, 4:].contiguous())

    # sort triplet_scores
    _, order = torch.sort(rel_scores_final, 0, True)

    for idx, top_N in enumerate(top_Ns):
        keep_ind = order[:top_N]
        rel_scores_topN = rel_scores_final[keep_ind]
        rel_rois_topN = rel_rois_final[keep_ind]
        rel_annot_topN = rel_annot_final[keep_ind]

        for k in range(gt_rel_rois.size(0)):
            gt = gt_rel_labels[k]
            gt_box = gt_rel_rois[k]

            valid_index = (((overlap_sobjs[keep_ind][:, k] > 0.5).int() + (overlap_oobjs[keep_ind][:, k] > 0.5).int()) == 2).nonzero()

            if len(valid_index.size()) == 0:
                continue

            # rel_correct_cnt[idx] += 1
            # continue

            valid_index = valid_index.squeeze()
            for i in range(valid_index.size(0)):
                rel = rel_annot_topN[valid_index[i]]
                # if gt[1] == rel[1]:
                #     rel_correct_cnt[idx] += 1
                #     break

                if gt[0] == rel[0] and gt[1] == rel[1] and gt[2] == rel[2]:
                    rel_correct_cnt[idx] += 1
                    break

    return rel_cnt, rel_correct_cnt, gt_rel_rois, gt_rel_labels

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_baseModel(model_name):
    if model_name == "vgg16":
        net = vgg16()
        model_path = 'data/pretrained_model/{}_caffe.pth'.format(model_name)
        net.load_pretrained_cnn(torch.load(model_path))
        return net.slice()
    elif model_name == "resnet50":
        return None

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def _softmax_with_loss(prob, target):
    loss = -prob * target
    loss = loss.sum() / (target.sum() + 1)
    return loss
