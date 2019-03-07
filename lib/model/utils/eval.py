import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model.rpn.bbox_transform import bbox_overlaps
from model.nms.nms_wrapper import nms
from model.utils.config import cfg
import pdb


def eval_objects_recall(gt_annot, obj_rois, obj_scores, top_Ns):

    gt_obj_labels = gt_annot[:, 4].contiguous().view(-1, 1)
    gt_obj_rois = gt_annot[:, :4]    	

    obj_cnt = gt_obj_rois.size(0)
    obj_correct_cnt = torch.zeros(len(top_Ns)).int()

    obj_scores[:, 0].zero_()
    max_obj_scores, max_obj_ind = torch.max(obj_scores, 1)
    obj_scores_final = max_obj_scores
    obj_labels_final = max_obj_ind

    # compute overlaps between gt_obj_rois and pre_obj_rois
    overlaps = bbox_overlaps(obj_rois.contiguous(), gt_obj_rois.contiguous())    

    # sort triplet_scores
    _, order = torch.sort(obj_scores_final, 0, True)

    for idx, top_N in enumerate(top_Ns):
        keep_ind = order[:top_N]
        obj_scores_topN = obj_scores_final[keep_ind]
        obj_rois_topN = obj_rois[keep_ind]
        obj_annot_topN = obj_labels_final[keep_ind]

        for k in range(gt_obj_rois.size(0)):
            gt = gt_obj_labels[k]
            gt_box = gt_obj_rois[k]

            valid_index = (overlaps[keep_ind][:, k] > 0.5).nonzero()

            if len(valid_index.size()) == 0:
                continue

            valid_index = valid_index.squeeze()
            for i in range(valid_index.size(0)):
                obj_label = obj_annot_topN[valid_index[i]]
                if gt[0] == obj_label[0]:
                    obj_correct_cnt[idx] += 1
                    break

	return obj_cnt, obj_correct_cnt


def eval_attribute_recall(gt_annot, obj_rois, obj_scores, att_scores, top_Ns):

    gt_obj_labels = gt_annot[:, 4].contiguous().view(-1, 1)
    gt_obj_rois = gt_annot[:, :4]
    gt_atts = gt_annot[:, 5:21]

    obj_scores[:, 0].zero_()
    max_obj_scores, max_obj_ind = torch.max(obj_scores, 1)
    obj_labels_final = max_obj_ind

    att_scores[:, 0].zero_()
    att_scores_sorted, order = torch.sort(att_scores, 1, True)

    # since the maximal number of attributes for each bbox is 16, we trim att_scores_sorted to 16
    att_scores_sorted_trim = att_scores_sorted[:, :16]
    order_trim = order[:, :16]

    # multiply two scores to get the final scores
    att_scores_final = max_obj_scores * att_scores_sorted_trim

    map_x = np.arange(0, att_scores_final.size(1))
    map_y = np.arange(0, att_scores_final.size(0))
    map_x_g, map_y_g = np.meshgrid(map_x, map_y)
    map_yx = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose()).cuda()

    overlaps = bbox_overlaps(obj_rois.contiguous(), gt_obj_rois.contiguous())    

    att_scores_final_v = att_scores_final.view(-1)
    map_yx_v = map_yx.view(-1, 2)

    _, order = torch.sort(att_scores_final_v, 0, True)

    for idx, top_N in enumerate(top_Ns):
        keep_ind = order[:top_N]

        map_yx_v_kept = map_yx_v[keep_ind]

        obj_kept = map_yx_v_kept[keep_ind, 0]
        att_kept = order_trim[map_yx_v_kept[keep_ind, 1]]

        obj_annot_topN = obj_labels_final[obj_kept]

        for k in range(gt_obj_rois.size(0)):
            gt_obj_label = gt_obj_labels[k]
            gt_box = gt_obj_rois[k]
            gt_att_label = gt_atts[k]

            
            valid_index = (overlaps[obj_kept][:, k] > 0.5).nonzero()

            if len(valid_index.size()) == 0:
                continue

            valid_index = valid_index.squeeze()
            for i in range(valid_index.size(0)):
                obj_label = obj_annot_topN[valid_index[i]]
                att_pos = att_kept[valid_index[i]]                
                if gt_obj_label[0] == obj_label[0] and gt_att_label[att_pos] == 1:
                    obj_correct_cnt[idx] += 1
                    break

	return None


def eval_relations_recall(gt_annot, obj_rois, obj_scores, rel_inds, rel_scores, top_Ns):
    
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
    _, order = torch.sort(max_obj_scores, 0, True)
    obj_scores_ordered = max_obj_scores[order]
    obj_rois_ordered = obj_rois[order]
    keep = nms_detections(obj_rois_ordered, obj_scores_ordered)
    
    notkeep_ind = order.clone().fill_(1)
    notkeep_ind[order[keep.squeeze().long()]] = 0

    notkeep_rels = notkeep_ind[sobj_inds].eq(1) | notkeep_ind[oobj_inds].eq(1)
    # set the 

    rel_scores[:, 0].zero_()
    max_rel_scores, max_rel_ind = torch.max(rel_scores, 1)

    rel_scores_final = max_rel_scores * max_obj_scores[sobj_inds] * max_obj_scores[oobj_inds]
    rel_scores_final[notkeep_rels] = 0

    rel_rois_final = torch.cat((obj_rois[sobj_inds], obj_rois[oobj_inds]), 1)

    max_obj_ind = max_obj_ind.contiguous().view(-1, 1)
    max_rel_ind = max_rel_ind.contiguous().view(-1, 1)
    rel_annot_final = torch.cat((max_obj_ind[sobj_inds], max_rel_ind, max_obj_ind[oobj_inds]), 1)

    # pdb.set_trace()

    # compute overlaps between gt_sobj and pred_sobj
    overlap_sobjs = bbox_overlaps(rel_rois_final[:, :4].contiguous(), gt_rel_rois[:, :4].contiguous())
    # compute overlaps between gt_oobj and pred_oobj
    overlap_oobjs = bbox_overlaps(rel_rois_final[:, 4:].contiguous(), gt_rel_rois[:, 4:].contiguous())
    
    # sort triplet_scores
    _, order = torch.sort(rel_scores_final, 0, True)

    for idx, top_N in enumerate(top_Ns):
        keep_ind = order[:top_N]
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
                if gt[0] == rel[0] and gt[1] == rel[1] and gt[2] == rel[2]:
                    rel_correct_cnt[idx] += 1
                    break

    return rel_cnt, rel_correct_cnt, gt_rel_rois, gt_rel_labels


def eval_graph_recall(gt_annot, obj_rois, obj_scores, att_scores, rel_inds, rel_scores, top_Ns):


	# compute 
	return None