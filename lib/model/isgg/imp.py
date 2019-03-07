import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import pdb
from imp_base import _IMP_BASE

### Iterative message passing

class _IMP(_IMP_BASE):
	def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
		super(_IMP, self).__init__(fea_size, dropout, gate_width, use_kernel_function=use_kernel_function)

	def forward(self, feature_obj, feature_phrase, mps_object, mps_phrase):

		# mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
		# mps_phrase [phrase_batchsize, 2]

		# object updating
		object_sub = self.prepare_message(feature_obj, feature_phrase, mps_object[:, 0, :], self.gate_edge2vert)
		object_obj = self.prepare_message(feature_obj, feature_phrase, mps_object[:, 1, :], self.gate_edge2vert)
		GRU_input_feature_object = object_sub + object_obj
		# out_feature_object = feature_obj + self.GRU_object(GRU_input_feature_object, feature_obj)
		out_feature_object = self.vert_rnn(GRU_input_feature_object, feature_obj)

		# phrase updating
		indices_sub = mps_phrase[:, 0].detach()
		indices_obj = mps_phrase[:, 1].detach()
		fea_sub2pred = torch.index_select(feature_obj, 0, indices_sub)
		fea_obj2pred = torch.index_select(feature_obj, 0, indices_obj)
		phrase_sub = self.gate_vert2edge(feature_phrase, fea_sub2pred)
		phrase_obj = self.gate_vert2edge(feature_phrase,  fea_obj2pred)
		GRU_input_feature_phrase =  phrase_sub + phrase_obj

		# out_feature_phrase = feature_phrase + self.GRU_phrase(GRU_input_feature_phrase, feature_phrase)
		out_feature_phrase = self.vert_rnn(GRU_input_feature_phrase, feature_phrase)

		return out_feature_object, out_feature_phrase