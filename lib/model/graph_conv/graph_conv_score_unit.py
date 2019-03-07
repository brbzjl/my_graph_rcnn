import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
import numpy as np
import math
import pdb
import time
import pdb
from model.utils.config import cfg

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class _Collection_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.fc_source = nn.Linear(dim_in, 32, bias=True)
        self.fc_target = nn.Linear(dim_out, 32, bias=True)
    # def forward(self, target, source, attention_base):
    #     assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
    #
    #     # Ninxdin -> Ninx32
    #     emb_source = self.fc_source(F.relu(source))
    #     # Noutxdout -> Noutx32
    #     emb_target = self.fc_target(F.relu(target))
    #
    #     # NoutxNin
    #     attention_prob = F.softmax(torch.mm(emb_target, emb_source.t())) * attention_base
    #
    #     fc_out = self.fc(F.relu(source))
    #
    #     collect = torch.mm(attention_prob, fc_out)
    #     return collect

    def forward(self, target, source, attention_base):
        # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
        fc_out = self.fc(F.relu(source))
        collect = torch.mm(attention_base, fc_out)
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
        return collect_avg

class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()
        self.fc_source = nn.Linear(dim, dim, bias=True)
        self.fc_target = nn.Linear(dim, dim, bias=True)
        normal_init(self.fc_source, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.fc_target, 0, 0.01, cfg.TRAIN.TRUNCATED)

    # def forward(self, target, source):
    #     assert target.size() == source.size(), "source dimension must be equal to target dimension"
    #     update = self.fc_target(F.relu(target)) + self.fc_source(F.relu(source))
    #     return update

    def forward(self, target, source):
        assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update

class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    def __init__(self, dim_obj, dim_att, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_obj, dim_att)) # att from obj
        self.collect_units.append(_Collection_Unit(dim_att, dim_obj)) # obj from att
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj)) # obj from obj
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (subject) from rel
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (object) from rel
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (subject)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (object)

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection

class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """
    def __init__(self, dim_obj, dim_att, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_att)) # att from others
        self.update_units.append(_Update_Unit(dim_obj)) # obj from others
        self.update_units.append(_Update_Unit(dim_rel)) # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update
