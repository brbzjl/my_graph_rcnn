import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
import numpy as np
import math
import pdb
import time

class _GraphAttentionLayer(nn.Module):
    """ graph attention layer """
    def __init__(self, dim):
        super(_GraphAttentionLayer, self).__init__()

        self.Att_linear_sub = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            )

        self.Att_linear_obj = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            )

    def forward(self, sub, obj):

        # roi_feat: N x D
        assert sub.size() == obj.size(), "sub and obj should have the same size"
        N = sub.size(0)
        D = sub.size(1)

        # we do not back-prop through this path
        x_sub = self.Att_linear_sub(sub.detach())
        x_obj = self.Att_linear_obj(obj.detach())

        # N x D, D x N ==> N x N
        x_dot = x_sub * x_obj

        att_score = F.sigmoid(x_dot.sum(1)) # (N * N)

        return att_score.view(-1)
