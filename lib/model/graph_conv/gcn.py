import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from proposal_layer import _ProposalLayer
from anchor_target_layer import _AnchorTargetLayer
from model.utils.network import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _GraphConvolutionLayer(nn.Module):
    """ graph convolutional layer """
    def __init__(self, dim):
        super(_GraphConvolutionLayer, self).__init__()

        
        # transformation matrix for gcn
        self.fc = nn.Sequential(
            nn.Linear(dim, dim)
        )

        # we directly use F.nonlinear during forward

    def forward(self, Input, I, C):
        # Input: input node features, NxD
        # I: index of K nearest neighbors
        # C: edge affinity NxK (to K-nearest neighbor)

        # KNN
        Input_KNN = []
        for i in range(I,size(0)):
            Input_KNN.append(Input[I[i]])

        Input_KNN = torch.stack(Input_KNN, 0)

        N, K, D = Input_KNN.size(0), Input_KNN.size(1), Input_KNN.size(2)

        Iv = Input_KNN.view(N * K, D)

        # Dim: (N*k, D)
        O = self.fc(Iv)

        # Dim: (N*k, D)
        Ov = O.view(N, K, D)

        # Dim: (N, K)
        C_exp = C.unsqueeze(2).expand_as(Ov)

        # Dim: (N, K, D), (N, K, D)
        Out = Ov * C_exp

        # Dim: (N, K, D)
        self.output = F.sigmoid(Out.sum(1))

        # Dim: (N, D)
        return self.output