from ast import pattern
import imp
import torch
import torch.nn as nn
from torch.autograd import Variable
from Classification_Branch import Cls_Branch
from Regression_Branch import Rgs_Branch
import numpy as np


class PL4A(nn.Module):
    def __init__(self, pattern_dim=256, num_pattern=20, cls_out_dim=6, tau=1, topK=4, drop_rate=0):
        super(PL4A, self).__init__()
        self.pattern = torch.randn(num_pattern, pattern_dim)
        self.patterns = Variable(self.pattern, requires_grad=True).cuda()
        self.cls_branch = Cls_Branch(out_dim=cls_out_dim, pattern_dim=pattern_dim, tau=tau)
        self.rgs_branch = Rgs_Branch(dim=pattern_dim, topK=topK, dropout=drop_rate)

    def forward(self, features):
        a_hat, cls_result = self.cls_branch(feat=features, patterns=self.patterns)
        pred_score = self.rgs_branch(patterns=self.patterns, feature=features, a_hat=a_hat)
        return  cls_result, pred_score


