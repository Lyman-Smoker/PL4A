import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class PatternMiner(nn.Module):
    def __init__(self, pattern_dim=256, num_pattern=20):
        super(PatternMiner, self).__init__()
        self.pattern = torch.randn(num_pattern, pattern_dim)
        self.patterns = Variable(self.pattern, requires_grad=True).cuda()

    def forward(self):
        return


def build_pattern_miner(pattern_dim=256, num_pattern=20):
    pattern = torch.randn(num_pattern, pattern_dim)
    patterns = Variable(pattern, requires_grad=True)
    return patterns