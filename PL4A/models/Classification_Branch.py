import torch
import torch.nn as nn

import numpy as np

class Cls_Branch(nn.Module):
    def __init__(self, out_dim, pattern_dim=1024, tau=1):
        super(Cls_Branch, self).__init__()
        self.out_dim = out_dim
        self.classifier = nn.Linear(pattern_dim, out_dim)
        self.tau = tau

    def forward(self, feat, patterns):
        """
        feat: B x 10 x 1024
        patterns: N x pattern_dim(1024)
        """
        
        # L2 norm
        feat_norm = feat / (torch.norm(feat, dim=2).unsqueeze(-1))      # B, T, D
        pattern_norm = patterns / (torch.norm(patterns, dim=1).unsqueeze(-1)) 
        pattern_norm = pattern_norm.unsqueeze(0)    # 1, N, pattern_dim
        # cosine similarity
        a_similarity = feat_norm @ pattern_norm.transpose(1,2)
        a_similarity = a_similarity.softmax(dim=-1) # B, T, N
        # obtain a_hat
        a_hat = torch.sum(a_similarity, dim=1)      # B, N
        # obtain cls_feature
        cls_feature = a_hat.softmax(-1) @ patterns  # B, pattern_dim 
        # fc head for classification
        pred_result = self.classifier(cls_feature)               # B, out_dim
        # print('here')
        return a_hat, pred_result
