import torch
import torch.nn as nn

import numpy as np

class Rgs_Branch(nn.Module):
    def __init__(self, dim, topK, dropout=0.):
        
        super(Rgs_Branch, self).__init__()
        self.topk = topK
        # Attention
        self.wkv = nn.Linear(dim, dim * 2, bias=False)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.scale = 1. / dim ** 0.5
        # LayerNormalization
        self.norm = nn.LayerNorm(dim)
        # FeedForward
        hidden_dim = dim
        self.ffc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
        # Regressor
        self.regressor = nn.Linear(dim, 1)

    def forward(self, patterns, feature, a_hat):
        batch_size, t, dim = feature.shape
        _, topk_pattern = torch.topk(a_hat, self.topk, sorted=True)  # torch.topk: return values and indices

        # obtain pattern subsets
        pattern_subset = None
        for b in range(batch_size):                 # for each batch
            subset4batch = patterns[topk_pattern[b]]
            if pattern_subset is None:
                pattern_subset = subset4batch
            else:
                pattern_subset = torch.cat([pattern_subset, subset4batch])
        pattern_subset = pattern_subset.cuda()      # B, k, D

        # Attention mecanism
        # project features and pattern_subset to KQV
        kv = self.wkv(feature).reshape(batch_size, t, 2, -1)    # B, T, 2, dim
        k, v = kv.permute(2, 0, 1, 3)                           # B, T, dim
        q = self.wq(pattern_subset)                             # B, K, dim
        # attention calculation
        dot = (q @ k.transpose(-2, -1)) * self.scale            # B, K, T
        attention = dot.softmax(-1)                             # B, K, T
        output_attn = attention @ v                             # B, K, dim
        # residual
        output_attn = output_attn + pattern_subset              # B, K, dim

        # FFC
        output_tr = self.ffc(output_attn) + output_attn         # B, K, dim

        # Feature aggregation and Regression
        fused_feature = output_tr.mean(1)                       # B, dim
        pred_score = self.regressor(fused_feature)
        return pred_score
