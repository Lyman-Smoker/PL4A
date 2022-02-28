import torch.nn as nn
import torch
from .I3D import I3D
import logging


class I3D_backbone(nn.Module):
    def __init__(self, I3D_class, gap=True):
        '''
        I3D_class: number of classes that need to be classified
        gap: whether to use global average pooling
        '''
        super(I3D_backbone, self).__init__()
        print('Using I3D backbone')
        self.backbone = I3D(num_classes=I3D_class, modality='rgb', dropout_prob=0.5)
        self.gap = gap

    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
        print('loading ckpt done')

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, target):
        start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        video_pack = torch.cat([target[:, :, i: i + 16] for i in start_idx])  # 10*B, C, 16, H, W
        # print(video_pack.shape)
        # pass the backbone
        total_feature = self.backbone(video_pack).reshape(10, len(target), -1).transpose(0, 1)  # B, 10, 1024
        # print(total_feature.shape)
        if self.gap:        # if using global average pooling
            total_feature = total_feature.mean(1)
        return total_feature


if __name__ == '__main__':
    base_model = I3D_backbone(I3D_class=400)
    # base_model.load_pretrain('/hdd/1/liyuanming/ComputerVision/pretrained_models/model_rgb.pth')
    input = torch.randn((2, 3, 103, 224, 224))
    out = base_model(input)
    print(out.shape)
