import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv
from ..basic.dcn import DeformableConv2d


class DCN(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 kernel_size=1, 
                 padding=0,
                 stride=1):
        super().__init__()
        self.dcn = nn.Sequential(
            DeformableConv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dcn(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_dim=512, expand_ratio=0.25):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.attn = nn.Sequential(
            Conv(in_dim, inter_dim, k=1),
            DCN(inter_dim, inter_dim, kernel_size=3, padding=1),
            Conv(inter_dim, in_dim, k=1)
        )

    def forward(self, x):
        return x + self.attn(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x + x * self.sigmoid(self.attn(x))


class DynamicAttention(nn.Module):
    def __init__(self, 
                 in_dim=2048,
                 out_dim=512,
                 expand_ratio=0.25, 
                 nblocks=1):
        super(DynamicAttention, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act=False),
            Conv(out_dim, out_dim, k=3, p=1, act=False)
        )
        # spatial attention
        attn_blocks = []
        for _ in range(nblocks):
            attn_blocks.append(SpatialAttention(out_dim, expand_ratio))
            attn_blocks.append(ChannelAttention(out_dim))

        self.attn_blocks = nn.Sequential(*attn_blocks)


    def forward(self, x):
        """
            in_feats: (List of Tensor) contains C3, C4 and C5 feature maps.
        """
        x = self.projector(x)
        x = self.attn_blocks(x)

        return self.attn_blocks(x)
