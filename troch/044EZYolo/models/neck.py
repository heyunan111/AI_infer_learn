# TinyYOLO Neck
# 实现特征金字塔网络(FPN)或PANet
# 用于多尺度特征融合

import torch
from torch import nn
from models.backbone import conv


class neck(nn.Module):
    def __init__(self, c3=128, c4=256, c5=512):
        super().__init__()
        self.reduceP5 = conv(c5, c4, 1)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuseP4 = conv(c4 + c4, c4, 3)
        self.reduceP4 = conv(c4, c3, 1)

        self.fuseP3 = conv(c3 + c3, c3, 3)

        self.downP3 = conv(c3, c4, 3, 2)
        self.fuseP4_2 = conv(c4 + c4, c4, 3)
        self.downP4 = conv(c4, c5, 3, 2)

        self.fuseP5_2 = conv(c5 + c5, c5, 3, 1)

    def forward(self, p3, p4, p5):
        p5Reduce = self.reduceP5(p5)
        p5Up = self.up(p5Reduce)
        P4Fuse = self.fuseP4(torch.cat([p5Up, p4], 1))
        p4Reduce = self.reduceP4(P4Fuse)
        p4Up = self.up(p4Reduce)
        p3Fuse = self.fuseP3(torch.cat([p4Up, p3], 1))  # 第一个输出
        p3Down = self.downP3(p3Fuse)
        p4Fuse2 = self.fuseP4_2(torch.cat([p3Down, P4Fuse], 1))  # 第2个输出
        p4Down = self.downP4(p4Fuse2)
        p5Fuse2 = self.fuseP5_2(torch.cat([p4Down, p5], 1))  # 第3个输出

        return p3Fuse, p4Fuse2, p5Fuse2
