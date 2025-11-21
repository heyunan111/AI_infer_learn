# TinyYOLO Backbone
# 实现轻量级的特征提取网络
# 使用卷积层、残差块等构建backbone

import torch
from torch import nn


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=None, s=1):
        super().__init__()
        # 支持两种参数风格: kernel_size/stride 或 k/s
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.siLu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.siLu(x)
        return x


class residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, 1)
        self.conv2 = conv(out_channels, out_channels, 3)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class c3(nn.Module):
    def __init__(self, in_C, out_C, resBlockNum=1, n=None):
        super().__init__()
        # 支持两种参数风格: resBlockNum 或 n
        n = n if n is not None else resBlockNum

        hidden = out_C // 2
        self.conv1 = conv(in_C, hidden, 1)
        self.conv2 = conv(in_C, hidden, 1)
        self.residual = nn.Sequential(*[residual(hidden, hidden) for _ in range(n)])
        self.conv3 = conv(hidden * 2, out_C, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.residual(x1)

        x2 = self.conv2(x)

        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)


class MiniCSPDarknet(nn.Module):
    """Mini CSPDarknet - 多尺度特征提取（返回P3, P4, P5）"""

    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            conv(3, 32, k=3, s=2),      # 640→320
            conv(32, 64, k=3, s=2),     # 320→160
            c3(64, 64, resBlockNum=1),
        )

        self.stage2 = nn.Sequential(
            conv(64, 128, k=3, s=2),    # 160→80
            c3(128, 128, resBlockNum=2),
        )

        self.stage3 = nn.Sequential(
            conv(128, 256, k=3, s=2),   # 80→40
            c3(256, 256, resBlockNum=3),
        )

        self.stage4 = nn.Sequential(
            conv(256, 512, k=3, s=2),   # 40→20
            c3(512, 512, resBlockNum=1),
        )

    def forward(self, x):
        x = self.stage1(x)              # 160×160
        P3 = self.stage2(x)             # 80×80, 128 channels
        P4 = self.stage3(P3)            # 40×40, 256 channels
        P5 = self.stage4(P4)            # 20×20, 512 channels
        return P3, P4, P5


if __name__ == '__main__':
    print("=" * 50)
    print("测试 MiniCSPDarknet (多尺度)")
    print("=" * 50)
    model2 = MiniCSPDarknet()
    x = torch.randn(1, 3, 640, 640)
    P3, P4, P5 = model2(x)
    print(f'输入: {x.shape}')
    print(f'P3 (80x80):  {P3.shape}')
    print(f'P4 (40x40):  {P4.shape}')
    print(f'P5 (20x20):  {P5.shape}')
    print(f'参数量: {sum(p.numel() for p in model2.parameters()) / 1e6:.2f}M')
