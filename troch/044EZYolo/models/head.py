# TinyYOLO Detection Head
# 实现检测头，输出边界框、类别概率和置信度
# 支持多尺度预测


import torch
from torch import nn


class head(nn.Module):
    def __init__(
            self,
            num_classes=80,
            anchors=None,
    ):
        super().__init__()
        if anchors is None:
            anchors = [
                [10, 13],
                [16, 30],
                [33, 23],
                [30, 61],
                [62, 45],
                [59, 119],
                [116, 90],
                [156, 198],
                [373, 326],
            ]
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.num_anchors = len(anchors) // 3

        self.anchors = torch.tensor(anchors).float().view(3, -1, 2)

        out_ch = self.num_anchors * self.num_outputs

        self.head3 = nn.Conv2d(128, out_ch, 1)
        self.head4 = nn.Conv2d(256, out_ch, 1)
        self.head5 = nn.Conv2d(512, out_ch, 1)

    def forward(self, features):
        p3, p4, p5 = features
        out3 = self.head3(p3)
        out4 = self.head4(p4)
        out5 = self.head5(p5)
        return [out3, out4, out5]

    def decode(self, pred, stride, scale_idx=0):
        """
        pred: [B, A*(C+5), H, W]
        stride: 特征图相对于输入图像的步长
        scale_idx: 尺度索引 (0, 1, 2)
        """
        B, _, H, W = pred.shape
        A = self.num_anchors
        C = pred.size(1) // A - 5

        pred = pred.view(B, A, C + 5, H, W).permute(0, 1, 3, 4, 2)
        # x y h w obj cls

        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        h = pred[..., 2]
        w = pred[..., 3]

        obj = torch.sigmoid(pred[..., 4])
        cls = torch.sigmoid(pred[..., 5:])

        grid_x = torch.arange(W).repeat(H, 1)
        grid_y = torch.arange(H).reshape(H, 1).repeat(1, W)

        grid_x = grid_x.to(pred.device)
        grid_y = grid_y.to(pred.device)

        # 获取对应尺度的 anchors
        anchors = self.anchors[scale_idx]  # [A, 2]

        bx = (x + grid_x) * stride
        by = (y + grid_y) * stride
        bw = torch.exp(w) * anchors[:, 0].view(1, A, 1, 1).to(pred.device)
        bh = torch.exp(h) * anchors[:, 1].view(1, A, 1, 1).to(pred.device)

        return bx, by, bw, bh, obj, cls

    def predict(self, features):
        """推理时使用，返回解码后的结果"""
        outputs = self.forward(features)
        strides = [8, 16, 32]
        decoded = []
        for i, (out, stride) in enumerate(zip(outputs, strides)):
            decoded.append(self.decode(out, stride, i))
        return decoded


if __name__ == "__main__":
    # 简单测试
    print("Testing YOLO Head...")

    # 创建检测头
    model = head(num_classes=80)
    print(f"✓ Model created with {model.num_classes} classes")
    print(f"✓ Number of anchors per scale: {model.num_anchors}")

    # 模拟三个尺度的特征图
    batch_size = 2
    p3 = torch.randn(batch_size, 128, 52, 52)  # 大尺度 (小物体)
    p4 = torch.randn(batch_size, 256, 26, 26)  # 中尺度
    p5 = torch.randn(batch_size, 512, 13, 13)  # 小尺度 (大物体)

    features = [p3, p4, p5]

    # 前向传播
    outputs = model(features)
    print(f"\n✓ Forward pass successful!")
    print(f"  Output shapes:")
    for i, out in enumerate(outputs):
        print(f"    Scale {i + 3}: {out.shape}")

    # 测试解码 (测试最小尺度的特征图)
    strides = [8, 16, 32]
    for i, (out, stride) in enumerate(zip(outputs, strides)):
        bx, by, bw, bh, obj, cls = model.decode(out, stride, scale_idx=i)
        print(f"\n✓ Decode scale {i + 3} (stride={stride}) successful!")
        print(f"  Decoded shapes:")
        print(f"    Box x: {bx.shape}")
        print(f"    Box y: {by.shape}")
        print(f"    Box w: {bw.shape}")
        print(f"    Box h: {bh.shape}")
        print(f"    Objectness: {obj.shape}")
        print(f"    Classes: {cls.shape}")

    print("\n✅ All tests passed!")
