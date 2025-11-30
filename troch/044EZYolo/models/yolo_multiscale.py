import torch
import torch.nn as nn
from models.backbone import MiniCSPDarknet, conv
from models.neck import neck
from models.head import head


class MultiScaleYOLO(nn.Module):
    """多尺度YOLO检测器 - 使用MiniCSPDarknet"""

    def __init__(self, num_classes=80, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

        # Backbone - 输出多尺度特征
        self.backbone = MiniCSPDarknet()
        self.head = head()
        self.neck = neck()

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        p3, p4, p5 = self.head((p3,p4,p5))
        return p3,p4,p5


if __name__ == '__main__':
    import sys

    sys.path.append('.')

    print("=" * 60)
    print("测试 MultiScaleYOLO")
    print("=" * 60)

    model = MultiScaleYOLO(num_classes=80)
    x = torch.randn(1, 3, 640, 640)

    out_P3, out_P4, out_P6 = model(x)

    print(f'输入: {x.shape}')
    print(f'\n多尺度输出:')
    print(f'  P3 (大目标, 80x80):  {out_P3.shape}')
    print(f'  P4 (中目标, 40x40):  {out_P4.shape}')
    print(f'  P6 (小目标, 10x10):  {out_P6.shape}')
    print(f'\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # 计算总检测框数量
    total_predictions = (
            out_P3.shape[2] * out_P3.shape[3] * 3
            + out_P4.shape[2] * out_P4.shape[3] * 3
            + out_P6.shape[2] * out_P6.shape[3] * 3
    )
    print(f'总预测框数量: {total_predictions}')
