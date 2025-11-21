import torch
from models.backbone import MiniCSPDarknet
from models.neck import neck

# 测试 Backbone + Neck
print("=" * 60)
print("测试 Backbone + Neck (FPN + PAN)")
print("=" * 60)

# 创建模型
backbone = MiniCSPDarknet()
neck_model = neck(c3=128, c4=256, c5=512)

# 输入
x = torch.randn(1, 3, 640, 640)
print(f"输入: {x.shape}")
print()

# Backbone 输出
p3, p4, p5 = backbone(x)
print("Backbone 输出:")
print(f"  P3: {p3.shape}")
print(f"  P4: {p4.shape}")
print(f"  P5: {p5.shape}")
print()

# Neck 输出
p3_out, p4_out, p5_out = neck_model(p3, p4, p5)
print("Neck 输出 (FPN + PAN):")
print(f"  P3_out: {p3_out.shape}")
print(f"  P4_out: {p4_out.shape}")
print(f"  P5_out: {p5_out.shape}")
print()

# 验证尺寸
assert p3_out.shape == (1, 128, 80, 80), f"P3_out shape错误: {p3_out.shape}"
assert p4_out.shape == (1, 256, 40, 40), f"P4_out shape错误: {p4_out.shape}"
assert p5_out.shape == (1, 512, 20, 20), f"P5_out shape错误: {p5_out.shape}"

print("✓ 所有尺寸验证通过!")
print()

# 参数量统计
backbone_params = sum(p.numel() for p in backbone.parameters()) / 1e6
neck_params = sum(p.numel() for p in neck_model.parameters()) / 1e6
total_params = backbone_params + neck_params

print(f"参数量统计:")
print(f"  Backbone: {backbone_params:.2f}M")
print(f"  Neck:     {neck_params:.2f}M")
print(f"  Total:    {total_params:.2f}M")
