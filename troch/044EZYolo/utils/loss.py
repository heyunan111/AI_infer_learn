# 损失函数
# 实现YOLO损失：边界框损失(CIoU/GIoU) + 分类损失 + 置信度损失
# 正负样本匹配策略

import torch


# CIoU = IoU
#       - (中心距离 / 外接框对角线)
#       - α * (宽高比惩罚)


def bbox_ciou(box1, box2, eps=1e-7):
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2

    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area
    iou = inter_area / (union_area + eps)

    center_dist = (box1[..., 0] - box2[..., 0]) ** 2 + (
        box1[..., 1] - box2[..., 1]
    ) ** 2

    # 中心距离
    center_dist = (box1[..., 0] - box2[..., 0]) ** 2 + (
        box1[..., 1] - box2[..., 1]
    ) ** 2

    # 包围框
    x1 = torch.min(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    y1 = torch.min(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    x2 = torch.max(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    y2 = torch.max(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)

    c2 = (x2 - x1) ** 2 + (y2 - y1) ** 2 + eps

    # 宽高比
    v = (4 / (3.1415926**2)) * torch.pow(
        torch.atan(box1[..., 2] / box1[..., 3])
        - torch.atan(box2[..., 2] / box2[..., 3]),
        2,
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - center_dist / c2 - alpha * v


def wh2xyxy(box):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
    x_c, y_c, w, h = box.unbind(-1)
    b1_x1 = x_c - w / 2
    b1_y1 = y_c - h / 2
    b1_x2 = x_c + w / 2
    b1_y2 = y_c + h / 2
    return torch.stack((b1_x1, b1_y1, b1_x2, b1_y2), dim=-1)


def xyxy2wh(box):
    """Convert [x1, y1, x2, y2] to [x_center, y_center, w, h]"""
    x1, y1, x2, y2 = box.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return torch.stack((x_c, y_c, w, h), dim=-1)


import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    YOLO 模型的损失函数实现。
    负责目标分配、标签编码和损失计算。
    """

    def __init__(self, anchors: list, strides: list, num_classes: int, image_size=640):
        """
        初始化损失函数所需的参数和组件。

        参数:
            anchors (list): 所有尺度的锚框尺寸列表，例如 [[10, 13], ..., [373, 326]]。
            strides (list): 每个特征图的步长，例如 [8, 16, 32]。
            num_classes (int): 数据集中的类别数量。
        """
        super().__init__()

        # 核心参数
        self.num_classes = num_classes
        self.num_anchors = len(anchors) // len(strides)

        # 锚框和步长
        # self.anchors 应该被重塑为 [3, A, 2] 的格式，方便按尺度索引
        self.anchors = torch.tensor(anchors).float().view(len(strides), -1, 2)
        self.strides = strides
        self.num_outputs = num_classes + 5  # 每个锚框的输出维度：5 (x,y,w,h,obj) + C

        # 损失权重（超参数）
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5

        # 损失组件实例化
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.ciou_loss = bbox_ciou

        self.image_size = image_size

    def forward(self, p_preds: list, targets: torch.Tensor) -> torch.Tensor:
        """
        YOLO Loss 框架（模板）

        参数:
            p_preds: 模型输出列表，每个元素形状 [B, A*(C+5), H, W]
            targets: GT 标签，[N, 6] = [b, cls, x, y, w, h]，x,y,w,h归一化

        返回:
            total_loss: 总损失
        """
        device = p_preds[0].device
        total_loss = torch.zeros(1, device=device)

        # 遍历每个尺度
        for scale_idx, pred_raw in enumerate(p_preds):
            # ===============================
            # Step 1: 准备预测张量
            # ===============================
            stride = self.strides[scale_idx]
            anchors_i = self.anchors[scale_idx].to(device)  # [A, 2]
            B, _, H, W = pred_raw.shape
            A = self.num_anchors
            C = self.num_classes

            # reshape: [B, A*(C+5), H, W] -> [B, A, H, W, C+5]
            pred = pred_raw.view(B, A, C + 5, H, W).permute(0, 1, 3, 4, 2)

            # 提取预测值
            xy_pred = pred[..., 0:2]  # [B, A, H, W, 2]
            wh_pred = pred[..., 2:4]  # [B, A, H, W, 2]
            obj_pred = pred[..., 4]  # [B, A, H, W]
            cls_pred = pred[..., 5:]  # [B, A, H, W, C]

            # 网格坐标
            grid_x = torch.arange(W, device=device).repeat(H, 1).float()
            grid_y = torch.arange(H, device=device).reshape(H, 1).repeat(1, W).float()

            # ===============================
            # Step 2: GT 映射到当前尺度
            # ===============================
            # targets 已经是像素坐标
            gt_xy = targets[:, 2:4]
            gt_wh = targets[:, 4:6]

            # 缩放到 grid
            gt_xy_scaled = gt_xy / stride
            gt_wh_scaled = gt_wh / stride

            # ===============================
            # Step 3: Anchor 匹配
            # ===============================
            ratios = gt_wh_scaled[:, None, :] / anchors_i[None, :, :]  # [N, A, 2]
            ratios = torch.max(ratios, 1 / ratios).max(dim=2)[0]  # [N, A]
            anchor_thresh = 4.0
            mask = ratios < anchor_thresh

            # ===============================
            # Step 4: 构造目标张量
            # ===============================
            t_obj = torch.zeros_like(obj_pred)  # [B, A, H, W]
            t_cls = torch.zeros_like(cls_pred)  # [B, A, H, W, C]
            t_box = torch.zeros(B, A, H, W, 4, device=device)  # [B, A, H, W, 4]

            for idx, target in enumerate(targets):
                b = int(target[0])
                cls_id = int(target[1])
                gx, gy = gt_xy_scaled[idx]
                gi, gj = int(gx.item()), int(gy.item())

                # 边界检查
                if gi >= W or gj >= H or gi < 0 or gj < 0:
                    continue

                # 遍历匹配的 anchor
                for a in range(A):
                    if mask[idx, a]:
                        # 注意：gj 是行(y)，gi 是列(x)
                        t_box[b, a, gj, gi, 0] = gx - gi  # x offset
                        t_box[b, a, gj, gi, 1] = gy - gj  # y offset
                        t_box[b, a, gj, gi, 2] = (
                            gt_wh_scaled[idx, 0] / anchors_i[a, 0]
                        ).log()
                        t_box[b, a, gj, gi, 3] = (
                            gt_wh_scaled[idx, 1] / anchors_i[a, 1]
                        ).log()

                        t_obj[b, a, gj, gi] = 1.0
                        t_cls[b, a, gj, gi, cls_id] = 1.0

            # ===============================
            # Step 5: 计算损失
            # ===============================
            # 置信度损失
            L_obj = self.bce_loss(obj_pred, t_obj).mean()

            # 分类损失（正样本）
            pos_mask = t_obj > 0
            if pos_mask.sum() > 0:
                L_cls = self.bce_loss(cls_pred[pos_mask], t_cls[pos_mask]).mean()
            else:
                L_cls = torch.zeros(1, device=device)

            # 坐标损失（CIoU）
            if pos_mask.sum() > 0:
                # decode 预测框到 grid 坐标
                xy_decoded = torch.sigmoid(xy_pred)
                grid_xy = torch.stack([grid_x, grid_y], dim=-1)[
                    None, None, :, :, :
                ]  # [1, 1, H, W, 2]
                pred_xy = xy_decoded + grid_xy  # [B, A, H, W, 2]
                pred_wh = (
                    torch.exp(wh_pred) * anchors_i[None, :, None, None, :]
                )  # [B, A, H, W, 2]
                pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)  # [B, A, H, W, 4]

                # decode 目标框到 grid 坐标
                t_xy = t_box[..., 0:2] + grid_xy
                t_wh = torch.exp(t_box[..., 2:4]) * anchors_i[None, :, None, None, :]
                t_boxes = torch.cat([t_xy, t_wh], dim=-1)

                # 只对正样本计算 CIoU
                pred_boxes_pos = pred_boxes[pos_mask]
                t_boxes_pos = t_boxes[pos_mask]
                ciou = bbox_ciou(pred_boxes_pos, t_boxes_pos)
                L_box = (1.0 - ciou).mean()
            else:
                L_box = torch.zeros(1, device=device)

            # ===============================
            # Step 6: 合并总损失
            # ===============================
            loss_i = (
                self.lambda_box * L_box
                + self.lambda_obj * L_obj
                + self.lambda_cls * L_cls
            )
            total_loss += loss_i

        return total_loss


def main():
    # --- 步骤 2: 定义参数 ---

    # 模拟输入图像尺寸
    IMG_SIZE = 640
    # 批次大小
    BATCH_SIZE = 4
    # 类别数 (例如 COCO 的 80 类)
    NUM_CLASSES = 3
    # 锚框参数 (需要与您的 head 初始化中的 anchors 匹配)
    ANCHORS = [
        [10, 13],
        [16, 30],
        [33, 23],  # P3/S8
        [30, 61],
        [62, 45],
        [59, 119],  # P4/S16
        [116, 90],
        [156, 198],
        [373, 326],  # P5/S32
    ]
    STRIDES = [8, 16, 32]

    # --- 步骤 3: 实例化 YOLOLoss ---

    criterion = YOLOLoss(anchors=ANCHORS, strides=STRIDES, num_classes=NUM_CLASSES)

    print(f"--- YOLO 损失函数测试启动 ---")
    print(
        f"输入尺寸: {IMG_SIZE}x{IMG_SIZE}, 批次大小: {BATCH_SIZE}, 类别数: {NUM_CLASSES}"
    )

    # --- 步骤 4: 模拟模型输出 (p_preds) ---

    # P3: H/8, W/8 = 80x80
    H_P3, W_P3 = IMG_SIZE // 8, IMG_SIZE // 8
    # P4: H/16, W/16 = 40x40
    H_P4, W_P4 = IMG_SIZE // 16, IMG_SIZE // 16
    # P5: H/32, W/32 = 20x20
    H_P5, W_P5 = IMG_SIZE // 32, IMG_SIZE // 32

    A = len(ANCHORS) // len(STRIDES)  # 3
    C5 = NUM_CLASSES + 5

    p_preds = [
        # 启用 requires_grad=True 模拟模型输出
        torch.randn(BATCH_SIZE, A * C5, H_P3, W_P3, requires_grad=True),
        torch.randn(BATCH_SIZE, A * C5, H_P4, W_P4, requires_grad=True),
        torch.randn(BATCH_SIZE, A * C5, H_P5, W_P5, requires_grad=True),
    ]

    print(f"\n模拟预测张量形状:")
    print(f"P3 (8x): {list(p_preds[0].shape)}")
    print(f"P4 (16x): {list(p_preds[1].shape)}")
    print(f"P5 (32x): {list(p_preds[2].shape)}")

    # --- 步骤 5: 模拟真实标签 (targets) ---

    # 假设当前批次有 7 个真实物体，每个物体由 6 个参数定义：
    # [batch_idx, class_id, x_center, y_center, w, h] (这里假设 x,y,w,h 是绝对像素坐标)
    targets_data = [
        [0, 1, 150, 200, 30, 40],  # Batch 0, 目标 1
        [0, 2, 500, 550, 80, 120],  # Batch 0, 目标 2
        [1, 0, 300, 320, 50, 50],  # Batch 1, 目标 3
        [2, 1, 10, 10, 15, 15],  # Batch 2, 目标 4 (小目标)
        [2, 2, 600, 600, 100, 100],  # Batch 2, 目标 5 (大目标)
        [3, 0, 400, 400, 20, 20],  # Batch 3, 目标 6
        [3, 1, 450, 450, 150, 150],  # Batch 3, 目标 7
    ]

    # 转换为 PyTorch Tensor
    targets = torch.tensor(targets_data).float()
    print(f"\n模拟真实标签 targets 形状: {list(targets.shape)}")

    # --- 步骤 6: 调用损失函数和反向传播 ---

    try:
        total_loss = criterion(p_preds, targets)

        # 模拟训练步骤：清零梯度 -> 反向传播 -> 更新权重 (这里只执行前两步)
        print(f"计算得到的 Total Loss: {total_loss.item():.4f}")

        # 模拟反向传播
        total_loss.backward()
        print("Loss.backward() 成功执行。")

    except Exception as e:
        print(f"\n❌ 损失函数执行失败！错误信息:")
        print(e)
        print("\n请检查您的 YOLOLoss 内部的张量操作和索引逻辑。")


if __name__ == "__main__":
    main()
