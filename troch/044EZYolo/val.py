# 验证脚本
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from utils.dataset import COCODataset
from configs.config import Config
from models.yolo_multiscale import MultiScaleYOLO
from utils.metrics import Evaluator


def collate_fn(batch):
    """自定义collate函数"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, labels


def decode_predictions(outputs, anchors, strides, conf_thresh=0.25, img_size=640):
    """
    解码模型输出为检测框
    outputs: [p3, p4, p5]
    返回: list of [N, 6] (x, y, w, h, conf, cls)
    """
    all_predictions = []

    for batch_idx in range(outputs[0].shape[0]):
        predictions = []

        for scale_idx, (pred, stride) in enumerate(zip(outputs, strides)):
            B, _, H, W = pred.shape
            A = 3
            C = (pred.shape[1] // A) - 5

            # reshape
            pred = pred.view(B, A, C + 5, H, W).permute(0, 1, 3, 4, 2)
            pred_batch = pred[batch_idx]  # [A, H, W, C+5]

            # 提取预测
            xy = torch.sigmoid(pred_batch[..., 0:2])
            wh = pred_batch[..., 2:4]
            obj = torch.sigmoid(pred_batch[..., 4])
            cls = torch.sigmoid(pred_batch[..., 5:])

            # 网格
            grid_x = torch.arange(W, device=pred.device).repeat(H, 1).float()
            grid_y = (
                torch.arange(H, device=pred.device).reshape(H, 1).repeat(1, W).float()
            )

            # 解码坐标
            anchor_scale = anchors[scale_idx]
            for a in range(A):
                # 解码
                bx = (xy[a, :, :, 0] + grid_x) * stride
                by = (xy[a, :, :, 1] + grid_y) * stride
                bw = torch.exp(wh[a, :, :, 0]) * anchor_scale[a, 0]
                bh = torch.exp(wh[a, :, :, 1]) * anchor_scale[a, 1]

                conf = obj[a]
                cls_prob = cls[a]

                # 过滤低置信度
                mask = conf > conf_thresh
                if mask.sum() == 0:
                    continue

                # 提取有效预测
                bx_valid = bx[mask]
                by_valid = by[mask]
                bw_valid = bw[mask]
                bh_valid = bh[mask]
                conf_valid = conf[mask]
                cls_prob_valid = cls_prob[mask]

                # 获取类别
                cls_conf, cls_idx = cls_prob_valid.max(dim=-1)
                final_conf = conf_valid * cls_conf

                # 组合
                pred_boxes = torch.stack(
                    [
                        bx_valid,
                        by_valid,
                        bw_valid,
                        bh_valid,
                        final_conf,
                        cls_idx.float(),
                    ],
                    dim=1,
                )
                predictions.append(pred_boxes)

        if len(predictions) > 0:
            predictions = torch.cat(predictions, 0)
        else:
            predictions = torch.zeros((0, 6), device=outputs[0].device)

        all_predictions.append(predictions)

    return all_predictions


@torch.no_grad()
def validate(model, dataloader, cfg, device):
    """验证函数"""
    model.eval()
    evaluator = Evaluator(num_classes=cfg.num_classes, conf_thresh=0.25, iou_thresh=0.5)

    # anchors
    anchors_flat = []
    for anchor_group in cfg.anchors:
        for i in range(0, len(anchor_group), 2):
            anchors_flat.append([anchor_group[i], anchor_group[i + 1]])

    anchors = torch.tensor(anchors_flat).float().view(3, 3, 2).to(device)
    strides = [8, 16, 32]

    print("\n开始验证...")
    for imgs, labels in tqdm(dataloader, desc="验证"):
        imgs = imgs.to(device)

        # 前向传播
        p3, p4, p5 = model(imgs)
        outputs = [p3, p4, p5]

        # 解码预测
        predictions = decode_predictions(outputs, anchors, strides, conf_thresh=0.25)

        # 处理每个样本
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if len(label) > 0:
                # 转换标签为像素坐标
                label_px = label.clone()
                label_px[:, 1:] *= cfg.img_size
                evaluator.process_batch(pred, label_px)

    # 计算指标
    metrics = evaluator.compute_metrics()
    print(f"\n验证结果:")
    print(f"  mAP@0.5: {metrics['mAP']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    return metrics


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载验证集
    print("\n加载验证集...")
    val_dataset = COCODataset(cfg.data_path, cfg.img_size)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 加载模型
    print("\n加载模型...")
    model = MultiScaleYOLO(num_classes=cfg.num_classes).to(device)

    # 加载权重
    checkpoint_path = Path(cfg.save_dir) / "final.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"已加载权重: {checkpoint_path}")
    else:
        print(f"警告: 未找到权重文件 {checkpoint_path}")

    # 验证
    validate(model, val_dataloader, cfg, device)


if __name__ == "__main__":
    main()
