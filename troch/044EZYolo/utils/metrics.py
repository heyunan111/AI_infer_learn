# 评估指标
import torch
import numpy as np


def box_iou(box1, box2):
    """
    计算两组框的IoU
    box1: [N, 4] (x, y, w, h)
    box2: [M, 4] (x, y, w, h)
    返回: [N, M]
    """
    # 转换为 x1, y1, x2, y2
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    # 交集
    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # 并集
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


def compute_ap(recall, precision):
    """计算AP (Average Precision)"""
    # 在开头和结尾添加哨兵值
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 计算precision包络
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算曲线下面积
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


class Evaluator:
    """YOLO评估器"""

    def __init__(self, num_classes, conf_thresh=0.25, iou_thresh=0.5):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        """重置统计"""
        self.stats = []  # (correct, conf, pred_cls, target_cls)

    def process_batch(self, predictions, targets):
        """
        处理一个batch的预测和目标
        predictions: [N, 6] (x, y, w, h, conf, cls)
        targets: [M, 5] (cls, x, y, w, h)
        """
        if len(predictions) == 0:
            if len(targets) > 0:
                self.stats.append(
                    (
                        torch.zeros(0, dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        targets[:, 0],
                    )
                )
            return

        # 过滤低置信度
        predictions = predictions[predictions[:, 4] > self.conf_thresh]

        if len(predictions) == 0:
            if len(targets) > 0:
                self.stats.append(
                    (
                        torch.zeros(0, dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        targets[:, 0],
                    )
                )
            return

        # 提取预测信息
        pred_boxes = predictions[:, :4]
        pred_conf = predictions[:, 4]
        pred_cls = predictions[:, 5]

        # 提取目标信息
        target_cls = targets[:, 0]
        target_boxes = targets[:, 1:5]

        # 计算IoU
        iou = box_iou(pred_boxes, target_boxes)

        # 匹配预测和目标
        correct = torch.zeros(len(predictions), dtype=torch.bool, device=iou.device)

        if len(targets) > 0:
            # 对每个目标找最佳预测
            for i in range(len(targets)):
                # 找到类别匹配且IoU最大的预测
                mask = pred_cls == target_cls[i]
                if mask.sum() > 0:
                    iou_match = iou[mask, i]
                    if iou_match.max() > self.iou_thresh:
                        j = iou_match.argmax()
                        pred_idx = torch.where(mask)[0][j]
                        correct[pred_idx] = True

        self.stats.append((correct, pred_conf, pred_cls, target_cls))

    def compute_metrics(self):
        """计算mAP等指标"""
        if len(self.stats) == 0:
            return {"mAP": 0.0, "precision": 0.0, "recall": 0.0}

        # 合并所有batch的统计
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            correct, conf, pred_cls, target_cls = stats
            # 按置信度排序
            i = np.argsort(-conf)
            correct, conf, pred_cls = correct[i], conf[i], pred_cls[i]

            # 计算每个类别的AP
            unique_classes = np.unique(target_cls)
            ap_per_class = []

            for c in unique_classes:
                # 该类别的预测和目标
                pred_mask = pred_cls == c
                target_count = (target_cls == c).sum()

                if pred_mask.sum() == 0 or target_count == 0:
                    continue

                # 累积TP和FP
                tp = np.cumsum(correct[pred_mask])
                fp = np.cumsum(~correct[pred_mask])

                # 计算precision和recall
                recall = tp / (target_count + 1e-16)
                precision = tp / (tp + fp + 1e-16)

                # 计算AP
                ap = compute_ap(recall, precision)
                ap_per_class.append(ap)

            # 计算mAP
            mAP = np.mean(ap_per_class) if len(ap_per_class) > 0 else 0.0

            # 计算总体precision和recall
            tp_total = correct.sum()
            fp_total = (~correct).sum()
            fn_total = len(target_cls) - tp_total

            precision = tp_total / (tp_total + fp_total + 1e-16)
            recall = tp_total / (tp_total + fn_total + 1e-16)

            return {
                "mAP": float(mAP),
                "precision": float(precision),
                "recall": float(recall),
            }
        else:
            return {"mAP": 0.0, "precision": 0.0, "recall": 0.0}
