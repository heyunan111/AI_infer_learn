import torch
from torch.utils.data import DataLoader
from pathlib import Path
import time
from utils.loss import YOLOLoss
from utils.dataset import COCODataset
from configs.config import Config
from models.yolo_multiscale import MultiScaleYOLO


def collate_fn(batch):
    """自定义collate函数处理不同数量的标签"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)

    # 将标签转换为 [N, 6] 格式: [batch_idx, class_id, x, y, w, h]
    targets = []
    for batch_idx, label in enumerate(labels):
        if len(label) > 0:
            # label 格式: [class_id, x_center, y_center, w, h] (归一化)
            batch_indices = torch.full((len(label), 1), batch_idx)
            # 转换为像素坐标
            label_px = label.clone()
            label_px[:, 1:] *= 640  # 假设图像大小为640
            targets.append(torch.cat([batch_indices, label_px], dim=1))

    if len(targets) > 0:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 6))

    return imgs, targets


def train():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    dataset = COCODataset(cfg.data_path, cfg.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设为0
        collate_fn=collate_fn,
    )

    # 创建模型
    print("\n创建模型...")
    model = MultiScaleYOLO(num_classes=cfg.num_classes).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 损失函数
    # 将 anchors 展平为列表
    anchors_flat = []
    for anchor_group in cfg.anchors:
        for i in range(0, len(anchor_group), 2):
            anchors_flat.append([anchor_group[i], anchor_group[i + 1]])

    criterion = YOLOLoss(
        anchors=anchors_flat,
        strides=[8, 16, 32],
        num_classes=cfg.num_classes,
        image_size=cfg.img_size,
    )

    print(f"\n开始训练 {cfg.epochs} 个epoch...\n")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 跳过没有目标的batch
            if len(targets) == 0:
                continue

            # 前向传播
            p3, p4, p5 = model(imgs)
            outputs = [p3, p4, p5]

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{cfg.epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time

        print(
            f"\nEpoch {epoch + 1} 完成 - "
            f"平均损失: {avg_loss:.4f} - "
            f"用时: {epoch_time:.2f}秒\n"
        )

        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f"epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存: {save_path}\n")

    # 保存最终模型
    final_path = save_dir / "final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n训练完成！最终模型: {final_path}")


if __name__ == "__main__":
    train()
