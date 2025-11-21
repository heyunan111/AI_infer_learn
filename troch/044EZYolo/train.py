import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time

from models.yolo import SimpleYOLO
from utils.dataset import COCODataset
from configs.config import Config

def collate_fn(batch):
    """自定义collate函数处理不同数量的标签"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, labels

def train():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print('\n加载数据集...')
    dataset = COCODataset(cfg.data_path, cfg.img_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设为0
        collate_fn=collate_fn
    )
    
    # 创建模型
    print('\n创建模型...')
    model = SimpleYOLO(num_classes=cfg.num_classes).to(device)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # 简单的损失函数（先用MSE，后续可以改进）
    criterion = nn.MSELoss()
    
    print(f'\n开始训练 {cfg.epochs} 个epoch...\n')
    
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # 前向传播
            outputs = model(imgs)
            
            # 简单的损失计算（这里只是示例，实际YOLO损失更复杂）
            # 真实项目需要实现YOLO的完整损失函数
            target = torch.zeros_like(outputs).to(device)
            loss = criterion(outputs, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch [{epoch+1}/{cfg.epochs}] '
                      f'Batch [{batch_idx}/{len(dataloader)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch {epoch+1} 完成 - '
              f'平均损失: {avg_loss:.4f} - '
              f'用时: {epoch_time:.2f}秒\n')
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f'epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存: {save_path}\n')
    
    # 保存最终模型
    final_path = save_dir / 'final.pth'
    torch.save(model.state_dict(), final_path)
    print(f'\n训练完成！最终模型: {final_path}')

if __name__ == '__main__':
    train()
