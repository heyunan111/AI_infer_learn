import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageFolderWithTxt(Dataset):
    def __init__(self, root_dir, txt_path, transform=None):
        """
        root_dir: 图片文件夹路径
        txt_path: 标签文件路径
        transform: 图像预处理函数（比如 ToTensor(), Resize(), Normalize() 等）
        """
        self.root_dir = root_dir
        self.transform = transform

        # 读取 txt 文件，存成 [(img_path, label), ...]
        self.samples = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, label = line.split(',')
                self.samples.append((img_name, label))
        all_labels = sorted(set(label for _, label in self.samples))
        self.label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # 打开图像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_id = self.label_to_id[label]
        return {
            "image": image,
            "label": torch.tensor(label_id, dtype=torch.long),
            "filename": img_name
        }
    
    def set_transform(self, transform):
        """动态设置transform"""
        self.transform = transform
        
data_path = "classify-leaves"

# 为训练集和测试集设置不同的transform
# 创建包装类来处理不同的transform
class TransformDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
            
    def __len__(self):
        return len(self.subset)
            
    def __getitem__(self, idx):
        item = self.subset[idx]
        # 重新应用transform
        img_name, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img_path = os.path.join(self.subset.dataset.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
            
        if self.transform:
            image = self.transform(image)
                
        label_id = self.subset.dataset.label_to_id[label]
        return {
            "image": image,
            "label": torch.tensor(label_id, dtype=torch.long),
            "filename": img_name
        }

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

train_ratio = 0.8  # 80% 训练集
test_ratio = 0.2   # 20% 测试集

# TODO 6: 实现训练函数 train_one_epoch()
def train(net, train_loader, criterion, optimizer, scheduler, epochs=100):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0  # 修正1: 移除重复定义
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')  # 修正2: 简化进度条
        
        for i, data in enumerate(pbar):  # 修正3: 使用pbar
            inputs = data["image"]
            labels = data["label"]
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}], Loss: {running_loss/100:.4f}, Acc: {100*correct/total:.2f}%')
                running_loss = 0.0
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_acc = 100 * correct / total
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} finished. LR: {current_lr:.6f}, Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
    
    print("Train Finished")
    return epoch_acc, avg_epoch_loss

# TODO 7: 实现验证函数 validate()

def validate(net,test_loader,criterion):
    net.eval()
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for data in test_loader :
            inputs = data["image"]
            labels = data["label"]
            
            inputs,labels = inputs.to(device),labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(test_loader)
    print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy, avg_val_loss

def train_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                             epochs=50, patience=10, min_delta=0.001):
    """
    带早停的训练函数
    """
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data in pbar:
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # 验证
        val_acc, val_loss = validate(model, test_loader, criterion)
        
        # 记录指标
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch+1}: LR={current_lr:.6f}, Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
        
        # 早停检查
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'💾 保存最佳模型，验证准确率: {val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'⏰ 早停计数器: {patience_counter}/{patience}')
            
        if patience_counter >= patience:
            print(f'🛑 早停触发！最佳验证准确率: {best_val_acc:.2f}%')
            break
    
    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# TODO 10: 模型评估和可视化
def plot_training_history(history):
    """
    绘制训练历史
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='Train Loss', color='blue')
    ax1.plot(history['val_losses'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_accs'], label='Train Acc', color='blue')
    ax2.plot(history['val_accs'], label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 损失差异
    loss_diff = [abs(t-v) for t, v in zip(history['train_losses'], history['val_losses'])]
    ax3.plot(loss_diff, color='green')
    ax3.set_title('Train-Val Loss Difference')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.grid(True)
    
    # 准确率差异
    acc_diff = [abs(t-v) for t, v in zip(history['train_accs'], history['val_accs'])]
    ax4.plot(acc_diff, color='orange')
    ax4.set_title('Train-Val Accuracy Difference')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, criterion, class_names):
    """
    详细评估模型性能
    """
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Evaluating'):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各种指标
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"测试集结果:")
    print(f"准确率: {accuracy:.2f}%")
    print(f"平均损失: {avg_test_loss:.4f}")
    print(f"总样本数: {len(all_labels)}")
    
    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names[:len(set(all_labels))]))
    
    return {
        'accuracy': accuracy,
        'test_loss': avg_test_loss,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def freeze_model_layers(model, freeze_backbone=True):
    """
    冻结或解冻模型层
    Args:
        model: ResNet模型
        freeze_backbone: True=冻结backbone(特征提取阶段), False=解冻所有层(微调阶段)
    """
    for name, param in model.named_parameters():
        if freeze_backbone:
            # 阶段1：只有最后的分类层(fc)可训练
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            # 阶段2：所有层都可训练
            param.requires_grad = True
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")

def two_stage_training_with_best_model(model, train_loader, test_loader, criterion, device):
    """
    两阶段训练函数 - 每个阶段都保存最佳模型
    """
    print("=" * 60)
    print("开始两阶段训练（带最佳模型保存）")
    print("=" * 60)
    
    # ============ 阶段1：特征提取 ============
    print("\n🔥 阶段1：特征提取训练（冻结backbone）")
    print("-" * 40)
    
    freeze_model_layers(model, freeze_backbone=True)
    
    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001, betas=(0.9, 0.999)
    )
    scheduler_stage1 = optim.lr_scheduler.StepLR(optimizer_stage1, step_size=5, gamma=0.5)
    
    # 阶段1：使用带早停的训练
    stage1_history = train_with_early_stopping(
        model, train_loader, test_loader, criterion, 
        optimizer_stage1, scheduler_stage1, 
        epochs=15, patience=5, min_delta=0.001
    )
    
    best_val_acc_stage1 = stage1_history['best_val_acc']
    print(f"✅ 阶段1完成！最佳验证准确率: {best_val_acc_stage1:.2f}%")
    
    # 加载阶段1最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # ============ 阶段2：微调 ============
    print("\n🚀 阶段2：微调训练（解冻所有层）")
    print("-" * 40)
    
    freeze_model_layers(model, freeze_backbone=False)
    
    # 为阶段2构建带分组学习率的优化器：
    # - backbone（除fc外的参数）使用较低学习率
    # - 分类头（model.fc）使用较高学习率
    optimizer_stage2 = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01, betas=(0.9, 0.999))

    # 学习率调度器（与阶段1保持一致的策略）
    scheduler_stage2 = optim.lr_scheduler.StepLR(optimizer_stage2, step_size=5, gamma=0.5)
    
    # 阶段2：使用带早停的训练
    stage2_history = train_with_early_stopping(
        model, train_loader, test_loader, criterion, 
        optimizer_stage2, scheduler_stage2, 
        epochs=30, patience=8, min_delta=0.001
    )
    
    best_val_acc_stage2 = stage2_history['best_val_acc']
    
    print(f"\n🎉 两阶段训练完成！")
    print(f"阶段1最佳准确率: {best_val_acc_stage1:.2f}%")
    print(f"阶段2最佳准确率: {best_val_acc_stage2:.2f}%")
    print(f"总提升: {best_val_acc_stage2 - best_val_acc_stage1:.2f}%")
    
    # 重命名最终模型
    torch.save(model.state_dict(), 'final_best_model.pth')
    
    return {
        'stage1_acc': best_val_acc_stage1,
        'stage2_acc': best_val_acc_stage2,
        'stage1_history': stage1_history,
        'stage2_history': stage2_history
    }

if __name__ == "__main__":
    # 创建完整数据集
    full_dataset = ImageFolderWithTxt(
        root_dir="classify-leaves",
        txt_path="classify-leaves/train.csv",
        transform=None  # 先不设置transform
    )
    
    # 分割数据集
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size]
    )
    
    # 应用不同的transform
    train_dataset = TransformDataset(train_dataset, train_transform)
    test_dataset = TransformDataset(test_dataset, val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,  # 验证集通常不需要shuffle
        num_workers=0,
    )
    
    print(f"数据集信息:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {len(full_dataset.label_to_id)}")
    print(f"类别列表: {list(full_dataset.label_to_id.keys())[:10]}...")  # 显示前10个类别
    
    model = models.resnet50(pretrained = True)
    num_classes = 176
    num_features   = model.fc.in_features
    model.fc = nn.Linear(num_features,num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)


    criterion = nn.CrossEntropyLoss()

    # two_stage_training_with_best_model(model, train_loader, test_loader, criterion, device)
    model.load_state_dict(torch.load('final_best_model.pth'))
    model.eval()  # 设置为评估模式

# 创建示例输入
    # 确保示例输入与模型在同一设备，避免 CPU/GPU 类型不匹配
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 根据你的输入尺寸调整

# 导出ONNX模型
    torch.onnx.export(
        model,                  # 要导出的模型
        dummy_input,            # 模型输入（示例）
        "model.onnx",           # 输出文件名
        export_params=True,     # 是否导出模型参数
        opset_version=11,       # ONNX算子集版本
        do_constant_folding=True,  # 是否进行常量折叠优化
        input_names=['input'],   # 输入节点名称
        output_names=['output'], # 输出节点名称
        dynamic_axes={          # 动态维度配置
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)