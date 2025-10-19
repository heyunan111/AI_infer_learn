# ResNet-50 图像分类框架

## 框架简介

这是一个基于 PyTorch 的 ResNet-50 图像分类框架，专门为叶子分类任务设计。该框架提供了完整的深度学习训练流水线，包括数据处理、模型训练、验证和可视化功能。

### 主要特性

- 🚀 **多种训练策略**：支持单阶段训练、早停训练和两阶段迁移学习
- 📊 **完整的数据管道**：自动化数据加载、预处理和增强
- 🎯 **灵活的配置系统**：支持配置文件和命令行参数
- 📈 **可视化支持**：训练过程可视化和结果分析
- 🔧 **模块化设计**：各组件独立，易于扩展和维护
- ⚡ **GPU 加速**：自动检测和使用 CUDA 设备

## 框架结构

```
src/
├── config/          # 配置管理模块
│   ├── __init__.py
│   └── settings.py  # 配置类和工具函数
├── data/            # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py   # 自定义数据集类
│   ├── manager.py   # 数据管理器
│   └── transforms.py # 数据变换和增强
├── models/          # 模型定义模块
│   ├── __init__.py
│   └── resnet.py    # ResNet 模型实现
├── training/        # 训练相关模块
│   ├── __init__.py
│   ├── evaluation.py # 模型评估器
│   ├── trainer.py   # 训练器实现
│   └── validator.py # 验证器
├── utils/           # 工具函数模块
│   ├── __init__.py
│   ├── helpers.py   # 辅助函数
│   └── visualization.py # 可视化工具
├── __init__.py
└── main.py          # 主程序入口
```

## 快速开始

### 1. 环境要求

```bash
# 必需依赖
torch >= 1.9.0
torchvision >= 0.10.0
Pillow >= 8.0.0
numpy >= 1.21.0

# 可选依赖（用于可视化）
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

### 2. 基本使用

#### 使用默认配置训练

```bash
# 在项目根目录下运行
python src/main.py
```

#### 两阶段迁移学习（推荐）

```bash
python src/main.py --training-strategy two_stage --epochs 50
```

#### 早停训练

```bash
python src/main.py --training-strategy early_stopping --patience 10
```

#### 自定义参数训练

```bash
python src/main.py \
    --data-path classify-leaves \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --training-strategy two_stage \
    --verbose
```

### 3. 配置文件使用

创建配置文件 `config.json`：

```json
{
    "data_path": "classify-leaves",
    "batch_size": 16,
    "num_classes": 176,
    "epochs": 50,
    "learning_rate": 0.001,
    "stage1_epochs": 15,
    "stage2_epochs": 30
}
```

使用配置文件：

```bash
python src/main.py --config config.json
```

## 训练策略详解

### 1. 单阶段训练 (single)
- 直接训练整个网络
- 适合数据量大的场景
- 训练时间较长但效果可能更好

### 2. 早停训练 (early_stopping)
- 监控验证集准确率
- 当性能不再提升时自动停止
- 防止过拟合，节省训练时间

### 3. 两阶段迁移学习 (two_stage) - 推荐
- **阶段1**：冻结主干网络，只训练分类头
- **阶段2**：解冻所有层，进行微调
- 适合小数据集，训练效率高

## 核心模块说明

### Config 模块
- `Config` 类：统一的配置管理
- `load_config()` 函数：从文件加载配置
- `validate_config()` 函数：配置验证

### Data 模块
- `ImageFolderWithTxt` 类：自定义数据集
- `DataManager` 类：数据加载和管理
- `get_transforms()` 函数：数据增强策略

### Models 模块
- `ResNetClassifier` 类：ResNet-50 分类器
- `create_model()` 函数：模型创建
- `freeze_resnet_layers()` 函数：层冻结控制

### Training 模块
- `BaseTrainer` 类：训练器基类
- `TwoStageTrainer` 类：两阶段训练器
- `EarlyStoppingTrainer` 类：早停训练器
- `Validator` 类：模型验证器

### Utils 模块
- `set_seed()` 函数：随机种子设置
- `get_device()` 函数：设备检测
- `save_all_plots()` 函数：结果可视化

## 命令行参数

| 参数                  | 说明             | 默认值          |
| --------------------- | ---------------- | --------------- |
| `--config`            | 配置文件路径     | None            |
| `--data-path`         | 数据目录路径     | classify-leaves |
| `--batch-size`        | 批次大小         | 16              |
| `--epochs`            | 训练轮数         | 50              |
| `--lr`                | 学习率           | 0.001           |
| `--training-strategy` | 训练策略         | two_stage       |
| `--patience`          | 早停耐心值       | 10              |
| `--pretrained`        | 使用预训练模型   | True            |
| `--seed`              | 随机种子         | 42              |
| `--verbose`           | 详细日志         | False           |
| `--save-plots`        | 保存训练图表     | True            |
| `--model-name`        | 模型架构选择     | resnet50        |
| `--dataset-type`      | 数据集类型       | csv             |
| `--annotation-file`   | JSON标注文件路径 | None            |

## 输出文件

训练完成后会生成以下文件：

- `best_model.pth` - 最佳模型权重
- `final_best_model.pth` - 最终模型权重
- `label_mapping.json` - 标签映射文件
- `plots/` - 训练过程可视化图表
- `logs/` - 训练日志文件

## 使用示例

### 完整训练流程

```python
from src.config import create_config_from_original
from src.data import DataManager
from src.models import create_model
from src.training import TwoStageTrainer

# 1. 创建配置
config = create_config_from_original()

# 2. 设置数据
data_manager = DataManager(config)
data_manager.setup_datasets()
data_manager.setup_dataloaders()
train_loader, val_loader = data_manager.get_dataloaders()

# 3. 创建模型
model = create_model(
    num_classes=config.num_classes,
    pretrained=config.pretrained,
    device=config.device
)

# 4. 训练模型
trainer = TwoStageTrainer(model, config, config.device)
results = trainer.train(train_loader, val_loader, None, None, criterion)
```

### 自定义数据集

```python
from src.data import ImageFolderWithTxt

# 创建自定义数据集
dataset = ImageFolderWithTxt(
    root_dir="path/to/images",
    txt_path="path/to/labels.txt",
    transform=transforms
)
```

## 常见问题

### Q: 如何调整训练参数？
A: 可以通过命令行参数或配置文件调整。推荐先使用默认参数，再根据结果调优。

### Q: 内存不足怎么办？
A: 减小 `batch_size` 参数，或者设置 `num_workers=0`。

### Q: 如何使用自己的数据？
A: 框架支持三种数据格式：

**CSV 格式**（默认）：
```
image,label
img1.jpg,class1
img2.jpg,class2
```

**JSON 格式**：
```json
{
  "images": [
    {"file_name": "img1.jpg", "category_id": 0},
    {"file_name": "img2.jpg", "category_id": 1}
  ],
  "categories": [
    {"id": 0, "name": "class1"},
    {"id": 1, "name": "class2"}
  ]
}
```

**目录结构格式**：
```
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class1/
    └── class2/
```

### Q: 训练中断了怎么办？
A: 框架会自动保存最佳模型，可以从保存的权重继续训练。

## 扩展开发

### 添加新模型

#### 1. 创建新模型文件

在 `models/` 目录下创建新的模型文件，例如 `efficientnet.py`：

```python
# models/efficientnet.py
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        # 加载预训练的 EfficientNet
        if pretrained:
            self.backbone = models.efficientnet_b0(pretrained=True)
        else:
            self.backbone = models.efficientnet_b0(pretrained=False)
        
        # 替换分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """冻结主干网络"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻主干网络"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

def create_efficientnet_model(num_classes, pretrained=True, device='cpu'):
    """创建 EfficientNet 模型"""
    model = EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model

def freeze_efficientnet_layers(model, freeze_backbone=True):
    """冻结或解冻 EfficientNet 层"""
    if freeze_backbone:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
```

#### 2. 更新模型模块导出

在 `models/__init__.py` 中添加新模型：

```python
# models/__init__.py
from .resnet import (
    ResNetClassifier,
    create_model,
    freeze_resnet_layers,
    # ... 其他 ResNet 相关函数
)

from .efficientnet import (
    EfficientNetClassifier,
    create_efficientnet_model,
    freeze_efficientnet_layers
)

__all__ = [
    # ResNet 相关
    'ResNetClassifier',
    'create_model',
    'freeze_resnet_layers',
    # EfficientNet 相关
    'EfficientNetClassifier', 
    'create_efficientnet_model',
    'freeze_efficientnet_layers',
    # ... 其他导出
]
```

#### 3. 修改配置支持新模型

在 `config/settings.py` 中添加模型选择配置：

```python
# config/settings.py
@dataclass
class Config:
    # ... 现有配置 ...
    model_name: str = "resnet50"  # 新增：模型名称选择
    
    def __post_init__(self):
        # ... 现有验证逻辑 ...
        
        # 验证模型名称
        valid_models = ["resnet50", "efficientnet_b0"]
        if self.model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")
```

#### 4. 更新主程序支持新模型

在 `main.py` 中修改模型创建逻辑：

```python
# main.py
def setup_data_and_model(config: Config) -> tuple:
    # ... 数据设置代码 ...
    
    # 根据配置创建不同模型
    if config.model_name == "resnet50":
        from models import create_model
        model = create_model(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            device=config.device
        )
    elif config.model_name == "efficientnet_b0":
        from models import create_efficientnet_model
        model = create_efficientnet_model(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            device=config.device
        )
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")
    
    print(f"✅ Model created successfully:")
    print(f"  Architecture: {config.model_name}")
    # ... 其余代码 ...
```

#### 5. 使用新模型

```bash
# 使用 EfficientNet 训练
python src/main.py --model-name efficientnet_b0 --training-strategy two_stage
```

### 添加新数据集

#### 1. 创建自定义数据集类

在 `data/` 目录下创建新的数据集文件，例如 `custom_dataset.py`：

```python
# data/custom_dataset.py
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """自定义图像数据集类 - 支持 JSON 标注格式"""
    
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir: 图片根目录
            annotation_file: JSON 标注文件路径
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 加载 JSON 标注文件
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 构建样本列表
        self.samples = []
        for item in self.annotations['images']:
            img_path = item['file_name']
            label = item['category_id']
            self.samples.append((img_path, label))
        
        # 构建标签映射
        categories = self.annotations['categories']
        self.label_to_id = {cat['name']: cat['id'] for cat in categories}
        self.id_to_label = {cat['id']: cat['name'] for cat in categories}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_id = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)
        
        # 加载图像
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label_id, dtype=torch.long),
            'filename': img_path
        }

class DirectoryDataset(Dataset):
    """目录结构数据集 - 每个子目录代表一个类别"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: 数据根目录，结构如下：
                root_dir/
                ├── class1/
                │   ├── img1.jpg
                │   └── img2.jpg
                └── class2/
                    ├── img3.jpg
                    └── img4.jpg
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 扫描目录结构
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.label_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id_to_label = {idx: cls for cls, idx in self.label_to_id.items()}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_name, img_name)
                        label_id = self.label_to_id[class_name]
                        self.samples.append((img_path, label_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_id = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)
        
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label_id, dtype=torch.long),
            'filename': img_path
        }
```

#### 2. 更新数据管理器

修改 `data/manager.py` 支持新数据集：

```python
# data/manager.py
class DataManager:
    def __init__(self, config):
        self.config = config
        # 添加数据集类型配置
        self.dataset_type = getattr(config, 'dataset_type', 'csv')  # csv, json, directory
        
    def setup_datasets(self):
        """根据配置创建不同类型的数据集"""
        if self.dataset_type == 'csv':
            # 原有的 CSV 数据集逻辑
            self._setup_csv_datasets()
        elif self.dataset_type == 'json':
            # 新的 JSON 数据集逻辑
            self._setup_json_datasets()
        elif self.dataset_type == 'directory':
            # 新的目录数据集逻辑
            self._setup_directory_datasets()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _setup_json_datasets(self):
        """设置 JSON 格式数据集"""
        from .custom_dataset import CustomImageDataset
        
        # 创建训练和验证数据集
        full_dataset = CustomImageDataset(
            root_dir=self.config.data_path,
            annotation_file=self.config.annotation_file,
            transform=None  # 先不设置变换
        )
        
        # 分割训练和验证集
        train_size = int(len(full_dataset) * self.config.train_ratio)
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 设置变换
        train_transform, val_transform = get_transforms(self.config)
        self.train_dataset.dataset.transform = train_transform
        self.val_dataset.dataset.transform = val_transform
    
    def _setup_directory_datasets(self):
        """设置目录结构数据集"""
        from .custom_dataset import DirectoryDataset
        
        train_transform, val_transform = get_transforms(self.config)
        
        # 假设有 train 和 val 子目录
        train_dir = os.path.join(self.config.data_path, 'train')
        val_dir = os.path.join(self.config.data_path, 'val')
        
        self.train_dataset = DirectoryDataset(train_dir, transform=train_transform)
        self.val_dataset = DirectoryDataset(val_dir, transform=val_transform)
```

#### 3. 更新配置支持新数据集

在 `config/settings.py` 中添加数据集配置：

```python
@dataclass
class Config:
    # ... 现有配置 ...
    dataset_type: str = "csv"  # csv, json, directory
    annotation_file: str = ""  # JSON 标注文件路径（当 dataset_type="json" 时使用）
    
    def __post_init__(self):
        # ... 现有验证 ...
        
        # 验证数据集类型
        valid_types = ["csv", "json", "directory"]
        if self.dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}")
        
        # JSON 数据集需要标注文件
        if self.dataset_type == "json" and not self.annotation_file:
            raise ValueError("annotation_file is required for JSON dataset")
```

#### 4. 使用新数据集

```bash
# 使用 JSON 格式数据集
python src/main.py --dataset-type json --annotation-file annotations.json

# 使用目录结构数据集
python src/main.py --dataset-type directory --data-path dataset_root

# 使用自定义配置文件
cat > custom_config.json << EOF
{
    "dataset_type": "directory",
    "data_path": "my_custom_dataset",
    "model_name": "efficientnet_b0",
    "batch_size": 32
}
EOF

python src/main.py --config custom_config.json
```

### 添加新的训练策略

1. 继承 `BaseTrainer` 类
2. 实现 `train()` 方法
3. 在 `main.py` 中注册新策略

### 添加新的数据增强

1. 在 `data/transforms.py` 中添加新的变换
2. 在配置中启用新的增强策略

## 技术支持

如果遇到问题，请检查：

1. 数据路径是否正确
2. 依赖包是否完整安装
3. CUDA 环境是否配置正确
4. 内存和显存是否充足

更多详细信息请参考代码注释和测试用例。