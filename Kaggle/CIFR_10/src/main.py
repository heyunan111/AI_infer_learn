from data_load import ImageFolderWithTxt
from data_load import TransformDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split, DataLoader
from train import train
import torch
from torchvision import models
import torch.nn as nn


def loadDataSet():
    fullDataset = ImageFolderWithTxt(
        "C:\\Users\\27427\\Desktop\\code\\AI_infer_learn\\Kaggle\\CIFR_10\\train\\train",
        "trainLabels.csv",
        ".",
    )
    total_size = len(fullDataset)
    train_ratio = 0.8
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(fullDataset, [train_size, test_size])

    # 返回为元组，保持顺序一致（训练集, 测试集）
    return train_dataset, test_dataset


def get_transforms():
    """
    返回CIFAR数据集的数据增强变换

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=40, min_width=40, p=1.0),
            A.RandomCrop(height=32, width=32, p=1.0),
            A.HorizontalFlip(p=0.5),
            # 只保留基础的数据增强
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform


def main():
    trainDataSet, testDataSet = loadDataSet()
    trainTransform, testTransform = get_transforms()
    trainDataSet, testDataSet = (
        TransformDataset(trainDataSet, trainTransform),
        TransformDataset(testDataSet, testTransform),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用预训练权重可以提供更好的初始化
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_classes = 10
    # 替换最后一层以适应CIFAR-10的10个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device=device)  # Move to device after modifying fc layer
    criterion = nn.CrossEntropyLoss()

    trainLoader = DataLoader(
        trainDataSet,
        batch_size=64,  # 适中的batch size
        shuffle=True,
        num_workers=0,
    )

    testLoader = DataLoader(
        testDataSet,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    # 回到AdamW但调整参数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,  # 保持原来的学习率
        weight_decay=0.01,  # 添加适度的权重衰减
    )

    # 使用更温和的学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,  # 总epoch数
        eta_min=1e-6,
    )
    print(
        train(
            model,
            trainLoader,
            testLoader,
            criterion,
            optimizer,
            scheduler,
            device,
            epochs=100,  # 减少epoch数先看效果
            patience=15,
            min_delta=0.001,
        )
    )


if __name__ == "__main__":
    main()
