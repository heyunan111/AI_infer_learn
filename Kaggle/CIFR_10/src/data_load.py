import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms, models
from PIL import Image
import json


class ImageFolderWithTxt(Dataset):
    def __init__(self, root_dir, txt_path, dictionarySavePath=None, transform=None):
        """
        root_dir: 图片文件夹路径
        txt_path: 标签文件路径
        transform: 图像预处理函数（比如 ToTensor(), Resize(), Normalize() 等）
        """
        self.root_dir = root_dir
        self.transform = transform

        # 读取 txt 文件并尽量把图片名规范化为包含扩展名的文件名
        self.samples = []
        with open(txt_path, "r") as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, label = line.split(",")
                # 如果 img_name 没有扩展名，尝试在 root_dir 中找到对应的文件
                if not os.path.splitext(img_name)[1]:
                    for ext in [".png", ".jpg", ".jpeg"]:
                        candidate = os.path.join(self.root_dir, img_name + ext)
                        if os.path.exists(candidate):
                            img_name = img_name + ext
                            break
                    # 如果没找到，保留原名并在运行时再尝试解析
                self.samples.append((img_name, label))
        all_labels = sorted(set(label for _, label in self.samples))
        self.label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        saveLabelDictionary(self.id_to_label, "")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # 如果路径不存在，尝试补全常见扩展名或搜索匹配文件名
        if not os.path.exists(img_path):
            # 尝试常见扩展
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = (
                    os.path.join(self.root_dir, img_name + ext)
                    if not os.path.splitext(img_name)[1]
                    else None
                )
                if candidate and os.path.exists(candidate):
                    img_path = candidate
                    break
            else:
                # 作为最后手段，尝试以 img_name 为前缀的文件（例如 '123' 匹配 '123.png'）
                try:
                    matches = [
                        p for p in os.listdir(self.root_dir) if p.startswith(img_name)
                    ]
                except FileNotFoundError:
                    matches = []
                if matches:
                    img_path = os.path.join(self.root_dir, matches[0])

        # 打开图像
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image file not found for sample '{img_name}'. Tried path: {img_path}"
            )
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            # Convert PIL image to numpy array for albumentations
            import numpy as np

            image_array = np.array(image)
            transformed = self.transform(image=image_array)
            image = transformed["image"]
        label_id = self.label_to_id[label]
        return {
            "image": image,
            "label": torch.tensor(label_id, dtype=torch.long),
            "filename": img_name,
        }

    def set_transform(self, transform):
        """动态设置transform"""
        self.transform = transform


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

        # 同样在这里尝试补全路径
        if not os.path.exists(img_path):
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = (
                    os.path.join(self.subset.dataset.root_dir, img_name + ext)
                    if not os.path.splitext(img_name)[1]
                    else None
                )
                if candidate and os.path.exists(candidate):
                    img_path = candidate
                    break
            else:
                try:
                    matches = [
                        p
                        for p in os.listdir(self.subset.dataset.root_dir)
                        if p.startswith(img_name)
                    ]
                except FileNotFoundError:
                    matches = []
                if matches:
                    img_path = os.path.join(self.subset.dataset.root_dir, matches[0])

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image file not found for sample '{img_name}' in subset. Tried path: {img_path}"
            )
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # Convert PIL image to numpy array for albumentations
            import numpy as np

            image_array = np.array(image)
            transformed = self.transform(image=image_array)
            image = transformed["image"]

        label_id = self.subset.dataset.label_to_id[label]
        return {
            "image": image,
            "label": torch.tensor(label_id, dtype=torch.long),
            "filename": img_name,
        }


def saveLabelDictionary(dictionary, dictionarySavePath=None):
    with open(dictionarySavePath + "dictionary.json", "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)


def loadLabelDictionary(dictionarySavePath=None):
    with open("dictionary.json", "r", encoding="utf-8") as f:
        loaded_map = json.load(f)
    return loaded_map
