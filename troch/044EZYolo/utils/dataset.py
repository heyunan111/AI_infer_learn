import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class COCODataset(Dataset):
    def __init__(self, data_path, img_size=640):
        self.img_size = img_size
        self.img_path = Path(data_path) / 'images' / 'train2017'
        self.label_path = Path(data_path) / 'labels' / 'train2017'
        
        # 获取所有图片
        self.img_files = sorted(self.img_path.glob('*.jpg'))
        print(f'找到 {len(self.img_files)} 张图片')
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 加载图片
        img_file = self.img_files[idx]
        img = Image.open(img_file).convert('RGB')
        
        # 加载标签
        label_file = self.label_path / f'{img_file.stem}.txt'
        labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    labels.append([float(x) for x in line.strip().split()])
        
        # 简单resize
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
        
        return img, labels
