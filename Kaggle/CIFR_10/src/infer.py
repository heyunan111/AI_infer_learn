import torch
from torchvision import models
import torch.nn as nn
import os
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from data_load import loadLabelDictionary
import numpy as np
import glob
from tqdm import tqdm

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载 - 只执行一次
model = models.resnet18(weights="IMAGENET1K_V1")
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device=device)
model.eval()

# 更新后的CUDA优化设置（新API）
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    # 使用新的TF32设置API
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

# 数据预处理
env_transform = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ToTensorV2(),
])

# 路径设置
root_path = "C:/Users/27427/Desktop/code/AI_infer_learn/Kaggle/CIFR_10/test/test"
submission_file = "C:/Users/27427/Desktop/code/AI_infer_learn/Kaggle/CIFR_10/sampleSubmission.csv"
dictionary_path = "C:/Users/27427/Desktop/code/AI_infer_learn/Kaggle/CIFR_10/dictionary.json"

# 预加载标签字典 - 只执行一次
label_dict = loadLabelDictionary(dictionary_path)

def load_and_preprocess_images_batch(image_paths, transform, batch_size=32):
    """批量加载和预处理图像"""
    images = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="预处理图像"):
        try:
            full_path = os.path.join(root_path, f"{img_path}.png")
            image = Image.open(full_path).convert("RGB")
            image_np = np.array(image)
            image_tensor = transform(image=image_np)['image']
            images.append(image_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # 如果图像数量为0，返回空列表
    if not images:
        return [], []
    
    # 分批处理
    batched_images = []
    batched_paths = []
    
    for i in range(0, len(images), batch_size):
        batch_images = torch.stack(images[i:i+batch_size])
        batch_paths = valid_paths[i:i+batch_size]
        batched_images.append(batch_images)
        batched_paths.append(batch_paths)
    
    return batched_images, batched_paths

def predict_batch(model, images_batch, device):
    """批量预测"""
    with torch.no_grad():
        if device.type == 'cuda':
            # 使用CUDA流异步传输
            with torch.cuda.stream(torch.cuda.Stream()):
                images_batch = images_batch.to(device, non_blocking=True)
        else:
            images_batch = images_batch.to(device)
        
        outputs = model(images_batch)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return predictions

# 读取测试文件
print("读取测试文件...")
image_paths = []
with open(submission_file, "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if line:
            img_path, _ = line.split(",")
            image_paths.append(img_path)

# 批量处理图像
batch_size = 64  # 根据GPU内存调整
image_batches, path_batches = load_and_preprocess_images_batch(image_paths, env_transform, batch_size)

# 批量推理
print("进行批量推理...")
results = []

for batch_images, batch_paths in tqdm(zip(image_batches, path_batches), total=len(image_batches), desc="批量推理"):
    if len(batch_images) == 0:
        continue
        
    predictions = predict_batch(model, batch_images, device)
    
    for img_path, pred_idx in zip(batch_paths, predictions):
        try:
            # 确保pred_idx是字符串形式用于字典查找
            label = label_dict.get(str(int(pred_idx)), "unknown")
            results.append(f"{img_path},{label}")
        except Exception as e:
            print(f"Error processing prediction for {img_path}: {e}")
            results.append(f"{img_path},unknown")

# 写入结果
print("写入结果文件...")
output_path = "test_results.csv"
with open(output_path, "w") as f:
    f.write("image,label\n")
    for result in results:
        f.write(result + "\n")

print(f"处理完成！共处理 {len(results)} 张图片")