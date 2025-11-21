"""
下载 MNIST 数据集并转换为 PNG 图像
"""
import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# MNIST 数据集 URL - 使用 GitHub 镜像或原始源
MNIST_URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}

def download_file(url, filepath):
    """下载文件"""
    print(f"正在下载: {url}")
    urllib.request.urlretrieve(url, filepath)
    print(f"已保存到: {filepath}")

def read_mnist_images(filepath):
    """读取 MNIST 图像文件"""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_mnist_labels(filepath):
    """读取 MNIST 标签文件"""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_as_png(images, labels, output_dir, dataset_name):
    """将图像保存为 PNG 格式"""
    print(f"\n正在保存 {dataset_name} 数据集为 PNG...")
    
    # 为每个数字创建目录
    for digit in range(10):
        digit_dir = output_dir / str(digit)
        digit_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计每个数字的数量
    digit_counts = {i: 0 for i in range(10)}
    
    # 保存图像
    for idx, (image, label) in enumerate(zip(images, labels)):
        digit_dir = output_dir / str(label)
        filename = digit_dir / f"{digit_counts[label]}.png"
        
        # 转换为 PIL Image 并保存
        img = Image.fromarray(image, mode='L')
        img.save(filename)
        
        digit_counts[label] += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{len(images)} 张图像")
    
    print(f"✓ {dataset_name} 数据集保存完成!")
    print(f"  总计: {len(images)} 张图像")
    for digit, count in digit_counts.items():
        print(f"  数字 {digit}: {count} 张")

def main():
    # 创建目录
    data_dir = Path('mnist_data')
    data_dir.mkdir(exist_ok=True)
    
    png_dir = Path('mnist_png')
    train_dir = png_dir / 'train'
    test_dir = png_dir / 'test'
    
    # 下载数据集
    print("=" * 50)
    print("开始下载 MNIST 数据集")
    print("=" * 50)
    
    files = {}
    for name, url in MNIST_URLS.items():
        filepath = data_dir / url.split('/')[-1]
        if not filepath.exists():
            download_file(url, filepath)
        else:
            print(f"文件已存在: {filepath}")
        files[name] = filepath
    
    # 读取训练集
    print("\n" + "=" * 50)
    print("处理训练集")
    print("=" * 50)
    train_images = read_mnist_images(files['train_images'])
    train_labels = read_mnist_labels(files['train_labels'])
    save_as_png(train_images, train_labels, train_dir, '训练集')
    
    # 读取测试集
    print("\n" + "=" * 50)
    print("处理测试集")
    print("=" * 50)
    test_images = read_mnist_images(files['test_images'])
    test_labels = read_mnist_labels(files['test_labels'])
    save_as_png(test_images, test_labels, test_dir, '测试集')
    
    print("\n" + "=" * 50)
    print("全部完成!")
    print("=" * 50)
    print(f"PNG 图像保存在: {png_dir.absolute()}")
    print(f"  训练集: {train_dir.absolute()}")
    print(f"  测试集: {test_dir.absolute()}")

if __name__ == '__main__':
    main()
