import os
import zipfile
import urllib.request
from pathlib import Path

def download_coco128():
    """下载COCO128数据集"""
    
    # 数据集URL
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    
    # 下载路径
    zip_path = "coco128.zip"
    extract_path = "."
    
    print("开始下载COCO128数据集...")
    print(f"URL: {url}")
    
    # 下载文件
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100)
            print(f"\r下载进度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end="")
        
        urllib.request.urlretrieve(url, zip_path, show_progress)
        print("\n下载完成！")
        
        # 解压文件
        print("正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("解压完成！")
        
        # 删除zip文件
        os.remove(zip_path)
        print(f"已删除压缩包: {zip_path}")
        
        # 显示数据集结构
        coco128_path = Path("coco128")
        if coco128_path.exists():
            print("\nCOCO128数据集结构:")
            print(f"📁 coco128/")
            for item in sorted(coco128_path.iterdir()):
                if item.is_dir():
                    file_count = len(list(item.glob("*")))
                    print(f"  📁 {item.name}/ ({file_count} 文件)")
                else:
                    print(f"  📄 {item.name}")
            
            # 统计图片数量
            images_path = coco128_path / "images" / "train2017"
            labels_path = coco128_path / "labels" / "train2017"
            
            if images_path.exists():
                image_count = len(list(images_path.glob("*.jpg")))
                print(f"\n✓ 图片数量: {image_count}")
            
            if labels_path.exists():
                label_count = len(list(labels_path.glob("*.txt")))
                print(f"✓ 标签数量: {label_count}")
            
            print(f"\n数据集路径: {coco128_path.absolute()}")
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    download_coco128()
