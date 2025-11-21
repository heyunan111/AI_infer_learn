# 训练配置
class Config:
    # 数据集
    data_path = 'coco128'
    num_classes = 80
    
    # 训练参数
    batch_size = 8
    epochs = 50
    img_size = 640
    learning_rate = 0.001
    
    # 模型参数
    anchors = [
        [10,13, 16,30, 33,23],      # P3/8
        [30,61, 62,45, 59,119],     # P4/16
        [116,90, 156,198, 373,326]  # P5/32
    ]
    
    # 设备
    device = 'cuda'
    
    # 保存路径
    save_dir = 'runs/train'
