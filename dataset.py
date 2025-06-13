import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FishDataset(Dataset):
    """
    鱼类图像数据集加载器
    
    继承自PyTorch的Dataset类，用于加载和处理鱼类图像数据集。
    支持训练集、验证集和测试集的加载，并应用相应的数据变换。
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        初始化数据集
        
        参数:
            root_dir (string): 数据集根目录，包含train, val, test子目录
            mode (string): 数据集模式，可选'train', 'val', 或 'test'
            transform (callable, optional): 应用于图像的转换函数
        """
        self.root_dir = os.path.join(root_dir, mode)  # 根据模式选择对应的子目录
        self.mode = mode  # 保存模式信息
        self.transform = transform  # 保存转换函数
        
        # 获取所有类别（文件夹名称）并排序，确保类别顺序一致
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        # 创建类别名称到索引的映射字典
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 获取所有图像文件路径和对应的标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                # 只处理jpg和png格式的图像
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    # 添加(图像路径, 类别索引)元组到samples列表
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        """
        返回数据集中样本的数量
        
        返回:
            int: 数据集大小
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (图像张量, 类别标签)
        """
        img_path, label = self.samples[idx]  # 获取图像路径和标签
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB模式
        
        if self.transform:
            image = self.transform(image)  # 应用图像转换
        
        return image, label

def get_data_loaders(data_dir='datas', batch_size=32, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    
    为三个数据集分别创建数据加载器，应用不同的数据转换策略：
    - 训练集：应用数据增强技术，包括随机裁剪、翻转、旋转和颜色抖动
    - 验证集和测试集：只进行标准的调整大小和中心裁剪
    
    参数:
        data_dir (string): 数据集根目录，默认为'datas'
        batch_size (int): 批量大小，默认为32
        num_workers (int): 数据加载的工作线程数，默认为4
        
    返回:
        tuple: (train_loader, val_loader, test_loader) 三个数据加载器
    """
    # 定义数据转换
    # 训练集使用数据增强，提高模型泛化能力
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小为224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转±15度
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 颜色随机抖动
        transforms.ToTensor(),  # 转换为张量，并将像素值缩放到[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet均值和标准差进行标准化
    ])
    
    # 验证集和测试集只进行标准化处理，不使用数据增强
    val_transform = transforms.Compose([
        transforms.Resize(256),  # 调整大小为256x256
        transforms.CenterCrop(224),  # 中心裁剪为224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    # 创建数据集实例
    train_dataset = FishDataset(root_dir=data_dir, mode='train', transform=train_transform)
    val_dataset = FishDataset(root_dir=data_dir, mode='val', transform=val_transform)
    test_dataset = FishDataset(root_dir=data_dir, mode='test', transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # 打乱训练数据，增加随机性
        num_workers=num_workers,  # 多线程加载数据
        pin_memory=True  # 将数据加载到CUDA固定内存，加速GPU训练
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 不打乱验证数据，保持一致性
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 不打乱测试数据，保持一致性
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_class_names(data_dir='datas'):
    """
    获取数据集的类别名称
    
    从训练集目录中读取所有子文件夹名称，作为类别名称。
    
    参数:
        data_dir (string): 数据集根目录，默认为'datas'
        
    返回:
        list: 类别名称列表
    """
    # 获取训练集目录路径
    train_dir = os.path.join(data_dir, 'train')
    # 读取所有子文件夹名称并排序
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return classes 