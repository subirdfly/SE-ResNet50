import os
import sys
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QComboBox, QFrame, QSplitter, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from torchvision import transforms
from models import ResNet50, SEResNet50

class InferenceThread(QThread):
    """
    单独的线程用于执行模型推理，避免GUI卡顿
    """
    result_ready = pyqtSignal(list)
    
    def __init__(self, model, image_path, transform, class_names, device):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.transform = transform
        self.class_names = class_names
        self.device = device
        
    def run(self):
        try:
            # 加载图像
            image = Image.open(self.image_path).convert('RGB')
            # 应用变换
            image_tensor = self.transform(image).unsqueeze(0)
            
            # 确保图像张量在正确的设备上
            image_tensor = image_tensor.to(self.device)
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            # 获取前5个预测结果
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            results = []
            for i in range(5):
                idx = top5_indices[i].item()
                prob = top5_prob[i].item() * 100
                results.append((self.class_names[idx], prob))
                
            self.result_ready.emit(results)
        except Exception as e:
            print(f"推理过程中出错: {e}")
            # 发送空结果，表示出错
            self.result_ready.emit([("错误", 0.0)])


class FishRecognitionApp(QMainWindow):
    """
    鱼类识别应用程序的主窗口
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鱼类识别系统")
        self.setMinimumSize(800, 600)
        
        # 加载模型和类别名称
        self.load_resources()
        
        # 设置界面
        self.setup_ui()
        
    def load_resources(self):
        """加载模型和类别名称"""
        # 检查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载类别名称
        self.load_class_names()
        
        # 设置图像转换
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 默认使用ResNet50模型
        self.current_model_name = "ResNet50"
        self.load_model(self.current_model_name)
    
    def load_class_names(self):
        """加载鱼类名称"""
        try:
            # 首先从训练目录加载类别名称，确保与训练时相同的顺序
            train_dir = os.path.join("datas", "train")
            if os.path.exists(train_dir):
                # 这里使用与训练时相同的方法获取类别名称
                self.class_names = sorted([d for d in os.listdir(train_dir) 
                                        if os.path.isdir(os.path.join(train_dir, d))])
                print(f"从训练目录加载了 {len(self.class_names)} 个类别")
                
                # 初始化中文名称字典
                self.class_names_ch = {name: name for name in self.class_names}
                
                # 尝试从en2ch.txt加载中英文对应关系
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
                for encoding in encodings:
                    try:
                        with open("datas/en2ch.txt", "r", encoding=encoding) as f:
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        en_name = parts[0]
                                        ch_name = parts[1]
                                        # 只更新已有的类别的中文名
                                        if en_name in self.class_names_ch:
                                            self.class_names_ch[en_name] = ch_name
                        print(f"成功使用 {encoding} 编码加载中文名称")
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                # 如果训练目录不存在，则尝试从en2ch.txt加载类别名称
                self.class_names = []
                self.class_names_ch = {}
                
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
                for encoding in encodings:
                    try:
                        with open("datas/en2ch.txt", "r", encoding=encoding) as f:
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        en_name = parts[0]
                                        ch_name = parts[1]
                                        self.class_names.append(en_name)
                                        self.class_names_ch[en_name] = ch_name
                        print(f"成功使用 {encoding} 编码加载类别名称")
                        break
                    except UnicodeDecodeError:
                        continue
            
            # 打印类别名称和索引，帮助调试
            print("类别名称和索引:")
            for i, name in enumerate(self.class_names):
                ch_name = self.class_names_ch.get(name, name)
                print(f"{i}: {name} ({ch_name})")
                
        except Exception as e:
            print(f"加载类别名称时出错: {e}")
            # 如果出错，使用默认类别名称
            self.class_names = [f"class_{i}" for i in range(30)]
            self.class_names_ch = {name: name for name in self.class_names}
    
    def load_model(self, model_name):
        """加载指定的模型"""
        try:
            num_classes = len(self.class_names)
            
            # 根据选择加载不同的模型
            if model_name == "ResNet50":
                self.model = ResNet50(num_classes=num_classes)
                model_path = "ResNet50_best.pth"
            elif model_name == "SE-ResNet50":
                self.model = SEResNet50(num_classes=num_classes)
                model_path = "SE-ResNet50_best.pth"
            elif model_name == "SE-ResNet50-NoSE1":
                self.model = SEResNet50(num_classes=num_classes, use_se_in_layer1=False)
                model_path = "SE-ResNet50-layer_best.pth"
            
            # 尝试加载模型权重
            if os.path.exists(model_path):
                # 使用map_location确保模型加载到正确的设备上
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"成功加载模型: {model_path}")
            else:
                print(f"警告: 模型文件 {model_path} 不存在，使用未训练的模型")
            
            # 将模型移至设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主窗口部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 创建顶部控制栏
        control_layout = QHBoxLayout()
        
        # 模型选择下拉框
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet50", "SE-ResNet50"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo)
        
        # 添加加载图片按钮
        self.load_button = QPushButton("加载图片")
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)
        
        # 添加识别按钮
        self.recognize_button = QPushButton("识别")
        self.recognize_button.clicked.connect(self.recognize_image)
        self.recognize_button.setEnabled(False)  # 初始状态禁用
        control_layout.addWidget(self.recognize_button)
        
        main_layout.addLayout(control_layout)
        
        # 创建分割器，用于图像显示和结果显示
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧图像显示区域
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout = QVBoxLayout(self.image_frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("请加载图片")
        image_layout.addWidget(self.image_label)
        
        # 右侧结果显示区域
        self.result_frame = QFrame()
        self.result_frame.setFrameShape(QFrame.StyledPanel)
        result_layout = QVBoxLayout(self.result_frame)
        
        result_layout.addWidget(QLabel("<h3>识别结果</h3>"))
        
        # 创建5个结果标签
        self.result_labels = []
        for i in range(5):
            label = QLabel(f"{i+1}. 等待识别...")
            result_layout.addWidget(label)
            self.result_labels.append(label)
        
        # 添加结果说明
        result_layout.addWidget(QLabel("<p>注: 显示前5个最可能的鱼类及其概率</p>"))
        
        # 添加弹性空间
        result_layout.addStretch()
        
        # 添加到分割器
        splitter.addWidget(self.image_frame)
        splitter.addWidget(self.result_frame)
        
        # 设置初始大小比例
        splitter.setSizes([500, 300])
        
        main_layout.addWidget(splitter)
        
        # 设置状态栏
        self.statusBar().showMessage(f"就绪 | 当前模型: {self.current_model_name} | 设备: {self.device}")
        
        # 设置中心窗口部件
        self.setCentralWidget(main_widget)
    
    def change_model(self, model_name):
        """切换模型"""
        self.current_model_name = model_name
        self.load_model(model_name)
        self.statusBar().showMessage(f"已切换到模型: {model_name} | 设备: {self.device}")
    
    def load_image(self):
        """加载图像文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.image_path = file_path
            
            # 显示图像
            pixmap = QPixmap(file_path)
            
            # 缩放图像以适应标签大小，保持纵横比
            pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(pixmap)
            self.recognize_button.setEnabled(True)
            self.statusBar().showMessage(f"已加载图片: {os.path.basename(file_path)}")
            
            # 清除之前的结果
            for label in self.result_labels:
                label.setText("等待识别...")
    
    def recognize_image(self):
        """识别图像中的鱼类"""
        if not hasattr(self, 'image_path'):
            self.statusBar().showMessage("请先加载图片")
            return
        
        # 禁用识别按钮，避免重复点击
        self.recognize_button.setEnabled(False)
        self.statusBar().showMessage("正在识别...")
        
        # 创建并启动推理线程
        self.inference_thread = InferenceThread(
            self.model, self.image_path, self.transform, self.class_names, self.device
        )
        self.inference_thread.result_ready.connect(self.update_results)
        self.inference_thread.start()
    
    def update_results(self, results):
        """更新识别结果"""
        if results[0][0] == "错误":
            # 显示错误信息
            self.statusBar().showMessage("识别过程中出错，请重试")
            for label in self.result_labels:
                label.setText("识别失败")
        else:
            # 更新结果标签
            for i, (class_name, probability) in enumerate(results):
                # 获取中文名称（如果有）
                chinese_name = self.class_names_ch.get(class_name, class_name)
                
                # 更新结果标签
                self.result_labels[i].setText(
                    f"{i+1}. {class_name} ({chinese_name}): {probability:.2f}%"
                )
            
            # 更新状态栏
            self.statusBar().showMessage(f"识别完成 | 最可能的鱼类: {results[0][0]} ({results[0][1]:.2f}%)")
        
        # 重新启用识别按钮
        self.recognize_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FishRecognitionApp()
    window.show()
    sys.exit(app.exec_()) 