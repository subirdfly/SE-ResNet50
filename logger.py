import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingLogger:
    """
    训练日志记录器
    
    该类用于记录和保存深度学习模型训练过程中的各种指标和信息，包括：
    1. 超参数配置
    2. 每个epoch的训练和验证指标
    3. 最佳模型信息
    4. 测试集评估结果
    5. 学习曲线图表
    
    通过这些记录，可以全面了解模型训练过程，便于后续分析和改进。
    """
    def __init__(self, log_dir, model_name):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志保存目录，将在此目录下创建各种日志文件
            model_name: 模型名称，用于命名日志文件
        """
        self.log_dir = log_dir  # 日志保存目录
        self.model_name = model_name  # 模型名称
        
        # 创建日志目录（如果不存在）
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成带时间戳的日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳，格式：年月日_时分秒
        self.log_file = os.path.join(log_dir, f"{model_name}_{timestamp}_log.txt")  # 文本日志文件
        self.csv_file = os.path.join(log_dir, f"{model_name}_{timestamp}_metrics.csv")  # CSV格式指标记录
        
        # 初始化指标记录字典，用于存储训练过程中的各项指标
        self.metrics = {
            'epoch': [],  # 轮次
            'train_loss': [],  # 训练损失
            'train_acc': [],  # 训练准确率
            'val_loss': [],  # 验证损失
            'val_acc': [],  # 验证准确率
            'learning_rate': [],  # 学习率
            'time_elapsed': []  # 已用时间（秒）
        }
        
        # 记录训练开始时间
        self.start_time = time.time()
        
        # 写入日志头部信息
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log for {model_name}\n")  # 日志标题
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 开始时间
            f.write("=" * 50 + "\n\n")  # 分隔线
    
    def log_hyperparameters(self, hyperparams):
        """
        记录超参数配置
        
        将模型训练的超参数配置写入日志文件，并同时保存为JSON格式，
        便于后续查看和分析模型训练的配置信息。
        
        参数:
            hyperparams: 包含超参数的字典，如学习率、批量大小、优化器等
        """
        # 写入文本日志
        with open(self.log_file, 'a') as f:
            f.write("Hyperparameters:\n")
            # 遍历并记录每个超参数
            for name, value in hyperparams.items():
                f.write(f"  {name}: {value}\n")
            f.write("\n" + "=" * 50 + "\n\n")  # 分隔线
        
        # 同时保存为JSON格式，便于程序读取
        hyperparam_file = os.path.join(self.log_dir, f"{self.model_name}_hyperparams.json")
        with open(hyperparam_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)  # 使用缩进格式化JSON，提高可读性
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate):
        """
        记录每个epoch的训练信息
        
        记录每个训练轮次的关键指标，包括损失、准确率和学习率等，
        并同时更新CSV文件，便于后续绘制学习曲线和分析训练过程。
        
        参数:
            epoch: 当前训练轮次
            train_loss: 训练集上的损失值
            train_acc: 训练集上的准确率（百分比）
            val_loss: 验证集上的损失值
            val_acc: 验证集上的准确率（百分比）
            learning_rate: 当前学习率
        """
        # 计算已经过去的时间（从训练开始到现在）
        time_elapsed = time.time() - self.start_time
        
        # 添加当前轮次的指标到记录字典
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['time_elapsed'].append(time_elapsed)
        
        # 写入文本日志
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")  # 训练指标
            f.write(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")  # 验证指标
            f.write(f"  Learning Rate: {learning_rate:.6f}\n")  # 学习率
            f.write(f"  Time: {time_elapsed/60:.2f} minutes\n\n")  # 已用时间（分钟）
        
        # 更新CSV文件，便于后续分析和绘图
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_file, index=False)
    
    def log_best_model(self, epoch, val_acc):
        """
        记录最佳模型信息
        
        当模型在验证集上达到更好的性能时，记录相关信息，
        便于后续了解模型的最佳状态。
        
        参数:
            epoch: 最佳模型对应的训练轮次
            val_acc: 最佳模型在验证集上的准确率（百分比）
        """
        with open(self.log_file, 'a') as f:
            f.write(f"Best model saved at epoch {epoch} with validation accuracy: {val_acc:.2f}%\n\n")
    
    def log_test_metrics(self, metrics):
        """
        记录测试集上的评估指标
        
        在训练完成后，记录模型在测试集上的各项评估指标，
        全面评估模型的泛化能力和实际性能。
        
        参数:
            metrics: 包含各种评估指标的字典，如准确率、F1分数等
        """
        # 写入文本日志
        with open(self.log_file, 'a') as f:
            f.write("Test Metrics:\n")
            f.write(f"  Loss: {metrics['loss']:.4f}\n")  # 测试损失
            f.write(f"  Accuracy: {metrics['accuracy']*100:.2f}%\n")  # 测试准确率
            f.write(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}\n")  # 宏平均F1分数
            f.write(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")  # 加权平均F1分数
            f.write(f"  Precision (Macro): {metrics['precision_macro']:.4f}\n")  # 宏平均精确率
            f.write(f"  Recall (Macro): {metrics['recall_macro']:.4f}\n")  # 宏平均召回率
            f.write(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")  # Cohen's Kappa系数
        
        # 保存测试指标为JSON格式，便于程序读取和分析
        test_metrics_file = os.path.join(self.log_dir, f"{self.model_name}_test_metrics.json")
        with open(test_metrics_file, 'w') as f:
            # 将numpy类型转换为Python原生类型，以便JSON序列化
            metrics_to_save = {}
            for k, v in metrics.items():
                if k == 'confusion_matrix' or k == 'classification_report':
                    continue  # 跳过复杂对象，避免序列化问题
                if hasattr(v, 'tolist'):
                    metrics_to_save[k] = v.tolist()  # 转换numpy数组为列表
                else:
                    metrics_to_save[k] = v
            json.dump(metrics_to_save, f, indent=4)  # 使用缩进格式化JSON
    
    def log_training_complete(self, total_time):
        """
        记录训练完成信息
        
        在训练结束时记录总结信息，包括完成时间和总训练时长，
        作为训练日志的结束部分。
        
        参数:
            total_time: 总训练时间（秒）
        """
        with open(self.log_file, 'a') as f:
            f.write("=" * 50 + "\n")  # 分隔线
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 完成时间
            f.write(f"Total training time: {total_time/60:.2f} minutes\n")  # 总训练时长（分钟）
            f.write("=" * 50 + "\n")  # 分隔线
    
    def save_learning_curves(self):
        """
        保存学习曲线图
        
        根据记录的训练和验证指标，生成学习曲线图和学习率变化曲线图，
        直观展示模型训练过程中性能的变化趋势。
        """
        # 创建图形对象，设置大小
        plt.figure(figsize=(12, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)  # 2行1列的第1个子图
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')  # 训练损失曲线
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Validation Loss')  # 验证损失曲线
        plt.xlabel('Epoch')  # x轴标签
        plt.ylabel('Loss')  # y轴标签
        plt.title('Training and Validation Loss')  # 标题
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
        
        # 绘制准确率曲线
        plt.subplot(2, 1, 2)  # 2行1列的第2个子图
        plt.plot(self.metrics['epoch'], self.metrics['train_acc'], label='Train Accuracy')  # 训练准确率曲线
        plt.plot(self.metrics['epoch'], self.metrics['val_acc'], label='Validation Accuracy')  # 验证准确率曲线
        plt.xlabel('Epoch')  # x轴标签
        plt.ylabel('Accuracy (%)')  # y轴标签
        plt.title('Training and Validation Accuracy')  # 标题
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
        
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        # 保存学习曲线图
        plt.savefig(os.path.join(self.log_dir, f"{self.model_name}_learning_curves.png"))
        plt.close()  # 关闭图形，释放内存
        
        # 绘制学习率变化曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['epoch'], self.metrics['learning_rate'])  # 学习率曲线
        plt.xlabel('Epoch')  # x轴标签
        plt.ylabel('Learning Rate')  # y轴标签
        plt.title('Learning Rate Schedule')  # 标题
        plt.grid(True)  # 显示网格
        # 保存学习率曲线图
        plt.savefig(os.path.join(self.log_dir, f"{self.model_name}_lr_schedule.png"))
        plt.close()  # 关闭图形，释放内存 