import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import sys
from datetime import datetime
from train import train_model, set_seed
from metrics import plot_metrics_comparison

def compare_models(data_dir, batch_size, num_epochs, lr, weight_decay, save_dir, seed):
    """
    比较ResNet-50和SE-ResNet-50模型的性能
    
    该函数依次训练ResNet-50和SE-ResNet-50两个模型，并对它们的性能进行全面比较。
    比较结果包括准确率、F1分数、精确率、召回率和Cohen's Kappa等多项指标，
    并生成可视化图表和详细的比较报告。
    
    参数:
        data_dir: 数据集目录路径，包含train、val和test子目录
        batch_size: 训练批量大小
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减系数
        save_dir: 结果保存目录
        seed: 随机种子，用于确保实验可重复性
        
    返回:
        dict: 包含两个模型指标和它们之间差异的字典
    """
    
    # 设置随机种子，确保实验可重复性
    set_seed(seed)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建比较日志目录
    comparison_log_dir = os.path.join(save_dir, 'comparison_logs')
    os.makedirs(comparison_log_dir, exist_ok=True)
    
    # 创建比较日志文件，使用时间戳命名以避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_log_file = os.path.join(comparison_log_dir, f"comparison_{timestamp}_log.txt")
    
    # 打开日志文件，保持打开状态直到比较结束
    log_file = open(comparison_log_file, 'w')
    
    try:
        # 写入比较日志头部信息
        log_file.write("ResNet-50 vs SE-ResNet-50 Comparison Log\n")
        log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 50 + "\n\n")
        log_file.write("Hyperparameters:\n")
        log_file.write(f"  Batch Size: {batch_size}\n")
        log_file.write(f"  Epochs: {num_epochs}\n")
        log_file.write(f"  Learning Rate: {lr}\n")
        log_file.write(f"  Weight Decay: {weight_decay}\n")
        log_file.write(f"  Random Seed: {seed}\n")
        log_file.write(f"  Dataset Path: {data_dir}\n\n")
        log_file.write("=" * 50 + "\n\n")
        log_file.flush()  # 立即写入文件，避免缓存
        
        # 记录比较开始时间
        comparison_start_time = time.time()
        
        # 训练ResNet-50模型
        print("\n" + "="*50)
        print("Training ResNet-50 model...")
        print("="*50)
        
        log_file.write("Training ResNet-50 model...\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.flush()  # 立即写入文件
        
        # 调用train_model函数训练ResNet-50模型
        resnet_metrics = train_model(
            model_type='resnet50',  # 模型类型为标准ResNet-50
            data_dir=data_dir,      # 数据集目录
            batch_size=batch_size,  # 批量大小
            num_epochs=num_epochs,  # 训练轮数
            lr=lr,                  # 学习率
            weight_decay=weight_decay,  # 权重衰减
            save_dir=save_dir       # 结果保存目录
        )
        
        # 记录ResNet-50训练完成信息
        log_file.write(f"ResNet-50 training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Test Accuracy: {resnet_metrics['accuracy']:.2f}%\n\n")
        log_file.flush()  # 立即写入文件
        
        # 训练SE-ResNet-50模型
        print("\n" + "="*50)
        print("Training SE-ResNet-50 model...")
        print("="*50)
        
        log_file.write("\nTraining SE-ResNet-50 model...\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.flush()  # 立即写入文件
        
        # 调用train_model函数训练SE-ResNet-50模型
        se_resnet_metrics = train_model(
            model_type='se_resnet50',  # 模型类型为SE-ResNet-50
            data_dir=data_dir,         # 数据集目录
            batch_size=batch_size,     # 批量大小
            num_epochs=num_epochs,     # 训练轮数
            lr=lr,                     # 学习率
            weight_decay=weight_decay, # 权重衰减
            save_dir=save_dir          # 结果保存目录
        )
        
        # 记录SE-ResNet-50训练完成信息
        log_file.write(f"SE-ResNet-50 training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Test Accuracy: {se_resnet_metrics['accuracy']:.2f}%\n\n")
        log_file.flush()  # 立即写入文件
        
        # 计算比较总时间
        comparison_total_time = time.time() - comparison_start_time
        
        # 打印比较结果
        print("\n" + "="*50)
        print("Comparison Results:")
        print("="*50)
        print(f"ResNet-50 Test Accuracy: {resnet_metrics['accuracy']:.2f}%")
        print(f"SE-ResNet-50 Test Accuracy: {se_resnet_metrics['accuracy']:.2f}%")
        print(f"Accuracy Improvement: {se_resnet_metrics['accuracy'] - resnet_metrics['accuracy']:.2f}%")
        
        print(f"\nResNet-50 F1 Score (Macro): {resnet_metrics['f1_macro']:.4f}")
        print(f"SE-ResNet-50 F1 Score (Macro): {se_resnet_metrics['f1_macro']:.4f}")
        print(f"F1 Score Improvement: {se_resnet_metrics['f1_macro'] - resnet_metrics['f1_macro']:.4f}")
        
        print(f"\nResNet-50 Cohen's Kappa: {resnet_metrics['cohen_kappa']:.4f}")
        print(f"SE-ResNet-50 Cohen's Kappa: {se_resnet_metrics['cohen_kappa']:.4f}")
        print(f"Cohen's Kappa Improvement: {se_resnet_metrics['cohen_kappa'] - resnet_metrics['cohen_kappa']:.4f}")
        
        # 写入比较结果到日志
        log_file.write("\n" + "="*50 + "\n")
        log_file.write("Comparison Results:\n")
        log_file.write("="*50 + "\n\n")
        
        # 准确率比较
        log_file.write(f"ResNet-50 Test Accuracy: {resnet_metrics['accuracy']:.2f}%\n")
        log_file.write(f"SE-ResNet-50 Test Accuracy: {se_resnet_metrics['accuracy']:.2f}%\n")
        log_file.write(f"Accuracy Improvement: {se_resnet_metrics['accuracy'] - resnet_metrics['accuracy']:.2f}%\n\n")
        
        # F1分数比较 (宏平均)
        log_file.write(f"ResNet-50 F1 Score (Macro): {resnet_metrics['f1_macro']:.4f}\n")
        log_file.write(f"SE-ResNet-50 F1 Score (Macro): {se_resnet_metrics['f1_macro']:.4f}\n")
        log_file.write(f"F1 Score Improvement: {se_resnet_metrics['f1_macro'] - resnet_metrics['f1_macro']:.4f}\n\n")
        
        # 精确率比较 (宏平均)
        log_file.write(f"ResNet-50 Precision (Macro): {resnet_metrics['precision_macro']:.4f}\n")
        log_file.write(f"SE-ResNet-50 Precision (Macro): {se_resnet_metrics['precision_macro']:.4f}\n")
        log_file.write(f"Precision Improvement: {se_resnet_metrics['precision_macro'] - resnet_metrics['precision_macro']:.4f}\n\n")
        
        # 召回率比较 (宏平均)
        log_file.write(f"ResNet-50 Recall (Macro): {resnet_metrics['recall_macro']:.4f}\n")
        log_file.write(f"SE-ResNet-50 Recall (Macro): {se_resnet_metrics['recall_macro']:.4f}\n")
        log_file.write(f"Recall Improvement: {se_resnet_metrics['recall_macro'] - resnet_metrics['recall_macro']:.4f}\n\n")
        
        # F1分数比较 (加权平均)
        log_file.write(f"ResNet-50 F1 Score (Weighted): {resnet_metrics['f1_weighted']:.4f}\n")
        log_file.write(f"SE-ResNet-50 F1 Score (Weighted): {se_resnet_metrics['f1_weighted']:.4f}\n")
        log_file.write(f"F1 Score (Weighted) Improvement: {se_resnet_metrics['f1_weighted'] - resnet_metrics['f1_weighted']:.4f}\n\n")
        
        # Cohen's Kappa比较
        log_file.write(f"ResNet-50 Cohen's Kappa: {resnet_metrics['cohen_kappa']:.4f}\n")
        log_file.write(f"SE-ResNet-50 Cohen's Kappa: {se_resnet_metrics['cohen_kappa']:.4f}\n")
        log_file.write(f"Cohen's Kappa Improvement: {se_resnet_metrics['cohen_kappa'] - resnet_metrics['cohen_kappa']:.4f}\n\n")
        
        # 记录总比较时间和完成时间
        log_file.write(f"Total comparison time: {comparison_total_time/60:.2f} minutes\n")
        log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 50 + "\n")
        log_file.flush()  # 立即写入文件
        
        # 创建比较图表 - 准确率对比柱状图
        log_file.write("\nGenerating comparison charts...\n")
        log_file.flush()  # 立即写入文件
        
        # 准备数据
        models = ['ResNet-50', 'SE-ResNet-50']
        accuracies = [resnet_metrics['accuracy'], se_resnet_metrics['accuracy']]
        
        # 创建准确率对比柱状图
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")  # 设置图表风格
        ax = sns.barplot(x=models, y=accuracies)
        
        # 在柱状图上添加数值标签
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.5, f"{acc:.2f}%", ha='center')
        
        # 设置图表标题和标签
        plt.title('Test Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(min(accuracies) - 5, max(accuracies) + 5)  # 设置y轴范围
        
        # 添加改进标注
        improvement = se_resnet_metrics['accuracy'] - resnet_metrics['accuracy']
        plt.annotate(
            f"Improvement: {improvement:.2f}%",
            xy=(0.5, max(accuracies) + 2),
            xytext=(0.5, max(accuracies) + 3),
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )
        
        # 保存准确率对比图
        plt.tight_layout()
        accuracy_chart_path = os.path.join(save_dir, 'accuracy_comparison.png')
        plt.savefig(accuracy_chart_path)
        plt.close()
        
        # 记录图表保存路径
        log_file.write(f"Accuracy comparison chart saved to: {accuracy_chart_path}\n")
        log_file.flush()  # 立即写入文件
        
        # 打印调试信息，查看指标值
        print("\n调试信息 - 指标值:")
        print(f"ResNet-50 Precision (Weighted): {resnet_metrics['precision_weighted']:.4f}")
        print(f"ResNet-50 Recall (Weighted): {resnet_metrics['recall_weighted']:.4f}")
        print(f"SE-ResNet-50 Precision (Weighted): {se_resnet_metrics['precision_weighted']:.4f}")
        print(f"SE-ResNet-50 Recall (Weighted): {se_resnet_metrics['recall_weighted']:.4f}")
        
        # 将指标转换为适合plot_metrics_comparison函数的格式
        # 注意：准确率需要从百分比转换为0-1范围
        resnet_metrics_converted = {
            'accuracy': resnet_metrics['accuracy'] / 100,  # 转换为0-1范围
            'precision_macro': resnet_metrics['precision_macro'],
            'recall_macro': resnet_metrics['recall_macro'],
            'f1_macro': resnet_metrics['f1_macro'],
            'precision_weighted': resnet_metrics['precision_weighted'],  # 直接使用字典中的值
            'recall_weighted': resnet_metrics['recall_weighted'],  # 直接使用字典中的值
            'f1_weighted': resnet_metrics['f1_weighted'],
            'cohen_kappa': resnet_metrics['cohen_kappa']
        }
        
        se_resnet_metrics_converted = {
            'accuracy': se_resnet_metrics['accuracy'] / 100,  # 转换为0-1范围
            'precision_macro': se_resnet_metrics['precision_macro'],
            'recall_macro': se_resnet_metrics['recall_macro'],
            'f1_macro': se_resnet_metrics['f1_macro'],
            'precision_weighted': se_resnet_metrics['precision_weighted'],  # 直接使用字典中的值
            'recall_weighted': se_resnet_metrics['recall_weighted'],  # 直接使用字典中的值
            'f1_weighted': se_resnet_metrics['f1_weighted'],
            'cohen_kappa': se_resnet_metrics['cohen_kappa']
        }
        
        # 绘制全面的指标比较图
        metrics_chart_path = os.path.join(save_dir, 'metrics_comparison.png')
        plot_metrics_comparison(
            resnet_metrics_converted,
            se_resnet_metrics_converted,
            'ResNet-50',
            'SE-ResNet-50',
            metrics_chart_path
        )
        
        # 记录指标比较图保存路径
        log_file.write(f"Metrics comparison chart saved to: {metrics_chart_path}\n")
        log_file.flush()  # 立即写入文件
        
        # 保存比较结果到CSV文件
        log_file.write("\nSaving comparison results to CSV and JSON...\n")
        log_file.flush()  # 立即写入文件
        
        # 创建包含所有指标的DataFrame
        metrics_names = [
            'Accuracy (%)', 'F1 Score (Macro)', 'F1 Score (Weighted)',
            'Precision (Macro)', 'Recall (Macro)', 'Cohen\'s Kappa'
        ]
        
        # 提取两个模型的指标值
        resnet_values = [
            resnet_metrics['accuracy'],
            resnet_metrics['f1_macro'],
            resnet_metrics['f1_weighted'],
            resnet_metrics['precision_macro'],
            resnet_metrics['recall_macro'],
            resnet_metrics['cohen_kappa']
        ]
        
        se_resnet_values = [
            se_resnet_metrics['accuracy'],
            se_resnet_metrics['f1_macro'],
            se_resnet_metrics['f1_weighted'],
            se_resnet_metrics['precision_macro'],
            se_resnet_metrics['recall_macro'],
            se_resnet_metrics['cohen_kappa']
        ]
        
        # 计算各指标的改进值
        improvements = [se - res for se, res in zip(se_resnet_values, resnet_values)]
        
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame({
            'Metric': metrics_names,
            'ResNet-50': resnet_values,
            'SE-ResNet-50': se_resnet_values,
            'Improvement': improvements
        })
        
        # 保存比较结果CSV
        csv_path = os.path.join(save_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # 记录CSV保存路径
        log_file.write(f"Comparison CSV saved to: {csv_path}\n")
        log_file.flush()  # 立即写入文件
        
        # 保存详细的比较结果为JSON格式
        comparison_results = {
            'resnet50': {k: float(v) if isinstance(v, (int, float)) else v 
                        for k, v in resnet_metrics.items()},  # ResNet-50的所有指标
            'se_resnet50': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in se_resnet_metrics.items()},  # SE-ResNet-50的所有指标
            'improvements': {  # 各项指标的改进值
                'accuracy': float(se_resnet_metrics['accuracy'] - resnet_metrics['accuracy']),
                'f1_macro': float(se_resnet_metrics['f1_macro'] - resnet_metrics['f1_macro']),
                'f1_weighted': float(se_resnet_metrics['f1_weighted'] - resnet_metrics['f1_weighted']),
                'precision_macro': float(se_resnet_metrics['precision_macro'] - resnet_metrics['precision_macro']),
                'recall_macro': float(se_resnet_metrics['recall_macro'] - resnet_metrics['recall_macro']),
                'cohen_kappa': float(se_resnet_metrics['cohen_kappa'] - resnet_metrics['cohen_kappa'])
            },
            'hyperparameters': {  # 实验超参数
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'random_seed': seed
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 时间戳
            'total_time_minutes': float(comparison_total_time / 60)  # 总比较时间（分钟）
        }
        
        # 保存JSON结果
        json_path = os.path.join(comparison_log_dir, f"comparison_{timestamp}_results.json")
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=4)  # 使用缩进格式化JSON
        
        # 记录JSON保存路径和完成信息
        log_file.write(f"Comparison JSON saved to: {json_path}\n")
        log_file.write("\nAll comparison tasks completed successfully!\n")
        log_file.flush()  # 立即写入文件
        
        # 打印保存路径信息
        print(f"\nComparison results saved to {save_dir}")
        print(f"Comparison logs saved to {comparison_log_dir}")
        
        # 返回比较结果字典
        return {
            'resnet_metrics': resnet_metrics,  # ResNet-50的指标
            'se_resnet_metrics': se_resnet_metrics,  # SE-ResNet-50的指标
            'improvements': {  # 主要指标的改进值
                'accuracy': se_resnet_metrics['accuracy'] - resnet_metrics['accuracy'],
                'f1_macro': se_resnet_metrics['f1_macro'] - resnet_metrics['f1_macro'],
                'f1_weighted': se_resnet_metrics['f1_weighted'] - resnet_metrics['f1_weighted'],
                'cohen_kappa': se_resnet_metrics['cohen_kappa'] - resnet_metrics['cohen_kappa']
            }
        }
    
    except Exception as e:
        # 记录错误信息
        error_msg = f"\nERROR: {str(e)}\n"
        print(error_msg)
        log_file.write(error_msg)
        log_file.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.flush()
        raise e  # 重新抛出异常，便于调试
    
    finally:
        # 确保日志文件被关闭，避免资源泄漏
        log_file.close()

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Compare ResNet-50 and SE-ResNet-50 on fish dataset")
    
    # 添加命令行参数
    parser.add_argument('--data_dir', type=str, default='datas',
                        help='Data directory containing train, val, and test folders')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='comparison_results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用比较模型函数
    results = compare_models(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        seed=args.seed
    )