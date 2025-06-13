import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def calculate_metrics(y_true, y_pred, y_score=None, class_names=None):
    """
    计算各种评价指标
    
    该函数计算多种分类评价指标，包括基本指标（准确率）、多类别指标（精确率、召回率、F1分数）
    以及混淆矩阵和Cohen's Kappa系数。如果提供预测概率，还会计算ROC-AUC。
    
    参数:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        y_score: 预测概率数组 (可选，用于计算ROC-AUC)
        class_names: 类别名称列表 (可选，用于生成分类报告)
        
    返回:
        metrics_dict: 包含各种评价指标的字典
    """
    metrics = {}
    
    # 基本指标 - 准确率（正确预测的样本比例）
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 多类别指标 - 宏平均（每个类别的指标平均值，不考虑类别不平衡）
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    
    # 多类别指标 - 微平均（将所有类别的样本合并后计算指标）
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    
    # 多类别指标 - 加权平均（根据每个类别的样本数加权平均）
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # 计算混淆矩阵 - 行表示真实类别，列表示预测类别
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Cohen's Kappa - 考虑随机分类器的一致性度量
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # 如果提供了预测概率，计算ROC-AUC
    if y_score is not None:
        # 对于多类别问题，使用OvR (One-vs-Rest)和OvO (One-vs-One)策略
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_score, multi_class='ovr')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_score, multi_class='ovo')
        except ValueError:
            # 如果预测概率形状不正确，跳过ROC-AUC计算
            metrics['roc_auc_ovr'] = None
            metrics['roc_auc_ovo'] = None
    
    # 详细的分类报告 - 包含每个类别的精确率、召回率、F1分数和支持度
    if class_names is not None:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    else:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path, normalize=True, figsize=(12, 10)):
    """
    绘制混淆矩阵热图
    
    该函数将混淆矩阵可视化为热图，并保存为图像文件。
    可以选择是否对混淆矩阵进行归一化，以便更好地显示分类器的性能。
    
    参数:
        cm: 混淆矩阵数组，形状为[n_classes, n_classes]
        class_names: 类别名称列表，用于标注坐标轴
        save_path: 图像保存路径
        normalize: 是否对混淆矩阵进行归一化，默认为True
        figsize: 图像大小，默认为(12, 10)
    """
    if normalize:
        # 按行归一化，使每行之和为1，表示每个真实类别的预测分布
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # 显示小数点后两位
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'  # 显示整数
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    # 使用seaborn的热图函数绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')  # y轴表示真实标签
    plt.xlabel('Predicted Label')  # x轴表示预测标签
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics_to_file(metrics, save_path):
    """
    将评价指标保存到文本文件
    
    该函数将计算得到的各种评价指标以易读的格式写入文本文件，
    包括基本指标、多类别指标和详细的分类报告。
    
    参数:
        metrics: 包含各种评价指标的字典，由calculate_metrics函数生成
        save_path: 文本文件保存路径
    """
    with open(save_path, 'w') as f:
        # 基本指标 - 准确率
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        
        # 多类别指标 - 宏平均
        f.write("Macro-average metrics (平均每个类别的指标):\n")
        f.write(f"Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_macro']:.4f}\n\n")
        
        # 多类别指标 - 微平均
        f.write("Micro-average metrics (将所有类别的样本合并后计算指标):\n")
        f.write(f"Precision: {metrics['precision_micro']:.4f}\n")
        f.write(f"Recall: {metrics['recall_micro']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_micro']:.4f}\n\n")
        
        # 多类别指标 - 加权平均
        f.write("Weighted-average metrics (根据每个类别的样本数加权平均):\n")
        f.write(f"Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall: {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_weighted']:.4f}\n\n")
        
        # Cohen's Kappa
        f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")
        
        # ROC-AUC（如果有）
        if 'roc_auc_ovr' in metrics and metrics['roc_auc_ovr'] is not None:
            f.write(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}\n")
        if 'roc_auc_ovo' in metrics and metrics['roc_auc_ovo'] is not None:
            f.write(f"ROC-AUC (OvO): {metrics['roc_auc_ovo']:.4f}\n\n")
        
        # 详细的分类报告 - 包含每个类别的精确率、召回率、F1分数和支持度
        f.write("Classification Report:\n")
        report = metrics['classification_report']
        
        # 将分类报告字典转换为表格形式，便于阅读
        if isinstance(report, dict):
            headers = ['precision', 'recall', 'f1-score', 'support']
            f.write(f"{'':20s} {headers[0]:10s} {headers[1]:10s} {headers[2]:10s} {headers[3]:10s}\n")
            
            # 写入每个类别的指标
            for class_name, class_metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
                    continue
                
                f.write(f"{class_name:20s} ")
                for metric in headers:
                    if metric in class_metrics:
                        if metric == 'support':
                            f.write(f"{class_metrics[metric]:10d} ")  # 支持度为整数
                        else:
                            f.write(f"{class_metrics[metric]:.4f}    ")  # 其他指标保留4位小数
                f.write("\n")
            
            # 写入平均指标
            for avg_name in ['micro avg', 'macro avg', 'weighted avg']:
                if avg_name in report:
                    f.write(f"\n{avg_name:20s} ")
                    for metric in headers:
                        if metric in report[avg_name]:
                            if metric == 'support':
                                f.write(f"{report[avg_name][metric]:10d} ")
                            else:
                                f.write(f"{report[avg_name][metric]:.4f}    ")
                    f.write("\n")
            
            # 写入总体准确率
            if 'accuracy' in report:
                f.write(f"\nAccuracy: {report['accuracy']:.4f}\n")
        else:
            # 如果分类报告不是字典形式，直接写入
            f.write(report)

def plot_metrics_comparison(model1_metrics, model2_metrics, model1_name, model2_name, save_path):
    """
    绘制两个模型的评价指标比较图
    
    该函数将两个模型的多种评价指标以柱状图的形式进行可视化比较，
    便于直观地对比两个模型在各项指标上的性能差异。
    
    参数:
        model1_metrics: 模型1的评价指标字典
        model2_metrics: 模型2的评价指标字典
        model1_name: 模型1的名称，用于图例
        model2_name: 模型2的名称，用于图例
        save_path: 图像保存路径
    """
    # 选择要比较的指标
    metrics_to_compare = [
        'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted',
        'cohen_kappa'
    ]
    
    # 为了更好的可视化，将指标名称映射为更友好的显示名称
    metric_display_names = {
        'accuracy': 'Accuracy',
        'precision_macro': 'Precision (Macro)',
        'recall_macro': 'Recall (Macro)',
        'f1_macro': 'F1-Score (Macro)',
        'precision_weighted': 'Precision (Weighted)',
        'recall_weighted': 'Recall (Weighted)',
        'f1_weighted': 'F1-Score (Weighted)',
        'cohen_kappa': 'Cohen\'s Kappa'
    }
    
    # 提取要比较的指标值
    model1_values = [model1_metrics[metric] for metric in metrics_to_compare]
    model2_values = [model2_metrics[metric] for metric in metrics_to_compare]
    
    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'Metric': [metric_display_names[m] for m in metrics_to_compare],
        model1_name: model1_values,
        model2_name: model2_values
    })
    
    # 转换为长格式，便于使用seaborn绘图
    df_melted = pd.melt(df, id_vars=['Metric'], var_name='Model', value_name='Value')
    
    # 绘制对比柱状图
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")  # 设置图表风格
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_melted)
    
    # 在柱状图上添加数值标签，便于精确比较
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,  # x坐标（柱子中心）
            height + 0.01,  # y坐标（柱子顶部上方）
            f'{height:.4f}',  # 显示值（保留4位小数）
            ha='center', va='bottom',  # 水平和垂直对齐
            fontsize=8  # 字体大小
        )
    
    plt.title('Performance Metrics Comparison')  # 图表标题
    plt.ylim(0, 1.1)  # y轴范围
    plt.xticks(rotation=45)  # x轴标签旋转45度，避免重叠
    plt.tight_layout()  # 自动调整布局，避免标签被裁剪
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图像，释放内存

def plot_per_class_metrics(metrics, model_name, save_path):
    """
    绘制每个类别的评价指标图
    
    该函数将模型在每个类别上的精确率、召回率和F1分数以柱状图的形式可视化，
    便于分析模型在不同类别上的表现差异。
    
    参数:
        metrics: 包含分类报告的评价指标字典
        model_name: 模型名称，用于图表标题
        save_path: 图像保存路径
    """
    # 确保分类报告存在
    if 'classification_report' not in metrics:
        print("Classification report not found in metrics dictionary.")
        return
    
    report = metrics['classification_report']
    
    # 提取每个类别的指标
    classes = []
    precision = []
    recall = []
    f1_score = []
    
    for class_name, class_metrics in report.items():
        # 跳过平均值和准确率
        if class_name in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            continue
        
        classes.append(class_name)
        precision.append(class_metrics['precision'])
        recall.append(class_metrics['recall'])
        f1_score.append(class_metrics['f1-score'])
    
    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
    
    # 转换为长格式，便于使用seaborn绘图
    df_melted = pd.melt(df, id_vars=['Class'], var_name='Metric', value_name='Value')
    
    # 绘制柱状图
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Class', y='Value', hue='Metric', data=df_melted)
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.2f}',
            ha='center', va='bottom',
            fontsize=8, rotation=90
        )
    
    plt.title(f'Per-Class Metrics - {model_name}')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90)  # 类别名称垂直显示，避免重叠
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, dataloader, criterion, device, class_names=None, save_dir=None, model_name=None):
    """
    全面评估模型性能并生成评价报告
    
    该函数在给定数据集上评估模型性能，计算多种评价指标，并生成可视化结果。
    如果提供保存目录和模型名称，将保存评价报告和可视化图表。
    
    参数:
        model: PyTorch模型
        dataloader: 数据加载器，通常为测试集
        criterion: 损失函数
        device: 计算设备（CPU或GPU）
        class_names: 类别名称列表，用于标注图表和报告
        save_dir: 结果保存目录，如果为None则不保存
        model_name: 模型名称，用于文件命名
        
    返回:
        metrics: 包含各种评价指标的字典
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    total = 0
    
    # 不计算梯度，加速推理过程
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 计算预测概率和类别
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # 累加损失和样本数
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            
            # 收集预测结果、真实标签和预测概率
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算平均损失
    avg_loss = running_loss / total
    
    # 计算各种评价指标
    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names)
    metrics['loss'] = avg_loss
    
    # 如果提供了保存目录和模型名称，保存评价报告和可视化图表
    if save_dir is not None and model_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存混淆矩阵图
        cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
        
        # 保存每个类别的评价指标图
        per_class_path = os.path.join(save_dir, f"{model_name}_per_class_metrics.png")
        plot_per_class_metrics(metrics, model_name, per_class_path)
        
        # 保存评价指标到文本文件
        metrics_path = os.path.join(save_dir, f"{model_name}_metrics.txt")
        save_metrics_to_file(metrics, metrics_path)
    
    return metrics 