import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import ResNet50, SEResNet50
from dataset import get_data_loaders, get_class_names
from metrics import evaluate_model, plot_confusion_matrix
from logger import TrainingLogger


def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复性

    通过设置PyTorch、NumPy和CUDA的随机种子，确保每次运行得到相同的结果，
    便于实验的对比和调试。

    参数:
        seed: 随机种子值，默认为42
    """
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.backends.cudnn.deterministic = True  # 确保cuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试，以确保结果可重复


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练模型一个完整的epoch

    该函数在给定的数据加载器上训练模型一个完整的epoch，
    更新模型参数，并返回平均损失和准确率。

    参数:
        model: PyTorch模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备（CPU或GPU）

    返回:
        tuple: (epoch_loss, epoch_acc) 本epoch的平均损失和准确率(%)
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累计损失
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数

    # 使用tqdm创建进度条，增强用户体验
    pbar = tqdm(dataloader, desc="训练中")
    for inputs, labels in pbar:
        # 将数据移至指定设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度，避免梯度累积
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)  # 累加批次损失
        _, predicted = outputs.max(1)  # 获取预测类别
        total += labels.size(0)  # 累加样本数
        correct += predicted.eq(labels).sum().item()  # 累加正确预测数

        # 更新进度条显示的信息
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })

    # 计算整个epoch的平均损失和准确率
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    在验证集上评估模型

    该函数在给定的数据加载器上评估模型性能，不更新模型参数，
    并返回平均损失、准确率以及所有预测结果和真实标签。

    参数:
        model: PyTorch模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备（CPU或GPU）

    返回:
        tuple: (epoch_loss, epoch_acc, all_preds, all_labels)
               平均损失、准确率(%)、所有预测结果和真实标签
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0  # 累计损失
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数
    all_preds = []  # 所有预测结果
    all_labels = []  # 所有真实标签

    # 不计算梯度，提高推理速度和减少内存使用
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for inputs, labels in pbar:
            # 将数据移至指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)  # 累加批次损失
            _, predicted = outputs.max(1)  # 获取预测类别
            total += labels.size(0)  # 累加样本数
            correct += predicted.eq(labels).sum().item()  # 累加正确预测数

            # 收集预测结果和真实标签，用于后续分析
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条显示的信息
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

    # 计算整个验证集的平均损失和准确率
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    绘制训练和验证曲线

    该函数绘制训练过程中的损失和准确率曲线，用于可视化模型的训练进展，
    并保存为图像文件。

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 图像保存路径
    """
    plt.figure(figsize=(12, 5))  # 创建图形对象并设置大小

    # 绘制损失曲线
    plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    plt.plot(train_losses, label='Train Loss')  # 绘制训练损失曲线
    plt.plot(val_losses, label='Validation Loss')  # 绘制验证损失曲线
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Loss')  # y轴标签
    plt.title('Validation Loss')  # 标题
    plt.legend()  # 显示图例

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    plt.plot(train_accs, label='Train Accuracy')  # 绘制训练准确率曲线
    plt.plot(val_accs, label='Validation Accuracy')  # 绘制验证准确率曲线
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Accuracy (%)')  # y轴标签
    plt.title('Training and Validation Accuracy')  # 标题
    plt.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图形，释放内存


def train_model(model_type, data_dir, batch_size, num_epochs, lr, weight_decay, save_dir, se_layers=None,
                se_reduction=16):
    """
    训练和评估模型的主函数

    该函数负责整个模型训练流程，包括数据加载、模型创建、训练循环、验证、
    模型保存、日志记录和最终评估。

    参数:
        model_type: 模型类型，'resnet50'或'se_resnet50'
        data_dir: 数据集目录路径
        batch_size: 训练批量大小
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减系数
        save_dir: 结果保存目录

    返回:
        dict: 包含测试集上各项评价指标的字典
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 创建日志目录
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 获取计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir, batch_size=batch_size
    )

    # 获取类别名称
    class_names = get_class_names(data_dir)

    # 根据指定类型创建模型
    if model_type == 'resnet50':
        model = ResNet50(num_classes=len(class_names))
        model_name = 'ResNet50'
    elif model_type == 'se_resnet50':
        model = SEResNet50(num_classes=len(class_names),
                           se_layers=se_layers if se_layers else ['layer1', 'layer2', 'layer3', 'layer4'],
                           reduction=se_reduction)
        model_name = f'SE-ResNet50-{"-".join(se_layers)}' if se_layers else 'SE-ResNet50-default'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 将模型移至指定设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW优化器
    # 余弦退火学习率调度器，使学习率在训练过程中周期性变化
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 初始化训练日志记录器
    logger = TrainingLogger(log_dir, model_name)

    # 记录超参数配置
    hyperparams = {
        'model_type': model_type,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'device': str(device),
        'num_classes': len(class_names),
        'dataset_path': data_dir
    }
    logger.log_hyperparameters(hyperparams)

    # 初始化训练历史记录
    train_losses = []  # 训练损失历史
    val_losses = []  # 验证损失历史
    train_accs = []  # 训练准确率历史
    val_accs = []  # 验证准确率历史
    best_val_acc = 0.0  # 最佳验证准确率

    # 记录训练开始时间
    start_time = time.time()

    # 训练循环 - 遍历每个epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 在验证集上评估
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录本轮训练信息到日志
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=current_lr
        )

        # 更新学习率
        scheduler.step()

        # 打印当前epoch的结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")

        # 保存验证准确率最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            logger.log_best_model(best_epoch, val_acc)

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")

    # 保存学习曲线
    logger.save_learning_curves()

    # 绘制训练曲线（保持原有功能）
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(save_dir, f"{model_name}_training_curves.png")
    )

    # 在测试集上评估
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_best.pth")))

    # 使用评估函数进行全面评估，生成各种指标和可视化结果
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        save_dir=save_dir,
        model_name=model_name
    )

    # 记录测试指标到日志
    logger.log_test_metrics(metrics)

    # 记录训练完成信息
    logger.log_training_complete(total_time)

    print(f"Test Loss: {metrics['loss']:.4f}, Test Acc: {metrics['accuracy'] * 100:.2f}%")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"训练日志保存至 {log_dir}")

    # 返回主要评价指标
    return {
        'accuracy': metrics['accuracy'] * 100,  # 转换为百分比
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'precision_weighted': metrics['precision_weighted'],  # 添加加权精确率
        'recall_weighted': metrics['recall_weighted'],  # 添加加权召回率
        'cohen_kappa': metrics['cohen_kappa']
    }


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="在鱼类数据集上训练ResNet50或SE-ResNet50模型")

    # 添加命令行参数
    parser.add_argument('--model', type=str, choices=['resnet50', 'se_resnet50'], required=True,
                        help='Model type: resnet50 or se_resnet50')
    parser.add_argument('--data_dir', type=str, default='datas',
                        help='Data directory containing train, val, and test folders')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--se_layers', nargs='+', default=None,
                        help="添加SE模块的层列表，如 '--se_layers layer1 layer3'")
    parser.add_argument('--se_reduction', type=int, default=16,
                        help="SE模块的通道压缩比例（默认16）")

    # 解析命令行参数
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 训练模型
    metrics = train_model(
        model_type=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        se_layers=args.se_layers,
        se_reduction=args.se_reduction
    )

    # 打印指标
    print("\nSummary of Test Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")