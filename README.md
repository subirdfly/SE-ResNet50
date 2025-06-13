# 鱼类分类 - ResNet-50 vs SE-ResNet-50 比较实验

本项目实现了一个鱼类图像分类实验，比较标准ResNet-50和自定义添加SE（Squeeze-and-Excitation）模块的SE-ResNet-50在鱼类分类任务上的性能差异。

## 数据集

数据集包含30种不同的鱼类图像，按照训练集、验证集和测试集进行了划分：

- 训练集：`datas/train/`
- 验证集：`datas/val/`
- 测试集：`datas/test/`

每个鱼类都有对应的中英文名称，保存在`datas/en2ch.txt`文件中。

## 项目结构

- `models.py`: 包含ResNet-50和自定义SE-ResNet-50模型的实现
- `dataset.py`: 数据集加载和预处理代码
- `train.py`: 模型训练和评估代码
- `metrics.py`: 各种评价指标的计算和可视化工具
- `logger.py`: 训练日志记录和保存工具
- `compare_models.py`: 比较两个模型性能的脚本（备用）
- `requirements.txt`: 项目依赖
- `testmodels.py`: 打印出模型形状，检测Se是否插入
- `fish_recognition_app(1)`:加载训练好的best.pth文件，使用模型进行图像识别
## 环境要求

项目依赖以下Python库：

```
torch>=1.10.0
torchvision>=0.11.1
numpy>=1.21.4
pillow>=8.4.0
matplotlib>=3.5.0
tqdm>=4.62.3
scikit-learn>=1.0.1
pandas>=1.5.3
seaborn>=0.12.0
```

## 使用方法

### 1. 训练单个模型

使用`train.py`脚本训练单个模型：

```bash
# 训练ResNet-50模型
python train.py --model resnet50 --batch_size 32 --epochs 30

# 训练SE-ResNet-50模型
python train.py --model se_resnet50 --se_layers layer1 layer 2--batch_size 32 --epochs 30
```

参数说明：
- `--model`: 模型类型，可选 'resnet50' 或 'se_resnet50'
- `--data_dir`: 数据集目录，默认为 'datas'
- `--se_layers`:自定义SE插入层级[layer1,layer2,layer3,layer4]
- `--batch_size`: 批量大小，默认为 32
- `--epochs`: 训练轮数，默认为 30
- `--lr`: 学习率，默认为 0.001
- `--weight_decay`: 权重衰减，默认为 0.01
- `--save_dir`: 结果保存目录，默认为 'results'
- `--seed`: 随机种子，默认为 42,实验中设为随机种子中的幸运种子3407

### 2. 比较两个模型（备用，做出来了，虽然没用还是贴出来）

使用`compare_models.py`脚本同时训练两个模型并比较它们的性能：

```bash
python compare_models.py --batch_size 32 --epochs 30
```
需要手动在train.py中修改默认的SE层数才能实现想要的比较，由于过于繁琐，文章中还是使用了单个模型运行，将结果并列进行比较。

    elif model_type == 'se_resnet50':
        model = SEResNet50(num_classes=len(class_names),
                           se_layers=se_layers if se_layers else ['layer1', 'layer2', 'layer3', 'layer4'],
                           reduction=se_reduction)
参数与`train.py`相同，但结果会保存在`comparison_results`目录中。

## 训练日志

本项目实现了详细的训练日志记录功能，可以帮助您跟踪和分析训练过程：

### 单模型训练日志

训练单个模型时，日志文件将保存在`{save_dir}/logs/`目录下，包括：

1. **文本日志**：`{model_name}_{timestamp}_log.txt`
   - 记录训练开始时间、超参数
   - 每个epoch的训练和验证指标
   - 最佳模型保存信息
   - 测试集评估结果
   - 总训练时间

2. **CSV指标记录**：`{model_name}_{timestamp}_metrics.csv`
   - 包含每个epoch的损失、准确率、学习率等
   - 方便导入到其他工具进行分析

3. **超参数记录**：`{model_name}_hyperparams.json`
   - 以JSON格式保存所有超参数

4. **测试指标**：`{model_name}_test_metrics.json`
   - 以JSON格式保存测试集上的详细评估指标

5. **学习曲线图**：
   - `{model_name}_learning_curves.png`：训练和验证的损失与准确率曲线
   - `{model_name}_lr_schedule.png`：学习率变化曲线

### 模型比较日志（备用）

比较两个模型时，除了各自的训练日志外，还会生成比较日志，保存在`{save_dir}/comparison_logs/`目录下：

1. **比较日志文件**：`comparison_{timestamp}_log.txt`
   - 记录比较开始时间和超参数
   - 两个模型的训练过程摘要
   - 详细的性能对比结果
   - 总比较时间

2. **比较结果JSON**：`comparison_{timestamp}_results.json`
   - 包含两个模型的所有指标
   - 各项指标的提升数值
   - 实验超参数和时间戳

3. **比较可视化**：
   - `accuracy_comparison.png`：准确率对比柱状图
   - `metrics_comparison.png`：多项指标对比图

4. **CSV比较表**：`model_comparison.csv`
   - 以表格形式记录所有对比指标
   - 方便导入到Excel等工具进行进一步分析

## 评价指标

本项目使用多种评价指标来全面评估模型性能：

1. **基本指标**：
   - 准确率 (Accuracy)：正确预测的样本比例
   - 损失值 (Loss)：交叉熵损失

2. **分类指标**：
   - 精确率 (Precision)：预测为正例中真正例的比例
   - 召回率 (Recall)：真正例中被正确预测的比例
   - F1分数 (F1-Score)：精确率和召回率的调和平均数
   - 各指标均包含宏平均(Macro)、微平均(Micro)和加权平均(Weighted)

3. **一致性指标**：
   - Cohen's Kappa：考虑随机正确的一致性指标

4. **可视化**：
   - 混淆矩阵 (Confusion Matrix)
   - 每个类别的性能指标
   - 模型性能比较图
## fish识别pth使用方法
在进行模型训练与对比后，将选中的最佳pth路径复制进fish_recognition.py中指定位置

``if model_name == "ResNet50":``

   ``self.model = ResNet50(num_classes=num_classes)``

   ``model_path = "ResNet50_best.pth"``
替换该处path

``elif model_name == "SE-ResNet50":``

   ``self.model = SEResNet50(num_classes=num_classes)``

   ``model_path = "SE-ResNet50_best.pth"``
替换该处path
## SE模块简介

SE（Squeeze-and-Excitation）模块是一种通道注意力机制，它通过显式地建模通道之间的相互依赖关系，自适应地重新校准通道特征响应。SE模块主要包含两个操作：

1. **Squeeze**：通过全局平均池化将每个通道的空间信息压缩成一个值，生成通道描述符
2. **Excitation**：通过两个全连接层和激活函数，生成每个通道的权重，并对原始特征进行重新加权

SE模块可以集成到各种网络架构中，通常能够提高模型的表示能力，而只增加很少的计算成本。

## 预期结果

通过添加SE模块，SE-ResNet-50模型预期会比标准ResNet-50模型获得更高的性能指标，通常能提升1-2个百分点的准确率。实验结果将包括：

1. 各模型的训练和验证曲线
2. 测试集上的全面评估指标
   - 准确率、精确率、召回率、F1分数等
   - 混淆矩阵和每个类别的性能分析
4. 详细的CSV格式比较结果
5. 完整的训练日志