import torch
import torch.nn as nn
from torchvision.models import resnet50


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBottleneck(nn.Module):
    """带有SE模块的Bottleneck块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    # 标准ResNet-50模型


class ResNet50(nn.Module):
    """
    标准ResNet-50模型

    基于PyTorch预训练的ResNet-50模型，修改最后的全连接层以适应鱼类分类任务。

    参数:
        num_classes: 分类类别数量，默认为30
    """

    def __init__(self, num_classes=30):
        super(ResNet50, self).__init__()
        # 加载预训练的ResNet-50模型
        self.model = resnet50(pretrained=True)
        # 修改最后的全连接层以匹配我们的类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x: 输入图像张量，形状为[batch_size, channels, height, width]

        返回:
            模型输出的类别预测分数，形状为[batch_size, num_classes]
        """
        return self.model(x)


class SEResNet50(nn.Module):
    """
    最小修改的SE-ResNet50，仅新增se_layers参数控制SE添加位置
    保持原始权重加载逻辑不变
    """

    def __init__(self, num_classes=30, se_layers=['layer1', 'layer2', 'layer3', 'layer4'], reduction=16):
        super().__init__()
        self.model = resnet50(pretrained=True)

        # 添加层选择逻辑
        for name, module in self.model.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if name not in se_layers:  # 跳过不需要SE的层
                    continue

                for i, bottleneck in enumerate(module):
                    # 仅替换当前层的Bottleneck为SEBottleneck
                    se_block = SEBottleneck(
                        inplanes=bottleneck.conv1.in_channels,
                        planes=bottleneck.conv1.out_channels,
                        stride=bottleneck.stride,
                        downsample=bottleneck.downsample,
                        reduction=reduction
                    )
                    # 保持原始权重复制逻辑
                    se_block.load_state_dict(bottleneck.state_dict(), strict=False)
                    module[i] = se_block

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)