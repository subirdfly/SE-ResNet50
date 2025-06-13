from torchvision.models import resnet50
from models import SEResNet50  # 假设这是你的SE-ResNet实现

model = SEResNet50(se_layers=['layer1'], reduction=16)
print(model)  # 检查输出中目标层是否包含SEBlock相关结构