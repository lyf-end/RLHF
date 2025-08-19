# PyTorch在计算机视觉（CV）中的应用

得益于其直观的API和强大的`torchvision`库，PyTorch在计算机视觉领域占据主导地位。本指南探讨了用于CV任务的基本PyTorch工具。

## 1. `torchvision`: CV工具箱

`torchvision`是一个库，为计算机视觉提供流行的数据集、模型架构和常见的图像转换。

### 关键组件:
- `torchvision.datasets`: 访问ImageNet, CIFAR10, MNIST等数据集。
- `torchvision.models`: 预训练模型，如ResNet, VGG, MobileNet。
- `torchvision.transforms`: 用于数据增强和预处理的常见图像转换。

### 示例: 数据加载和转换

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 定义一系列转换
transform = transforms.Compose(
    [transforms.Resize((224, 224)), # 调整图像大小
     transforms.ToTensor(), # 将图像转换为PyTorch张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 标准化张量

# 加载CIFAR10训练数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 创建一个数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# 获取一批图像和标签
dataiter = iter(trainloader)
images, labels = next(dataiter)

print("图像批次形状:", images.shape) # 预期: (4, 3, 224, 224)
print("标签批次形状:", labels.shape) # 预期: (4)
```

## 2. 构建卷积神经网络（CNNs）

CNN是大多数现代CV模型的支柱。`torch.nn`提供了所有必要的层。

### 关键层:
- `nn.Conv2d`: 2D卷积层。
- `nn.MaxPool2d`: 最大池化层。
- `nn.ReLU`: 修正线性单元激活函数。
- `nn.BatchNorm2d`: 批归一化，用于稳定训练。
- `nn.Flatten`: 将张量展平以传递给线性层。

### 示例: 一个简单的CNN

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第1个卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 第2个卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 256) # 假设输入图像为32x32，池化两次
        self.fc2 = nn.Linear(256, 10) # 10个输出类别

    def forward(self, x):
        # 输入形状: (batch_size, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x))) # -> (batch_size, 16, 16, 16)
        x = self.pool(F.relu(self.conv2(x))) # -> (batch_size, 32, 8, 8)
        
        # 展平图像张量
        x = x.view(-1, 32 * 8 * 8) # -> (batch_size, 2048)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleCNN()
print(model)

# 虚拟输入 (batch_size=4, channels=3, height=32, width=32)
dummy_input = torch.randn(4, 3, 32, 32)
output = model(dummy_input)
print("输出形状:", output.shape) # 预期: (4, 10)
```

## 3. 使用预训练模型（迁移学习）

迁移学习是一种强大的技术，将在大型数据集上训练的模型应用于一个较小的、不同的数据集。`torchvision.models`使这变得容易。

### 示例: 微调ResNet模型

```python
import torchvision.models as models
import torch.nn as nn

# 加载一个预训练的ResNet18模型
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 冻结网络中的所有参数
for param in resnet18.parameters():
    param.requires_grad = False

# 获取分类器的输入特征数
num_ftrs = resnet18.fc.in_features

# 用一个新的全连接层替换最后一个全连接层以适应我们的任务
# 假设我们有100个类别
resnet18.fc = nn.Linear(num_ftrs, 100)

# 现在，只有最后一层的参数会被训练。
# 之后可以解冻更多层进行微调。

# 虚拟输入
dummy_input = torch.randn(1, 3, 224, 224)
output = resnet18(dummy_input)
print("修改后的ResNet输出形状:", output.shape) # 预期: (1, 100)
```

## 4. 目标检测和分割

对于更高级的任务，如目标检测和分割，PyTorch和`torchvision`提供了像Faster R-CNN和Mask R-CNN这样的模型。

### 示例: 使用预训练的Faster R-CNN

```python
import torchvision.models as models
import torch

# 加载一个预训练的Faster R-CNN模型
model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# 将模型设置为评估模式
model.eval()

# 虚拟输入: 一个张量列表
# 每个张量是一个 (C, H, W) 图像
dummy_input = [torch.rand(3, 300, 400), torch.rand(3, 500, 600)]

# 获取预测结果
predictions = model(dummy_input)

print("预测数量:", len(predictions))
print("一个预测中的键:", predictions[0].keys())
# 预期键: 'boxes', 'labels', 'scores'
```

---

## PyTorch CV 功能图

```mermaid
graph TD
    A[PyTorch CV 核心] --> B(torchvision);
    A --> C(核心 nn 层);
    A --> D(高级模型);

    subgraph torchvision
        B1(数据集);
        B2(模型);
        B3(转换);
    end
    
    B1 --> B1a(CIFAR10, ImageNet);
    B2 --> B2a(ResNet, VGG);
    B3 --> B3a(ToTensor, Normalize, Resize);

    subgraph 核心 nn 层
        C1(nn.Conv2d);
        C2(nn.MaxPool2d);
        C3(nn.BatchNorm2d);
        C4(nn.ReLU);
        C5(nn.Linear);
    end

    subgraph 高级模型
        D1(迁移学习);
        D2(目标检测);
        D3(分割);
    end
    
    B2 -- 预训练权重 --> D1;
    D1 -- 微调 --> D2;
    D1 -- 微调 --> D3;
    
    C --> D1;
    B3 -- 预处理 --> C;
