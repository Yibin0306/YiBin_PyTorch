# 导入torchvision库，用于计算机视觉相关任务（如预训练模型、数据集加载等）
import torchvision
# 从torch中导入神经网络模块nn
from torch import nn

# 以下代码被注释掉，用于加载ImageNet训练数据集（需要下载约138GB数据）
# train_data = torchvision.datasets.ImageNet(root='./ImageNet_DATA',
#                                        split='train',
#                                        download=True,
#                                        transform=torchvision.transforms.ToTensor())

# 加载未预训练的VGG16模型（随机初始化权重）
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 加载预训练的VGG16模型（在ImageNet数据集上训练过的权重）
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 打印预训练模型的结构
print(vgg16_true)

# 加载CIFAR10训练数据集（10分类小图片数据集）
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10_DATA',
                                       train=True,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

# 修改预训练模型（输出1000类）以适应CIFAR10的10分类任务
# 在classifier末尾添加一个新的全连接层（1000->10）
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# 打印修改后的模型结构
print(vgg16_true)

# 修改未预训练模型的最后一层全连接层（原始输出1000类）
# 直接替换第6层（索引6）为新的全连接层（4096->10）
vgg16_false.classifier[6] = nn.Linear(4096, 10)
# 打印修改后的模型结构
print(vgg16_false)