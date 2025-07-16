# 导入必要的库
import torchvision.datasets  # PyTorch视觉数据集模块
from torch import nn  # 神经网络模块
from torch.nn import *  # 导入所有神经网络组件
from torch.utils.data import DataLoader  # 数据加载器

# ===================== 数据集准备 =====================
# 下载并加载CIFAR-10测试集（train=False）
# 参数说明:
#   root: 数据集存储路径
#   train: False表示使用测试集
#   download: True表示如果数据集不存在则自动下载
#   transform: 将图像转换为Tensor格式，并自动归一化到[0,1]
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10_DATA',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

# 创建数据加载器，batch_size=1表示每次加载一个样本
dataloader = DataLoader(dataset, batch_size=1)


# ===================== 神经网络定义 =====================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 使用Sequential容器按顺序构建网络
        self.model1 = Sequential(
            # 3通道输入 -> 32通道输出, 5x5卷积核, padding=2保持特征图大小不变
            Conv2d(3, 32, 5, padding=2),
            # 2x2最大池化，步幅默认为2，特征图尺寸减半
            MaxPool2d(2),
            # 32通道 -> 32通道, 5x5卷积核
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            # 32通道 -> 64通道, 5x5卷积核
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            # 将多维特征图展平为一维向量
            Flatten(),
            # 全连接层: 输入特征数1024 -> 输出64
            # 计算来源: 最后一次池化后特征图大小为(64, 4, 4) -> 64 * 4 * 4=1024
            Linear(1024, 64),
            # 全连接层: 64 -> 10 (对应CIFAR-10的10个类别)
            Linear(64, 10)
        )

    def forward(self, x):
        # 前向传播: 将输入通过整个序列模型
        x = self.model1(x)
        return x


# ===================== 模型训练准备 =====================
# 创建交叉熵损失函数 (常用于分类任务)
loss = nn.CrossEntropyLoss()
# 实例化神经网络
net = Net()

# ===================== 训练循环 =====================
# 遍历数据集中的所有样本
for data in dataloader:
    # 解包数据: 图像和对应标签
    imgs, targets = data

    # 前向传播: 获取模型预测输出
    output = net(imgs)

    # 计算损失: 比较预测输出和真实标签
    result_loss = loss(output, targets)

    # 反向传播: 计算所有参数的梯度
    result_loss.backward()

    # 注意: 这里缺少优化器步骤(如optimizer.step())和梯度清零(optimizer.zero_grad())
    # 因此实际训练中需要添加这些步骤才能更新权重

    pass  # 占位符，实际训练中可添加更多逻辑