# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的视觉数据集模块
import torchvision.datasets
# 从torch中导入神经网络模块
from torch import nn
# 导入神经网络中的所有层类型
from torch.nn import *
# 导入数据加载工具
from torch.utils.data import DataLoader

# 加载CIFAR-10测试数据集（实际训练中通常使用train=True加载训练集）
# root：数据集存储路径
# train=False：加载测试集（应为True加载训练集）
# download=True：如果本地不存在则自动下载
# transform=ToTensor()：将PIL图像转换为张量并归一化到[0,1]
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10_DATA',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

# 创建数据加载器
# batch_size=1：每次加载一个样本（实际训练中通常设为更大的批次）
dataloader = DataLoader(dataset, batch_size=1)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 使用Sequential容器定义网络结构
        self.model1 = Sequential(
            # 卷积层1：输入通道3(RGB), 输出通道32, 5x5卷积核, padding=2保持空间尺寸
            Conv2d(3, 32, 5, padding=2),
            # 2x2最大池化层，步幅默认为2，尺寸减半
            MaxPool2d(2),
            # 卷积层2：输入32通道, 输出32通道, 5x5卷积核
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),  # 再次池化
            # 卷积层3：输入32通道, 输出64通道, 5x5卷积核
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),  # 最终尺寸变为原图1/8
            # 展平层：将多维特征图转换为一维向量
            Flatten(),
            # 全连接层1：1024维输入（需检查实际输入尺寸），输出64维
            Linear(1024, 64),  # 注意：CIFAR-10图像32x32，经过三次池化后为4x4，64通道 -> 4 * 4 * 64=1024
            # 全连接层2（输出层）：64维输入，10维输出（对应10个类别）
            Linear(64, 10)
        )

    def forward(self, x):
        # 前向传播：顺序执行模型中的各层
        x = self.model1(x)
        return x


# 创建交叉熵损失函数（包含Softmax）
loss = nn.CrossEntropyLoss()
# 实例化神经网络
net = Net()
# 创建随机梯度下降优化器，学习率0.01
optim = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练循环：20个epoch
for epoch in range(20):
    running_loss = 0.0  # 当前epoch的累计损失

    # 遍历数据集中的所有批次
    for data in dataloader:
        # 解包数据：图像和标签
        imgs, targets = data
        # 前向传播获取预测结果
        output = net(imgs)
        # 计算当前批次的损失
        result_loss = loss(output, targets)

        # 梯度清零（防止梯度累加）
        optim.zero_grad()
        # 反向传播计算梯度
        result_loss.backward()
        # 使用优化器更新权重
        optim.step()

        # 累加损失（注意：此处直接相加而非取平均）
        running_loss += result_loss

    # 打印当前epoch的总损失
    print(f'Epoch {epoch + 1} Loss: {running_loss}')