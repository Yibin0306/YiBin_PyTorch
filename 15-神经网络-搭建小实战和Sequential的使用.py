# 导入PyTorch核心库
import torch
from torch import nn  # 神经网络模块
from torch.nn import *  # 导入所有神经网络层
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化工具


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 使用Sequential容器创建多层网络结构
        self.model1 = Sequential(
            # 卷积层1：输入3通道(RGB)，输出32通道，5x5卷积核，填充2保持尺寸
            Conv2d(3, 32, 5, padding=2),
            # 最大池化层1：2x2窗口，步长2（尺寸减半）
            MaxPool2d(2),

            # 卷积层2：输入32通道，输出32通道，5x5卷积核，填充2
            Conv2d(32, 32, 5, padding=2),
            # 最大池化层2：2x2窗口，步长2（尺寸再减半）
            MaxPool2d(2),

            # 卷积层3：输入32通道，输出64通道，5x5卷积核，填充2
            Conv2d(32, 64, 5, padding=2),
            # 最大池化层3：2x2窗口，步长2（尺寸第三次减半）
            MaxPool2d(2),

            # 展平层：将多维特征图转换为一维向量
            Flatten(),
            # 全连接层1：输入1024维，输出64维
            Linear(1024, 64),
            # 全连接层2（输出层）：输入64维，输出10维（对应10个类别）
            Linear(64, 10)
        )

    def forward(self, x):
        # 前向传播：依次通过所有层
        x = self.model1(x)
        return x


# 实例化神经网络
net = Net()

# 创建模拟输入数据：64张32x32的RGB图像（全1张量）
input = torch.ones((64, 3, 32, 32))  # 形状：[批量大小, 通道数, 高度, 宽度]

# 前向传播
output = net(input)
# 打印输出形状（应为[64, 10]）
print(output.shape)  # 输出：torch.Size([64, 10])

# 创建TensorBoard记录器
writer = SummaryWriter('TensorBoard')  # 日志保存到'TensorBoard'目录
# 添加计算图到TensorBoard
writer.add_graph(net, input)  # 记录网络结构和输入数据
# 关闭写入器
writer.close()