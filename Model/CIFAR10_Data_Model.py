# 导入PyTorch库
import torch
from torch import nn  # 神经网络模块


# 定义神经网络模型类，继承自nn.Module
class Net(nn.Module):
    def __init__(self):
        """网络模型初始化"""
        super(Net, self).__init__()  # 调用父类初始化方法

        # 使用Sequential容器构建网络层（顺序执行）
        self.model = nn.Sequential(
            # 第一个卷积层：输入通道3(RGB)，输出通道32，5x5卷积核，步长1，填充2（保持尺寸）
            nn.Conv2d(3, 32, 5, 1, 2),
            # 2x2最大池化层（下采样，尺寸减半）
            nn.MaxPool2d(2),

            # 第二个卷积层：输入32通道，输出32通道
            nn.Conv2d(32, 32, 5, 1, 2),
            # 最大池化层
            nn.MaxPool2d(2),

            # 第三个卷积层：输入32通道，输出64通道
            nn.Conv2d(32, 64, 5, 1, 2),
            # 最大池化层
            nn.MaxPool2d(2),

            # 展平层：将三维特征图转换为一维向量
            nn.Flatten(),

            # 全连接层：输入维度64 * 4 * 4=1024，输出维度64
            nn.Linear(64 * 4 * 4, 64),
            # 输出层：64维特征→10维分类输出（对应CIFAR-10的10个类别）
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """定义前向传播过程"""
        x = self.model(x)  # 输入数据通过定义好的网络结构
        return x


# 当直接运行此脚本时执行以下代码
if __name__ == '__main__':
    # 创建网络实例
    net = Net()

    # 创建一个模拟输入张量（批大小64，3通道，32x32像素）
    # 符合CIFAR-10图像尺寸要求
    input = torch.ones(64, 3, 32, 32)

    # 通过网络前向传播
    output = net(input)

    # 打印输出形状（应为[64,10]）
    # 64：批大小，10：类别数量
    print(output.shape)