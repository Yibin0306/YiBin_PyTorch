"""
GoogLeNet 神经网络实现
论文: Going Deeper with Convolutions (2014)
特点: 使用Inception模块，在保持计算效率的同时增加网络深度和宽度
"""

import torch
import torch.nn as nn
from torchsummary import summary


class Inception(nn.Module):
    """
    Inception模块 - GoogLeNet的核心组件
    通过并行使用不同大小的卷积核来捕获多尺度特征

    参数:
        in_channels: 输入通道数
        c1, c2, c3, c4: 四个分支的输出通道数配置
            c1: 1x1卷积分支的输出通道数
            c2: 1x1卷积 + 3x3卷积分支的输出通道数元组
            c3: 1x1卷积 + 5x5卷积分支的输出通道数元组  
            c4: 最大池化 + 1x1卷积分支的输出通道数
    """

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()

        # 分支1: 1x1卷积 - 主要用于降维和特征提取
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 分支2: 1x1卷积 + 3x3卷积 - 捕获中等感受野的特征
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        # 分支3: 1x1卷积 + 5x5卷积 - 捕获大感受野的特征
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        # 分支4: 最大池化 + 1x1卷积 - 保留原始特征并进行降维
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        """
        前向传播过程
        四个分支并行计算，最后在通道维度上拼接结果

        参数:
            x: 输入张量，形状为 [batch_size, in_channels, height, width]

        返回:
            拼接后的特征图，通道数 = c1 + c2[1] + c3[1] + c4
        """
        # 分支1: 直接1x1卷积
        p1 = self.ReLU(self.p1_1(x))

        # 分支2: 1x1卷积 -> ReLU -> 3x3卷积 -> ReLU
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))

        # 分支3: 1x1卷积 -> ReLU -> 5x5卷积 -> ReLU  
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))

        # 分支4: 最大池化 -> 1x1卷积 -> ReLU
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))

        # 在通道维度(dim=1)上拼接四个分支的输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet网络主体结构
    包含5个主要模块(block)，逐渐降低空间分辨率，增加通道数
    """

    def __init__(self, Inception):
        super(GoogLeNet, self).__init__()

        # 模块1: 初始特征提取
        # 输入: 1x224x224 -> 输出: 64x56x56
        self.block1 = nn.Sequential(
            # 7x7卷积，大步长(2)快速下采样
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # 最大池化进一步下采样
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块2: 特征增强
        # 输入: 64x56x56 -> 输出: 192x28x28  
        self.block2 = nn.Sequential(
            # 1x1卷积进行特征变换
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            # 3x3卷积扩展通道数
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            # 下采样
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块3: 第一个Inception模块组
        # 输入: 192x28x28 -> 输出: 480x14x14
        self.block3 = nn.Sequential(
            # 第一个Inception: 192 -> 256通道
            Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32),
            # 第二个Inception: 256 -> 480通道  
            Inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64),
            # 下采样
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块4: 主要Inception模块组（网络最宽的部分）
        # 输入: 480x14x14 -> 输出: 832x7x7
        self.block4 = nn.Sequential(
            # 5个连续的Inception模块，通道数在512-832之间
            Inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),  # 480 -> 512
            Inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),  # 512 -> 512
            Inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),  # 512 -> 512
            Inception(in_channels=512, c1=112, c2=(128, 288), c3=(32, 64), c4=64),  # 512 -> 528
            Inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128),  # 528 -> 832
            # 最后一次下采样
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块5: 分类头部
        # 输入: 832x7x7 -> 输出: 10个类别分数
        self.block5 = nn.Sequential(
            # 最后两个Inception模块
            Inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),  # 832 -> 832
            Inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128),  # 832 -> 1024
            # 自适应平均池化：将任意大小的特征图池化为1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            # 展平特征图用于全连接层
            nn.Flatten(),
            # 最终分类层：1024特征 -> 10个类别
            nn.Linear(in_features=1024, out_features=10),
        )

        # 权重初始化部分 - 对网络中的所有卷积层和全连接层进行初始化
        for m in self.modules():
            # 如果是卷积层，使用Kaiming初始化（针对ReLU优化）
            if isinstance(m, nn.Conv2d):
                # Kaiming正态分布初始化，考虑ReLU的非线性特性
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果存在偏置项，初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # 如果是全连接层，使用较小的正态分布初始化
            elif isinstance(m, nn.Linear):
                # 正态分布初始化，均值为0，标准差为0.01
                nn.init.normal_(m.weight, 0, 0.01)
                # 偏置项初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        网络前向传播流程
        参数:
            x: 输入图像张量，形状为 [batch_size, 1, 224, 224]
        返回:
            类别预测分数，形状为 [batch_size, 10]
        """
        # 数据流: 1x224x224 -> 64x56x56 -> 192x28x28 -> 480x14x14 -> 832x7x7 -> 10
        x = self.block1(x)  # 初始特征提取
        x = self.block2(x)  # 特征增强
        x = self.block3(x)  # Inception模块组1
        x = self.block4(x)  # Inception模块组2（主要特征学习）
        x = self.block5(x)  # 分类头部
        return x


if __name__ == '__main__':
    # 设置设备（GPU如果可用，否则CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例并移动到相应设备
    model = GoogLeNet(Inception).to(device)

    # 打印模型结构摘要
    # 输入尺寸: (通道数, 高度, 宽度) = (1, 224, 224)
    print(summary(model, (1, 224, 224)))