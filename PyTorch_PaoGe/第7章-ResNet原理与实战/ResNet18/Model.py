import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块
from torchsummary import summary  # 导入模型结构摘要工具


class ResidualBlock(nn.Module):
    """
    ResNet的核心组件：残差块（Residual Block）
    通过跳跃连接（skip connection）解决深层网络梯度消失问题
    """

    def __init__(self, in_planes, planes, stride=1, use_1x1=False):
        """
        初始化残差块

        参数:
            in_planes: 输入通道数
            planes: 输出通道数（也是中间层的通道数）
            stride: 卷积步长，用于下采样
            use_1x1: 是否使用1x1卷积调整跳跃连接的维度
        """
        super(ResidualBlock, self).__init__()
        self.ReLU = nn.ReLU()  # ReLU激活函数

        # 第一个卷积层：3x3卷积，可能包含下采样（当stride>1时）
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes,
                               kernel_size=3, stride=stride, padding=1)

        # 第二个卷积层：3x3卷积，保持特征图尺寸
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,
                               kernel_size=3, padding=1)

        # 批量归一化层：加速训练，提高稳定性
        self.bn1 = nn.BatchNorm2d(planes)  # 第一个BN层
        self.bn2 = nn.BatchNorm2d(planes)  # 第二个BN层

        # 1x1卷积：当输入输出维度不匹配时，用于调整跳跃连接的维度
        if use_1x1:
            self.conv1x1 = nn.Conv2d(in_channels=in_planes, out_channels=planes,
                                     kernel_size=1, stride=stride, padding=0)
        else:
            self.conv1x1 = None  # 维度匹配时不需要1x1卷积

    def forward(self, x):
        """
        前向传播过程：实现残差连接 F(x) + x

        参数:
            x: 输入张量

        返回:
            y: 经过残差块处理后的输出
        """
        # 主路径：两个卷积层 + 批量归一化 + 激活函数
        y = self.ReLU(self.bn1(self.conv1(x)))  # 卷积1 -> BN1 -> ReLU
        y = self.bn2(self.conv2(y))  # 卷积2 -> BN2（注意：这里没有立即加ReLU）

        # 跳跃连接：如果维度不匹配，使用1x1卷积调整输入x的维度
        if self.conv1x1 is not None:
            x = self.conv1x1(x)  # 使用1x1卷积调整跳跃连接的维度

        # 残差连接：主路径输出 + 跳跃连接，然后应用ReLU
        y = self.ReLU(y + x)  # F(x) + x -> ReLU

        return y


class ResNet(nn.Module):
    """
    ResNet网络主体结构
    包含多个残差块，实现深度残差网络
    """

    def __init__(self, ResidualBlock):
        """
        初始化ResNet网络
        """
        super(ResNet, self).__init__()

        # 模块1：初始特征提取（非残差块）
        # 输入: 1x224x224 -> 输出: 64x56x56
        self.block1 = nn.Sequential(
            # 7x7大卷积核，快速提取特征并下采样
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),  # ReLU激活
            # 最大池化，进一步下采样
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块2：第一个残差块组（2个残差块）
        # 输入: 64x56x56 -> 输出: 64x56x56（尺寸不变）
        self.block2 = nn.Sequential(
            ResidualBlock(in_planes=64, planes=64, stride=1, use_1x1=False),  # 维度匹配，不需要1x1卷积
            ResidualBlock(in_planes=64, planes=64, stride=1, use_1x1=False),  # 维度匹配，不需要1x1卷积
        )

        # 模块3：第二个残差块组（包含下采样）
        # 输入: 64x56x56 -> 输出: 128x28x28（通道数翻倍，尺寸减半）
        self.block3 = nn.Sequential(
            ResidualBlock(in_planes=64, planes=128, stride=2, use_1x1=True),  # 下采样，需要1x1卷积调整维度
            ResidualBlock(in_planes=128, planes=128, stride=1, use_1x1=False),  # 维度匹配，不需要1x1卷积
        )

        # 模块4：第三个残差块组（进一步下采样）
        # 输入: 128x28x28 -> 输出: 256x14x14（通道数翻倍，尺寸减半）
        self.block4 = nn.Sequential(
            ResidualBlock(in_planes=128, planes=256, stride=2, use_1x1=True),  # 下采样，需要1x1卷积
            ResidualBlock(in_planes=256, planes=256, stride=1, use_1x1=False),  # 维度匹配
        )

        # 模块5：第四个残差块组（最后下采样）
        # 输入: 256x14x14 -> 输出: 512x7x7（通道数翻倍，尺寸减半）
        self.block5 = nn.Sequential(
            ResidualBlock(in_planes=256, planes=512, stride=2, use_1x1=True),  # 下采样，需要1x1卷积
            ResidualBlock(in_planes=512, planes=512, stride=1, use_1x1=False),  # 维度匹配
        )

        # 模块6：分类头部
        # 输入: 512x7x7 -> 输出: 10个类别分数
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化：将任意尺寸特征图池化为1x1
            nn.Flatten(),  # 展平为向量
            nn.Linear(in_features=512, out_features=10),  # 全连接层，输出10个类别
        )

    def forward(self, x):
        """
        前向传播：数据依次通过6个模块

        参数:
            x: 输入图像张量，形状为 [batch_size, 1, 224, 224]

        返回:
            x: 分类结果，形状为 [batch_size, 10]
        """
        # 数据流: 1x224x224 -> 64x56x56 -> 64x56x56 -> 128x28x28 -> 256x14x14 -> 512x7x7 -> 10
        x = self.block1(x)  # 初始特征提取
        x = self.block2(x)  # 第一个残差块组
        x = self.block3(x)  # 第二个残差块组（下采样）
        x = self.block4(x)  # 第三个残差块组（下采样）
        x = self.block5(x)  # 第四个残差块组（下采样）
        x = self.block6(x)  # 分类头部
        return x


# 主程序入口
if __name__ == '__main__':
    # 设置计算设备：优先使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建ResNet模型实例并移动到相应设备
    model = ResNet(ResidualBlock).to(device)

    # 打印模型结构摘要
    # 输入尺寸: (通道数, 高度, 宽度) = (1, 224, 224)
    print(summary(model, (1, 224, 224)))