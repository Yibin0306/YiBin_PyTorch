import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from torchsummary import summary  # 导入模型结构摘要工具


class VGGNet(nn.Module):
    """
    VGGNet神经网络模型类
    这是牛津大学Visual Geometry Group提出的经典卷积神经网络
    主要特点：使用小卷积核(3×3)堆叠深层网络，结构简洁规整
    原始论文：Very Deep Convolutional Networks for Large-Scale Image Recognition
    """

    def __init__(self):
        """
        初始化函数：定义VGG网络的所有层结构
        继承自nn.Module，必须调用父类的初始化方法
        VGGNet有多种配置（VGG11, VGG13, VGG16, VGG19），这里实现的是VGG16-like结构
        """
        super(VGGNet, self).__init__()  # 调用父类nn.Module的初始化方法

        # 第一个卷积块：2个卷积层 + 1个最大池化层
        self.block1 = nn.Sequential(
            # 第一个卷积层：输入通道1（灰度图），输出通道64，卷积核3×3，填充1
            # 填充1可以保持特征图尺寸不变（输入输出尺寸相同）
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数，引入非线性
            # 第二个卷积层：输入输出通道都是64，保持特征图尺寸
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数
            # 最大池化层：池化窗口2×2，步长2（下采样，尺寸减半）
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第二个卷积块：2个卷积层 + 1个最大池化层
        self.block2 = nn.Sequential(
            # 卷积层：通道数从64增加到128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # 最大池化层：再次下采样，尺寸减半
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三个卷积块：3个卷积层 + 1个最大池化层
        self.block3 = nn.Sequential(
            # 卷积层：通道数从128增加到256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 第三个卷积层：保持256通道
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 最大池化层：下采样，尺寸减半
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第四个卷积块：3个卷积层 + 1个最大池化层
        self.block4 = nn.Sequential(
            # 卷积层：通道数从256增加到512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # 最大池化层：下采样，尺寸减半
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第五个卷积块：3个卷积层 + 1个最大池化层
        self.block5 = nn.Sequential(
            # 三个卷积层都保持512通道
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # 最后一个最大池化层：下采样，尺寸减半
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第六个全连接块：展平 + 3个全连接层
        self.block6 = nn.Sequential(
            # 展平层：将多维特征图转换为一维向量
            # 输入形状: (batch_size, 512, 7, 7) → 输出形状: (batch_size, 512 * 7 * 7=25088)
            nn.Flatten(),
            # 第一个全连接层：输入25088特征，输出4096特征
            nn.Linear(in_features=7 * 7 * 512, out_features=256),
            nn.ReLU(),  # ReLU激活函数
            # 第二个全连接层：输入4096特征，输出4096特征
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # 第三个全连接层（输出层）：输入4096特征，输出10特征（对应10个类别）
            # 注意：这里没有使用激活函数，通常配合交叉熵损失函数
            nn.Linear(in_features=128, out_features=10),
        )

        # 权重初始化：对模型中的所有层进行自定义权重初始化
        # 遍历模型中的所有模块
        for m in self.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态分布初始化（He初始化），针对ReLU激活函数优化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # 如果存在偏置项，初始化为0（初始值）
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 使用正态分布初始化，均值为0，标准差为0.01
                nn.init.normal_(m.weight, 0, std=0.01)
                # 如果存在偏置项，初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播函数：定义数据通过网络的计算流程

        参数：
        x - 输入张量，形状为(batch_size, 1, 224, 224)

        返回：
        x - 输出张量，形状为(batch_size, 10)

        数据流经过的尺寸变化：
        输入: (batch_size, 1, 224, 224)
        block1后: (batch_size, 64, 112, 112)   [池化尺寸减半]
        block2后: (batch_size, 128, 56, 56)    [池化尺寸减半]
        block3后: (batch_size, 256, 28, 28)    [池化尺寸减半]
        block4后: (batch_size, 512, 14, 14)    [池化尺寸减半]
        block5后: (batch_size, 512, 7, 7)      [池化尺寸减半]
        block6后: (batch_size, 10)             [全连接层输出]
        """

        # 依次通过6个模块
        x = self.block1(x)  # 第一个卷积块
        x = self.block2(x)  # 第二个卷积块
        x = self.block3(x)  # 第三个卷积块
        x = self.block4(x)  # 第四个卷积块
        x = self.block5(x)  # 第五个卷积块
        x = self.block6(x)  # 第六个全连接块

        return x  # 返回最终的分类结果


# 主程序入口：当直接运行此脚本时执行以下代码
if __name__ == '__main__':
    """
    测试代码：创建VGGNet模型实例并打印模型结构摘要
    """

    # 设置计算设备：优先使用GPU（CUDA），如果没有则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建VGGNet模型实例并移动到指定设备
    model = VGGNet().to(device)

    # 打印模型结构摘要
    # summary函数显示每层的输出形状、参数数量等信息
    # 输入尺寸为(1, 224, 224) - 单通道224x224图像
    print(summary(model, (1, 224, 224)))