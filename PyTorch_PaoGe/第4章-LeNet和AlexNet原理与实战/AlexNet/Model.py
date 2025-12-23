import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from torchsummary import summary  # 导入模型结构摘要工具
import torch.nn.functional as F  # 导入函数式接口，包含Dropout等操作


class AlexNet(nn.Module):
    """
    AlexNet神经网络模型类
    这是2012年ImageNet竞赛冠军模型，开创了深度学习在计算机视觉领域的应用
    原始论文：ImageNet Classification with Deep Convolutional Neural Networks
    """

    def __init__(self):
        """
        初始化函数：定义网络的所有层结构
        继承自nn.Module，必须调用父类的初始化方法
        """
        super(AlexNet, self).__init__()  # 调用父类nn.Module的初始化方法

        # 定义激活函数 - ReLU（Rectified Linear Unit）激活函数
        # ReLU能够有效解决梯度消失问题，加速模型收敛
        self.ReLU = nn.ReLU()

        # 第一个卷积层：输入通道1（灰度图），输出通道96，卷积核11x11，步长4
        # 使用大卷积核和较大步长来快速降低特征图尺寸，同时捕获较大范围的视觉特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)

        # 第一个最大池化层：池化窗口3x3，步长2
        # 进一步降低特征图尺寸，增强特征的平移不变性
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 第二个卷积层：输入通道96，输出通道256，卷积核5x5，填充2
        # 填充2是为了保持特征图尺寸不变（输入输出尺寸相同）
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)

        # 第二个最大池化层：池化窗口3x3，步长2
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 第三个卷积层：输入通道256，输出通道384，卷积核3x3，填充1
        # 使用更小的卷积核来捕获更精细的特征
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        # 第四个卷积层：输入通道384，输出通道384，卷积核3x3，填充1
        # 通道数保持不变，加深网络深度
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        # 第五个卷积层：输入通道384，输出通道256，卷积核3x3，填充1
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        # 第三个最大池化层：池化窗口3x3，步长2
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 展平层：将多维特征图转换为一维向量，为全连接层做准备
        self.flatten = nn.Flatten()

        # 第一个全连接层：输入特征256 * 6 * 6=9216，输出特征4096
        # 原始AlexNet使用Dropout防止过拟合，这里在forward中实现
        self.Linear1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)

        # 第二个全连接层：输入特征4096，输出特征4096
        self.Linear2 = nn.Linear(in_features=4096, out_features=4096)

        # 第三个全连接层（输出层）：输入特征4096，输出特征10（对应10个类别）
        # 输出层不使用激活函数，通常配合交叉熵损失函数
        self.Linear3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        """
        前向传播函数：定义数据通过网络的计算流程

        参数：
        x - 输入张量，形状为(batch_size, 1, 227, 227)

        返回：
        x - 输出张量，形状为(batch_size, 10)
        """

        # 第一个卷积块：卷积 -> ReLU激活 -> 最大池化
        x = self.ReLU(self.conv1(x))  # 输出形状: (batch_size, 96, 55, 55)
        x = self.maxpool1(x)  # 输出形状: (batch_size, 96, 27, 27)

        # 第二个卷积块：卷积 -> ReLU激活 -> 最大池化
        x = self.ReLU(self.conv2(x))  # 输出形状: (batch_size, 256, 27, 27)
        x = self.maxpool2(x)  # 输出形状: (batch_size, 256, 13, 13)

        # 第三个卷积层：卷积 -> ReLU激活
        x = self.ReLU(self.conv3(x))  # 输出形状: (batch_size, 384, 13, 13)

        # 第四个卷积层：卷积 -> ReLU激活
        x = self.ReLU(self.conv4(x))  # 输出形状: (batch_size, 384, 13, 13)

        # 第五个卷积块：卷积 -> ReLU激活 -> 最大池化
        x = self.ReLU(self.conv5(x))  # 输出形状: (batch_size, 256, 13, 13)
        x = self.maxpool3(x)  # 输出形状: (batch_size, 256, 6, 6)

        # 展平特征图：将(batch_size, 256, 6, 6)转换为(batch_size, 256 * 6 * 6=9216)
        x = self.flatten(x)  # 输出形状: (batch_size, 9216)

        # 全连接层部分
        x = self.ReLU(self.Linear1(x))  # 输出形状: (batch_size, 4096)
        x = F.dropout(x, p=0.5)  # Dropout正则化，随机丢弃50%的神经元，防止过拟合
        x = self.ReLU(self.Linear2(x))  # 输出形状: (batch_size, 4096)
        x = F.dropout(x, p=0.5)  # 再次Dropout
        x = self.Linear3(x)  # 输出形状: (batch_size, 10) - 最终分类结果

        return x


# 主程序入口：当直接运行此脚本时执行以下代码
if __name__ == '__main__':
    """
    测试代码：创建AlexNet模型实例并打印模型结构摘要
    """

    # 设置计算设备：优先使用GPU（CUDA），如果没有则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前使用的设备:', device)  # 打印设备信息

    # 创建AlexNet模型实例并移动到指定设备
    model = AlexNet().to(device)

    # 打印模型结构摘要
    # summary函数显示每层的输出形状、参数数量等信息
    # 输入尺寸为(1, 227, 227) - 单通道227x227图像
    print(summary(model, (1, 227, 227)))