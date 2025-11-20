import torch  # 导入必要的库
from torch import nn  # nn模块包含构建神经网络所需的各种类
from torchsummary import summary  # 用于打印模型结构摘要


# 定义LeNet类，继承自nn.Module
class LeNet(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super(LeNet, self).__init__()

        # 第一个卷积层：输入通道1（灰度图），输出通道6，卷积核大小5x5，填充2
        # 填充2是为了保持输入输出尺寸一致（28x28 -> 28x28）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        # 第一个平均池化层：池化窗口2x2，步长2（下采样）
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二个卷积层：输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 第二个平均池化层：池化窗口2x2，步长2
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Sigmoid激活函数（原始LeNet使用的激活函数）
        self.sigmoid = nn.Sigmoid()

        # 展平层：将多维特征图转换为一维向量
        self.flatten = nn.Flatten()

        # 第一个全连接层：输入特征16 * 5 * 5=400，输出特征120
        self.linear = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # 第二个全连接层：输入特征120，输出特征84
        self.linear2 = nn.Linear(in_features=120, out_features=84)

        # 第三个全连接层（输出层）：输入特征84，输出特征10（对应10个数字类别）
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 第一个卷积块：卷积 -> Sigmoid激活 -> 池化
        x = self.sigmoid(self.conv1(x))
        x = self.avgpool1(x)

        # 第二个卷积块：卷积 -> Sigmoid激活 -> 池化
        x = self.sigmoid(self.conv2(x))
        x = self.avgpool2(x)

        # 将特征图展平为一维向量
        x = self.flatten(x)

        # 全连接层部分
        x = self.linear(x)  # 第一层全连接
        x = self.linear2(x)  # 第二层全连接
        x = self.linear3(x)  # 输出层（未使用激活函数，通常配合交叉熵损失函数使用）

        return x


# 主程序入口
if __name__ == '__main__':
    # 设置设备：优先使用GPU（CUDA），如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建LeNet模型实例并移动到相应设备
    model = LeNet().to(device)

    # 打印模型结构摘要，输入尺寸为(1, 28, 28)（单通道28x28图像，如MNIST数据集）
    print(summary(model, (1, 28, 28)))