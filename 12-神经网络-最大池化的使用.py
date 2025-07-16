"""
最大池化(Max Pooling)：
    目的：降低特征图的空间维度，减少计算量
    操作：在窗口内取最大值作为输出
    效果：保留最显著特征，增强位置不变性
    优点：减少计算量，增加网络对微小位置变化的鲁棒性
    缺点：可能丢失一些空间信息
"""
# 导入必要的PyTorch库
import torch
import torchvision.datasets  # PyTorch的计算机视觉数据集模块
from torch import nn  # 神经网络模块
from torch.nn import MaxPool2d  # 二维最大池化层
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化工具

# 加载CIFAR10数据集
dataset = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=False,  # 使用测试集（非训练集）
    download=True,  # 如果本地没有数据集则自动下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为PyTorch张量格式
)

# 创建数据加载器，用于批量处理数据
dataLoader = DataLoader(
    dataset,  # 加载的数据集
    batch_size=64  # 每批加载64张图像
)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 调用父类nn.Module的初始化方法

        # 创建最大池化层
        self.maxpool1 = MaxPool2d(
            kernel_size=3,  # 池化窗口大小3x3
            ceil_mode=False  # 当剩余部分不足池化窗口大小时，是否保留边缘
            # ceil_mode=False 表示忽略不足窗口大小的部分
            # ceil_mode=True 表示保留并池化剩余部分
        )

    def forward(self, input):
        # 前向传播：应用最大池化操作
        output = self.maxpool1(input)
        return output


# 实例化神经网络
net = Net()

# 创建TensorBoard日志记录器
writer = SummaryWriter('TensorBoard')  # 日志保存在"TensorBoard"目录
step = 0  # 初始化步数计数器，用于跟踪训练进度

# 遍历数据加载器中的批次
for data in dataLoader:
    # 解包数据：获取图像和标签
    imgs, targets = data

    # 记录输入图像到TensorBoard
    # 输入图像形状: [64, 3, 32, 32] - 64张32x32的RGB图像
    writer.add_images('input', imgs, step)

    # 将图像输入网络进行最大池化操作
    output = net(imgs)

    # 记录池化后的输出图像到TensorBoard
    # 输出图像形状: [64, 3, 10, 10] - 64张10x10的RGB图像
    # 计算过程: (32 - 3)/3 + 1 = 10 (向下取整)
    writer.add_images('output', output, step)

    step += 1  # 增加步数计数器

# 关闭TensorBoard写入器
writer.close()

# 运行后在终端执行以下命令查看结果:
# tensorboard --logdir=TensorBoard