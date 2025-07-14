# 导入必要的PyTorch库
import torch
import torchvision  # PyTorch的计算机视觉库
from torch import nn  # 神经网络模块
from torch.nn import Conv2d  # 卷积层
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化工具

# 加载CIFAR10数据集
dataset = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=False,  # 使用测试集（非训练集）
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为PyTorch张量格式
    download=True  # 如果本地没有数据集则自动下载
)

# 创建数据加载器
dataLoader = DataLoader(
    dataset,  # 加载的数据集
    batch_size=64  # 每批加载64张图像
)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 调用父类初始化方法

        # 创建卷积层
        self.conv1 = Conv2d(
            in_channels=3,  # 输入通道数（RGB图像为3通道）
            out_channels=6,  # 输出通道数（卷积核数量）
            kernel_size=3,  # 卷积核大小3x3
            stride=1,  # 卷积步长
            padding=0  # 不添加填充
        )

    def forward(self, x):
        # 前向传播：应用卷积操作
        x = self.conv1(x)
        return x


# 实例化神经网络
net = Net()

# 创建TensorBoard日志记录器
writer = SummaryWriter("TensorBoard")  # 日志保存在"TensorBoard"目录
step = 0  # 初始化步数计数器

# 遍历数据加载器中的批次
for data in dataLoader:
    # 解包数据：获取图像和标签
    imgs, targets = data

    # 将图像输入网络进行卷积操作
    output = net(imgs)

    # 记录输入图像到TensorBoard
    # 输入图像形状: [batch_size=64, channels=3, height=32, width=32]
    writer.add_images('input', imgs, step)

    # 处理输出图像以便可视化：
    # 卷积后输出形状: [64, 6, 30, 30] - 6通道无法直接显示
    # 重塑为3通道格式: [batch_size*2, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))

    # 记录输出图像到TensorBoard
    writer.add_images('output', output, step)

    step += 1  # 增加步数计数器

# 关闭TensorBoard写入器
writer.close()

# 运行后在终端执行以下命令查看结果:
# tensorboard --logdir=TensorBoard