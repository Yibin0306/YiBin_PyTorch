# 导入必要的PyTorch库
import torch
import torchvision.datasets  # PyTorch的计算机视觉数据集模块
from torch import nn  # 神经网络模块
from torch.nn import ReLU  # ReLU激活函数
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化工具

# 创建一个手动输入的2x2张量用于演示
input = torch.tensor([
    [1, -0.5],
    [-1, 3]
], dtype=torch.float32)  # 指定为float32类型

# 重塑张量维度为神经网络标准输入格式
# 形状: [batch_size, channels, height, width]
# 这里: 1个样本, 1个通道, 2x2图像
input = torch.reshape(input, (-1, 1, 2, 2))

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

        # 创建激活函数
        self.relu = ReLU()  # ReLU激活函数 (未在forward中使用)
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, input):
        # 前向传播：应用Sigmoid激活函数
        # Sigmoid将输入压缩到(0,1)范围
        output = self.sigmoid(input)
        return output


# 实例化神经网络
net = Net()

# 创建TensorBoard日志记录器
writer = SummaryWriter("TensorBoard")  # 日志保存在"TensorBoard"目录
step = 0  # 初始化步数计数器，用于跟踪训练进度

# 遍历数据加载器中的批次
for data in dataLoader:
    # 解包数据：获取图像和标签
    imgs, targets = data

    # 记录输入图像到TensorBoard
    # 输入图像形状: [64, 3, 32, 32] - 64张32x32的RGB图像
    writer.add_images('input', imgs, global_step=step)

    # 将图像输入网络进行激活函数处理
    output = net(imgs)

    # 记录Sigmoid处理后的输出图像到TensorBoard
    # 输出图像形状: [64, 3, 32, 32] - 尺寸不变，但值域在(0,1)之间
    writer.add_images('output', output, step)

    step += 1  # 增加步数计数器

# 关闭TensorBoard写入器
writer.close()

# 运行后在终端执行以下命令查看结果:
# tensorboard --logdir=TensorBoard