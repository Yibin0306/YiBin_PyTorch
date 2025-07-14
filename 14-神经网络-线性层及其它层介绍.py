# 导入必要的PyTorch库
import torch
import torchvision.datasets  # 提供常用数据集
from torch import nn  # 神经网络模块
from torch.nn import Linear  # 全连接层
from torch.utils.data import DataLoader  # 数据加载器

# 下载并加载CIFAR10测试数据集
dataset = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=False,  # 使用测试集（非训练集）
    download=True,  # 如果本地不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将PIL图像转换为Tensor格式
)

# 创建数据加载器，每次迭代返回64张图像
dataloader = DataLoader(dataset, batch_size=64)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义单个全连接层：输入维度196608，输出维度10
        self.linear1 = Linear(196608, 10)  # 196608 = 3 * 32 * 32 * 64（通道*高*宽*批量）

    def forward(self, input):
        output = self.linear1(input)  # 前向传播：通过全连接层
        return output


# 实例化网络
net = Net()

# 遍历数据集
for data in dataloader:
    imgs, targets = data  # 解包数据（图像张量和标签）

    # 将当前批次的所有图像展平成1维向量
    # imgs形状原为[64, 3, 32, 32] -> 展平后变为[64 * 3 * 32 * 32] = [196608]
    output = torch.flatten(imgs)

    # 通过神经网络
    output = net(output)  # 输出形状变为[10]

    # 打印网络输出形状（应为[10]）
    print(output.shape)