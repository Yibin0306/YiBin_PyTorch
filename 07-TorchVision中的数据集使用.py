# 导入torchvision库，包含常用数据集、模型架构和图像变换方法
import torchvision
# 导入TensorBoard的SummaryWriter，用于可视化数据
from torch.utils.tensorboard import SummaryWriter

# 定义数据集预处理变换管道
dataset_transform = torchvision.transforms.Compose([
    # 将PIL图像或numpy数组转换为PyTorch张量
    torchvision.transforms.ToTensor(),
])

# 创建CIFAR10训练数据集
train_set = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=True,  # 加载训练集
    transform=dataset_transform,  # 应用定义的预处理变换
    download=True  # 如果本地不存在则下载数据集
)

# 创建CIFAR10测试数据集
test_set = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=False,  # 加载测试集
    transform=dataset_transform,  # 应用相同的预处理变换
    download=True  # 如果本地不存在则下载数据集
)

# 创建TensorBoard写入器，日志保存在"TensorBoard"目录
writer = SummaryWriter("TensorBoard")

# 循环处理测试集的前10个样本
for i in range(10):
    # 获取测试集中第i个样本
    # img: 经过ToTensor变换后的图像张量 (C×H×W)
    # label: 对应的类别标签 (0-9)
    img, label = test_set[i]

    # 将当前图像添加到TensorBoard
    # "test_set": 图像在TensorBoard中的标签
    # img: 要显示的图像张量
    # i: 步数/索引，用于区分不同图像
    writer.add_image("test_set", img, i)

# 关闭TensorBoard写入器，确保所有数据已写入
writer.close()