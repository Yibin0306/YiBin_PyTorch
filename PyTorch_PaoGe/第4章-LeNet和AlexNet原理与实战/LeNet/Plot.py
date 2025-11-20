# 导入必要的库
from torchvision.datasets import FashionMNIST  # FashionMNIST数据集
from torchvision import transforms  # 图像预处理变换
import torch.utils.data as Data  # 数据加载工具
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库

# 加载FashionMNIST训练数据集
train_data = FashionMNIST(root='./Data',  # 数据集保存路径
                          train=True,     # 加载训练集（True=训练集，False=测试集）
                          download=True,  # 如果本地不存在则下载数据集
                          # 定义图像预处理流程
                          transform=transforms.Compose([
                              transforms.Resize(224),  # 将图像大小调整为224x224（通常用于适配预训练模型）
                              transforms.ToTensor()    # 将PIL图像或numpy数组转换为张量，并归一化到[0,1]
                          ]))

# 创建数据加载器，用于批量加载数据
train_loader = Data.DataLoader(dataset=train_data,  # 要加载的数据集
                               batch_size=64,       # 每个批次的样本数量
                               shuffle=True,        # 每个epoch打乱数据顺序
                               num_workers=0)       # 使用0个进程加载数据（0=主进程加载）

# 从数据加载器中获取一个批次的数据进行演示
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:  # 只取第一个批次的数据
        break

# 处理批次数据
batch_x = b_x.squeeze().numpy()  # 移除张量中维度为1的维度，并转换为numpy数组
                                 # squeeze()将形状从[64,1,224,224]变为[64,224,224]
batch_y = b_y.numpy()            # 将标签张量转换为numpy数组

class_label = train_data.classes  # 获取数据集的类别标签名称
print("类别标签:", class_label)
print("训练数据批次形状:", batch_x.shape)  # 输出批次数据的维度（64个样本，每个224x224像素）

# 可视化一个批次中的图像
plt.figure(figsize=(12, 5))  # 创建图形窗口，设置尺寸
for ii in np.arange(len(batch_y)):  # 遍历批次中的每个样本
    plt.subplot(4, 16, ii + 1)     # 创建4行16列的子图，当前绘制第ii+1个
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)  # 显示灰度图像
    plt.title(class_label[batch_y[ii]], size=10)     # 设置标题为对应的类别名称
    plt.axis("off")  # 关闭坐标轴显示
    plt.subplots_adjust(wspace=0.05)  # 调整子图之间的水平间距

plt.show()  # 显示图形