# 导入必要的库
import torchvision  # 计算机视觉相关数据集和模型
from torch import nn  # 神经网络模块
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.tensorboard import SummaryWriter  # 训练可视化工具

# 从自定义模块导入CIFAR10模型
from Model.CIFAR10_Data_Model import *  # 导入Net类定义

# 准备数据集
# 下载并加载CIFAR10训练集
train_data = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=True,  # 使用训练集
    download=True,  # 如果不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为张量
)

# 下载并加载CIFAR10测试集
test_data = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=False,  # 使用测试集
    download=True,  # 如果不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为张量
)

# 获取数据集大小
train_data_size = len(train_data)  # 训练集样本数量(50,000)
test_data_size = len(test_data)  # 测试集样本数量(10,000)
print(f'训练数据集的长度为：{train_data_size}\n测试数据集的长度为：{test_data_size}')

# 创建数据加载器(DataLoader)
train_DataLoader = DataLoader(train_data, batch_size=64)  # 训练数据加载器，批量大小64
test_DataLoader = DataLoader(test_data, batch_size=64)  # 测试数据加载器，批量大小64

# 创建神经网络模型实例
net = Net()  # 实例化自定义模型（来自CIFAR10_Data_Model）

# 定义损失函数（交叉熵损失，适用于多分类问题）
loss_fn = nn.CrossEntropyLoss()

# 设置优化器（随机梯度下降）
learning_rate = 0.01  # 学习率
optimizer = torch.optim.SGD(net.parameters(), learning_rate)  # 优化模型所有参数

# 设置训练参数
total_train_step = 0  # 记录总训练迭代次数
total_test_step = 0  # 记录总测试次数
epoch = 10  # 训练轮数

# 初始化TensorBoard记录器（保存到"TensorBoard"目录）
writer = SummaryWriter("TensorBoard")

# 开始训练循环
for i in range(epoch):
    print(f"---------第 {i + 1} 轮训练开始----------")

    # 训练步骤（遍历训练集所有批次）
    net.train()
    for data in train_DataLoader:
        img, targets = data  # 解包数据批次（图像和标签）
        output = net(img)  # 前向传播得到预测输出
        loss = loss_fn(output, targets)  # 计算损失值

        # 反向传播优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_train_step += 1  # 更新训练迭代计数

        # 每100次迭代记录一次训练损失
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}，Loss：{loss.item()}")
            writer.add_scalar('Train_Loss', loss.item(), total_train_step)  # 记录到TensorBoard

    # 测试步骤（在整个测试集上评估）
    net.eval()
    total_test_loss = 0  # 累计测试损失
    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for data in test_DataLoader:
            img, targets = data
            output = net(img)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()  # 累加批次损失

    print(f"整体测试集上的Loss：{total_test_loss}")
    writer.add_scalar('Test_Loss', total_test_loss, total_test_step)  # 记录测试损失
    total_test_step += 1  # 更新测试计数

    # 保存当前轮次的模型
    torch.save(net, f"Net_MODEL_{i}.pth")  # 保存整个模型
    print("模型已保存")

# 关闭TensorBoard记录器
writer.close()