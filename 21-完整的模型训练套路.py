import torchvision
from torch import nn
from torch.utils.data import DataLoader
from Model.CIFAR10_Data_Model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./CIFAR10_DATA',
                                       train=True,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root='./CIFAR10_DATA',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10，训练数据集的长度为：10
print(f'训练数据集的长度为：{train_data_size}\n测试数据集的长度为：{test_data_size}')

# 利用DataLoader来加载数据
train_DataLoader = DataLoader(train_data, batch_size=64)
test_DataLoader = DataLoader(test_data, batch_size=64)

# 创建网络模型
net = Net()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

for i in range(epoch):
    print(f"---------第 {i+1} 轮训练开始----------")

    # 训练步骤开始
    for data in train_DataLoader:
        img, targets = data
        output = net(img)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        print(f"训练次数：{total_train_step}，Loss：{loss.item()}")

    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_DataLoader:
            img, targets = data
            output = net(img)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
    print(f"整体测试集上的Loss：{total_test_loss}")