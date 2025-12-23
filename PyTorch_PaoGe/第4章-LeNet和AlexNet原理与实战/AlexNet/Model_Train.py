import copy  # 深度拷贝模块，用于创建对象的完全独立副本
import time  # 时间模块，用于计时和测量训练时间
import pandas as pd  # 数据处理库，用于创建DataFrame和数据分析
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 数据加载器，用于批量加载数据
from torchvision.datasets import FashionMNIST  # FashionMNIST数据集
from torchvision import transforms  # 图像预处理变换
import torch.utils.data as Data  # PyTorch数据工具
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from Model import AlexNet  # 从Model模块导入AlexNet模型定义


def train_val_data_process():
    """
    数据处理函数：加载FashionMNIST数据集并划分为训练集和验证集
    返回: train_dataloader, val_dataloader - 训练和验证数据加载器
    """
    # 加载FashionMNIST训练数据集
    dataset = FashionMNIST(root='./Data',  # 数据集保存路径
                           train=True,  # 加载训练集（True=训练集，False=测试集）
                           download=True,  # 如果本地不存在则自动下载数据集
                           # 定义图像预处理流程
                           transform=transforms.Compose([
                               transforms.Resize(size=227),  # 将图像大小调整为227x227（原始FashionMNIST是28x28）
                               transforms.ToTensor()  # 将PIL图像或numpy数组转换为张量，并归一化到[0,1]
                           ]))

    # 将数据集划分为训练集和验证集（80%训练，20%验证）
    # random_split函数随机划分数据集，返回两个子集
    train_data, val_data = Data.random_split(dataset,
                                             [round(len(dataset) * 0.8),  # 训练集大小（80%，四舍五入取整）
                                              round(len(dataset) * 0.2)])  # 验证集大小（20%，四舍五入取整）

    # 创建训练数据加载器
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=128,  # 每个批次的样本数量
                                  shuffle=True,  # 每个epoch打乱数据顺序，防止模型记忆顺序
                                  num_workers=0)

    # 创建验证数据加载器
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=128,  # 验证批次大小与训练一致
                                shuffle=True,  # 验证集也可以打乱，但不是必须的
                                num_workers=0)

    # 返回创建好的数据加载器
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    模型训练和验证函数
    参数:
        model: 要训练的神经网络模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        num_epochs: 训练的总轮数
    返回:
        train_process: 包含训练过程的DataFrame，记录每个epoch的损失和准确率
    """
    # 设置设备（优先使用GPU，如果没有则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义优化器（Adam优化器，学习率0.001）
    # Adam优化器结合了Momentum和RMSProp的优点，通常收敛较快
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数（交叉熵损失，适用于多分类问题）
    # 交叉熵损失直接比较预测概率分布和真实标签分布
    criterion = torch.nn.CrossEntropyLoss()

    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(device)

    # 初始化最佳模型权重（深拷贝当前模型权重）
    # 使用深拷贝确保保存的权重副本完全独立，不受后续训练影响
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化最佳准确率为0.0
    best_acc = 0.0

    # 初始化列表用于记录训练过程中的损失和准确率
    train_loss_all = []  # 记录每个epoch的训练损失
    train_acc_all = []  # 记录每个epoch的训练准确率
    val_loss_all = []  # 记录每个epoch的验证损失
    val_acc_all = []  # 记录每个epoch的验证准确率

    # 记录训练开始时间，用于计算总训练时间
    since = time.time()

    # 开始训练循环，遍历每个epoch
    for epoch in range(num_epochs):
        # 打印当前epoch信息
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)  # 打印分隔线

        # 初始化每个epoch的统计变量
        train_loss = 0.0  # 累计训练损失
        val_loss = 0.0  # 累计验证损失
        train_acc = 0.0  # 累计训练正确预测数
        val_acc = 0.0  # 累计验证正确预测数
        train_num = 0  # 累计训练样本总数
        val_num = 0  # 累计验证样本总数

        # ========== 训练阶段 ==========
        # 遍历训练数据加载器中的所有批次
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将当前批次的数据和标签移动到指定设备
            b_x = b_x.to(device)  # 输入数据
            b_y = b_y.to(device)  # 真实标签

            # 设置模型为训练模式（启用dropout和batch normalization）
            model.train()

            # 前向传播：将输入数据传入模型，得到输出
            output = model(b_x)

            # 获取预测标签：选择输出中概率最大的类别作为预测结果
            pre_lab = torch.argmax(output, dim=1)

            # 计算损失：比较模型输出和真实标签
            # crossEntropyLoss函数内置了softmax函数，所以只需要最后输出值和标签值就可以了。概率会自己算出来
            loss = criterion(output, b_y)

            # 反向传播前的准备：清空优化器中的梯度
            optimizer.zero_grad()

            # 反向传播：计算损失关于模型参数的梯度
            loss.backward()

            # 参数更新：根据梯度更新模型参数
            optimizer.step()

            # 累计统计信息
            # 损失乘以批次大小，因为损失是批次内样本的平均损失
            train_loss += loss.item() * b_x.size(0)
            # 累计正确预测的数量
            train_acc += torch.sum(pre_lab == b_y)
            # 累计已处理的样本数量
            train_num += b_x.size(0)

        # ========== 验证阶段 ==========
        # 遍历验证数据加载器中的所有批次
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将验证数据移动到指定设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为评估模式（禁用dropout和batch normalization）
            model.eval()

            # 前向传播（验证阶段不需要计算梯度）
            output = model(b_x)

            # 获取预测标签
            pre_lab = torch.argmax(output, dim=1)

            # 计算验证损失
            loss = criterion(output, b_y)

            # 累计验证统计信息
            val_loss += loss.item() * b_x.size(0)
            val_acc += torch.sum(pre_lab == b_y)
            val_num += b_x.size(0)

        # 计算当前epoch的平均训练损失并添加到列表
        train_loss_all.append(train_loss / train_num)
        # 计算当前epoch的平均验证损失并添加到列表
        val_loss_all.append(val_loss / val_num)

        # 计算当前epoch的训练准确率并添加到列表
        # double()确保使用双精度计算，item()将张量转换为Python数值
        train_acc_all.append(train_acc.double().item() / train_num)
        # 计算当前epoch的验证准确率并添加到列表
        val_acc_all.append(val_acc.double().item() / val_num)

        # 打印当前epoch的训练和验证结果
        print("第{}轮 train loss:{:.4f} train acc: {:.4f}".format(
            epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("第{}轮 val loss:{:.4f} val acc: {:.4f}".format(
            epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        # 检查当前验证准确率是否优于历史最佳
        if val_acc_all[-1] > best_acc:
            # 更新最佳准确率
            best_acc = val_acc_all[-1]
            # 深拷贝当前模型权重作为新的最佳模型
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算并打印当前已使用的训练时间
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(
            time_use // 60, time_use % 60))  # 格式化为分钟和秒
        print()

    # 保存最佳模型到文件
    torch.save(best_model_wts, 'Model_Save/best_model.pth')

    # 创建DataFrame记录整个训练过程
    train_process = pd.DataFrame(data={
        'epoch': range(num_epochs),  # 训练轮次
        'train_loss_all': train_loss_all,  # 训练损失
        'train_acc_all': train_acc_all,  # 训练准确率
        'val_loss_all': val_loss_all,  # 验证损失
        'val_acc_all': val_acc_all  # 验证准确率
    })

    # 返回训练过程记录
    return train_process


def matplot_acc_loss(train_process):
    """
    可视化训练过程中的损失和准确率变化
    参数:
        train_process: 包含训练过程的DataFrame
    """
    # 创建图形窗口，设置尺寸为12x4英寸
    plt.figure(figsize=(12, 4))

    # 第一个子图：损失变化曲线
    plt.subplot(1, 2, 1)  # 1行2列，第1个子图
    # 绘制训练损失曲线（红色圆点连线）
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    # 绘制验证损失曲线（蓝色方块连线）
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()  # 显示图例
    plt.xlabel("epoch")  # x轴标签
    plt.ylabel("Loss")  # y轴标签

    # 第二个子图：准确率变化曲线
    plt.subplot(1, 2, 2)  # 1行2列，第2个子图
    # 绘制训练准确率曲线（红色圆点连线）
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    # 绘制验证准确率曲线（蓝色方块连线）
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")  # x轴标签
    plt.ylabel("acc")  # y轴标签
    plt.legend()  # 显示图例

    # 显示图形
    plt.show()


if __name__ == '__main__':
    """
    主程序入口
    当直接运行此脚本时执行完整的训练流程
    """
    # 创建LeNet模型实例
    AlexNet = AlexNet()

    # 处理数据，获取训练和验证数据加载器
    train_dataloader, val_dataloader = train_val_data_process()

    # 训练模型，获取训练过程记录
    train_process = train_model_process(AlexNet, train_dataloader, val_dataloader, 5)

    # 可视化训练过程中的损失和准确率变化
    matplot_acc_loss(train_process)