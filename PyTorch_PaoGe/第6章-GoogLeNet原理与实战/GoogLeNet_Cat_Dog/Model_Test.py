import torch  # 导入PyTorch深度学习框架
import torch.utils.data as Data  # 导入PyTorch数据加载和处理模块
from torchvision import transforms  # 导入图像预处理变换模块
from torch import nn  # 导入神经网络模块
from torchvision.datasets import ImageFolder
from PIL import Image
from Model import GoogLeNet, Inception  # 从自定义的Model模块中导入GoogLeNet, Inception模型定义


def test_data_process():
    # 设置训练数据的根目录路径
    # 使用原始字符串(r前缀)避免反斜杠转义问题
    Root_Test = r"data\test"

    # 定义图像归一化操作
    # 使用数据集的均值和标准差进行标准化，有助于模型训练收敛
    # 这里的均值[0.16207035, 0.15101879, 0.1384724]和标准差[0.05801719, 0.05213359, 0.0477778]通常是预先计算得到的
    normalize = transforms.Normalize(mean=[0.16207035, 0.15101879, 0.1384724],
                                     std=[0.05801719, 0.05213359, 0.0477778])

    # 定义训练数据的图像变换管道（数据预处理流程）
    test_transform = transforms.Compose([
        # 1. 调整图像尺寸为224x224像素（符合GoogLeNet等经典CNN网络的输入要求）
        transforms.Resize((224, 224)),

        # 2. 将PIL图像或numpy数组转换为PyTorch张量，并自动将像素值从[0,255]缩放到[0,1]范围
        transforms.ToTensor(),

        # 3. 应用标准化处理：对每个通道进行 (input - mean) / std 的归一化
        normalize
    ])

    # 创建训练数据集实例
    # ImageFolder会自动根据目录结构加载数据，要求目录结构为：
    # data/train/
    #   ├── class1/
    #   │   ├── image1.jpg
    #   │   └── image2.jpg
    #   └── class2/
    #       ├── image1.jpg
    #       └── image2.jpg
    test_data = ImageFolder(root=Root_Test, transform=test_transform)

    # 创建测试数据加载器
    test_dataloader = Data.DataLoader(
        dataset=test_data,  # 指定要加载的数据集
        batch_size=1,  # 批次大小为1，即每次处理1张图像（可以改为更大的批次以提高效率）
        shuffle=True,  # 打乱数据顺序，确保模型评估的随机性
        num_workers=0  # 数据加载的进程数（0表示使用主进程加载）
    )

    # 返回创建好的测试数据加载器
    return test_dataloader


def test_model_process(model, test_dataloader):
    """
    模型测试函数：使用测试集评估模型性能

    参数：
    model - 已经训练好的模型实例
    test_dataloader - 测试数据加载器

    功能：
    1. 将模型设置为评估模式
    2. 在测试集上进行前向传播（不计算梯度）
    3. 统计预测正确的样本数量
    4. 计算并输出测试准确率
    """

    # 设置计算设备（优先使用GPU，如果不可用则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到指定的计算设备（GPU或CPU）
    model = model.to(device)

    # 初始化测试统计变量
    test_corrects = 0.0  # 记录预测正确的样本数量
    test_num = 0  # 记录测试样本总数

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    # 在测试阶段不需要计算梯度，可以节省内存和计算资源
    with torch.no_grad():
        # 遍历测试数据加载器中的所有批次
        for test_data_x, test_data_y in test_dataloader:
            # 将测试数据移动到指定的计算设备
            test_data_x = test_data_x.to(device)  # 输入图像数据
            test_data_y = test_data_y.to(device)  # 对应的真实标签

            # 将模型设置为评估模式（影响Dropout和BatchNorm等层的行为）
            # 在评估模式下，这些层会使用训练阶段学到的统计信息，而不是重新计算
            model.eval()

            # 前向传播：将测试数据输入模型，得到输出预测
            output = model(test_data_x)

            # 获取预测结果：选择输出中概率最大的类别作为预测标签
            # dim=1表示在类别维度（第1维）上取最大值索引
            pre_lab = torch.argmax(output, dim=1)

            # 统计预测正确的样本数量
            # pre_lab == test_data_y.data：比较预测标签和真实标签，返回布尔张量
            # torch.sum()：计算True的数量（即预测正确的样本数）
            test_corrects += torch.sum(pre_lab == test_data_y.data)

            # 更新已处理的测试样本总数
            test_num += test_data_x.size(0)  # size(0)返回当前批次的样本数量

    # 计算测试准确率：正确预测数 / 总样本数
    # test_corrects.double()：转换为双精度浮点数以确保计算精度
    # .item()：将单元素张量转换为Python数值
    test_acc = test_corrects.double().item() / test_num

    # 打印测试准确率结果
    print("测试的准确率为：", test_acc)
    print()  # 打印空行以便阅读


# 主程序入口：当直接运行此脚本时执行以下代码
if __name__ == '__main__':
    # 创建LeNet模型实例
    model = GoogLeNet(Inception)

    # 加载预训练的最佳模型权重
    # torch.load()：从文件中加载保存的模型状态字典
    # 'Model_Save/best_model.pth'：模型权重文件的路径
    model.load_state_dict(torch.load('Model_Save/best_model.pth'))

    # 处理测试数据，获取测试数据加载器
    # test_dataloader = test_data_process()

    # 使用测试集评估模型性能
    # test_model_process(model, test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    classes = ['猫', '狗']
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测值：",classes[result],"------","真实值：",classes[label])

    image = Image.open('data_test/5.jpg')
    normalize = transforms.Normalize(mean=[0.16207035, 0.15101879, 0.1384724],
                                     std=[0.05801719, 0.05213359, 0.0477778])
    test_transform = transforms.Compose([
        # 1. 调整图像尺寸为224x224像素（符合GoogLeNet等经典CNN网络的输入要求）
        transforms.Resize((224, 224)),

        # 2. 将PIL图像或numpy数组转换为PyTorch张量，并自动将像素值从[0,255]缩放到[0,1]范围
        transforms.ToTensor(),

        # 3. 应用标准化处理：对每个通道进行 (input - mean) / std 的归一化
        normalize
    ])
    image = test_transform(image)

    # 添加批次维度
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        # 获取预测结果：选择输出中概率最大的类别作为预测标签
        # dim=1表示在类别维度（第1维）上取最大值索引
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()

    print("预测值：", classes[result])

