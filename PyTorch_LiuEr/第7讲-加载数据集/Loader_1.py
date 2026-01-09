"""
糖尿病预测模型：数据集加载与训练优化

本代码展示了使用PyTorch进行糖尿病预测的完整流程，包括：
1. 自定义数据集类实现
2. 数据加载器使用(批量加载、乱序、并行)
3. 三层神经网络模型设计
4. 训练循环优化(小批量训练)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ===== 自定义数据集类 =====
class DiabetesDataset(Dataset):
    """
    糖尿病数据集加载器

    继承自torch.utils.Data.Dataset，实现以下方法:
    1. __init__: 初始化数据集
    2. __getitem__: 获取单个样本
    3. __len__: 获取数据集大小
    """

    def __init__(self, filepath):
        """
        初始化数据集
        参数:
            filepath: 数据文件路径
        """
        # 从CSV文件加载数据
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # 记录数据集大小
        self.len = xy.shape[0]  # 样本数量
        # 创建特征张量 (所有行, 前8列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        # 创建标签张量 (所有行, 最后一列，保持二维形式)
        self.y_data = torch.from_numpy(xy[:, [-1]])
        """
        数据格式说明:
          特征: 8个医学指标
          标签: 1 (0表示未患病, 1表示患病)
        """

    def __getitem__(self, index):
        """
        获取单个样本
        参数:
            index: 样本索引
        返回:
            (特征, 标签) 元组
        """
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """
        获取数据集大小
        返回:
            样本数量
        """
        return self.len


# ===== 数据准备 =====
# 创建数据集实例
dataset = DiabetesDataset("../数据/diabetes.csv.gz")
"""
数据集特性:
  样本数: 768
  特征数: 8 (怀孕次数、血糖、血压等)
  标签: 二分类 (0或1)
"""

# 创建数据加载器
train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,  # 每批32个样本
    shuffle=True,  # 每epoch打乱数据顺序
    num_workers=2  # 使用2个进程加载数据
)
"""
DataLoader参数说明:
  batch_size=32: 小批量大小
  shuffle=True:  每个epoch重新洗牌数据
  num_workers=2: 并行数据加载的进程数
作用:
  1. 自动分批
  2. 支持乱序
  3. 并行数据加载加速
"""


# ===== 神经网络模型 =====
class Model(torch.nn.Module):
    """
    三层全连接神经网络

    网络结构:
      输入层: 8个神经元
      隐藏层1: 6个神经元
      隐藏层2: 4个神经元
      输出层: 1个神经元 (概率输出)
    """

    def __init__(self):
        super(Model, self).__init__()
        # 第一层: 8输入 -> 6输出
        self.linear1 = torch.nn.Linear(8, 6)
        # 第二层: 6输入 -> 4输出
        self.linear2 = torch.nn.Linear(6, 4)
        # 第三层: 4输入 -> 1输出
        self.linear3 = torch.nn.Linear(4, 1)
        # Sigmoid激活函数 (概率转换)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入特征
        返回:
            糖尿病患病概率
        """
        # 三层线性变换 + Sigmoid激活
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 创建模型实例
model = Model()

# ===== 损失函数与优化器 =====
# 二元交叉熵损失函数
criterion = torch.nn.BCELoss(reduction='mean')
"""
参数说明:
  reduction='mean': 返回批量的平均损失
  其他选项: 'sum'(损失总和), 'none'(每个样本损失)
"""

# 随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
"""
参数说明:
  model.parameters(): 获取所有可训练参数
  lr=0.01: 学习率(常用值)
"""

if __name__ == '__main__':
    # ===== 训练循环 =====
    for epoch in range(100):  # 100个训练周期
        # 使用enumerate获取批次索引和数据
        for i, data in enumerate(train_loader, 0):
            # 解包数据: 特征和标签
            inputs, labels = data

            # 前向传播: 计算预测值
            y_pred = model(inputs)

            # 计算损失: 批量平均损失
            loss = criterion(y_pred, labels)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播: 计算梯度
            loss.backward()

            # 参数更新
            optimizer.step()

            # 可选: 打印每100个batch的损失
            if i % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}')

"""
糖尿病预测模型训练详解:
=========================================================
1. 数据集加载优化:
   - 使用Dataset类封装数据加载逻辑
   - 支持__len__和__getitem__方法
   - 便于DataLoader批量处理

2. DataLoader优势:
   - 自动分批: 处理大批量数据
   - 乱序功能: 每个epoch重新洗牌，提高泛化能力
   - 并行加载: num_workers加速数据加载
   - 内存高效: 仅加载需要的批次数据

3. 训练流程:
   A. 外层循环: 训练周期(epoch)
   B. 内层循环: 遍历数据批次(batch)
       1. 获取批次数据
       2. 前向传播计算预测值
       3. 计算批量损失
       4. 梯度清零
       5. 反向传播计算梯度
       6. 优化器更新参数
=========================================================
"""