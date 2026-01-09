"""
PyTorch线性回归完整实现：使用神经网络模块、损失函数和优化器

本代码展示了使用PyTorch高级API实现线性回归模型的完整流程，包括：
1. 数据准备与张量化
2. 自定义神经网络模块
3. 内置损失函数
4. 优化器使用
5. 训练循环
6. 模型测试
"""

import torch

# ===== 数据准备 =====
# 创建特征张量 (3x1矩阵)
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# 创建标签张量 (3x1矩阵)
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# ===== 模型定义 =====
class LinearModule(torch.nn.Module):
    """
    自定义线性回归模块

    继承自torch.nn.Module，包含：
    - 初始化函数(__init__)：定义网络层
    - 前向传播函数(forward)：定义计算流程
    """

    def __init__(self):
        """
        初始化函数：定义网络层
        """
        super(LinearModule, self).__init__()  # 调用父类构造函数
        # 创建全连接层: 输入特征数=1, 输出特征数=1 (线性回归)
        self.linear = torch.nn.Linear(1, 1)
        """
        nn.Linear详解:
          参数:
            in_features: 输入特征维度 (1)
            out_features: 输出特征维度 (1)
          自动初始化:
            weight: 权重参数 (1x1矩阵)
            bias: 偏置参数 (标量)
        """

    def forward(self, x):
        """
        前向传播函数：定义计算流程
        参数:
            x: 输入数据张量
        返回:
            y_pred: 预测值
        """
        y_pred = self.linear(x)  # 执行线性变换: y = wx + b
        return y_pred

# 创建模型实例
model = LinearModule()
"""
模型结构:
  model.linear.weight: 权重参数 (初始随机值)
  model.linear.bias: 偏置参数 (初始随机值)
"""

# ===== 损失函数 =====
# 创建均方误差损失函数
criterion = torch.nn.MSELoss(size_average=False)
"""
参数说明:
  size_average=False: 返回所有样本的损失总和
  若为True: 返回损失的平均值
  公式: loss = Σ(y_pred_i - y_true_i)^2
"""

# ===== 优化器 =====
# 创建随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
"""
参数说明:
  model.parameters(): 获取模型所有可训练参数(权重和偏置)
  lr=0.01: 学习率(控制更新步长)
  功能: 实现参数更新规则: param = param - lr * param.grad
"""

# ===== 训练循环 =====
for epoch in range(1000):  # 1000个训练周期
    # 前向传播: 计算预测值
    y_pred = model(x_data)

    # 计算损失: 比较预测值和真实值
    loss = criterion(y_pred, y_data)

    # 打印训练进度(每100个epoch打印一次)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    # 梯度清零: 清空上一次计算的梯度值
    optimizer.zero_grad()

    # 反向传播: 计算损失关于参数的梯度
    loss.backward()

    # 参数更新: 使用优化器更新权重和偏置
    optimizer.step()

# ===== 结果输出 =====
# 打印训练后的参数
print('训练后权重 w =', model.linear.weight.item())  # 获取标量值
print('训练后偏置 b =', model.linear.bias.item())

# 使用训练好的模型进行预测
x_test = torch.tensor([[4.0]])  # 创建测试数据(形状需与训练数据一致)
y_test = model(x_test)  # 预测
print('预测值 y_pred =', y_test.data.item())

"""
PyTorch模型训练流程详解:
=========================================================
1. 数据准备:
   - 将数据转换为PyTorch张量
   - 注意保持特征和标签的形状匹配
   - 本例中: 输入形状为(3,1)，输出形状为(3,1)

2. 模型定义:
   - 继承nn.Module类
   - __init__中定义网络层(nn.Linear)
   - forward方法定义数据流向
   - 自动管理参数: weight和bias

3. 损失函数:
   - MSELoss: 均方误差损失
   - size_average选项控制损失计算方式
   - 其他可选损失函数: L1Loss, CrossEntropyLoss等

4. 优化器:
   - SGD: 随机梯度下降
   - 参数: 模型参数和学习率
   - 其他优化器: Adam, RMSprop, Adagrad等

5. 训练循环:
   A. 前向传播:
        y_pred = model(x_data)
        - 自动调用forward方法
        - 计算预测值

   B. 损失计算:
        loss = criterion(y_pred, y_data)
        - 衡量预测值与真实值的差异

   C. 梯度清零:
        optimizer.zero_grad()
        - 清除上一次迭代的梯度
        - 防止梯度累积

   D. 反向传播:
        loss.backward()
        - 自动计算所有参数的梯度
        - 梯度存储在参数的.grad属性中

   E. 参数更新:
        optimizer.step()
        - 使用梯度更新参数
        - 更新规则: param = param - lr * param.grad

6. 参数访问:
   - model.linear.weight: 权重张量
   - .item(): 获取单元素张量的Python值

7. 模型预测:
   - 使用训练好的模型处理新数据
   - 注意输入张量形状需与训练数据一致
   - .Data: 获取张量数据(不含梯度信息)

8. 训练过程分析:
   - 初始损失较大(随机参数)
   - 随着训练进行，损失逐渐降低
   - 最终参数接近理想值(w=2.0, b≈0.0)
=========================================================
"""