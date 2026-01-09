"""
PyTorch自动微分实现：线性回归模型训练

本代码展示了如何使用PyTorch的自动微分功能实现线性回归模型的训练过程。
重点演示了：
1. Tensor的基本操作
2. requires_grad属性与自动微分
3. 前向传播和损失计算
4. 反向传播与梯度计算
5. 手动参数更新与梯度清零
"""

import torch

# ===== 训练数据 =====
# 输入特征值 (x)
x_data = [1.0, 2.0, 3.0]
# 对应标签值 (y)
y_data = [2.0, 4.0, 6.0]  # 数据关系为 y = 2x，理想权重应为2.0

# ===== 模型参数 =====
# 初始化权重张量，初始值1.0
w = torch.tensor([1.0])
# 启用自动梯度追踪：允许PyTorch计算关于w的导数
w.requires_grad = True


# ===== 模型函数 =====
def forward(x):
    """
    前向传播函数：计算预测值
    参数:
        x: 输入特征值（Python标量）
    返回:
        w * x: 预测值（Tensor）
    """
    return w * x


def loss(x, y):
    """
    计算单个样本的平方误差损失
    参数:
        x: 输入特征值（Python标量）
        y: 真实标签值（Python标量）
    返回:
        损失值（Tensor）
    """
    y_pred = forward(x)  # 计算预测值（Tensor）
    return (y_pred - y) ** 2  # 返回平方误差（Tensor）


# ===== 训练前预测 =====
# 使用初始权重预测x=4时的输出
print("predict (before training)", 4, forward(4).item())  # .item()获取Python标量值

# ===== 训练循环 =====
for epoch in range(100):  # 100个训练周期
    # 遍历所有训练样本（随机梯度下降：逐个样本更新）
    for x, y in zip(x_data, y_data):
        # 步骤1: 前向传播计算损失
        l = loss(x, y)  # 计算当前样本的损失（Tensor）

        # 步骤2: 反向传播计算梯度
        l.backward()  # 自动计算关于w的梯度并存储在w.grad中

        # 步骤3: 打印梯度信息
        # w.grad.item()获取梯度的Python标量值
        print('\tgrad:', x, y, w.grad.item(), w.data)

        # 步骤4: 手动更新权重（梯度下降）
        # w.data获取权重张量的数据部分（不追踪梯度）
        w.data = w.data - 0.01 * w.grad.data

        # 步骤5: 清零梯度
        # 防止梯度累积到下一次迭代
        w.grad.data.zero_()

    # 每个epoch结束时打印损失
    print('epoch:', epoch, l.item())

# ===== 训练后预测 =====
print('predict (after training)', 4, forward(4).item())

"""
PyTorch自动微分机制详解:
=========================================================
1. Tensor与requires_grad:
   - w = torch.tensor([1.0], requires_grad=True)
   - 设置requires_grad=True告诉PyTorch需要计算该张量的梯度

2. 计算图:
   - 当进行前向计算时，PyTorch自动构建计算图
   - 示例:
        y_pred = w * x
        loss = (y_pred - y)**2
   - 反向传播时，PyTorch根据计算图自动计算梯度

3. 梯度计算:
   - l.backward() 自动计算所有requires_grad=True的张量的梯度
   - 梯度存储在张量的.grad属性中

4. 手动更新参数:
   - 使用w.data访问张量值（不追踪梯度）
   - 更新公式: w = w - η * ∂loss/∂w
   - 注意: 使用w.data避免在更新时构建计算图

5. 梯度清零:
   - w.grad.Data.zero_() 清零梯度
   - 必须执行: PyTorch会累积梯度，不清零会导致错误更新

6. 获取Python值:
   - .item()方法: 将单元素张量转换为Python标量
   - .data属性: 获取不追踪梯度的张量值

7. 重要概念区分:
   - w.Data: 权重值本身（不带梯度信息）
   - w.grad: 损失函数关于w的梯度
   - w.grad.Data: 梯度张量的数据部分

8. 训练过程分析:
   - 每个epoch遍历所有样本(3个)
   - 每个样本执行完整的前向-反向-更新流程
   - 损失值逐渐减小，权重w逐渐接近2.0

9. 为什么使用PyTorch自动微分:
   - 避免手动推导和实现梯度公式
   - 支持复杂模型（多层网络、卷积等）
   - 自动处理计算图构建和梯度计算
=========================================================
"""