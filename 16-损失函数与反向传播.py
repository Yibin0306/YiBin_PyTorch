# 文档说明：
"""
Loss（损失函数）
1、计算实际输出和目标之间的差距
2、为我们更新输出提供一定的依据（反向传播）
"""

import torch
from torch import nn
from torch.nn import L1Loss, MSELoss  # 导入L1损失和均方误差损失

# 创建输入张量和目标张量（使用相同的数值类型float32）
input = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)

# 将张量重塑为4维格式 (batch_size, channels, height, width)
# 本例中实际维度为(1, 1, 1, 3)表示：1个样本，1个通道，1行，3列
input = torch.reshape(input, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# ===================== L1损失计算 =====================
# 创建L1损失函数实例，设置reduction='sum'表示直接求和损失值（默认为'mean'求平均）
loss = L1Loss(reduction='sum')
# 计算输入和目标之间的L1损失：|1-1| + |2-2| + |3-5| = 0+0+2 = 2.0
result = loss(input, targets)

# ===================== 均方误差损失计算 =====================
loss_mse = MSELoss()  # 创建MSE损失函数实例（默认使用求平均）
# 计算MSE损失：[(1-1)^2 + (2-2)^2 + (3-5)^2]/3 = (0+0+4)/3 ≈ 1.333
result_mse = loss_mse(input, targets)

# 打印两种损失结果
print(result)      # 输出：tensor(2.)
print(result_mse)  # 输出：tensor(1.3333)

# ===================== 交叉熵损失计算 =====================
# 创建3分类的输入数据（未归一化的logits）
x = torch.tensor([0.1, 0.2, 0.3])
# 目标标签：表示类别索引1（第二个类别）
y = torch.tensor([1])
# 重塑为2D张量：(batch_size, num_classes) -> (1, 3)
x = torch.reshape(x, (1, 3))
# 创建交叉熵损失函数
loss_cross = nn.CrossEntropyLoss()
# 计算交叉熵损失：先对x进行softmax([0.1,0.2,0.3]->[0.30,0.33,0.37])，再取目标类别1的概率取负对数
result_cross = loss_cross(x, y)
print(result_cross)  # 输出：tensor(1.1019)