"""
NN(Neural Network)神经网络
input -> forward(前向传播) -> output

def forward(self, x):
    x = F.relu(self.conv1(x))
    return F.relu(self.conv2(x))
输入x -> 卷积 -> 非线性 -> 卷积 -> 非线性 -> 输出
"""

# 导入PyTorch库，用于构建神经网络
import torch
# 从torch中导入nn模块，包含各种神经网络层和模型基类
from torch import nn

# 定义一个名为Net的神经网络类，继承自nn.Module
class Net(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的初始化方法
        super(Net, self).__init__()
        # 此处可以添加神经网络层定义
        # 使用直接数值操作演示前向传播

    def forward(self, input):
        # 定义前向传播过程
        # 输入: input - 任意张量
        # 操作: 对输入值加1 (简单的数值操作演示)
        output = input + 1
        return output  # 返回计算结果

# 实例化神经网络
net = Net()  # 创建Net类的实例

# 创建测试张量 (标量1.0)
x = torch.tensor(1.0)  # 浮点类型张量

# 执行前向传播
output = net(x)  # 将输入x传入网络，触发forward方法

# 输出结果
print(output)  # 应输出: tensor(2.)