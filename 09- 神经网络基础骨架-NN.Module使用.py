"""
NN(Neural Network)神经网络
input -> forward(前向传播) -> output

def forward(self, x):
    x = F.relu(self.conv1(x))
    return F.relu(self.conv2(x))
输入x -> 卷积 -> 非线性 -> 卷积 -> 非线性 -> 输出
"""
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        output = input + 1
        return output

Net = Net()
x = torch.tensor(1.0)
output = Net(x)
print(output)