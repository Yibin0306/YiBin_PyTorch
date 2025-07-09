"""
dir()函数: 打开，看见，知道工具箱内有什么东西
help()函数: 说明书，知道工具使用的方法
"""
import torch
print(torch.cuda.is_available())
print(dir(torch))
print(dir(torch.cuda))
print(dir(torch.cuda.is_available()))
print(help(torch.cuda.is_available))