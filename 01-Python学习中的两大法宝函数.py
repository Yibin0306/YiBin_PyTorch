"""
dir()函数: 打开，看见，知道工具箱内有什么东西
help()函数: 说明书，知道工具使用的方法
"""
# 导入PyTorch深度学习库
import torch

# 检查当前环境CUDA（GPU加速）是否可用：
# - 返回True表示已安装GPU驱动且PyTorch支持CUDA
# - 返回False表示仅能使用CPU运行
print(torch.cuda.is_available())

# 查看torch顶级模块的所有属性与方法列表：
# - 包含所有核心功能：神经网络、张量操作、优化器等
# - 相当于PyTorch的全局功能目录
print(dir(torch))

# 查看torch.cuda子模块的所有属性与方法：
# - CUDA相关操作：设备管理、内存分配、流控制等
# - 仅当CUDA可用时部分功能才有效
print(dir(torch.cuda))

# - 此函数返回的是bool值(bool类型只有__doc__等基础属性)
# - 打印结果通常为Python基本类型的默认属性列表
print(dir(torch.cuda.is_available()))

# 打印torch.cuda.is_available函数的帮助文档：
# - 显示函数用途、返回值说明和使用示例
# - help()是Python内置的交互式帮助工具
print(help(torch.cuda.is_available))