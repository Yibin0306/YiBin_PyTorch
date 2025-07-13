"""
transform模块是PyTorch中的图像预处理工具箱
包含各种图像转换工具，用于将原始数据转换为神经网络可处理的格式

关键概念：
1. 创建具体工具：如 Tool = transforms.ToTensor()
2. Tensor数据类型：PyTorch特有的数据结构，封装了神经网络所需的理论基础参数
    - 自动支持梯度计算
    - 优化GPU加速
    - 提供丰富的张量操作
"""

# 导入必要的库
from PIL import Image  # Python图像处理库
from torch.utils.tensorboard import SummaryWriter  # 训练可视化工具
from torchvision import transforms  # PyTorch图像预处理模块

# 本示例演示两个核心问题：
# 1. transform工具的使用方法
# 2. 为什么神经网络需要Tensor数据类型

# 指定图像文件路径
img_path = "D:\\Resources\\PyTorch_Learn\\hymenoptera_data\\hymenoptera_data\\train\\ants_image\\0013035.jpg"

# 使用PIL打开图像文件
img = Image.open(img_path)

# 创建SummaryWriter实例，指定日志目录
# 用于后续在TensorBoard中可视化图像
writer = SummaryWriter("TensorBoard")

# 创建ToTensor转换工具实例
# 该工具将PIL图像或NumPy数组转换为PyTorch Tensor
tensor_trans = transforms.ToTensor()

# 应用转换：将PIL图像转换为Tensor
tensor_img = tensor_trans(img)

# 将转换后的Tensor图像添加到TensorBoard
# TensorBoard默认接受CHW格式的Tensor，无需额外指定dataformats
writer.add_image("Tensor_img", tensor_img)

"""
为什么需要Tensor数据类型：
1. 统一数据格式：神经网络各层要求统一的张量格式
2. 自动微分：内置梯度计算支持反向传播
3. GPU加速：可无缝转移到GPU进行并行计算
4. 批处理支持：天然支持批量数据处理
5. 丰富操作：提供数百种张量运算优化
"""

# 使用中应关闭writer释放资源
writer.close()