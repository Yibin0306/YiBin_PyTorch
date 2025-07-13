# 导入必要的库
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录工具
import numpy as np  # 数值计算库
from PIL import Image  # 图像处理库

# 创建SummaryWriter实例，指定日志保存目录为"TensorBoard"
# 所有记录的数据将保存在该目录中，可通过TensorBoard可视化
writer = SummaryWriter("TensorBoard")

# 定义图像文件路径
img_path = "D:\\Resources\\PyTorch_Learn\\hymenoptera_data\\hymenoptera_data\\train\\ants_image\\0013035.jpg"

# 使用PIL库打开图像文件
img_PIL = Image.open(img_path)

# 将PIL图像转换为NumPy数组格式
# 转换后的数组维度为[高度, 宽度, 通道] (HWC)
img_array = np.array(img_PIL)

# 将图像添加到TensorBoard
# 参数说明：
# 'test' - 图像在TensorBoard中的标签名
# img_array - 图像数据（NumPy数组）
# 1 - 全局步骤(step)标识符，可用于时间序列
# dataformats='HWC' - 指定数据格式为(高度, 宽度, 通道)
writer.add_image('test', img_array, 1, dataformats='HWC')

# 循环100次，模拟训练过程记录标量数据
for i in range(100):
    # 添加标量数据到TensorBoard
    # 参数说明：
    # 'y=2x' - 标量在TensorBoard中的标签名
    # 2*i - 标量值（此处模拟y=2x函数）
    # i - X轴值（训练步数）
    writer.add_scalar('y=2x', 2*i, i)

# 关闭SummaryWriter，确保所有数据写入磁盘
# 重要：不关闭可能导致部分数据丢失
writer.close()