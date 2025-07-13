"""
pytorch中读取类中主要涉及两个类：

Dataset:数据集（提供一种方式去获取数据及其label）
两个功能：如何获取每一个数据及其label
        告诉我们总共有多少数据

Dataloader:数据装载器（为后面网络提供不同的数据形式）
"""

# 导入PyTorch数据集基类
from torch.utils.data import Dataset
# 导入Python Imaging Library处理图像
from PIL import Image
# 导入操作系统接口模块
import os

# help(Dataset)

# 自定义数据集类，继承自PyTorch的Dataset基类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        """初始化数据集
        Args:
            root_dir (str): 根目录路径
            label_dir (str): 当前类别的子目录名
        """
        self.root_dir = root_dir        # 存储根目录路径
        self.label_dir = label_dir      # 存储标签目录名
        self.path = os.path.join(self.root_dir, self.label_dir)  # 拼接完整数据路径
        self.img_path = os.listdir(self.path)  # 获取该类别下的所有文件名列表

    def __getitem__(self, idx):
        """获取单个样本（实现Dataset的核心接口）"""
        img_name = self.img_path[idx]  # 根据索引获取文件名
        # 拼接图像的完整路径：根目录/标签目录/文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)  # 使用PIL加载图像
        label = self.label_dir           # 直接使用目录名作为标签
        return img, label                # 返回图像对象和标签

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.img_path)  # 当前类别的图像数量

# 数据集路径配置
root_dir = "D:\\Resources\\PyTorch_Learn\\hymenoptera_data\\hymenoptera_data\\train"  # 训练集根目录
ants_label_dir = "ants_image"  # 蚂蚁类别子目录名
bees_label_dir = "bees_image"  # 蜜蜂类别子目录名

# 创建蚂蚁数据集实例
ants_dataset = MyData(root_dir, ants_label_dir)
# 创建蜜蜂数据集实例
bees_dataset = MyData(root_dir, bees_label_dir)

# 合并数据集
train_dataset = ants_dataset + bees_dataset

# 尝试访问合并后数据集的第一个样本
img, label = train_dataset[0]  # 获取图像和标签
img.show()  # 调用PIL的show方法显示图像