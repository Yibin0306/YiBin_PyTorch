# 导入必要的库
import torch
import torchvision.transforms  # 提供图像预处理工具
from PIL import Image  # Python图像处理库
from torch import nn, device  # PyTorch神经网络模块和设备管理

# 加载并预处理图像
image_path = 'dog.png'  # 图像文件路径
image = Image.open(image_path)  # 打开图像文件
image = image.convert('RGB')  # 转换为RGB格式（3通道）

# 定义图像预处理流程
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),  # 调整图像大小为32x32像素
    torchvision.transforms.ToTensor()  # 将PIL图像转换为PyTorch张量（并自动归一化到[0,1]范围）
])

image = transform(image)  # 应用预处理变换


# 定义神经网络模型类（继承自nn.Module）
class Net(nn.Module):
    def __init__(self):
        """网络模型初始化"""
        super(Net, self).__init__()  # 调用父类初始化方法

        # 使用Sequential容器按顺序构建网络层
        self.model = nn.Sequential(
            # 卷积层1：输入通道3(RGB)，输出32通道，5x5卷积核，步长1，填充2（保持特征图尺寸）
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),  # 2x2最大池化（下采样，尺寸减半）

            # 卷积层2：输入32通道，输出32通道
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),  # 再次池化

            # 卷积层3：输入32通道，输出64通道
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),  # 最终池化后特征图尺寸：32->16->8->4

            nn.Flatten(),  # 展平层：将三维特征图转换为一维向量（64通道*4 * 4=1024维）

            # 全连接层1：1024维输入 -> 64维输出
            nn.Linear(64 * 4 * 4, 64),
            # 输出层：64维 -> 10维（对应CIFAR-10数据集的10个类别）
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """定义前向传播过程"""
        x = self.model(x)  # 数据通过顺序模型
        return x


# 加载预训练模型（强制使用CPU设备）
model = torch.load("Net_MODEL_0.pth", map_location=torch.device('cpu'))
# 调整图像张量形状：添加批次维度 (C,H,W) -> (1,C,H,W)
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()  # 设置模型为评估模式（禁用dropout/batchnorm等训练专用层）
with torch.no_grad():  # 禁用梯度计算（节省内存，加速推理）
    output = model(image)  # 模型前向传播

# 输出原始预测张量（10个类别的分数）
print(output)
# 输出预测类别索引（沿维度1取argmax获得预测类别）
print(output.argmax(1))