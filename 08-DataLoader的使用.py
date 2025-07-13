import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10(root='./CIFAR10_DATA', train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
# target是数据的标签属性
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("TensorBoard")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('Epoch：{}'.format(epoch), imgs, step)
        step += 1

writer.close()