from pygments.formatters import img
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("TensorBoard")

# writer.add_image()

for i in range(100):
    writer.add_scalar('y=2x', 2*i, i)

writer.close()