import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch import nn
from Model import LeNet

def test_data_process():

    test_data = FashionMNIST(root='./Data',
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor()
                           ]))

    test_dataloader = Data.DataLoader(dataset=test_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0)

    return test_dataloader