import torch
from torch import nn
from torch.nn import init
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 方法一
# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#     def forward(self, x:torch.Tensor): # x shape: (batch, 1, 28, 28)
#         # 将x转化为(batch,1*28*28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return y

# net = LinearNet(num_inputs, num_outputs)
# init.normal_(net.linear.weight, mean=0, std=0.01)
# init.constant_(net.linear.bias, val=0) 


# 方法二 将形象转换使用一个FlattenLayer层
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
    
from collections import OrderedDict

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_outputs)
    # OrderedDict([
    #     ('flatten', FlattenLayer()),
    #     ('linear', nn.Linear(num_inputs, num_outputs))
    # ])
)
init.normal_(net[1].weight, mean=0, std=0.01)
init.constant_(net[1].bias, val=0) 

loss = nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 5


d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
