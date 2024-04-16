import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l
import torchvision


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

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    # * 表示两个矩阵对应元素相乘
    return mask * X / keep_prob

X = torch.arange(16).view(2, 8)
print(X)
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1.0))


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

drop_prob1, drop_prob2 = 0.2, 0.5
# 方法一
# def net(X, is_training=True):
#     X = X.view(-1, num_inputs)
#     H1 = (torch.matmul(X, W1) + b1).relu()
#     if is_training:  # 只在训练模型时使用丢弃法
#         H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
#     H2 = (torch.matmul(H1, W2) + b2).relu()
#     if is_training:
#         H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
#     return torch.matmul(H2, W3) + b3

# # 本函数已保存在d2lzh_pytorch
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         if isinstance(net, torch.nn.Module):
#             net.eval() # 评估模式, 这会关闭dropout
#             acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#             net.train() # 改回训练模式
#         else: # 自定义的模型
#             if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
#                 # 将is_training设置成False
#                 acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
#             else:
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
#         n += y.shape[0]
#     return acc_sum / n


# 方法二
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hiddens1, num_hiddens2), 
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hiddens2, 10)
        )

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = load_data_fashion_mnist(batch_size)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)