import torch
from torch import nn

# 由于使用苹果的设备，所以以下都是苹果设备相关的测试
print(torch.backends.mps.is_available())
mps_device = torch.device("mps")

# 必须通过.float()转换为浮点型。mps不支持整型
x = torch.tensor([1, 2, 3]).float()

net = nn.Linear(3, 1).float()
net = net.to(mps_device)

x = x.to(mps_device)
print(net(x))
