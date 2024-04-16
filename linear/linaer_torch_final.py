import torch
import os
import random
import numpy as np

# 1-prepare dataset
mps_device = torch.device("mps")
# 生成200*3的矩阵，值的范围是0，100
# randint产生的是dtype为long的张量，需要转化为float
data_x = torch.randint(0,100,(200,3),).float()
# l为3*1的矩阵
l = torch.tensor([[5.0],[2.0],[2.0]])

# 线性回滚的偏差符合正太分布
data_y = torch.matmul(data_x,l)+ torch.randn((200, 1)) * 0.5+10
# 2-design model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = torch.nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)
    
model = LinearRegression()

# 3-construct loss and optimizer
criterion = torch.nn.MSELoss(size_average=True)
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001,alpha=0.9)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.003)



# 4-training cycle forward, backward, update
for epoch in range(5):
        outputs = model(data_x)# model的预测
        loss = criterion(outputs,data_y)# model的损失的计算
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for parameter in model.parameters():
    print(parameter)
