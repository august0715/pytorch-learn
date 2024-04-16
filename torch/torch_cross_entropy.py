import torch

def soft_max(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
   
    # 广播partition
    return x_exp / partition
def cross_entropy(y, y_hat):
    i = len(y_hat)
    # 这里可以使用gather函数
    x = y_hat[range(i), y]
    print("取出对应元素:", x, '真实label:', y)

    return -torch.log(x)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat_softmax = soft_max(y_hat)
print(y_hat_softmax)
out = cross_entropy(y, y_hat_softmax)
print('手动计算的损失', out)
cr_loss = torch.nn.CrossEntropyLoss(reduction="none")
out = cr_loss(y_hat, y)
print('公式计算的损失', out)

cross_loss = torch.nn.CrossEntropyLoss(reduction='none')
input = torch.tensor([[4, 14, 19, 15],
                       [18, 6, 14, 7],
                       [18, 5, 3, 16]], dtype=torch.float)

target = torch.tensor([0, 3, 2])
loss = cross_loss(input, target)
print(loss)


# shape [1, 3, 4]
input = torch.tensor([[[4, 14, 19, 15],
                       [18, 6, 14, 7],
                       [18, 5, 3, 16]]], dtype=torch.float)

input = input.permute(0, 2, 1)
    # shape [1, 3]
target = torch.tensor([[0, 3, 2]])

loss = cross_loss(input, target)
print(loss)
