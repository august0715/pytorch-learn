import torch


# 创建一个形状为(2, 3, 4)的张量
x = torch.randn(2, 3, 4)
print("原始张量形状:", x.shape,x.shape[2])
print("原始张量:", x)

# 使用.view()改变张量形状为(6, 4)
y = x.view(6, 4)
print("改变形状后的张量形状:", y.shape)
print("改变形状后的张量:", y)

y = x.view(3, 8)
print("改变形状后的张量形状:", y.shape)
print("改变形状后的张量:", y)


# 对多维Tensor按维度操作。
# 在下面的例子中，给定一个Tensor矩阵X。我们可以只对其中同一列（dim=0）或同一行（dim=1）的元素求和，
# 并在结果中保留行和列这两个维度（keepdim=True）。
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.dim())
# 同一列
# tensor([[5, 7, 9]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=0, keepdim=True).shape)


# 同一行
# tensor([[ 6],
#         [15]])
print(X.sum(dim=1, keepdim=True))
print(X.sum(dim=1, keepdim=True).shape)

# 如果是一个3*4*5*6的张量，那么
# sum(dim=0, keepdim=True) 会得到1*4*5*6的张量
# sum(dim=1, keepdim=True) 会得到3*1*5*6的张量
# sum(dim=2, keepdim=True) 会得到3*4*1*6的张量
# sum(dim=3, keepdim=True) 会得到3*4*5*1的张量
X = torch.tensor([[[1, 2],[3,4]], [[4, 5],[6,7]], [[40, 50],[60,70]]])
print(X.shape)
print(X.sum(dim=0, keepdim=True)) #[[3,7]]
print(X.sum(dim=0, keepdim=True).shape)
print(X.sum(dim=1, keepdim=True))
print(X.sum(dim=1, keepdim=True).shape)
print(X.sum(dim=2, keepdim=True))
print(X.sum(dim=2, keepdim=True).shape)


