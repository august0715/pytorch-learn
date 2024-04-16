import torch

tensor_0 = torch.arange(0, 12).view(4, 3)
print(tensor_0)

index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(0, index)
print(tensor_1)

index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(1, index)
print(tensor_1)