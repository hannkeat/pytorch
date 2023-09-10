import torch

print(torch.__version__)
print("GPU", torch.cuda.is_available())

a = torch.randn(2, 3)
print(a)
print(type(a))
print(isinstance(a, torch.FloatTensor))

a = a.cuda()

print(isinstance(a, torch.FloatTensor))