import torch

# dimension 2  ---liner
a = torch.randn(2, 3)
print(a)

print(a.shape)

print(a.size(0))
print(a.size(1))
print(a.shape[0])
print(a.shape[1])


# dimension 3   ----  recurrent neural network
b = torch.rand(1, 2, 3)
print(b)
print(b[0])
print(b[0][0])
print(b.shape)
print(b.size(0))
print(b.size(1))
print(b.shape[0])
print(b.shape[1])


# dimension 4  ---- convolutional neural networks
c = torch.randn(2, 3, 28, 28)
print(c)
print(c[0])
print(c[0][0])
print(c[0][0][0])
print(c.shape)
print(c.numel())