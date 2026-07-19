import torch

print("=" * 40)
print("PyTorch Test")
print("=" * 40)

print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())
print("CUDA Device Cnt :", torch.cuda.device_count())

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

print("\nTensor X:")
print(x)

print("\nTensor Y:")
print(y)

print("\nX + Y =")
print(x + y)

print("\nX × Y =")
print(torch.matmul(x, y))