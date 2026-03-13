import torch

print("Before allocation:")
print(torch.cuda.memory_reserved(0))

# Allocate tensor on GPU
x = torch.randn(1000, 1000, device="cuda")

print("\nAfter allocation:")
print(torch.cuda.memory_reserved(0))