# from datasets import load_dataset

# ds = load_dataset("EPFL-CVLAB-SPACECRAFT/SwissCube")

import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Architectures:", torch.cuda.get_arch_list())

# Test an actual GPU computation
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("GPU computation successful:", y.shape)