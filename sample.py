import torch
import time

device = "cpu"

x = torch.randn(5000, 5000).to(device=device)
y = torch.randn(5000, 5000).to(device=device)

t = time.time()
z = x @ y
torch.cuda.synchronize()
print("Time:", time.time() - t)
