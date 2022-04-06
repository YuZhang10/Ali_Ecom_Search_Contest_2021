import torch
print("clear gpu memory!")
with torch.no_grad():
    torch.cuda.empty_cache()