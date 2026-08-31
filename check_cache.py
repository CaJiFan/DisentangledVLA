import torch
import os

path = "checkpoints/projectors/pi0/libero_spatial_no_noops/pi0_emb_cache_cond_prior_fusion3.pt"
size_mb = os.path.getsize(path) / (1024 * 1024)
print(f"File Size: {size_mb:.2f} MB")

try:
    d = torch.load(path, map_location='cpu', weights_only=False)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape} ({v.dtype})")
        elif isinstance(v, dict):
            print(f"{k}: {len(v)} elements")
        else:
            print(f"{k}: {type(v)}")
except Exception as e:
    print("Error loading:", e)
