import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

device = "cuda"
print("Loading policy...")
policy = PI0Policy.from_pretrained("lerobot/pi0").to(device)
print("Policy loaded.")

called_modules = []
def trace_hook(name):
    def hook(m, i, o):
        called_modules.append(name)
    return hook

for name, module in policy.named_modules():
    if "layers" in name or "block" in name:
        module.register_forward_hook(trace_hook(name))

B = 1
img_t = torch.zeros((B, 3, 480, 640), device=device)
dummy_inputs = {
    "observation.images.camera0": img_t,
    "observation.images.camera1": img_t,
    "observation.images.camera2": img_t,
    "observation.language.tokens": torch.zeros((B, 48), dtype=torch.long, device=device),
    "observation.language.attention_mask": torch.ones((B, 48), dtype=torch.bool, device=device),
    "observation.state": torch.zeros((B, 14), dtype=torch.float32, device=device),
    "action": torch.zeros((B, 50, 14), dtype=torch.float32, device=device),
}

print("Running forward pass...")
try:
    policy(dummy_inputs)
except Exception as e:
    print(f"Forward pass failed with: {e}")

print("Modules called:")
for m in called_modules:
    print(m)
