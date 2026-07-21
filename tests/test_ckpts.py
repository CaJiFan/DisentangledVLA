import torch
for p in [
    "checkpoints/projectors/octo/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128.pt",
    "checkpoints/projectors/octo/libero_spatial/vla_emb_cache_text_cvae_arch_cvae_beta0.001_z64.pt",
    "checkpoints/projectors/openvla/libero_spatial/vla_emb_cache_text_cond_beta_tcvae.pt",
    "checkpoints/projectors/openvla/libero_spatial/vla_emb_cache_text_cvae.pt",
]:
    c = torch.load(p, map_location="cpu")
    print(f"{p.split('/')[-1]}")
    print(f"  keys: {list(c.keys())}")
    print(f"  train_emb: {c['train_emb'].shape}, teacher_mu: {c['train_teacher_mu'].shape}")
    print(f"  has_actions: {'train_actions' in c}")