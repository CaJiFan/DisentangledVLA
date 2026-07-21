import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.disentanglers import TCNTextActionBetaTCVAE
from utils.data import get_full_trajectory_dataloader

DEVICE = "cuda"
CKPT = "./checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.1_z64_alpha1.0_chunk8_std_cyc4_vel0.5_tcn_seed_2_step_100000.pt"

CHUNK_SIZE = 8

vae = TCNTextActionBetaTCVAE(action_dim=7, chunk_size=CHUNK_SIZE, latent_dim=64, text_emb_dim=512).to(DEVICE)
vae.load_state_dict(torch.load(CKPT))
vae.eval()

train_dl, test_dl, _ = get_full_trajectory_dataloader(suite="libero_spatial", batch_size=256)

all_mus = []
with torch.no_grad():
    for actions, texts in test_dl:
        # actions: [B, 256, 7] — slice into non-overlapping chunks of CHUNK_SIZE
        B, T, D = actions.shape
        n_chunks = T // CHUNK_SIZE
        # [B, n_chunks, CHUNK_SIZE, D] -> [B*n_chunks, CHUNK_SIZE, D]
        chunks = actions[:, :n_chunks * CHUNK_SIZE, :].reshape(B * n_chunks, CHUNK_SIZE, D)
        # Repeat text embeddings to match chunks
        text_rep = texts.unsqueeze(1).expand(B, n_chunks, -1).reshape(B * n_chunks, -1)
        mu, _ = vae.encode(chunks.to(DEVICE))
        all_mus.append(mu.cpu())

dim_stds = torch.cat(all_mus).std(dim=0)
print(f"Active (>0.10): {(dim_stds > 0.10).sum().item()}")
print(f"Zombie (0.05–0.10): {((dim_stds >= 0.05) & (dim_stds <= 0.10)).sum().item()}")
print(f"Dead   (<0.05): {(dim_stds < 0.05).sum().item()}")
print(f"Range: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]")
print(f"Dead dim indices: {(dim_stds < 0.05).nonzero().squeeze().tolist()}")