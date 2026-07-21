import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent HF tokenizer warnings after DataLoader fork
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #


import torch
from src.disentanglers import TCNTextActionBetaTCVAE

CHUNK_SIZE = 8
N_BLOCKS   = max(3, (CHUNK_SIZE - 1).bit_length())
VAE_CKPT   = "checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.001_z64_alpha1.0_chunk8_std_cyc4_vel0.5_tcn_seed_2_step_100000.pt"

vae = TCNTextActionBetaTCVAE(
    action_dim=7, chunk_size=CHUNK_SIZE, latent_dim=64, text_emb_dim=512,
    beta=0.001, dropout=0.15, hidden_channels=64, n_blocks=N_BLOCKS,
).to("cuda")
vae.load_state_dict(torch.load(VAE_CKPT))
vae.eval()

import torch.nn.functional as F
import h5py
import glob

HDF5_ROOT = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK_SIZE_TEST = 8

SUITES = ["libero_spatial", "libero_object", "libero_goal"]

for suite in SUITES:
    hdf5_dir = os.path.join(HDF5_ROOT, f"{suite}_no_noops_hdf5")
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    if not hdf5_files:
        print(f"[{suite}]  no HDF5 files found at {hdf5_dir} — skipping")
        continue

    all_chunks = []
    files_read = 0
    for fpath in hdf5_files:
        files_read += 1
        with h5py.File(fpath, "r") as f:
            for demo_key in f["data"].keys():
                acts = f["data"][demo_key]["actions"][:]  # (T, 7)
                # Slide a window over the trajectory to extract fixed-length chunks
                for start in range(0, len(acts) - CHUNK_SIZE_TEST + 1, CHUNK_SIZE_TEST):
                    chunk = acts[start:start + CHUNK_SIZE_TEST]
                    if len(chunk) == CHUNK_SIZE_TEST:
                        all_chunks.append(torch.from_numpy(chunk).float())

    # Use all chunks for a representative estimate (no early break)
    acts = torch.stack(all_chunks).cuda()
    with torch.no_grad():
        mu, lv = vae.encode(acts)
        recon  = vae.decode(mu, torch.zeros(len(acts), 512, device="cuda"))
        mse    = F.mse_loss(recon, acts)
    print(f"[{suite}]  recon MSE: {mse:.4f}  |  mean logvar: {lv.mean():.3f}  (n={len(acts)} chunks from {files_read}/{len(hdf5_files)} tasks)")