#!/usr/bin/env python3
"""
Diagnostic: is z task-discriminative?

Usage (inside container):
    python3 scripts/diagnose_vae.py \
        --checkpoint checkpoints/new_protocol_cvae/libero_object/rw100_d0.75_beta0.1_z128_chunk8_protA_cyc4_cond_prior_seed_1_best.pt \
        --suite libero_object --use-cond-prior --latent_dim 128 --chunk_size 8 \
        [--plot]
"""
import os, sys, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.disentanglers import (
    ConvTextActionBetaTCVAE, MLPTextActionBetaTCVAE,
    TCNTextActionBetaTCVAE, TCNTextActionCVAE,
    TCNTextCondPriorCVAE, TCNTextWAE,
)
from utils.data import get_text_action_ram_cached_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--suite',      required=True,
                   choices=["libero_spatial","libero_object","libero_goal","libero_long","libero_10"])
    p.add_argument('--latent_dim',    type=int,   default=128)
    p.add_argument('--chunk_size',    type=int,   default=8)
    p.add_argument('--text_backbone', type=str,   default='clip')
    p.add_argument('--dropout',       type=float, default=0.0)
    p.add_argument('--use-mlp',        action='store_true')
    p.add_argument('--use-tcn',        action='store_true')
    p.add_argument('--use-cvae',       action='store_true')
    p.add_argument('--use-cond-prior', action='store_true')
    p.add_argument('--use-wae',        action='store_true')
    p.add_argument('--plot', action='store_true', help='Save t-SNE PNG next to checkpoint')
    return p.parse_args()


def build_model(args, text_emb_dim):
    n_blocks = max(3, (args.chunk_size - 1).bit_length())
    kw = dict(action_dim=7, chunk_size=args.chunk_size, latent_dim=args.latent_dim,
              text_emb_dim=text_emb_dim, beta=0.1, dropout=args.dropout,
              hidden_channels=64, n_blocks=n_blocks)
    if args.use_cond_prior: return TCNTextCondPriorCVAE(**kw)
    if args.use_wae:        return TCNTextWAE(**kw)
    if args.use_cvae:       return TCNTextActionCVAE(**kw)
    if args.use_tcn:        return TCNTextActionBetaTCVAE(**kw)
    if args.use_mlp:        return MLPTextActionBetaTCVAE(
        action_dim=7, chunk_size=args.chunk_size, latent_dim=args.latent_dim,
        text_emb_dim=text_emb_dim, beta=0.1, dropout=args.dropout)
    return ConvTextActionBetaTCVAE(
        action_dim=7, chunk_size=args.chunk_size, latent_dim=args.latent_dim,
        text_emb_dim=text_emb_dim, beta=0.1, dropout=args.dropout)


@torch.no_grad()
def collect_latents(model, dataloader, args, device):
    model.eval()
    all_mus, all_texts = [], []
    needs_text_in_encode = args.use_cvae  # only TCNTextActionCVAE.encode() takes text

    for actions, text_feats in dataloader:
        actions   = actions.to(device)
        text_feats = text_feats.to(device)

        if needs_text_in_encode:
            mu, _ = model.encode(actions, text_feats)
        else:
            mu, _ = model.encode(actions)   # works for WAE (returns z, zeros) too

        all_mus.append(mu.cpu())
        all_texts.append(text_feats.cpu())

    return torch.cat(all_mus, 0), torch.cat(all_texts, 0)


def assign_task_ids(all_texts):
    seen, ids = {}, []
    for t in all_texts:
        k = t.numpy().tobytes()
        if k not in seen:
            seen[k] = len(seen)
        ids.append(seen[k])
    return torch.tensor(ids), len(seen)


def main():
    args = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_emb_dim = {'smollm': 960, 'octo_t5': 768, 'openvla_llama': 4096, 'clip': 512}[args.text_backbone]

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(args, text_emb_dim).to(DEVICE)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    print(f"✅ Loaded: {args.checkpoint}")

    # ── Load data (no shuffle so task ordering is stable) ───────────────────
    print("⏳ Loading dataset (this may take a few minutes)...")
    train_dl, _, _ = get_text_action_ram_cached_dataloader(
        suite=args.suite, batch_size=512,
        text_backbone=args.text_backbone,
        train_split_ratio=None,
    )
    # Rebuild a no-shuffle loader from the same TensorDataset
    from torch.utils.data import DataLoader
    noshuffle_dl = DataLoader(train_dl.dataset, batch_size=512, shuffle=False, num_workers=0)

    # ── Encode ───────────────────────────────────────────────────────────────
    print("⏳ Encoding all trajectories...")
    all_mus, all_texts = collect_latents(model, noshuffle_dl, args, DEVICE)
    task_ids, n_tasks = assign_task_ids(all_texts)
    N, Z = all_mus.shape
    print(f"\n{'='*60}")
    print(f"  Suite: {args.suite}  |  N={N}  |  Tasks={n_tasks}  |  z_dim={Z}")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"{'='*60}")

    # ── 1. Active dims ───────────────────────────────────────────────────────
    dim_stds = all_mus.std(0)
    n_active = (dim_stds > 0.1).sum().item()
    n_dead   = (dim_stds < 0.05).sum().item()
    print(f"\n[1] Latent Dim Activity  (out of {Z})")
    print(f"    Active (std>0.10): {n_active:3d}  ({100*n_active/Z:.1f}%)")
    print(f"    Dead   (std<0.05): {n_dead:3d}  ({100*n_dead/Z:.1f}%)")

    # ── 2. Per-task means & Fisher B/W ratio ─────────────────────────────────
    task_means = torch.stack([all_mus[task_ids == t].mean(0) for t in range(n_tasks)])
    task_vars  = torch.stack([all_mus[task_ids == t].var(0).mean() for t in range(n_tasks)])
    global_mean   = all_mus.mean(0)
    between_var   = ((task_means - global_mean)**2).mean().item()
    within_var    = task_vars.mean().item()
    fisher        = between_var / (within_var + 1e-9)

    print(f"\n[2] Cluster Quality (Fisher B/W ratio)")
    print(f"    Between-task var: {between_var:.6f}  (↑ want large)")
    print(f"    Within-task  var: {within_var:.6f}  (↓ want small)")
    print(f"    Fisher ratio:     {fisher:.4f}   (>0.5 = decent, >1.0 = good)")

    # ── 3. Mean inter-task cosine similarity ─────────────────────────────────
    mn = task_means / (task_means.norm(dim=1, keepdim=True) + 1e-8)
    cos = mn @ mn.T
    off = cos[~torch.eye(n_tasks, dtype=torch.bool)].mean().item()
    print(f"\n[3] Inter-task cosine similarity (task centroids)")
    print(f"    Mean: {off:.4f}   (0=orthogonal/good, 1=identical/bad)")

    # ── 4. k-NN task classification accuracy ─────────────────────────────────
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        X = all_mus.numpy()
        y = task_ids.numpy()
        if len(X) > 8000:
            idx = np.random.choice(len(X), 8000, replace=False)
            X, y = X[idx], y[idx]

        X = StandardScaler().fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        chance = 1.0 / n_tasks
        print(f"\n[4] k-NN (k=5, 5-fold CV) Task Classification")
        print(f"    Chance:   {chance*100:.1f}%")
        print(f"    k-NN:     {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%  ({scores.mean()/chance:.1f}x above chance)")
    except ImportError:
        print("\n[4] sklearn not found — skipping k-NN")
        scores = None

    # ── 5. Reconstruction sanity: per-task MSE variance ──────────────────────
    # High variance = model treats tasks differently; low = treats all the same
    print(f"\n[5] Per-task centroid L2 distances (max vs min pair)")
    dists = torch.cdist(task_means, task_means)
    off_mask = ~torch.eye(n_tasks, dtype=torch.bool)
    if off_mask.any():
        print(f"    Max pairwise dist: {dists[off_mask].max().item():.4f}")
        print(f"    Min pairwise dist: {dists[off_mask].min().item():.4f}")
        print(f"    Mean pairwise dist:{dists[off_mask].mean().item():.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT:")
    if fisher < 0.05:
        print("  🔴 z is NOT task-discriminative (Fisher<0.05).")
        print("     The decoder is solving reconstruction via text alone.")
        print("     z encodes only within-task noise (demo-level position variance).")
    elif fisher < 0.3:
        print("  🟡 z has WEAK task structure (Fisher 0.05–0.30).")
        print("     Some task signal in z, but mostly noise.")
    else:
        print("  🟢 z has reasonable task structure (Fisher>0.30).")
        print("     Problem may be in the downstream projector or evaluation.")
    print(f"{'='*60}\n")

    # ── t-SNE plot ────────────────────────────────────────────────────────────
    if args.plot:
        try:
            from sklearn.manifold import TSNE
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            sub = min(3000, N)
            idx = np.random.choice(N, sub, replace=False)
            X2 = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(all_mus[idx].numpy())
            y2 = task_ids[idx].numpy()

            fig, ax = plt.subplots(figsize=(8, 7))
            sc = ax.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='tab10', s=12, alpha=0.7)
            plt.colorbar(sc, ax=ax, label='Task ID')
            ax.set_title(f't-SNE of z — {args.suite}\nFisher={fisher:.3f}  k-NN={scores.mean()*100:.1f}% (chance {chance*100:.1f}%)'
                         if scores is not None else f't-SNE of z — {args.suite}')
            out = args.checkpoint.replace('.pt', '_tsne.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"📊 t-SNE saved to: {out}")
        except Exception as e:
            print(f"Plot failed: {e}")


if __name__ == '__main__':
    main()
