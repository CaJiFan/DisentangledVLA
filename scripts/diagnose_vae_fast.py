#!/usr/bin/env python3
"""
Fast VAE diagnostic — reads directly from HDF5 (no TFRecord streaming).

Usage:
    docker exec smolvla_worker python3 /workspace/scripts/diagnose_vae_fast.py \
        --checkpoint checkpoints/new_protocol_cvae/libero_object/rw100_d0.15_beta0.1_z128_chunk8_protA_cyc4_cond_prior_seed_1_best.pt \
        --suite libero_object --use-cond-prior --latent_dim 128 [--plot]
"""
import os, sys, argparse, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from src.disentanglers import (
    ConvTextActionBetaTCVAE, MLPTextActionBetaTCVAE,
    TCNTextActionBetaTCVAE, TCNTextActionCVAE,
    TCNTextCondPriorCVAE, TCNTextWAE,
)

HDF5_ROOT = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK = 8          # must match training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--suite',      required=True)
    p.add_argument('--latent_dim',    type=int,   default=128)
    p.add_argument('--chunk_size',    type=int,   default=8)
    p.add_argument('--text_backbone', type=str,   default='clip')
    p.add_argument('--dropout',       type=float, default=0.0)
    p.add_argument('--use-mlp',        action='store_true')
    p.add_argument('--use-tcn',        action='store_true')
    p.add_argument('--use-cvae',       action='store_true')
    p.add_argument('--use-cond-prior', action='store_true')
    p.add_argument('--use-wae',        action='store_true')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--stats_path', type=str, default=None,
                   help='Path to dataset_statistics.pt (auto-detected if omitted)')
    return p.parse_args()


def build_model(args, text_emb_dim):
    n_blocks = max(3, (args.chunk_size - 1).bit_length())
    kw = dict(action_dim=7, chunk_size=args.chunk_size, latent_dim=args.latent_dim,
              text_emb_dim=text_emb_dim, beta=0.1, dropout=0.0,
              hidden_channels=64, n_blocks=n_blocks)
    if args.use_cond_prior: return TCNTextCondPriorCVAE(**kw)
    if args.use_wae:        return TCNTextWAE(**kw)
    if args.use_cvae:       return TCNTextActionCVAE(**kw)
    if args.use_tcn:        return TCNTextActionBetaTCVAE(**kw)
    if args.use_mlp:
        return MLPTextActionBetaTCVAE(action_dim=7, chunk_size=args.chunk_size,
                                      latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
                                      beta=0.1, dropout=0.0)
    return ConvTextActionBetaTCVAE(action_dim=7, chunk_size=args.chunk_size,
                                   latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
                                   beta=0.1, dropout=0.0)


def load_data_from_hdf5(suite, chunk_size, stats_path, text_backbone, device):
    """
    Reads every HDF5 demo, slices into chunks of `chunk_size`, normalises,
    and embeds the task instruction with CLIP.
    Returns:
        chunks  : (N, chunk_size, 7) float32 tensor
        text_embs: (N, text_dim) float32 tensor
        task_ids : (N,) long tensor
        task_names: list[str]
    """
    hdf5_dir = os.path.join(HDF5_ROOT, f"{suite}_no_noops_hdf5")
    if not os.path.isdir(hdf5_dir):
        hdf5_dir = os.path.join(HDF5_ROOT, suite)
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {hdf5_dir}")

    # Load normalisation stats
    if stats_path and os.path.exists(stats_path):
        action_stats = torch.load(stats_path, map_location='cpu', weights_only=False)
    else:
        # Try auto-detect
        candidates = glob.glob(f"checkpoints/new_protocol_cvae/{suite}/dataset_statistics.pt")
        candidates += glob.glob(f"checkpoints/**/{suite}/dataset_statistics.pt")
        if candidates:
            action_stats = torch.load(candidates[0], map_location='cpu', weights_only=False)
            print(f"  Stats: {candidates[0]}")
        else:
            action_stats = None
            print("  ⚠️  No stats found — skipping normalisation (using raw actions).")

    if action_stats is not None:
        key = f"{suite}_no_noops"
        stats = action_stats[key]['action']
        action_min  = torch.tensor(stats['min']).float()
        action_max  = torch.tensor(stats['max']).float()
        action_mask = torch.tensor(stats['mask']).float()
    else:
        action_min = action_max = action_mask = None

    # CLIP text encoder
    print("  Loading CLIP text encoder...")
    tokenizer     = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder  = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_cache    = {}

    all_chunks, all_texts, all_task_ids = [], [], []
    task_names = []

    for hdf5_path in hdf5_files:
        # Derive task name from filename
        fname = os.path.basename(hdf5_path).replace(".hdf5", "")
        # Task instruction: try to extract from HDF5 attrs, else use filename
        with h5py.File(hdf5_path, 'r') as f:
            # Some LIBERO HDF5 store the instruction in attrs
            instr = None
            if 'data' in f and hasattr(f['data'], 'attrs'):
                instr = f['data'].attrs.get('language_instruction', None)
                if isinstance(instr, bytes): instr = instr.decode()
            if instr is None:
                # Fall back to filename-derived instruction
                instr = fname.replace("_demo", "").replace("_", " ")

            task_id = len(task_names)
            task_names.append(instr)

            if instr not in text_cache:
                with torch.no_grad():
                    tok = tokenizer([instr], padding=True, truncation=True, return_tensors='pt').to(device)
                    text_cache[instr] = text_encoder(**tok).pooler_output[0].cpu().float()

            text_emb = text_cache[instr]

            demo_keys = sorted(f['data'].keys())
            for dk in demo_keys:
                raw = f[f'data/{dk}/actions'][:]          # (T, 7)
                gt  = torch.tensor(raw, dtype=torch.float32)

                # Normalise
                if action_min is not None:
                    norm = (gt - action_min) / (action_max - action_min + 1e-5) * 2.0 - 1.0
                    norm = norm * action_mask + gt * (1.0 - action_mask)
                else:
                    norm = gt

                # Slice into non-overlapping chunks
                T = norm.shape[0]
                for start in range(0, T - chunk_size + 1, chunk_size):
                    chunk = norm[start : start + chunk_size]   # (chunk_size, 7)
                    all_chunks.append(chunk)
                    all_texts.append(text_emb)
                    all_task_ids.append(task_id)

        print(f"  [{task_id+1}/{len(hdf5_files)}] {fname}: {len(demo_keys)} demos")

    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    chunks   = torch.stack(all_chunks)     # (N, chunk_size, 7)
    texts    = torch.stack(all_texts)      # (N, text_dim)
    task_ids = torch.tensor(all_task_ids, dtype=torch.long)
    return chunks, texts, task_ids, task_names


@torch.no_grad()
def collect_latents(model, chunks, texts, args, device, batch_size=512):
    model.eval()
    needs_text = args.use_cvae
    all_mus = []
    N = len(chunks)
    for i in range(0, N, batch_size):
        a = chunks[i:i+batch_size].to(device)
        t = texts[i:i+batch_size].to(device)
        if needs_text:
            mu, _ = model.encode(a, t)
        else:
            mu, _ = model.encode(a)
        all_mus.append(mu.cpu())
    return torch.cat(all_mus, 0)


def main():
    args = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_emb_dim = {'smollm': 960, 'octo_t5': 768, 'openvla_llama': 4096, 'clip': 512}[args.text_backbone]

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(args, text_emb_dim).to(DEVICE)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    print(f"✅ Loaded checkpoint: {os.path.basename(args.checkpoint)}")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"⏳ Loading HDF5 data for {args.suite}...")
    chunks, texts, task_ids, task_names = load_data_from_hdf5(
        args.suite, args.chunk_size, args.stats_path, args.text_backbone, DEVICE
    )
    N, Z_in = chunks.shape[0], chunks.shape[-1]
    n_tasks = len(task_names)
    print(f"  Loaded {N} chunks from {n_tasks} tasks")

    # ── Encode ───────────────────────────────────────────────────────────────
    print("⏳ Encoding all chunks...")
    all_mus = collect_latents(model, chunks, texts, args, DEVICE)
    Z = all_mus.shape[1]

    print(f"\n{'='*62}")
    print(f"  {args.suite}  N={N}  tasks={n_tasks}  z_dim={Z}")
    print(f"  {os.path.basename(args.checkpoint)}")
    print(f"{'='*62}")

    # ── 1. Active dims ───────────────────────────────────────────────────────
    dim_stds = all_mus.std(0)
    n_active = (dim_stds > 0.10).sum().item()
    n_dead   = (dim_stds < 0.05).sum().item()
    print(f"\n[1] Latent Activity")
    print(f"    Active (std>0.10): {n_active}/{Z}  ({100*n_active/Z:.1f}%)")
    print(f"    Dead   (std<0.05): {n_dead}/{Z}  ({100*n_dead/Z:.1f}%)")
    print(f"    Overall std mean:  {dim_stds.mean().item():.4f}")

    # ── 2. Fisher B/W ratio ──────────────────────────────────────────────────
    task_means = torch.stack([all_mus[task_ids == t].mean(0) for t in range(n_tasks)])
    task_var   = torch.stack([all_mus[task_ids == t].var(0).mean() for t in range(n_tasks)])
    global_mean  = all_mus.mean(0)
    between_var  = ((task_means - global_mean)**2).mean().item()
    within_var   = task_var.mean().item()
    fisher       = between_var / (within_var + 1e-9)

    print(f"\n[2] Fisher B/W Ratio  (want > 0.5)")
    print(f"    Between-task var: {between_var:.6f}")
    print(f"    Within-task  var: {within_var:.6f}")
    print(f"    Fisher ratio:     {fisher:.4f}  {'🟢' if fisher>0.5 else '🟡' if fisher>0.1 else '🔴'}")

    # ── 3. Per-task centroid distances ───────────────────────────────────────
    dists = torch.cdist(task_means, task_means)
    off   = ~torch.eye(n_tasks, dtype=torch.bool)
    print(f"\n[3] Task Centroid Distances")
    print(f"    Mean: {dists[off].mean().item():.4f}  Max: {dists[off].max().item():.4f}  Min: {dists[off].min().item():.4f}")

    # ── 4. Inter-task cosine similarity ─────────────────────────────────────
    mn   = task_means / (task_means.norm(dim=1, keepdim=True) + 1e-8)
    cos  = (mn @ mn.T)[off].mean().item()
    print(f"\n[4] Inter-task Cosine Similarity  (want close to 0)")
    print(f"    Mean: {cos:.4f}  {'🟢' if cos<0.5 else '🟡' if cos<0.8 else '🔴'}")

    # ── 5. k-NN accuracy ─────────────────────────────────────────────────────
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        X = all_mus.numpy(); y = task_ids.numpy()
        if len(X) > 10000:
            idx = np.random.default_rng(0).choice(len(X), 10000, replace=False)
            X, y = X[idx], y[idx]
        X = StandardScaler().fit_transform(X)
        scores = cross_val_score(KNeighborsClassifier(5), X, y, cv=5, scoring='accuracy')
        chance = 1.0 / n_tasks
        print(f"\n[5] k-NN (k=5, 5-fold) Task Classification")
        print(f"    Chance:  {chance*100:.1f}%")
        print(f"    k-NN:    {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%  ({scores.mean()/chance:.1f}x above chance)  {'🟢' if scores.mean()>0.5 else '🟡' if scores.mean()>2*chance else '🔴'}")
    except ImportError:
        scores = None
        print("\n[5] sklearn not available — skipping k-NN")

    # ── 6. Per-task breakdown ────────────────────────────────────────────────
    print(f"\n[6] Per-task centroid norms (should differ if z is task-specific)")
    for t in range(n_tasks):
        nm = task_means[t].norm().item()
        n_samples = (task_ids == t).sum().item()
        short_name = task_names[t][:55]
        print(f"    Task {t:2d} ({n_samples:5d} chunks)  ||centroid||={nm:.3f}  {short_name}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("VERDICT:")
    if fisher < 0.05:
        print("  🔴 z is NOT task-discriminative  (Fisher < 0.05)")
        print("     → Decoder is cheating via text; z ≈ within-task noise")
        print("     → Architecture change needed (not just hyperparams)")
    elif fisher < 0.30:
        print("  🟡 z has WEAK task structure  (Fisher 0.05–0.30)")
        print("     → Some signal but dominated by noise")
    else:
        print("  🟢 z is reasonably task-discriminative  (Fisher > 0.30)")
        print("     → Problem may be in projector or VLA evaluation pipeline")
    print(f"{'='*62}\n")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    if args.plot:
        try:
            from sklearn.manifold import TSNE
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            sub = min(3000, N)
            idx = np.random.default_rng(42).choice(N, sub, replace=False)
            X2  = TSNE(n_components=2, random_state=42, perplexity=40
                       ).fit_transform(all_mus[idx].numpy())
            y2  = task_ids[idx].numpy()

            fig, ax = plt.subplots(figsize=(9, 7))
            sc = ax.scatter(X2[:,0], X2[:,1], c=y2, cmap='tab10', s=14, alpha=0.75)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_ticks(range(n_tasks))
            cbar.set_ticklabels([f"T{i}" for i in range(n_tasks)])
            title_knn = f"  k-NN={scores.mean()*100:.1f}%" if scores is not None else ""
            ax.set_title(f"t-SNE of z — {args.suite}\nFisher={fisher:.3f}{title_knn}  (chance={100/n_tasks:.0f}%)")
            out = args.checkpoint.replace('.pt', '_tsne.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"📊 t-SNE saved → {out}")
        except Exception as e:
            print(f"Plot error: {e}")


if __name__ == '__main__':
    main()
