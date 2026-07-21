"""
Latent Space Visualization — DecOnly vs CVAE
=============================================
Produces side-by-side PCA + t-SNE plots colored by task, comparing:
  - TCNTextActionBetaTCVAE  (decoder-only text conditioning)
  - TCNTextActionCVAE       (full text conditioning in encoder + decoder)

Key design decisions:
  - Data loaded from HDF5 (task-level labels, same split as projector training)
  - CVAE encoder gets real CLIP text; decoder-only encoder gets no text (correct)
  - Colors by task (10 per suite); train tasks = filled circles, held-out = stars
  - PCA first (fast, global structure), then t-SNE (local clusters)

Usage:
    python3 scripts/viz_latent_space.py \
        --suite libero_spatial \
        --deconly_ckpt checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.001_z64_alpha1.0_chunk8_std_cyc4_vel0.5_tcn_seed_2_step_100000.pt \
        --cvae_ckpt   checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.001_z64_alpha1.0_chunk8_std_cyc4_vel0.5_cvae_seed_2_step_100000.pt \
        --out_dir     plots/latent_space \
        --samples_per_task 200
"""

import os, sys, argparse, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import CLIPTokenizer, CLIPTextModel
import tqdm

from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE

# ── Config ──────────────────────────────────────────────────────────────────
HDF5_ROOT    = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK_SIZE   = 8
ACTION_DIM   = 7
Z_DIM        = 64
N_BLOCKS     = max(3, (CHUNK_SIZE - 1).bit_length())
TRAIN_SPLIT  = 7   # first 7 tasks = train, rest = held-out
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_COLORS  = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",            type=str, required=True,
                   choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--deconly_ckpt",     type=str, required=True,
                   help="TCNTextActionBetaTCVAE checkpoint (decoder-only text)")
    p.add_argument("--cvae_ckpt",        type=str, default=None,
                   help="TCNTextActionCVAE checkpoint (full text conditioning). "
                        "Omit to plot only the decoder-only model.")
    p.add_argument("--out_dir",          type=str, default="plots/latent_space")
    p.add_argument("--samples_per_task", type=int, default=200)
    p.add_argument("--tsne_perplexity",  type=int, default=30)
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────
def load_task_data(suite, samples_per_task):
    hdf5_dir   = os.path.join(HDF5_ROOT, f"{suite}_no_noops_hdf5")
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    assert hdf5_files, f"No HDF5 files at {hdf5_dir}"

    tasks = []  # list of dicts: {name, instruction, actions (T,8,7), is_train}
    for idx, fpath in enumerate(hdf5_files):
        task_name = os.path.splitext(os.path.basename(fpath))[0]
        instr     = task_name.replace("_demo", "").replace("_", " ")
        is_train  = idx < TRAIN_SPLIT

        chunks, phases = [], []
        with h5py.File(fpath, "r") as f:
            # Try to recover instruction from attrs
            if "problem_info" in f.attrs:
                import json
                info  = json.loads(f.attrs["problem_info"])
                instr = info.get("language_instruction", instr)

            for demo_key in f["data"].keys():
                acts = f["data"][demo_key]["actions"][:]   # (T, 7)
                n_chunks = (len(acts) - CHUNK_SIZE) // CHUNK_SIZE + 1
                for ci, start in enumerate(range(0, len(acts) - CHUNK_SIZE + 1, CHUNK_SIZE)):
                    chunk = acts[start:start + CHUNK_SIZE]
                    if len(chunk) == CHUNK_SIZE:
                        chunks.append(chunk)
                        # early = first half of trajectory, late = second half
                        phase = "early" if ci < n_chunks // 2 else "late"
                        phases.append(phase)

        if not chunks:
            continue

        rng    = np.random.default_rng(42 + idx)
        chosen = rng.choice(len(chunks), size=min(samples_per_task, len(chunks)), replace=False)
        acts   = torch.from_numpy(np.stack([chunks[i] for i in chosen])).float()
        phases_chosen = [phases[i] for i in chosen]

        tasks.append({"name": task_name, "instruction": instr,
                      "actions": acts, "phases": phases_chosen,
                      "is_train": is_train, "idx": idx})

    print(f"✅ Loaded {len(tasks)} tasks from {suite} "
          f"({TRAIN_SPLIT} train / {len(tasks)-TRAIN_SPLIT} held-out)")
    return tasks


# ── CLIP text embeddings ──────────────────────────────────────────────────────
@torch.no_grad()
def get_clip_emb(instruction, tokenizer, encoder, device):
    toks = tokenizer([instruction], return_tensors="pt",
                     padding=True, truncation=True, max_length=77).to(device)
    return encoder(**toks).pooler_output[0].float()   # (512,)


# ── Encode all tasks with a given model ──────────────────────────────────────
@torch.no_grad()
def encode_tasks(model, tasks, is_cvae, clip_tokenizer, clip_encoder):
    all_z, all_labels, all_markers, all_phases = [], [], [], []

    for task in tqdm.tqdm(tasks, desc="Encoding tasks"):
        acts = task["actions"].to(DEVICE)    # (N, 8, 7)
        N    = acts.shape[0]

        if is_cvae:
            # CVAE encoder: action + real CLIP text → z (text-shifted clusters)
            clip_emb = get_clip_emb(task["instruction"], clip_tokenizer,
                                    clip_encoder, DEVICE)         # (512,)
            clip_emb = clip_emb.unsqueeze(0).expand(N, -1)        # (N, 512)
            mu, _    = model.encode(acts, clip_emb)
        else:
            # Decoder-only: encoder sees ONLY actions → trajectory-driven z
            mu, _    = model.encode(acts)

        all_z.append(mu.float().cpu().numpy())
        all_labels.extend([task["idx"]] * N)
        all_markers.extend(["train" if task["is_train"] else "test"] * N)
        all_phases.extend(task["phases"])

    return (np.concatenate(all_z, axis=0),
            np.array(all_labels),
            np.array(all_markers),
            np.array(all_phases))


# ── Phase plot ───────────────────────────────────────────────────────────────
PHASE_COLORS = {"early": "#2196F3", "late": "#FF5722"}  # blue=early, orange=late

def make_phase_figure(embeddings_2d, phases, task_labels, tasks, title, out_path):
    """Color by trajectory phase (early/late); marker shape encodes task index."""
    MARKERS = ["o", "s", "^", "D", "v", "p", "h", "8", "P", "X"]
    fig, ax = plt.subplots(figsize=(10, 8))

    for task in tasks:
        m    = MARKERS[task["idx"] % len(MARKERS)]
        mask = task_labels == task["idx"]
        for phase, color in PHASE_COLORS.items():
            pm = mask & (phases == phase)
            if pm.any():
                ax.scatter(embeddings_2d[pm, 0], embeddings_2d[pm, 1],
                           c=color, alpha=0.55, s=14, marker=m, edgecolors="none")

    # Legend: phases (color) + a few task shapes
    phase_handles = [mpatches.Patch(color=c, label=f"{p} trajectory")
                     for p, c in PHASE_COLORS.items()]
    task_handles  = [plt.scatter([], [], c="gray", marker=MARKERS[t["idx"] % len(MARKERS)],
                                 s=20, label=t["instruction"][:30])
                     for t in tasks]
    ax.legend(handles=phase_handles + task_handles, fontsize=7,
              loc="upper right", framealpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  💾 Saved → {out_path}")


# ── Task plot ─────────────────────────────────────────────────────────────────
def make_figure(embeddings_2d, labels, markers, tasks, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    legend_handles = []
    for task in tasks:
        idx     = task["idx"]
        color   = TASK_COLORS[idx % len(TASK_COLORS)]
        mask    = labels == idx
        split   = markers[mask]

        # Train samples: filled circle; held-out: star
        tr_mask = mask & (markers == "train")
        te_mask = mask & (markers == "test")

        if tr_mask.any():
            ax.scatter(embeddings_2d[tr_mask, 0], embeddings_2d[tr_mask, 1],
                       c=color, alpha=0.55, s=12, marker="o", edgecolors="none")
        if te_mask.any():
            ax.scatter(embeddings_2d[te_mask, 0], embeddings_2d[te_mask, 1],
                       c=color, alpha=0.85, s=40, marker="*", edgecolors="k",
                       linewidths=0.3)

        short_name = task["instruction"][:35] + ("…" if len(task["instruction"]) > 35 else "")
        suffix     = " [held-out]" if not task["is_train"] else ""
        legend_handles.append(
            mpatches.Patch(color=color, label=f"{short_name}{suffix}")
        )

    # Legend for marker shapes
    legend_handles += [
        plt.scatter([], [], c="gray", marker="o", s=12, label="train demos"),
        plt.scatter([], [], c="gray", marker="*", s=40, edgecolors="k",
                    linewidths=0.3, label="held-out demos"),
    ]

    ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
              framealpha=0.7, ncol=1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  💾 Saved → {out_path}")


def run_pca_tsne(Z, seed):
    # PCA → 50 dims first for stability, then t-SNE
    n_pca = min(50, Z.shape[1], Z.shape[0] - 1)
    pca   = PCA(n_components=n_pca, random_state=seed)
    Z_pca = pca.fit_transform(Z)
    return Z_pca, pca


def run_tsne(Z_pca, perplexity, seed):
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed, n_jobs=1)
    return tsne.fit_transform(Z_pca)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────
    tasks = load_task_data(args.suite, args.samples_per_task)

    # ── Load CLIP (shared) ───────────────────────────────────────────────────
    print("📝 Loading CLIP text encoder…")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_encoder   = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    for p in clip_encoder.parameters():
        p.requires_grad = False

    models_to_run = []

    # ── Decoder-only model ───────────────────────────────────────────────────
    print(f"\n🏗  Loading decoder-only VAE from {args.deconly_ckpt}…")
    vae_deconly = TCNTextActionBetaTCVAE(
        action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=Z_DIM,
        text_emb_dim=512, beta=0.001, dropout=0.15, hidden_channels=64, n_blocks=N_BLOCKS,
    ).to(DEVICE)
    vae_deconly.load_state_dict(torch.load(args.deconly_ckpt, map_location=DEVICE))
    vae_deconly.eval()
    models_to_run.append(("DecOnly (text in decoder only)", vae_deconly, False))

    # ── CVAE model (optional) ────────────────────────────────────────────────
    if args.cvae_ckpt:
        print(f"🏗  Loading CVAE from {args.cvae_ckpt}…")
        vae_cvae = TCNTextActionCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=Z_DIM,
            text_emb_dim=512, beta=0.001, dropout=0.15, hidden_channels=64, n_blocks=N_BLOCKS,
            enc_text_gate_init=0.0,
        ).to(DEVICE)
        vae_cvae.load_state_dict(torch.load(args.cvae_ckpt, map_location=DEVICE))
        vae_cvae.eval()
        models_to_run.append(("CVAE (text in encoder + decoder)", vae_cvae, True))

    # ── Encode + plot each model ─────────────────────────────────────────────
    all_results = {}
    for label, model, is_cvae in models_to_run:
        print(f"\n⚙️  Encoding with {label}…")
        Z, task_labels, split_markers, phases = encode_tasks(
            model, tasks, is_cvae, clip_tokenizer, clip_encoder
        )
        all_results[label] = (Z, task_labels, split_markers, phases)

        slug = "deconly" if not is_cvae else "cvae"

        # PCA (2D)
        print("  📊 PCA…")
        pca_2d = PCA(n_components=2, random_state=args.seed).fit_transform(Z)
        make_figure(
            pca_2d, task_labels, split_markers, tasks,
            title=f"PCA — {label}\n{args.suite}",
            out_path=os.path.join(args.out_dir, f"{args.suite}_{slug}_pca.png"),
        )
        make_phase_figure(
            pca_2d, phases, task_labels, tasks,
            title=f"PCA (phase) — {label}\n{args.suite}  [blue=early, orange=late]",
            out_path=os.path.join(args.out_dir, f"{args.suite}_{slug}_pca_phase.png"),
        )

        # t-SNE (2D, via PCA50 intermediate)
        print("  📊 t-SNE…")
        Z_pca50, _ = run_pca_tsne(Z, args.seed)
        tsne_2d    = run_tsne(Z_pca50, args.tsne_perplexity, args.seed)
        make_figure(
            tsne_2d, task_labels, split_markers, tasks,
            title=f"t-SNE — {label}\n{args.suite}  (perplexity={args.tsne_perplexity})",
            out_path=os.path.join(args.out_dir, f"{args.suite}_{slug}_tsne.png"),
        )
        make_phase_figure(
            tsne_2d, phases, task_labels, tasks,
            title=f"t-SNE (phase) — {label}\n{args.suite}  [blue=early, orange=late]",
            out_path=os.path.join(args.out_dir, f"{args.suite}_{slug}_tsne_phase.png"),
        )

    # ── Side-by-side comparison (if both models ran) ─────────────────────────
    if len(all_results) == 2:
        print("\n📊 Side-by-side comparison plot…")
        labels_list = list(all_results.keys())

        for method_name, proj_fn in [("PCA", lambda Z: PCA(2, random_state=args.seed).fit_transform(Z)),
                                      ("t-SNE", lambda Z: run_tsne(run_pca_tsne(Z, args.seed)[0],
                                                                    args.tsne_perplexity, args.seed))]:
            # --- task-colored comparison ---
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            for ax, label in zip(axes, labels_list):
                Z, task_labels, split_markers, phases = all_results[label]
                emb_2d = proj_fn(Z)

                for task in tasks:
                    idx   = task["idx"]
                    color = TASK_COLORS[idx % len(TASK_COLORS)]
                    tr_m  = (task_labels == idx) & (split_markers == "train")
                    te_m  = (task_labels == idx) & (split_markers == "test")
                    if tr_m.any():
                        ax.scatter(emb_2d[tr_m, 0], emb_2d[tr_m, 1],
                                   c=color, alpha=0.5, s=10, marker="o", edgecolors="none")
                    if te_m.any():
                        ax.scatter(emb_2d[te_m, 0], emb_2d[te_m, 1],
                                   c=color, alpha=0.9, s=45, marker="*",
                                   edgecolors="k", linewidths=0.3)

                ax.set_title(f"{method_name} — {label}", fontsize=11, fontweight="bold")
                ax.axis("off")

            handles = [mpatches.Patch(color=TASK_COLORS[t["idx"] % len(TASK_COLORS)],
                                      label=t["instruction"][:30] +
                                            (" [held-out]" if not t["is_train"] else ""))
                       for t in tasks]
            handles += [plt.scatter([], [], c="gray", marker="o", s=10, label="train"),
                        plt.scatter([], [], c="gray", marker="*", s=45,
                                    edgecolors="k", linewidths=0.3, label="held-out")]
            fig.legend(handles=handles, fontsize=7, loc="lower center",
                       ncol=5, framealpha=0.7, bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(f"{method_name} Latent Space Comparison — {args.suite}",
                         fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            out = os.path.join(args.out_dir, f"{args.suite}_comparison_{method_name.lower()}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  💾 Saved → {out}")

            # --- phase-colored comparison ---
            MARKERS = ["o", "s", "^", "D", "v", "p", "h", "8", "P", "X"]
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            for ax, label in zip(axes, labels_list):
                Z, task_labels, split_markers, phases = all_results[label]
                emb_2d = proj_fn(Z)

                for task in tasks:
                    m    = MARKERS[task["idx"] % len(MARKERS)]
                    mask = task_labels == task["idx"]
                    for phase, color in PHASE_COLORS.items():
                        pm = mask & (phases == phase)
                        if pm.any():
                            ax.scatter(emb_2d[pm, 0], emb_2d[pm, 1],
                                       c=color, alpha=0.55, s=14, marker=m,
                                       edgecolors="none")

                ax.set_title(f"{method_name} (phase) — {label}", fontsize=11, fontweight="bold")
                ax.axis("off")

            phase_handles = [mpatches.Patch(color=c, label=f"{p} trajectory")
                             for p, c in PHASE_COLORS.items()]
            task_handles  = [plt.scatter([], [], c="gray",
                                         marker=MARKERS[t["idx"] % len(MARKERS)],
                                         s=20, label=t["instruction"][:28])
                             for t in tasks]
            fig.legend(handles=phase_handles + task_handles, fontsize=7,
                       loc="lower center", ncol=5, framealpha=0.7,
                       bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(f"{method_name} Phase Comparison — {args.suite}  "
                         f"[blue=early, orange=late]",
                         fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            out = os.path.join(args.out_dir,
                               f"{args.suite}_comparison_{method_name.lower()}_phase.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  💾 Saved → {out}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
