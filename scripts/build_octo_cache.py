"""
Octo Embedding Cache Builder
=============================
Extracts Octo last-transformer-layer embeddings for all LIBERO demos and saves
them in the same cache format used by the OpenVLA projector pipeline (v2):

    {
        train_emb        [N, OCTO_DIM]  — Octo readout_action embedding (mean-pooled over readout tokens)
        train_teacher_mu [N, 64]        — VAE encoder mu  (computed in PyTorch after JAX extraction)
        train_teacher_lv [N, 64]        — VAE encoder logvar
        train_clip_emb   [N, DIM]       — Text backbone embeddings (kept 'clip_emb' key for compat)
        train_actions    [N, 8, 7]      — normalised action chunks
        test_* same structure
    }

Usage:
    python3 scripts/build_octo_cache.py \
        --suite libero_spatial \
        --octo_model hf://rail-berkeley/octo-small-1.5 \
        --vae_type text_cond_beta_tcvae \
        --beta 0.001 --z_dim 64 \
        --text_backbone octo_t5
"""



import os
import sys
import argparse
import glob
import gc

# Prevent tokenizers from deadlocking
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prevent JAX from hogging 100% of the VRAM, leaving none for PyTorch VAE/Text Encoders
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import h5py
from PIL import Image

# PyTorch
import torch
import torch.nn.functional as F

from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE, TCNTextCondPriorCVAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HDF5_ROOT       = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK_SIZE      = 8
ACTION_DIM      = 7
N_BLOCKS        = max(3, (CHUNK_SIZE - 1).bit_length())
TRAIN_SPLIT     = 7          
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP            = 100000

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",          type=str, required=True, choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--octo_model",     type=str, default="hf://rail-berkeley/octo-small-1.5")
    p.add_argument("--out_dir",        type=str, default="./checkpoints/projectors/octo")
    p.add_argument("--vae_type",       type=str, default="text_cond_beta_tcvae", choices=["text_cond_beta_tcvae", "text_cvae", "cond_prior"])
    p.add_argument("--vae_seed",       type=int, default=2)
    p.add_argument("--beta",           type=float, default=0.001)
    p.add_argument("--z_dim",          type=int,   default=64)
    p.add_argument("--vla_layer_idx",  type=int, default=-1, help="Which VLA layer to tap (e.g. 16 for intermediate, -1 for last)")
    p.add_argument("--text_backbone",  type=str, default="clip", choices=["clip", "smollm", "octo_t5", "openvla_llama"])
    p.add_argument("--embed_batch_size", type=int, default=8)
    p.add_argument("--protocol",       type=str, default="A", choices=["A", "B"], help="A: Train on all tasks. B: Hold out tasks for testing.")
    p.add_argument("--emb_cache_from", type=str, default=None)
    p.add_argument("--vae_checkpoint", type=str, default=None, help="Direct path to VAE checkpoint (overrides auto-generated path)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Load HDF5
# ---------------------------------------------------------------------------
def load_suite(suite: str):
    hdf5_dir   = os.path.join(HDF5_ROOT, f"{suite}_no_noops_hdf5")
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    assert hdf5_files, f"No HDF5 files found at {hdf5_dir}"

    task_dict = {}
    for fpath in tqdm.tqdm(hdf5_files, desc=f"Loading {suite}"):
        task_name = os.path.splitext(os.path.basename(fpath))[0]
        task_dict[task_name] = {"images": [], "actions": []}

        with h5py.File(fpath, "r") as f:
            if "problem_info" in f.attrs:
                import json
                info = json.loads(f.attrs["problem_info"])
                instr = info.get("language_instruction", task_name.replace("_", " ").replace(" demo", ""))
            else:
                instr = task_name.replace("_demo", "").replace("_", " ")

            task_dict[task_name]["instruction"] = instr

            for demo_key in f["data"].keys():
                imgs = f["data"][demo_key]["obs"]["agentview_rgb"][:]
                acts = f["data"][demo_key]["actions"][:]

                for start in range(0, len(acts) - CHUNK_SIZE + 1):
                    img_chunk = imgs[start]
                    act_chunk = acts[start:start + CHUNK_SIZE]
                    if len(act_chunk) == CHUNK_SIZE:
                        task_dict[task_name]["images"].append(img_chunk)
                        task_dict[task_name]["actions"].append(act_chunk)

    return task_dict


# ---------------------------------------------------------------------------
# 2. Octo embedding pass (JAX)
# ---------------------------------------------------------------------------
def embed_octo(model, images_np, instructions, batch_size=8):
    if not images_np:
        return np.zeros((0, 384), dtype=np.float32)

    from octo.model.octo_model import OctoModel

    all_embs = []
    for i in tqdm.tqdm(range(0, len(images_np), batch_size), desc="Octo embed"):
        batch_imgs   = np.stack([
            np.array(Image.fromarray(img).resize((256, 256)))
            for img in images_np[i:i+batch_size]
        ])
        batch_instrs = instructions[i:i+batch_size]

        obs = {
            "image_primary":      batch_imgs[:, np.newaxis],
            "timestep_pad_mask":  np.ones((len(batch_imgs), 1), dtype=bool),
        }
        task = model.create_tasks(texts=batch_instrs)

        transformer_out = model.run_transformer(
            observations=obs,
            tasks=task,
            timestep_pad_mask=obs["timestep_pad_mask"],
            train=False,
        )

        readout = transformer_out["readout_action"].tokens
        emb     = readout.mean(axis=(1, 2))
        all_embs.append(np.array(emb))

    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# 3. Dynamic Text Backbone (PyTorch)
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_vla_text_backbone(instructions, text_backbone, device):
    from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel
    
    print(f"📝 Pre-computing text embeddings using {text_backbone}...")
    
    if text_backbone == "smollm":
        model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
        max_len = 128
    elif text_backbone == "octo_t5":
        model_id = "t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
        max_len = 128
    elif text_backbone == "openvla_llama":
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
        max_len = 128
    else: 
        model_id = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
        max_len = 77  # ⚠️ CLIP crash prevention

    text_cache, all_embs = {}, []
    for instr in tqdm.tqdm(instructions, desc="Text Embed"):
        if instr not in text_cache:
            toks = tokenizer([instr], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            if text_backbone == "clip":
                emb = text_encoder(**toks).pooler_output[0]
            else:
                outputs = text_encoder(**toks)
                hidden_states = outputs.last_hidden_state
                mask = toks.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = (sum_embeddings / sum_mask)[0]
            text_cache[instr] = emb.float().cpu()
        all_embs.append(text_cache[instr])
        
    text_encoder.cpu()
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(all_embs)


# ---------------------------------------------------------------------------
# 4. VAE teacher encode (PyTorch)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_teacher_vae(vae, actions_tensor, vae_type, device, text_embs, batch_size=256):
    all_mu, all_lv = [], []
    for i in tqdm.tqdm(range(0, len(actions_tensor), batch_size), desc="VAE teacher"):
        a_batch = actions_tensor[i:i+batch_size].to(device)
        
        if vae_type in ["text_cond_beta_tcvae", "cond_prior"]:
            mu, lv = vae.encode(a_batch)
        else:
            t_batch = text_embs[i:i+batch_size].to(device)
            # mu, lv = vae.encode(a_batch, t_batch)
            zero_t = torch.zeros(a_batch.size(0), t_batch.shape[-1], device=device)
            mu, lv = vae.encode(a_batch, zero_t)
            
        all_mu.append(mu.float().cpu())
        all_lv.append(lv.float().cpu())
    return torch.cat(all_mu), torch.cat(all_lv)


# ---------------------------------------------------------------------------
# 5. Normalise actions
# ---------------------------------------------------------------------------
def load_action_stats(suite):
    stats_path = f"./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt"
    if not os.path.exists(stats_path):
        return None, None, None
    full_stats = torch.load(stats_path, map_location="cpu")
    suite_key = f"{suite}_no_noops" if f"{suite}_no_noops" in full_stats else "libero_spatial_no_noops"
    if suite_key not in full_stats:
        return None, None, None
    stats = full_stats[suite_key]["action"]
    action_min  = torch.tensor(stats["min"]).float()
    action_max  = torch.tensor(stats["max"]).float()
    action_mask = torch.tensor(stats["mask"]).float()
    return action_min, action_max, action_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    vae_arch = {"text_cond_beta_tcvae": "tcn", "text_cvae": "cvae", "cond_prior": "cond_prior"}.get(args.vae_type, "cvae")
    
    os.makedirs(f"{args.out_dir}/{args.suite}", exist_ok=True)
    _layer_suffix = f"_layer{args.vla_layer_idx}" if args.vla_layer_idx != -1 else ""
    _seed_suffix = f"_seed{args.vae_seed}"
    out_path = f"{args.out_dir}/{args.suite}/vla_emb_cache_{args.vae_type}_arch_{vae_arch}_beta{args.beta}_z{args.z_dim}_text_{args.text_backbone}{_seed_suffix}{_layer_suffix}.pt"

    text_embed_dim_dict = {'smollm': 960, 'octo_t5': 768, 'openvla_llama': 4096, 'clip': 512}
    text_emb_dim = text_embed_dim_dict.get(args.text_backbone, 512)

    # ── 1. Load Data / Embeddings ───────────────────────────────────────────
    if args.emb_cache_from and os.path.exists(args.emb_cache_from):
        print(f"⚡ Loading embeddings from existing cache: {args.emb_cache_from}")
        _prev = torch.load(args.emb_cache_from, map_location="cpu")
        train_emb      = _prev["train_emb"]
        test_emb       = _prev["test_emb"]
        tr_acts        = _prev["train_actions"]
        te_acts        = _prev["test_actions"]
        train_text_emb = _prev["train_clip_emb"]
        test_text_emb  = _prev["test_clip_emb"]
        OCTO_DIM       = train_emb.shape[-1]
    else:
        from octo.model.octo_model import OctoModel
        print(f"🤖 Loading Octo from {args.octo_model}…")
        octo_model = OctoModel.load_pretrained(args.octo_model)
        OCTO_DIM   = octo_model.module.octo_transformer.token_embedding_size

        task_dict  = load_suite(args.suite)
        task_names = sorted(task_dict.keys())
        
        if args.protocol == "A":
            print(f"🚀 PROTOCOL A: Using all {len(task_names)} tasks for training (no held-out test split).")
            train_tasks = task_names
            test_tasks  = []
        else:
            print(f"🚀 PROTOCOL B: Holding out {len(task_names) - TRAIN_SPLIT} tasks for testing.")
            train_tasks = task_names[:TRAIN_SPLIT]
            test_tasks  = task_names[TRAIN_SPLIT:]

        def gather(task_list):
            if not task_list:
                return [], torch.zeros(0, CHUNK_SIZE, ACTION_DIM), []
            imgs, acts, instrs = [], [], []
            for t in task_list:
                n = len(task_dict[t]["images"])
                imgs.extend(task_dict[t]["images"])
                acts.extend(task_dict[t]["actions"])
                instrs.extend([task_dict[t]["instruction"]] * n)
            acts_tensor = torch.from_numpy(np.stack(acts)).float()
            return imgs, acts_tensor, instrs

        tr_imgs, tr_acts, tr_instrs = gather(train_tasks)
        te_imgs, te_acts, te_instrs = gather(test_tasks)

        act_min, act_max, act_mask = load_action_stats(args.suite)
        if act_min is not None:
            def _normalize(acts):
                norm = (acts - act_min) / (act_max - act_min + 1e-5) * 2.0 - 1.0
                return norm * act_mask + acts * (1.0 - act_mask)
            tr_acts = _normalize(tr_acts)
            te_acts = _normalize(te_acts)
        
        print("\n🧠 Octo embedding pass (train)…")
        train_emb_np = embed_octo(octo_model, tr_imgs, tr_instrs, batch_size=args.embed_batch_size)
        print("🧠 Octo embedding pass (test)…")
        test_emb_np  = embed_octo(octo_model, te_imgs, te_instrs, batch_size=args.embed_batch_size)

        train_emb = torch.from_numpy(train_emb_np).float()
        test_emb  = torch.from_numpy(test_emb_np).float()
        
        # Free JAX memory before PyTorch models load
        del octo_model
        gc.collect()

        train_text_emb = embed_vla_text_backbone(tr_instrs, args.text_backbone, DEVICE)
        test_text_emb  = embed_vla_text_backbone(te_instrs, args.text_backbone, DEVICE) if te_instrs else torch.zeros(0, text_emb_dim, device="cpu")

    # ── 2. Load VAE (PyTorch) ────────────────────────────────────────────────
    if args.vae_checkpoint is not None:
        vae_ckpt = args.vae_checkpoint
    else:
        if args.text_backbone == "clip":
            vae_ckpt = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_cyc4_vel0.5_{vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'
        else:
            vae_ckpt = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_text_{args.text_backbone}_seed_{args.vae_seed}_cyc4_vel0.5_{vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'

    print(f"🎓 Loading VAE from {os.path.basename(vae_ckpt)}…")
    
    if vae_arch == "tcn":
        vae = TCNTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=args.z_dim,
            text_emb_dim=text_emb_dim, beta=args.beta, dropout=0.15, hidden_channels=64, n_blocks=N_BLOCKS,
        ).to(DEVICE)
    elif vae_arch == "cvae":
        vae = TCNTextActionCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=args.z_dim,
            text_emb_dim=text_emb_dim, beta=args.beta, dropout=0.15,
            hidden_channels=64, n_blocks=N_BLOCKS, enc_text_gate_init=0.0
        ).to(DEVICE)
    elif vae_arch == "cond_prior":
        vae = TCNTextCondPriorCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=args.z_dim,
            text_emb_dim=text_emb_dim, beta=args.beta, dropout=0.15,
            hidden_channels=64, n_blocks=N_BLOCKS
        ).to(DEVICE)
        
    vae.load_state_dict(torch.load(vae_ckpt, map_location=DEVICE))
    vae.eval()

    # ── 3. VAE teacher targets (PyTorch) ──────────────────────────────────────
    print("\n🎓 VAE teacher encode…")
    train_teacher_mu, train_teacher_lv = encode_teacher_vae(vae, tr_acts, args.vae_type, DEVICE, text_embs=train_text_emb)
    
    if len(te_acts) > 0:
        test_teacher_mu,  test_teacher_lv  = encode_teacher_vae(vae, te_acts, args.vae_type, DEVICE, text_embs=test_text_emb)
    else:
        test_teacher_mu = torch.zeros(0, args.z_dim, device="cpu")
        test_teacher_lv = torch.zeros(0, args.z_dim, device="cpu")

    if args.vla_layer_idx != -1:
        print(f"⚠️  WARNING: --vla_layer_idx {args.vla_layer_idx} was passed, but intermediate layer extraction is not implemented for Octo (JAX). Octo will use the final layer (-1).")

    # ── 4. Save cache ────────────────────────────────────────────────────────
    torch.save({
        "train_emb":        train_emb,
        "train_teacher_mu": train_teacher_mu,
        "train_teacher_lv": train_teacher_lv,
        "train_clip_emb":   train_text_emb, # Keeping key name 'clip_emb' for downstream compatibility
        "train_actions":    tr_acts,
        "test_emb":         test_emb,
        "test_teacher_mu":  test_teacher_mu,
        "test_teacher_lv":  test_teacher_lv,
        "test_clip_emb":    test_text_emb,
        "test_actions":     te_acts,
        "octo_dim":         OCTO_DIM,
    }, out_path)
    
    print(f"\n💾 Cache saved → {out_path}")

if __name__ == "__main__":
    main()