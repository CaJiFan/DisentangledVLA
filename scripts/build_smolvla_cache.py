"""
SmolVLA Embedding Cache Builder
=================================
Extracts SmolVLA VLM-backbone hidden states for all LIBERO demos and saves
them in the same v2 cache format used by the OpenVLA/Octo projector pipeline:

    {
        train_emb        [N, 960]   — SmolVLA VLM prefix hidden states (mean-pooled)
        train_teacher_mu [N, Z]     — VAE encoder mu  (PyTorch)
        train_teacher_lv [N, Z]     — VAE encoder logvar
        train_clip_emb   [N, 512]   — CLIP pooler_output
        train_actions    [N, 8, 7]  — normalised action chunks
        test_*           same structure
    }

Run this script INSIDE the smolvla_worker Docker container (lerobot installed,
PyTorch + CUDA 12.4).  Training runs unchanged in openvla_worker — only the
.pt cache files need to cross the container boundary (they live in the shared
checkpoints/ directory).

Architecture facts (lerobot/smolvla_base):
    hidden_size       = 960   ← VLM backbone (SmolLM2-135M)
    expert_hidden_size = 720   ← action expert  (not used here)
    max_state_dim      = 32    ← zero-padded; zeros used for all timesteps

Usage:
    # Build TCN (z=128) cache for libero_spatial:
    python3 scripts/build_smolvla_cache.py \\
        --suite libero_spatial \\
        --smolvla_model lerobot/smolvla_base \\
        --vae_checkpoint ./checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.001_z128_alpha1.0_chunk8_std_cyc4_vel0.5_tcn_seed_2_step_100000.pt \\
        --vae_type text_cond_beta_tcvae --vae_seed 2 --beta 0.001 --z_dim 128

    # Build CVAE (z=64) cache — reuse embeddings, recompute teacher only:
    python3 scripts/build_smolvla_cache.py \\
        --suite libero_spatial \\
        --vae_checkpoint ./checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.001_z64_alpha1.0_chunk8_std_cyc4_vel0.5_cvae_seed_1_step_100000.pt \\
        --vae_type text_cvae --vae_seed 1 --beta 0.001 --z_dim 64 \\
        --emb_cache_from ./checkpoints/projectors/smolvla/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128.pt
"""

import os
import sys
import argparse
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import json
import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HDF5_ROOT   = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK_SIZE  = 8
ACTION_DIM  = 7
N_BLOCKS    = max(3, (CHUNK_SIZE - 1).bit_length())
TRAIN_SPLIT = 7       # first 7 tasks = train, rest = test
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",          type=str, required=True,
                   choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--smolvla_model",  type=str, default="lerobot/smolvla_base",
                   help="LeRobot SmolVLA checkpoint (HF repo or local dir).")
    p.add_argument("--vae_checkpoint", type=str, required=False)
    p.add_argument("--out_dir",        type=str, default="./checkpoints/projectors/smolvla")
    p.add_argument("--vae_type",       type=str, default="text_cond_beta_tcvae",
                   choices=["text_cond_beta_tcvae", "text_cvae"])
    p.add_argument("--vae_seed",       type=int, default=2)
    p.add_argument("--beta",           type=float, default=0.001)
    p.add_argument("--z_dim",          type=int,   default=128)
    p.add_argument("--text_backbone", type=str, default="clip",
                   choices=["smollm", "octo_t5", "openvla_llama","clip"],
                   help="Which language model tokenizer to cache as the baseline target.")
    p.add_argument("--embed_batch_size", type=int, default=8,
                   help="Batch size for SmolVLA forward pass.")
    p.add_argument("--vla_layer_idx",  type=int, default=-1, help="Which VLA layer to tap (e.g. 16 for intermediate, -1 for last)")
    p.add_argument("--emb_cache_from", type=str, default=None,
                   help="Path to an existing cache .pt to reuse train_emb/test_emb/"
                        "train_actions/test_actions/train_clip_emb/test_clip_emb. "
                        "Only VAE teacher targets are recomputed. Use when changing "
                        "VAE type without re-running the expensive VLM pass.")
    return p.parse_args()

STEP = 100000
# ---------------------------------------------------------------------------
# 1. Load HDF5 → task_dict  {task_name: {images, actions, instructions}}
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
                info  = json.loads(f.attrs["problem_info"])
                instr = info.get("language_instruction",
                                 task_name.replace("_demo", "").replace("_", " "))
            else:
                instr = task_name.replace("_demo", "").replace("_", " ")

            task_dict[task_name]["instruction"] = instr

            for demo_key in f["data"].keys():
                imgs = f["data"][demo_key]["obs"]["agentview_rgb"][:]  # (T, H, W, 3)
                acts = f["data"][demo_key]["actions"][:]               # (T, 7)

                for start in range(0, len(acts) - CHUNK_SIZE + 1):
                    act_chunk = acts[start:start + CHUNK_SIZE]
                    if len(act_chunk) == CHUNK_SIZE:
                        task_dict[task_name]["images"].append(imgs[start])
                        task_dict[task_name]["actions"].append(act_chunk)

    return task_dict


# ---------------------------------------------------------------------------
# 2. SmolVLA embedding pass (PyTorch — no JAX required)
#
#    Extracts the VLM-backbone prefix hidden states ONLY.
#    The flow-matching action expert is never run — we only want the
#    image+language representation, not action generation.
#
#    Embedding = mean-pool over all valid (non-padded) prefix token positions.
#    Shape: (N, 960)
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_smolvla(policy, images_np, instructions, batch_size=8, device=DEVICE, layer_idx=-1):
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    flow = policy.model            # VLAFlowMatching
    vlm  = flow.vlm_with_expert    # SmolVLMWithExpertModel
    vlm_dtype = next(vlm.parameters()).dtype  # match model dtype (bf16 or f32)

    # Use the resize dimensions from the policy config so that the patch grid
    # is divisible by the pixel_shuffle scale_factor (typically 4).
    # e.g. 256×256 → 16×16 patches (16 % 4 == 0 ✓); 224×224 → 14×14 (14 % 4 ≠ 0 ✗).
    _resize = getattr(policy.config, "resize_imgs_with_padding", None) or (256, 256)
    _tgt_h, _tgt_w = _resize  # PIL.resize takes (width, height)

    target_layer = vlm.vlm.model.layers[layer_idx]
    captured = {}
    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured['hs'] = hs.float().detach().cpu()
    handle = target_layer.register_forward_hook(hook_fn)

    all_embs = []
    for i in tqdm.tqdm(range(0, len(images_np), batch_size), desc="SmolVLA embed"):
        batch_imgs   = images_np[i:i + batch_size]
        batch_instrs = instructions[i:i + batch_size]
        B = len(batch_imgs)

        # ── Language tokenisation ──────────────────────────────────────────
        lang_enc    = vlm.processor.tokenizer(
            batch_instrs, return_tensors="pt", padding=True,
            truncation=True, max_length=48,
        )
        lang_tokens = lang_enc.input_ids.to(device)            # (B, L)
        lang_masks  = lang_enc.attention_mask.to(device).bool()  # (B, L)

        # ── Image preprocessing ────────────────────────────────────────────
        # Resize to the policy's native resolution (read from config), then
        # normalize to [-1, 1] as expected by SigLIP.
        imgs_pt = []
        for img in batch_imgs:
            img_t = torch.from_numpy(
                np.array(Image.fromarray(img).resize((_tgt_w, _tgt_h)))
            ).float() / 255.0              # [0, 1], HWC
            img_t = img_t.permute(2, 0, 1) # CHW
            img_t = img_t * 2.0 - 1.0     # [-1, 1]
            imgs_pt.append(img_t)
        images_tensor = torch.stack(imgs_pt).to(device, dtype=vlm_dtype)  # (B, 3, 224, 224)
        img_masks     = torch.ones(B, dtype=torch.bool, device=device)

        # ── Zero state (32-dim) ────────────────────────────────────────────
        # SmolVLA projects state into the prefix sequence.  We use zeros so
        # the embedding reflects only image+language, matching the projector's
        # inference setting where we won't have robot state available.
        # state_proj is float32 even when the VLM backbone is bfloat16 (mixed-precision),
        # so derive state dtype from the projection layer directly.
        state_dtype = flow.state_proj.weight.dtype
        state = torch.zeros(B, flow.config.max_state_dim, device=device, dtype=state_dtype)

        # ── Build prefix token sequence ────────────────────────────────────
        prefix_embs, prefix_pad, prefix_att = flow.embed_prefix(
            images=[images_tensor],   # list[Tensor(B, 3, H, W)]
            img_masks=[img_masks],    # list[Tensor(B,) bool]
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

        att_2d  = make_att_2d_masks(prefix_pad, prefix_att)
        pos_ids = torch.cumsum(prefix_pad, dim=1) - 1

        # ── VLM forward (prefix only — action expert skipped) ─────────────
        # fill_kv_cache=True forces all layers through forward_attn_layer,
        # which handles None gracefully (line ~215: "if hidden_states is None: continue").
        # fill_kv_cache=False would route cross-attention layers to
        # forward_cross_attn_layer, which unconditionally accesses inputs_embeds[1]
        # and crashes on None.  With use_cache=False no KV cache is actually built.
        # Returns ((prefix_out, None), past_kv)
        (prefix_out, _), _ = vlm.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        
        prefix_out = captured['hs']

        # ── Mean-pool over valid prefix token positions ────────────────────
        # prefix_out: (B, seq_len, 960)
        valid = prefix_pad.unsqueeze(-1).float()          # (B, seq_len, 1)
        emb   = (prefix_out.float() * valid).sum(1) / valid.sum(1).clamp(min=1)
        all_embs.append(emb.cpu())
        
        captured.pop('hs', None)

    handle.remove()
    return torch.cat(all_embs, dim=0)   # (N, 960)


# ---------------------------------------------------------------------------
# 3. CLIP text embedding (PyTorch) — identical to build_octo_cache.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_clip(instructions, device):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    encoder   = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    cache, all_embs = {}, []
    for instr in tqdm.tqdm(instructions, desc="CLIP embed"):
        if instr not in cache:
            toks = tokenizer([instr], return_tensors="pt", padding=True,
                             truncation=True, max_length=77).to(device)
            cache[instr] = encoder(**toks).pooler_output[0].float().cpu()
        all_embs.append(cache[instr])

    encoder.cpu()
    return torch.stack(all_embs)   # (N, 512)

@torch.no_grad()
def embed_vla_text_backbone(instructions, text_backbone, device):
    from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel
    
    print(f"📝 Pre-computing text embeddings using {text_backbone}...")
    
    # --- 1. Load the appropriate Tokenizer, Model, and MAX LENGTH ---
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
        
    else: # CLIP
        model_id = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
        max_len = 77  # ⚠️ CRITICAL: CLIP crashes if max_length > 77

    text_cache, all_embs = {}, []
    
    # --- 2. Encode and Cache Instructions ---
    for instr in tqdm.tqdm(instructions, desc="Text Embed"):
        if instr not in text_cache:
            toks = tokenizer(
                [instr], return_tensors="pt", padding=True, truncation=True, max_length=max_len
            ).to(device)
            
            if text_backbone == "clip":
                # CLIP natively provides a pooler_output (the [CLS] token equivalent)
                emb = text_encoder(**toks).pooler_output[0]
            else:
                # LLMs require manual mean-pooling over the sequence length, ignoring padding
                outputs = text_encoder(**toks)
                hidden_states = outputs.last_hidden_state
                mask = toks.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = (sum_embeddings / sum_mask)[0]
                
            text_cache[instr] = emb.float().cpu()
            
        all_embs.append(text_cache[instr])
        
    # --- 3. Memory Cleanup ---
    text_encoder.cpu()
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    
    return torch.stack(all_embs)


# ---------------------------------------------------------------------------
# 4. VAE teacher encode (PyTorch) — identical to build_octo_cache.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_teacher_vae(vae, actions_tensor, vae_type, device, batch_size=256, text_emb_dim=512, text_emb=None):
    all_mu, all_lv = [], []
    for i in tqdm.tqdm(range(0, len(actions_tensor), batch_size), desc="VAE teacher"):
        a = actions_tensor[i:i + batch_size].to(device)
        if vae_type == "text_cond_beta_tcvae":
            mu, lv = vae.encode(a)
        else:
            zero_t = torch.zeros(a.size(0), text_emb_dim, device=a.device)
            mu, lv = vae.encode(a, zero_t)
            # t_emb = text_emb[i:i + batch_size].to(device) if text_emb is not None else torch.zeros(a.size(0), text_emb_dim, device=a.device)
            # mu, lv = vae.encode(a, t_emb)  # pass real text embeddings for CVAE
        all_mu.append(mu.float().cpu())
        all_lv.append(lv.float().cpu())
    return torch.cat(all_mu), torch.cat(all_lv)


# ---------------------------------------------------------------------------
# 5. Action normalisation — same min-max [-1,1] as the VAE training pipeline
# ---------------------------------------------------------------------------
def load_action_stats(suite):
    stats_path = f"./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt"
    if not os.path.exists(stats_path):
        print(f"⚠️  No stats file at {stats_path} — actions will NOT be normalised.")
        return None, None, None
    full_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    suite_key  = f"{suite}_no_noops"
    if suite_key not in full_stats:
        suite_key = "libero_spatial_no_noops"
    if suite_key not in full_stats:
        print(f"⚠️  Suite key '{suite_key}' not found — actions will NOT be normalised.")
        return None, None, None
    stats = full_stats[suite_key]["action"]
    return stats["min"], stats["max"], stats.get("mask", None)


def normalise_actions(actions_np, a_min, a_max, mask):
    """Min-max normalise to [-1, 1]; masked dims (gripper) kept raw."""
    actions = torch.from_numpy(actions_np).float()   # (N, T, 7)
    if a_min is None:
        return actions
    a_min_t = torch.tensor(a_min, dtype=torch.float32)
    a_max_t = torch.tensor(a_max, dtype=torch.float32)
    rng     = (a_max_t - a_min_t).clamp(min=1e-6)
    normed  = 2.0 * (actions - a_min_t) / rng - 1.0
    if mask is not None:
        mask_t = torch.tensor(mask, dtype=torch.float32)
        normed = normed * mask_t + actions * (1.0 - mask_t)
    return normed


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    args  = parse_args()
    BETA  = args.beta
    Z_DIM = args.z_dim
    _vae_arch = {"text_cond_beta_tcvae": "tcn", "text_cvae": "cvae"}.get(args.vae_type, "")
    _cache_suffix = f"_arch_{_vae_arch}_beta{BETA}_z{Z_DIM}"
    out_dir  = os.path.join(args.out_dir, args.suite)
    os.makedirs(out_dir, exist_ok=True)
    _layer_suffix = f"_layer{args.vla_layer_idx}" if args.vla_layer_idx != -1 else ""
    _seed_suffix = f"_seed{args.vae_seed}"
    out_path = os.path.join(out_dir, f"vla_emb_cache_{args.vae_type}{_cache_suffix}_text_{args.text_backbone}{_seed_suffix}{_layer_suffix}.pt")

    print(f"📦 Cache will be saved to: {out_path}")

    text_embed_dim_dict = {
        'smollm': 960,
        'octo_t5': 768,
        'openvla_llama': 4096,
        'clip': 512,
    }

    text_emb_dim = text_embed_dim_dict.get(args.text_backbone, 512)
    # ── Load HDF5 data ────────────────────────────────────────────────────
    task_dict = load_suite(args.suite)
    task_names = sorted(task_dict.keys())
    train_tasks = task_names[:TRAIN_SPLIT]
    test_tasks  = task_names[TRAIN_SPLIT:]
    print(f"Tasks — train: {len(train_tasks)}, test: {len(test_tasks)}")

    def collect(tasks):
        images, actions, instructions = [], [], []
        for t in tasks:
            n = len(task_dict[t]["images"])
            images     += task_dict[t]["images"]
            actions    += task_dict[t]["actions"]
            instructions += [task_dict[t]["instruction"]] * n
        return images, np.array(actions, dtype=np.float32), instructions

    tr_imgs, tr_acts, tr_instrs = collect(train_tasks)
    te_imgs, te_acts, te_instrs = collect(test_tasks)
    print(f"Samples — train: {len(tr_imgs)}, test: {len(te_imgs)}")

    # ── Normalise actions ─────────────────────────────────────────────────
    a_min, a_max, mask = load_action_stats(args.suite)
    tr_acts_norm = normalise_actions(tr_acts, a_min, a_max, mask)
    te_acts_norm = normalise_actions(te_acts, a_min, a_max, mask)

    # ── Reuse existing embeddings if requested ────────────────────────────
    if args.emb_cache_from and os.path.exists(args.emb_cache_from):
        print(f"⚡ Reusing embeddings from {args.emb_cache_from} — skipping SmolVLA pass.")
        src = torch.load(args.emb_cache_from, map_location="cpu", weights_only=False)
        tr_emb     = src["train_emb"]
        te_emb     = src["test_emb"]
        tr_text_embs    = src["train_clip_emb"]
        te_text_embs    = src["test_clip_emb"]
        # Sanity-check sample count matches current HDF5 load
        assert len(tr_emb) == len(tr_imgs), \
            f"train size mismatch: cache {len(tr_emb)} vs HDF5 {len(tr_imgs)}"
        assert len(te_emb) == len(te_imgs), \
            f"test size mismatch: cache {len(te_emb)} vs HDF5 {len(te_imgs)}"
    else:
        # ── Load SmolVLAPolicy ────────────────────────────────────────────
        print(f"🤖 Loading SmolVLA from {args.smolvla_model} …")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(args.smolvla_model).to(DEVICE).eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        print(f"✅ SmolVLA loaded.  hidden_size={policy.model.vlm_with_expert.config.text_config.hidden_size}")
        _resize_cfg = getattr(policy.config, "resize_imgs_with_padding", None) or (256, 256)
        print(f"   image resize  : {_resize_cfg}  (from policy.config.resize_imgs_with_padding)")

        # ── SmolVLA embedding pass ────────────────────────────────────────
        print(f"🧠 Generating SmolVLA embeddings from layer {args.vla_layer_idx}...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            tr_emb = embed_smolvla(policy, tr_imgs, tr_instrs, args.embed_batch_size, DEVICE, args.vla_layer_idx)
            te_emb = embed_smolvla(policy, te_imgs, te_instrs, args.embed_batch_size, DEVICE, args.vla_layer_idx)

        # ── CLIP embeddings ───────────────────────────────────────────────
        tr_text_embs = embed_vla_text_backbone(tr_instrs, args.text_backbone, DEVICE)
        te_text_embs = embed_vla_text_backbone(te_instrs, args.text_backbone, DEVICE)

        # Free SmolVLA VRAM before VAE pass
        policy.cpu()
        del policy
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"📐 Embedding shape: {tr_emb.shape}  (expect [N, 960])")

    # ── Load VAE and encode teacher targets ───────────────────────────────
    # vae_checkpoint = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_cyc4_vel0.5_{_vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'
    if args.text_backbone == "clip":
        # Legacy CLIP checkpoints don't have 'text_clip' in the filename
        vae_checkpoint = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_cyc4_vel0.5_{_vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'
    else:
        vae_checkpoint = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_text_{args.text_backbone}_seed_{args.vae_seed}_cyc4_vel0.5_{_vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'
    
    if not os.path.exists(vae_checkpoint):
        raise FileNotFoundError(f"❌ VAE checkpoint not found: {vae_checkpoint}")
    VAE_DROPOUT = 0.15
    if args.vae_type == "text_cond_beta_tcvae":
        vae = TCNTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=Z_DIM,
            text_emb_dim=text_emb_dim, beta=BETA, dropout=VAE_DROPOUT,
            hidden_channels=64, n_blocks=N_BLOCKS,
        ).to(DEVICE)
    elif args.vae_type == "text_cvae":
        vae = TCNTextActionCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=Z_DIM,
            text_emb_dim=text_emb_dim, beta=BETA, dropout=VAE_DROPOUT,
            hidden_channels=64, n_blocks=N_BLOCKS, enc_text_gate_init=0.0,
        ).to(DEVICE)

    vae.load_state_dict(torch.load(vae_checkpoint, map_location=DEVICE, weights_only=False))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    tr_mu, tr_lv = encode_teacher_vae(vae, tr_acts_norm, args.vae_type, DEVICE, text_emb_dim=text_emb_dim,
                                       text_emb=tr_text_embs)
    te_mu, te_lv = encode_teacher_vae(vae, te_acts_norm, args.vae_type, DEVICE, text_emb_dim=text_emb_dim,
                                       text_emb=te_text_embs)

    # ── Save cache ────────────────────────────────────────────────────────
    cache = {
        "train_emb":        tr_emb.float(),
        "train_teacher_mu": tr_mu.float(),
        "train_teacher_lv": tr_lv.float(),
        "train_clip_emb":   tr_text_embs.float(),
        "train_actions":    tr_acts_norm.float(),
        "test_emb":         te_emb.float(),
        "test_teacher_mu":  te_mu.float(),
        "test_teacher_lv":  te_lv.float(),
        "test_clip_emb":    te_text_embs.float(),
        "test_actions":     te_acts_norm.float(),
        "smolvla_dim":      tr_emb.shape[-1],   # 960 — for auto-detection in train_projector
    }
    torch.save(cache, out_path)
    print(f"✅ Cache saved: {out_path}")
    print(f"   train_emb  : {tr_emb.shape}")
    print(f"   test_emb   : {te_emb.shape}")
    print(f"   train_mu   : {tr_mu.shape}")
    print(f"   train_clip_emb   : {tr_text_embs.shape}")


if __name__ == "__main__":
    main()

'''
python3 scripts/build_smolvla_cache.py \
  --suite libero_spatial \
  --vae_type text_cond_beta_tcvae \
  --beta 0.001 --z_dim 128 --vae_seed 2
'''
