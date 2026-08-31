"""
OpenVLA Embedding Cache Builder
=================================
Extracts OpenVLA backbone hidden states (last token, last layer) for LIBERO demos.

Usage:
    python3 scripts/build_openvla_cache.py \
        --suite libero_spatial \
        --vae_type text_cond_beta_tcvae \
        --beta 0.001 --z_dim 128 --vae_seed 2 --text_backbone clip
"""

import os
import sys
import argparse
import glob
import h5py
import json
import tqdm
import numpy as np
from PIL import Image
import gc

import torch
import torch.nn.functional as F

# Disable tokenizer parallelism to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HDF5_ROOT   = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK_SIZE  = 8
ACTION_DIM  = 7
N_BLOCKS    = max(3, (CHUNK_SIZE - 1).bit_length())
TRAIN_SPLIT = 7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP        = 100000

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",          type=str, required=True, choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--openvla_model",  type=str, default="openvla/openvla-7b", help="HF Repo or local path")
    p.add_argument("--out_dir",        type=str, default="./checkpoints/projectors/openvla")
    p.add_argument("--vae_type",       type=str, default="text_cvae", choices=["text_cond_beta_tcvae", "text_cvae"])
    p.add_argument("--vae_seed",       type=int, default=1)
    p.add_argument("--beta",           type=float, default=0.001)
    p.add_argument("--z_dim",          type=int,   default=128)
    p.add_argument("--text_backbone",  type=str, default="clip", choices=["clip", "smollm", "octo_t5", "openvla_llama"])
    p.add_argument("--embed_batch_size", type=int, default=2, help="Batch size for OpenVLA (Keep low! 7B models use immense VRAM)")
    p.add_argument("--vla_layer_idx",  type=int, default=-1, help="Which VLA layer to tap (e.g. 16 for intermediate, -1 for last)")
    p.add_argument("--num_fusion_layers", type=int, default=1, help="Number of intermediate layers to extract. Overrides vla_layer_idx if > 1.")
    p.add_argument("--emb_cache_from", type=str, default=None, help="Reuse VLM pass if changing VAE")
    return p.parse_args()


def load_suite(suite: str):
    hdf5_dir   = os.path.join(HDF5_ROOT, f"{suite}_no_noops_hdf5")
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    
    task_dict = {}
    for fpath in tqdm.tqdm(hdf5_files, desc=f"Loading {suite}"):
        task_name = os.path.splitext(os.path.basename(fpath))[0]
        task_dict[task_name] = {"images": [], "actions": []}

        with h5py.File(fpath, "r") as f:
            if "problem_info" in f.attrs:
                info  = json.loads(f.attrs["problem_info"])
                instr = info.get("language_instruction", task_name.replace("_demo", "").replace("_", " "))
            else:
                instr = task_name.replace("_demo", "").replace("_", " ")

            task_dict[task_name]["instruction"] = instr

            for demo_key in f["data"].keys():
                imgs = f["data"][demo_key]["obs"]["agentview_rgb"][:]
                acts = f["data"][demo_key]["actions"][:]

                for start in range(0, len(acts) - CHUNK_SIZE + 1):
                    act_chunk = acts[start:start + CHUNK_SIZE]
                    if len(act_chunk) == CHUNK_SIZE:
                        task_dict[task_name]["images"].append(imgs[start])
                        task_dict[task_name]["actions"].append(act_chunk)
    return task_dict


@torch.no_grad()
def embed_openvla(vla_model, processor, images_np, instructions, batch_size=2, device=DEVICE, layer_idx=-1, num_fusion_layers=1):
    all_embs = []
    
    # Locate requested transformer layers
    if num_fusion_layers > 1:
        total_layers = len(vla_model.language_model.model.layers)
        step = total_layers // (num_fusion_layers + 1)
        target_indices = [step * (i + 1) - 1 for i in range(num_fusion_layers)]
        target_layers = [vla_model.language_model.model.layers[idx] for idx in target_indices]
        print(f"🔥 Extracting fusion layers at indices: {target_indices}")
    else:
        target_layers = [vla_model.language_model.model.layers[layer_idx]]
        
    captured = {i: None for i in range(len(target_layers))}
    handles = []

    def make_hook(i):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[i] = hs.float().detach().cpu()
        return hook_fn
        
    for i, target_layer in enumerate(target_layers):
        handles.append(target_layer.register_forward_hook(make_hook(i)))
    
    for i in tqdm.tqdm(range(0, len(images_np), batch_size), desc="OpenVLA embed"):
        batch_imgs = [Image.fromarray(img) for img in images_np[i:i+batch_size]]
        batch_instrs = instructions[i:i+batch_size]
        prompts = [f"In: {ins}\nOut: " for ins in batch_instrs]
        
        inputs = processor(
            text=prompts, images=batch_imgs,
            padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        
        # Cast images to bfloat16 to match OpenVLA native dtype
        if "pixel_values" in inputs:
            inputs = processor(text=prompts, images=batch_imgs, return_tensors="pt", padding=True).to(device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            outputs = vla_model(
                **inputs,
                output_hidden_states=False,
                return_dict=True
            )
            
        layer_embs = []
        for j in range(len(target_layers)):
            hs = captured[j]
            # Extract last token of the instruction (causal reasoning token)
            last_tok_idx = (inputs.attention_mask.sum(dim=1) - 1).cpu()
            emb = hs[torch.arange(len(prompts)), last_tok_idx] # [B, 4096]
            layer_embs.append(emb)
            captured[j] = None
            
        if num_fusion_layers > 1:
            stacked_emb = torch.stack(layer_embs, dim=1) # [B, num_fusion_layers, 4096]
            all_embs.append(stacked_emb)
        else:
            all_embs.append(layer_embs[0])

        torch.cuda.empty_cache()
        
    for handle in handles:
        handle.remove()
    return torch.cat(all_embs, dim=0)


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


@torch.no_grad()
def encode_teacher_vae(vae, actions_tensor, vae_type, device, text_embs, batch_size=256):
    all_mu, all_lv = [], []
    for i in tqdm.tqdm(range(0, len(actions_tensor), batch_size), desc="VAE teacher"):
        a_batch = actions_tensor[i:i+batch_size].to(device)
        
        
        if vae_type == "text_cond_beta_tcvae":
            mu, lv = vae.encode(a_batch)
        else:
            t_batch = text_embs[i:i+batch_size].to(device)
            # mu, lv = vae.encode(a_batch, t_batch) 
            zero_t = torch.zeros(a_batch.size(0), t_batch.shape[-1], device=device)
            mu, lv = vae.encode(a_batch, zero_t)
            
        all_mu.append(mu.float().cpu())
        all_lv.append(lv.float().cpu())
    return torch.cat(all_mu), torch.cat(all_lv)


def load_action_stats(suite):
    stats_path = f"./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt"
    if not os.path.exists(stats_path):
        return None, None, None
    full_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    suite_key  = f"{suite}_no_noops" if f"{suite}_no_noops" in full_stats else "libero_spatial_no_noops"
    stats = full_stats[suite_key]["action"]
    return stats["min"], stats["max"], stats.get("mask", None)


def normalise_actions(actions_np, a_min, a_max, mask):
    actions = torch.from_numpy(actions_np).float()
    if a_min is None: return actions
    a_min_t = torch.tensor(a_min, dtype=torch.float32)
    a_max_t = torch.tensor(a_max, dtype=torch.float32)
    rng     = (a_max_t - a_min_t).clamp(min=1e-6)
    normed  = 2.0 * (actions - a_min_t) / rng - 1.0
    if mask is not None:
        mask_t = torch.tensor(mask, dtype=torch.float32)
        normed = normed * mask_t + actions * (1.0 - mask_t)
    return normed


def main():
    args = parse_args()
    _vae_arch = {"text_cond_beta_tcvae": "tcn", "text_cvae": "cvae"}.get(args.vae_type, "")
    _cache_suffix = f"_arch_{_vae_arch}_beta{args.beta}_z{args.z_dim}"
    out_dir  = os.path.join(args.out_dir, args.suite)
    os.makedirs(out_dir, exist_ok=True)
    if args.num_fusion_layers > 1:
        _layer_suffix = f"_fusion{args.num_fusion_layers}"
    else:
        _layer_suffix = f"_layer{args.vla_layer_idx}" if args.vla_layer_idx != -1 else ""
    _seed_suffix = f"_seed{args.vae_seed}"
    out_path = os.path.join(out_dir, f"vla_emb_cache_{args.vae_type}{_cache_suffix}{_seed_suffix}{_layer_suffix}.pt")

    print(f"📦 Cache will be saved to: {out_path}")

    text_embed_dim_dict = {'smollm': 960, 'octo_t5': 768, 'openvla_llama': 4096, 'clip': 512}
    text_emb_dim = text_embed_dim_dict.get(args.text_backbone, 512)

    # ── 1. Load Data ────────────────────────────────────────────────────
    task_dict = load_suite(args.suite)
    task_names = sorted(task_dict.keys())
    train_tasks = task_names[:TRAIN_SPLIT]
    test_tasks  = task_names[TRAIN_SPLIT:]

    def collect(tasks):
        images, actions, instructions = [], [], []
        for t in tasks:
            images += task_dict[t]["images"]
            actions += task_dict[t]["actions"]
            instructions += [task_dict[t]["instruction"]] * len(task_dict[t]["images"])
        return images, np.array(actions, dtype=np.float32), instructions

    tr_imgs, tr_acts, tr_instrs = collect(train_tasks)
    te_imgs, te_acts, te_instrs = collect(test_tasks)

    a_min, a_max, mask = load_action_stats(args.suite)
    tr_acts_norm = normalise_actions(tr_acts, a_min, a_max, mask)
    te_acts_norm = normalise_actions(te_acts, a_min, a_max, mask)

    # ── 2. Embeddings ───────────────────────────────────────────────────
    if args.emb_cache_from and os.path.exists(args.emb_cache_from):
        print(f"⚡ Reusing VLA/Text embeddings from {args.emb_cache_from}")
        src = torch.load(args.emb_cache_from, map_location="cpu", weights_only=False)
        tr_emb, te_emb = src["train_emb"], src["test_emb"]
        tr_text_embs, te_text_embs = src["train_clip_emb"], src["test_clip_emb"]
    else:
        # ── 2. Load OpenVLA and embed ─────────────────────────────────────────
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"🏗️ Loading OpenVLA from {args.openvla_model}...")
        processor = AutoProcessor.from_pretrained(args.openvla_model, trust_remote_code=True)
        vla_model = AutoModelForVision2Seq.from_pretrained(
            args.openvla_model, 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to(DEVICE)
        vla_model.eval()

        print(f"🧠 Generating OpenVLA embeddings from layer {args.vla_layer_idx}...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            tr_emb = embed_openvla(vla_model, processor, tr_imgs, tr_instrs, args.embed_batch_size, DEVICE, args.vla_layer_idx, args.num_fusion_layers)
            te_emb = embed_openvla(vla_model, processor, te_imgs, te_instrs, args.embed_batch_size, DEVICE, args.vla_layer_idx, args.num_fusion_layers)
            
            # AGGRESSIVE VRAM CLEARING BEFORE TEXT/VAE PASS
            vla_model.cpu()
            del vla_model, processor
            gc.collect()
            torch.cuda.empty_cache()

        tr_text_embs = embed_vla_text_backbone(tr_instrs, args.text_backbone, DEVICE)
        te_text_embs = embed_vla_text_backbone(te_instrs, args.text_backbone, DEVICE)

    # ── 3. Teacher VAE Targets ──────────────────────────────────────────
    if args.text_backbone == "clip":
        vae_checkpoint = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_cyc4_vel0.5_{_vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'
    else:
        vae_checkpoint = f'checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta{args.beta}_z{args.z_dim}_alpha1.0_chunk8_std_text_{args.text_backbone}_seed_{args.vae_seed}_cyc4_vel0.5_{_vae_arch}_seed_{args.vae_seed}_step_{STEP}.pt'

    if args.vae_type == "text_cond_beta_tcvae":
        vae = TCNTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=args.z_dim, text_emb_dim=text_emb_dim, 
            beta=args.beta, dropout=0.15, hidden_channels=64, n_blocks=N_BLOCKS
        ).to(DEVICE)
    else:
        vae = TCNTextActionCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=args.z_dim, text_emb_dim=text_emb_dim, 
            beta=args.beta, dropout=0.15, hidden_channels=64, n_blocks=N_BLOCKS, enc_text_gate_init=0.0
        ).to(DEVICE)

    vae.load_state_dict(torch.load(vae_checkpoint, map_location=DEVICE, weights_only=False))
    vae.eval()

    tr_mu, tr_lv = encode_teacher_vae(vae, tr_acts_norm, args.vae_type, DEVICE, text_embs=tr_text_embs)
    te_mu, te_lv = encode_teacher_vae(vae, te_acts_norm, args.vae_type, DEVICE, text_embs=te_text_embs)

    # ── 4. Save ─────────────────────────────────────────────────────────
    torch.save({
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
        "vla_out_dim":      tr_emb.shape[-1],
    }, out_path)
    
    print(f"✅ Cache saved to {out_path}")

if __name__ == "__main__":
    main()


# 📦 Cache will be saved to: ./checkpoints/projectors/openvla/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128_text_clip_zeroes.pt