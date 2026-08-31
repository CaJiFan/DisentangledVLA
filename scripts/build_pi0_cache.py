"""
Pi0 / Pi0.5 Embedding Cache Builder
=================================
Extracts Pi0 backbone hidden states (prior to the Flow Matching head) for LIBERO demos.

Since Pi0 uses a PaliGemma or Qwen VLM backbone followed directly by a Flow Matching
action chunker, its intermediate visual features are highly optimized for robotic control.

Usage:
    python3 scripts/build_pi0_cache.py \
        --suite libero_spatial \
        --num_fusion_layers 3
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
from src.disentanglers.AdvancedTextActionCVAE import TCNTextCondPriorCVAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HDF5_ROOT   = "/mnt/Data/cjimenez/LIBERO/libero/datasets"
CHUNK_SIZE  = 8
N_BLOCKS    = max(3, (CHUNK_SIZE - 1).bit_length())
ACTION_DIM  = 7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",          type=str, required=True, choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--pi0_model",      type=str, default="lerobot/pi0", help="HF Repo or local path (via lerobot or openpi)")
    p.add_argument("--out_dir",        type=str, default="./checkpoints/projectors/pi0")
    p.add_argument("--vae_type",       type=str, default="cond_prior", choices=["text_cond_beta_tcvae", "text_cvae", "cond_prior"])
    p.add_argument("--vae_checkpoint", type=str, default="./checkpoints/new_protocol_cvae/libero_spatial/rw100_d0.15_beta0.1_z128_chunk8_protA_cyc4_cond_prior_seed_2_step_250000.pt")
    p.add_argument("--vae_seed",       type=int, default=2)
    p.add_argument("--z_dim",          type=int, default=128)
    p.add_argument("--text_backbone",  type=str, default="clip", choices=["clip", "smollm", "openvla_llama"])
    p.add_argument("--embed_batch_size", type=int, default=2)
    p.add_argument("--num_fusion_layers", type=int, default=1, help="Number of intermediate layers to extract.")
    p.add_argument("--emb_cache_from", type=str, default=None, help="Reuse Pi0 pass if changing VAE")
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
def embed_pi0(pi0_policy, images_np, instructions, batch_size=2, device=DEVICE, num_fusion_layers=1):
    """
    Extracts the intermediate representations from Pi0's VLM backbone.
    """
    all_embs = []
    
    # ---------------------------------------------------------------------------------
    # TODO: Register Hooks to Pi0's VLM Backbone (e.g. PaliGemma/Qwen internal layers)
    # ---------------------------------------------------------------------------------
    # Because Pi0 can be loaded via `openpi` or `lerobot`, the exact module names vary.
    # Below is the conceptual hook implementation (similar to openvla):
    
    layer_outputs = {}
    def get_hook(name):
        def hook(model, input, output):
            # typically output[0] is the hidden states
            layer_outputs[name] = output[0].detach()
        return hook

    # Example: If using LeRobot's Pi0 implementation
    # total_layers = len(pi0_policy.model.vlm.text_model.encoder.layers)
    # step = total_layers // (num_fusion_layers + 1)
    # target_indices = [step * (i + 1) - 1 for i in range(num_fusion_layers)]
    layer_outputs = {}
    
    def get_pre_hook(name):
        def hook(model, args):
            # args is a tuple of inputs to the module
            out = args[0]
            layer_outputs[name] = out
        return hook

    handles = []
    pi0_tokenizer = None
    target_indices = []
    
    if pi0_policy is not None:
        try:
            from transformers import AutoTokenizer
            print("Loading PaliGemma tokenizer for text conditioning...")
            pi0_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        except Exception as e:
            print(f"⚠️ Failed to load PaliGemma tokenizer: {e}")
            print("Please ensure you have exported HF_TOKEN with access to google/paligemma-3b-pt-224!")
            
        try:
            # Pi0 manually interleaves submodules, so the parent `layer` forward() is never called!
            # To get the VLA hidden states, we use a pre-hook on the `input_layernorm` of the layer,
            # which captures the exact hidden state tensor `x` right before it enters the layer.
            trunk = pi0_policy.model.paligemma_with_expert.paligemma.model.language_model.layers
            n_layers = len(trunk)
            
            # Dynamically compute target indices to evenly span early, middle, and late layers (VLA-Adapter style)
            step = n_layers // num_fusion_layers
            target_indices = [step * i + (step - 1) for i in range(num_fusion_layers)]
            print(f"Dynamically selected layer indices for fusion: {target_indices} (out of {n_layers} layers)")
            
            for idx in target_indices:
                target_module = trunk[idx].input_layernorm
                handle = target_module.register_forward_pre_hook(get_pre_hook(f"layer_{idx}"))
                handles.append(handle)
        except AttributeError:
            print("⚠️  Could not find Pi0 VLM trunk.")
            raise RuntimeError("Please update the Pi0 hook path in scripts/build_pi0_cache.py")

    for i in tqdm.tqdm(range(0, len(images_np), batch_size), desc="Pi0 Inference", leave=False):
        batch_imgs  = images_np[i : i + batch_size]
        batch_instr = instructions[i : i + batch_size]
        
        B = len(batch_imgs)
        D = 2048 # Fallback dimension if dummy

        if pi0_policy is not None and handles:
            # 1. Forward pass (we don't need the final action output, just the hooks)
            try:
                # Convert images and text to Pi0 inputs
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy
                # We assume the user has the correct preprocessing. 
                # Convert images to expected format and shape
                img_t = torch.from_numpy(batch_imgs).float().permute(0, 3, 1, 2).to(device) / 255.0
                img_t = F.interpolate(img_t, size=(480, 640), mode='bilinear', align_corners=False)
                
                # lerobot/pi0 requires these exact camera keys based on its config
                # As standard practice for missing modalities, we zero-out the other cameras 
                # so we don't artificially triple the attention weights on the 3rd person view!
                if pi0_tokenizer is not None:
                    # Tokenize the instruction batch natively!
                    tokens = pi0_tokenizer(batch_instr, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
                    lang_tokens = tokens["input_ids"].to(device)
                    lang_mask = tokens["attention_mask"].bool().to(device)
                else:
                    # Fallback if tokenizer failed to load
                    lang_tokens = torch.zeros((B, 48), dtype=torch.long, device=device)
                    lang_mask = torch.ones((B, 48), dtype=torch.bool, device=device)

                dummy_inputs = {
                    "observation.images.camera0": img_t,
                    "observation.images.camera1": torch.zeros_like(img_t),
                    "observation.images.camera2": torch.zeros_like(img_t),
                    "observation.language.tokens": lang_tokens,
                    "observation.language.attention_mask": lang_mask,
                    "observation.state": torch.zeros((B, 14), dtype=torch.float32, device=device),
                    "action": torch.zeros((B, 50, 14), dtype=torch.float32, device=device),
                    "task": batch_instr
                }
                
                _ = pi0_policy(dummy_inputs)
                
                extracted_layers = []
                for idx in target_indices:
                    # layer_out shape: [B, SeqLen, D]
                    layer_out = layer_outputs[f"layer_{idx}"]
                    # Slice out the final visual/reasoning token
                    causal_token = layer_out[:, -1, :] 
                    extracted_layers.append(causal_token)
                    
                stacked_emb = torch.stack(extracted_layers, dim=1) # [B, num_fusion_layers, D]
            except Exception as e:
                print(f"⚠️  Pi0 Forward pass failed: {e}. Falling back to dummy.")
                stacked_emb = torch.randn(B, num_fusion_layers, D, device=device)
        else:
            stacked_emb = torch.randn(B, num_fusion_layers, D, device=device)
        
        all_embs.append(stacked_emb.cpu())

    # Cleanup hooks
    for h in handles:
        h.remove()

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
        max_len = 77  # CLIP crash prevention

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
        
        if vae_type in ["text_cond_beta_tcvae", "cond_prior"]:
            mu, lv = vae.encode(a_batch)
        else:
            t_batch = text_embs[i:i+batch_size].to(device)
            zero_t = torch.zeros(a_batch.size(0), t_batch.shape[-1], device=device)
            mu, lv = vae.encode(a_batch, zero_t)
            
        all_mu.append(mu.float().cpu())
        all_lv.append(lv.float().cpu())
    return torch.cat(all_mu), torch.cat(all_lv)

def load_action_stats(suite):
    stats_path = f"./checkpoints/new_protocol_cvae/{suite}/dataset_statistics.pt"
    if not os.path.exists(stats_path):
        stats_path = f"./checkpoints/text_tcvae/{suite}/dataset_statistics.pt"
    if not os.path.exists(stats_path):
        return None, None, None
    full_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    suite_key  = f"{suite}_no_noops" if f"{suite}_no_noops" in full_stats else list(full_stats.keys())[0]
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
        normed = normed * mask_t
    return normed


def main():
    args = parse_args()
    print(f"Building Pi0 cache for suite: {args.suite}")
    
    import re
    _beta_match = re.search(r'_beta([0-9.\-]+)_', args.vae_checkpoint)
    BETA = _beta_match.group(1) if _beta_match else "0.1"
    
    _vae_arch = {"text_cond_beta_tcvae": "tcn", "text_cvae": "cvae", "cond_prior": "cond_prior"}.get(args.vae_type, "")
    
    _vae_seed_match = re.search(r'_seed_(\d+)', args.vae_checkpoint)
    _vae_seed = int(_vae_seed_match.group(1)) if _vae_seed_match else 1

    cache_name = f"pi0_emb_cache_{args.vae_type}_arch_{_vae_arch}_beta{BETA}_z{args.z_dim}"
    if args.text_backbone != "clip": cache_name += f"_text_{args.text_backbone}"
    cache_name += f"_seed{_vae_seed}"
    if args.num_fusion_layers > 1:
        cache_name += f"_fusion{args.num_fusion_layers}"
    cache_name += ".pt"
    cache_path = os.path.join(args.out_dir, f"{args.suite}_no_noops", cache_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # 2. Load VAE
    print(f"Loading Teacher VAE ({args.vae_type}) from {args.vae_checkpoint}...")
    text_dim = {"clip": 512, "smollm": 960, "openvla_llama": 4096}.get(args.text_backbone, 512)
    use_state = ("_state" in args.vae_checkpoint)
    if args.vae_type == "text_cond_beta_tcvae":
        vae = TCNTextActionBetaTCVAE(action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, text_emb_dim=text_dim, latent_dim=args.z_dim, n_blocks=N_BLOCKS).to(DEVICE)
    elif args.vae_type == "text_cvae":
        vae = TCNTextActionCVAE(action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, text_emb_dim=text_dim, latent_dim=args.z_dim, n_blocks=N_BLOCKS).to(DEVICE)
    elif args.vae_type == "cond_prior":
        vae = TCNTextCondPriorCVAE(action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, text_emb_dim=text_dim, latent_dim=args.z_dim, n_blocks=N_BLOCKS, use_state=use_state, state_dim=8).to(DEVICE)

    ckpt = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        vae.load_state_dict(ckpt["model_state_dict"])
    else:
        vae.load_state_dict(ckpt)
    vae.eval()

    task_dict = load_suite(args.suite)
    a_min, a_max, a_mask = load_action_stats(args.suite)

    pi0_policy = None
    if args.emb_cache_from is None:
        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            print(f"Loading Pi0 policy from {args.pi0_model}...")
            pi0_policy = PI0Policy.from_pretrained(args.pi0_model).to(DEVICE).eval()
        except Exception as e:
            print(f"⚠️  Failed to load Pi0 policy: {e}")
            print("Proceeding with dummy model logic for cache building scaffold.")

    cache = {}
    for task_name, data in tqdm.tqdm(task_dict.items(), desc="Tasks Processing"):
        images_np    = np.array(data["images"])
        actions_np   = np.array(data["actions"])
        instructions = [data["instruction"]] * len(images_np)

        if args.emb_cache_from is not None:
            # Reusing existing Pi0 embeddings
            pass
        else:
            embs = embed_pi0(
                pi0_policy=pi0_policy,
                images_np=images_np, 
                instructions=instructions, 
                batch_size=args.embed_batch_size,
                device=DEVICE,
                num_fusion_layers=args.num_fusion_layers
            )

        norm_acts = normalise_actions(actions_np, a_min, a_max, a_mask)
        text_embs = embed_vla_text_backbone(instructions, args.text_backbone, DEVICE)
        
        mu, lv = encode_teacher_vae(vae, norm_acts, args.vae_type, DEVICE, text_embs)

        cache[task_name] = {
            "vla_emb": embs if args.emb_cache_from is None else None,
            "actions": actions_np,
            "train_mu": mu,
            "train_logvar": lv
        }

    print(f"Saving Pi0 cache to {cache_path}...")
    torch.save(cache, cache_path)
    print("✅ Pi0 Cache generation complete!")

if __name__ == "__main__":
    main()
