#!/usr/bin/env python3
"""
VLA Representation Visualization via t-SNE
===========================================
Extracts intermediate layer representations from pre-trained VLA models
(OpenVLA, SmolVLA, or Pi0) and projects them using t-SNE.
Enables comparing how "entangled" or "disentangled" the object semantic categories
are in the base VLM model before and after robotic control training.

Usage:
    python3 scripts/viz_vla_tsne.py \
        --vla_type openvla \
        --vla_checkpoint openvla/openvla-7b \
        --layers "1,16,31" \
        --mode vision_tokens \
        --selected_classes "milk,orange_juice,alphabet_soup,ketchup,butter"
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import h5py
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HDF5_ROOT = "/mnt/Data/cjimenez/LIBERO/libero/datasets"

# Colors for plotting
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=str, default="libero_object",
                   help="LIBERO suite to extract frames from.")
    p.add_argument("--vla_type", type=str, default="openvla", choices=["openvla", "smolvla", "pi0"],
                   help="VLA architecture model type.")
    p.add_argument("--vla_checkpoint", type=str, default="openvla/openvla-7b",
                   help="Model Hub path or directory to local checkpoint.")
    p.add_argument("--layers", type=str, default="1,16,31",
                   help="Comma-separated list of 1-indexed layer numbers to visualize.")
    p.add_argument("--mode", type=str, default="vision_tokens", choices=["vision_tokens", "last_token"],
                   help="Extraction mode. 'vision_tokens' averages visual patch embeddings in the VLM. "
                        "'last_token' extracts the causal text/action prediction token.")
    p.add_argument("--samples_per_class", type=int, default=50,
                   help="Number of image observations to extract per object class.")
    p.add_argument("--selected_classes", type=str, default="milk,orange_juice,alphabet_soup,ketchup,butter",
                   help="Comma-separated class names/nouns to select.")
    p.add_argument("--out_dir", type=str, default="plots/vla_tsne",
                   help="Directory to save the plotted figures.")
    p.add_argument("--perplexity", type=int, default=15,
                   help="t-SNE perplexity.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_class_dataset(suite, selected_classes, samples_per_class):
    """Loads image observations grouped by their target object classes."""
    hdf5_dir = os.path.join(HDF5_ROOT, f"{suite}_no_noops_hdf5")
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    assert hdf5_files, f"No HDF5 files found at {hdf5_dir}"

    data_by_class = {cls: [] for cls in selected_classes}

    for fpath in hdf5_files:
        task_name = os.path.splitext(os.path.basename(fpath))[0]
        
        # Detect which class this file belongs to
        matched_class = None
        for cls in selected_classes:
            # Match word with underscores or exact match in filename
            if f"_{cls}_" in f"_{task_name}_" or task_name.startswith(cls) or task_name.endswith(cls):
                matched_class = cls
                break
        
        if matched_class is None:
            continue
            
        with h5py.File(fpath, "r") as f:
            if "problem_info" in f.attrs:
                info = json.loads(f.attrs["problem_info"])
                instr = info.get("language_instruction", task_name.replace("_demo", "").replace("_", " "))
            else:
                instr = task_name.replace("_demo", "").replace("_", " ")

            for demo_key in f["data"].keys():
                imgs = f["data"][demo_key]["obs"]["agentview_rgb"][:]  # (T, H, W, 3)
                
                # Sample frames from this demo
                step_sz = max(1, len(imgs) // 5)
                for idx in range(0, len(imgs), step_sz):
                    data_by_class[matched_class].append({
                        "image": imgs[idx],
                        "instruction": instr
                    })

    # Subsample to exact limit per class
    rng = np.random.default_rng(42)
    final_dataset = []
    for cls in selected_classes:
        samples = data_by_class[cls]
        if len(samples) > samples_per_class:
            chosen_indices = rng.choice(len(samples), size=samples_per_class, replace=False)
            samples = [samples[i] for i in chosen_indices]
        
        for s in samples:
            s["class"] = cls
            final_dataset.append(s)
            
    print(f"📊 Loaded dataset. Samples per class:")
    for cls in selected_classes:
        n_samples = sum(1 for s in final_dataset if s["class"] == cls)
        print(f"  - {cls}: {n_samples} samples")
        
    return final_dataset


# ── Feature Extraction Hooks & Forward Runs ───────────────────────────────────
class VLAFeatureExtractor:
    def __init__(self, vla_type, checkpoint, device):
        self.vla_type = vla_type
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.vla_type == "openvla":
            from transformers import AutoModelForVision2Seq, AutoProcessor
            print(f"🏗️  Loading OpenVLA ({self.checkpoint}) in bfloat16...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
            self.model.eval()
            for p in self.model.parameters(): p.requires_grad = False
            
        elif self.vla_type == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            print(f"🏗️  Loading SmolVLA ({self.checkpoint})...")
            policy = SmolVLAPolicy.from_pretrained(self.checkpoint).to(self.device).eval()
            self.policy = policy
            self.model = policy.model.vlm_with_expert
            self.processor = self.model.processor
            for p in self.model.parameters(): p.requires_grad = False

        elif self.vla_type == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            from transformers import AutoTokenizer
            print(f"🏗️  Loading Pi0 ({self.checkpoint})...")
            policy = PI0Policy.from_pretrained(self.checkpoint).to(self.device).eval()
            self.policy = policy
            self.model = policy.model.paligemma_with_expert.paligemma
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            for p in self.model.parameters(): p.requires_grad = False

    def extract_features(self, samples, layers, mode):
        """Extracts features for all samples at specified layers."""
        # Convert layers to 0-indexed indices
        # Layers is 1-indexed to keep user-friendly interface
        
        all_features = {layer: [] for layer in layers}
        
        # We hook into language model layers
        captured_states = {}
        handles = []
        
        def make_hook(layer_num):
            def hook_fn(module, input, output):
                # output can be tuple of (hidden_states, attention, etc.)
                hs = output[0] if isinstance(output, tuple) else output
                captured_states[layer_num] = hs.float().detach().cpu()
            return hook_fn

        # Register hooks
        for layer_num in layers:
            idx = layer_num - 1 # 0-indexed
            if self.vla_type == "openvla":
                target_module = self.model.language_model.model.layers[idx]
                handles.append(target_module.register_forward_hook(make_hook(layer_num)))
            elif self.vla_type == "smolvla":
                if hasattr(self.model.vlm.model, "text_model") and hasattr(self.model.vlm.model.text_model, "layers"):
                    target_module = self.model.vlm.model.text_model.layers[idx].input_layernorm
                else:
                    target_module = self.model.vlm.model.layers[idx].input_layernorm
                
                def make_smolvla_pre_hook(l_num):
                    def hook_fn(module, args):
                        hs = args[0]
                        captured_states[l_num] = hs.float().detach().cpu()
                    return hook_fn
                
                handles.append(target_module.register_forward_pre_hook(make_smolvla_pre_hook(layer_num)))
            elif self.vla_type == "pi0":
                target_module = self.model.model.language_model.layers[idx].input_layernorm
                
                def make_pi0_pre_hook(l_num):
                    def hook_fn(module, args):
                        hs = args[0]
                        captured_states[l_num] = hs.float().detach().cpu()
                    return hook_fn
                
                handles.append(target_module.register_forward_pre_hook(make_pi0_pre_hook(layer_num)))

        # Process samples one-by-one or in very small batches to avoid OOM
        for sample in tqdm.tqdm(samples, desc=f"Extracting VLA features"):
            image_pil = Image.fromarray(sample["image"])
            instruction = sample["instruction"]

            # Clear state dict
            captured_states.clear()

            if self.vla_type == "openvla":
                prompt = f"In: {instruction}\nOut: "
                inputs = self.processor(text=[prompt], images=[image_pil], return_tensors="pt").to(self.device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
                with torch.no_grad():
                    _ = self.model(**inputs, output_hidden_states=False, return_dict=True)
                
                # Image token processing
                img_token_id = getattr(self.model.config, "image_token_index", getattr(self.model.config, "image_token_id", None))
                
                for layer_num in layers:
                    hs = captured_states[layer_num] # (1, SeqLen, D)
                    if mode == "vision_tokens" and img_token_id is not None:
                        img_mask = (inputs["input_ids"] == img_token_id).unsqueeze(-1).float().cpu() # (1, SeqLen, 1)
                        vision_tokens = hs * img_mask
                        n_img = img_mask.sum(dim=1).clamp(min=1)
                        emb = (vision_tokens.sum(dim=1) / n_img)[0] # (D,)
                    else: # last_token mode
                        last_tok_idx = inputs.attention_mask.sum(dim=1) - 1
                        emb = hs[0, last_tok_idx[0]] # (D,)
                    all_features[layer_num].append(emb.numpy())

            elif self.vla_type == "smolvla":
                # SmolVLA using its policy model helper
                from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
                
                # Language
                lang_enc = self.processor.tokenizer(
                    [instruction], return_tensors="pt", padding=True,
                    truncation=True, max_length=48,
                )
                lang_tokens = lang_enc.input_ids.to(self.device)
                lang_masks  = lang_enc.attention_mask.to(self.device).bool()

                # Image
                _resize = getattr(self.policy.config, "resize_imgs_with_padding", None) or (256, 256)
                _tgt_h, _tgt_w = _resize
                img_t = torch.from_numpy(
                    np.array(image_pil.resize((_tgt_w, _tgt_h)))
                ).float() / 255.0
                img_t = img_t.permute(2, 0, 1) * 2.0 - 1.0          # CHW, [-1, 1]
                images_tensor = img_t.unsqueeze(0).to(self.device, dtype=next(self.model.parameters()).dtype)  # (1, 3, H, W)
                img_masks = torch.ones(1, dtype=torch.bool, device=self.device)

                # State
                state = torch.zeros(1, self.policy.model.config.max_state_dim, device=self.device, dtype=self.policy.model.state_proj.weight.dtype)

                with torch.no_grad():
                    # Prefix embeddings
                    prefix_embs, prefix_pad, prefix_att = self.policy.model.embed_prefix(
                        images=[images_tensor],
                        img_masks=[img_masks],
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=state,
                    )
                    att_2d  = make_att_2d_masks(prefix_pad, prefix_att)
                    pos_ids = torch.cumsum(prefix_pad, dim=1) - 1

                    _ = self.model.forward(
                        attention_mask=att_2d,
                        position_ids=pos_ids,
                        past_key_values=None,
                        inputs_embeds=[prefix_embs, None],
                        use_cache=True,
                    )

                n_img_tokens = prefix_pad[0].sum().item() - lang_masks[0].sum().item() - 1
                for layer_num in layers:
                    hs = captured_states[layer_num] # (1, SeqLen, D)
                    if mode == "vision_tokens":
                        emb = hs[0, :n_img_tokens].mean(dim=0)
                    else:
                        emb = hs[0, -1]
                    all_features[layer_num].append(emb.numpy())

            elif self.vla_type == "pi0":
                # Pi0 (PaliGemma-based)
                img_t = torch.from_numpy(np.array(image_pil)).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                img_t = F.interpolate(img_t, size=(480, 640), mode='bilinear', align_corners=False)
                img_t = img_t.to(next(self.model.parameters()).dtype)
                
                # Tokenize prompt
                tokens = self.tokenizer([instruction], padding="max_length", max_length=48, truncation=True, return_tensors="pt")
                lang_tokens = tokens["input_ids"].to(self.device)
                lang_mask = tokens["attention_mask"].bool().to(self.device)
                
                dummy_inputs = {
                    "observation.images.camera0": img_t,
                    "observation.images.camera1": torch.zeros_like(img_t),
                    "observation.images.camera2": torch.zeros_like(img_t),
                    "observation.language.tokens": lang_tokens,
                    "observation.language.attention_mask": lang_mask,
                    "observation.state": torch.zeros((1, 14), dtype=torch.float32, device=self.device),
                    "action": torch.zeros((1, 50, 14), dtype=torch.float32, device=self.device),
                    "task": instruction
                }

                # Forward pass
                with torch.no_grad():
                    _ = self.policy(dummy_inputs)

                # PaliGemma has visual tokens mapped first or concatenated
                # Visual tokens count = 256 for standard PaliGemma
                n_img_tokens = 256
                for layer_num in layers:
                    hs = captured_states[layer_num] # (1, SeqLen, D)
                    if mode == "vision_tokens":
                        # Standard PaliGemma puts visual prefix tokens first in the sequence
                        emb = hs[0, :n_img_tokens].mean(dim=0)
                    else:
                        emb = hs[0, -1]
                    all_features[layer_num].append(emb.numpy())

        # Cleanup hooks
        for handle in handles:
            handle.remove()
            
        # Convert list of arrays to stacked numpy arrays
        for layer_num in layers:
            all_features[layer_num] = np.stack(all_features[layer_num], axis=0)
            
        return all_features


# ── Main Execution ────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selected_classes = [c.strip() for c in args.selected_classes.split(",")]
    layers = [int(l.strip()) for l in args.layers.split(",")]
    
    # 1. Load HDF5 observations
    dataset = load_class_dataset(args.suite, selected_classes, args.samples_per_class)
    labels = np.array([s["class"] for s in dataset])

    # 2. Extract model features
    extractor = VLAFeatureExtractor(args.vla_type, args.vla_checkpoint, DEVICE)
    features_by_layer = extractor.extract_features(dataset, layers, args.mode)

    # 3. Create the t-SNE plot grid
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5.5))
    if n_layers == 1:
        axes = [axes]

    for ax, layer_num in zip(axes, layers):
        print(f"📉 Computing t-SNE for layer {layer_num}...")
        Z = features_by_layer[layer_num]
        
        # Fit t-SNE
        tsne = TSNE(n_components=2, perplexity=min(args.perplexity, len(Z) - 1), 
                    random_state=args.seed, init="pca", learning_rate="auto")
        Z_2d = tsne.fit_transform(Z)

        # Plot scatter points
        for c_idx, cls in enumerate(selected_classes):
            mask = labels == cls
            color = CLASS_COLORS[c_idx % len(CLASS_COLORS)]
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=color, label=cls, alpha=0.75, s=20)
            
        ax.set_title(f"Layer {layer_num}", fontsize=14, fontweight="bold")
        ax.axis("off")

    # Add single legend below subplots
    handles = [mpatches.Patch(color=CLASS_COLORS[i % len(CLASS_COLORS)], label=cls) 
               for i, cls in enumerate(selected_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(selected_classes), 
               fontsize=11, bbox_to_anchor=(0.5, -0.05))

    title_model_name = os.path.basename(args.vla_checkpoint.rstrip("/"))
    fig.suptitle(f"t-SNE of {args.vla_type.upper()} ({title_model_name}) {args.mode.replace('_', ' ').title()} space\nDataset: {args.suite.upper()}", 
                 fontsize=16, fontweight="bold", y=1.05)
    
    # Save the output image
    out_filename = f"tsne_{args.vla_type}_{title_model_name}_{args.mode}.png"
    out_path = os.path.join(args.out_dir, out_filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"🎉 SUCCESS! Saved t-SNE plot to {out_path}")

if __name__ == "__main__":
    main()
