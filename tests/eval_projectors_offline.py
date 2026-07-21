import os
import re
import glob
import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import argparse
import gc
import sys

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ban TensorFlow from stealing GPU VRAM
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass

from utils.data import get_vla_projector_dataloader_cached
from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE
from src.projectors import ProbabilisticActionProjector

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Multi-VLA Projector Evaluator")
    # Base dir is now the root projectors folder
    parser.add_argument("--base_proj_dir", type=str, default="checkpoints/projectors/", help="Base folder for projectors")
    parser.add_argument("--base_vae_dir", type=str, default="checkpoints/text_tcvae/libero_spatial/", help="Base folder for VAEs")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--normalize_emb", action="store_true", default=True)
    return parser.parse_args()

@torch.no_grad()
def evaluate_projector(projector, vae, dataloader, device, normalize_emb):
    projector.eval()
    vae.eval()
    
    total_sum_sq_err_mu = 0.0
    total_sum_sq_err_act = 0.0
    
    total_elements_mu = 0
    total_elements_act = 0

    for vla_emb, target_mu, target_logvar, text_emb, gt_actions in dataloader:
        vla_emb = vla_emb.to(device)
        target_mu = target_mu.to(device)
        text_emb = text_emb.to(device)
        gt_actions = gt_actions.to(device)
        
        if normalize_emb:
            vla_emb = F.normalize(vla_emb, dim=-1)

        # 1. Projector Forward (extract mu for deterministic alignment)
        _, pred_mu, _ = projector(vla_emb)
        
        # 2. VAE Decode 
        pred_actions = vae.decode(pred_mu, text_emb)
        
        # Isolate continuous dimensions
        pred_cont = pred_actions[..., :6]
        gt_cont = gt_actions[..., :6]

        # Calculate Sum Squared Errors (Matches F.mse_loss reduction="mean")
        total_sum_sq_err_mu += F.mse_loss(pred_mu, target_mu, reduction="sum").item()
        total_sum_sq_err_act += F.mse_loss(pred_cont, gt_cont, reduction="sum").item()
        
        total_elements_mu += pred_mu.numel()
        total_elements_act += pred_cont.numel()

    # Exact Means
    mse_mu = total_sum_sq_err_mu / total_elements_mu
    action_mse = total_sum_sq_err_act / total_elements_act
    
    return mse_mu, action_mse

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dim_map = {"smollm": 960, "octo_t5": 768, "openvla_llama": 4096, "clip": 512}
    
    # Map each VLA to its underlying text backbone
    vla_configs = {
        "openvla": "clip",
        "octo": "clip",
        "smolvla": "clip",
    }
    
    z_configs = [64, 128]
    all_results = []

    for vla_type, text_backbone in vla_configs.items():
        text_emb_dim = dim_map[text_backbone]
        
        for z_dim in z_configs:
            print("\n" + "="*70)
            print(f"🚀 INITIATING EVALUATION: {vla_type.upper()} | Z_DIM = {z_dim}")
            print("="*70)
            
            # --- 1. Architecture & Cache Logic ---
            if z_dim == 64:
                vae_type = "text_cvae"
                cond_pretty = "Fully Cond."
                arch_str = "cvae"
                seed = 1
            else:
                vae_type = "text_cond_beta_tcvae"
                cond_pretty = "Decoder-Only"
                arch_str = "tcn"
                seed = 2
        
            cache_file = f"vla_emb_cache_{vae_type}_arch_{arch_str}_beta0.001_z{z_dim}_text_{text_backbone}.pt"
            proj_dir = os.path.join(args.base_proj_dir, vla_type, args.suite, f"chunk_8_zdim_{z_dim}")
            cache_path = os.path.join(args.base_proj_dir, vla_type, args.suite, cache_file)
            
            if not os.path.exists(proj_dir):
                print(f"⚠️ Projector directory not found: {proj_dir} (Skipping)")
                continue
                
            if not os.path.exists(cache_path):
                print(f"⚠️ Cache file not found: {cache_path} (Skipping)")
                continue
            
            # --- 2. Find the correct Teacher VAE for this backbone ---
            vae_pattern = os.path.join(args.base_vae_dir, f"*beta0.001_z{z_dim}*std_cyc*{arch_str}*_seed_{seed}_step_100000.pt")
            vae_candidates = glob.glob(vae_pattern)

            print('vae candidates', vae_candidates)

            # print(args.base_vae_dir, vae_candidates)
            
            vae_ckpt = None
            for c in vae_candidates:
                if f"text_{text_backbone}" in os.path.basename(c):
                    vae_ckpt = c
                    break
            
            # Fallback for old models (like CLIP) that might not have "text_" in the name
            if not vae_ckpt and text_backbone == "clip":
                for c in vae_candidates:
                    if "text_" not in os.path.basename(c):
                        vae_ckpt = c
                        break
            
            # Last resort fallback
            if not vae_ckpt and vae_candidates:
                vae_ckpt = vae_candidates[0] 
                
            if not vae_ckpt:
                print(f"❌ Could not find Teacher VAE for Z={z_dim}, Backbone={text_backbone}")
                continue
                
            print(f"📦 Loading Teacher VAE: {os.path.basename(vae_ckpt)}")
            if vae_type == "text_cvae":
                vae = TCNTextActionCVAE(
                    action_dim=7, chunk_size=8, latent_dim=z_dim, text_emb_dim=text_emb_dim,
                    beta=0.001, dropout=0.15, hidden_channels=64, n_blocks=max(3, (8-1).bit_length()), enc_text_gate_init=0.0
                ).to(device)
            else:
                vae = TCNTextActionBetaTCVAE(
                    action_dim=7, chunk_size=8, latent_dim=z_dim, text_emb_dim=text_emb_dim,
                    beta=0.001, dropout=0.15, hidden_channels=64, n_blocks=max(3, (8-1).bit_length())
                ).to(device)
            
            vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
            vae.eval()

            # --- 3. Load DataLoader ---
            print(f"💽 Loading Dataloader from cache: {cache_file}")
            train_dataloader, _ = get_vla_projector_dataloader_cached(
                vla_model=None, processor=None, suite=args.suite,
                batch_size=args.batch_size, embed_batch_size=1,
                cache_path=cache_path, fallback_cache_path=None,
                device=device, vae=vae, vae_type=vae_type, use_vision_pool=False
            )
            
            sample_vla_emb, _, _, _, _ = next(iter(train_dataloader))
            vla_out_dim = sample_vla_emb.shape[-1]
            
            # --- 4. Find Best Projector Checkpoints per Seed ---
            proj_pattern = os.path.join(proj_dir, "prob_mlp_loss_kl_seed_*_step_100000.pt")
            proj_paths = glob.glob(proj_pattern)
            
            regex = re.compile(r"seed_(?P<seed>\d+)_step_(?P<step>\d+)\.pt")
            best_ckpts = {}
            for path in proj_paths:
                match = regex.search(os.path.basename(path))
                if match:
                    seed, step = int(match.group("seed")), int(match.group("step"))
                    if seed not in best_ckpts or step > best_ckpts[seed]["step"]:
                        best_ckpts[seed] = {"step": step, "path": path}

            if not best_ckpts:
                print(f"❌ No projector checkpoints found in {proj_dir}")
                continue

            print(f"🔍 Found final checkpoints for {len(best_ckpts)} seeds.")
            
            # --- 5. Evaluate Projectors ---
            for seed, data in sorted(best_ckpts.items()):
                step = data["step"]
                path = data["path"]
                
                projector = ProbabilisticActionProjector(
                    input_dim=vla_out_dim, latent_dim=z_dim, dropout=0.1, architecture="mlp"
                ).to(device)
                projector.load_state_dict(torch.load(path, map_location=device))
                is_normalized = True if vla_type == "openvla" else False
                mse_mu, action_mse = evaluate_projector(projector, vae, train_dataloader, device, is_normalized)
                
                all_results.append({
                    "vla_type": vla_type, "z_dim": z_dim, "cond_pretty": cond_pretty, 
                    "seed": seed, "step": step, "mse_mu": mse_mu, "action_mse": action_mse  
                })
                print(f"   => Seed {seed} (Step {step}) | MSE_μ: {mse_mu:.6f} | Action MSE: {action_mse:.6f}")
                
                del projector
                torch.cuda.empty_cache()

            del vae, train_dataloader
            gc.collect()
            torch.cuda.empty_cache()

    # --- 6. Aggregate & Print Multi-VLA LaTeX Table ---
    grouped = {}
    for r in all_results:
        key = (r["vla_type"], r["cond_pretty"], r["z_dim"])
        if key not in grouped:
            grouped[key] = {"steps": [], "mse_mu": [], "action_mse": []}
        grouped[key]["steps"].append(r["step"])
        grouped[key]["mse_mu"].append(r["mse_mu"])
        grouped[key]["action_mse"].append(r["action_mse"])

    print("\n\n" + "%" * 70)
    print("% COPY THIS BLOCK DIRECTLY INTO YOUR .TEX FILE")
    print("%" * 70 + "\n")
    
    print(r"\begin{table*}[htb!]")
    print(r"    \centering")
    print(r"    \caption{\textbf{Projector Alignment Fidelity Across VLA Backbones.} Averaged across 3 random seeds on the training set of \texttt{libero\_spatial}. We report the mean and standard deviation ($\pm$) of the Latent KL Divergence and continuous Action Mean Squared Error (MSE).}")
    print(r"    \label{tab:projector_alignment_all}")
    print(r"    \begin{tabular}{@{}llcc|cc@{}}")
    print(r"        \toprule")
    print(r"        \textbf{VLA Backbone} & \textbf{Teacher VAE (Stage 1)} & $\mathbf{Z_{\text{dim}}}$ & \textbf{Max Steps} & \textbf{KL(Teacher $\parallel$ Proj)} $\downarrow$ & \textbf{Action-Space Gap} $\downarrow$ \\")
    print(r"        \midrule")
    
    # Sort logically: VLA Type -> Z_dim
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[2])):
        vla, cond, z = key
        avg_step = int(np.mean(grouped[key]["steps"]))
        
        avg_kl = np.mean(grouped[key]["mse_mu"]) # Reusing variable name but it contains KL now
        std_kl = np.std(grouped[key]["mse_mu"])
        
        avg_act_mse = np.mean(grouped[key]["action_mse"])
        std_act_mse = np.std(grouped[key]["action_mse"])
        
        vla_pretty = {"openvla": "OpenVLA", "octo": "Octo", "smolvla": "SmolVLA"}.get(vla, vla.upper())
        
        print(f"        {vla_pretty} & {cond} & {z} & ~{avg_step} & ${avg_kl:.6f} \\pm {std_kl:.6f}$ & ${avg_act_mse:.6f} \\pm {std_act_mse:.6f}$ \\\\")
        
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()



"""


\begin{table*}[htb!]
    \centering
    \caption{\textbf{Projector Alignment Fidelity Across VLA Backbones.} Averaged across 3 random seeds on the training set of \texttt{libero\_spatial}. We report the mean and standard deviation ($\pm$) of the Latent KL Divergence and continuous Action Mean Squared Error (MSE).}
    \label{tab:projector_alignment_all}
    \begin{tabular}{@{}llcc|cc@{}}
        \toprule
        \textbf{VLA Backbone} & \textbf{Teacher VAE (Stage 1)} & $\mathbf{Z_{\text{dim}}}$ & \textbf{Max Steps} & \textbf{KL(Teacher $\parallel$ Proj)} $\downarrow$ & \textbf{Action-Space Gap} $\downarrow$ \\
        \midrule
        Octo & Fully Cond. & 64 & ~100000 & $0.217979 \pm 0.001162$ & $0.026491 \pm 0.000058$ \\
        Octo & Decoder-Only & 128 & ~100000 & $0.118291 \pm 0.000464$ & $0.023339 \pm 0.000098$ \\
        OpenVLA & Fully Cond. & 64 & ~100000 & $1.094872 \pm 0.023248$ & $0.202695 \pm 0.004706$ \\
        OpenVLA & Decoder-Only & 128 & ~100000 & $0.933299 \pm 0.010491$ & $0.196446 \pm 0.003293$ \\
        SmolVLA & Fully Cond. & 64 & ~100000 & $0.868563 \pm 0.743224$ & $0.102922 \pm 0.090223$ \\
        SmolVLA & Decoder-Only & 128 & ~100000 & $0.451858 \pm 0.591988$ & $0.059606 \pm 0.071404$ \\
        \bottomrule
    \end{tabular}
\end{table*}

"""