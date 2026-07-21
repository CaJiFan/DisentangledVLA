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

# Ensure local imports work regardless of where the script is run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ban TensorFlow from stealing GPU VRAM
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass

from utils.data import get_text_action_ram_cached_dataloader
from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Universal VAE Checkpoint Evaluator")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/text_tcvae/libero_spatial/", help="Path to checkpoints")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    return parser.parse_args()

@torch.no_grad()
def evaluate_checkpoint(model, dataloader, device):
    model.eval()
    total_cont_mse = 0.0
    total_grip_bce = 0.0
    total_elements_cont = 0
    total_elements_grip = 0

    for actions, text_features in dataloader:
        actions, text_features = actions.to(device), text_features.to(device)
        
        recon_actions, _, _, _ = model(actions, text_features)
        
        # --- Exactly matching your action_recon_loss logic ---
        # 1. Continuous (Dims 0-5)
        pred_cont = recon_actions[..., :6]
        gt_cont = actions[..., :6]
        total_cont_mse += F.mse_loss(pred_cont, gt_cont, reduction="sum").item()
        total_elements_cont += pred_cont.numel()
        
        # 2. Gripper (Dim 6)
        gt_bin = (actions[..., 6] > 0.0).float()
        pred_prob = torch.clamp((recon_actions[..., 6] + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
        total_grip_bce += F.binary_cross_entropy(pred_prob, gt_bin, reduction="sum").item()
        total_elements_grip += gt_bin.numel()

    # Calculate exact Means
    action_mse = total_cont_mse / total_elements_cont
    gripper_bce = total_grip_bce / total_elements_grip
    
    # Your combined metric
    combined_recon_loss = action_mse + (0.5 * gripper_bce)

    return action_mse, gripper_bce, combined_recon_loss

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_map = {"smollm": 960, "octo_t5": 768, "openvla_llama": 4096, "clip": 512}

    print(f"🔍 Scanning directory: {args.ckpt_dir}")
    pattern = os.path.join(args.ckpt_dir, "rw100_dropout0.15_beta0.*std_cyc*")
    ckpt_paths = glob.glob(pattern)
    
    regex_hyperparams = re.compile(r"beta(?P<beta>[\d\.]+)_z(?P<z>\d+)_.*_(?P<model_type>cvae|tcn)_seed_(?P<seed>\d+)_")
    
    ckpts_by_backbone = {}
    
    for path in ckpt_paths:
        filename = os.path.basename(path)
        bb_match = re.search(r"text_(smollm|octo_t5|openvla_llama|clip)", filename)
        backbone = bb_match.group(1) if bb_match else "clip"
        
        match = regex_hyperparams.search(filename)
        if not match: continue
        
        meta = match.groupdict()
        meta["path"] = path
        meta["backbone"] = backbone
        
        if backbone not in ckpts_by_backbone:
            ckpts_by_backbone[backbone] = []
        ckpts_by_backbone[backbone].append(meta)

    raw_results = []

    for backbone, ckpts in ckpts_by_backbone.items():
        print("\n" + "="*50)
        print(f"🚀 Loading Data & Evaluating Backbone: {backbone.upper()} ({len(ckpts)} files)")
        print("="*50)
        
        text_emb_dim = dim_map[backbone]
        
        train_dataloader, _, _ = get_text_action_ram_cached_dataloader(
            suite="libero_spatial",
            batch_size=args.batch_size,
            text_backbone=backbone
        )
        
        pbar = tqdm.tqdm(ckpts, desc=f"Evaluating {backbone}")
        for meta in pbar:
            beta = float(meta['beta'])
            z_dim = int(meta['z'])
            seed = int(meta['seed'])
            model_type = meta['model_type']
            
            cond_label = "Decoder-Only" if model_type == "cvae" else "Fully Cond."
            
            if model_type == "cvae":
                model = TCNTextActionCVAE(
                    action_dim=7, chunk_size=8, latent_dim=z_dim, text_emb_dim=text_emb_dim,
                    beta=beta, dropout=0.15, hidden_channels=64, n_blocks=max(3, (8 - 1).bit_length()), enc_text_gate_init=0.0
                )
            else:
                model = TCNTextActionBetaTCVAE(
                    action_dim=7, chunk_size=8, latent_dim=z_dim, text_emb_dim=text_emb_dim,
                    beta=beta, dropout=0.15, hidden_channels=64, n_blocks=max(3, (8 - 1).bit_length())
                )

            model.load_state_dict(torch.load(meta['path'], map_location=device))
            model.to(device)
            
            # Get exact metrics
            action_mse, grip_bce, combined_recon = evaluate_checkpoint(model, train_dataloader, device)
            
            # Live updates on progress bar
            pbar.set_postfix({"Act MSE": f"{action_mse:.5f}", "Grip BCE": f"{grip_bce:.5f}"})
            
            raw_results.append({
                "backbone": backbone, "cond": cond_label, "beta": beta, "z": z_dim, "seed": seed,
                "mse": action_mse, "bce": grip_bce, "combined": combined_recon
            })
            
            del model
            torch.cuda.empty_cache()

        del train_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate results for LaTeX table
    grouped = {}
    for res in raw_results:
        key = (res["backbone"], res["cond"], res["beta"], res["z"])
        if key not in grouped:
            grouped[key] = {"mse": [], "bce": [], "combined": []}
        grouped[key]["mse"].append(res["mse"])
        grouped[key]["bce"].append(res["bce"])
        grouped[key]["combined"].append(res["combined"])

    print("\n\n" + "%" * 60)
    print("% COPY THIS BLOCK DIRECTLY INTO YOUR .TEX FILE")
    print("%" * 60 + "\n")
    
    print(r"\begin{table*}[htb!]")
    print(r"    \centering")
    print(r"    \caption{\textbf{Quantitative Intrinsic Metrics for VAE Ablations.} Averaged across random seeds on the training set of \texttt{libero\_spatial}. We report the continuous Action Mean Squared Error (MSE), Gripper Binary Cross Entropy (BCE), and the unified Reconstruction Loss.}")
    print(r"    \label{tab:vae_intrinsic_full}")
    print(r"    \begin{tabular}{@{}llcc|ccc@{}}")
    print(r"        \toprule")
    print(r"        \textbf{Text Backbone} & \textbf{Conditioning} & $\boldsymbol{\beta}$ & $\mathbf{Z_{\text{dim}}}$ & \textbf{Action MSE} $\downarrow$ & \textbf{Gripper BCE} $\downarrow$ & \textbf{Total Recon Loss} $\downarrow$ \\")
    print(r"        \midrule")
    
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1], float(x[2]), int(x[3]))):
        backbone, cond, beta, z = key
        avg_mse = np.mean(grouped[key]["mse"])
        avg_bce = np.mean(grouped[key]["bce"])
        avg_comb = np.mean(grouped[key]["combined"])
        
        bb_pretty = {"clip": "CLIP (512-D)", "smollm": "SmolLM (960-D)", "octo_t5": "T5 (768-D)", "openvla_llama": "Llama-2 (4096-D)"}.get(backbone, backbone)
        
        print(f"        {bb_pretty} & {cond} & {beta} & {z} & {avg_mse:.6f} & {avg_bce:.6f} & {avg_comb:.6f} \\\\")
        
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()