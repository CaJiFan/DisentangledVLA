import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.disentanglers.TextActionDecOnlyBetaTCVAE import TextActionBetaTCVAE
from utils.data import get_text_action_ram_cached_dataloader, get_text_action_ram_cached_dataloader2

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_LABELS = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

def load_model(checkpoint_path, beta=1.0, z_dim=16):
    model = TextActionBetaTCVAE(
        action_dim=7, chunk_size=CHUNK_SIZE, latent_dim=z_dim, 
        text_emb_dim=512, beta=beta, dropout=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def plot_latent_traversals(model, dataloader, suite, z_dim=16, sweep_range=(-3, 3), num_steps=5):
    """
    Sweeps each latent dimension while keeping others fixed and plots the decoded physics.
    """
    print("🔬 Running Latent Traversal Analysis...")
    
    # 1. Grab a single representative batch
    actions, text_features = next(iter(dataloader))
    actions = actions.to(DEVICE)
    text_features = text_features.to(DEVICE)
    
    # Use the very first trajectory in the batch as our "Base"
    base_action = actions[0:1]       # Shape: [1, 16, 7]
    base_text = text_features[0:1]   # Shape: [1, 512]
    
    with torch.no_grad():
        # Encode the base trajectory to get the base latent vector
        mu, _ = model.encode(base_action)
        base_z = mu[0].clone()       # Shape: [16]
        
        # We will only plot the top 6 most "active" latent dimensions to save space
        # Active = highest variance across a batch
        batch_mu, _ = model.encode(actions)
        active_dims = torch.var(batch_mu, dim=0).argsort(descending=True)[:6].cpu().numpy()

        fig, axes = plt.subplots(nrows=len(active_dims), ncols=7, figsize=(20, 12))
        sweep_vals = np.linspace(sweep_range[0], sweep_range[1], num_steps)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_steps))

        for row_idx, dim_idx in enumerate(active_dims):
            for step_idx, val in enumerate(sweep_vals):
                # Create a swept z vector
                swept_z = base_z.clone()
                swept_z[dim_idx] = val
                swept_z = swept_z.unsqueeze(0) # [1, 16]
                
                # Decode the swept z back into a physical trajectory
                decoded_action = model.decode(swept_z, base_text)
                decoded_np = decoded_action[0].cpu().numpy() # [16, 7]
                
                # Plot the 7 physical dimensions
                for col_idx in range(7):
                    ax = axes[row_idx, col_idx]
                    ax.plot(decoded_np[:, col_idx], color=colors[step_idx], alpha=0.8, linewidth=2)
                    
                    if step_idx == num_steps - 1: # Formatting on last sweep
                        if row_idx == 0:
                            ax.set_title(ACTION_LABELS[col_idx], fontweight='bold')
                        if col_idx == 0:
                            ax.set_ylabel(f"Latent $z_{{{dim_idx}}}$", fontweight='bold', fontsize=12)
                        
                        ax.set_xticks([]) # Hide x-ticks for cleaner look
                        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Latent Traversals (Beta-TCVAE) - Sweeping from {sweep_range[0]} to {sweep_range[1]}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = f"kinematic_disentanglement_beta{BETA}_z{Z_DIM}_chunk{CHUNK_SIZE}_grid_{suite}.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved Traversal Grid to {out_path}")

import torch
import numpy as np
import matplotlib.pyplot as plt

# ... (Load your trained GUIDED model and dataloader as usual) ...

def plot_guided_traversals(model, dataloader, suite, sweep_range=(-2.0, 2.0), num_steps=5):
    print("🔬 Running Guided Kinematic Traversal...")
    
    actions, text_features = next(iter(dataloader))
    base_action = actions[0:1].to(DEVICE)     # [1, 8, 7]
    base_text = text_features[0:1].to(DEVICE) # [1, 512]
    
    with torch.no_grad():
        mu, _ = model.encode(base_action)
        base_z = mu[0].clone()
        base_decoded = model.decode(base_z.unsqueeze(0), base_text)[0].cpu().numpy()

        # We will explicitly plot the 7 guided dimensions
        fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(20, 15))
        sweep_vals = np.linspace(sweep_range[0], sweep_range[1], num_steps)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_steps))
        
        action_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

        for row_idx in range(7): # Loop strictly over z0 to z15
            for step_idx, val in enumerate(sweep_vals):
                swept_z = base_z.clone()
                # Instead of replacing the value, we ADD to it (a true delta shift)
                swept_z[row_idx] += val 
                
                decoded_np = model.decode(swept_z.unsqueeze(0), base_text)[0].cpu().numpy()
                
                # We plot the DIFFERENCE from the base trajectory (makes disentanglement obvious)
                difference = decoded_np - base_decoded
                
                for col_idx in range(7):
                    ax = axes[row_idx, col_idx]
                    ax.plot(difference[:, col_idx], color=colors[step_idx], alpha=0.8, linewidth=2)
                    
                    if step_idx == num_steps - 1: 
                        if row_idx == 0: ax.set_title(action_labels[col_idx], fontweight='bold')
                        if col_idx == 0: ax.set_ylabel(f"Neuron $z_{row_idx}$ \n({action_labels[col_idx]} Guid.)", fontweight='bold')
                        
                        # Fix Y-axis limits so all subplots are on the same scale
                        # ax.set_ylim([-0.5, 0.5]) 
                        ax.set_xticks([]) 
                        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
                        ax.grid(True, alpha=0.2)

    plt.suptitle("Weakly Supervised Kinematic Disentanglement (Difference from Base)", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = f"kinematic_disentanglement_beta{BETA}_z{Z_DIM}_alpha{ALPHA}_chunk{CHUNK_SIZE}_std_trav7_seed_{SEED}_grid_{suite}.png"

    plt.savefig(out_path, dpi=300)
    print("✅ Saved CoRL Figure!")



if __name__ == "__main__":
    global BETA, Z_DIM, CHUNK_SIZE, ALPHA, SEED
    
    BETA = 0.1
    Z_DIM = 16
    CHUNK_SIZE = 8
    SEED = 1
    ALPHA = 1.0
    
    for suite in ['libero_spatial', 'libero_object',]:
        if CHUNK_SIZE == 8:
            STEP = 80_000
            # CHECKPOINT = f"./checkpoints/text_tcvae/{suite}/{suite}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_chunk{CHUNK_SIZE}_step_{STEP}.pt"
            CHECKPOINT = f"./checkpoints/text_tcvae/{suite}/{suite}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_alpha{ALPHA}_chunk{CHUNK_SIZE}_step_{STEP}.pt"
            if SEED >= 1:
                CHECKPOINT = f"./checkpoints/text_tcvae/{suite}/{suite}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_alpha{ALPHA}_chunk{CHUNK_SIZE}_std_seed_{SEED}_step_{STEP}.pt"

            dataloader, _ = get_text_action_ram_cached_dataloader(suite=suite, batch_size=256)
        elif CHUNK_SIZE == 16:
            STEP = 50_000
            CHECKPOINT = f"./checkpoints/text_tcvae/{suite}/{suite}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_step_{STEP}.pt"
            dataloader, _ = get_text_action_ram_cached_dataloader2(suite=suite, batch_size=256)

        model = load_model(CHECKPOINT, beta=BETA, z_dim=Z_DIM)
        plot_guided_traversals(model, dataloader, suite)