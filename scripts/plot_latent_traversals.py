import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
import sys
import os
import tensorflow as tf 
import tqdm
import random

# --- YOUR IMPORTS ---
# Ensure these are pointing to the correct paths
from src.disentanglers.ActionBetaTCVAE import ActionBetaTCVAE
from src.disentanglers.ActionCNNVQVAE import ActionVQVAE 
from utils.data import FastActionRLDSDataset, ActionOnlyTransform

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/mnt/Data/cjimenez/LIBERO/libero/datasets/"
SEED = 42

MODEL_CONFIGS = {
    "beta_tcvae": {
        "class": ActionBetaTCVAE,
        "path": "./checkpoints/disentanglers/beta_tcvae_step_50000.pt",
        "params": {"action_dim": 7, "chunk_size": 16, "latent_dim": 16, "beta": 6.0},
        "is_probabilistic": True
    },
    "vqvae": {
        "class": ActionVQVAE,
        "path": "./checkpoints/disentanglers/vqvae_step_50000.pt",
        # Ensure 'embed_dim' matches your checkpoint (likely 256 based on previous errors)
        "params": {"action_dim": 7, "horizon": 16, "embed_dim": 256, "num_codes": 1024},
        "is_probabilistic": False
    }
}

SUITES = {
    "libero_goal": "Goal Tasks", 
    "libero_spatial": "Spatial Tasks"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed) # CRITICAL for TF Data Loaders
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Global Seed set to {seed}")

def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    cfg = MODEL_CONFIGS[model_name]
    model = cfg["class"](**cfg["params"])
    
    if not os.path.exists(cfg["path"]):
        print(f"❌ Checkpoint not found at {cfg['path']}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(cfg["path"], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded {model_name}")
    return model, cfg

def get_fixed_dataset(samples_per_suite=1000):
    """Loads data ONCE into memory to ensure identical inputs for all models."""
    all_actions = []
    print("💾 Caching fixed validation dataset...")
    
    for label_idx, (suite_key, suite_name) in enumerate(SUITES.items()):
        set_seed(SEED + label_idx) 
        
        ds = FastActionRLDSDataset(
            data_root_dir=DATA_ROOT,
            data_mix=f'{suite_key}_no_noops', 
            batch_transform=ActionOnlyTransform(action_horizon=16),
            resize_resolution=(224, 224),
            train=True 
        )
        iterator = iter(ds)
        
        count = 0
        suite_actions = []
        
        with tqdm.tqdm(total=samples_per_suite, desc=f"Loading {suite_key}") as pbar:
            while count < samples_per_suite:
                try:
                    batch = next(iterator)
                    if isinstance(batch, dict): actions = batch['actions']
                    elif isinstance(batch, (list, tuple)): actions = batch[0]
                    else: actions = batch

                    if isinstance(actions, np.ndarray): actions = torch.tensor(actions).float()
                    else: actions = actions.clone().detach().float()
                    
                    if actions.ndim == 2: actions = actions.unsqueeze(0)
                    
                    suite_actions.append(actions)
                    count += actions.shape[0]
                    pbar.update(actions.shape[0])
                except StopIteration:
                    break
        
        # Stack and crop to exact number
        all_actions.append(torch.cat(suite_actions, dim=0)[:samples_per_suite])
        
    return torch.cat(all_actions, dim=0)

def get_latents_from_fixed(model, actions_tensor, config):
    """Extracts latents from the fixed tensor for Correlation Matrix"""
    print(f"   -> Extracting latents for correlation matrix...")
    all_zs = []
    batch_size = 256
    total = actions_tensor.shape[0]
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            actions = actions_tensor[i : i+batch_size].to(DEVICE)
            
            if config["is_probabilistic"]:
                # TCVAE
                mu, _ = model.encode(actions)
                z = mu
            else:
                # VQ-VAE
                z = model.encoder(actions.permute(0, 2, 1))
                if isinstance(z, tuple): z = z[0]
                if z.ndim == 3: z = z.mean(dim=-1) # Mean pool over time
                
            all_zs.append(z.cpu().numpy())
            
    return np.concatenate(all_zs, axis=0)

def plot_traversals(model, config, sample_action, model_name):
    print(f"   -> Generating Latent Traversals for {model_name}...")
    
    # 1. Encode single sample to get base_z
    with torch.no_grad():
        actions = sample_action.to(DEVICE)
        if config["is_probabilistic"]:
            mu, _ = model.encode(actions)
            base_z = mu[0] # (Latent_Dim,)
        else:
            # VQVAE Encode
            z = model.encoder(actions.permute(0, 2, 1))
            if isinstance(z, tuple): z = z[0]
            # Handle Temporal Dim for VQVAE
            if z.ndim == 3: base_z = z.mean(dim=-1)[0]
            else: base_z = z[0]

    dims_to_sweep = min(5, base_z.shape[0])
    sweep_steps = 7
    sweep_range = torch.linspace(-3, 3, sweep_steps).to(DEVICE)
    
    fig, axes = plt.subplots(dims_to_sweep, 4, figsize=(15, 2 * dims_to_sweep), sharex=True)
    col_titles = ["End-Effector X", "End-Effector Y", "End-Effector Z", "Gripper"]
    action_indices = [0, 1, 2, -1] 
    colors = plt.cm.viridis(np.linspace(0, 1, sweep_steps))

    for row_idx in range(dims_to_sweep):
        # Create batch of z's with one dim swept
        z_batch = base_z.repeat(sweep_steps, 1)
        z_batch[:, row_idx] = sweep_range
        
        # Decode
        with torch.no_grad():
            if config["is_probabilistic"]:
                decoded = model.decode(z_batch)
            else:
                # VQVAE Manual Decode
                # Expand z for quantizer if needed (Batch, Channels, 1)
                z_in = z_batch.unsqueeze(-1) if hasattr(model, 'encoder') and model.encoder[0].weight.ndim == 3 else z_batch
                z_q, _, _ = model.quantizer(z_in)
                decoded = model.decoder(z_q)
                # Permute back (Batch, Time, Channels) if needed
                if decoded.shape[1] != 16: decoded = decoded.permute(0, 2, 1)

            decoded = decoded.cpu().numpy()

        # Plotting
        for col_idx, act_idx in enumerate(action_indices):
            ax = axes[row_idx, col_idx]
            for step_i in range(sweep_steps):
                ax.plot(decoded[step_i, :, act_idx], color=colors[step_i], linewidth=1.5)
            
            if row_idx == 0: ax.set_title(col_titles[col_idx], fontweight='bold')
            if col_idx == 0: ax.set_ylabel(f"Latent Dim {row_idx}", rotation=90)
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Latent Traversals ({model_name})", fontsize=14)
    plt.tight_layout()
    output_path = f"results_{model_name}_traversal.png"
    plt.savefig(output_path, dpi=300)
    plt.close() # Close figure to free memory
    print(f"✅ Saved {output_path}")

def plot_correlation(data, model_name):
    print(f"   -> Generating Correlation Matrix for {model_name}...")
    
    plt.figure(figsize=(8, 6))
    corr = np.corrcoef(data.T)
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix: {model_name}\n(Diagonal=Good, Checkerboard=Entangled)")
    
    output_path = f"results_{model_name}_correlation.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0, help="Which sample index to use for traversal plots")
    args = parser.parse_args()

    # 1. DATA LOADING (Once)
    set_seed(SEED)
    fixed_actions = get_fixed_dataset(samples_per_suite=20_000)
    print(f"📦 Loaded Fixed Dataset: {fixed_actions.shape}")
    
    # 2. SELECT TRAVERSAL SAMPLE (Once)
    if args.sample_idx >= len(fixed_actions):
        print(f"⚠️ Sample index {args.sample_idx} out of bounds. Using 0.")
        idx = 0
    else:
        idx = args.sample_idx
        
    print(f"🎨 Using Sample Index: {idx} for ALL traversal plots")
    sample_action = fixed_actions[idx : idx+1] # Keep (1, 16, 7) shape

    # 3. ITERATE MODELS
    for model_name in ["beta_tcvae", "vqvae"]:
        print(f"\n--- Processing {model_name} ---")
        
        # Load Model
        model, cfg = load_model(model_name)
        
        # Plot A: Traversals (Using fixed sample)
        plot_traversals(model, cfg, sample_action, model_name)
        
        # Plot B: Correlation (Using full fixed dataset)
        latents = get_latents_from_fixed(model, fixed_actions, cfg)
        plot_correlation(latents, model_name)
        
        # Cleanup GPU
        del model
        torch.cuda.empty_cache()

    print("\n🎉 All plots generated successfully!")