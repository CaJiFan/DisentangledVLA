import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import tensorflow as tf
import tqdm
import random
import argparse
import sys
import os

# --- YOUR IMPORTS ---
from src.disentanglers.ActionBetaTCVAE import ActionBetaTCVAE
from src.disentanglers.ActionCNNVQVAE import ActionVQVAE 
from utils.data import FastActionRLDSDataset, ActionOnlyTransform

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    # Optional: Deterministic algorithms (slower but strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Global Seed set to {seed}")

SEED = 42

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/mnt/Data/cjimenez/LIBERO/libero/datasets/"

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
        "params": {"action_dim": 7, "horizon": 16, "embed_dim": 256, "num_codes": 1024},
        "is_probabilistic": False
    }
}

SUITES = {
    "libero_goal": "Goal Tasks", 
    "libero_spatial": "Spatial Tasks"
}


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

def get_labeled_latents(model, config, samples_per_suite=1000):
    all_z = []
    all_labels = []
    
    print("🏷️  Extracting Labeled Latents...")
    
    for label_idx, (suite_key, suite_name) in enumerate(SUITES.items()):
        # Initialize dataset (Standard loading logic)

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
        
        with torch.no_grad():
            with tqdm.tqdm(total=samples_per_suite, desc=f"Encoding {suite_key}") as pbar:
                while count < samples_per_suite:
                    try:
                        batch = next(iterator)
                        # Handle Dict vs Tuple vs Tensor
                        if isinstance(batch, dict): actions = batch['actions']
                        elif isinstance(batch, (list, tuple)): actions = batch[0]
                        else: actions = batch

                        # Clean Tensor
                        if isinstance(actions, np.ndarray): actions = torch.tensor(actions).float()
                        else: actions = actions.clone().detach().float()
                        actions = actions.to(DEVICE)
                        if actions.ndim == 2: actions = actions.unsqueeze(0)

                        if count == 0:
                            print(f"DEBUG: First batch for {suite_name}, action sum: {actions.sum().item():.4f}")

                        # --- ENCODING LOGIC ---
                        if config["is_probabilistic"]:
                            # TCVAE
                            mu, _ = model.encode(actions)
                            z_out = mu
                        else:
                            # VQ-VAE
                            z = model.encoder(actions.permute(0, 2, 1))
                            if isinstance(z, tuple): z = z[0]
                            # If VQVAE returns (B, Embed, T), flatten or mean-pool
                            if z.ndim == 3: 
                                z_out = z.mean(dim=-1) 
                            else:
                                z_out = z

                        all_z.append(z_out.cpu().numpy())
                        batch_size = z_out.shape[0]
                        all_labels.extend([label_idx] * batch_size)
                        
                        count += batch_size
                        pbar.update(batch_size)
                    except StopIteration:
                        break
    
    X = np.concatenate(all_z, axis=0)
    y = np.array(all_labels)
    return X, y

def plot_semantic_tsne(X, y, model_name):
    print(f"🎨 Running t-SNE for {model_name}...")
    
    tsne = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto', random_state=SEED)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4'] 
    
    for label_idx, (suite_key, suite_name) in enumerate(SUITES.items()):
        indices = (y == label_idx)
        plt.scatter(
            X_embedded[indices, 0], X_embedded[indices, 1], 
            c=colors[label_idx], label=suite_name,
            alpha=0.6, s=10, edgecolors='none'
        )
        
    plt.title(f"Latent Semantic Structure: {model_name.upper()}", fontsize=14, fontweight='bold')
    plt.legend(markerscale=3, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('off')
    
    output_path = f"results_semantic_{model_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved {output_path}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True, choices=["beta_tcvae", "vqvae"], help="Which model to visualize")
#     args = parser.parse_args()

#     set_seed(seed=SEED)
    
#     model, cfg = load_model(args.model)
#     X, y = get_labeled_latents(model, cfg, samples_per_suite=2000)
#     plot_semantic_tsne(X, y, args.model)

def get_fixed_dataset(samples_per_suite=1000):
    """Loads data once into memory to ensure identical inputs for both models."""
    fixed_data = []
    
    print("💾 Caching fixed validation dataset...")
    
    for label_idx, (suite_key, suite_name) in enumerate(SUITES.items()):
        # Set seed just for this initial load
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

                    if count == 0:
                        print(f"DEBUG: First batch for {suite_name}, action sum: {actions.sum().item():.4f}")

                    # Store as CPU tensor
                    if isinstance(actions, np.ndarray): actions = torch.tensor(actions).float()
                    else: actions = actions.clone().detach().float()
                    
                    if actions.ndim == 2: actions = actions.unsqueeze(0)
                    
                    suite_actions.append(actions)
                    count += actions.shape[0]
                    pbar.update(actions.shape[0])
                except StopIteration:
                    break
        
        # Stack this suite's data
        fixed_data.append({
            "actions": torch.cat(suite_actions, dim=0)[:samples_per_suite], # Crop to exact
            "label": label_idx,
            "name": suite_name
        })
        
    return fixed_data

def encode_fixed_data(model, config, fixed_data):
    all_z = []
    all_labels = []
    
    model.eval()
    
    print(f"⚙️  Encoding with {config['class'].__name__}...")
    
    with torch.no_grad():
        for suite_data in fixed_data:
            actions_tensor = suite_data["actions"].to(DEVICE) # Move to GPU only when needed
            label = suite_data["label"]
            
            # Process in batches to avoid OOM
            batch_size = 256
            total = actions_tensor.shape[0]
            
            for i in range(0, total, batch_size):
                batch = actions_tensor[i : i+batch_size]
                
                if config["is_probabilistic"]:
                    mu, _ = model.encode(batch)
                    z_out = mu
                else:
                    # VQ-VAE
                    z = model.encoder(batch.permute(0, 2, 1))
                    if isinstance(z, tuple): z = z[0]
                    if z.ndim == 3: z_out = z.mean(dim=-1)
                    else: z_out = z
                
                all_z.append(z_out.cpu().numpy())
                all_labels.extend([label] * batch.shape[0])

    return np.concatenate(all_z, axis=0), np.array(all_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # No "model" arg needed anymore, we run BOTH
    args = parser.parse_args()

    set_seed(SEED)
    
    # 1. Load Data ONCE
    fixed_data = get_fixed_dataset(samples_per_suite=2000)
    
    # 2. Run Beta-TCVAE
    model_tcvae, cfg_tcvae = load_model("beta_tcvae")
    X_tcvae, y_tcvae = encode_fixed_data(model_tcvae, cfg_tcvae, fixed_data)
    plot_semantic_tsne(X_tcvae, y_tcvae, "beta_tcvae")
    
    # Clear GPU to be safe
    del model_tcvae
    torch.cuda.empty_cache()
    
    # 3. Run VQ-VAE (on exact same data)
    model_vqvae, cfg_vqvae = load_model("vqvae")
    X_vqvae, y_vqvae = encode_fixed_data(model_vqvae, cfg_vqvae, fixed_data)
    plot_semantic_tsne(X_vqvae, y_vqvae, "vqvae")