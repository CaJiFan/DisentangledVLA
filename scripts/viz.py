import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # Added PCA
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict
import tqdm

# --- Import your model and dataset classes ---
# (Assumes the classes ActionVQVAE and RLDSDataset are in your scope)
from vlas.openvla_oft.prismatic.vla.datasets import RLDSDataset 
from vlas.openvla_oft.prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from src.ActionVQVAE import ActionVQVAE


EMBED_DIM = 256
NUM_CODES = 1024

@dataclass
class VizTransform:
    action_horizon: int = 8  
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Get Actions
        actions = torch.from_numpy(rlds_batch["action"]).float()
        if actions.shape[0] >= self.action_horizon:
            actions = actions[:self.action_horizon]
        else:
            pad = torch.zeros(self.action_horizon - actions.shape[0], actions.shape[1])
            actions = torch.cat([actions, pad], dim=0)
            
        # 2. Get Task Description (The "Cluster Label")
        # In RLDS/LIBERO, this is stored in 'task' -> 'language_instruction'
        # It is usually a bytes string, so we decode it.
        task_lang = rlds_batch["task"]["language_instruction"]
        if isinstance(task_lang, bytes):
            task_lang = task_lang.decode("utf-8").lower()
        elif isinstance(task_lang, torch.Tensor):
             # Handle rare case where it's already a tensor
            task_lang = task_lang.item().decode("utf-8").lower()
            
        return {
            "actions": actions,
            "label": task_lang # Use this for plotting
        }
    
# --- Main Plotting Function ---
def plot_latent_space(
    checkpoint_path, 
    data_root_dir, 
    dataset_name="libero_spatial", 
    num_samples=2000,
    use_pca=True  # Toggle for PCA
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Weights
    # (Assuming ActionVQVAE class is defined)
    model = ActionVQVAE(
        action_dim=ACTION_DIM, 
        horizon=NUM_ACTIONS_CHUNK,
        embed_dim=EMBED_DIM,
        num_codes=NUM_CODES
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Load Data
    dataset = RLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=dataset_name,
        batch_transform=VizTransform(),
        resize_resolution=(224, 224),
        shuffle_buffer_size=1000,
        train=True,
        image_aug=False
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0)
    
    # 3. Extract Latents
    print(f"Extracting {num_samples} latent codes...")
    latents = []
    labels = []
    
    iter_loader = iter(dataloader)
    pbar = tqdm.tqdm(total=num_samples)
    
    while len(latents) * 64 < num_samples:
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
            
        actions = batch["actions"].to(device)
        batch_labels = batch["task_name"]
        
        with torch.no_grad():
            # Get Continuous Latent 'z'
            z = model.encode(actions)
            latents.append(z.cpu().numpy())
            labels.extend(batch_labels)
            
        pbar.update(64)
    
    X = np.concatenate(latents, axis=0)[:num_samples]
    y = np.array(labels)[:num_samples]
    
    # --- 4. PCA Preprocessing (The "Pro" Move) ---
    if use_pca and X.shape[1] > 50:
        print(f"Running PCA to reduce dim from {X.shape[1]} to 50...")
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
        print(f"Explained Variance by 50 PCs: {np.sum(pca.explained_variance_ratio_):.2f}")

    # --- 5. t-SNE ---
    print("Running t-SNE...")
    # 'init'='pca' is another trick for stability
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # --- 6. Plotting ---
    df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Task': y})
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df, x='x', y='y', hue='Task', 
        palette='tab10', s=60, alpha=0.8, legend='full'
    )
    
    title = f"Latent Space: {dataset_name}"
    if use_pca: title += " (PCA -> t-SNE)"
    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig("best_practice_tsne.png", dpi=300)
    plt.show()

def plot_latent_clusters(
    checkpoint_path, 
    data_root_dir, 
    dataset_name="libero_spatial_no_noops",
    num_samples=2500
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Weights
    # (Assuming ActionVQVAE class is defined)
    model = ActionVQVAE(
        action_dim=ACTION_DIM, 
        horizon=NUM_ACTIONS_CHUNK,
        embed_dim=EMBED_DIM,
        num_codes=NUM_CODES
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Initialize Dataset with NEW VizTransform
    dataset = RLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=dataset_name,
        batch_transform=VizTransform(action_horizon=8), # Uses language as label
        resize_resolution=(224, 224),
        shuffle_buffer_size=1, # disable shuffling for consistent results
        train=True,
        image_aug=False
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0)
    
    # Extract
    latents = []
    labels = []
    
    print(f"Extracting {num_samples} samples from {dataset_name}...")
    iter_loader = iter(dataloader)
    
    # Loop to collect data
    while len(latents) * 64 < num_samples:
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
        
        actions = batch["actions"].to(device)
        batch_labels = batch["label"] # Now contains "pick up the red cup", etc.
        
        with torch.no_grad():
            z = model.encode(actions)
            latents.append(z.cpu().numpy())
            labels.extend(batch_labels)

    # Concat
    X = np.concatenate(latents, axis=0)
    y = np.array(labels)
    
    
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca')
    X_emb = tsne.fit_transform(X_pca)

    
    df = pd.DataFrame({'x': X_emb[:,0], 'y': X_emb[:,1], 'Task': y})
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df, x='x', y='y', hue='Task', 
        palette='tab10', # 10 distinct colors for 10 tasks
        s=80, alpha=0.8
    )
    
    plt.title(f"Action Latent Space by Task: {dataset_name}", fontsize=16)
    # Place legend outside to not cover clusters
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize='small')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_clusters.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    
    plot_latent_clusters(
        checkpoint_path= "checkpoints/vqvae_action_step_50000.pt" ,
        data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/", 
        dataset_name="libero_spatial_no_noops"
    )