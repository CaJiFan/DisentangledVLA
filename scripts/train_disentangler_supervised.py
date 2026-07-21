import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import wandb
import h5py
from transformers import CLIPTokenizer, CLIPTextModel

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.disentanglers.TextActionDecOnlyBetaTCVAE import TextActionBetaTCVAE
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- CONFIG ---
DATA_PATH = "./processed_data/semantic_libero_goal.hdf5"
SAVE_DIR = "./checkpoints/text_conditioned_tcvae/"
BATCH_SIZE = 256
LR = 1e-4
MAX_STEPS = 50_000
VAE_TYPE = "beta_tcvae"

MAX_SEQ_LEN = 168
ACTION_DIM = 7

USE_WANDB = True
WANDB_PROJECT = "DisentangledVLA"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. SKILL VOCABULARY ---
# Map the integer labels from your dataset builder to rich, semantic text prompts
SKILL_VOCAB = {
    0: "Move the arm through free space to align with the target",
    1: "Reach, close the gripper, and lift the object",
    2: "Move the held object to the target and open the gripper to release it",
    3: "Push the object horizontally across the surface",
    4: "Pull the drawer handle backward to open it",
    5: "Twist the wrist to turn the stove knob",
    6: "Retract the empty gripper away from the target to finish"
}

# --- 2. CUSTOM DATASET ---
class SemanticSkillDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['actions'])
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            actions = f['actions'][idx]
            mask = f['masks'][idx]
            label = f['labels'][idx]
        return torch.tensor(actions), torch.tensor(mask), torch.tensor(label, dtype=torch.long)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a Text-Conditioned Disentangled VAE")
    parser.add_argument("--beta", type=float, default=6.0, help="Beta value for the TC loss term")
    parser.add_argument("--latent_dim", type=int, default=16, help="Size of the physics bottleneck (e.g., 16, 32, 64)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for the text conditioning (0.0 = no dropout)")
    return parser.parse_args()

def train_conditional_disentangler():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    WANDB_RUN_NAME = f"SemanticText_{VAE_TYPE}_beta_{args.beta}_z{args.latent_dim}_dropout_{args.dropout}"
    
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "learning_rate": LR, "batch_size": BATCH_SIZE, "max_steps": MAX_STEPS, 
            "vae_type": VAE_TYPE, "chunk_size": MAX_SEQ_LEN
        })

    # --- 3. PRE-COMPUTE CLIP EMBEDDINGS ---
    print("🧠 Pre-computing CLIP Text Embeddings for the 8 Semantic Skills...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    
    with torch.no_grad():
        texts = [SKILL_VOCAB[i] for i in range(len(SKILL_VOCAB))]
        inputs = tokenizer(texts, padding=True, return_tensors="pt").to(DEVICE)
        # Get the (8, 512) embedding matrix for all skills
        precomputed_text_embs = text_encoder(**inputs).pooler_output 
        
    # Free up VRAM! We don't need the massive CLIP model in memory anymore.
    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()

    # --- 4. MODEL INIT ---
    print(f"⚙️ Initializing Conditional {VAE_TYPE} (Seq Len: {MAX_SEQ_LEN}, Beta: {args.beta})...")
    vae = TextActionBetaTCVAE(
        action_dim=ACTION_DIM, 
        chunk_size=MAX_SEQ_LEN, 
        latent_dim=args.latent_dim, 
        text_emb_dim=512, 
        beta=args.beta
    ).to(DEVICE)
        
    optimizer = optim.AdamW(vae.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS)
    
    # --- 5. DATA LOADER ---
    dataset = SemanticSkillDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    
    def infinite_loader(dl):
        while True:
            for batch in dl: yield batch
    data_iter = infinite_loader(dataloader)
    
    print("🔥 Starting Masked Training...")
    with tqdm.tqdm(total=MAX_STEPS) as pbar:
        for step in range(MAX_STEPS):
            
            actions, masks, labels = next(data_iter)
            actions = actions.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if labels.max() >= 7 or labels.min() < 0:
                raise ValueError(f"🧨 LABEL ERROR: Found an out-of-bounds label! "
                                 f"Min: {labels.min()}, Max: {labels.max()}. Expected 0-6.")
            # The Magic Trick: Index the pre-computed (8, 512) matrix using our batch labels
            # This instantly creates a (Batch_Size, 512) text embedding tensor!
            text_features = precomputed_text_embs[labels]
            
            if torch.rand(1).item() < args.dropout:
                text_features = torch.zeros_like(text_features)
            # Forward Pass VAE
            recon_actions, mu, logvar, z = vae(actions, text_features)
            
            # --- 6. THE MASKED LOSS SWAP ---
            # Ask the VAE for its losses. It will give us a mathematically correct TC/KL term, 
            # but a flawed (unmasked) reconstruction term.
            _, raw_recon_loss_scalar, tc_loss = vae.compute_loss(actions, recon_actions, mu, logvar, z)

            # Calculate our own flawless Masked MSE
            expanded_masks = masks.unsqueeze(-1)
            raw_mse = F.mse_loss(recon_actions, actions, reduction='none')
            masked_mse = raw_mse * expanded_masks
            
            # Average ONLY over the valid frames
            masked_recon_loss_scalar = masked_mse.sum() / (expanded_masks.sum() * ACTION_DIM)

            # Recombine our Masked MSE with the VAE's TC/KL term
            total_loss = masked_recon_loss_scalar + tc_loss

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Loss: {total_loss.item():.4f} (Masked MSE: {masked_recon_loss_scalar.item():.4f}, TC: {tc_loss.item():.4f})")
            
            if USE_WANDB:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/masked_recon_loss": masked_recon_loss_scalar.item(),
                    "train/tc_loss": tc_loss.item(),
                    "train/learning_rate": current_lr,
                    "global_step": step
                })
            
            pbar.update(1)
            if (step+1) % 10_000 == 0 and step > 0:
                torch.save(vae.state_dict(), f"{SAVE_DIR}/supervised_{VAE_TYPE}_DecOnly_step_{step+1}_z_{args.latent_dim}.pt")

                
    if USE_WANDB: wandb.finish()
    print("✅ Training Complete!")

if __name__ == "__main__":
    train_conditional_disentangler()