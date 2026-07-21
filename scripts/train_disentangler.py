import argparse
import tqdm
import os
import torch
import torch.nn.functional as F
from vlas.openvla_oft.prismatic.vla.constants import ACTION_DIM

from src.disentanglers.ActionBetaTCVAE import ActionBetaTCVAE
from src.disentanglers.ActionCNNVQVAE import ActionVQVAE

from utils.data import get_ram_cached_dataloader

# --- CONFIG ---
EMBED_DIM = 256
NUM_CODES = 1024
BATCH_SIZE = 256
LATENT_DIM = 16
NUM_ACTIONS_CHUNK = 16
BETA_TC = 6.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for disentangled action models.")
    parser.add_argument("--model", type=str, required=True, help="Model type (e.g., vqvae, beta_tcvae)")
    parser.add_argument("--save_to", type=str, default="./checkpoints/disentanglers", help="Path to save the trained model.")
    return parser.parse_args()

def train(data_root_dir, max_steps=50000):
    args = parse_arguments()
    MODEL_TYPE = args.model.lower()
    SAVE_DIR = args.save_to

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 1. Model Selection ---
    print(f"🚀 Initializing Model: {MODEL_TYPE.upper()}")
    
    if MODEL_TYPE == "vqvae":
        model = ActionVQVAE(
            action_dim=ACTION_DIM,
            horizon=NUM_ACTIONS_CHUNK,
            embed_dim=EMBED_DIM,
            num_codes=NUM_CODES
        ).to(device)
    elif MODEL_TYPE == "beta_tcvae":
        model = ActionBetaTCVAE(
            action_dim=ACTION_DIM,
            chunk_size=NUM_ACTIONS_CHUNK,
            latent_dim=LATENT_DIM,
            beta=BETA_TC
        ).to(device)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # --- 2. Data Loading (RAM Cached) ---
    dataloader = get_ram_cached_dataloader(
        data_root_dir=data_root_dir,
        batch_size=BATCH_SIZE, 
        action_horizon=NUM_ACTIONS_CHUNK
    )
    # Cycle the dataloader indefinitely for max_steps
    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch
    
    iter_loader = infinite_loader(dataloader)

    # --- 3. Training Loop ---
    model.train()
    print(f"🔥 Starting Training for {max_steps} steps...")
    
    with tqdm.tqdm(total=max_steps) as pbar:
        for step in range(max_steps):
            
            # Helper: Get batch from infinite loader
            # TensorDataset returns a tuple (actions,), so we unpack
            batch_tuple = next(iter_loader)
            actions = batch_tuple[0].to(device) # (B, 16, 7)
            
            optimizer.zero_grad()
            
            # --- VQ-VAE Forward ---
            if MODEL_TYPE == "vqvae":
                pred_actions, vq_loss, _ = model(actions)
                rec_loss = F.mse_loss(pred_actions, actions)
                total_loss = rec_loss + vq_loss
                
                logs = {
                    "L_Total": total_loss.item(), 
                    "L_Rec": rec_loss.item(), 
                    "L_VQ": vq_loss.item()
                }

            # --- Beta-TCVAE Forward ---
            elif MODEL_TYPE == "beta_tcvae":
                # Ensure input is (B, D, T) if your class expects it, 
                # or rely on the class to permute. 
                # (Assuming the class provided earlier handles (B,T,D) -> (B,D,T) internally via .permute)
                
                # Note: The class I gave you returns 4 items from forward()
                x_recon, mu, logvar, z = model(actions) 
                
                # IMPORTANT: Use the strict compute_loss signature
                # We permute target actions to (B, D, T) to match reconstruction shape if needed
                total_loss, rec_loss_val, tc_loss_val = model.compute_loss(
                    # actions.permute(0, 2, 1), 
                    actions, 
                    x_recon, 
                    mu, logvar, z
                )
                
                logs = {
                    "L_Total": total_loss.item(), 
                    "L_Rec": rec_loss_val.item(), 
                    "L_TC": tc_loss_val.item() # Explicitly log Total Correlation
                }

            total_loss.backward()
            optimizer.step()
            
            # Update Progress Bar
            pbar.set_description(f"{MODEL_TYPE.upper()} | " + " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))
            pbar.update(1)
            
            # Checkpointing
            if (step + 1) % 5000 == 0:
                save_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_step_{step+1}.pt")
                torch.save(model.state_dict(), save_path)
                
    print(f"✅ Training Complete. Saved to {SAVE_DIR}")
    
if __name__ == "__main__":
    train(
        data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/"
    )
