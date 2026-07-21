import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from src.disentanglers.ActionBetaTCVAE import ActionBetaTCVAE
from src.projectors.ProbabilisticActionProjector import ProbabilisticActionProjector
from utils.data import FastActionRLDSDataset, identity_transform
from PIL import Image

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLA_PATH = "openvla/openvla-7b"
VAE_PATH = "./checkpoints/disentanglers/beta_tcvae_step_50000.pt"
# Use your latest checkpoint
PROJ_PATH = "./checkpoints/probabilistic_projector/projector_step_50000.pt" 

def load_models():
    print("Loading VAE...")
    vae = ActionBetaTCVAE(action_dim=7, chunk_size=16, latent_dim=16).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH))
    vae.eval()

    print("Loading Projector...")
    proj = ProbabilisticActionProjector(4096, 1024, 16).to(DEVICE)
    proj.load_state_dict(torch.load(PROJ_PATH))
    proj.eval()

    print("Loading OpenVLA (Backbone)...")
    processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        VLA_PATH, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(DEVICE)
    vla.eval()
    
    return vae, proj, vla, processor

def get_vla_embedding(model, processor, image, instruction):
    prompt = f"In: {instruction}\nOut: "
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE)
    if hasattr(inputs, "pixel_values"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    last_hidden = outputs.hidden_states[-1]
    idx = inputs.attention_mask.sum(dim=1) - 1
    embedding = last_hidden[0, idx]
    return embedding.float() # (1, 4096)

def evaluate():
    vae, proj, vla, processor = load_models()
    
    # Load ONE batch of validation data
    dataset = FastActionRLDSDataset(
        data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/",
        data_mix=["libero_goal_no_noops", "libero_spatial_no_noops"],
        batch_transform=identity_transform,
        resize_resolution=(224, 224),
        train=True, # VALIDATION SET
        # split="train[-10%:]",
        return_visuals=True
    )

    # Create Iterator
    iterator = iter(dataset)
    
    # --- MANUAL SKIP ---
    print("Skipping first 200 batches to simulate validation...")
    for _ in range(200): 
        next(iterator)
    
    # Just grab the first item manually to test
    print("\n--- Running Inference ---")
    iterator = iter(dataset)
    
    # Let's test 3 examples
    for i in range(30):
        item = next(iterator)
        img_raw = item['observation']['image_primary']

        if isinstance(img_raw, torch.Tensor):
            img_raw = img_raw.detach().cpu().numpy()
            
        # 3. Squeeze "Singleton" Dimensions
        # This turns (1, 1, 224, 224, 3) -> (224, 224, 3)
        img_raw = img_raw.squeeze()
        
        # 4. Handle Channel First/Last
        # If shape is (3, 224, 224), transpose to (224, 224, 3)
        if img_raw.shape[0] == 3:
            img_raw = np.transpose(img_raw, (1, 2, 0))
            
        # 5. Handle Normalization (Float -> Uint8)
        if img_raw.dtype == np.float32 or img_raw.dtype == np.float64:
            if img_raw.max() <= 1.0:
                img_raw = (img_raw * 255).astype(np.uint8)
            else:
                img_raw = img_raw.astype(np.uint8)
        else:
            img_raw = img_raw.astype(np.uint8)

        # 6. Verify Shape (Debug Check)
        # If the shape is still weird (e.g., missing width), this will tell us
        if img_raw.ndim != 3:
            print(f"⚠️ Warning: Weird image shape after squeeze: {img_raw.shape}")
            # Fallback: Just take the first frame if it's still 4D
            if img_raw.ndim == 4: 
                img_raw = img_raw[0]
            
        # 3. Create PIL Image (Corrected Line)
        img_pil = Image.fromarray(img_raw)
        
        instr = item['task']['language_instruction']
        if isinstance(instr, bytes): instr = instr.decode("utf-8")
            
        gt_action = torch.tensor(item['action']).float().to(DEVICE).unsqueeze(0) # (1, 16, 7)
        
        print(f"\nTest {i+1}: '{instr}'")
        
        # 1. VLA Embedding
        emb = get_vla_embedding(vla, processor, img_pil, instr)
        
        # 2. Projector
        # We take the MEAN (mu) as the deterministic prediction for evaluation
        _, pred_mu, _ = proj(emb) 
        
        # 3. VAE Decode
        pred_action = vae.decode(pred_mu) # (1, 16, 7)
        
        # 4. Metric (MSE)
        mse = torch.nn.functional.mse_loss(pred_action, gt_action).item()
        print(f"  > Action MSE: {mse:.4f}")
        
        # Optional: Compare first step of chunk
        print(f"  > GT Start:   {gt_action[0,0,:3].cpu().numpy().round(3)}")
        print(f"  > Pred Start: {pred_action[0,0,:3].detach().cpu().numpy().round(3)}")

if __name__ == "__main__":
    evaluate()