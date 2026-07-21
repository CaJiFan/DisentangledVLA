import torch
import numpy as np
import os
import tqdm
from PIL import Image

# --- LIBERO IMPORTS ---
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# --- YOUR MODEL IMPORTS ---
from transformers import AutoModelForVision2Seq, AutoProcessor
from src.disentanglers.ActionBetaTCVAE import ActionBetaTCVAE
from src.projectors.ProbabilisticActionProjector import ProbabilisticActionProjector
from utils.data import FastActionRLDSDataset, identity_transform # To get stats

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLA_PATH = "openvla/openvla-7b"
VAE_PATH = "./checkpoints/disentanglers/beta_tcvae_step_50000.pt"
PROJ_PATH = "./checkpoints/probabilistic_projector/projector_step_50000.pt"

BENCHMARK_NAME = "libero_10_no_noops" # The "Long Horizon" suite
EXECUTION_HORIZON = 8       # How many steps to execute before replanning (chunk size is 16)
MAX_STEPS = 600             # Max steps per episode (timeout)
NUM_EPISODES = 20           # Episodes per task to evaluate

def load_models():
    print("⏳ Loading Models...")
    # 1. VAE
    vae = ActionBetaTCVAE(action_dim=7, chunk_size=16, latent_dim=16).to(DEVICE).eval()
    vae.load_state_dict(torch.load(VAE_PATH))
    
    # 2. Projector
    proj = ProbabilisticActionProjector(4096, 1024, 16).to(DEVICE).eval()
    proj.load_state_dict(torch.load(PROJ_PATH))
    
    # 3. OpenVLA
    processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        VLA_PATH, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(DEVICE).eval()
    
    return vae, proj, vla, processor

def get_dataset_stats():
    """
    We need the dataset statistics to UN-NORMALIZE the actions.
    The model predicts [-1, 1], but the robot needs real values (e.g. [-0.5, 0.5]).
    """
    print("📊 Loading Dataset Stats...")
    # Initialize dataset just to grab the stats dictionary
    # We use a small subset to make initialization fast
    ds = FastActionRLDSDataset(
        data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/",
        data_mix=[BENCHMARK_NAME], 
        batch_transform=identity_transform,
        resize_resolution=(224, 224),
        train=True,
    )
    return ds.dataset_statistics

def unnormalize_action(action, stats):
    """
    Applies the reverse of the normalization used during training.
    Assuming 'bounds_q99' or similar min/max normalization.
    """
    mask = stats['action']['mask'].numpy()
    min_val = stats['action']['min'].numpy()
    max_val = stats['action']['max'].numpy()
    
    # Formula: x_real = (x_norm + 1) / 2 * (max - min) + min
    # (Assuming training normalized to [-1, 1])
    action_unnorm = (action + 1) / 2 * (max_val - min_val) + min_val
    return action_unnorm

@torch.no_grad()
def predict_action_chunk(vae, proj, vla, processor, img_pil, instruction):
    # 1. Prepare Inputs
    prompt = f"In: {instruction}\nOut: "
    inputs = processor(text=[prompt], images=[img_pil], return_tensors="pt").to(DEVICE)
    if hasattr(inputs, "pixel_values"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
    # 2. VLA Embedding
    outputs = vla(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-1] # Shape: (1, Seq, 4096)
    
    # Get last token
    idx = inputs.attention_mask.sum(dim=1) - 1
    emb = last_hidden[0, idx].float() # Cast BF16 -> FP32 for Projector
    
    # 3. Projector & VAE
    _, pred_mu, _ = proj(emb.unsqueeze(0)) # (1, 16)
    action_chunk = vae.decode(pred_mu)     # (1, 16, 7)
    
    return action_chunk[0].cpu().numpy()

def main():
    # 1. Setup
    vae, proj, vla, processor = load_models()
    stats = get_dataset_stats()
    
    # 2. Initialize Benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    print(benchmark_dict)
    task_suite = benchmark_dict[BENCHMARK_NAME]()
    num_tasks = task_suite.get_num_tasks()
    print(f"🚀 Starting Evaluation on {BENCHMARK_NAME} ({num_tasks} tasks)")

    total_successes = 0
    total_episodes = 0

    # 3. Task Loop
    for task_id in range(num_tasks):
        # Get Task Info
        task = task_suite.get_task(task_id)
        task_name = task.name
        init_states = task_suite.get_task_init_states(task_id)
        instruction = task.language 
        
        print(f"\n📍 Task {task_id+1}/{num_tasks}: {task_name}")
        print(f"   Instruction: '{instruction}'")

        # Initialize Environment
        env_args = {
            "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            "render_gpu_device_id": 0
        }
        env = OffScreenRenderEnv(**env_args)
        
        task_successes = 0

        # 4. Episode Loop
        for ep in range(NUM_EPISODES):
            env.reset()
            # Set initial state (important for reproducibility)
            init_state = init_states[ep % len(init_states)]
            env.set_init_state(init_state)
            
            # Start Evaluation
            steps = 0
            done = False
            success = False
            
            pbar = tqdm.tqdm(total=MAX_STEPS, desc=f"Ep {ep+1}", leave=False)
            
            while steps < MAX_STEPS:
                # A. Get Observation
                obs = env._get_observations()
                # LIBERO uses 'agentview_image' by default. Shape: (256, 256, 3) usually.
                # It comes upside down in some versions, but usually standard RGB.
                img_array = obs['agentview_image'] 
                
                # Flip/Resize Logic
                # LIBERO images are often flipped vertically. Check visually if failing.
                # img_array = img_array[::-1] 
                
                img_pil = Image.fromarray(img_array).resize((224, 224))
                
                # B. Predict Chunk
                raw_action_chunk = predict_action_chunk(vae, proj, vla, processor, img_pil, instruction)
                
                # C. Unnormalize
                # (16, 7) -> Real Robot Units
                action_chunk = unnormalize_action(raw_action_chunk, stats)
                
                # D. Execute Horizon (RHC)
                # We execute 'EXECUTION_HORIZON' steps from the chunk
                for k in range(EXECUTION_HORIZON):
                    if steps >= MAX_STEPS: break
                    
                    action = action_chunk[k] # (7,)
                    
                    # Step Env
                    # Action format: [x, y, z, ax, ay, az, gripper]
                    obs, reward, done, info = env.step(action)
                    steps += 1
                    pbar.update(1)
                    
                    # Check Success
                    if env.check_done():
                        success = True
                        break
                
                if success: break
            
            pbar.close()
            
            if success:
                task_successes += 1
                print(f"   ✅ Episode {ep+1} Success!")
            else:
                print(f"   ❌ Episode {ep+1} Failed.")
                
        # Task Summary
        print(f"   👉 Task Success Rate: {task_successes}/{NUM_EPISODES} ({(task_successes/NUM_EPISODES)*100:.1f}%)")
        total_successes += task_successes
        total_episodes += NUM_EPISODES
        
        env.close()

    print(f"\n🏆 FINAL SCORE: {total_successes}/{total_episodes} ({(total_successes/total_episodes)*100:.1f}%)")

if __name__ == "__main__":
    main()