import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #

import gc
from glob import glob
import h5py
import torch
import numpy as np
import imageio
import cv2
from transformers import CLIPTokenizer, CLIPTextModel

# --- Your Custom Imports ---
from src.disentanglers import TextActionBetaTCVAE, MLPTextActionBetaTCVAE
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from utils.data import get_text_action_ram_cached_dataloader, get_text_action_ram_cached_dataloader2
from vlas.openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
)

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUITE = "libero_spatial"
UNNORM_KEY = f'{SUITE}_no_noops'
DATASET_PATH = "/mnt/Data/cjimenez/LIBERO/libero/datasets/"
MAX_STEPS = 300

# 64 seed 4
# 128 seed 1

BETA=0.1
Z_DIM=128
CHUNK_SIZE = 8
ALPHA = 1.0
STEP = 100_000
SEED = 1
RECON_WEIGHT = 100
SAVE_VIDEOS = True

if CHUNK_SIZE == 16:
    VAE_PATH = f"./checkpoints/text_tcvae/{SUITE}/{SUITE}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_step_50000.pt"
else:
    # VAE_PATH = f"./checkpoints/text_tcvae/{SUITE}/{SUITE}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_chunk{CHUNK_SIZE}_step_50000.pt"
    # VAE_PATH = f"./checkpoints/text_tcvae/{SUITE}/{SUITE}_text_cond_beta_tcvae_dropout0.15_beta{BETA}_z{Z_DIM}_alpha{ALPHA}_chunk{CHUNK_SIZE}_std_seed_{SEED}_step_{STEP}.pt"
    VAE_PATH = f"./checkpoints/text_tcvae/{SUITE}/rw{RECON_WEIGHT}_dropout0.15_beta{BETA}_z{Z_DIM}_alpha{ALPHA}_chunk{CHUNK_SIZE}_std_mlp_seed_{SEED}_step_{STEP}.pt"
STATS_PATH = f"./checkpoints/text_tcvae/{SUITE}/dataset_statistics.pt"

# --- HELPER FUNCTIONS ---
def normalize_actions(actions_unnormalized, stats):
    stats = stats[UNNORM_KEY]
    
    x = actions_unnormalized.to(DEVICE)
    low = torch.tensor(stats['action']['q01']).to(DEVICE)
    high = torch.tensor(stats['action']['q99']).to(DEVICE)
    _min = torch.tensor(stats['action']['min']).to(DEVICE)
    _max = torch.tensor(stats['action']['max']).to(DEVICE)
    
    # Check if a mask exists, otherwise default to all True
    if 'mask' in stats['action']:
        mask = torch.tensor(stats['action']['mask'], dtype=torch.bool).to(DEVICE)
    else:
        mask = torch.ones_like(low, dtype=torch.bool).to(DEVICE)

    # 1. Base Normalization matching OpenVLA exact math
    norm_x = 2.0 * (x - low) / (high - low + 1e-8) - 1.0
    norm_x = torch.clamp(norm_x, min=-1.0, max=1.0)
    
    # 2. Apply Mask (if mask is False for a dim, keep the original x)
    x = torch.where(mask, norm_x, x)
    
    # 3. Handle unused dimensions (where min == max, output 0.0)
    zeros_mask = (_min == _max)
    x = torch.where(zeros_mask, 0.0, x)
    
    return x

def unnormalize_actions(actions_normalized, stats):
    stats = stats[UNNORM_KEY]
    
    x = actions_normalized.clone().to(DEVICE)
    low = torch.tensor(stats['action']['q01']).to(DEVICE)
    high = torch.tensor(stats['action']['q99']).to(DEVICE)
    _min = torch.tensor(stats['action']['min']).to(DEVICE)
    _max = torch.tensor(stats['action']['max']).to(DEVICE)
    
    if 'mask' in stats['action']:
        mask = torch.tensor(stats['action']['mask'], dtype=torch.bool).to(DEVICE)
    else:
        mask = torch.ones_like(low, dtype=torch.bool).to(DEVICE)

    # 1. Reverse the base normalization
    unnorm_x = ((x + 1.0) / 2.0) * (high - low + 1e-8) + low
    
    # 2. Apply Mask (if mask is False, the value was never normalized, so don't unnormalize it)
    x = torch.where(mask, unnorm_x, x)
    
    # 3. Handle unused dimensions
    # If min == max, the network predicted 0.0. In physical reality, the value should just be the min (or max).
    zeros_mask = (_min == _max)
    x = torch.where(zeros_mask, _min, x)
    
    return x

def load_models():
    """ Load heavy models ONCE to save VRAM """
    print("⏳ Loading Models & Stats...")
    vae = MLPTextActionBetaTCVAE(
        action_dim=7,
        chunk_size=CHUNK_SIZE, 
        latent_dim=Z_DIM,
        text_emb_dim=512
    ).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH))
    vae.eval()

    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    stats = torch.load(STATS_PATH)
    
    return vae, clip_tok, clip_enc, stats

def run_oracle_for_task(task_path, bmark, models):
    vae, clip_tok, clip_enc, stats = models

    # 1. Dynamically match the HDF5 filename to the LIBERO Task ID
    task_name = os.path.basename(task_path).replace(".hdf5", "")
    # print(task_name)
    # print(bmark.get_num_tasks())
    # if task_name not in ["pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo", \
    #                       "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo", \
    #                       "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo"]:
    #     return  # Skip tasks that are not in the validation split
    task_id = None

    for i in range(bmark.get_num_tasks()):
        # print(bmark.get_task(i).name)
        if bmark.get_task(i).name + '_demo' == task_name:
            task_id = i
            break
            
    if task_id is None:
        print(f"⚠️ Warning: Could not find {task_name} in benchmark. Skipping.")
        return
    
    instruction = bmark.get_task(task_id).language
    print(f"🎯 Instruction: {instruction}")

     # Embed Instruction
    text_inputs = clip_tok([instruction], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_features = clip_enc(**text_inputs).pooler_output

    
    print(f"\n==================================================")
    print(f"📖 Processing Task: {task_name}")
    print(f"==================================================")

    task_mse_list = []
    task_dim_mse_list = []

    with h5py.File(task_path, "r") as f:
        demos = list(f["data"].keys())
        
        for demo_id in demos:
            print(f"  -> Running {demo_id}...")
            
            # --- Extract Data ---
            gt_actions_numpy = f[f"data/{demo_id}/actions"][:]
            # instruction = f[f"data/{demo_id}/obs/task_language"][0].decode("utf-8")

            # Pad actions with the final resting state so the last chunk is exactly 16 steps
            padding = np.zeros((CHUNK_SIZE, 7))
            padding[:] = gt_actions_numpy[-1]
            gt_actions_padded = np.concatenate([gt_actions_numpy, padding], axis=0)
            
            gt_actions_tensor = torch.tensor(gt_actions_padded).float()
            gt_actions_norm = normalize_actions(gt_actions_tensor, stats)

            # --- Initialize Environment ---
            env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
            env_gt = OffScreenRenderEnv(**env_args)
            env_pred = OffScreenRenderEnv(**env_args)
            env_gt.reset()
            env_pred.reset()

            # --- Video Setup ---
            if SAVE_VIDEOS:
                output_dir = f"videos/{SUITE}/{task_name}/rw{RECON_WEIGHT}_dropout0.15_beta{BETA}_z{Z_DIM}_alpha{ALPHA}_chunk{CHUNK_SIZE}_std_mlp"
                os.makedirs(output_dir, exist_ok=True)
                video_gt_path = f"{output_dir}/{demo_id}_gt.mp4"
                video_pred_path = f"{output_dir}/{demo_id}_pred.mp4"
                writer_gt = imageio.get_writer(video_gt_path, fps=30, macro_block_size=1)
                writer_pred = imageio.get_writer(video_pred_path, fps=30, macro_block_size=1)

            demo_mse_list = [] 
            demo_dim_mse_list = []
            
            for t in range(min(len(gt_actions_numpy)+15, MAX_STEPS+15)):

                if t < 15:
                    # env.step returns (obs, reward, done, info)
                    obs, reward, done, info = env_gt.step(get_libero_dummy_action('openvla'))
                    obs, reward, done, info = env_pred.step(get_libero_dummy_action('openvla'))
                    t += 1
                    continue
                
                t -= 15
                gt_chunk = gt_actions_norm[t : t + CHUNK_SIZE].unsqueeze(0).float().to(DEVICE)
                
                with torch.no_grad():
                    mu, _ = vae.encode(gt_chunk)
                    # print(f"DEBUG MU: {np.round(mu.squeeze().cpu().numpy()[:5], 4)}") # <--- ADD THIS
                    pred_chunk = vae.decode(mu, text_features)
                    pred_chunk_norm = unnormalize_actions(pred_chunk, stats).squeeze(0).cpu().numpy()
                    # pred_chunk_norm = pred_chunk.squeeze(0).cpu().numpy()
                
                gt_chunk_physical = gt_actions_padded[t : t + CHUNK_SIZE]
                
                # --- NEW: Binarize the Gripper ---
                # Index 6 is the gripper. If it's >= 0, snap to 1.0. If < 0, snap to -1.0.
                pred_chunk_norm[:, 6] = np.where(pred_chunk_norm[:, 6] >= 0.15, 1.0, -1.0)
                
                # We compare the physical predictions against the raw padded ground truth
                step_mse = np.mean((pred_chunk_norm - gt_chunk_physical) ** 2)
                dim_mse = np.mean((pred_chunk_norm - gt_chunk_physical) ** 2, axis=0)

                demo_mse_list.append(step_mse)
                demo_dim_mse_list.append(dim_mse)

                if t % 50 == 0:
                    print(f"\n--- STEP {t} DEBUG ---")
                    print(f"GT Action UnNorm: {np.round(gt_chunk_physical[0], 4)}")
                    print(f"Pred Action UnNorm: {np.round(pred_chunk_norm[0], 4)}")
                    print(f"Step Total MSE: {step_mse:.6f}")
                    print(f"--------------------\n")
                
                # Execute the reconstructed action
                obs_gt, reward_gt, done_gt, info = env_gt.step(gt_chunk_physical[0])
                obs_pred, reward_pred, done_pred, info = env_pred.step(pred_chunk_norm[0])

                if SAVE_VIDEOS:
                    raw_img_gt = np.flipud(obs_gt['agentview_image'])
                    raw_img_pred = np.flipud(obs_pred['agentview_image'])

                    writer_gt.append_data(cv2.resize(raw_img_gt, (224, 224), interpolation=cv2.INTER_CUBIC))
                    writer_pred.append_data(cv2.resize(raw_img_pred, (224, 224), interpolation=cv2.INTER_CUBIC))
                    
                # if done_gt:
                #     print(f"     ✅ Task Completed at step {t} in GT!")
                    
                if done_pred:
                    print(f"     ✅ Task Completed at step {t} in PRED!")
                    break
            else:
                print(f"     ❌ Task Failed in both/ Timed out.")

            # --- NEW: Print the Final Metrics ---
            avg_demo_mse = np.mean(demo_mse_list)
            avg_demo_dim_mse = np.mean(demo_dim_mse_list, axis=0)

            task_mse_list.append(avg_demo_mse)
            task_dim_mse_list.append(avg_demo_dim_mse)

            print(f"     📊 Final Physical MSE for {demo_id}: {avg_demo_mse:.6f}")

            # --- Clean up MuJoCo Aggressively ---
            if SAVE_VIDEOS:
                writer_gt.close()
                writer_pred.close()

            env_gt.close()
            env_pred.close()
            del env_gt 
            del env_pred 
            gc.collect()
    final_task_mse = np.mean(task_mse_list)
    final_task_dim_mse = np.mean(task_dim_mse_list, axis=0)
    print(f"\n✅ Finished {task_name}. Overall Task MSE: {final_task_mse:.6f}")

    return task_name, final_task_mse, final_task_dim_mse

if __name__ == "__main__":
    # Load heavy components ONCE
    models = load_models()
    bmark = benchmark.get_benchmark_dict()[SUITE]()
    
    # Grab all task files
    HDF5_PATHS = glob(f"{DATASET_PATH}/{SUITE}_no_noops_hdf5/*.hdf5")

    # print(HDF5_PATHS)
    validation_results = {}
    for task_path in HDF5_PATHS:
        result = run_oracle_for_task(task_path, bmark, models)

        if result is not None:
            name, mse, dim_mse = result
            validation_results[name] = {
                "mse": mse,
                "dim_mse": dim_mse
            }
    
    # =================================================================
    # 🏆 FINAL EVALUATION SCOREBOARD
    # =================================================================
    print("\n\n" + "="*60)
    print("🏆 FINAL VALIDATION METRICS (ZERO-SHOT)")
    print("="*60)

    total_val_mse = []

    for name, metrics in validation_results.items():
        print(f"\nTask: {name}")
        print(f"  -> Total MSE: {metrics['mse']:.6f}")
        print(f"  -> [X, Y, Z, Roll, Pitch, Yaw, Gripper] MSE:")
        print(f"  -> {np.round(metrics['dim_mse'], 6)}")
        total_val_mse.append(metrics['mse'])

    if total_val_mse:
        print("-" * 60)
        print(f"🚀 OVERALL VALIDATION MSE: {np.mean(total_val_mse):.6f}")
    print("="*60 + "\n")
        
    print("\n🎉 Full Suite Oracle Sweep Complete!")