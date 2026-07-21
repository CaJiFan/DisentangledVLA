import os
import sys
import json
import argparse
import glob
import h5py
import torch
import numpy as np
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.disentanglers.AdvancedTextActionCVAE import TCNTextCondPriorCVAE, TCNTextWAE
from src.disentanglers.TextActionDecOnlyBetaTCVAE import TCNTextActionBetaTCVAE, TCNTextActionCVAE
from transformers import CLIPTokenizer, CLIPTextModel

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_dummy_action

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUITE = "libero_spatial"
DATASET_PATH = "/mnt/Data/cjimenez/LIBERO/libero/datasets/libero_spatial"
STATS_PATH = f"./checkpoints/new_protocol_cvae/libero_spatial/dataset_statistics.pt"
UNNORM_KEY = f'{SUITE}_no_noops'

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    # Example: rw100_d0.15_beta0.1_z64_chunk8_protA_cyc4_cond_prior_seed_1_step_250000.pt
    parts = filename.replace(".pt", "").split("_")
    
    cfg = {}
    for i, p in enumerate(parts):
        if p.startswith("z") and p[1:].isdigit():
            cfg['z_dim'] = int(p[1:])
        elif p.startswith("beta"):
            cfg['beta'] = float(p.replace("beta", ""))
        elif p.startswith("chunk"):
            cfg['chunk_size'] = int(p.replace("chunk", ""))
        elif p == "seed":
            cfg['seed'] = int(parts[i+1])
        elif p == "step":
            cfg['step'] = int(parts[i+1])
            
    # Determine arch
    if "cond_prior" in filename:
        cfg['arch'] = "cond_prior"
    elif "wae" in filename:
        cfg['arch'] = "wae"
    elif "dec_only" in filename:
        cfg['arch'] = "dec_only"
    elif "full_cond" in filename:
        cfg['arch'] = "full_cond"
    else:
        cfg['arch'] = "unknown"
        
    return cfg

def load_model(filepath, cfg):
    print(f"Loading {cfg['arch']} with z={cfg.get('z_dim')} from {filepath}")
    
    chunk_size = cfg.get('chunk_size', 8)
    n_blocks = max(3, (chunk_size - 1).bit_length())
    
    if cfg['arch'] == "cond_prior":
        vae = TCNTextCondPriorCVAE(
            action_dim=7, chunk_size=chunk_size, 
            latent_dim=cfg.get('z_dim', 64), text_emb_dim=512,
            beta=cfg.get('beta', 0.1), dropout=0.0, n_blocks=n_blocks
        ).to(DEVICE)
    elif cfg['arch'] == "wae":
        vae = TCNTextWAE(
            action_dim=7, chunk_size=chunk_size, 
            latent_dim=cfg.get('z_dim', 64), text_emb_dim=512,
            beta=cfg.get('beta', 0.1), dropout=0.0, n_blocks=n_blocks
        ).to(DEVICE)
    elif cfg['arch'] == "dec_only":
        vae = TCNTextActionBetaTCVAE(
            action_dim=7, chunk_size=chunk_size, 
            latent_dim=cfg.get('z_dim', 64), text_emb_dim=512,
            beta=cfg.get('beta', 0.1), dropout=0.0, n_blocks=n_blocks
        ).to(DEVICE)
    elif cfg['arch'] == "full_cond":
        vae = TCNTextActionCVAE(
            action_dim=7, chunk_size=chunk_size, 
            latent_dim=cfg.get('z_dim', 64), text_emb_dim=512,
            beta=cfg.get('beta', 0.1), dropout=0.0,
            enc_text_gate_init=0.0, n_blocks=n_blocks # Full cond enables the text gate
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown architecture in filename: {filepath}")

    vae.load_state_dict(torch.load(filepath, map_location=DEVICE))
    vae.eval()
    return vae

def evaluate_disentangler(filepath, num_rollouts=20):
    cfg = parse_filename(filepath)
    vae = load_model(filepath, cfg)
    
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    
    action_stats = torch.load(STATS_PATH)
    stats = action_stats[UNNORM_KEY]['action']
    action_min = torch.tensor(stats['min']).float().to(DEVICE)
    action_max = torch.tensor(stats['max']).float().to(DEVICE)
    action_mask = torch.tensor(stats['mask']).float().to(DEVICE)

    bmark = benchmark.get_benchmark_dict()[SUITE]()
    
    total_success = 0
    total_demos = 0
    
    chunk_size = cfg.get('chunk_size', 8)

    print(f"\nEvaluating: {os.path.basename(filepath)}")
    print("="*60)
    
    for task_id in range(bmark.get_num_tasks()):
        task = bmark.get_task(task_id)
        instruction = task.language
        task_name = task.name
        
        # Open HDF5 for this task
        hdf5_path = os.path.join(DATASET_PATH, f"{task_name}_demo.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"Skipping {task_name}, dataset not found at {hdf5_path}")
            continue
            
        text_inputs = clip_tok([instruction], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_emb = clip_enc(**text_inputs).pooler_output
            
        task_successes = 0
        
        with h5py.File(hdf5_path, "r") as f:
            demos = list(f["data"].keys())
            demos_to_eval = demos[:num_rollouts]
            
            env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
            env = OffScreenRenderEnv(**env_args)
            
            for demo_id in tqdm.tqdm(demos_to_eval, desc=f"Task {task_id+1}/{bmark.get_num_tasks()} ({task_name})", leave=False):
                init_state = f[f"data/{demo_id}/states"][0]
                gt_actions = f[f"data/{demo_id}/actions"][:]
                
                gt_tensor = torch.tensor(gt_actions).float().to(DEVICE)
                norm_gt = (gt_tensor - action_min) / (action_max - action_min + 1e-5)
                norm_gt = norm_gt * 2.0 - 1.0
                norm_gt = norm_gt * action_mask + gt_tensor * (1.0 - action_mask)
                
                env.reset()
                env.set_init_state(init_state)
                
                seq_len = len(norm_gt)
                demo_success = False
                
                # Execute dummy actions for 15 steps (Libero wait period)
                for _ in range(15):
                    env.step(get_libero_dummy_action('openvla'))
                
                t = 0
                while t < seq_len:
                    if t + chunk_size <= seq_len:
                        chunk = norm_gt[t : t + chunk_size]
                    else:
                        valid_len = seq_len - t
                        pad_len = chunk_size - valid_len
                        padding = norm_gt[-1].repeat(pad_len, 1)
                        chunk = torch.cat([norm_gt[t : seq_len], padding], dim=0)

                    chunk = chunk.unsqueeze(0)

                    with torch.no_grad():
                        encode_args = (chunk, text_emb) if vae.encode.__code__.co_argcount > 2 else (chunk,)
                        mu, _ = vae.encode(*encode_args)
                        pred_chunk_norm = vae.decode(mu, text_emb)[0]

                    steps_to_execute = chunk_size if t + chunk_size <= seq_len else seq_len - t
                    
                    for i in range(steps_to_execute):
                        pred_action_norm = pred_chunk_norm[i]
                        pred_action_norm = torch.clamp(pred_action_norm, -1.0, 1.0)
                        pred_action_norm = torch.where(action_mask > 0, pred_action_norm, norm_gt[t+i])
                        
                        unnorm_action = ((pred_action_norm + 1.0) / 2.0) * (action_max - action_min + 1e-5) + action_min
                        unnorm_action = torch.where(action_mask > 0, unnorm_action, gt_tensor[t+i])
                        
                        act_np = unnorm_action.cpu().numpy()
                        act_np[6] = 1.0 if act_np[6] >= 0.0 else -1.0
                        
                        _, _, done, _ = env.step(act_np)
                        if done:
                            demo_success = True
                            break
                            
                    if demo_success:
                        break
                    t += chunk_size
                    
                if demo_success:
                    task_successes += 1
                total_demos += 1
                total_success += 1 if demo_success else 0
                
        print(f"  {task_name}: {task_successes}/{len(demos_to_eval)} ({task_successes/len(demos_to_eval)*100:.1f}%)")
        
    final_sr = total_success / total_demos * 100
    print("="*60)
    print(f"FINAL SUCCESS RATE: {final_sr:.2f}% ({total_success}/{total_demos})")
    print("="*60)
    return final_sr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to a specific .pt file")
    parser.add_argument("--dir", type=str, help="Path to a directory containing .pt files (will evaluate all)")
    parser.add_argument("--rollouts", type=int, default=20, help="Number of rollouts per task")
    args = parser.parse_args()
    
    if args.checkpoint:
        evaluate_disentangler(args.checkpoint, args.rollouts)
    elif args.dir:
        files = glob.glob(os.path.join(args.dir, "*.pt"))
        # Strictly filter for the new Protocol A runs (excluding legacy ones)
        files = [f for f in files if "step_250000.pt" in f and ("protA" in f or "d0.15" in f)]
        
        results_file = os.path.join(args.dir, "eval_results.json")
        results = {}
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                
        for f in sorted(files):
            ckpt_name = os.path.basename(f)
            if ckpt_name in results:
                print(f"Skipping {ckpt_name}, already evaluated: {results[ckpt_name]:.2f}%")
                continue
                
            sr = evaluate_disentangler(f, args.rollouts)
            results[ckpt_name] = sr
            
            with open(results_file, 'w') as out_f:
                json.dump(results, out_f, indent=4)
            
        print("\n\n" + "#"*60)
        print("SUMMARY TABLE")
        print("#"*60)
        for k, v in results.items():
            print(f"{k}: {v:.2f}%")
    else:
        print("Please provide --checkpoint or --dir")
