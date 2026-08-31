import os
import sys
import json
import argparse
import glob
import h5py
import torch
import numpy as np
import tqdm
import imageio
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.disentanglers.AdvancedTextActionCVAE import TCNTextCondPriorCVAE, TCNTextWAE
from src.disentanglers.TextActionDecOnlyBetaTCVAE import TCNTextActionBetaTCVAE, TCNTextActionCVAE
from transformers import CLIPTokenizer, CLIPTextModel

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_dummy_action

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_filename(filepath):
    filename = os.path.basename(filepath)
    # Example: rw100_d0.15_beta0.1-0.5_z64_chunk8_protA_cyc4_cond_prior_seed_1_step_250000.pt
    parts = filename.replace(".pt", "").split("_")
    
    cfg = {}
    for i, p in enumerate(parts):
        if p.startswith("z") and p[1:].isdigit():
            cfg['z_dim'] = int(p[1:])
        elif p.startswith("beta"):
            b_parts = p.replace("beta", "").split("-")
            cfg['beta'] = float(b_parts[0])
            cfg['beta_high'] = float(b_parts[1]) if len(b_parts) > 1 else float(b_parts[0])
        elif p.startswith("chunk"):
            cfg['chunk_size'] = int(p.replace("chunk", ""))
        elif p == "seed":
            cfg['seed'] = int(parts[i+1])
        elif p == "step":
            cfg['step'] = int(parts[i+1])
            
    cfg['use_state'] = "_state" in filename
    
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
            beta=cfg.get('beta', 0.1), dropout=0.0, n_blocks=n_blocks,
            use_state=cfg.get('use_state', False), state_dim=8
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

    ckpt = torch.load(filepath, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        vae.load_state_dict(ckpt["model_state_dict"])
    else:
        vae.load_state_dict(ckpt)
    vae.eval()
    return vae

def evaluate_disentangler(filepath, num_rollouts=20, suite="libero_object", save_video_dir=None, use_prior=False, exec_steps=1, temporal_ensemble=False, ensemble_k=0.01):
    cfg = parse_filename(filepath)
    vae = load_model(filepath, cfg)
    
    ckpt_name = os.path.splitext(os.path.basename(filepath))[0]
    out_video_dir = save_video_dir or f"./eval_results_videos/{suite}/{ckpt_name}"
    os.makedirs(out_video_dir, exist_ok=True)
    
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    
    stats_path = f"./checkpoints/new_protocol_cvae/{suite}/dataset_statistics.pt"
    dataset_path = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{suite}"
    unnorm_key = f"{suite}_no_noops"
    
    action_stats = torch.load(stats_path)
    stats = action_stats[unnorm_key]['action']
    action_min = torch.tensor(stats['min']).float().to(DEVICE)
    action_max = torch.tensor(stats['max']).float().to(DEVICE)
    action_mask = torch.tensor(stats['mask']).float().to(DEVICE)

    bmark = benchmark.get_benchmark_dict()[suite]()
    
    total_success = 0
    total_demos = 0
    
    chunk_size = cfg.get('chunk_size', 8)

    print(f"\nEvaluating: {os.path.basename(filepath)}")
    if temporal_ensemble:
        print(f"🔄 Execution Mode: Temporal Ensembling (k={ensemble_k}, chunk_size={chunk_size}) | Prior Mode: {use_prior}")
    else:
        print(f"🔄 Execution Mode: Receding Horizon (exec_steps={exec_steps}/{chunk_size}) | Prior Mode: {use_prior}")
    print(f"📁 Success Videos Output Dir: {out_video_dir}")
    print("="*60)
    
    for task_id in range(bmark.get_num_tasks()):
        task = bmark.get_task(task_id)
        instruction = task.language
        task_name = task.name
        
        task_emb_path = f"./checkpoints/new_protocol_cvae/{suite}/task_embeddings/{task_name}_clip.pt"
        if os.path.exists(task_emb_path):
            text_emb = torch.load(task_emb_path).to(DEVICE)
        else:
            inputs = clip_tok(instruction, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                text_emb = clip_enc(**inputs).last_hidden_state[:, 0, :]
            
        # Open HDF5 for this task
        hdf5_path = os.path.join(dataset_path, f"{task_name}_demo.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"Skipping {task_name}, dataset not found at {hdf5_path}")
            continue
            
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
                frames = []
                
                # Execute dummy actions for 15 steps (Libero wait period)
                for _ in range(15):
                    obs, _, _, _ = env.step(get_libero_dummy_action('openvla'))
                    img = np.flipud(obs['agentview_image'])
                    frames.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
                
                t = 0
                if temporal_ensemble:
                    action_dim = len(action_min)
                    all_time_actions = np.zeros((seq_len, seq_len + chunk_size, action_dim), dtype=np.float32)
                    
                    while t < seq_len:
                        if getattr(vae, 'use_state', False):
                            from experiments.robot.libero.libero_utils import quat2axisangle
                            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                            state_tensor = torch.tensor(state).float().to(DEVICE).unsqueeze(0)
                        else:
                            state_tensor = None

                        with torch.no_grad():
                            decode_text = torch.zeros_like(text_emb) if getattr(vae, 'no_text_decoder', False) or "SPIRL" in filepath or "spirl" in filepath else text_emb
                            if use_prior and hasattr(vae, 'get_prior'):
                                if state_tensor is not None:
                                    mu, _ = vae.get_prior(text_emb, state_tensor)
                                else:
                                    mu, _ = vae.get_prior(text_emb)
                            else:
                                if t + chunk_size <= seq_len:
                                    chunk = norm_gt[t : t + chunk_size]
                                else:
                                    valid_len = seq_len - t
                                    pad_len = chunk_size - valid_len
                                    padding = norm_gt[-1].repeat(pad_len, 1)
                                    chunk = torch.cat([norm_gt[t : seq_len], padding], dim=0)
                                chunk = chunk.unsqueeze(0)
                                encode_args = (chunk, text_emb) if vae.encode.__code__.co_argcount > 2 else (chunk,)
                                mu, _ = vae.encode(*encode_args)
                            
                            if state_tensor is not None:
                                pred_chunk_norm = vae.decode(mu, decode_text, state_tensor)[0]
                            else:
                                pred_chunk_norm = vae.decode(mu, decode_text)[0]

                        all_time_actions[t, t : t + chunk_size] = pred_chunk_norm.cpu().numpy()
                        
                        start_idx = max(0, t - chunk_size + 1)
                        actions_for_curr_step = all_time_actions[start_idx : t + 1, t]
                        
                        # Weighting: newest prediction (index -1 in actions_for_curr_step) has age 0
                        num_actions = len(actions_for_curr_step)
                        ages = np.arange(num_actions)[::-1]
                        weights = np.exp(-ensemble_k * ages)
                        weights = weights / weights.sum()
                        
                        pred_action_norm = (actions_for_curr_step * weights[:, None]).sum(axis=0)
                        pred_action_norm = np.clip(pred_action_norm, -1.0, 1.0)
                        
                        unnorm_action = ((pred_action_norm + 1.0) / 2.0) * (action_max.cpu().numpy() - action_min.cpu().numpy() + 1e-5) + action_min.cpu().numpy()
                        unnorm_action = unnorm_action * action_mask.cpu().numpy() + pred_action_norm * (1.0 - action_mask.cpu().numpy())
                        
                        act_np = unnorm_action.copy()
                        act_np[6] = 1.0 if pred_action_norm[6] > 0.0 else -1.0
                        
                        obs, _, done, _ = env.step(act_np)
                        img = np.flipud(obs['agentview_image'])
                        frames.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
                        
                        if done:
                            demo_success = True
                            break
                        t += 1
                else:
                    while t < seq_len:
                        if t + chunk_size <= seq_len:
                            chunk = norm_gt[t : t + chunk_size]
                        else:
                            valid_len = seq_len - t
                            pad_len = chunk_size - valid_len
                            padding = norm_gt[-1].repeat(pad_len, 1)
                            chunk = torch.cat([norm_gt[t : seq_len], padding], dim=0)

                        chunk = chunk.unsqueeze(0)

                        if getattr(vae, 'use_state', False):
                            from experiments.robot.libero.libero_utils import quat2axisangle
                            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                            state_tensor = torch.tensor(state).float().to(DEVICE).unsqueeze(0)
                        else:
                            state_tensor = None

                        with torch.no_grad():
                            decode_text = torch.zeros_like(text_emb) if getattr(vae, 'no_text_decoder', False) or "SPIRL" in filepath or "spirl" in filepath else text_emb
                            if use_prior and hasattr(vae, 'get_prior'):
                                if state_tensor is not None:
                                    mu, _ = vae.get_prior(text_emb, state_tensor)
                                else:
                                    mu, _ = vae.get_prior(text_emb)
                            else:
                                encode_args = (chunk, text_emb) if vae.encode.__code__.co_argcount > 2 else (chunk,)
                                mu, _ = vae.encode(*encode_args)
                            
                            if state_tensor is not None:
                                pred_chunk_norm = vae.decode(mu, decode_text, state_tensor)[0]
                            else:
                                pred_chunk_norm = vae.decode(mu, decode_text)[0]

                        steps_to_execute = chunk_size if t + chunk_size <= seq_len else seq_len - t
                        num_steps_this_loop = min(exec_steps, steps_to_execute)
                        
                        for i in range(num_steps_this_loop):
                            pred_action_norm = pred_chunk_norm[i]
                            pred_action_norm = torch.clamp(pred_action_norm, -1.0, 1.0)

                            gripper_norm = pred_action_norm[6]
                            unnorm_action = ((pred_action_norm + 1.0) / 2.0) * (action_max - action_min + 1e-5) + action_min

                            unnorm_action = unnorm_action * action_mask + pred_action_norm * (1.0 - action_mask)

                            act_np = unnorm_action.cpu().numpy()
                            act_np[6] = 1.0 if pred_action_norm[6] > 0.0 else -1.0
                            
                            obs, _, done, _ = env.step(act_np)
                            img = np.flipud(obs['agentview_image'])
                            frames.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))

                            if done:
                                demo_success = True
                                break
                                
                        if demo_success:
                            break
                        t += num_steps_this_loop
                    
                if demo_success:
                    task_successes += 1
                    video_save_path = os.path.join(out_video_dir, f"success_{task_name}_{demo_id}.mp4")
                    writer = imageio.get_writer(video_save_path, fps=30, macro_block_size=1)
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()
                    print(f"  🎥 Saved SUCCESS video: {video_save_path}")

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
    parser.add_argument("--suite", type=str, default="libero_object", choices=["libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--checkpoint", type=str, help="Path to a specific .pt file")
    parser.add_argument("--dir", type=str, help="Path to a directory containing .pt files (will evaluate all)")
    parser.add_argument("--rollouts", type=int, default=20, help="Number of rollouts per task")
    parser.add_argument("--save_video_dir", type=str, default=None, help="Custom output directory to save success MP4 videos")
    parser.add_argument("--use_prior", action="store_true", help="Sample latent z from prior p(z|text) instead of GT chunk posterior q(z|chunk)")
    parser.add_argument("--exec_steps", type=int, default=1, help="Number of steps to execute per chunk before re-planning (1 = full receding horizon)")
    parser.add_argument("--temporal_ensemble", action="store_true", help="Enable test-time temporal ensembling over overlapping action chunks")
    parser.add_argument("--ensemble_k", type=float, default=0.01, help="Exponential weighting parameter for temporal ensembling (default: 0.01)")
    args = parser.parse_args()
    
    if args.checkpoint:
        evaluate_disentangler(
            args.checkpoint,
            num_rollouts=args.rollouts,
            suite=args.suite,
            save_video_dir=args.save_video_dir,
            use_prior=args.use_prior,
            exec_steps=args.exec_steps,
            temporal_ensemble=args.temporal_ensemble,
            ensemble_k=args.ensemble_k
        )
    elif args.dir:
        files = glob.glob(os.path.join(args.dir, "*.pt"))
        files = [f for f in files if "best.pt" in f or "step_1000000.pt" in f or "step_250000.pt" in f]
        
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
                
            sr = evaluate_disentangler(
                f,
                num_rollouts=args.rollouts,
                suite=args.suite,
                use_prior=args.use_prior,
                exec_steps=args.exec_steps,
                temporal_ensemble=args.temporal_ensemble,
                ensemble_k=args.ensemble_k
            )
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
