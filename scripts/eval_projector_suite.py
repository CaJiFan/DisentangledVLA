import os
import sys
import argparse
import glob
import h5py
import torch
import numpy as np
import tqdm
import imageio
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForVision2Seq, AutoProcessor, CLIPTokenizer, CLIPTextModel
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_dummy_action, quat2axisangle

from src.disentanglers import TCNTextCondPriorCVAE, TCNTextActionBetaTCVAE, TCNTextActionCVAE
from src.projectors import ProbabilisticActionProjector, TransformerActionProjector, ContinuousFlowMatcher
from utils.data import _make_openvla_emb_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-Loop Projector Policy Evaluation with Temporal Ensembling")
    parser.add_argument("--projector_checkpoint", type=str, required=True,
                        help="Path to trained projector checkpoint (.pt)")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to trained VAE checkpoint (.pt)")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_long"])
    parser.add_argument("--vla_checkpoint", type=str, default="openvla/openvla-7b")
    parser.add_argument("--vla_type", type=str, default="openvla")
    parser.add_argument("--vla_layer_idx", type=int, default=-1)
    parser.add_argument("--use_vision_pool", action="store_true", default=False)
    
    # Projector Arch
    parser.add_argument("--projector_type", type=str, default=None, choices=["mlp", "prob", "flow"],
                        help="Auto-detected if None")
    parser.add_argument("--projector_arch", type=str, default=None,
                        choices=["mlp", "transformer", "flow_transformer", "bottleneck", "linear"],
                        help="Auto-detected if None")
    parser.add_argument("--vae_type", type=str, default=None,
                        choices=["cond_prior", "text_cond_beta_tcvae", "text_cvae"],
                        help="Auto-detected if None")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--xfmr_d_model", type=int, default=256)
    parser.add_argument("--xfmr_num_heads", type=int, default=8)
    parser.add_argument("--xfmr_num_layers", type=int, default=3)
    parser.add_argument("--xfmr_ffn_dim", type=int, default=512)
    parser.add_argument("--flow_steps", type=int, default=10, help="Number of ODE steps for Flow Matching")
    
    # Temporal Ensembling
    parser.add_argument("--temporal_ensemble", action="store_true", default=True,
                        help="Enable temporal ensembling across overlapping action chunks")
    parser.add_argument("--ensemble_k", type=float, default=0.01,
                        help="Exponential decay factor k for temporal ensembling (0.01 = flat average, 0.1-0.25 = reactive)")
    parser.add_argument("--exec_steps", type=int, default=1,
                        help="Execution horizon if temporal ensembling is disabled")
    
    # Eval Settings
    parser.add_argument("--num_rollouts", type=int, default=20,
                        help="Number of evaluation rollouts per task (e.g. 20 or 50)")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Max simulation steps per episode")
    parser.add_argument("--save_videos", type=lambda x: str(x).lower() not in ('false', '0', 'no'), default=True,
                        help="Save MP4 videos for all successful rollouts (default: True)")
    parser.add_argument("--video_dir", type=str, default="./eval_results_videos/projectors/")
    parser.add_argument("--seed", type=int, default=7)
    
    return parser.parse_args()


def detect_config_from_path(proj_path, vae_path, args):
    proj_name = os.path.basename(proj_path)
    vae_name = os.path.basename(vae_path)
    
    # Detect VAE Type
    if args.vae_type is None:
        if "cond_prior" in vae_name:
            args.vae_type = "cond_prior"
        elif "cvae" in vae_name or "full_cond" in vae_name:
            args.vae_type = "text_cvae"
        else:
            args.vae_type = "text_cond_beta_tcvae"
            
    # Detect Projector Type & Arch
    if args.projector_arch is None:
        if "flow_transformer" in proj_name:
            args.projector_arch = "flow_transformer"
            args.projector_type = "flow"
        elif "transformer" in proj_name:
            args.projector_arch = "transformer"
            args.projector_type = "prob"
        elif "bottleneck" in proj_name:
            args.projector_arch = "bottleneck"
            args.projector_type = "prob"
        elif "linear" in proj_name:
            args.projector_arch = "linear"
            args.projector_type = "prob"
        else:
            args.projector_arch = "mlp"
            args.projector_type = "prob" if "prob" in proj_name else "mlp"

    if "flow" in proj_name:
        args.projector_type = "flow"

    # Detect z_dim
    import re
    z_match = re.search(r'z(?:dim_)?(\d+)', proj_path) or re.search(r'z(\d+)', vae_name)
    if z_match:
        args.z_dim = int(z_match.group(1))

    # Detect state conditioning
    args.use_state_cond = ("_state" in vae_name) or ("_state" in proj_name)
    
    print(f"📋 Detected Config: VAE={args.vae_type} (z={args.z_dim}, use_state={args.use_state_cond}) | Projector={args.projector_arch} (type={args.projector_type})")


def load_models(args):
    print("\n📦 Loading Models...")
    
    # 1. Load VAE
    chunk_size = args.chunk_size
    n_blocks = max(3, (chunk_size - 1).bit_length())
    
    if args.vae_type == "cond_prior":
        vae = TCNTextCondPriorCVAE(
            action_dim=7, chunk_size=chunk_size, latent_dim=args.z_dim,
            text_emb_dim=512, beta=0.1, dropout=0.0, n_blocks=n_blocks,
            use_state=args.use_state_cond, state_dim=8
        ).to(DEVICE)
    elif args.vae_type == "text_cvae":
        vae = TCNTextActionCVAE(
            action_dim=7, chunk_size=chunk_size, latent_dim=args.z_dim,
            text_emb_dim=512, beta=0.1, dropout=0.0, n_blocks=n_blocks
        ).to(DEVICE)
    else:
        vae = TCNTextActionBetaTCVAE(
            action_dim=7, chunk_size=chunk_size, latent_dim=args.z_dim,
            text_emb_dim=512, beta=0.1, dropout=0.0, n_blocks=n_blocks
        ).to(DEVICE)
        
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=DEVICE)
    if isinstance(vae_ckpt, dict) and "model_state_dict" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["model_state_dict"])
    else:
        vae.load_state_dict(vae_ckpt)
    vae.eval()
    print("  ✅ Frozen VAE Decoder loaded successfully.")
    
    # 2. Load Projector
    vla_dim = 4096  # OpenVLA default
    if args.projector_arch == "flow_transformer":
        inner_proj = TransformerActionProjector(
            input_dim=vla_dim, latent_dim=args.z_dim,
            d_model=args.xfmr_d_model, num_heads=args.xfmr_num_heads,
            num_layers=args.xfmr_num_layers, ffn_dim=args.xfmr_ffn_dim,
            is_flow=True
        )
        projector = ContinuousFlowMatcher(inner_proj, latent_dim=args.z_dim).to(DEVICE)
    elif args.projector_arch == "transformer":
        projector = TransformerActionProjector(
            input_dim=vla_dim, latent_dim=args.z_dim,
            d_model=args.xfmr_d_model, num_heads=args.xfmr_num_heads,
            num_layers=args.xfmr_num_layers, ffn_dim=args.xfmr_ffn_dim,
            is_flow=False
        ).to(DEVICE)
    elif args.projector_arch == "mlp":
        projector = ProbabilisticActionProjector(
            input_dim=vla_dim, latent_dim=args.z_dim
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported projector arch: {args.projector_arch}")
        
    proj_ckpt = torch.load(args.projector_checkpoint, map_location=DEVICE)
    if isinstance(proj_ckpt, dict) and "model_state_dict" in proj_ckpt:
        projector.load_state_dict(proj_ckpt["model_state_dict"])
    elif isinstance(proj_ckpt, dict) and "state_dict" in proj_ckpt:
        projector.load_state_dict(proj_ckpt["state_dict"])
    else:
        projector.load_state_dict(proj_ckpt)
    projector.eval()
    print("  ✅ Projector loaded successfully.")
    
    # 3. Load OpenVLA
    print(f"  ⏳ Loading VLA ({args.vla_checkpoint})...")
    processor = AutoProcessor.from_pretrained(args.vla_checkpoint, trust_remote_code=True)
    vla_model = AutoModelForVision2Seq.from_pretrained(
        args.vla_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE).eval()
    
    emb_fn = _make_openvla_emb_fn(
        vla_model, processor, DEVICE,
        use_vision_pool=args.use_vision_pool,
        vla_layer_idx=args.vla_layer_idx
    )
    print("  ✅ OpenVLA visual encoder ready.")
    
    # 4. Load CLIP for text conditioning
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    
    return vae, projector, emb_fn, clip_tok, clip_enc


def evaluate_suite(args):
    detect_config_from_path(args.projector_checkpoint, args.vae_checkpoint, args)
    vae, projector, emb_fn, clip_tok, clip_enc = load_models(args)
    
    # Stats for action unnormalization
    raw_suite = args.suite.replace("_no_noops", "")
    stats_path = f"./checkpoints/new_protocol_cvae/{raw_suite}/dataset_statistics.pt"
    if not os.path.exists(stats_path):
        stats_path = f"./checkpoints/text_tcvae/{raw_suite}/dataset_statistics.pt"
    action_stats = torch.load(stats_path, map_location=DEVICE)
    suite_key = f"{raw_suite}_no_noops" if f"{raw_suite}_no_noops" in action_stats else list(action_stats.keys())[0]
    stats = action_stats[suite_key]['action']
    action_min = torch.tensor(stats['min']).float().to(DEVICE)
    action_max = torch.tensor(stats['max']).float().to(DEVICE)
    action_mask = torch.tensor(stats['mask']).float().to(DEVICE)
    
    bmark = benchmark.get_benchmark_dict()[args.suite]()
    dataset_path = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{raw_suite}_no_noops_hdf5"
    
    total_successes = 0
    total_rollouts = 0
    task_results = {}
    
    print("\n" + "="*70)
    print(f"🚀 STARTING PROJECTOR EVALUATION: {args.suite.upper()}")
    print(f"   Projector: {os.path.basename(args.projector_checkpoint)}")
    print(f"   VAE:       {os.path.basename(args.vae_checkpoint)}")
    print(f"   Mode:      {'Temporal Ensembling (k=' + str(args.ensemble_k) + ')' if args.temporal_ensemble else 'Receding Horizon (exec=' + str(args.exec_steps) + ')'}")
    print("="*70)
    
    for task_id in range(bmark.get_num_tasks()):
        task = bmark.get_task(task_id)
        task_name = task.name
        instruction = task.language
        
        # Get CLIP text embedding
        t_inputs = clip_tok([instruction], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_emb = clip_enc(**t_inputs).pooler_output
            
        hdf5_path = os.path.join(dataset_path, f"{task_name}_demo.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"⚠️ Dataset missing for {task_name}, skipping.")
            continue
            
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
            demos_to_eval = demo_keys[:args.num_rollouts]
            
            env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
            env = OffScreenRenderEnv(**env_args)
            task_successes = 0
            
            pbar = tqdm.tqdm(demos_to_eval, desc=f"[{task_id+1}/{bmark.get_num_tasks()}] {task_name[:35]}", leave=True)
            for demo_id in pbar:
                init_state = f[f"data/{demo_id}/states"][0]
                obs = env.reset()
                env.set_init_state(init_state)
                
                # Settle physics for 15 steps
                for _ in range(15):
                    obs, _, _, _ = env.step(get_libero_dummy_action('openvla'))
                    
                frames = []
                task_done = False
                
                if args.temporal_ensemble:
                    action_dim = len(action_min)
                    all_time_actions = np.zeros((args.max_steps, args.max_steps + args.chunk_size, action_dim), dtype=np.float32)
                    
                    for t in range(args.max_steps):
                        img_pil = Image.fromarray(obs['agentview_image'].astype(np.uint8))
                        
                        with torch.no_grad():
                            vla_emb = emb_fn(img_pil, instruction).to(DEVICE)
                            vla_emb = torch.nn.functional.normalize(vla_emb, dim=-1)
                            
                            # Projector forward
                            if args.projector_type == "flow":
                                z_dim = projector.latent_dim
                                B_vla = vla_emb.size(0)
                                z_t = torch.randn(B_vla, z_dim, device=DEVICE)
                                dt = 1.0 / args.flow_steps
                                for step_i in range(args.flow_steps):
                                    t_tensor = torch.ones(B_vla, device=DEVICE) * (step_i / float(args.flow_steps))
                                    v = projector(vla_emb, z_t, t_tensor)
                                    z_t = z_t + v * dt
                                pred_mu = z_t
                            else:
                                _, pred_mu, _ = projector(vla_emb)
                                
                            # VAE decode
                            decode_text = torch.zeros_like(text_emb) if getattr(vae, 'no_text_decoder', False) or "SPIRL" in args.vae_checkpoint else text_emb
                            if getattr(vae, 'use_state', False):
                                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                                state_tensor = torch.tensor(state).float().to(DEVICE).unsqueeze(0)
                                pred_chunk_norm = vae.decode(pred_mu, decode_text, state_tensor)[0]
                            else:
                                pred_chunk_norm = vae.decode(pred_mu, decode_text)[0]
                                
                        all_time_actions[t, t : t + args.chunk_size] = pred_chunk_norm.cpu().numpy()
                        
                        start_idx = max(0, t - args.chunk_size + 1)
                        actions_for_curr_step = all_time_actions[start_idx : t + 1, t]
                        
                        num_actions = len(actions_for_curr_step)
                        ages = np.arange(num_actions)[::-1]
                        weights = np.exp(-args.ensemble_k * ages)
                        weights = weights / weights.sum()
                        
                        pred_action_norm = (actions_for_curr_step * weights[:, None]).sum(axis=0)
                        pred_action_norm = np.clip(pred_action_norm, -1.0, 1.0)
                        
                        unnorm_action = (pred_action_norm + 1.0) / 2.0 * (action_max.cpu().numpy() - action_min.cpu().numpy() + 1e-5) + action_min.cpu().numpy()
                        pred_action = unnorm_action * action_mask.cpu().numpy() + pred_action_norm * (1.0 - action_mask.cpu().numpy())
                        action_np = pred_action.copy()
                        action_np[-1] = 1.0 if pred_action_norm[-1] > 0.0 else -1.0
                        
                        obs, reward, done, info = env.step(action_np)
                        if args.save_videos:
                            img = np.flipud(obs['agentview_image'])
                            frames.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
                            
                        if done:
                            task_done = True
                            break
                else:
                    # Receding Horizon
                    for t in range(0, args.max_steps, args.exec_steps):
                        img_pil = Image.fromarray(obs['agentview_image'].astype(np.uint8))
                        with torch.no_grad():
                            vla_emb = emb_fn(img_pil, instruction).to(DEVICE)
                            vla_emb = torch.nn.functional.normalize(vla_emb, dim=-1)
                            if args.projector_type == "flow":
                                z_dim = projector.latent_dim
                                B_vla = vla_emb.size(0)
                                z_t = torch.randn(B_vla, z_dim, device=DEVICE)
                                dt = 1.0 / args.flow_steps
                                for step_i in range(args.flow_steps):
                                    t_tensor = torch.ones(B_vla, device=DEVICE) * (step_i / float(args.flow_steps))
                                    v = projector(vla_emb, z_t, t_tensor)
                                    z_t = z_t + v * dt
                                pred_mu = z_t
                            else:
                                _, pred_mu, _ = projector(vla_emb)
                                
                            decode_text = torch.zeros_like(text_emb) if getattr(vae, 'no_text_decoder', False) or "SPIRL" in args.vae_checkpoint else text_emb
                            if getattr(vae, 'use_state', False):
                                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                                state_tensor = torch.tensor(state).float().to(DEVICE).unsqueeze(0)
                                pred_chunk_norm = vae.decode(pred_mu, decode_text, state_tensor)[0]
                            else:
                                pred_chunk_norm = vae.decode(pred_mu, decode_text)[0]
                                
                        for k in range(args.exec_steps):
                            pred_action_norm = pred_chunk_norm[k]
                            unnorm_action = (pred_action_norm + 1.0) / 2.0 * (action_max - action_min + 1e-5) + action_min
                            pred_action = unnorm_action * action_mask + pred_action_norm * (1.0 - action_mask)
                            action_np = pred_action.cpu().numpy()
                            action_np[-1] = 1.0 if pred_action_norm[-1] > 0.0 else -1.0
                            
                            obs, reward, done, info = env.step(action_np)
                            if args.save_videos:
                                img = np.flipud(obs['agentview_image'])
                                frames.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
                            if done:
                                task_done = True
                                break
                        if task_done:
                            break
                            
                if task_done:
                    task_successes += 1
                    if args.save_videos and len(frames) > 0:
                        os.makedirs(out_video_dir, exist_ok=True)
                        video_save_path = os.path.join(out_video_dir, f"success_{task_name}_{demo_id}.mp4")
                        writer = imageio.get_writer(video_save_path, fps=30, macro_block_size=1)
                        for frame in frames:
                            writer.append_data(frame)
                        writer.close()
                        print(f"    🎥 Saved SUCCESS video: {os.path.basename(video_save_path)}")
                    
                pbar.set_postfix({"SR": f"{task_successes / (pbar.n + 1) * 100:.1f}%"})
                
            env.close()
            
        task_sr = task_successes / len(demos_to_eval) * 100.0
        task_results[task_name] = (task_successes, len(demos_to_eval), task_sr)
        total_successes += task_successes
        total_rollouts += len(demos_to_eval)
        print(f"  👉 Task {task_id+1}: {task_name} → {task_sr:.1f}% ({task_successes}/{len(demos_to_eval)})")
        
    overall_sr = total_successes / total_rollouts * 100.0 if total_rollouts > 0 else 0.0
    print("\n" + "="*70)
    print(f"🏆 FINAL SUCCESS RATE: {overall_sr:.2f}% ({total_successes}/{total_rollouts})")
    if args.save_videos:
        print(f"📁 Success Videos Saved To: {out_video_dir}")
    print("="*70)
    
    return overall_sr, task_results


if __name__ == "__main__":
    args = parse_args()
    evaluate_suite(args)
