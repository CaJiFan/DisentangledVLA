import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLE_SERVICE"] = "True"
os.environ["WANDB_START_METHOD"] = "thread"


from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import torch
from PIL import Image
from libero.libero import benchmark
import wandb

# --- YOUR CUSTOM IMPORTS ---
sys.path.append(os.getcwd()) 
from transformers import AutoModelForVision2Seq, AutoProcessor
from src.disentanglers.ActionBetaTCVAE import ActionBetaTCVAE
from src.disentanglers.TextActionDecOnlyBetaTCVAE import TextActionBetaTCVAE
from src.disentanglers.ActionCNNVQVAE import ActionVQVAE  
from src.projectors.ProbabilisticActionProjector import ProbabilisticActionProjector
from src.projectors import ProbabilisticActionProjector, ProbVLMProjector, MLPActionProjector
from utils.data import FastActionRLDSDataset, identity_transform

from transformers import AutoModelForVision2Seq, AutoProcessor, CLIPTokenizer, CLIPTextModel
# from src.disentanglers.TextActionBetaTCVAE import TextActionBetaTCVAE

# Append current directory for OpenVLA utils
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)

# --- CONFIGURATION ---
@dataclass
class GenerateConfig:
    # --- 1. Base Model Paths ---
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b" 
    
    # --- 2. Dynamic Checkpoint Paths (Fed by the Bash Loop) ---
    projector_weights: str = ""  
    tcvae_weights: str = ""        
    
    # --- 3. Architecture & Experiment Types ---
    loss_type: str = "nll"                 # ["mse", "nll", "w2"]
    z_dim: int = 64                        # Latent dimension
    chunk_size: int = 8                   # Number of steps in each action chunk
    projector_type: str = "mlp"         # ["normal", "generalized_gaussian"]
    
    # FIX: Removed the duplicate vae_type!
    vae_type: str = "text_cond_beta_tcvae" # ["beta_tcvae", "vqvae", "text_cond_beta_tcvae"]
    
    # --- 4. Eval Settings ---
    task_suite_name: str = "libero_spatial" 
    num_steps_wait: int = 10 
    num_trials_per_task: int = 50
    center_crop: bool = True 
    
    # --- 5. RHC Settings ---
    execution_horizon: int = 8      
    
    # --- 6. Logging ---
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = True
    wandb_project: str = "DisentangledVLA_Eval"
    wandb_entity: str = "cjimenezf17-iri"
    seed: int = 7


class DisentangledPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.loss_type == "mse":
            cfg.projector_type = "mlp"
        else:
            cfg.projector_type = "normal"
        print(cfg.projector_type, cfg.loss_type, cfg.tcvae_weights, cfg.projector_weights)
        
        print("LOADING MODELS...")
        
        # 1. Load VAE (ROBUST SELECTION)x
        print(f"Initializing VAE Type: {cfg.vae_type}")
        if cfg.vae_type == "text_cond_beta_tcvae":
            self.vae = TextActionBetaTCVAE(action_dim=7, chunk_size=cfg.chunk_size, latent_dim=cfg.z_dim, text_emb_dim=512).to(self.device).eval()
            print("Loading Frozen CLIP for VAE Decoder...")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        elif cfg.vae_type == "beta_tcvae":
            self.vae = ActionBetaTCVAE(action_dim=7, chunk_size=cfg.chunk_size, latent_dim=cfg.z_dim).to(self.device).eval()
        elif cfg.vae_type == "vqvae":
            # Note: Adjust hidden_dims/num_embeddings if your VQ-VAE init requires them!
            self.vae = ActionVQVAE(action_dim=7, chunk_size=cfg.chunk_size, latent_dim=cfg.z_dim).to(self.device).eval()
        else:
            raise ValueError(f"Unknown vae_type: {cfg.vae_type}")
                
        self.vae.load_state_dict(torch.load(cfg.tcvae_weights, map_location=self.device))
        
        # 2. Load Projector (ROBUST SELECTION)
        print(f"Initializing Projector Type: {cfg.projector_type}")
        if cfg.projector_type == "normal":
            self.proj = ProbabilisticActionProjector(input_dim=4096, latent_dim=cfg.z_dim).to(self.device).eval()
        elif cfg.projector_type == "generalized_gaussian":
            self.proj = ProbVLMProjector(input_dim=4096, latent_dim=cfg.z_dim).to(self.device).eval()
        elif cfg.projector_type == "mlp":
            self.proj = MLPActionProjector(input_dim=4096, latent_dim=cfg.z_dim).to(self.device).eval()
        else:
            raise ValueError(f"Unknown projector_type: {cfg.projector_type}")
            
        self.proj.load_state_dict(torch.load(cfg.projector_weights, map_location=self.device))
        
        # Load OpenVLA (Backbone)
        self.processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to(self.device).eval()

        # Buffer for RHC
        self.action_buffer = []
        self.steps_since_plan = 0

    def reset(self):
        """Called at the start of every episode"""
        self.action_buffer = []
        self.steps_since_plan = 0

    def unnormalize_action(self, action):
        """Restores action from [-1, 1] to Real Robot Units"""
        def to_np(x):
            return x.numpy() if hasattr(x, 'numpy') else np.array(x)
            
        mask = to_np(self.stats[self.full_task_suite_name]['action']['mask'])
        min_val = to_np(self.stats[self.full_task_suite_name]['action']['min'])
        max_val = to_np(self.stats[self.full_task_suite_name]['action']['max'])
        
        return (action + 1) / 2 * (max_val - min_val) + min_val
    
    def setup_task_suite_ds(self, task_suite_name):
         # 4. Load Statistics for Un-normalization
        print("LOADING DATASET STATS...")
        
        if "no_noops" not in task_suite_name:
            self.full_task_suite_name = f"{task_suite_name}_no_noops"
        else:
            self.full_task_suite_name = task_suite_name
            
        print(f"   > Benchmark Name: {task_suite_name}")
        print(f"   > Dataset Name:   {self.full_task_suite_name}")

        ds = FastActionRLDSDataset(
            data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/", 
            data_mix=[self.full_task_suite_name], 
            batch_transform=identity_transform,
            resize_resolution=(224, 224),
            train=True,
        )
        self.stats = ds.dataset_statistics

        print(f"   > Available Stat Keys: {list(self.stats[self.full_task_suite_name].keys())}")

    @torch.no_grad()
    def step(self, image_numpy, instruction):
        """
        Returns a SINGLE action step.
        Handles chunking and replanning internally.
        """
        if len(self.action_buffer) == 0 or self.steps_since_plan >= self.cfg.execution_horizon:
            
            # --- PREDICTION LOGIC ---
            img_pil = Image.fromarray(image_numpy)
            prompt = f"In: {instruction}\nOut: "
            
            inputs = self.processor(text=[prompt], images=[img_pil], return_tensors="pt").to(self.device)
            if hasattr(inputs, "pixel_values"):
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            # 1. VLA Embedding
            outputs = self.vla(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]
            idx = inputs.attention_mask.sum(dim=1) - 1
            # emb = last_hidden[0, idx].float() 
            emb = last_hidden[torch.arange(1), idx].float()

            # 2. Projector (ROBUST SELECTION)
            if self.cfg.projector_type == "normal":
                _, pred_mu, _ = self.proj(emb)
            elif self.cfg.projector_type == "mlp":
                pred_mu = self.proj(emb)
            elif self.cfg.projector_type == "generalized_gaussian":
                pred_mu, _, _ = self.proj(emb)

            # 3. VAE Decode (ROBUST SELECTION)
            # Depending on your VQ-VAE, you might need to map pred_mu through a quantizer first.
            # E.g., z_q, _, _ = self.vae.quantize(pred_mu) and then decode(z_q)
            # Standard decoding flow assuming pred_mu acts as the continuous code:
            if self.cfg.vae_type == "text_cond_beta_tcvae":
                # Get the 512-D CLIP embedding for the current instruction
                text_inputs = self.clip_tokenizer([instruction], padding=True, truncation=True, return_tensors="pt").to(self.device)
                text_emb = self.clip_encoder(**text_inputs).pooler_output
                text_emb = text_emb.to(pred_mu.dtype)
                # Decode using BOTH the projected mu and the text condition
                raw_chunk = self.vae.decode(pred_mu, text_emb)[0].cpu().numpy()
            else:
                # Standard unsupervised decode
                raw_chunk = self.vae.decode(pred_mu)[0].cpu().numpy()

            # 4. Unnormalize Entire Chunk
            # real_chunk = self.unnormalize_action(raw_chunk)
            real_chunk = raw_chunk 
            # 5. Refill Buffer
            self.action_buffer = list(real_chunk)
            self.steps_since_plan = 0

        action = self.action_buffer.pop(0)
        self.steps_since_plan += 1
        return action

# --- MAIN EVALUATION LOOP ---
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)
    policy = DisentangledPolicy(cfg)

    run_id = f"libero_spatial/proj-{cfg.projector_type}-loss-{cfg.loss_type}-trials-{cfg.num_trials_per_task}"
    if cfg.run_id_note: run_id += f"--{cfg.run_id_note}"

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    total_episodes, total_successes = 0, 0
    suite_results = dict()
    for task_suite_name in ["libero_spatial",]:
        cfg.task_suite_name = task_suite_name

        policy.setup_task_suite_ds(cfg.task_suite_name)

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        resize_size = get_image_resize_size(cfg) 

        suite_episodes, suite_successes = 0, 0
        
        # 10 loops over tasks
        for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

            task_episodes, task_successes = 0, 0
            
            # 50 loops per task (episodes)
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                env.reset()
                policy.reset() 
                
                obs = env.set_init_state(initial_states[episode_idx])
                
                if cfg.task_suite_name == "libero_spatial": max_steps = 250 
                elif cfg.task_suite_name == "libero_object": max_steps = 280 
                elif cfg.task_suite_name == "libero_goal": max_steps = 300 
                elif cfg.task_suite_name == "libero_10": max_steps = 520 
                elif cfg.task_suite_name == "libero_90": max_steps = 400
                else: max_steps = 600

                t = 0
                replay_images = []
                done = False  

                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        img = get_libero_image(obs) 
                        replay_images.append(img)
                        
                        action = policy.step(img, task_description)

                        gripper_val = 1.0 if action[-1] > 0.5 else 0.0
                        action[-1] = 1.0 - (2.0 * gripper_val)

                        obs, reward, done, info = env.step(action.tolist())
                        
                        if done:
                            task_successes += 1
                            suite_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"Caught exception: {e}")
                        traceback.print_exc() # <--- REVEALS THE EXACT LINE OF THE CRASH
                        log_file.write(f"Caught exception: {e}\n")
                        log_file.write(traceback.format_exc() + "\n")
                        break

                task_episodes += 1
                suite_episodes += 1
                total_episodes += 1
                
                save_rollout_video(
                    replay_images, total_episodes, success=done, 
                    task_description=task_description, log_file=log_file
                )
            
            task_rate = task_successes / task_episodes
            if cfg.use_wandb:
                wandb.log({f"Task_Success/{cfg.task_suite_name}/{task_description}": task_rate})
            
            print(f"\n✅ COMPLETED TASK: {task_description}")
            print(f"✅ TASK SUCCESS RATE: {task_rate:.1%}")
            log_file.write(f"--- COMPLETED TASK: {task_description} | Success Rate: {task_rate:.1%} ---\n")
            log_file.flush()

            env.close()
            
        suite_rate = suite_successes / suite_episodes
        suite_results[cfg.task_suite_name] = suite_rate
        print(f"\n🏆 COMPLETED SUITE: {cfg.task_suite_name}")
        print(f"🏆 SUITE SUCCESS RATE: {suite_rate:.1%}")

        # log_file.close()
    
    print(f"\n🏆 EVALUATION COMPLETED!")
    total_success_rate = total_successes / total_episodes
    print(f"🏆 OVERALL SUCCESS RATE: {total_success_rate:.1%}")
    
    if cfg.use_wandb:
        print(suite_results)
        
        # Format the data for the W&B Table: [[Suite Name, Score], ...]
        bar_data = [[suite, score * 100] for suite, score in suite_results.items()]
        table = wandb.Table(data=bar_data, columns=["Task Suite", "Success Rate (%)"])
        
        wandb.log({
            "Overall/Total_Success_Rate": total_success_rate,
            "Overall/Suite_Comparison": wandb.plot.bar(
                table, 
                "Task Suite", 
                "Success Rate (%)", 
                title="Success Rate per Suite"
            )
        })
        wandb.finish() 
    

if __name__ == "__main__":
    eval_libero()


