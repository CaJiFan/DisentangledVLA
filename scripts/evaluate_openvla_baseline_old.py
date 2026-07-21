import os
import sys
from typing import Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #

import argparse
import torch
import numpy as np
import tqdm
import wandb
from libero.libero import benchmark

from vlas.openvla_oft.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from vlas.openvla_oft.experiments.robot.robot_utils import (
    get_action, get_model, invert_gripper_action, 
    normalize_gripper_action, set_seed_everywhere
)
from vlas.openvla_oft.experiments.robot.openvla_utils import get_processor

def evaluate_simplified_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, default="openvla/openvla-7b")
    parser.add_argument("--task_suite", type=str, required=True, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num_trials_per_task", type=int, default=20)
    args = parser.parse_args()

    set_seed_everywhere(7)
    wandb.init(project="DisentangledVLA_Eval", name=f"Baseline_OpenVLA_{args.task_suite}")

    print(f"🚀 Loading OpenVLA from: {args.pretrained_checkpoint}")
    
    # We mock the config object just enough to satisfy their internal get_model() function
    class MockConfig:
        model_family = "openvla"
        pretrained_checkpoint = args.pretrained_checkpoint
        load_in_8bit = False
        load_in_4bit = False
        center_crop = True
        unnorm_key = args.task_suite
        use_film = False
        num_images_in_input = 2
        use_proprio = True

        checkpoint_step: Optional[int] = None         # Checkpoint step to load (if None, loads latest)
        use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
        use_diffusion: bool = True                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
        num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
        num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference

        num_open_loop_steps: int = 4                     # Number of actions to execute open-loop before requerying policy

        #################################################################################################################
        # LIBERO environment-specific parameters
        #################################################################################################################
        task_suite_name: str = args.task_suite  # Task suite
        num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
        num_trials_per_task: int = 50                    # Number of rollouts per task
        initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
        env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

        seed: int = 42                                 # Random Seed (for reproducibility)

        
    cfg = MockConfig()
    model = get_model(cfg)
    processor = get_processor(cfg)

    # Handle their specific un-norm key hack
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    print(f"🌍 Initializing Task Suite: {args.task_suite}")
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    
    total_episodes, total_successes = 0, 0

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Set specific step limits based on the suite
        max_steps_dict = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
        max_steps = max_steps_dict.get(args.task_suite, 600)

        task_successes = 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Eval: {task_description}"):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            steps = 0
            done = False
            
            while steps < max_steps + 10: # +10 for the wait steps
                try:
                    # HACK 1: Wait for physics to settle
                    if steps < 10:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        steps += 1
                        continue

                    # Extract just the image for OpenVLA
                    img = obs["agentview_image"]
                    if img.shape[0] == 3: # Flip channels if needed
                        img = np.transpose(img, (1, 2, 0))
                    img = img[::-1, :, :] # Flip upside down (OpenGL fix)
                    
                    observation = {"full_image": img, "state": np.zeros(14)} # Dummy state, OpenVLA ignores it

                    # Get Action
                    action = get_action(cfg, model, observation, task_description, processor=processor)

                    # HACK 2: Fix the Gripper logic
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                    # Step Env
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                        
                    steps += 1

                except Exception as e:
                    print(f"Simulation crashed: {e}")
                    break
            
            total_episodes += 1

        task_rate = task_successes / args.num_trials_per_task
        print(f"✅ {task_description} Success: {task_rate*100}%")
        wandb.log({f"success_rate/{task_description}": task_rate})

    final_rate = total_successes / total_episodes
    print(f"\n🏆 FINAL SUCCESS RATE ({args.task_suite}): {final_rate*100}%")
    wandb.log({"success_rate/total": final_rate})
    wandb.finish()

if __name__ == "__main__":
    evaluate_simplified_baseline()