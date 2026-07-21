import os
import sys
import argparse
import traceback
import torch
import numpy as np
import tqdm
import wandb
import json
from PIL import Image
from typing import Optional
from libero.libero import benchmark
from transformers import AutoModelForVision2Seq, AutoProcessor

from huggingface_hub import HfApi, hf_hub_download

# Import the original OpenVLA utilities so we don't have to rewrite their gripper logic
sys.path.append("../..")
from experiments.robot.robot_utils import (
    get_action, invert_gripper_action, 
    normalize_gripper_action, set_seed_everywhere
)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)

def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """

    dataset_statistics_path = "/workspace/DisentangledFlow/vlas/openvla_oft/experiments/robot/libero/statistics/all.json"
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay

def evaluate_simplified_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, default="openvla/openvla-7b")
    parser.add_argument("--task_suite_name", type=str, required=True, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    # parser.add_argument("--num_trials_per_task", type=int, default=20)
    args = parser.parse_args()

    set_seed_everywhere(42)
    wandb.init(project="DisentangledVLA_Eval", name=f"Baseline_OpenVLA_{args.task_suite_name}")

    # We mock the config object just enough to satisfy the get_action() statistics logic
    class MockConfig:
        model_family = "openvla"
        pretrained_checkpoint = args.pretrained_checkpoint
        load_in_8bit = False
        load_in_4bit = False
        center_crop = True
        unnorm_key = args.task_suite_name
        use_film = False
        num_images_in_input = 2
        use_proprio = True

        checkpoint_step: Optional[int] = None         # Checkpoint step to load (if None, loads latest)
        use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
        use_diffusion: bool = True                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
        num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
        num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference

        num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

        #################################################################################################################
        # LIBERO environment-specific parameters
        #################################################################################################################
        task_suite_name: str = args.task_suite_name  # Task suite
        num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
        num_trials_per_task: int = 25                    # Number of rollouts per task
        initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
        env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

        seed: int = 42                                 # Random Seed (for reproducibility)


    cfg = MockConfig()

    device = "cuda"

    # ---> THE NATIVE HUGGING FACE LOADER (Bypasses all OpenVLA Utils) <---
    print(f"🚀 Loading Processor natively from {args.pretrained_checkpoint}...")
    processor = AutoProcessor.from_pretrained(args.pretrained_checkpoint, trust_remote_code=True)
    
    print(f"🚀 Loading Model natively from {args.pretrained_checkpoint}...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_checkpoint, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device).eval()

    
    print(model.norm_stats)
    load_dataset_stats(model, args.pretrained_checkpoint)

    # Handle their specific un-norm key hack
    print(model.norm_stats)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    print(f"🌍 Initializing Task Suite: {args.task_suite_name}")
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    
    total_episodes, total_successes = 0, 0

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Set specific step limits based on the suite
        max_steps_dict = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
        max_steps = max_steps_dict.get(args.task_suite_name, 600)

        task_successes = 0
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Eval: {task_description}"):
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

                    observation, img = prepare_observation(obs, 224)

                    if "wrist_image" in observation:
                        observation.pop("wrist_image") # OpenVLA baseline doesn't use wrist image, so we remove it to avoid confusion

                    # Get Action using their script, but our native model!
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
                    traceback.print_exc() 
                    break
            
            total_episodes += 1

        task_rate = task_successes / args.num_trials_per_task
        print(f"✅ {task_description} Success: {task_rate*100}%")
        wandb.log({f"success_rate/{task_description}": task_rate})

    final_rate = total_successes / total_episodes
    print(f"\n🏆 FINAL SUCCESS RATE ({args.task_suite_name}): {final_rate*100}%")
    wandb.log({"success_rate/total": final_rate})
    wandb.finish()

if __name__ == "__main__":
    evaluate_simplified_baseline()