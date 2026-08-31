import os
from dataclasses import dataclass
from typing import Any, Dict
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel

import h5py
from PIL import Image
from torch.utils.data import Dataset

import torch.nn.functional as F

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

# ---------------------------------------------------------------------------
# Optional heavy dependencies — only required for video probe functions and
# non-cached dataloaders.  Training on a frozen cache needs none of these.
# Wrapped in try/except so smolvla_worker (no LIBERO/prismatic) can import
# this module and use get_vla_projector_dataloader_cached without issues.
# ---------------------------------------------------------------------------
try:
    import imageio
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# PyOpenGL defaults to GLX (requires X11 display) on Linux, leaving
# PLATFORM.EGL=None and crashing robosuite's egl_context.py at import time
# even when libegl1 is installed.  Force the EGL platform so libEGL loads.
# os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Args:
        quat (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den

try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    _HAS_LIBERO = True
except Exception as e:
    print(e)
    benchmark = None
    OffScreenRenderEnv = None
    _HAS_LIBERO = False

try:
    from vlas.openvla_oft.prismatic.vla.datasets import FastActionRLDSDataset, FastActionRLDSDataset2
    _HAS_PRISMATIC = True
except ImportError:
    FastActionRLDSDataset = None
    FastActionRLDSDataset2 = None
    _HAS_PRISMATIC = False

import wandb

# The RLDS Data Transform
@dataclass
class ActionOnlyTransform:
    """
    Extracts only the action chunks from the RLDS batch.
    Ignores images and text.
    """
    action_horizon: int = 32
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # rlds_batch["action"] comes from the dataset with `future_action_window_size`
        # print('og keys', rlds_batch.keys())
        actions_np = rlds_batch["action"] 
        
        # Convert to Tensor
        # actions = torch.from_numpy(actions_np).float()
        actions = torch.tensor(actions_np).float()
        
        # Ensure exact horizon length
        if actions.shape[0] >= self.action_horizon:
            actions = actions[:self.action_horizon]
        else:
            # Pad with zeros if shorter (rare in standard RLDS)
            pad_len = self.action_horizon - actions.shape[0]
            pad = torch.zeros(pad_len, actions.shape[1])
            actions = torch.cat([actions, pad], dim=0)
            
        return {"actions": actions}


SUITES = ["libero_goal", "libero_spatial"]

def identity_transform(x):
    return x

def get_ram_cached_dataloader(data_root_dir, SUITES=SUITES, batch_size=256, action_horizon=16):
    all_actions = []
    print("\n⏳ Caching Actions from TFRecords into RAM...")
    
    for suite in SUITES:
        print(f"   - Streaming {suite}...")
        
        # Initialize our Fast Loader
        ds = FastActionRLDSDataset(
            data_root_dir=data_root_dir,
            data_mix=f'{suite}_no_noops', # Ensure this matches your folder name
            batch_transform=ActionOnlyTransform(action_horizon=action_horizon),
            resize_resolution=(224, 224), # Ignored but required by init
            train=True
        )
        
        # Manually iterate to avoid DataLoader worker overhead
        # The underlying TF dataset is already optimized
        num_samples = len(ds)
        iterator = iter(ds)
        
        count = 0
        for _ in tqdm.tqdm(range(num_samples), desc=f"Extracting {suite}"):
            # Grab next item from the stream
            rlds_batch = next(iterator)
            processed = rlds_batch # Already transformed by __iter__ in RLDSDataset?
            
            # Convert to Torch and store
            # processed['actions'] might be numpy, ensure torch conversion later
            # action_chunk = torch.tensor(processed['actions']).float()
            action_chunk = processed['actions'].clone().detach().float()
            all_actions.append(action_chunk)
            count += 1
            
    print(f"📦 Stacking {len(all_actions)} chunks...")
    full_action_tensor = torch.stack(all_actions, dim=0)
    
    print(f"✅ Cached in RAM. Shape: {full_action_tensor.shape}")
    
    # Create standard Torch Loader
    dataset = TensorDataset(full_action_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def get_text_action_ram_cached_dataloader(
    data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/", 
    suite="libero_spatial",
    batch_size=128,
    train_split_ratio=None,  # None = Protocol A (all tasks, no held-out split)
                              # int  = Protocol B (task-level split, e.g. 7 of 10)
    text_backbone="clip",
    return_states=False
):
    """
    Protocol A (train_split_ratio=None, industry standard for LIBERO):
      - All tasks go into training.
      - Returns (train_dataloader, None, action_stats).
      - Evaluation is done exclusively via the LIBERO simulator.

    Protocol B (train_split_ratio=7):
      - 7 tasks train, 3 tasks held-out test.
      - Returns (train_dataloader, test_dataloader, action_stats).
      - Used for OOD generalisation experiments (e.g. LIBERO-Object).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⏳ Loading Text Backbone for '{text_backbone}'...")

    # 1. Dynamically Load the Tokenizer and Model
    if text_backbone == "smollm":
        model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
    elif text_backbone == "openvla_llama":
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
    else: # Fallback to CLIP
        model_id = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()

    suite_name = f'{suite}_no_noops'
    print(f"\n⏳ Caching {suite_name} into RAM and grouping by task...")
    
    ds = FastActionRLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=[suite_name], 
        batch_transform=identity_transform, 
        train=True,
        resize_resolution=(224, 224), 
        return_visuals=False 
    )
    
    task_dict = {}
    text_cache = {}  # Embed each unique instruction only once

    iterator = iter(ds)
    for _ in tqdm.tqdm(range(len(ds)), desc=f"Extracting {suite}"):
        item = next(iterator)
        
        # Extract Action & Instruction
        if isinstance(item, (tuple, list)): _, instr, act = item 
        else:
            instr = item['task']['language_instruction']
            act = item['action']

        if isinstance(act, np.ndarray): act = torch.from_numpy(np.copy(act)).float()
        elif isinstance(act, torch.Tensor): act = act.float()
            
        if isinstance(instr, bytes): instr = instr.decode("utf-8")
        elif isinstance(instr, np.ndarray): instr = str(instr.item()) if instr.ndim == 0 else instr[0].decode("utf-8")
        
        # 2. Dynamic Embedding Logic (CLIP vs LLMs)
        if instr not in text_cache:
            with torch.no_grad():
                text_inputs = tokenizer([instr], padding=True, truncation=True, return_tensors="pt").to(device)
                
                if text_backbone == "clip":
                    # CLIP uses the dedicated pooler output
                    emb = text_encoder(**text_inputs).pooler_output[0]
                else:
                    # LLMs require mean-pooling the hidden states over the sequence
                    outputs = text_encoder(**text_inputs)
                    hidden_states = outputs.last_hidden_state
                    mask = text_inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    emb = (sum_embeddings / sum_mask)[0]
                    
                text_cache[instr] = emb.cpu().float()

        if instr not in task_dict:
            task_dict[instr] = {'actions': [], 'texts': [], 'states': []}

        task_dict[instr]['actions'].append(act)
        task_dict[instr]['texts'].append(text_cache[instr])
        if return_states:
            state = item['observation']['proprio'][0]
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(np.copy(state)).float()
            elif isinstance(state, torch.Tensor):
                state = state.float()
            task_dict[instr]['states'].append(state)
            
    # 3. Aggressive VRAM Cleanup (Crucial before VAE training)
    del text_encoder, tokenizer
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print("🧹 Cleared Text Backbone from VRAM to make room for VAE.")

    # ---- Task split --------------------------------------------------------
    unique_tasks = sorted(list(task_dict.keys()))
    if len(unique_tasks) != 10:
        print(f"⚠️ Warning: Found {len(unique_tasks)} tasks instead of 10.")

    if train_split_ratio is None:
        # Protocol A: all tasks → training, no held-out test split.
        # Evaluation is done exclusively via the LIBERO simulator.
        train_task_names    = unique_tasks
        held_out_task_names = []
        print("\n" + "="*50)
        print(f"🚀 PROTOCOL A (full suite): {suite.upper()}")
        print(f"✅ TRAINING ON ALL {len(train_task_names)} TASKS — no held-out test split.")
        print("   Evaluation via LIBERO simulator only.")
        print("="*50 + "\n")
    else:
        # Protocol B: task-level split (OOD generalisation experiments).
        train_task_names    = unique_tasks[:train_split_ratio]
        held_out_task_names = unique_tasks[train_split_ratio:]
        print("\n" + "="*50)
        print(f"🔬 PROTOCOL B (task split): {suite.upper()}")
        print(f"✅ TRAINING ON ({len(train_task_names)} Tasks):")
        for t in train_task_names: print(f"  - {t}")
        print(f"🛑 HELD-OUT FOR ZERO-SHOT EVAL ({len(held_out_task_names)} Tasks):")
        for t in held_out_task_names: print(f"  - {t}")
        print("="*50 + "\n")

    train_actions, train_texts = [], []
    train_states = []
    for task in train_task_names:
        train_actions.extend(task_dict[task]['actions'])
        train_texts.extend(task_dict[task]['texts'])
        if return_states:
            train_states.extend(task_dict[task]['states'])

    train_action_tensor = torch.stack(train_actions, dim=0)
    train_text_tensor   = torch.stack(train_texts,   dim=0)
    
    if return_states:
        train_state_tensor = torch.stack(train_states, dim=0)
        print(f"Train Actions: {train_action_tensor.shape}, Texts: {train_text_tensor.shape}, States: {train_state_tensor.shape}")
        train_dataset    = TensorDataset(train_action_tensor, train_text_tensor, train_state_tensor)
    else:
        print(f"Train Actions: {train_action_tensor.shape}, Texts: {train_text_tensor.shape}")
        train_dataset    = TensorDataset(train_action_tensor, train_text_tensor)

    action_stats = ds.dataset_statistics
    print(f"✅ Extracted Official Dataset Statistics for Un-normalization.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)

    if held_out_task_names:
        test_actions, test_texts = [], []
        test_states = []
        for task in held_out_task_names:
            test_actions.extend(task_dict[task]['actions'])
            test_texts.extend(task_dict[task]['texts'])
            if return_states:
                test_states.extend(task_dict[task]['states'])
        test_action_tensor = torch.stack(test_actions, dim=0)
        test_text_tensor   = torch.stack(test_texts,   dim=0)
        
        if return_states:
            test_state_tensor = torch.stack(test_states, dim=0)
            print(f"Test  Actions: {test_action_tensor.shape}, Texts: {test_text_tensor.shape}, States: {test_state_tensor.shape}")
            test_dataset    = TensorDataset(test_action_tensor, test_text_tensor, test_state_tensor)
        else:
            print(f"Test  Actions: {test_action_tensor.shape}, Texts: {test_text_tensor.shape}")
            test_dataset    = TensorDataset(test_action_tensor, test_text_tensor)
            
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=0, drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, action_stats

def get_text_action_ram_cached_dataloader2(
    data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/", 
    suite="libero_spatial", # Now takes a SINGLE suite
    batch_size=128,
    split="train",
    train_split_ratio=7  # Out of 10 tasks
):
    print(f"\n⏳ Loading Frozen CLIP for Text Caching...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    
    suite_name = f'{suite}_no_noops'
    print(f"\n⏳ Caching {suite_name} into RAM and grouping by task...")
    
    ds = FastActionRLDSDataset2(
        data_root_dir=data_root_dir,
        data_mix=[suite_name], 
        batch_transform=identity_transform, 
        train=True,
        resize_resolution=(224, 224), 
        return_visuals=False 
    )
    
    # Dictionary to group data by task instruction
    task_dict = {}
    
    iterator = iter(ds)
    # print('len ds', len(ds))
    for _ in tqdm.tqdm(range(len(ds)), desc=f"Extracting {suite}"):
        item = next(iterator)
        
        # Extract Action & Instruction
        if isinstance(item, (tuple, list)): _, instr, act = item 
        else:
            instr = item['task']['language_instruction']
            act = item['action']

        if isinstance(act, np.ndarray): act = torch.from_numpy(np.copy(act)).float()
        elif isinstance(act, torch.Tensor): act = act.float()
            
        if isinstance(instr, bytes): instr = instr.decode("utf-8")
        elif isinstance(instr, np.ndarray): instr = str(instr.item()) if instr.ndim == 0 else instr[0].decode("utf-8")
        
        # Group by instruction
        if instr not in task_dict:
            task_dict[instr] = {'actions': [], 'texts': []}
            
        with torch.no_grad():
            text_inputs = tokenizer([instr], padding=True, truncation=True, return_tensors="pt").to(device)
            text_emb = text_encoder(**text_inputs).pooler_output[0].cpu() 
        
        # if 'basket' in instr and 'cream' in instr:
        #     print(instr)
        #     print(act)
        #     break
        # break
        task_dict[instr]['actions'].append(act)
        task_dict[instr]['texts'].append(text_emb)
            
    # --- Execute the 7/3 CoRL Split ---
    unique_tasks = sorted(list(task_dict.keys()))
    if len(unique_tasks) != 10:
        print(f"⚠️ Warning: Found {len(unique_tasks)} tasks instead of 10. Adjusting split logic.")
    
    train_task_names = unique_tasks[:train_split_ratio]
    held_out_task_names = unique_tasks[train_split_ratio:]
    
    print("\n" + "="*50)
    print(f"🏆 CoRL EVALUATION SPLIT FOR: {suite.upper()}")
    print(f"✅ TRAINING ON ({len(train_task_names)} Tasks):")
    for t in train_task_names: print(f"  - {t}")
    print(f"🛑 HELD-OUT FOR ZERO-SHOT EVAL ({len(held_out_task_names)} Tasks):")
    for t in held_out_task_names: print(f"  - {t}")
    print("="*50 + "\n")


    actions, texts = [], []
    if split == "train":
        for task in train_task_names:
            actions.extend(task_dict[task]['actions'])
            texts.extend(task_dict[task]['texts'])
    else: # split = "test"
        for task in held_out_task_names:
            actions.extend(task_dict[task]['actions'])
            texts.extend(task_dict[task]['texts'])

    full_action_tensor = torch.stack(actions, dim=0)
    full_text_tensor = torch.stack(texts, dim=0)
    
    action_stats = ds.dataset_statistics
    
    print(f"✅ Cached {suite} {split} Split. Actions: {full_action_tensor.shape}")
    print(f"✅ Extracted Official Dataset Statistics for Un-normalization.")
    
    dataset = TensorDataset(full_action_tensor, full_text_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # Return both the dataloader AND the official stats
    return dataloader, action_stats

def get_full_trajectory_dataloader(
    data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/", 
    suite="libero_spatial",
    batch_size=128,
    train_split_ratio=7,
    max_seq_len=256,
    stats_path="./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt" # Path to your previously saved stats!
):
    print(f"\n⏳ Loading Frozen CLIP for Text Caching...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    
    suite_dir = os.path.join(data_root_dir, f"{suite}_no_noops_hdf5")
    if not os.path.exists(suite_dir):
        suite_dir = os.path.join(data_root_dir, suite) # Fallback
        
    task_files = sorted([f for f in os.listdir(suite_dir) if f.endswith('.hdf5')])
    
    task_dict = {}
    
    print(f"\n⏳ Extracting FULL Trajectories from {suite} HDF5s...")
    action_stats = torch.load(stats_path, weights_only=False)
    
    # 2. Extract the exact normalization bounds for the specific suite
    suite_name_in_stats = f"{suite}_no_noops"
    stats = action_stats[suite_name_in_stats]['action']
    action_min = torch.tensor(stats['min']).float()
    action_max = torch.tensor(stats['max']).float()
    action_mask = torch.tensor(stats['mask']).float()

    bmark = benchmark.get_benchmark_dict()[suite]()

    for file_name in tqdm.tqdm(task_files):
        task_path = os.path.join(suite_dir, file_name)
        task_name = file_name.replace(".hdf5", "")
        task_id = None
        for i in range(bmark.get_num_tasks()):
            if bmark.get_task(i).name + '_demo' == task_name:
                task_id = i
                break
                
        if task_id is None:
            print(f"⚠️ Warning: Could not find {task_name} in benchmark. Skipping.")
            continue

        instr = bmark.get_task(task_id).language
        
        if instr not in task_dict:
            task_dict[instr] = {'actions': [], 'texts': []}
            
        # Embed text once per task
        with torch.no_grad():
            text_inputs = tokenizer([instr], padding=True, truncation=True, return_tensors="pt").to(device)
            text_emb = text_encoder(**text_inputs).pooler_output[0].cpu()
        
        with h5py.File(task_path, "r") as f:
            # Extract and Pad every demonstration
            for demo_id in f["data"].keys():
                gt_actions = f[f"data/{demo_id}/actions"][:]
                
                # --- PADDING LOGIC ---
                # Pad up to max_seq_len (256) by repeating the final resting state action
                seq_len = len(gt_actions)
                if seq_len < max_seq_len:
                    padding_len = max_seq_len - seq_len
                    padding = np.tile(gt_actions[-1], (padding_len, 1))
                    padded_actions = np.concatenate([gt_actions, padding], axis=0)
                else:
                    padded_actions = gt_actions[:max_seq_len] # Truncate if somehow over 256
                    
                act_tensor = torch.tensor(padded_actions).float()
                norm_act = (act_tensor - action_min) / (action_max - action_min + 1e-5)
                norm_act = norm_act * 2.0 - 1.0
                
                # If there are dimensions we shouldn't normalize (mask == 0), keep original
                norm_act = norm_act * action_mask + act_tensor * (1.0 - action_mask)
                
                # Now append the perfectly normalized trajectory
                task_dict[instr]['actions'].append(norm_act)
                task_dict[instr]['texts'].append(text_emb)

    # --- Execute the 7/3 CoRL Split ---
    unique_tasks = sorted(list(task_dict.keys()))
    train_task_names = unique_tasks[:train_split_ratio]
    held_out_task_names = unique_tasks[train_split_ratio:]
    
    print(train_task_names)
    print(held_out_task_names) 

    print("\n" + "="*50)
    print(f"🏆 CoRL EVALUATION SPLIT FOR: {suite.upper()}")
    print(f"✅ TRAINING ON ({len(train_task_names)} Tasks)")
    for t in train_task_names: print(f"  - {t}")
    print(f"🛑 HELD-OUT FOR ZERO-SHOT EVAL ({len(held_out_task_names)} Tasks)")
    for t in held_out_task_names: print(f"  - {t}")
    print("="*50 + "\n")

    train_actions, train_texts = [], []
    test_actions, test_texts = [], []

    for task in train_task_names:
        train_actions.extend(task_dict[task]['actions'])
        train_texts.extend(task_dict[task]['texts'])

    for test_task in held_out_task_names:
        test_actions.extend(task_dict[test_task]['actions'])
        test_texts.extend(task_dict[test_task]['texts'])

    train_action_tensor = torch.stack(train_actions, dim=0) # Shape: (N, 256, 7)
    train_text_tensor = torch.stack(train_texts, dim=0)
    test_action_tensor = torch.stack(test_actions, dim=0)
    test_text_tensor = torch.stack(test_texts, dim=0)

    print(train_action_tensor.shape, train_text_tensor.shape)
    print(test_action_tensor.shape, test_text_tensor.shape)
    
    # Load previously saved stats to keep math identical to OpenVLA
    action_stats = torch.load(stats_path, weights_only=False)
    
    train_dataset = TensorDataset(train_action_tensor, train_text_tensor)
    test_dataset = TensorDataset(test_action_tensor, test_text_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    return train_dataloader, test_dataloader, action_stats

def log_video_probe(vae, step, suite_name, stats_path, device, probe_task_name, split_name, chunk_size, text_backbone, exec_steps=1, temporal_ensemble=True, ensemble_k=0.01):
    """Runs a video probe for chunk-based VAEs (with Temporal Ensembling by default) and logs it to WandB."""
    if temporal_ensemble:
        print(f"\n🎥 Generating {split_name.upper()} Video Probe (Temporal Ensembling k={ensemble_k}, Chunk Size {chunk_size}) for Step {step}...")
    else:
        print(f"\n🎥 Generating {split_name.upper()} Video Probe (Chunk Size {chunk_size}, exec_steps={exec_steps}) for Step {step}...")
    
    probe_demo_id = "demo_0"
    
    # 1. Match task to benchmark
    bmark = benchmark.get_benchmark_dict()[suite_name]()
    task_id = None
    for i in range(bmark.get_num_tasks()):
        if bmark.get_task(i).name + '_demo' == probe_task_name:
            task_id = i
            break
            
    if task_id is None:
        print(f"⚠️ Probe task '{probe_task_name}' not found. Skipping video.")
        return

    instruction = bmark.get_task(task_id).language

    # 2. Extract GT Data and Init State from HDF5
    raw_suite_name = suite_name.replace("_no_noops", "")
    data_dir = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{raw_suite_name}_no_noops_hdf5"
    hdf5_path = os.path.join(data_dir, f"{probe_task_name}.hdf5")
    
    if not os.path.exists(hdf5_path):
        print(f"🧨 Video Probe Error: Could not find HDF5 at {hdf5_path}")
        return
        
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
        probe_demo_id = demo_keys[0] if len(demo_keys) > 0 else "demo_0"
        gt_actions = f[f"data/{probe_demo_id}/actions"][:]
        init_state = f[f"data/{probe_demo_id}/states"][0]
    
    # 3. Embed Instruction Dynamically
    if text_backbone == "smollm":
        model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
    elif text_backbone == "openvla_llama":
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
    else: # Fallback to CLIP
        model_id = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()

    text_inputs = tokenizer([instruction], padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if text_backbone == "clip":
            text_emb = text_encoder(**text_inputs).pooler_output
        else:
            outputs = text_encoder(**text_inputs)
            hidden_states = outputs.last_hidden_state
            mask = text_inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            text_emb = (sum_embeddings / sum_mask) # Shape: (1, dim)
    
    # 4. Pre-Normalize the ENTIRE Ground Truth Sequence
    action_stats = torch.load(stats_path, weights_only=False)
    stats = action_stats[f"{raw_suite_name}_no_noops"]['action']
    action_min = torch.tensor(stats['min']).float().to(device)
    action_max = torch.tensor(stats['max']).float().to(device)
    action_mask = torch.tensor(stats['mask']).float().to(device)

    gt_tensor = torch.tensor(gt_actions).float().to(device)
    norm_gt = (gt_tensor - action_min) / (action_max - action_min + 1e-5)
    norm_gt = norm_gt * 2.0 - 1.0
    norm_gt = norm_gt * action_mask + gt_tensor * (1.0 - action_mask)
    
    # 5. Initialize MuJoCo Environment
    env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    env.set_init_state(init_state) 
    # Execute dummy actions for 15 steps to let physics settle (standard LIBERO protocol)
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(15):
        obs, _, _, _ = env.step(dummy_action)
    
    video_path = f"/tmp/vae_probe_{split_name}_{step}.mp4"
    writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)
    
    seq_len = len(norm_gt)
    task_success = False

    if temporal_ensemble:
        action_dim = len(action_min)
        all_time_actions = np.zeros((seq_len, seq_len + chunk_size, action_dim), dtype=np.float32)
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
                decode_text = torch.zeros_like(text_emb) if getattr(vae, 'no_text_decoder', False) else text_emb
                encode_args = (chunk, text_emb) if vae.encode.__code__.co_argcount > 2 else (chunk,)
                mu, _ = vae.encode(*encode_args)
                if getattr(vae, 'use_state', False):
                    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                    state_tensor = torch.tensor(state).float().to(device).unsqueeze(0)
                    pred_chunk_norm = vae.decode(mu, decode_text, state_tensor)[0]
                else:
                    pred_chunk_norm = vae.decode(mu, decode_text)[0]

            all_time_actions[t, t : t + chunk_size] = pred_chunk_norm.cpu().numpy()

            start_idx = max(0, t - chunk_size + 1)
            actions_for_curr_step = all_time_actions[start_idx : t + 1, t]

            num_actions = len(actions_for_curr_step)
            ages = np.arange(num_actions)[::-1]
            weights = np.exp(-ensemble_k * ages)
            weights = weights / weights.sum()

            pred_action_norm = (actions_for_curr_step * weights[:, None]).sum(axis=0)
            pred_action_norm = np.clip(pred_action_norm, -1.0, 1.0)

            unnorm_action = (pred_action_norm + 1.0) / 2.0 * (action_max.cpu().numpy() - action_min.cpu().numpy() + 1e-5) + action_min.cpu().numpy()
            pred_action = unnorm_action * action_mask.cpu().numpy() + pred_action_norm * (1.0 - action_mask.cpu().numpy())
            action_np = pred_action.copy()
            action_np[-1] = 1.0 if pred_action_norm[-1] > 0.0 else -1.0

            obs, reward, done, info = env.step(action_np)
            img = np.flipud(obs['agentview_image'])
            writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))

            if done:
                task_success = True
                break
            t += 1
    else:
        # Standard chunk execution loop: re-plan every exec_steps steps
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
                decode_text = torch.zeros_like(text_emb) if getattr(vae, 'no_text_decoder', False) else text_emb
                encode_args = (chunk, text_emb) if vae.encode.__code__.co_argcount > 2 else (chunk,)
                mu, _ = vae.encode(*encode_args)
                if getattr(vae, 'use_state', False):
                    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                    state_tensor = torch.tensor(state).float().to(device).unsqueeze(0)
                    pred_chunk_norm = vae.decode(mu, decode_text, state_tensor)[0]
                else:
                    pred_chunk_norm = vae.decode(mu, decode_text)[0]

            steps_to_execute = chunk_size if t + chunk_size <= seq_len else seq_len - t
            num_steps_this_loop = min(exec_steps, steps_to_execute)
            done = False
            for i in range(num_steps_this_loop):
                pred_action_norm = pred_chunk_norm[i]

                unnorm_action = (pred_action_norm + 1.0) / 2.0 * (action_max - action_min + 1e-5) + action_min
                pred_action = unnorm_action * action_mask + pred_action_norm * (1.0 - action_mask)
                action_np = pred_action.cpu().numpy()
                action_np[-1] = 1.0 if pred_action_norm[-1] > 0.0 else -1.0

                obs, reward, done, info = env.step(action_np)
                img = np.flipud(obs['agentview_image'])
                writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))

                if done:
                    task_success = True
                    break

            t += num_steps_this_loop
            if done:
                break
            
    writer.close()
    env.close()
    del text_encoder, tokenizer
    import gc; gc.collect()
    torch.cuda.empty_cache()
    
    # 7. Log to WandB
    if wandb.run is not None:
        wandb.log({
            f"eval_videos/{split_name}_probe": wandb.Video(video_path, fps=30, format="mp4"),
            f"eval_metrics/probe_{split_name}_success": float(task_success),
            "global_step": step
        }, step=step)
    status = "✅ SUCCESS" if task_success else "❌ FAILED"
    print(f"🎥 {split_name.upper()} Video uploaded to WandB! Outcome: {status}")

def _make_openvla_emb_fn(vla_model, processor, device, use_vision_pool=False, vla_layer_idx=-1, num_fusion_layers=1):
    """
    Build an emb_fn callable from an OpenVLA model + processor.
    emb_fn(image_pil, instruction) → torch.Tensor (1, vla_dim) float32 on `device`
    """
    import torch
    vla_prompt_tmpl = "In: {}\nOut: "
    _img_token_id = getattr(vla_model.config, "image_token_index",
                    getattr(vla_model.config, "image_token_id", None)) if use_vision_pool else None

    if num_fusion_layers > 1:
        total_layers = len(vla_model.language_model.model.layers)
        step = total_layers // (num_fusion_layers + 1)
        target_indices = [step * (i + 1) - 1 for i in range(num_fusion_layers)]
    else:
        target_indices = [vla_layer_idx]

    def emb_fn(image_pil, instruction: str):
        prompt = vla_prompt_tmpl.format(instruction)
        inputs = processor(
            text=[prompt], images=[image_pil],
            padding=True, truncation=True, return_tensors="pt"
        ).to(vla_model.device)
        if hasattr(inputs, "pixel_values"):
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            out = vla_model(**inputs, output_hidden_states=True, return_dict=True)
            
        layer_embs = []
        for idx in target_indices:
            hs = out.hidden_states[idx]  # (1, L, D)
            if use_vision_pool and _img_token_id is not None:
                img_mask = (inputs["input_ids"] == _img_token_id).unsqueeze(-1).float()  # (1, L, 1)
                vision_tokens = hs * img_mask
                n_img = img_mask.sum(dim=1).clamp(min=1)
                emb = (vision_tokens.sum(dim=1) / n_img).float().to(device)  # (1, D)
            else:
                last_tok_idx = inputs.attention_mask.sum(dim=1) - 1
                emb = hs[0, last_tok_idx[0]].unsqueeze(0).float().to(device)  # (1, D)
            layer_embs.append(emb)
            
        if num_fusion_layers > 1:
            return torch.stack(layer_embs, dim=1) # (1, num_fusion_layers, D)
        else:
            return layer_embs[0]

    return emb_fn


def _get_clip_text_emb(instruction: str, device):
    """Load CLIP once per unique (instruction, device) pair and cache the result."""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_inputs = tokenizer([instruction], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_text_encoder(**text_inputs).pooler_output  # [1, 512]
    del clip_text_encoder
    return emb


def _make_smolvla_emb_fn(policy, device, vla_layer_idx=-1, num_fusion_layers=1):
    """
    Build an emb_fn for SmolVLA.
    emb_fn(image_pil, instruction) -> torch.Tensor float32 on `device`
    """
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    flow = policy.model
    vlm  = flow.vlm_with_expert
    vlm_dtype   = next(vlm.parameters()).dtype
    state_dtype = flow.state_proj.weight.dtype

    _resize = getattr(policy.config, "resize_imgs_with_padding", None) or (256, 256)
    _tgt_h, _tgt_w = _resize

    total_layers = 16
    if num_fusion_layers > 1:
        step = total_layers // (num_fusion_layers + 1)
        target_indices = [step * (i + 1) - 1 for i in range(num_fusion_layers)]
    else:
        target_indices = [vla_layer_idx]

    captured = {}
    handles = []
    
    def make_hook_fn(layer_idx_key):
        def hook_fn(module, args):
            hs = args[0]
            captured[layer_idx_key] = hs.float().detach().cpu()
        return hook_fn

    for l_idx in target_indices:
        actual_idx = l_idx if l_idx != -1 else 15
        if hasattr(vlm.vlm.model, "text_model") and hasattr(vlm.vlm.model.text_model, "layers"):
            target_layer = vlm.vlm.model.text_model.layers[actual_idx].input_layernorm
            handle = target_layer.register_forward_pre_hook(make_hook_fn(l_idx))
        else:
            target_layer = vlm.vlm.model.layers[actual_idx]
            def legacy_hook_fn(module, input, output, key=l_idx):
                hs = output[0] if isinstance(output, tuple) else output
                captured[key] = hs.float().detach().cpu()
            handle = target_layer.register_forward_hook(legacy_hook_fn)
        handles.append(handle)

    @torch.no_grad()
    def emb_fn(image_pil, instruction: str):
        # Language
        lang_enc    = vlm.processor.tokenizer(
            [instruction], return_tensors="pt", padding=True,
            truncation=True, max_length=48,
        )
        lang_tokens = lang_enc.input_ids.to(device)
        lang_masks  = lang_enc.attention_mask.to(device).bool()

        # Image: resize to policy's native resolution, normalise to [-1, 1]
        img_t = torch.from_numpy(
            np.array(image_pil.resize((_tgt_w, _tgt_h)))
        ).float() / 255.0
        img_t = img_t.permute(2, 0, 1) * 2.0 - 1.0          # CHW, [-1, 1]
        images_tensor = img_t.unsqueeze(0).to(device, dtype=vlm_dtype)  # (1, 3, H, W)
        img_masks = torch.ones(1, dtype=torch.bool, device=device)

        # Zero state
        state = torch.zeros(1, flow.config.max_state_dim, device=device, dtype=state_dtype)

        # Prefix embeddings
        prefix_embs, prefix_pad, prefix_att = flow.embed_prefix(
            images=[images_tensor],
            img_masks=[img_masks],
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        att_2d  = make_att_2d_masks(prefix_pad, prefix_att)
        pos_ids = torch.cumsum(prefix_pad, dim=1) - 1

        (prefix_out, _), _ = vlm.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        valid = prefix_pad.unsqueeze(-1).float().cpu()          # (1, seq_len, 1)
        if num_fusion_layers > 1:
            batch_layer_embs = []
            for l_idx in target_indices:
                prefix_out_l = captured[l_idx]
                emb_l = (prefix_out_l.float() * valid).sum(1) / valid.sum(1).clamp(min=1)
                batch_layer_embs.append(emb_l)
            emb = torch.stack(batch_layer_embs, dim=1)  # (1, num_fusion_layers, 960)
        else:
            prefix_out_l = captured[target_indices[0]]
            emb = (prefix_out_l.float() * valid).sum(1) / valid.sum(1).clamp(min=1)  # (1, 960)
            
        captured.clear()
        return emb.to(device)

    return emb_fn


def _make_pi0_emb_fn(pi0_policy, pi0_tokenizer, device, num_fusion_layers=3):
    """
    Build an emb_fn for Pi0.
    emb_fn(image_pil, instruction) -> torch.Tensor (1, num_fusion_layers, 2048) float32 on `device`
    """
    import torch.nn.functional as F
    
    trunk = pi0_policy.model.paligemma_with_expert.paligemma.model.language_model.layers
    n_layers = len(trunk)
    step = n_layers // num_fusion_layers
    target_indices = [step * i + (step - 1) for i in range(num_fusion_layers)]
    
    layer_outputs = {}
    def get_pre_hook(name):
        def hook(model, args):
            out = args[0]
            layer_outputs[name] = out
        return hook

    handles = []
    for idx in target_indices:
        target_module = trunk[idx].input_layernorm
        handle = target_module.register_forward_pre_hook(get_pre_hook(f"layer_{idx}"))
        handles.append(handle)

    @torch.no_grad()
    def emb_fn(image_pil, instruction: str):
        img_np = np.array(image_pil)
        img_t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        # Resize to (480, 640) as expected by lerobot/pi0
        img_t = F.interpolate(img_t, size=(480, 640), mode='bilinear', align_corners=False)

        if pi0_tokenizer is not None:
            tokens = pi0_tokenizer([instruction], padding="max_length", max_length=48, truncation=True, return_tensors="pt")
            lang_tokens = tokens["input_ids"].to(device)
            lang_mask = tokens["attention_mask"].bool().to(device)
        else:
            lang_tokens = torch.zeros((1, 48), dtype=torch.long, device=device)
            lang_mask = torch.ones((1, 48), dtype=torch.bool, device=device)

        dummy_inputs = {
            "observation.images.camera0": img_t,
            "observation.images.camera1": torch.zeros_like(img_t),
            "observation.images.camera2": torch.zeros_like(img_t),
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask,
            "observation.state": torch.zeros((1, 14), dtype=torch.float32, device=device),
            "action": torch.zeros((1, 50, 14), dtype=torch.float32, device=device),
            "task": instruction
        }

        layer_outputs.clear()
        _ = pi0_policy(dummy_inputs)

        extracted_layers = []
        for idx in target_indices:
            layer_out = layer_outputs[f"layer_{idx}"]
            causal_token = layer_out[:, -1, :] # [1, 2048]
            extracted_layers.append(causal_token)

        stacked_emb = torch.stack(extracted_layers, dim=1) # [1, num_fusion_layers, 2048]
        return stacked_emb.to(device)
        
    return emb_fn


def log_projector_video_probe(
    vae, projector, vla_model, processor,
    step, suite_name, stats_path, device,
    probe_task_name, split_name, chunk_size,
    emb_fn=None, normalize_emb=False, use_vision_pool=False,
    text_backbone="clip", vla_layer_idx=-1, num_fusion_layers=1, exec_steps=None,
    temporal_ensemble=True, ensemble_k=0.01
):
    if not _HAS_LIBERO:
        print("⚠️  LIBERO not available — skipping projector video probe.")
        return
    # Respect MUJOCO_GL env var (osmesa or egl); default to egl only if unset.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("PYOPENGL_PLATFORM", None)
    """
    Chunk-based video probe for the full projector pipeline (with Temporal Ensembling by default):
      env obs image + instruction → VLA → projector → z_mu → VAE decoder → action chunk

    Args:
        emb_fn: Optional callable (image_pil, instruction) → torch.Tensor (1, D) on `device`.
                If None, falls back to building one from vla_model + processor (OpenVLA path).
                Pass OctoWorker.make_emb_fn() here for Octo-based projectors.
        text_backbone: The text backbone to use for embedding the instruction.
    """
    if emb_fn is None:
        # Backwards-compatible: build from OpenVLA model/processor
        if vla_model is None or processor is None:
            raise ValueError("Must provide either emb_fn OR (vla_model, processor) to log_projector_video_probe")
        emb_fn = _make_openvla_emb_fn(vla_model, processor, device, use_vision_pool=use_vision_pool,
                                      vla_layer_idx=vla_layer_idx, num_fusion_layers=num_fusion_layers)

    # 1. Match task to benchmark
    bmark = benchmark.get_benchmark_dict()[suite_name]()
    task_id = None
    for i in range(bmark.get_num_tasks()):
        if bmark.get_task(i).name + '_demo' == probe_task_name:
            task_id = i
            break

    if task_id is None:
        print(f"⚠️  [probe] Task '{probe_task_name}' not in benchmark {suite_name}. Skipping probe.")
        return

    instruction = bmark.get_task(task_id).language

    # 2. Extract GT init state from HDF5 (same starting position as demonstrations)
    raw_suite_name = suite_name.replace("_no_noops", "")
    data_dir = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{raw_suite_name}_no_noops_hdf5"
    hdf5_path = os.path.join(data_dir, f"{probe_task_name}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"⚠️  [probe] HDF5 file not found: {hdf5_path}. Skipping probe.")
        return

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()),
                           key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
        probe_demo_id = demo_keys[0] if demo_keys else "demo_0"
        init_state = f[f"data/{probe_demo_id}/states"][0]
        gt_actions = f[f"data/{probe_demo_id}/actions"][:]

    # 3. Load stats for action un-normalisation
    action_stats = torch.load(stats_path, weights_only=False)
    stats = action_stats[f"{raw_suite_name}_no_noops"]['action']
    action_min = torch.tensor(stats['min']).float().to(device)
    action_max = torch.tensor(stats['max']).float().to(device)
    action_mask = torch.tensor(stats['mask']).float().to(device)

    # 4. Text embedding for VAE decoder (if text_cond VAE)
    vae_text_emb = None
    if getattr(vae, 'text_emb_dim', None) is not None and getattr(vae, 'text_emb_dim') > 0:
        if text_backbone == "clip":
            tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            t_inp = tok([instruction], padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                vae_text_emb = enc(**t_inp).pooler_output
            del tok, enc
        else:
            tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            enc = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct").to(device).eval()
            t_inp = tok([instruction], padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = enc(**t_inp)
                mask = t_inp.attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                vae_text_emb = (out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            del tok, enc
        torch.cuda.empty_cache()

    # 5. Initialise MuJoCo environment
    env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    env.set_init_state(init_state)
    obs, _, _, _ = env.step(np.zeros(7))

    video_path = f"/tmp/proj_probe_{split_name}_{step}.mp4"
    writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)

    projector.eval()
    seq_len = min(len(gt_actions) * 2, 400)
    done = False
    task_success = False
    first_action_logged = False

    if temporal_ensemble:
        action_dim = len(action_min)
        all_time_actions = np.zeros((seq_len, seq_len + chunk_size, action_dim), dtype=np.float32)
        for t in range(seq_len):
            if done:
                break
            obs_img_pil = Image.fromarray(obs['agentview_image'].astype(np.uint8))
            with torch.no_grad():
                vla_emb = emb_fn(obs_img_pil, instruction).to(device)
                if normalize_emb:
                    vla_emb = F.normalize(vla_emb, dim=-1)

                if "Flow" in projector.__class__.__name__:
                    z_dim = projector.latent_dim
                    B_vla = vla_emb.size(0)
                    z_t = torch.randn(B_vla, z_dim, device=vla_emb.device)
                    dt = 1.0 / 10
                    for i in range(10):
                        t_steps = torch.ones(B_vla, device=vla_emb.device) * (i / 10.0)
                        v = projector(vla_emb, z_t, t_steps)
                        z_t = z_t + v * dt
                    pred_mu = z_t
                else:
                    _, pred_mu, _ = projector(vla_emb)
                    
                if getattr(vae, 'use_state', False):
                    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                    state_tensor = torch.tensor(state).float().to(device).unsqueeze(0)
                    pred_chunk_norm = vae.decode(pred_mu, vae_text_emb, state_tensor)[0]
                else:
                    pred_chunk_norm = vae.decode(pred_mu, vae_text_emb)[0]

            all_time_actions[t, t : t + chunk_size] = pred_chunk_norm.cpu().numpy()

            start_idx = max(0, t - chunk_size + 1)
            actions_for_curr_step = all_time_actions[start_idx : t + 1, t]

            num_actions = len(actions_for_curr_step)
            ages = np.arange(num_actions)[::-1]
            weights = np.exp(-ensemble_k * ages)
            weights = weights / weights.sum()

            pred_action_norm = (actions_for_curr_step * weights[:, None]).sum(axis=0)
            pred_action_norm = np.clip(pred_action_norm, -1.0, 1.0)

            pred_action_unnorm = (pred_action_norm + 1.0) / 2.0 * (action_max.cpu().numpy() - action_min.cpu().numpy()) + action_min.cpu().numpy()
            pred_action = pred_action_unnorm * action_mask.cpu().numpy() + pred_action_norm * (1.0 - action_mask.cpu().numpy())
            action_np = pred_action.copy()
            action_np[-1] = 1.0 if pred_action_norm[-1] > 0.0 else -1.0

            obs, reward, done, info = env.step(action_np)
            if done:
                task_success = True
            img = np.flipud(obs['agentview_image'])
            writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
            if done:
                break
    else:
        if exec_steps is None:
            exec_steps = chunk_size
        for t in range(0, seq_len, exec_steps):
            if done:
                break
            obs_img_pil = Image.fromarray(obs['agentview_image'].astype(np.uint8))
            with torch.no_grad():
                vla_emb = emb_fn(obs_img_pil, instruction).to(device)
                if normalize_emb:
                    vla_emb = F.normalize(vla_emb, dim=-1)

                if "Flow" in projector.__class__.__name__:
                    z_dim = projector.latent_dim
                    B_vla = vla_emb.size(0)
                    z_t = torch.randn(B_vla, z_dim, device=vla_emb.device)
                    dt = 1.0 / 10
                    for i in range(10):
                        t_steps = torch.ones(B_vla, device=vla_emb.device) * (i / 10.0)
                        v = projector(vla_emb, z_t, t_steps)
                        z_t = z_t + v * dt
                    pred_mu = z_t
                else:
                    _, pred_mu, _ = projector(vla_emb)
                pred_chunk_norm = vae.decode(pred_mu, vae_text_emb)

            for k in range(exec_steps):
                if t + k >= seq_len:
                    break
                pred_action_norm = pred_chunk_norm[0, k]
                pred_action_unnorm = (pred_action_norm + 1.0) / 2.0 * (action_max - action_min) + action_min
                pred_action = pred_action_unnorm * action_mask + pred_action_norm * (1.0 - action_mask)
                action_np = pred_action.cpu().numpy()
                action_np[-1] = 1.0 if action_np[-1] > 0.0 else -1.0

                obs, reward, done, info = env.step(action_np)
                if done:
                    task_success = True
                img = np.flipud(obs['agentview_image'])
                writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
                if done:
                    break

    writer.close()
    env.close()
    status = "✅ SUCCESS" if task_success else f"❌ FAILED (timeout at {seq_len} steps)"
    print(f"  [probe] {split_name} probe outcome: {status}")

    if wandb.run is not None:
        exec_tag = f"temporal_k{ensemble_k}" if temporal_ensemble else f"exec{exec_steps}"
        wandb.log({
            f"eval_videos/projector_{split_name}_{exec_tag}_{probe_task_name}": wandb.Video(video_path, fps=30, format="mp4"),
            f"eval_metrics/{split_name}_success_{exec_tag}_{probe_task_name}": float(task_success),
            "global_step": step
        }, step=step)
    print(f"🎥 Projector {split_name.upper()} Video uploaded to WandB!")

def log_gt_video_probe(
    step, suite_name, stats_path, device,
    probe_task_name, split_name,
):
    if not _HAS_LIBERO:
        print("⚠️  LIBERO not available — skipping GT video probe.")
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("PYOPENGL_PLATFORM", None)
    """
    Executes the raw ground-truth actions from the HDF5 demo and logs the
    resulting video to WandB. Use this once as a reference baseline to compare
    projector probe videos against.
    """
    print(f"\n🎥 Generating GT {split_name.upper()} Reference Video for Step {step}...")
    probe_demo_id = "demo_0"

    bmark = benchmark.get_benchmark_dict()[suite_name]()
    task_id = None
    for i in range(bmark.get_num_tasks()):
        if bmark.get_task(i).name + '_demo' == probe_task_name:
            task_id = i
            break
    if task_id is None:
        print(f"⚠️ GT probe task '{probe_task_name}' not found. Skipping.")
        return

    raw_suite_name = suite_name.replace("_no_noops", "")
    data_dir = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{raw_suite_name}_no_noops_hdf5"
    hdf5_path = os.path.join(data_dir, f"{probe_task_name}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"🧨 GT Video Probe Error: Could not find HDF5 at {hdf5_path}")
        return
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
        probe_demo_id = demo_keys[0] if len(demo_keys) > 0 else "demo_0"
        gt_actions = f[f"data/{probe_demo_id}/actions"][:]
        init_state = f[f"data/{probe_demo_id}/states"][0]

    env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    env.set_init_state(init_state)
    obs, _, _, _ = env.step(np.zeros(7))

    video_path = f"/tmp/gt_probe_{split_name}_{step}.mp4"
    writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)

    for action_np in gt_actions:
        obs, reward, done, info = env.step(action_np)
        img = np.flipud(obs['agentview_image'])
        writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
        if done:
            break

    writer.close()
    env.close()

    if wandb.run is not None:
        wandb.log({
            f"gt_videos/{split_name}_reference": wandb.Video(video_path, fps=30, format="mp4"),
            "global_step": step
        }, step=step)
    print(f"🎥 GT {split_name.upper()} reference video uploaded to WandB!")
 


class VLARAMDataset(Dataset):
    def __init__(self, images, instructions, actions):
        self.images = images
        self.instructions = instructions
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # Return PIL image for OpenVLA processor, raw string, and tensor action
        return {
            "image": Image.fromarray(self.images[idx]), 
            "instruction": self.instructions[idx], 
            "actions": self.actions[idx]
        }

def get_vla_projector_dataloader(
    data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/", 
    suite="libero_spatial", 
    batch_size=4,        # ⚠️ MUST BE SMALL FOR OPENVLA! (e.g., 2 or 4)
    train_split_ratio=7  # Out of 10 tasks
):
    suite_name = f'{suite}_no_noops'
    print(f"\n⏳ Caching {suite_name} VISUALS into RAM and grouping by task...")
    
    ds = FastActionRLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=[suite_name], 
        batch_transform=identity_transform, 
        train=True,
        resize_resolution=(224, 224), 
        return_visuals=True # ⚠️ CRITICAL: We need images now!
    )
    
    task_dict = {}
    iterator = iter(ds)
    
    for _ in tqdm.tqdm(range(len(ds)), desc=f"Extracting {suite} Visuals"):
        item = next(iterator)
        
        # 1. Extract
        if isinstance(item, (tuple, list)): 
            img, instr, act = item 
        else:
            img = item['observation']['image_primary']
            instr = item['task']['language_instruction']
            act = item['action']

        # 2. Process Image to uint8 Numpy to save RAM (PIL conversion happens in __getitem__)
        if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        if img.shape[0] == 3 and img.shape[2] != 3: img = np.transpose(img, (1, 2, 0))
        if img.ndim == 4: img = img[0] # Take first frame if chunked

        # 3. Process Action & Text
        if isinstance(act, np.ndarray): act = torch.from_numpy(np.copy(act)).float()
        elif isinstance(act, torch.Tensor): act = act.float()
            
        if isinstance(instr, bytes): instr = instr.decode("utf-8")
        elif isinstance(instr, np.ndarray): instr = str(instr.item()) if instr.ndim == 0 else instr[0].decode("utf-8")
        
        # 4. Group
        if instr not in task_dict:
            task_dict[instr] = {'images': [], 'actions': [], 'texts': []}
            
        task_dict[instr]['images'].append(img)
        task_dict[instr]['actions'].append(act)
        task_dict[instr]['texts'].append(instr)
            
    # --- Execute the 7/3 CoRL Split ---
    unique_tasks = sorted(list(task_dict.keys()))
    train_task_names = unique_tasks[:train_split_ratio]
    held_out_task_names = unique_tasks[train_split_ratio:]
    
    print("\n" + "="*50)
    print(f"🎯 PROJECTOR SPLIT FOR: {suite.upper()}")
    print(f"✅ TRAINING ON ({len(train_task_names)} Tasks)")
    print(f"🛑 HELD-OUT ({len(held_out_task_names)} Tasks) - LEAKAGE PREVENTED")
    print("="*50 + "\n")

    # Aggregate only the 7 training tasks
    train_images, train_actions, train_texts = [], [], []
    for task in train_task_names:
        train_images.extend(task_dict[task]['images'])
        train_actions.extend(task_dict[task]['actions'])
        train_texts.extend(task_dict[task]['texts'])

    test_images, test_actions, test_texts = [], [], []
    for task in held_out_task_names:
        test_images.extend(task_dict[task]['images'])
        test_actions.extend(task_dict[task]['actions'])
        test_texts.extend(task_dict[task]['texts'])

    # Convert actions to a single tensor
    train_action_tensor = torch.stack(train_actions, dim=0)
    test_action_tensor = torch.stack(test_actions, dim=0)
    
    # Create our custom dataset
    train_dataset = VLARAMDataset(train_images, train_texts, train_action_tensor)
    test_dataset = VLARAMDataset(test_images, test_texts, test_action_tensor)

    def dict_collate(batch):
        return {
            "image": [item["image"] for item in batch],
            "instruction": [item["instruction"] for item in batch],
            "actions": torch.stack([item["actions"] for item in batch])
        }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=dict_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, collate_fn=dict_collate)
    return train_dataloader, test_dataloader

def get_vla_projector_dataloader_cached(
    vla_model,
    processor,
    data_root_dir="/mnt/Data/cjimenez/LIBERO/libero/datasets/",
    suite="libero_spatial",
    batch_size=128,
    embed_batch_size=4,
    train_split_ratio=None,  # None = Protocol A (all tasks, no held-out test split)
                              # int  = Protocol B (task-level split, e.g. 7 of 10)
    cache_path=None,
    fallback_cache_path=None,
    device=None,
    vae=None,
    vae_type="text_cond_beta_tcvae",
    use_vision_pool=False,
    text_backbone="clip",
    vla_layer_idx=-1,
    num_fusion_layers=1,
):
    """
    Protocol A (train_split_ratio=None, industry standard for LIBERO):
      - All tasks go into training; no held-out test split is built.
      - Returns (train_dataloader, None).
      - Evaluation is done exclusively via the LIBERO simulator.

    Protocol B (train_split_ratio=7):
      - 7 tasks train / 3 tasks held-out test.
      - Returns (train_dataloader, test_dataloader).
      - Used for OOD generalisation experiments.
    """
    """
    Pre-computes all teacher targets ONCE and returns TensorDataset-backed dataloaders.

    Cache format (v2, CVAE-aware):
      · train_emb        [N, VLA_DIM]   — VLA last-token embeddings
      · train_teacher_mu [N, Z_DIM]     — frozen VAE encoder mu (with text for CVAE)
      · train_teacher_lv [N, Z_DIM]     — frozen VAE encoder logvar
      · train_clip_emb   [N, TEXT_DIM]  — Text embedding (name kept 'clip_emb' for backwards compatibility)
      · test_* same structure

    The hot loop only sees (vla_emb, teacher_mu, teacher_logvar, clip_emb) tensors.
    No VLA, Text Backbone, or VAE inference happens during training.
    """
    if device is None:
        if vla_model is not None:
            device = next(vla_model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_cvae = (vae_type == "text_cvae")
    
    # Map the text dimension for fallback tensors
    text_emb_dim = {"smollm": 960, "openvla_llama": 4096, "clip": 512}.get(text_backbone, 512)

    # ---- 1. Load / build the cache ----------------------------------------
    if cache_path is not None and os.path.exists(cache_path):
        print(f"⚡ Loading pre-computed teacher targets from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        _cache_save_path = cache_path
    elif fallback_cache_path is not None and os.path.exists(fallback_cache_path):
        print(f"⚡ Target cache not found — loading from fallback {fallback_cache_path} (will reteach + save to {cache_path})")
        cached = torch.load(fallback_cache_path, map_location="cpu", weights_only=False)
        _cache_save_path = cache_path  # save reteached result to the correct target path
    else:
        cached = None
        _cache_save_path = cache_path

    if cached is not None:
        if "train_emb" not in cached:
            print("⚡ Task-grouped cache format detected — flattening dynamically...")
            unique_tasks = sorted(list(cached.keys()))
            if train_split_ratio is None:
                train_task_names = unique_tasks
                held_out_task_names = []
                print(f"🚀 Protocol A: training on ALL {len(train_task_names)} tasks — no held-out test split.")
            else:
                train_task_names = unique_tasks[:train_split_ratio]
                held_out_task_names = unique_tasks[train_split_ratio:]
                print(f"🔬 Protocol B: {len(train_task_names)} train tasks | {len(held_out_task_names)} held-out tasks")

            def gather_task_data(task_names):
                if not task_names:
                    return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), []
                
                all_vla = []
                all_mu = []
                all_lv = []
                all_actions = []
                all_instrs = []
                
                for t in task_names:
                    t_data = cached[t]
                    vla = t_data["vla_emb"]
                    if isinstance(vla, np.ndarray):
                        vla = torch.from_numpy(vla)
                    elif vla is None:
                        raise ValueError(f"vla_emb is None for task {t}!")
                    
                    mu = t_data["train_mu"]
                    if isinstance(mu, np.ndarray): mu = torch.from_numpy(mu)
                    
                    lv = t_data["train_logvar"]
                    if isinstance(lv, np.ndarray): lv = torch.from_numpy(lv)
                    
                    acts = t_data["actions"]
                    if isinstance(acts, np.ndarray): acts = torch.from_numpy(acts)
                    
                    all_vla.append(vla)
                    all_mu.append(mu)
                    all_lv.append(lv)
                    all_actions.append(acts)
                    
                    instr = t.replace("_demo", "").replace("_", " ")
                    all_instrs.extend([instr] * len(vla))
                    
                flat_vla = torch.cat(all_vla, dim=0).float()
                flat_mu = torch.cat(all_mu, dim=0).float()
                flat_lv = torch.cat(all_lv, dim=0).float()
                flat_acts = torch.cat(all_actions, dim=0).float()
                
                return flat_vla, flat_mu, flat_lv, flat_acts, all_instrs

            tr_emb, train_teacher_mu, train_teacher_lv, tr_acts_raw, train_instrs = gather_task_data(train_task_names)
            te_emb, test_teacher_mu,  test_teacher_lv,  te_acts_raw, test_instrs  = gather_task_data(held_out_task_names)

            # Load action stats for normalization
            stats_path = "./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt"
            if os.path.exists(stats_path):
                action_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
                suite_name_in_stats = f"{suite}_no_noops"
                if suite_name_in_stats not in action_stats:
                    suite_name_in_stats = "libero_spatial_no_noops"
                stats = action_stats[suite_name_in_stats]['action']
                a_min_t = torch.tensor(stats['min'], dtype=torch.float32)
                a_max_t = torch.tensor(stats['max'], dtype=torch.float32)
                rng     = (a_max_t - a_min_t).clamp(min=1e-6)
                
                # Normalize train
                if len(tr_acts_raw) > 0:
                    train_actions = 2.0 * (tr_acts_raw - a_min_t) / rng - 1.0
                    if 'mask' in stats:
                        mask_t = torch.tensor(stats['mask'], dtype=torch.float32)
                        train_actions = train_actions * mask_t + tr_acts_raw * (1.0 - mask_t)
                else:
                    train_actions = tr_acts_raw
                
                # Normalize test
                if len(te_acts_raw) > 0:
                    test_actions = 2.0 * (te_acts_raw - a_min_t) / rng - 1.0
                    if 'mask' in stats:
                        mask_t = torch.tensor(stats['mask'], dtype=torch.float32)
                        test_actions = test_actions * mask_t + te_acts_raw * (1.0 - mask_t)
                else:
                    test_actions = te_acts_raw
            else:
                print(f"⚠️  No stats file at {stats_path} — actions will NOT be normalised.")
                train_actions = tr_acts_raw
                test_actions = te_acts_raw

            from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel
            print(f"\n📝 Pre-computing text embeddings using {text_backbone}...")
            if text_backbone == "smollm":
                model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
                max_len = 128
            elif text_backbone == "openvla_llama":
                model_id = "meta-llama/Llama-2-7b-hf"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
                max_len = 128
            else: # CLIP
                model_id = "openai/clip-vit-base-patch32"
                tokenizer = CLIPTokenizer.from_pretrained(model_id)
                text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
                max_len = 77

            for p in text_encoder.parameters(): p.requires_grad = False
            text_cache = {}

            def embed_text(instructions):
                all_text = []
                for instr in tqdm.tqdm(instructions, desc=f"{text_backbone} text emb"):
                    if instr not in text_cache:
                        toks = tokenizer(
                            [instr], return_tensors="pt", padding=True, truncation=True, max_length=max_len
                        ).to(device)
                        with torch.no_grad():
                            if text_backbone == "clip":
                                emb = text_encoder(**toks).pooler_output[0]
                            else:
                                outputs = text_encoder(**toks)
                                hidden_states = outputs.last_hidden_state
                                mask = toks.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                                sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                                emb = (sum_embeddings / sum_mask)[0]
                            text_cache[instr] = emb.float().cpu()
                    all_text.append(text_cache[instr])
                return torch.stack(all_text) if all_text else torch.zeros(0, text_emb_dim)

            train_clip_emb = embed_text(train_instrs)
            test_clip_emb = embed_text(test_instrs)

            text_encoder.cpu(); del text_encoder, tokenizer; torch.cuda.empty_cache()

            train_emb = tr_emb
            test_emb = te_emb

            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save({
                    "train_emb":        train_emb,
                    "train_teacher_mu": train_teacher_mu,
                    "train_teacher_lv": train_teacher_lv,
                    "train_clip_emb":   train_clip_emb,
                    "train_actions":    train_actions,
                    "test_emb":         test_emb,
                    "test_teacher_mu":  test_teacher_mu,
                    "test_teacher_lv":  test_teacher_lv,
                    "test_clip_emb":    test_clip_emb,
                    "test_actions":     test_actions,
                }, cache_path)
                print(f"💾 Flattened v2 cache saved to {cache_path}")
        else:
            # v2 cache has teacher_mu/lv; v1 cache has raw actions → upgrade in-place.
            # If actions are present we can always re-teach cheaply (VAE encode only).
            has_teacher = "train_teacher_mu" in cached
            has_actions = "train_actions" in cached

            if has_teacher and not has_actions:
                # Old v2 without saved actions — load as-is, can't re-teach without full rebuild
                train_emb        = cached["train_emb"]
                train_teacher_mu = cached["train_teacher_mu"]
                train_teacher_lv = cached["train_teacher_lv"]
                train_clip_emb   = cached["train_clip_emb"]
                test_emb         = cached["test_emb"]
                test_teacher_mu  = cached["test_teacher_mu"]
                test_teacher_lv  = cached["test_teacher_lv"]
                test_clip_emb    = cached["test_clip_emb"]
                # No actions saved — placeholder zeros (action recon loss disabled for this cache)
                train_actions    = torch.zeros(len(train_emb), 8, 7)
                test_actions     = torch.zeros(len(test_emb),  8, 7)
                print("✅ v2 cache loaded.")
            elif has_actions:
                # v2 with actions (or v1): always recompute teacher targets from saved actions.
                if has_teacher:
                    print("🔄 Cache has actions — recomputing teacher targets (VAE encode only).")
                else:
                    print("⚠️  v1 cache detected — upgrading to v2 (CLIP + VAE encode only, no re-extraction).")
                train_emb     = cached["train_emb"]
                train_actions = cached["train_actions"]
                test_emb      = cached["test_emb"]
                test_actions  = cached["test_actions"]
                
                # Use dynamic text_emb_dim for fallbacks instead of hardcoded 512
                train_clip_emb = cached.get("train_clip_emb", torch.zeros(len(train_emb), text_emb_dim))
                test_clip_emb  = cached.get("test_clip_emb",  torch.zeros(len(test_emb),  text_emb_dim))

                assert vae is not None, "vae must be provided to compute teacher targets"
                _vae_was_training = vae.training; vae.eval()
                _is_cvae_local = (vae_type == "text_cvae")

                def _reteach(actions, text_embs):
                    if len(actions) == 0:
                        return torch.empty(0), torch.empty(0)
                    all_mu, all_lv = [], []
                    for i in tqdm.tqdm(range(0, len(actions), 256), desc="Re-teaching"):
                        a = actions[i:i+256].to(device)
                        t_emb = text_embs[i:i+256].to(device)
                        with torch.no_grad():
                            if _is_cvae_local:
                                # mu, lv = vae.encode(a, t_emb)
                                zero_t = torch.zeros(a.size(0), t_emb.shape[-1], device=device)
                                mu, lv = vae.encode(a, zero_t)
                            else:
                                mu, lv = vae.encode(a)
                        all_mu.append(mu.float().cpu()); all_lv.append(lv.float().cpu())
                    return torch.cat(all_mu), torch.cat(all_lv)

                train_teacher_mu, train_teacher_lv = _reteach(train_actions, train_clip_emb)
                test_teacher_mu,  test_teacher_lv  = _reteach(test_actions,  test_clip_emb)
                if _cache_save_path is not None:
                    os.makedirs(os.path.dirname(_cache_save_path), exist_ok=True)
                    torch.save({
                        "train_emb": train_emb, "train_teacher_mu": train_teacher_mu,
                        "train_teacher_lv": train_teacher_lv, "train_clip_emb": train_clip_emb,
                        "train_actions": train_actions,
                        "test_emb": test_emb, "test_teacher_mu": test_teacher_mu,
                        "test_teacher_lv": test_teacher_lv, "test_clip_emb": test_clip_emb,
                        "test_actions": test_actions,
                    }, _cache_save_path)
                    print(f"💾 Cache updated with fresh teacher targets → {_cache_save_path}")
    else:
        # ---- 1a. Gather raw data -----------------------------------------------
        suite_name = f'{suite}_no_noops'
        print(f"\n⏳ Loading {suite_name} visuals for embedding pass...")

        ds = FastActionRLDSDataset(
            data_root_dir=data_root_dir,
            data_mix=[suite_name],
            batch_transform=identity_transform,
            train=True,
            resize_resolution=(224, 224),
            return_visuals=True,
        )

        task_dict = {}
        iterator = iter(ds)
        for _ in tqdm.tqdm(range(len(ds)), desc=f"Extracting {suite} Visuals"):
            item = next(iterator)
            if isinstance(item, (tuple, list)):
                img, instr, act = item
            else:
                img   = item['observation']['image_primary']
                instr = item['task']['language_instruction']
                act   = item['action']

            if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
            if img.dtype in [np.float32, np.float64]:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            if img.shape[0] == 3 and img.shape[2] != 3: img = np.transpose(img, (1, 2, 0))
            if img.ndim == 4: img = img[0]

            if isinstance(act, np.ndarray): act = torch.from_numpy(np.copy(act)).float()
            elif isinstance(act, torch.Tensor): act = act.float()
            if isinstance(instr, bytes): instr = instr.decode("utf-8")
            elif isinstance(instr, np.ndarray):
                instr = str(instr.item()) if instr.ndim == 0 else instr[0].decode("utf-8")

            if instr not in task_dict:
                task_dict[instr] = {'images': [], 'actions': []}
            task_dict[instr]['images'].append(img)
            task_dict[instr]['actions'].append(act)

        unique_tasks = sorted(list(task_dict.keys()))

        if train_split_ratio is None:
            # Protocol A: all tasks → training, no held-out test split.
            train_task_names    = unique_tasks
            held_out_task_names = []
            print(f"🚀 Protocol A: training on ALL {len(train_task_names)} tasks — no held-out test split.")
        else:
            # Protocol B: task-level split.
            train_task_names    = unique_tasks[:train_split_ratio]
            held_out_task_names = unique_tasks[train_split_ratio:]
            print(f"🔬 Protocol B: {len(train_task_names)} train tasks | {len(held_out_task_names)} held-out tasks")

        def gather(task_names):
            imgs, acts, instrs = [], [], []
            for t in task_names:
                imgs.extend(task_dict[t]['images'])
                acts.extend(task_dict[t]['actions'])
                instrs.extend([t] * len(task_dict[t]['images']))
            return imgs, torch.stack(acts), instrs

        tr_imgs, tr_acts, tr_instrs = gather(train_task_names)
        te_imgs, te_acts, te_instrs = gather(held_out_task_names) if held_out_task_names else ([], torch.zeros(0), [])

        # ---- 1b. VLA embedding pass (hook-based) --------------------------------
        def _get_last_transformer_layer(model):
            candidates = [
                lambda m: m.language_model.model.layers[-1],
                lambda m: m.model.language_model.model.layers[-1],
                lambda m: m.model.layers[-1],
                lambda m: m.transformer.h[-1],
                lambda m: m.model.decoder.layers[-1],
            ]
            for fn in candidates:
                try: return fn(model)
                except AttributeError: continue
            return None

        def embed_vla(images, instructions, desc, use_vision_pool=False, layer_idx=-1):
            all_embs = []
            last_layer = _get_last_transformer_layer(vla_model)
            use_hook   = last_layer is not None
            captured   = {}
            img_token_id = None
            if use_vision_pool:
                img_token_id = getattr(getattr(vla_model, "config", None), "image_token_index", None)
                if img_token_id is None:
                    use_vision_pool = False  
            if use_hook:
                def hook_fn(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    captured['last_hs'] = hs.float().detach().cpu()
                handle = last_layer.register_forward_hook(hook_fn)
            effective_bs = embed_batch_size if use_hook else 1
            try:
                for i in tqdm.tqdm(range(0, len(images), effective_bs), desc=desc):
                    batch_imgs   = [Image.fromarray(img) for img in images[i:i+effective_bs]]
                    batch_instrs = instructions[i:i+effective_bs]
                    prompts      = [f"In: {ins}\nOut: " for ins in batch_instrs]
                    inputs = processor(
                        text=prompts, images=batch_imgs,
                        padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                    if hasattr(inputs, "pixel_values"):
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                    with torch.no_grad():
                        out = vla_model(**inputs, output_hidden_states=not use_hook, return_dict=True)
                    last_hs = captured['last_hs'] if use_hook else out.hidden_states[-1].float().cpu()
                    if use_vision_pool:
                        ids_cpu = inputs.input_ids.cpu()
                        batch_embs = []
                        for b in range(last_hs.shape[0]):
                            vis_mask = (ids_cpu[b] == img_token_id)
                            if vis_mask.any():
                                batch_embs.append(last_hs[b][vis_mask].mean(dim=0))
                            else:
                                idx = int(inputs.attention_mask[b].sum().item()) - 1
                                batch_embs.append(last_hs[b, idx])
                        emb = torch.stack(batch_embs, dim=0)
                    else:
                        if layer_idx == -1:
                            last_tok_idx = (inputs.attention_mask.sum(dim=1) - 1).cpu()
                            emb = last_hs[torch.arange(len(prompts)), last_tok_idx]
                        else:
                            max_len = inputs.attention_mask.sum(dim=1).max().item()
                            seq_emb = last_hs[:, :max_len, :]
                            emb = seq_emb.view(seq_emb.size(0), seq_emb.size(1), 256, 16).mean(dim=-1)
                    all_embs.append(emb)
                    del out; captured.pop('last_hs', None); torch.cuda.empty_cache()
            finally:
                if use_hook: handle.remove()
            return torch.cat(all_embs, dim=0)

        print("\n🧠 Pre-computing VLA embeddings (train)...")
        train_emb = embed_vla(tr_imgs, tr_instrs, "Embedding train", use_vision_pool=use_vision_pool, layer_idx=vla_layer_idx)
        if te_imgs:
            print("🧠 Pre-computing VLA embeddings (test)...")
            test_emb = embed_vla(te_imgs, te_instrs, "Embedding test", use_vision_pool=use_vision_pool, layer_idx=vla_layer_idx)
        else:
            test_emb = torch.zeros(0, train_emb.shape[-1])  # empty placeholder

        # ---- 1c. Text embedding pass (Adaptive Tokenizer) -----------------------
        from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel
        print(f"\n📝 Pre-computing text embeddings using {text_backbone}...")
        
        if text_backbone == "smollm":
            model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            text_encoder = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        elif text_backbone == "openvla_llama":
            model_id = "meta-llama/Llama-2-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            text_encoder = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        else: # Fallback to CLIP
            model_id = "openai/clip-vit-base-patch32"
            tokenizer = CLIPTokenizer.from_pretrained(model_id)
            text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()

        for p in text_encoder.parameters(): p.requires_grad = False

        text_cache = {}  # instruction → [DIM] to avoid re-encoding duplicates

        def embed_text(instructions):
            all_text = []
            for instr in tqdm.tqdm(instructions, desc=f"{text_backbone} text emb"):
                if instr not in text_cache:
                    toks = tokenizer(
                        [instr], return_tensors="pt", padding=True, truncation=True, max_length=128
                    ).to(device)
                    with torch.no_grad():
                        if text_backbone == "clip":
                            emb = text_encoder(**toks).pooler_output[0]
                        else:
                            outputs = text_encoder(**toks)
                            hidden_states = outputs.last_hidden_state
                            mask = toks.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                            emb = (sum_embeddings / sum_mask)[0]
                        text_cache[instr] = emb.float().cpu()
                all_text.append(text_cache[instr])
            return torch.stack(all_text)

        # Keeping variable name 'clip_emb' for backwards compatibility with existing caches
        train_clip_emb = embed_text(tr_instrs)
        test_clip_emb  = embed_text(te_instrs) if te_instrs else torch.zeros(0, text_emb_dim)

        # Free the text model from GPU to make room for VAE
        text_encoder.cpu(); del text_encoder, tokenizer; torch.cuda.empty_cache()

        # ---- 1d. VAE teacher encode pass ----------------------------------------
        print("\n🎓 Pre-computing VAE teacher targets (mu, logvar)...")
        assert vae is not None, "vae must be provided to pre-compute teacher targets"
        vae_was_training = vae.training
        vae.eval()

        def encode_teacher(actions, text_embs, desc):
            all_mu, all_lv = [], []
            bs = 256
            for i in tqdm.tqdm(range(0, len(actions), bs), desc=desc):
                a_batch = actions[i:i+bs].to(device)
                t_batch = text_embs[i:i+bs].to(device)  # FIX: Slice and send real text embeddings!
                with torch.no_grad():
                    if is_cvae:
                        mu, lv = vae.encode(a_batch, t_batch) 
                    else:
                        mu, lv = vae.encode(a_batch)
                all_mu.append(mu.float().cpu())
                all_lv.append(lv.float().cpu())
            return torch.cat(all_mu, dim=0), torch.cat(all_lv, dim=0)

        train_teacher_mu, train_teacher_lv = encode_teacher(tr_acts, train_clip_emb, "Teacher encode train")
        if len(te_acts) > 0:
            test_teacher_mu, test_teacher_lv = encode_teacher(te_acts, test_clip_emb, "Teacher encode test")
        else:
            test_teacher_mu = torch.zeros(0, train_teacher_mu.shape[-1])
            test_teacher_lv = torch.zeros(0, train_teacher_lv.shape[-1])

        if vae_was_training: vae.train()

        # ---- 1e. Save v2 cache --------------------------------------------------
        train_actions = tr_acts
        test_actions  = te_acts if len(te_acts) > 0 else torch.zeros(0, *tr_acts.shape[1:])

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                "train_emb":        train_emb,
                "train_teacher_mu": train_teacher_mu,
                "train_teacher_lv": train_teacher_lv,
                "train_clip_emb":   train_clip_emb,
                "train_actions":    train_actions,
                "test_emb":         test_emb,
                "test_teacher_mu":  test_teacher_mu,
                "test_teacher_lv":  test_teacher_lv,
                "test_clip_emb":    test_clip_emb,
                "test_actions":     test_actions,
            }, cache_path)
            print(f"💾 v2 cache saved to {cache_path}")

    print(f"✅ Cache shapes — train emb: {train_emb.shape}, teacher_mu: {train_teacher_mu.shape}")

    # ---- 2. Build dataloaders -----------------------------------------------
    train_dataset    = TensorDataset(train_emb, train_teacher_mu, train_teacher_lv, train_clip_emb, train_actions)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)

    # Protocol A: no held-out test split — return None for test_dataloader.
    # Protocol B: build test dataloader from held-out task embeddings.
    if len(test_emb) > 0:
        test_dataset    = TensorDataset(test_emb, test_teacher_mu, test_teacher_lv, test_clip_emb, test_actions)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=0, drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader