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
    text_backbone="clip"
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
    elif text_backbone == "octo_t5":
        model_id = "t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
            task_dict[instr] = {'actions': [], 'texts': []}

        task_dict[instr]['actions'].append(act)
        task_dict[instr]['texts'].append(text_cache[instr])
            
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
    for task in train_task_names:
        train_actions.extend(task_dict[task]['actions'])
        train_texts.extend(task_dict[task]['texts'])

    train_action_tensor = torch.stack(train_actions, dim=0)
    train_text_tensor   = torch.stack(train_texts,   dim=0)
    print(f"Train Actions: {train_action_tensor.shape}, Texts: {train_text_tensor.shape}")

    action_stats = ds.dataset_statistics
    print(f"✅ Extracted Official Dataset Statistics for Un-normalization.")

    train_dataset    = TensorDataset(train_action_tensor, train_text_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)

    if held_out_task_names:
        test_actions, test_texts = [], []
        for task in held_out_task_names:
            test_actions.extend(task_dict[task]['actions'])
            test_texts.extend(task_dict[task]['texts'])
        test_action_tensor = torch.stack(test_actions, dim=0)
        test_text_tensor   = torch.stack(test_texts,   dim=0)
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
    action_stats = torch.load(stats_path)
    
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
    action_stats = torch.load(stats_path)
    
    train_dataset = TensorDataset(train_action_tensor, train_text_tensor)
    test_dataset = TensorDataset(test_action_tensor, test_text_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    return train_dataloader, test_dataloader, action_stats

def log_video_probe(vae, step, suite_name, stats_path, device, probe_task_name, split_name, chunk_size, text_backbone):
    """Runs a receding-horizon video probe for chunk-based VAEs and logs it to WandB."""
    print(f"\n🎥 Generating {split_name.upper()} Video Probe (Chunk Size {chunk_size}) for Step {step}...")
    
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
        gt_actions = f[f"data/{probe_demo_id}/actions"][:]
        init_state = f[f"data/{probe_demo_id}/states"][0] 
    
    # 3. Embed Instruction Dynamically
    if text_backbone == "smollm":
        model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id).to(device).eval()
    elif text_backbone == "octo_t5":
        model_id = "t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    action_stats = torch.load(stats_path)
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
    env.reset()
    env.set_init_state(init_state) 
    
    video_path = f"/tmp/vae_probe_{split_name}_{step}.mp4"
    writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)
    
    seq_len = len(norm_gt)
    
    task_success = False

    # 6. Full-chunk execution loop: re-plan every chunk_size steps
    t = 0
    while t < seq_len:
        # Extract the chunk window
        if t + chunk_size <= seq_len:
            chunk = norm_gt[t : t + chunk_size]
        else:
            valid_len = seq_len - t
            pad_len = chunk_size - valid_len
            padding = norm_gt[-1].repeat(pad_len, 1)
            chunk = torch.cat([norm_gt[t : seq_len], padding], dim=0)

        chunk = chunk.unsqueeze(0)  # (1, chunk_size, 7)

        with torch.no_grad():
            encode_args = (chunk, text_emb) if vae.encode.__code__.co_argcount > 2 else (chunk,)
            mu, _ = vae.encode(*encode_args)
            pred_chunk_norm = vae.decode(mu, text_emb)[0]  # (chunk_size, 7)

        # Execute all steps in the predicted chunk before re-planning
        steps_to_execute = chunk_size if t + chunk_size <= seq_len else seq_len - t
        done = False
        for i in range(steps_to_execute):
            pred_action_norm = pred_chunk_norm[i]

            pred_action = (pred_action_norm + 1.0) / 2.0 * (action_max - action_min) + action_min
            pred_action = pred_action_norm * action_mask + pred_action * (1.0 - action_mask)
            action_np = pred_action.cpu().numpy()
            action_np[-1] = 1.0 if action_np[-1] > 0.0 else -1.0

            obs, reward, done, info = env.step(action_np)
            img = np.flipud(obs['agentview_image'])
            writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))

            if done:
                task_success = True
                break

        t += chunk_size
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
        })
    status = "✅ SUCCESS" if task_success else "❌ FAILED"
    print(f"🎥 {split_name.upper()} Video uploaded to WandB! Outcome: {status}")

def _make_openvla_emb_fn(vla_model, processor, device, use_vision_pool=False, vla_layer_idx=-1):
    """
    Build an emb_fn callable from an OpenVLA model + processor.
    emb_fn(image_pil, instruction) → torch.Tensor (1, vla_dim) float32 on `device`
    """
    import torch
    vla_prompt_tmpl = "In: {}\nOut: "
    _img_token_id = getattr(vla_model.config, "image_token_index",
                    getattr(vla_model.config, "image_token_id", None)) if use_vision_pool else None

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
        hs = out.hidden_states[vla_layer_idx]  # (1, L, D)
        if use_vision_pool and _img_token_id is not None:
            img_mask = (inputs["input_ids"] == _img_token_id).unsqueeze(-1).float()  # (1, L, 1)
            vision_tokens = hs * img_mask
            n_img = img_mask.sum(dim=1).clamp(min=1)
            emb = (vision_tokens.sum(dim=1) / n_img).float().to(device)  # (1, D)
        else:
            last_tok_idx = inputs.attention_mask.sum(dim=1) - 1
            emb = hs[0, last_tok_idx[0]].unsqueeze(0).float().to(device)  # (1, D)
        return emb

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


def _make_smolvla_emb_fn(policy, device, vla_layer_idx=-1):
    """
    Build an emb_fn for SmolVLA.
    emb_fn(image_pil, instruction) -> torch.Tensor (1, 960) float32 on `device`

    The policy should already be on `device` when emb_fn is called.
    """
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    flow = policy.model
    vlm  = flow.vlm_with_expert
    vlm_dtype   = next(vlm.parameters()).dtype
    state_dtype = flow.state_proj.weight.dtype

    _resize = getattr(policy.config, "resize_imgs_with_padding", None) or (256, 256)
    _tgt_h, _tgt_w = _resize

    target_layer = vlm.vlm.model.layers[vla_layer_idx]
    captured = {}
    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured['hs'] = hs.float().detach().cpu()
    handle = target_layer.register_forward_hook(hook_fn)

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

        # Zero state (state_proj is float32 even when VLM backbone is bf16)
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

        # VLM forward — fill_kv_cache=True routes all layers through forward_attn_layer,
        # which handles inputs_embeds[1]=None gracefully (skips action expert).
        (prefix_out, _), _ = vlm.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        
        prefix_out = captured['hs']

        # Mean-pool over valid prefix token positions → (1, 960)
        valid = prefix_pad.unsqueeze(-1).float()
        emb   = (prefix_out.float() * valid).sum(1) / valid.sum(1).clamp(min=1)
        
        # We don't remove the hook here because emb_fn is called multiple times
        captured.pop('hs', None)
        return emb.to(device)  # (1, 960)

    return emb_fn


def log_projector_video_probe(
    vae, projector, vla_model, processor,
    step, suite_name, stats_path, device,
    probe_task_name, split_name, chunk_size,
    emb_fn=None, normalize_emb=False, use_vision_pool=False,
    text_backbone="clip", vla_layer_idx=-1
):
    if not _HAS_LIBERO:
        print("⚠️  LIBERO not available — skipping projector video probe.")
        return
    # Respect MUJOCO_GL env var (osmesa or egl); default to egl only if unset.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("PYOPENGL_PLATFORM", None)
    """
    Chunk-based receding-horizon video probe for the full projector pipeline:
      env obs image + instruction → VLA → projector → z_mu → VAE decoder → action chunk

    Executes the FULL predicted chunk between re-planning calls (not just the first
    action). This is the correct execution strategy for a chunk policy:
      - Temporal coherence within the chunk is preserved.
      - Gripper transitions encoded mid-chunk are actually executed.
      - VLA is called ~T/chunk_size times instead of T times (less jitter).

    Args:
        emb_fn: Optional callable (image_pil, instruction) → torch.Tensor (1, D) on `device`.
                If None, falls back to building one from vla_model + processor (OpenVLA path).
                Pass OctoWorker.make_emb_fn() here for Octo-based projectors.
        text_backbone: The text backbone to use for embedding the instruction.
    """
    if emb_fn is None:
        # Backwards-compatible: build from OpenVLA model/processor
        emb_fn = _make_openvla_emb_fn(vla_model, processor, device, use_vision_pool=use_vision_pool, vla_layer_idx=vla_layer_idx)

    print(f"\n🎥 Generating Projector {split_name.upper()} Video Probe for Step {step}...")
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

    # 2. Load init state from HDF5
    raw_suite_name = suite_name.replace("_no_noops", "")
    data_dir = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{raw_suite_name}_no_noops_hdf5"
    hdf5_path = os.path.join(data_dir, f"{probe_task_name}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"🧨 Projector Video Probe Error: Could not find HDF5 at {hdf5_path}")
        return
    with h5py.File(hdf5_path, "r") as f:
        gt_actions = f[f"data/{probe_demo_id}/actions"][:]
        init_state = f[f"data/{probe_demo_id}/states"][0]

    print(f"  [probe] Loading {text_backbone} to embed VAE instruction...")
    if text_backbone == "smollm":
        model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
    elif text_backbone == "octo_t5":
        model_id = "t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    text_inputs = tokenizer([instruction], padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        if text_backbone == "clip":
            vae_text_emb = text_encoder(**text_inputs).pooler_output
        else:
            outputs = text_encoder(**text_inputs)
            hidden_states = outputs.last_hidden_state
            mask = text_inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            vae_text_emb = (sum_embeddings / sum_mask) # Shape: (1, D)

    # 🚨 CRITICAL: Free VRAM immediately before booting up MuJoCo!
    del text_encoder, tokenizer, text_inputs
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print("  [probe] Text backbone VRAM freed.")

    # 4. Action stats for un-normalisation
    action_stats = torch.load(stats_path, weights_only=False)
    stats = action_stats[f"{raw_suite_name}_no_noops"]['action']
    action_min = torch.tensor(stats['min']).float().to(device)
    action_max = torch.tensor(stats['max']).float().to(device)
    action_mask = torch.tensor(stats['mask']).float().to(device)

    # 5. Initialise MuJoCo environment
    env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    env.set_init_state(init_state)
    obs, _, _, _ = env.step(np.zeros(7))  # warmup step to get first rendered obs

    video_path = f"/tmp/proj_probe_{split_name}_{step}.mp4"
    writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)

    projector.eval()
    # Give the policy generous time: GT demos are optimal-speed; a learned chunk
    # policy can follow a slightly longer trajectory and still succeed.
    # min 600 steps prevents trivially short demos from cutting off evaluation.
    seq_len = max(len(gt_actions) * 2, 600)
    done = False
    task_success = False
    first_action_logged = False

    # 6. Chunk-based receding horizon: re-plan every chunk_size steps
    for t in range(0, seq_len, chunk_size):
        if done:
            break

        # Raw image — no rotation, consistent with build_octo_cache.py which
        # reads raw HDF5 images (same MuJoCo output format) with no preprocessing.
        obs_img_pil = Image.fromarray(obs['agentview_image'].astype(np.uint8))

        with torch.no_grad():
            vla_emb = emb_fn(obs_img_pil, instruction).to(device)  # (1, D)
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
            pred_chunk_norm = vae.decode(pred_mu, vae_text_emb)  # [1, chunk_size, 7]

        # Execute all steps in the predicted chunk
        for k in range(chunk_size):
            if t + k >= seq_len:
                break
            pred_action_norm = pred_chunk_norm[0, k]  # [7], in [-1, 1] (Tanh decoder output)
            # Unnorm: inverse of (x - min) / (max - min) * 2 - 1
            pred_action_unnorm = (pred_action_norm + 1.0) / 2.0 * (action_max - action_min) + action_min
            # mask=1 → dim was normalised → use unnorm; mask=0 → dim kept raw (e.g. gripper ±1) → use norm
            pred_action = pred_action_unnorm * action_mask + pred_action_norm * (1.0 - action_mask)
            action_np = pred_action.cpu().numpy()
            action_np[-1] = 1.0 if action_np[-1] > 0.0 else -1.0  # binarise gripper

            # Log first action of first chunk for quick diagnostic
            if not first_action_logged:
                gt_a0 = gt_actions[0] if len(gt_actions) > 0 else None
                print(f"  [probe diag] pred_action (step 0, unnorm): {np.round(action_np, 3)}")
                if gt_a0 is not None:
                    print(f"  [probe diag] gt_action   (step 0, raw  ): {np.round(gt_a0, 3)}")
                first_action_logged = True

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
        wandb.log({
            f"eval_videos/projector_{split_name}_probe": wandb.Video(video_path, fps=30, format="mp4"),
            f"eval_metrics/probe_{split_name}_success": float(task_success),
            "global_step": step
        })
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
        })
    print(f"🎥 GT {split_name.upper()} reference video uploaded to WandB!")
 
def log_octo_gt_video_probe(
    step, suite_name, stats_path, device,
    probe_task_name, split_name,
    octo_gt_fn, chunk_size=4,
):
    if not _HAS_LIBERO:
        print("⚠️  LIBERO not available — skipping Octo GT video probe.")
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("PYOPENGL_PLATFORM", None)
    """
    Octo GT video probe: runs the raw Octo model (no projector, no VAE) on the
    live environment and logs the video to WandB.

    The `octo_gt_fn` callable (from OctoWorker.make_gt_fn()) handles:
      - 180° image rotation [::-1,::-1] matching Octo's training distribution
      - 2-frame history window inside the child process
      - sample_actions() with unnormalization

    Gripper is binarised identically to the projector probe for fair comparison.

    Args:
        octo_gt_fn:  callable(image_np, instruction, reset=False) → (H, 7) np.ndarray
                     where H is Octo's actual prediction horizon (typically 4).
        chunk_size:  re-planning cadence — how many steps to execute before
                     querying Octo again. Should match Octo's prediction horizon
                     (4 for octo-base/small). Outer loop steps by len(action_chunk)
                     so skipping never occurs even if this differs.
    """
    print(f"\n🎥 Generating Octo-GT {split_name.upper()} Video Probe for Step {step}...")
    probe_demo_id = "demo_0"

    # 1. Match task to benchmark
    bmark = benchmark.get_benchmark_dict()[suite_name]()
    task_id = None
    for i in range(bmark.get_num_tasks()):
        if bmark.get_task(i).name + '_demo' == probe_task_name:
            task_id = i
            break
    if task_id is None:
        print(f"⚠️ Octo GT probe task '{probe_task_name}' not found. Skipping.")
        return

    instruction = bmark.get_task(task_id).language

    # 2. Load init state from HDF5
    raw_suite_name = suite_name.replace("_no_noops", "")
    data_dir = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{raw_suite_name}_no_noops_hdf5"
    hdf5_path = os.path.join(data_dir, f"{probe_task_name}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"🧨 Octo GT Probe Error: Could not find HDF5 at {hdf5_path}")
        return
    with h5py.File(hdf5_path, "r") as f:
        gt_actions = f[f"data/{probe_demo_id}/actions"][:]
        init_state = f[f"data/{probe_demo_id}/states"][0]

    # 3. Initialise MuJoCo environment
    env_args = {"bddl_file_name": os.path.join(bmark.get_task_bddl_file_path(task_id))}
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    env.set_init_state(init_state)
    obs, _, _, _ = env.step(np.zeros(7))  # warmup step

    video_path = f"/tmp/octo_gt_probe_{split_name}_{step}.mp4"
    writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)

    seq_len = len(gt_actions)
    done = False
    t = 0

    # 4. Chunk-based receding horizon: re-query Octo every len(action_chunk) steps.
    # t increments by the ACTUAL number of actions returned, not a fixed chunk_size,
    # so no demo steps are ever skipped regardless of Octo's prediction horizon.
    while t < seq_len and not done:
        # Pass raw env image; OctoWorker applies [::-1,::-1] internally
        image_np = obs['agentview_image'].astype(np.uint8)
        reset_hist = (t == 0)

        try:
            action_chunk = octo_gt_fn(image_np, instruction, reset=reset_hist)
            # action_chunk: (H, 7) float32, already unnormalized
        except Exception as exc:
            print(f"[Octo GT probe] act_gt failed at t={t}: {exc}")
            break

        executed = 0
        for k in range(len(action_chunk)):
            if t + k >= seq_len:
                break
            action_np = action_chunk[k].copy()
            # Binarise gripper (dim 6): positive → open (+1), negative → close (−1)
            action_np[-1] = 1.0 if action_np[-1] > 0.0 else -1.0

            obs, reward, done, info = env.step(action_np)
            img = np.flipud(obs['agentview_image'])
            writer.append_data(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
            executed += 1
            if done:
                break

        t += executed if executed > 0 else len(action_chunk)  # advance past this chunk

    writer.close()
    env.close()

    if wandb.run is not None:
        wandb.log({
            f"gt_videos/octo_gt_{split_name}_probe": wandb.Video(video_path, fps=30, format="mp4"),
            "global_step": step
        })
    print(f"🎥 Octo GT {split_name.upper()} video uploaded to WandB!")

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
    text_emb_dim = {"smollm": 960, "octo_t5": 768, "openvla_llama": 4096, "clip": 512}.get(text_backbone, 512)

    # ---- 1. Load / build the cache ----------------------------------------
    if cache_path is not None and os.path.exists(cache_path):
        print(f"⚡ Loading pre-computed teacher targets from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        _cache_save_path = cache_path
    elif fallback_cache_path is not None and os.path.exists(fallback_cache_path):
        print(f"⚡ Target cache not found — loading from fallback {fallback_cache_path} (will reteach + save to {cache_path})")
        cached = torch.load(fallback_cache_path, map_location="cpu")
        _cache_save_path = cache_path  # save reteached result to the correct target path
    else:
        cached = None
        _cache_save_path = cache_path

    if cached is not None:
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
            if _vae_was_training: vae.train()

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
        elif text_backbone == "octo_t5":
            model_id = "t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
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