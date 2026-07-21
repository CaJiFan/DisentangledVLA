import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
from .analyze_trajectories import find_task_aware_boundaries

# Define the absolute maximum length of a skill. 
# Based on your logs, Picks can be ~140 frames. Let's use 160 to be safe.
MAX_SEQ_LEN = 168 # multi-liple of 8 for the conv architecture

def build_padded_dataset(source_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Trackers for our new dataset
    all_padded_actions = []
    all_masks = []
    all_labels = []
    
    # Create a simple vocabulary mapping for conditioning later
    skill_vocab = {
        "reach":    0,
        "pick":     1,
        "place":    2,
        "push":     3, 
        "pull":     4, 
        "turn":     5,
        "retract":  6
    }

    hdf5_files = list(glob(f'{source_dir}/*.hdf5'))
    
    print(f"📦 Extracting Semantic Skills from {len(hdf5_files)} files...")
    longest_skill = 0
    for filepath in tqdm(hdf5_files):
        task_name = os.path.basename(filepath).replace('.hdf5', '')
        
        with h5py.File(filepath, 'r') as f:
            for demo_key in f['data'].keys():
                demo_group = f['data'][demo_key]
                actions = demo_group['actions'][:]
                eef_pos = demo_group['obs']['ee_pos'][:]
                
                # Use your perfected heuristic!
                split_indices, skill_names = find_task_aware_boundaries(actions, eef_pos, task_name)
                
                for i in range(len(split_indices) - 1):
                    start_idx = split_indices[i]
                    end_idx = split_indices[i + 1]
                    skill_name = skill_names[i]
                    
                    # Extract the core skill name fro og (e.g., "0_reach", "1_pick", etc.)
                    skill_name = skill_name.split("_")[-1].lower()  
                    
                    chunk_actions = actions[start_idx:end_idx]
                    seq_len = len(chunk_actions)
                    
                    if seq_len == 0: continue

                    # if seq_len > longest_skill:
                    #     longest_skill = seq_len
                    #     print(f"⚠️ New longest skill found: {longest_skill} frames in {skill_name} of {demo_key} ({task_name})")

                    # Truncate if anomalously long
                    if seq_len > MAX_SEQ_LEN:
                        chunk_actions = chunk_actions[:MAX_SEQ_LEN]
                        seq_len = MAX_SEQ_LEN
                        
                    # 1. Zero Padding
                    padded_actions = np.zeros((MAX_SEQ_LEN, 7), dtype=np.float32)
                    padded_actions[:seq_len] = chunk_actions
                    
                    # 2. Binary Mask (1 for real data, 0 for padding)
                    mask = np.zeros((MAX_SEQ_LEN,), dtype=np.float32)
                    mask[:seq_len] = 1.0
                    
                    all_padded_actions.append(padded_actions)
                    all_masks.append(mask)
                    label = skill_vocab.get(skill_name, -1)
                    # print(f"Mapping skill '{skill_name}' to label {label}")
                    if label == -1:
                        print(f"⚠️ Warning: Skill '{skill_name}' not found in vocab. Assigning label -1.")
                    all_labels.append(label)

    # Save to a single, hyper-optimized HDF5 file
    print(f"\n💾 Saving {len(all_padded_actions)} distinct skills to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('actions', data=np.stack(all_padded_actions))
        f.create_dataset('masks', data=np.stack(all_masks))
        f.create_dataset('labels', data=np.array(all_labels, dtype=np.int32))
        
    print("✅ Dataset built successfully!")

if __name__ == "__main__":

    for SUITE_NAME in ["libero_goal", "libero_spatial"]:
        DATASET_DIR = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{SUITE_NAME}_no_noops_hdf5"
        OUTPUT_PATH = f"./processed_data/semantic_{SUITE_NAME}.hdf5"
        build_padded_dataset(DATASET_DIR, OUTPUT_PATH)