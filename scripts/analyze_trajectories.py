from glob import glob
import os
import h5py
import numpy as np
import imageio
from tqdm import tqdm

def find_semantic_skill_boundaries(actions, eef_pos, gripper_idx=6, threshold=0.0):
    """
    Splits a standard Pick-and-Place trajectory into 3 semantic phases:
    1. Pick (Reach + Grasp + Lift to Apex)
    2. Place (Transport to target + Release)
    3. Retract (Move away empty-handed)
    """
    gripper_actions = actions[:, gripper_idx]
    binary_gripper = (gripper_actions > threshold).astype(int)
    
    # Find all moments the gripper state flips
    flips = np.where(np.diff(binary_gripper) != 0)[0]
    
    # Fallback if the robot never uses the gripper (e.g., just pushing)
    if len(flips) == 0:
        return [0, len(actions)]
        
    # 1. GRASP INDEX: First time the gripper closes
    grasp_idx = flips[0] + 1
    
    # 2. RELEASE INDEX: First time the gripper opens AFTER the grasp
    subsequent_flips = flips[flips >= grasp_idx]
    if len(subsequent_flips) == 0:
        release_idx = len(actions) - 1 # Held the object until the video ended
    else:
        release_idx = subsequent_flips[0] + 1
        
    # 3. APEX INDEX: Highest Z-coordinate (height) between grasp and release
    # eef_pos[:, 2] is the Z-axis. 
    z_trajectory_during_transport = eef_pos[grasp_idx:release_idx, 2]
    
    if len(z_trajectory_during_transport) > 0:
        apex_offset = np.argmax(z_trajectory_during_transport)
        apex_idx = grasp_idx + apex_offset
    else:
        apex_idx = grasp_idx # Fallback
        
    # Compile the boundaries (Start -> Apex -> Release -> End)
    boundaries = [0, apex_idx, release_idx, len(actions)]
    
    # Clean up duplicate indices in case of instantly dropped objects
    boundaries = sorted(list(set(boundaries)))
    return boundaries

def find_task_aware_boundaries(actions, eef_pos, task_name, gripper_idx=6, threshold=0.0):
    """
    Routes the segmentation strategy based on the semantic name of the task,
    handling atomic actions, hooking, picking, AND composite multi-stage tasks.
    """
    task_name = task_name.lower()
  
    # ==========================================
    # STRATEGY 0: HYBRID / COMPOSITE TASKS
    # e.g., "open_the_top_drawer_and_put_the_bowl_inside"
    # ==========================================
    if "open" in task_name and ("put" in task_name or "place" in task_name or "store" in task_name):
       # --- PHASE 1: The Drawer (Kinematics via 3D Velocity & Strict Plunge Rejection) ---
        # phase 1: hook and pull
        hook_idx, end_pull_idx = state_machine_approach(eef_pos)
        
        # --- PHASE 2: The Bowl (Look for gripper ONLY AFTER the pull) ---
        gripper_actions = actions[:, gripper_idx]
        binary_gripper = (gripper_actions > threshold).astype(int)
        flips = np.where(np.diff(binary_gripper) != 0)[0]
        
        # Filter to only look at gripper actions that happen AFTER the drawer is open
        valid_grasps = flips[flips >= end_pull_idx]
        
        if len(valid_grasps) > 0:
            # print(f"  > Found {len(valid_grasps)} gripper flips after the pull, starting at frame {valid_grasps[0]}")
            # First grasp after the pull is the bowl
            grasp_idx = valid_grasps[0] + 1
            
            # Last release is the drop
            subsequent_flips = flips[flips >= grasp_idx]
            release_idx = subsequent_flips[-1] + 1 if len(subsequent_flips) > 0 else len(actions) - 1
            
            z_trajectory = eef_pos[grasp_idx:release_idx, 2]
            apex_idx = grasp_idx + np.argmax(z_trajectory) if len(z_trajectory) > 0 else grasp_idx
            
            # Added reach_end_idx here!
            raw_boundaries = [0, hook_idx, end_pull_idx, apex_idx, release_idx, len(actions)]
            raw_names = ["0_Reach", "1_Pull", "2_Pick", "3_Place", "4_Retract"]
        else:
            # Added reach_end_idx here too!
            raw_boundaries = [0, hook_idx, end_pull_idx, len(actions)]
            raw_names = ["0_Reach", "1_Pull", "2_Retract"]
            
        # Clean up overlapping boundaries
        boundaries, names = [raw_boundaries[0]], []
        for i in range(1, len(raw_boundaries)):
            if raw_boundaries[i] > raw_boundaries[i-1]:
                boundaries.append(raw_boundaries[i])
                names.append(raw_names[i-1])
                
        return boundaries, names
     
    # ==========================================
    # STRATEGY 1: PUSH / TURN (Tabletop & Constrained Interactions)
    # ==========================================
    if "push" in task_name:
        # Physics: Reaching is vertical/diagonal. Pushing is purely horizontal.
        # We track the Z-axis to find when the arm hits the table/plate level.
        z_pos = eef_pos[:, 2]
        
        # Find the absolute lowest point the arm reaches
        min_z = np.min(z_pos)
        
        # The 'Contact' point is the first frame the arm gets within 5mm of this floor
        contact_indices = np.where(z_pos < min_z + 0.005)[0]
        
        if len(contact_indices) > 0:
            contact_idx = contact_indices[0]
            # Add a tiny 5-frame grace period for the human to stabilize before pushing
            split_idx = min(contact_idx + 5, len(actions) - 1)
        else:
            split_idx = len(actions) // 2 # Absolute fallback
            
        boundaries = sorted(list(set([0, split_idx, len(actions)])))
        return boundaries, ["0_Reach", "1_Push"]

    elif "turn_on" in task_name:
        # Physics: Reaching involves large 3D translations. Turning a knob is almost 
        # purely rotational (wrist movement), meaning the XYZ translation drops to near zero.
        # Therefore, the very end of the trajectory IS the exact location of the knob.
        
        # 1. Find the knob's exact XYZ coordinate (average of the last 10 frames)
        knob_pos = np.mean(eef_pos[-10:, :3], axis=0)
        
        # 2. Calculate how far the arm is from the knob at every frame
        dist_to_knob = np.linalg.norm(eef_pos[:, :3] - knob_pos, axis=1)
        
        # 3. Calculate 3D velocity to ensure it actually stopped to grab the knob
        vel_3d = np.linalg.norm(np.diff(eef_pos[:, :3], axis=0), axis=1)
        vel_3d = np.append(vel_3d, 0) # Pad to match array length
        
        # 4. The split is the first frame the robot is within 1.5cm of the knob AND slows down
        arrival_indices = np.where((dist_to_knob < 0.015) & (vel_3d < 0.005))[0]
        
        if len(arrival_indices) > 0:
            split_idx = arrival_indices[0]
        else:
            # Fallback: Just take the last 20 frames as the "Turn" if the math misses
            split_idx = max(0, len(actions) - 20)
            
        boundaries = sorted(list(set([0, split_idx, len(actions)])))
        return boundaries, ["0_Reach", "1_Turn"]
   
    # ==========================================
    # STRATEGY 2: NON-PREHENSILE (Drawers ONLY)
    # ==========================================
    if "open" in task_name:
        # --- Kinematics via 3D Velocity & Strict Plunge Rejection ---

        # phase 1: hook and pull
        hook_idx, end_pull_idx = state_machine_approach(eef_pos)

        boundaries = sorted(list(set([0, hook_idx, end_pull_idx, len(actions)])))
        return boundaries, ["0_Reach", "1_Pull", "2_Retract"]

    # ==========================================
    # STRATEGY 3: PREHENSILE (Pick and Place ONLY)
    # ==========================================
    gripper_actions = actions[:, gripper_idx]
    binary_gripper = (gripper_actions > threshold).astype(int)
    flips = np.where(np.diff(binary_gripper) != 0)[0]
    
    if len(flips) > 0:
        grasp_idx = flips[0] + 1
        subsequent_flips = flips[flips >= grasp_idx]
        release_idx = subsequent_flips[-1] + 1 if len(subsequent_flips) > 0 else len(actions) - 1
            
        z_trajectory = eef_pos[grasp_idx:release_idx, 2]
        apex_idx = grasp_idx + np.argmax(z_trajectory) if len(z_trajectory) > 0 else grasp_idx
        
        # --- NEW: Split Reach from Pre-Grasp ---
        # vel_3d = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
        # smooth_v = np.convolve(vel_3d, np.ones(5)/5, mode='valid')
        
        # reach_end_idx = 0
        # if grasp_idx > 10:
        #     pre_grasp_vel = smooth_v[:grasp_idx - 2]
        #     if len(pre_grasp_vel) > 0:
        #         reach_end_idx = np.argmax(pre_grasp_vel) + 2
        #     else:
        #         reach_end_idx = grasp_idx // 2
        # else:
        #     reach_end_idx = grasp_idx
            
        boundaries = sorted(list(set([0, apex_idx, release_idx, len(actions)])))
        return boundaries, ["0_Pick", "1_Place", "2_Retract"]
        
    return [0, len(actions)], ["Unknown_Continuous"]

def state_machine_approach(eef_pos):
    # --- PHASE 1: The Drawer (Kinematics via 3D Velocity & Sustained Stops) ---
    xy_pos = eef_pos[:, :2]
    start_pos = xy_pos[0]
    
    vel_3d = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
    smooth_v = np.convolve(vel_3d, np.ones(5)/5, mode='valid') 
    
    vel_xy = np.linalg.norm(np.diff(xy_pos, axis=0), axis=1)
    vel_z = np.abs(np.diff(eef_pos[:, 2], axis=0))
    smooth_vxy = np.convolve(vel_xy, np.ones(5)/5, mode='valid')
    smooth_vz = np.convolve(vel_z, np.ones(5)/5, mode='valid')
    
    hook_idx = 0
    end_pull_idx = 0
    state = 0 
    max_dist = 0
    hook_candidate = 0
    
    # TRACKERS FOR THE JITTER FIX
    state_2_start_idx = 0
    stop_counter = 0
    
    for i, v in enumerate(smooth_v):
        frame_idx = i + 2 
        
        if state == 0:
            dist = np.linalg.norm(xy_pos[frame_idx] - start_pos)
            if dist > max_dist:
                max_dist = dist
                hook_candidate = frame_idx

            if dist > 0.10 and v < 0.002:
                hook_idx = frame_idx
                state = 1
            elif max_dist > 0.10 and (max_dist - dist) > 0.015:
                hook_idx = hook_candidate
                state = 2
                state_2_start_idx = frame_idx # Record when the pull started
                
        elif state == 1:
            if v > 0.003:
                state = 2
                state_2_start_idx = frame_idx # Record when the pull started
                
        elif state == 2:
            frames_in_state_2 = frame_idx - state_2_start_idx
            
            # 1. REFINED REJECTION CHECK (Plunges vs Sloppy Pulls)
            # Only check during the first 15 frames. If we've been pulling longer, it's real.
            if frames_in_state_2 < 15:
                # Vertical speed must be > 4mm AND it must dominate the horizontal speed
                if smooth_vz[i] > 0.004 and smooth_vz[i] > (smooth_vxy[i] * 1.5):
                    state = 0 
                    hook_idx = 0
                    max_dist = np.linalg.norm(xy_pos[frame_idx] - start_pos)
                    hook_candidate = frame_idx
                    continue
                
            # 2. THE CUSHION: Ignore all speed drops for the first 15 frames of the pull
            if frames_in_state_2 < 15:
                continue
                
            # 3. THE SUSTAINED STOP: Must be stopped for 5 consecutive frames
            if v < 0.002:
                stop_counter += 1
                if stop_counter >= 5:
                    end_pull_idx = frame_idx - 5 # Rewind to the exact frame the stop began
                    break
            else:
                stop_counter = 0 # Reset counter if they twitch again

    # Fallbacks
    if hook_idx == 0:
        distances = np.linalg.norm(xy_pos - start_pos, axis=1)
        hook_idx = np.argmax(distances) 
        
    if end_pull_idx == 0:
        end_pull_idx = hook_idx + 25 

    return hook_idx, end_pull_idx

def save_video(images, filepath, fps=20):
    if images.dtype != np.uint8:
        images = (images * 255).astype(np.uint8)
    imageio.mimwrite(filepath, images, fps=fps, macro_block_size=1)

def visualize_semantic_splits(hdf5_path, output_dir):
    dataset_name = os.path.basename(hdf5_path).replace('.hdf5', '')
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🔍 Analyzing Dataset: {dataset_name}")
    
    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())
        
        for i in range(len(demos)):
            demo_key = demos[i]
            demo_group = f['data'][demo_key]
            
            actions = demo_group['actions'][:]
            
            # --- EXTRACT END-EFFECTOR POSITION ---
            # This is standard across all LIBERO tasks
            # print(demo_group.keys(), demo_group['obs'].keys())
            eef_pos = demo_group['obs']['ee_pos'][:]
            
            obs_keys = list(demo_group['obs'].keys())
            # possible_keys = ['agentview_rgb', 'agentview_image', 'robot0_agentview_left_image']
            # image_key = next((k for k in possible_keys if k in obs_keys), None)
            image_key = 'agentview_rgb'
                    
            if image_key is None:
                raise KeyError(f"Could not find camera! Keys: {obs_keys}")
                
            images = demo_group['obs'][image_key][:]
            images = images[:, ::-1, :, :]
            if images.shape[1] == 3:
                images = np.transpose(images, (0, 2, 3, 1))

            save_video(images, os.path.join(save_dir, f"{demo_key}_full.mp4"))
            split_indices, skill_names = find_task_aware_boundaries(actions, eef_pos, dataset_name)
            
            print(f"  > {demo_key}: Split at frames {split_indices}")

            print("#"*150)
            
            for chunk_idx in range(len(split_indices) - 1):
                start_idx = split_indices[chunk_idx]
                end_idx = split_indices[chunk_idx + 1]
                
                chunk_images = images[start_idx:end_idx]
                
                if len(chunk_images) > 0:
                    skill_label = skill_names[chunk_idx] if chunk_idx < len(skill_names) else f"Extra_{chunk_idx}"
                    chunk_vid_path = os.path.join(save_dir, f"{demo_key}_{skill_label}_(f{start_idx}-{end_idx}).mp4")
                    save_video(chunk_images, chunk_vid_path)

if __name__ == "__main__":
    SUITE_NAME = "libero_goal"
    DATASET_DIR = f"/mnt/Data/cjimenez/LIBERO/libero/datasets/{SUITE_NAME}_no_noops_hdf5"
    OUTPUT_FOLDER = f"./semantic_skill_visualizations/{SUITE_NAME}"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)   

    # Just test it on the first 2 files so you don't wait forever while debugging!
    for path in list(glob(f'{DATASET_DIR}/*.hdf5'))[:]:
        if "drawer" not in path:
            continue
        print(f"Processing {path}...")
        visualize_semantic_splits(path, OUTPUT_FOLDER)