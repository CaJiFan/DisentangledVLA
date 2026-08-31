import torch
import os
import h5py
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from scripts.eval_disentangler_suite import load_model, parse_filename

def main():
    filepath = "checkpoints/new_protocol_cvae/libero_object/rw100_d0.1_beta0.1_z1024_chunk8_protA_cond_prior_sc1.0_htlv2_seed_1_step_1000000.pt"
    cfg = parse_filename(filepath)
    vae = load_model(filepath, cfg)
    vae.eval()
    
    # Load dataset statistics
    stats_path = "./checkpoints/new_protocol_cvae/libero_object/dataset_statistics.pt"
    action_stats = torch.load(stats_path)
    stats = action_stats['libero_object_no_noops']['action']
    action_min = torch.tensor(stats['min']).float().cuda()
    action_max = torch.tensor(stats['max']).float().cuda()
    action_mask = torch.tensor(stats['mask']).float().cuda()
    
    # Load a demo
    hdf5_path = "/mnt/Data/cjimenez/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        demo_id = list(f["data"].keys())[0]
        gt_actions = f[f"data/{demo_id}/actions"][:]
        
    gt_tensor = torch.tensor(gt_actions).float().cuda()
    norm_gt = (gt_tensor - action_min) / (action_max - action_min + 1e-5)
    norm_gt = norm_gt * 2.0 - 1.0
    norm_gt = norm_gt * action_mask + gt_tensor * (1.0 - action_mask)
    
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    
    text_inputs = clip_tok(["pick up the alphabet soup and place it in the basket"], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_emb = clip_enc(**text_inputs).pooler_output
        
    # Take a window where gripper is closed (e.g. middle of demo)
    print("GT Gripper values:")
    print(gt_actions[60:76, 6])
    
    chunk = norm_gt[60:68].unsqueeze(0)
    with torch.no_grad():
        mu, _ = vae.encode(chunk, text_emb)
        pred_chunk = vae.decode(mu, text_emb)[0]
        
    print("\nReconstructed Gripper values (raw model outputs before mapping):")
    print(pred_chunk[:, 6].cpu().numpy())
    
    print("\nReconstructed Gripper mapped values:")
    mapped = (pred_chunk[:, 6] > 0.0).float() * 2.0 - 1.0
    print(mapped.cpu().numpy())

if __name__ == "__main__":
    main()
