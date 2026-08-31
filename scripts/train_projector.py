import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent HF tokenizer warnings after DataLoader fork
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #

import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import tqdm
import numpy as np
import wandb
import gc

# --- Imports from your codebase ---
# from transformers import AutoModelForVision2Seq, AutoProcessor

from src.projectors import ProbabilisticActionProjector, MLPActionProjector, KLTransformerProjector, FlowTransformerProjector
from utils.data import get_vla_projector_dataloader_cached, log_projector_video_probe, log_gt_video_probe, _make_openvla_emb_fn, _make_pi0_emb_fn
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader
from utils.losses import KLDistillationLoss, ClosedFormW2Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE
from src.disentanglers.AdvancedTextActionCVAE import TCNTextCondPriorCVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_recon_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """MSE for continuous dims 0-5; sigmoid-BCE for gripper dim 6.
    Matches the VAE training objective so the floor is truly zero.
    Returns a scalar mean loss."""
    cont_loss = F.mse_loss(pred[..., :6], gt[..., :6])
    gt_bin    = (gt[..., 6] > 0.0).float()
    pred_prob = torch.clamp((pred[..., 6] + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
    grip_loss = F.binary_cross_entropy(pred_prob, gt_bin)
    return cont_loss + 0.5 * grip_loss

def seed_everything(seed: int):
    """Locks down all random number generators for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Global Seed set to {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Robust Projector Training Script")
    
    # Model Types
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "prob"])
    parser.add_argument("--projector_arch", type=str, default="mlp",
                        choices=["mlp", "bottleneck", "linear", "transformer", "flow_transformer"],
                        help="Internal architecture of the projector. "
                             "'mlp' (default): 2-layer funnel (4096→512→256→64). "
                             "'bottleneck': 1-layer compression (4096→128→64). "
                             "'linear': direct linear map, best extrapolation to unseen task clusters. "
                             "'transformer': cross-attention with per-dimension latent queries; "
                             "'flow_transformer': Flow Matching objective transformer;"
                             "uses TransformerActionProjector (d_model=256, heads=8, layers=3).")
    # TransformerActionProjector hyperparameters (only used when --projector_arch transformer)
    parser.add_argument("--xfmr_d_model",   type=int,   default=256,  help="Transformer hidden dim (transformer arch only).")
    parser.add_argument("--xfmr_num_heads",  type=int,   default=8,   help="Number of attention heads (transformer arch only).")
    parser.add_argument("--xfmr_num_layers", type=int,   default=3,   help="Number of cross-attention blocks (transformer arch only).")
    parser.add_argument("--xfmr_ffn_dim",    type=int,   default=512, help="FFN inner dim (transformer arch only).")
    
    parser.add_argument("--vae_type", type=str, default="text_cond_beta_tcvae",
                        choices=["text_cond_beta_tcvae", "text_cvae", "cond_prior"])

    parser.add_argument("--use_state_cond", action="store_true", default=False,
                        help="Condition the VAE prior and decoder on robot proprioceptive state (position, quat, gripper).")
    # Loss Types
    parser.add_argument("--loss", type=str, default="kl", choices=["mse", "nll", "w2", "kl", "flow"], help="Loss function to use. 'kl' (recommended) distills the full teacher posterior (mu+logvar). 'nll' only fits mu. 'w2' collapses logvar to zero. 'flow' trains a Continuous Normalizing Flow via CFM.")

    # Checkpoints
    parser.add_argument("--vla_type", type=str, default="openvla", choices=["openvla", "smolvla", "pi0"],
                        help="Which VLA the embedding cache was built from. Controls cache and save paths.")
    parser.add_argument("--smolvla_model", type=str, default="lerobot/smolvla_base",
                        help="SmolVLA checkpoint (HuggingFace or local). Used only if video probes are added for smolvla.")
    parser.add_argument("--pi0_model", type=str, default="lerobot/pi0",
                        help="Pi0 checkpoint (HuggingFace or local). Used only if video probes are added for pi0.")
    parser.add_argument("--vla_checkpoint", type=str, default="openvla/openvla-7b")
    parser.add_argument("--vla_out_dim", type=int, default=4096,
                        help="VLA last-hidden-state dimension. Used to avoid loading the VLA "
                             "when the embedding cache already exists (saves ~14 GB VRAM at startup).")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to your trained VAE .pt file")
    parser.add_argument("--vae_z_dim", type=int, default=64,
                        help="Latent dimension of the VAE checkpoint. Sets Z_DIM for projector output and cache lookup.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/projectors/")
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=256)  # TensorDataset — no VLA bottleneck
    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--lr_decay_steps", type=int, default=None,
                        help="Number of steps to decay learning rate over. Defaults to max_steps if not provided.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_eta_min", type=float, default=1e-6,
                        help="Minimum learning rate to decay to.")
    parser.add_argument("--patience", type=int, default=15,
                        help="If >0, enables early stopping with this patience window in terms of validation evaluations.")
    parser.add_argument("--eval_every", type=int, default=10000,
                        help="Number of training steps between validation evaluations.")
    parser.add_argument("--auto_resume", type=int, default=1, help="If 1, auto-resume from latest checkpoint for this config")
    parser.add_argument("--resume_step", type=int, default=0, help="Manual override to resume from specific step")
    parser.add_argument("--suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_long", "libero_all"],
                        help="'libero_all' trains on spatial+object+goal simultaneously (best coverage).")
    parser.add_argument("--train_split_ratio", type=int, default=None,
                        help="Tasks (out of 10) to use for training. "
                             "None (default) = Protocol A: all tasks, no held-out test split — standard for LIBERO papers. "
                             "7 = Protocol B: 7 train / 3 held-out test tasks (OOD experiments, e.g. LIBERO-Object).")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (set >1 only if batch_size is memory-constrained)")
    parser.add_argument("--embed_batch_size", type=int, default=4, help="Batch size for one-time VLA embedding pass (ignored if cache exists)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    # Regularisation — critical for few-task generalisation (7 train / 3 test in LIBERO)
    parser.add_argument("--emb_noise_std", type=float, default=0.01,
                        help="Std of Gaussian noise added to VLA embeddings during training. "
                             "Apply AFTER normalisation. 0.01 is a safe default.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate inside projector feature_net (0.3 recommended for <=10 tasks).")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW weight decay.")
    parser.add_argument("--normalize_emb", type=lambda x: str(x).lower() not in ('false', '0', 'no'),
                        default=True,
                        help="L2-normalise VLA embeddings to unit sphere before projection. "
                             "Puts all task embeddings on the same manifold (same effect as "
                             "CLIP's normalisation for zero-shot transfer). Strongly recommended.")
    parser.add_argument("--use_vision_pool", action="store_true", default=False,
                        help="OpenVLA only: use mean-pooled vision patch tokens instead of the "
                             "last LM text token as the VLA embedding. Vision tokens are more "
                             "grounded in the observed scene; may improve action_recon_loss.")
    parser.add_argument("--action_recon_weight", type=float, default=0.5,
                        help="Weight for auxiliary action-reconstruction loss: "
                             "MSE(vae.decode(pred_mu, text_emb), gt_actions). "
                             "Gives direct action-space gradient signal. Start with 0.1.")
    parser.add_argument("--text_backbone", type=str, default="smollm",
                        choices=["clip", "smollm", "openvla_llama"],
                        help="Which language model tokenizer was used for the baseline VAE.")
    parser.add_argument("--vla_layer_idx", type=int, default=-1, 
                        help="Which OpenVLA layer to extract (e.g. 16 for intermediate, -1 for last)")
    parser.add_argument("--num_fusion_layers", type=int, default=1, help="Number of intermediate layers to extract. Overrides vla_layer_idx if > 1.")
    parser.add_argument("--ortho_weight", type=float, default=0.0,
                        help="Weight for the orthogonality regularizer on the Flow Matcher's latent queries. Set to >0 (e.g. 0.01) to enable.")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="DisentangledVLA")
    
    return parser.parse_args()

def get_vla_embedding(model, processor, images, instructions, use_vision_pool=False, vla_layer_idx=-1):
    prompts = [f"In: {instr}\nOut: " for instr in instructions]
    
    inputs = processor(text=prompts, images=images, padding=True, truncation=True, return_tensors="pt").to(model.device)
    if hasattr(inputs, "pixel_values"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    hs = outputs.hidden_states[vla_layer_idx]

    if use_vision_pool:
        # Mean-pool over vision patch tokens (image tokens precede text tokens in OpenVLA).
        # Vision tokens are more grounded in the observed scene than the final LM text token,
        # which is optimized for discrete action-token prediction, not continuous z prediction.
        # Identify image token positions using the image_token_id from the processor/model config.
        img_token_id = getattr(model.config, "image_token_index",
                       getattr(model.config, "image_token_id", None))
        if img_token_id is not None:
            input_ids = inputs["input_ids"]  # (B, L)
            img_mask = (input_ids == img_token_id).unsqueeze(-1).float()  # (B, L, 1)
            vision_tokens = hs * img_mask                  # (B, L, D)
            n_img = img_mask.sum(dim=1).clamp(min=1)                      # (B, 1)
            embedding = vision_tokens.sum(dim=1) / n_img                  # (B, D)
        else:
            # Fallback: pool over all tokens
            embedding = hs.mean(dim=1)
    else:
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        embedding = hs[torch.arange(len(prompts)), last_token_indices]
    
    return embedding.float()

def load_openvla_model(model_path):
    print(f"🏗️  Loading OpenVLA from {model_path} in bfloat16...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    print("✅ OpenVLA Loaded & Frozen.")
    return model, processor

CHUNK_SIZE = 8
ACTION_DIM = 7
N_BLOCKS = max(3, (CHUNK_SIZE - 1).bit_length())  # RF covers chunk_size
VAE_DROPOUT = 0.15

# Probe tasks (must exist in LIBERO HDF5 dir)
PROBE_VAL_TASK   = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo"
PROBE_TRAIN_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"
PROBE_EXTRA_TASK_1 = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo"
PROBE_EXTRA_TASK_2 = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo"

@torch.no_grad()
def sample_flow(projector, vla_embedding, z_dim, num_steps=10):
    B = vla_embedding.size(0)
    z_t = torch.randn(B, z_dim, device=vla_embedding.device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.ones(B, device=vla_embedding.device) * (i / num_steps)
        v = projector(vla_embedding, z_t, t)
        z_t = z_t + v * dt
    return z_t

def train_projector():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Dynamically assign probe tasks for the active suite
    if args.suite == "libero_object":
        PROBE_VAL_TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo"
        PROBE_TRAIN_TASK = "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo"
        PROBE_EXTRA_TASK_1 = "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo"
        PROBE_EXTRA_TASK_2 = "pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo"
    elif args.suite == "libero_goal":
        PROBE_VAL_TASK = "open_the_middle_drawer_of_the_cabinet_demo"
        PROBE_TRAIN_TASK = "put_the_bowl_on_the_plate_demo"
        PROBE_EXTRA_TASK_1 = "push_the_plate_to_the_front_of_the_stove_demo"
        PROBE_EXTRA_TASK_2 = "put_the_cream_cheese_in_the_bowl_demo"
    else:
        PROBE_VAL_TASK = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo"
        PROBE_TRAIN_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"
        PROBE_EXTRA_TASK_1 = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo"
        PROBE_EXTRA_TASK_2 = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo"

    text_embed_dim_dict = {
        'smollm': 960,
        'openvla_llama': 4096,
        'clip': 512,
    }

    text_emb_dim = text_embed_dim_dict.get(args.text_backbone, 512)


    Z_DIM = args.vae_z_dim
    _vae_arch = {"text_cond_beta_tcvae": "tcn", "text_cvae": "cvae", "cond_prior": "cond_prior"}.get(args.vae_type, "")
    
    # Dynamically extract beta from the checkpoint name so caches are named correctly!
    import re
    _beta_match = re.search(r'_beta([0-9.\-]+)_', args.vae_checkpoint)
    BETA = _beta_match.group(1) if _beta_match else "0.1"

    # State-conditioned VAEs require live robot proprioception (not present in offline embedding cache).
    if args.action_recon_weight > 0.0 and args.use_state_cond:
        print("⚠️  --action_recon_weight > 0 is not supported with state-conditioned VAEs (offline cache stores pure latents). Forcing action_recon_weight=0.0.")
        args.action_recon_weight = 0.0

    # Determine protocol
    _protocol_tag = "protA" if args.train_split_ratio is None else f"protB_split{args.train_split_ratio}"

    _layer_tag = f"_layer{args.vla_layer_idx}" if args.vla_layer_idx != -1 else ""
    run_name = f"{args.vla_type}_{args.suite}_{args.projector_arch}_fusion{args.num_fusion_layers}_loss_{args.loss}_arw{args.action_recon_weight}_ortho{args.ortho_weight}_z{Z_DIM}_{_protocol_tag}{_layer_tag}"

    if args.use_vision_pool:
        run_name += "_vpool"

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.vla_type == "openvla":
        args.max_steps *= 2.5
        if args.lr_decay_steps is not None:
            args.lr_decay_steps *= 2.5
    elif args.vla_type == "smolvla":
        args.max_steps *= 2.5
        if args.lr_decay_steps is not None:
            args.lr_decay_steps *= 2.5
        if args.vla_layer_idx != -1:
            print(f"⚠️  WARNING: --vla_layer_idx {args.vla_layer_idx} passed, but SmolVLA only supports final layer extraction. Forcing --vla_layer_idx=-1.")
            args.vla_layer_idx = -1

    # --- 1. LOAD MODELS ---
    print(f"Loading Frozen Teacher VAE ({args.vae_type})...")
    if args.vae_type == "text_cond_beta_tcvae":
        vae = TCNTextActionBetaTCVAE(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            latent_dim=Z_DIM,
            text_emb_dim=text_emb_dim,
            beta=BETA,
            dropout=VAE_DROPOUT,
            hidden_channels=64,
            n_blocks=N_BLOCKS,
        ).to(DEVICE)
    elif args.vae_type == "text_cvae":
        vae = TCNTextActionCVAE(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            latent_dim=Z_DIM,
            text_emb_dim=text_emb_dim,
            beta=BETA,
            dropout=VAE_DROPOUT,
            hidden_channels=64,
            n_blocks=N_BLOCKS,
            enc_text_gate_init=0.0
        ).to(DEVICE)
    elif args.vae_type == "cond_prior":
        vae = TCNTextCondPriorCVAE(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            latent_dim=Z_DIM,
            text_emb_dim=text_emb_dim,
            beta=BETA,
            dropout=VAE_DROPOUT,
            hidden_channels=64,
            n_blocks=N_BLOCKS,
            use_state=args.use_state_cond,
        ).to(DEVICE)

    ckpt_data = torch.load(args.vae_checkpoint, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
        vae.load_state_dict(ckpt_data["model_state_dict"])
    else:
        vae.load_state_dict(ckpt_data)
    vae.eval() 
    for param in vae.parameters(): param.requires_grad = False

    # Only load OpenVLA if the embedding cache doesn't exist yet.
    # Once the cache is on disk, VLA is only needed for video probes (lazy-loaded there).
    _ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal"]
    _cache_root = f"./checkpoints/projectors/{args.vla_type}"
    # Octo and SmolVLA caches include VAE-arch/beta/z in the filename to distinguish
    # caches built for different VAE types without re-running the expensive VLM pass.
    _vae_arch = {"text_cond_beta_tcvae": "tcn", "text_cvae": "cvae", "cond_prior": "cond_prior"}.get(args.vae_type, "")
    _prefix = "pi0_emb_cache" if args.vla_type == "pi0" else "vla_emb_cache"
    _cache_name = f"{_prefix}_{args.vae_type}_arch_{_vae_arch}_beta{BETA}_z{Z_DIM}"
    if args.text_backbone != "clip": _cache_name += f"_text_{args.text_backbone}"
    _vpool_suffix = "_vpool" if getattr(args, "use_vision_pool", False) else ""
    _cache_name += _vpool_suffix
    import re
    _vae_seed_match = re.search(r'_seed_(\d+)', args.vae_checkpoint)
    _vae_seed = int(_vae_seed_match.group(1)) if _vae_seed_match else 1
    _cache_name += f"_seed{_vae_seed}"
    if args.num_fusion_layers > 1:
        _cache_name += f"_fusion{args.num_fusion_layers}"
    elif args.vla_layer_idx != -1: 
        _cache_name += f"_layer{args.vla_layer_idx}"
    _cache_name += ".pt"

    # For OpenVLA: if the target cache doesn't exist, look for any other VAE-type cache
    # that has saved actions. If found, we can reteach teacher targets (~1 min VAE encode)
    # instead of re-running the full 14 GB VLA embedding extraction pass.
    # Same logic applies for smolvla.
    _ALT_VAE_TYPES = {"text_cond_beta_tcvae": "text_cvae", "text_cvae": "text_cond_beta_tcvae"}
    _alt_cache_name = f"vla_emb_cache_{_ALT_VAE_TYPES.get(args.vae_type, '')}{_vpool_suffix}.pt" \
        if args.vla_type == "openvla" else None

    def _find_fallback(suite_name):
        """Return path to an alternate cache with saved actions, or None.
        For openvla: checks the known alternate VAE-type cache name."""
        suite_dirs = [f"{_cache_root}/{suite_name}", f"{_cache_root}/{suite_name}_no_noops"]
        target = _cache_name
        layer_suffix = f"_layer{args.vla_layer_idx}" if args.vla_layer_idx != -1 else ""
        seed_suffix = f"_seed{_vae_seed}"
        for suite_dir in suite_dirs:
            if not os.path.isdir(suite_dir):
                continue
            candidates = [
                os.path.join(suite_dir, f)
                for f in os.listdir(suite_dir)
                if (f.startswith("vla_emb_cache_") or f.startswith("pi0_emb_cache_")) 
                and (f.endswith(f"{layer_suffix}.pt") or "fusion" in f) and f != target
                and (args.vla_layer_idx != -1 or "_layer" not in f)
            ]
            for p in candidates:
                try:
                    c = torch.load(p, map_location="cpu", weights_only=False)
                    if "train_actions" in c or "train_emb" in c:
                        # Enforce Protocol isolation (don't mix protA and protB caches!)
                        has_test_split = len(c.get("test_emb", [])) > 0
                        if args.train_split_ratio is None and has_test_split:
                            continue # Skip protB cache when running protA
                        if args.train_split_ratio is not None and not has_test_split:
                            continue # Skip protA cache when running protB
                            
                        return p
                except Exception:
                    continue
        return None

    if args.suite == "libero_all":
        cache_path   = None  # per-suite paths used in the multi-suite loop below
        cache_exists = all(
            os.path.exists(f"{_cache_root}/{s}/{_cache_name}") or os.path.exists(f"{_cache_root}/{s}_no_noops/{_cache_name}")
            for s in _ALL_SUITES
        )
        _fallback_paths = {s: _find_fallback(s) for s in _ALL_SUITES} if not cache_exists else {}
        fallback_exists = all(_fallback_paths.get(s) for s in _ALL_SUITES)
    else:
        cache_path   = f"{_cache_root}/{args.suite}/{_cache_name}"
        if not os.path.exists(cache_path) and os.path.exists(f"{_cache_root}/{args.suite}_no_noops/{_cache_name}"):
            cache_path = f"{_cache_root}/{args.suite}_no_noops/{_cache_name}"
        cache_exists = os.path.exists(cache_path)
        _fallback_path = _find_fallback(args.suite) if not cache_exists else None
        fallback_exists = _fallback_path is not None

    if cache_exists or fallback_exists:
        if fallback_exists and not cache_exists:
            _fb = _fallback_path if args.suite != "libero_all" else next(iter(_fallback_paths.values()))
            print(f"⚡ Fallback cache found ({_fb}) — will reteach teacher z only, skipping VLA extraction.")
        else:
            print(f"⚡ Embedding cache found at {cache_path} — skipping {args.vla_type.upper()} load at startup.")
        vla_model, processor = None, None
        _probe_path = cache_path if (cache_path and os.path.exists(str(cache_path))) else \
            (_fallback_path if args.suite != "libero_all" else _fallback_paths.get("libero_spatial")) or \
            f"{_cache_root}/libero_spatial/{_cache_name}"
        DEFAULT_VLA_DIMS = {"openvla": 4096, "pi0": 2048, "octo": 768, "smolvla": 960}
        default_dim = DEFAULT_VLA_DIMS.get(args.vla_type, args.vla_out_dim)
        if _probe_path and os.path.exists(_probe_path):
            _probe = torch.load(_probe_path, map_location="cpu", weights_only=False)
            if "train_emb" in _probe:
                VLA_OUT_DIM = int(_probe["train_emb"].shape[-1])
            elif "octo_dim" in _probe:
                VLA_OUT_DIM = int(_probe["octo_dim"])
            else:
                first_k = next(iter(_probe.keys()), None)
                if first_k and isinstance(_probe[first_k], dict) and "train_emb" in _probe[first_k]:
                    VLA_OUT_DIM = int(_probe[first_k]["train_emb"].shape[-1])
                else:
                    VLA_OUT_DIM = default_dim
            del _probe
        else:
            VLA_OUT_DIM = default_dim
    else:
        if args.vla_type != "openvla":
            raise ValueError(f"Target embedding cache not found at {cache_path} and no fallback cache was found.\n"
                             f"For vla_type='{args.vla_type}', you MUST pre-build the cache using scripts/build_{args.vla_type}_cache.py.\n"
                             f"train_projector.py only supports on-the-fly cache building for OpenVLA.")
                             
        vla_model, processor = load_openvla_model(args.vla_checkpoint)
        if hasattr(vla_model.config, "hidden_size"):
            VLA_OUT_DIM = vla_model.config.hidden_size
        elif hasattr(vla_model.config, "text_config"):
            VLA_OUT_DIM = vla_model.config.text_config.hidden_size
        else:
            VLA_OUT_DIM = DEFAULT_VLA_DIMS.get(args.vla_type, args.vla_out_dim)
    print(f"📐 VLA Embedding Dimension: {VLA_OUT_DIM}")

    print(f"Initializing Student Projector ({args.projector_type})...")
    if args.projector_type == "mlp":
        projector = MLPActionProjector(input_dim=VLA_OUT_DIM, latent_dim=Z_DIM).to(DEVICE)
        criterion = nn.MSELoss()
    elif args.projector_type == "prob":
        if args.projector_arch == "transformer":
            projector = KLTransformerProjector(
                input_dim=VLA_OUT_DIM,
                latent_dim=Z_DIM,
                d_model=args.xfmr_d_model,
                num_heads=args.xfmr_num_heads,
                num_layers=args.xfmr_num_layers,
                ffn_dim=args.xfmr_ffn_dim,
                dropout=args.dropout,
            ).to(DEVICE)
        elif args.projector_arch == "flow_transformer":
            projector = FlowTransformerProjector(
                input_dim=VLA_OUT_DIM,
                latent_dim=Z_DIM,
                d_model=args.xfmr_d_model,
                num_heads=args.xfmr_num_heads,
                num_layers=args.xfmr_num_layers,
                ffn_dim=args.xfmr_ffn_dim,
                dropout=args.dropout,
            ).to(DEVICE)
        else:
            projector = ProbabilisticActionProjector(
                input_dim=VLA_OUT_DIM, latent_dim=Z_DIM, dropout=args.dropout,
                architecture=args.projector_arch,
            ).to(DEVICE)
        if args.loss == "kl":
            kl_criterion = KLDistillationLoss()
        elif args.loss == "w2":
            w2_criterion = ClosedFormW2Loss().to(DEVICE)

    optimizer = optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else args.max_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=args.lr_eta_min, last_epoch=-1)

    # ---------------------------------------------------------
    # Checkpoint Auto-Resume Logic
    # ---------------------------------------------------------
    base_dir = f"{args.save_dir}/{args.vla_type}/{args.suite}/chunk_{CHUNK_SIZE}_zdim_{Z_DIM}"
    ckpt_prefix = f"{args.projector_type}_{args.projector_arch}_fusion{args.num_fusion_layers}_ortho{args.ortho_weight}_loss_{args.loss}_seed_{args.seed}"
    if args.auto_resume:
        if os.path.exists(base_dir):
            prefix = f"{ckpt_prefix}_step_"
            ckpts = [f for f in os.listdir(base_dir) if f.startswith(prefix) and f.endswith(".pt")]
            if ckpts:
                latest_step = max([int(f.replace(prefix, "").replace(".pt", "")) for f in ckpts])
                args.resume_step = latest_step

    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume_step > 0:
        ckpt = f"{base_dir}/{ckpt_prefix}_step_{args.resume_step}.pt"
        if os.path.exists(ckpt):
            print(f"🔄 Resuming from {ckpt}")
            checkpoint = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            if "model_state_dict" in checkpoint:
                projector.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                args.resume_step = checkpoint["step"]
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                patience_counter = checkpoint.get("patience_counter", 0)
                print(f"✅ Full state restored from step {args.resume_step} (Model + Optimizer + Scheduler) | Best Val Loss: {best_val_loss:.6f}")
            else:
                # Legacy checkpoint that only saved model weights
                projector.load_state_dict(checkpoint)
                scheduler.last_epoch = args.resume_step - 1
                print(f"⚠️ Legacy checkpoint loaded. Optimizer fresh, step {args.resume_step}.")
        else:
            print(f"❌ Checkpoint not found at {ckpt}, starting from scratch.")
            args.resume_step = 0
    # ---------------------------------------------------------
    
    # Dataloader — pre-compute all VLA embeddings once, then train on pure tensors.
    # This eliminates 7B-model inference from the hot loop (~32h → ~30min for 25k steps).
    if args.suite == "libero_all":
        train_sub, test_sub = [], []
        for _s in _ALL_SUITES:
            _cp = f"{_cache_root}/{_s}/{_cache_name}"
            _fb = _fallback_paths.get(_s) if not cache_exists else None
            _tr_dl, _te_dl = get_vla_projector_dataloader_cached(
                vla_model=vla_model, processor=processor, suite=_s,
                batch_size=args.batch_size, embed_batch_size=args.embed_batch_size,
                cache_path=_cp, fallback_cache_path=_fb, device=DEVICE, vae=vae, vae_type=args.vae_type,
                use_vision_pool=args.use_vision_pool, text_backbone=args.text_backbone,
                train_split_ratio=args.train_split_ratio, vla_layer_idx=args.vla_layer_idx,
                num_fusion_layers=args.num_fusion_layers,
            )
            train_sub.append(_tr_dl.dataset)
            if _te_dl is not None:
                test_sub.append(_te_dl.dataset)
        train_dataloader = TorchDataLoader(ConcatDataset(train_sub), batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, drop_last=True)
        test_dataloader  = TorchDataLoader(ConcatDataset(test_sub), batch_size=args.batch_size,
                                           shuffle=False, num_workers=0, drop_last=False) if test_sub else None
    else:
        train_dataloader, test_dataloader = get_vla_projector_dataloader_cached(
            vla_model=vla_model,
            processor=processor,
            suite=args.suite,
            batch_size=args.batch_size,
            embed_batch_size=args.embed_batch_size,
            cache_path=cache_path,
            fallback_cache_path=_fallback_path if not cache_exists else None,
            device=DEVICE,
            vae=vae,
            vae_type=args.vae_type,
            use_vision_pool=args.use_vision_pool,
            train_split_ratio=args.train_split_ratio,
            vla_layer_idx=args.vla_layer_idx,
            num_fusion_layers=args.num_fusion_layers,
        )



    if test_dataloader is None:
        print(f"🚀 {_protocol_tag}: no held-out test split — eval is simulator-only.")
    else:
        print(f"🔬 {_protocol_tag}: held-out test split active — {len(test_dataloader.dataset)} test samples.")

    # If VLA was loaded for the embedding pass, offload it now to free ~14 GB VRAM.
    # It will be lazy-loaded back to GPU only inside video probe calls.
    if vla_model is not None:
        vla_model.cpu()
        torch.cuda.empty_cache()
        print("📦 VLA offloaded to CPU. VRAM freed for projector training.")

    # stats_path needed for video probes (same path as VAE normalisation stats)
    # For libero_all, probe tasks are from libero_spatial — use its stats.
    _stats_suite = "libero_spatial" if args.suite == "libero_all" else args.suite
    stats_path = f"./checkpoints/text_tcvae/{_stats_suite}/dataset_statistics.pt"

    smolvla_policy = None
    if args.vla_type == "smolvla" and args.use_wandb:
        # SmolVLA is PyTorch — no subprocess needed.
        # Load once onto CPU; moved to GPU only for each video probe interval.
        # lerobot may not be installed in every container (e.g. openvla_worker);
        # training only needs the frozen cache, so probes are silently skipped if
        # the import fails.
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            print(f"🤖 Loading SmolVLA for video probes from {args.smolvla_model} …")
            smolvla_policy = SmolVLAPolicy.from_pretrained(args.smolvla_model).cpu().eval()
            for _p in smolvla_policy.parameters():
                _p.requires_grad_(False)
            print("✅ SmolVLA loaded (on CPU, will move to GPU per probe).")
        except ImportError:
            print("⚠️  lerobot not found — SmolVLA video probes will be skipped (training unaffected).")

    pi0_policy = None
    pi0_tokenizer = None
    if args.vla_type == "pi0" and args.use_wandb:
        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            from transformers import AutoTokenizer
            print(f"🤖 Loading Pi0 for video probes from {args.pi0_model} …")
            pi0_policy = PI0Policy.from_pretrained(args.pi0_model).cpu().eval()
            for _p in pi0_policy.parameters():
                _p.requires_grad_(False)
            pi0_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            print("✅ Pi0 loaded (on CPU, will move to GPU per probe).")
        except Exception as e:
            print(f"⚠️  Failed to load Pi0 policy/tokenizer for video probes: {e} — Pi0 video probes will be skipped.")

    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch

    data_iter = infinite_loader(train_dataloader)

    running_train_action_mses = []

    # Training loop
    print(f"🚀 Starting Training from step {args.resume_step} to {args.max_steps}...")
    steps = args.resume_step

    # Log HDF5 demo replay once at step 0 — perfect reference, same for all vla_types
    if args.use_wandb and args.resume_step == 0:
        log_gt_video_probe(step=0, suite_name=_stats_suite, stats_path=stats_path,
                           device=DEVICE, probe_task_name=PROBE_VAL_TASK, split_name="val")
        log_gt_video_probe(step=0, suite_name=_stats_suite, stats_path=stats_path,
                           device=DEVICE, probe_task_name=PROBE_TRAIN_TASK, split_name="train")
        log_gt_video_probe(step=0, suite_name=_stats_suite, stats_path=stats_path,
                           device=DEVICE, probe_task_name=PROBE_EXTRA_TASK_1, split_name="val_extra1")
        log_gt_video_probe(step=0, suite_name=_stats_suite, stats_path=stats_path,
                           device=DEVICE, probe_task_name=PROBE_EXTRA_TASK_2, split_name="val_extra2")
        gc.collect()
        torch.cuda.empty_cache()

    try:
        with tqdm.tqdm(initial=args.resume_step, total=args.max_steps) as pbar:
            while steps < args.max_steps:

                # --- Gradient Accumulation: accumulate over accum_steps micro-batches ---
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_mse_mu = 0.0
                accum_logvar = 0.0       # tracks teacher target_logvar
                accum_pred_logvar = 0.0  # tracks projector pred_logvar
                accum_cos_sim = 0.0
                accum_recon_loss = 0.0   # action-space reconstruction loss (pred_mu)
                accum_ortho_loss = 0.0

                for _ in range(args.accum_steps):
                    vla_embedding, target_mu, target_logvar, text_emb, gt_actions = next(data_iter)
                    vla_embedding = vla_embedding.to(DEVICE)
                    target_mu     = target_mu.to(DEVICE)
                    target_logvar = target_logvar.to(DEVICE)
                    # text_emb and gt_actions only sent to GPU when recon_weight > 0

                    # Normalise first (fixes inter-task norm differences), then add noise.
                    if args.normalize_emb:
                        vla_embedding = F.normalize(vla_embedding, dim=-1)
                    if args.emb_noise_std > 0.0:
                        vla_embedding = vla_embedding + args.emb_noise_std * torch.randn_like(vla_embedding)

                    # Teacher targets are pre-computed in the cache — no VAE call needed.
                    accum_logvar += target_logvar.mean().item() / args.accum_steps

                    # --- B. PROJECTOR FORWARD + LOSS (scaled for accumulation) ---
                    if args.projector_type == "mlp":
                        pred_z  = projector(vla_embedding)
                        loss    = criterion(pred_z, target_mu) / args.accum_steps
                        z_for_recon = pred_z
                        with torch.no_grad():
                            accum_cos_sim += F.cosine_similarity(pred_z, target_mu, dim=-1).mean().item() / args.accum_steps
                    elif args.projector_type == "prob":
                        if args.loss == "flow":
                            B_batch = target_mu.size(0)
                            t = torch.rand(B_batch, device=DEVICE)
                            z_0 = torch.randn_like(target_mu)
                            z_1 = target_mu
                            # CFM interpolant
                            z_t = t.unsqueeze(1) * z_1 + (1 - t.unsqueeze(1)) * z_0
                            v_target = z_1 - z_0
                            pred_v = projector(vla_embedding, z_t, t)
                            loss = F.mse_loss(pred_v, v_target) / args.accum_steps
                            
                            ortho_loss_val = 0.0
                            if args.ortho_weight > 0.0:
                                o_loss = projector.get_ortho_loss()
                                loss = loss + (args.ortho_weight * o_loss) / args.accum_steps
                                ortho_loss_val = o_loss.item()
                            accum_ortho_loss += ortho_loss_val / args.accum_steps
                            
                            if args.action_recon_weight > 0.0:
                                z_for_recon = sample_flow(projector, vla_embedding, Z_DIM, num_steps=10)
                            else:
                                z_for_recon = z_1 # dummy if not used
                            
                            with torch.no_grad():
                                # Evaluate tracking MSE during training
                                accum_mse_mu += F.mse_loss(sample_flow(projector, vla_embedding, Z_DIM, num_steps=10), target_mu).item() / args.accum_steps
                                accum_pred_logvar += 0.0
                        else:
                            dist, pred_mu, pred_logvar = projector(vla_embedding)
                            if args.loss == "kl":
                                loss = kl_criterion(pred_mu, pred_logvar, target_mu, target_logvar) / args.accum_steps
                            elif args.loss == "w2":
                                loss = w2_criterion(pred_mu, pred_logvar, target_mu) / args.accum_steps
                            elif args.loss == "nll":
                                loss = -dist.log_prob(target_mu).mean() / args.accum_steps
                            z_for_recon = pred_mu
                            with torch.no_grad():
                                accum_mse_mu      += F.mse_loss(pred_mu, target_mu).item() / args.accum_steps
                                accum_pred_logvar += pred_logvar.mean().item()              / args.accum_steps

                    # Auxiliary reconstruction loss: decode pred_mu through frozen VAE decoder
                    # and compare to ground-truth actions. Provides direct action-space gradient.
                    # Uses the same loss as VAE training: MSE on dims 0-5, sigmoid-BCE on gripper.
                    if args.action_recon_weight > 0.0:
                        _text_gpu      = text_emb.to(DEVICE)
                        _gt_actions    = gt_actions.to(DEVICE)
                        pred_actions   = vae.decode(z_for_recon, _text_gpu)
                        recon_loss     = action_recon_loss(pred_actions, _gt_actions)
                        loss           = loss + args.action_recon_weight * recon_loss / args.accum_steps
                        accum_recon_loss += recon_loss.item() / args.accum_steps

                    loss.backward()
                    accum_loss += loss.item()

                # --- D. OPTIMIZER STEP (once per effective batch) ---
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                optimizer.step()
                if steps < decay_steps:
                    scheduler.step()
                steps += 1
                if args.action_recon_weight > 0.0:
                    running_train_action_mses.append(accum_recon_loss)
                    running_train_action_mses = running_train_action_mses[-1000:]
                current_lr = optimizer.param_groups[0]["lr"]

                # --- E. TRAINING LOGGING ---
                if args.projector_type == "mlp":
                    pbar.set_description(f"Loss(MSE): {accum_loss:.4f} | CosSim: {accum_cos_sim:.3f}")
                    if args.use_wandb:
                        wandb.log({"train/loss_total": accum_loss, "train/cos_sim": accum_cos_sim,
                                "train/learning_rate": current_lr, "global_step": steps}, step=steps)
                elif args.projector_type == "prob":
                    pbar.set_description(f"Loss({args.loss.upper()}): {accum_loss:.4f} | MSE(μ): {accum_mse_mu:.4f} | pred_lv: {accum_pred_logvar:.2f} | tgt_lv: {accum_logvar:.2f}")
                    if args.use_wandb:
                        log_entry = {"train/loss_total": accum_loss, "train/mse_mu": accum_mse_mu,
                                "train/pred_logvar": accum_pred_logvar,
                                "train/teacher_logvar": accum_logvar,
                                "train/learning_rate": current_lr,
                                "global_step": steps}
                        if args.action_recon_weight > 0.0:
                            log_entry["train/action_recon_loss"] = accum_recon_loss
                        if args.ortho_weight > 0.0:
                            log_entry["train/ortho_loss"] = accum_ortho_loss
                        wandb.log(log_entry, step=steps)

                pbar.update(1)
                if steps % 1000 == 0: gc.collect()

                # --- F. EVAL LOOP + VIDEO PROBE ---
                if steps % args.eval_every == 0:
                    projector.eval()

                    early_stop = False
                    val_loss = None
                    if test_dataloader is not None:
                        test_losses, test_mse_mus, test_logvars, test_teacher_logvars = [], [], [], []
                        test_action_mses, test_z_mean_norms = [], []

                        with torch.no_grad():
                            for t_vla_emb, t_mu, t_logvar, t_text_emb, _t_actions in test_dataloader:
                                t_vla_emb  = t_vla_emb.to(DEVICE)
                                t_mu       = t_mu.to(DEVICE)
                                t_logvar   = t_logvar.to(DEVICE)
                                t_text_emb = t_text_emb.to(DEVICE)
                                _t_actions = _t_actions.to(DEVICE)
                                if args.normalize_emb:
                                    t_vla_emb = F.normalize(t_vla_emb, dim=-1)

                                if args.projector_type == "mlp":
                                    p_z   = projector(t_vla_emb)
                                    t_loss = F.mse_loss(p_z, t_mu).item()
                                    test_losses.append(t_loss)
                                elif args.projector_type == "prob":
                                    if args.loss == "flow":
                                        p_mu = sample_flow(projector, t_vla_emb, Z_DIM, num_steps=10)
                                        t_loss = F.mse_loss(p_mu, t_mu).item()
                                        p_lv = torch.zeros_like(p_mu)
                                    else:
                                        t_dist, p_mu, p_lv = projector(t_vla_emb)
                                        if args.loss == "kl":
                                            t_loss = kl_criterion(p_mu, p_lv, t_mu, t_logvar).item()
                                        elif args.loss == "nll":
                                            t_loss = -t_dist.log_prob(t_mu).mean().item()
                                        elif args.loss == "w2":
                                            t_loss = w2_criterion(p_mu, p_lv, t_mu).item()
                                    test_losses.append(t_loss)
                                    test_mse_mus.append(F.mse_loss(p_mu, t_mu).item())
                                    test_logvars.append(p_lv.mean().item())
                                    test_teacher_logvars.append(t_logvar.mean().item())

                                    pred_actions_recon    = vae.decode(p_mu, t_text_emb)
                                    test_action_mses.append(action_recon_loss(pred_actions_recon, _t_actions).item())
                                    test_z_mean_norms.append(p_mu.norm(dim=-1).mean().item())

                        val_loss = sum(test_losses) / len(test_losses)

                        if args.use_wandb:
                            log_dict = {"test/loss": val_loss,
                                        "global_step": steps}
                            if test_mse_mus:
                                log_dict["test/mse_mu"]        = sum(test_mse_mus)        / len(test_mse_mus)
                                log_dict["test/pred_logvar"]   = sum(test_logvars)         / len(test_logvars)
                                log_dict["test/teacher_logvar"]= sum(test_teacher_logvars) / len(test_teacher_logvars)
                                log_dict["test/action_mse"]    = sum(test_action_mses)     / len(test_action_mses)
                                log_dict["test/z_pred_norm"]   = sum(test_z_mean_norms)    / len(test_z_mean_norms)
                            wandb.log(log_dict, step=steps)

                    else:
                        # Protocol A: calculate running average of training action_mse
                        if len(running_train_action_mses) > 0:
                            val_loss = sum(running_train_action_mses) / len(running_train_action_mses)
                            print(f"📈 Step {steps} — Running average training action_mse (last 1000 steps): {val_loss:.6f}")
                            if args.use_wandb:
                                wandb.log({"train/running_action_mse": val_loss, "global_step": steps}, step=steps)

                    if val_loss is not None:
                        # Early Stopping & Best Model Saving Logic
                        base_dir = f"{args.save_dir}/{args.vla_type}/{args.suite}/chunk_{CHUNK_SIZE}_zdim_{Z_DIM}"
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_path = f"{base_dir}/{ckpt_prefix}_best.pt"
                            os.makedirs(os.path.dirname(best_path), exist_ok=True)
                            torch.save({
                                "model_state_dict": projector.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "step": steps,
                                "best_val_loss": best_val_loss,
                                "patience_counter": patience_counter
                            }, best_path)
                            print(f"🏆 Step {steps} — New best validation metric ({'test loss' if test_dataloader is not None else 'running train action_mse'}): {best_val_loss:.6f}! Saved best checkpoint.")
                        else:
                            patience_counter += 1
                            print(f"📉 Step {steps} — Validation metric did not improve: {val_loss:.6f} (Best: {best_val_loss:.6f}). Patience: {patience_counter}/{args.patience}")

                        if args.patience > 0 and patience_counter >= args.patience:
                            print(f"🛑 Early stopping triggered! Validation metric did not improve for {args.patience} evaluations.")
                            early_stop = True

                    # Video probes run regardless of protocol (A or B).
                    # Pre-load the VLA model and create the embedding function
                    emb_fn_to_use = None
                    if args.vla_type == "openvla":
                        if vla_model is None:
                            vla_model, processor = load_openvla_model(args.vla_checkpoint)
                        vla_model.to(DEVICE)
                        torch.cuda.empty_cache()
                        emb_fn_to_use = _make_openvla_emb_fn(vla_model, processor, DEVICE, use_vision_pool=args.use_vision_pool, vla_layer_idx=args.vla_layer_idx, num_fusion_layers=args.num_fusion_layers)
                    elif args.vla_type == "smolvla" and smolvla_policy is not None:
                        from utils.data import _make_smolvla_emb_fn
                        smolvla_policy.to(DEVICE)
                        torch.cuda.empty_cache()
                        emb_fn_to_use = _make_smolvla_emb_fn(smolvla_policy, DEVICE, vla_layer_idx=args.vla_layer_idx, num_fusion_layers=args.num_fusion_layers)
                    elif args.vla_type == "pi0" and pi0_policy is not None:
                        pi0_policy.to(DEVICE)
                        torch.cuda.empty_cache()
                        emb_fn_to_use = _make_pi0_emb_fn(pi0_policy, pi0_tokenizer, DEVICE, num_fusion_layers=args.num_fusion_layers)
                    elif args.vla_type == "octo" and octo_emb_fn is not None:
                        emb_fn_to_use = octo_emb_fn

                    if emb_fn_to_use is not None:
                        probe_tasks = [
                            (PROBE_VAL_TASK, "val"),
                            (PROBE_TRAIN_TASK, "train"),
                            (PROBE_EXTRA_TASK_1, "val_extra1"),
                            (PROBE_EXTRA_TASK_2, "val_extra2")
                        ]
                        for task_name, split in probe_tasks:
                            for ex_steps in [CHUNK_SIZE, 1]:
                                log_projector_video_probe(
                                    vae=vae, projector=projector, vla_model=None, processor=None,
                                    step=steps, suite_name=_stats_suite, stats_path=stats_path,
                                    device=DEVICE, probe_task_name=task_name, split_name=split,
                                    chunk_size=CHUNK_SIZE, emb_fn=emb_fn_to_use, normalize_emb=args.normalize_emb,
                                    use_vision_pool=args.use_vision_pool,
                                    text_backbone=args.text_backbone,
                                    vla_layer_idx=args.vla_layer_idx,
                                    num_fusion_layers=args.num_fusion_layers,
                                    exec_steps=ex_steps
                                )

                    # Offload the VLA model back to CPU to save VRAM for training
                    if args.vla_type == "openvla":
                        vla_model.cpu()
                    elif args.vla_type == "smolvla":
                        smolvla_policy.cpu()
                    elif args.vla_type == "pi0" and pi0_policy is not None:
                        pi0_policy.cpu()
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Save checkpoint
                    base_dir = f"{args.save_dir}/{args.vla_type}/{args.suite}/chunk_{CHUNK_SIZE}_zdim_{Z_DIM}"
                    os.makedirs(base_dir, exist_ok=True)
                    torch.save({
                        "model_state_dict": projector.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step": steps,
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter
                    }, f"{base_dir}/{ckpt_prefix}_step_{steps}.pt")

                    projector.train()

                    if early_stop:
                        break

        pbar.close()

    finally:
        pass

    if args.use_wandb: wandb.finish()
    print("🎯 Projector Training Complete!")

if __name__ == "__main__":
    train_projector()