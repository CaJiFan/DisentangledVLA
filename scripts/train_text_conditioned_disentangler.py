import os
# Put this before ANY other imports!
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import random
import numpy as np
import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.disentanglers import (ConvTextActionBetaTCVAE, MLPTextActionBetaTCVAE, 
                               TCNTextActionBetaTCVAE, TCNTextActionCVAE,
                               TCNTextCondPriorCVAE, TCNTextWAE)
from utils.data import get_text_action_ram_cached_dataloader, log_video_probe, log_gt_video_probe
from utils.metrics import compute_mig, cyclic_beta_schedule, get_beta_schedule, compute_supcon_loss

# --- CONFIG ---
SAVE_DIR_BASE = "./checkpoints/new_protocol_cvae/"
os.makedirs(SAVE_DIR_BASE, exist_ok=True)

BATCH_SIZE  = 128
LR          = 1e-4
MAX_STEPS   = 250_000
ACTION_DIM  = 7
TEST_STEPS  = 10_000
USE_WANDB   = True
WANDB_PROJECT = "DisentangledVLA"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROBE_VAL_TASK   = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo"
PROBE_TRAIN_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Global Seed set to {seed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Disentangled VAE per LIBERO Suite")
    parser.add_argument("--suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_long", "libero_10"])
    parser.add_argument("--beta",         type=float, default=0.1,  help="KL weight (standard beta-VAE)")
    parser.add_argument("--dropout",      type=float, default=0.15, help="Classifier-free guidance dropout on text")
    parser.add_argument("--latent_dim",   type=int,   default=128,  help="Size of latent bottleneck")
    parser.add_argument("--chunk_size",   type=int,   default=8,    help="Action chunk size")
    parser.add_argument("--recon_weight", type=int,   default=100,  help="Reconstruction loss weight")
    parser.add_argument("--n_cycles",     type=int,   default=4,    help="Number of beta annealing cycles")
    parser.add_argument("--beta_schedule", type=str,  default="warmup",
                        choices=["fixed", "warmup", "high_to_low", "cyclic"],
                        help="Schedule type for beta: 'fixed', 'warmup' (0->beta_max over first N%), "
                             "'high_to_low' (beta_high->beta_max early for initial disentanglement), "
                             "or 'cyclic' (n_cycles periodic cosine curves).")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of max_steps used for warmup/decay phase (default 0.05 = first 5% of training).")
    parser.add_argument("--beta_high",    type=float, default=1.0,
                        help="Starting beta value for 'high_to_low' schedule (default 1.0).")

    parser.add_argument("--use-mlp",  action="store_true", help="Use MLP encoder/decoder")
    parser.add_argument("--use-tcn",  action="store_true", help="Use dilated-TCN decoder-only CVAE")
    parser.add_argument("--use-cvae", action="store_true", help="Use full CVAE (text in encoder+decoder)")
    parser.add_argument("--use-cond-prior", action="store_true", help="Use TCN Decoder-Only with Conditional Prior")
    parser.add_argument("--use-wae", action="store_true", help="Use Wasserstein AE with TCN Decoder-Only")
    parser.add_argument("--enc_text_gate_init", type=float, default=0.0,
                        help="Initial value of encoder text gate (0=fully closed at init).")
    parser.add_argument("--text_backbone", type=str, default="clip",
                        choices=["smollm", "octo_t5", "openvla_llama", "clip"])
    parser.add_argument("--train_split_ratio", type=int, default=None,
                        help="None=Protocol A (all tasks); 7=Protocol B (7 train / 3 held-out test).")
    parser.add_argument("--max_steps", type=int, default=250000, help="Maximum number of training steps")
    parser.add_argument("--val_freq", type=int, default=10000, help="Evaluation and logging frequency in steps")
    parser.add_argument("--patience", type=int, default=-1, help="Early stopping patience (number of eval checks); -1 disables early stopping")
    parser.add_argument("--seed", type=int, default=0)
    # ── Task-discriminative losses ───────────────────────────────────────────
    parser.add_argument("--supcon_weight", type=float, default=0.0,
                        help="Weight for Supervised Contrastive loss on z (0 = disabled). "
                             "Pulls same-task z together, pushes diff-task z apart. "
                             "Compatible with ALL arch flags and cross-attn pooling.")
    parser.add_argument("--supcon_temp", type=float, default=0.07,
                        help="Temperature for SupCon loss (default 0.07).")
    parser.add_argument("--no_text_decoder", action="store_true",
                        help="Zero out text in decode() so z must carry all task info. "
                             "--use-cvae + this flag  = Play-LMP style (text in encoder only). "
                             "--use-cond-prior + this = SPIRL style (text in prior only). "
                             "Compatible with cross-attn pooling.")
    parser.add_argument("--gripper_weight", type=float, default=5.0,
                        help="Weight multiplier for binary cross entropy gripper loss (default 5.0).")
    parser.add_argument("--exec_steps", type=int, default=1,
                        help="Number of steps to execute per chunk in video probes before re-planning (default 1).")
    parser.add_argument("--use_state_cond", action="store_true",
                        help="Enable robot state conditioning on VAE prior and decoder (SPIRL style).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint (.pt file) to resume model weights from.")
    return parser.parse_args()


def train_conditional_disentangler():
    args = parse_args()
    seed_everything(args.seed)

    global PROBE_VAL_TASK, PROBE_TRAIN_TASK
    if args.suite == "libero_spatial":
        PROBE_VAL_TASK   = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo"
        PROBE_TRAIN_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"
    elif args.suite == "libero_object":
        PROBE_VAL_TASK   = "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo"
        PROBE_TRAIN_TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo"
    elif args.suite == "libero_goal":
        PROBE_VAL_TASK   = "open_the_middle_drawer_of_the_cabinet_demo"
        PROBE_TRAIN_TASK = "put_the_bowl_on_the_stove_demo"

    CHUNK_SIZE = args.chunk_size
    MAX_STEPS = args.max_steps
    TEST_STEPS = args.val_freq
    suite_save_dir = os.path.join(SAVE_DIR_BASE, args.suite)
    os.makedirs(suite_save_dir, exist_ok=True)

    text_embed_dim_dict = {'smollm': 960, 'octo_t5': 768, 'openvla_llama': 4096, 'clip': 512}
    text_emb_dim = text_embed_dim_dict.get(args.text_backbone, 512)

    _protocol = "protA" if args.train_split_ratio is None else f"protB_split{args.train_split_ratio}"
    _text_tag = "" if args.text_backbone == "clip" else f"_text_{args.text_backbone}"
    _arch_tag = "_mlp" if args.use_mlp else (
        "_cond_prior" if args.use_cond_prior else (
            "_wae" if args.use_wae else (
                "_dec_only" if args.use_tcn else (
                    "_full_cond" if args.use_cvae else "_conv"
                )
            )
        )
    )
    _cyc_tag  = f"_cyc{args.n_cycles}" if args.n_cycles > 0 else ""
    _supcon_tag = f"_sc{args.supcon_weight}" if args.supcon_weight > 0 else ""
    _nodec_tag  = "_SPIRL" if args.no_text_decoder else ""
    _state_tag = "_state" if args.use_state_cond else ""
    RUN_NAME = (f"rw{args.recon_weight}_d{args.dropout}_beta{args.beta}-{args.beta_high}"
                f"_z{args.latent_dim}_chunk{args.chunk_size}"
                f"{_text_tag}_{_protocol}{_cyc_tag}{_arch_tag}{_supcon_tag}{_nodec_tag}{_state_tag}_h{args.exec_steps}_grip{args.gripper_weight}")


    train_dataloader, test_dataloader, action_stats = get_text_action_ram_cached_dataloader(
        suite=args.suite,
        batch_size=BATCH_SIZE,
        text_backbone=args.text_backbone,
        train_split_ratio=args.train_split_ratio,
        return_states=args.use_state_cond,
    )

    train_actions_all = train_dataloader.dataset.tensors[0]  # (N, T, 7)
    train_texts_all   = train_dataloader.dataset.tensors[1]  # (N, text_dim)
    if args.use_state_cond:
        train_states_all = train_dataloader.dataset.tensors[2]  # (N, 8)
    else:
        train_states_all = None
    global_mean = train_actions_all.mean(dim=1).mean(dim=0)
    global_std  = train_actions_all.mean(dim=1).std(dim=0)

    # ── Precompute integer task IDs for SupCon (same text = same task) ───────
    # Works for any number of tasks; no LIBERO-specific code needed.
    _text_to_id: dict = {}
    _id_list = []
    for _t in train_texts_all:
        _key = _t.numpy().tobytes()
        if _key not in _text_to_id:
            _text_to_id[_key] = len(_text_to_id)
        _id_list.append(_text_to_id[_key])
    train_task_ids_all = torch.tensor(_id_list, dtype=torch.long)
    n_unique_tasks = len(_text_to_id)
    print(f"🏷️  Precomputed task IDs: {n_unique_tasks} unique tasks found.")

    # Rebuild the dataloader with a 3rd/4th tensor (task_ids/states) for training.
    from torch.utils.data import TensorDataset, DataLoader
    if args.use_state_cond:
        train_dataset_full = TensorDataset(train_actions_all, train_texts_all, train_task_ids_all, train_states_all)
    else:
        train_dataset_full = TensorDataset(train_actions_all, train_texts_all, train_task_ids_all)
    train_dataloader   = DataLoader(train_dataset_full, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=0, drop_last=True)

    if test_dataloader is None:
        val_dataloader = None
        print("📊 Protocol A: Using full training dataset with running training loss for convergence/patience.")
    else:
        val_dataloader = test_dataloader

    stats_path = f"{suite_save_dir}/dataset_statistics.pt"
    torch.save(action_stats, stats_path)
    print(f"💾 Saved Dataset Stats to: {stats_path}")

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=f'{args.suite}/{RUN_NAME}',
                   settings=wandb.Settings(console="off"), config={
                       "suite": args.suite, "lr": LR, "batch_size": BATCH_SIZE,
                       "max_steps": MAX_STEPS, "beta": args.beta, "recon_weight": args.recon_weight,
                       "chunk_size": args.chunk_size, "latent_dim": args.latent_dim,
                       "n_cycles": args.n_cycles, "text_backbone": args.text_backbone,
                       "arch": _arch_tag, "seed": args.seed,
                       "supcon_weight": args.supcon_weight, "supcon_temp": args.supcon_temp,
                       "no_text_decoder": args.no_text_decoder,
                   })

    n_blocks = max(3, (args.chunk_size - 1).bit_length())
    if args.use_mlp:
        print("⚡ Using MLP Architecture")
        vae = MLPTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
        ).to(DEVICE)
    elif args.use_tcn:
        print(f"⚡ Using TCN Decoder-Only CVAE (n_blocks={n_blocks})")
        vae = TCNTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
            hidden_channels=64, n_blocks=n_blocks,
        ).to(DEVICE)
    elif args.use_cond_prior:
        print(f"⚡ Using Conditional Prior CVAE (n_blocks={n_blocks}, use_state={args.use_state_cond})")
        vae = TCNTextCondPriorCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
            hidden_channels=64, n_blocks=n_blocks,
            use_state=args.use_state_cond, state_dim=8,
        ).to(DEVICE)
    elif args.use_wae:
        print(f"⚡ Using Wasserstein AE (n_blocks={n_blocks})")
        vae = TCNTextWAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
            hidden_channels=64, n_blocks=n_blocks,
        ).to(DEVICE)
    elif args.use_cvae:
        print(f"⚡ Using Full CVAE (text encoder+decoder, gate_init={args.enc_text_gate_init})")
        vae = TCNTextActionCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
            hidden_channels=64, n_blocks=n_blocks,
            enc_text_gate_init=args.enc_text_gate_init,
        ).to(DEVICE)
    else:
        print("⚡ Using Convolutional TCVAE")
        vae = ConvTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
        ).to(DEVICE)

    # set_action_stats is still needed for checkpoint saving consistency;
    # compute_loss no longer uses it but keep it for potential future use.
    vae.set_action_stats(global_mean, global_std)
    vae.no_text_decoder = args.no_text_decoder

    optimizer = optim.AdamW(vae.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

    start_step = 0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
        ckpt_data = torch.load(args.resume_from_checkpoint, map_location=DEVICE)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            vae.load_state_dict(ckpt_data["model_state_dict"])
            if "optimizer_state_dict" in ckpt_data:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                print("  ✅ Optimizer state restored.")
            if "scheduler_state_dict" in ckpt_data:
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                print("  ✅ Scheduler state restored.")
            if "step" in ckpt_data:
                start_step = ckpt_data["step"] + 1
                print(f"  ✅ Resuming from step {start_step}.")
        else:
            # Fallback for plain state_dict
            vae.load_state_dict(ckpt_data)
            print("  ✅ Model state_dict restored.")

    # Precompute integer task labels for MIG (Protocol B only).
    if test_dataloader is not None:
        test_texts_all = test_dataloader.dataset.tensors[1]
        _seen: dict = {}
        _test_label_list = []
        for _t in test_texts_all:
            _key = _t.numpy().tobytes()
            if _key not in _seen:
                _seen[_key] = len(_seen)
            _test_label_list.append(_seen[_key])
        test_labels_all = torch.tensor(_test_label_list, dtype=torch.long)
        print(f"📊 {len(_seen)} unique test task(s) found for MIG")
    else:
        test_labels_all = None
        print("📊 Protocol A: no held-out test tasks — MIG tracking disabled.")

    print(f"✅ Data loaded — Train batches: {len(train_dataloader)}, "
          f"Test batches: {len(test_dataloader) if test_dataloader else 'N/A'}")

    if USE_WANDB:
        log_gt_video_probe(step=0, suite_name=args.suite, stats_path=stats_path,
                           device=DEVICE, probe_task_name=PROBE_VAL_TASK, split_name="val")
        log_gt_video_probe(step=0, suite_name=args.suite, stats_path=stats_path,
                           device=DEVICE, probe_task_name=PROBE_TRAIN_TASK, split_name="train")

    best_val_loss = float('inf')
    best_val_mse = float('inf')
    patience_counter = 0
    running_train_losses = []
    running_train_mses = []
    running_train_grippers = []
    running_train_kls = []
    running_train_supcons = []

    print(f"Starting Training for {args.suite}...")
    print(f"  Beta Schedule: {args.beta_schedule} (beta_max={args.beta}, warmup_ratio={args.warmup_ratio}, n_cycles={args.n_cycles})")
    print(f"  SupCon: {'ON  (weight={}, temp={})'.format(args.supcon_weight, args.supcon_temp) if args.supcon_weight > 0 else 'OFF'}")
    print(f"  No-text-decoder: {'ON  (Play-LMP / SPIRL style)' if args.no_text_decoder else 'OFF'}")
    train_data_iter = iter(train_dataloader)

    with tqdm.tqdm(total=MAX_STEPS, initial=start_step) as pbar:
        for step in range(start_step, MAX_STEPS):
            try:
                batch = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_dataloader)
                batch = next(train_data_iter)

            if args.use_state_cond:
                actions, text_features, task_ids_batch, states_batch = batch
                states_batch = states_batch.to(DEVICE)
            else:
                actions, text_features, task_ids_batch = batch
                states_batch = None

            actions        = actions.to(DEVICE)
            text_features  = text_features.to(DEVICE)
            task_ids_batch = task_ids_batch.to(DEVICE)

            # decode_text: real text unless --no_text_decoder (Play-LMP / SPIRL style)
            decode_text = torch.zeros_like(text_features) if args.no_text_decoder else text_features

            current_beta = get_beta_schedule(
                step, MAX_STEPS, args.beta,
                n_cycles=args.n_cycles,
                warmup_ratio=args.warmup_ratio,
                schedule_type=args.beta_schedule,
                beta_high=args.beta_high,
            )

            # ── Forward pass (encode / reparameterise / decode separately so we
            #    can inject decode_text independently of encode_text) ──────────
            if args.use_cond_prior:
                mu_q, logvar_q = vae.encode(actions)
                mu_p, logvar_p = vae.get_prior(text_features, states_batch)   # prior always sees real text
                z              = vae.reparameterize(mu_q, logvar_q)
                recon_actions  = vae.decode(z, decode_text, states_batch)
                loss, recon_loss, loss_continuous, loss_gripper, kl_loss = vae.compute_loss(
                    actions, recon_actions, mu_q, logvar_q, mu_p, logvar_p,
                    beta=current_beta, recon_weight=args.recon_weight, gripper_weight=args.gripper_weight,
                )
                mu = mu_q
            elif args.use_wae:
                z, _           = vae.encode(actions)             # deterministic; _ is zeros
                recon_actions  = vae.decode(z, decode_text)
                loss, recon_loss, loss_continuous, loss_gripper, kl_loss = vae.compute_loss(
                    actions, recon_actions, z,
                    beta=current_beta, recon_weight=args.recon_weight, gripper_weight=args.gripper_weight,
                )
                mu = z
            else:
                # ConvTCVAE / MLP / TCN (text-free encoder) / full CVAE (text encoder)
                if args.use_cvae:  # TCNTextActionCVAE.encode() needs text_features
                    mu, logvar = vae.encode(actions, text_features)
                else:
                    mu, logvar = vae.encode(actions)
                z             = vae.reparameterize(mu, logvar)
                recon_actions = vae.decode(z, decode_text)
                loss, recon_loss, loss_continuous, loss_gripper, kl_loss = vae.compute_loss(
                    actions, recon_actions, mu, logvar,
                    beta=current_beta, recon_weight=args.recon_weight, gripper_weight=args.gripper_weight,
                )

            # ── Supervised Contrastive auxiliary loss (optional) ────────────
            supcon_loss_val = torch.tensor(0.0, device=DEVICE)
            if args.supcon_weight > 0:
                supcon_loss_val = compute_supcon_loss(z, task_ids_batch, args.supcon_temp)
                loss = loss + args.supcon_weight * supcon_loss_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                dim_stds = mu.std(dim=0)

            pure_mse_metric = loss_continuous.item() / (CHUNK_SIZE * 6)
            if USE_WANDB:
                wandb.log({
                    "train/total_loss":   loss.item(),
                    "train/recon_mse":    pure_mse_metric,
                    "train/bce_gripper":  loss_gripper.item() / CHUNK_SIZE,
                    "train/kl_loss":      kl_loss.item(),
                    "train/current_beta": current_beta,
                    "train/lr":           scheduler.get_last_lr()[0],
                    "train/supcon_loss":  supcon_loss_val.item(),
                    "diagnostics/n_active_dims": (dim_stds > 0.1).sum().item(),
                    "diagnostics/n_mid_dims":    ((dim_stds >= 0.05) & (dim_stds <= 0.1)).sum().item(),
                    "diagnostics/n_dead_dims":   (dim_stds < 0.05).sum().item(),
                    "diagnostics/dim_std_hist":  wandb.Histogram(dim_stds.cpu().numpy()),
                    "global_step": step,
                }, step=step)

            running_train_losses.append(loss.item())
            running_train_mses.append(pure_mse_metric)
            running_train_grippers.append(loss_gripper.item() / CHUNK_SIZE)
            running_train_kls.append(kl_loss.item())
            running_train_supcons.append(supcon_loss_val.item())

            pbar.set_description(f"Loss: {loss.item():.2f} (MSE: {pure_mse_metric:.4f})")
            pbar.update(1)

            if (step + 1) % TEST_STEPS == 0:
                vae.eval()

                if val_dataloader is not None:
                    # Protocol B: run actual validation evaluation
                    val_losses, val_mses, val_grippers, val_kls = [], [], [], []
                    all_val_mus = []

                    for batch in val_dataloader:
                        if args.use_state_cond:
                            val_actions, val_text_features, val_states = batch
                            val_states = val_states.to(DEVICE)
                        else:
                            val_actions, val_text_features = batch
                            val_states = None
                        val_actions = val_actions.to(DEVICE)
                        val_text_features = val_text_features.to(DEVICE)
                        val_beta = current_beta

                        with torch.no_grad():
                            if args.use_cond_prior:
                                val_recon, val_mu, val_logvar, val_mu_p, val_logvar_p, val_z = vae(val_actions, val_text_features, val_states)
                                v_loss, _, v_cont, v_grip, v_kl = vae.compute_loss(
                                    val_actions, val_recon, val_mu, val_logvar, val_mu_p, val_logvar_p,
                                    beta=val_beta, recon_weight=args.recon_weight, gripper_weight=args.gripper_weight,
                                )
                            elif args.use_wae:
                                val_recon, val_mu = vae(val_actions, val_text_features)
                                v_loss, _, v_cont, v_grip, v_kl = vae.compute_loss(
                                    val_actions, val_recon, val_mu,
                                    beta=val_beta, recon_weight=args.recon_weight, gripper_weight=args.gripper_weight,
                                )
                            else:
                                val_recon, val_mu, val_logvar, val_z = vae(val_actions, val_text_features)
                                v_loss, _, v_cont, v_grip, v_kl = vae.compute_loss(
                                    val_actions, val_recon, val_mu, val_logvar,
                                    beta=val_beta, recon_weight=args.recon_weight, gripper_weight=args.gripper_weight,
                                )
                            val_losses.append(v_loss.item())
                            val_mses.append(v_cont.item() / (CHUNK_SIZE * 6))
                            val_grippers.append(v_grip.item() / CHUNK_SIZE)
                            val_kls.append(v_kl.item())
                            all_val_mus.append(val_mu.cpu())

                    mean_val_loss = sum(val_losses) / len(val_losses)
                    mean_val_mse = sum(val_mses) / len(val_mses)
                    mean_val_gripper = sum(val_grippers) / len(val_grippers)
                    mean_val_kl = sum(val_kls) / len(val_kls)

                    if test_labels_all is not None:
                        all_val_mus_cat = torch.cat(all_val_mus, dim=0)
                        mig_score = compute_mig(all_val_mus_cat, test_labels_all[:len(all_val_mus_cat)])
                    else:
                        mig_score = 0.0

                    if USE_WANDB:
                        log_dict = {
                            "val/total_loss":  mean_val_loss,
                            "val/recon_mse":   mean_val_mse,
                            "val/bce_gripper": mean_val_gripper,
                            "val/kl_loss":     mean_val_kl,
                            "global_step":      step,
                        }
                        if test_labels_all is not None:
                            log_dict["val/mig"] = mig_score
                        wandb.log(log_dict, step=step)

                    if test_labels_all is not None:
                        print(f"\n📊 Step {step+1} | Val Loss: {mean_val_loss:.4f} | MIG: {mig_score:.4f}")
                    else:
                        print(f"\n📊 Step {step+1} | Val Loss: {mean_val_loss:.4f} (Protocol B)")
                else:
                    # Protocol A: calculate running average of training metrics
                    mean_val_loss = sum(running_train_losses) / len(running_train_losses)
                    mean_val_mse = sum(running_train_mses) / len(running_train_mses)
                    mean_val_gripper = sum(running_train_grippers) / len(running_train_grippers)
                    mean_val_kl = sum(running_train_kls) / len(running_train_kls)
                    mig_score = 0.0

                    if USE_WANDB:
                        wandb.log({
                            "val/running_total_loss": mean_val_loss,
                            "val/running_recon_mse":  mean_val_mse,
                            "val/running_bce_gripper": mean_val_gripper,
                            "val/running_kl_loss":     mean_val_kl,
                            "global_step":              step,
                        }, step=step)

                    print(f"\n📊 Step {step+1} | Running Train Loss: {mean_val_loss:.4f} (Protocol A)")

                # Reset running lists
                running_train_losses = []
                running_train_mses = []
                running_train_grippers = []
                running_train_kls = []

                if USE_WANDB:
                    log_video_probe(
                        vae=vae, step=step, suite_name=args.suite, stats_path=stats_path,
                        device=DEVICE, probe_task_name=PROBE_VAL_TASK,
                        chunk_size=args.chunk_size, split_name="val", text_backbone=args.text_backbone,
                        exec_steps=args.exec_steps,
                    )
                    log_video_probe(
                        vae=vae, step=step, suite_name=args.suite, stats_path=stats_path,
                        device=DEVICE, probe_task_name=PROBE_TRAIN_TASK,
                        chunk_size=args.chunk_size, split_name="train", text_backbone=args.text_backbone,
                        exec_steps=args.exec_steps,
                    )

                metric_name = "validation loss" if val_dataloader is not None else "running total loss"
                ckpt_state = {
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "best_val_mse": best_val_mse,
                    "val_loss": mean_val_loss,
                    "val_mse": mean_val_mse,
                }
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_mse = mean_val_mse
                    patience_counter = 0
                    best_ckpt_path = f"{suite_save_dir}/{RUN_NAME}_seed_{args.seed}_best.pt"
                    torch.save(ckpt_state, best_ckpt_path)
                    print(f"🏆 Step {step+1} | New best {metric_name}: {best_val_loss:.4f} (MSE: {mean_val_mse:.4f})! Saved best checkpoint.")
                else:
                    patience_counter += 1
                    print(f"📉 Step {step+1} | {metric_name.capitalize()} did not improve: {mean_val_loss:.4f} (Best: {best_val_loss:.4f} | MSE: {mean_val_mse:.4f}). Patience: {patience_counter}/{args.patience}")

                torch.save(ckpt_state, f"{suite_save_dir}/{RUN_NAME}_seed_{args.seed}_step_{step+1}.pt")
                
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"🛑 Early stopping triggered after {step+1} steps.")
                    break

                vae.train()

    if USE_WANDB:
        wandb.finish()
    print(f"✅ Training Complete for {args.suite}!")


if __name__ == "__main__":
    train_conditional_disentangler()