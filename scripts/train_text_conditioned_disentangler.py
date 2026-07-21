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
from utils.metrics import compute_mig, cyclic_beta_schedule

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
    parser.add_argument("--n_cycles",     type=int,   default=4,    help="Number of beta annealing cycles (0 = constant)")
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
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def train_conditional_disentangler():
    args = parse_args()
    seed_everything(args.seed)

    CHUNK_SIZE = args.chunk_size
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
    RUN_NAME = (f"rw{args.recon_weight}_d{args.dropout}_beta{args.beta}"
                f"_z{args.latent_dim}_chunk{args.chunk_size}"
                f"{_text_tag}_{_protocol}{_cyc_tag}{_arch_tag}")

    train_dataloader, test_dataloader, action_stats = get_text_action_ram_cached_dataloader(
        suite=args.suite,
        batch_size=BATCH_SIZE,
        text_backbone=args.text_backbone,
        train_split_ratio=args.train_split_ratio,
    )

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
        print(f"⚡ Using Conditional Prior CVAE (n_blocks={n_blocks})")
        vae = TCNTextCondPriorCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
            latent_dim=args.latent_dim, text_emb_dim=text_emb_dim,
            beta=args.beta, dropout=args.dropout,
            hidden_channels=64, n_blocks=n_blocks,
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
    train_actions_all = train_dataloader.dataset.tensors[0]
    global_mean = train_actions_all.mean(dim=1).mean(dim=0)
    global_std  = train_actions_all.mean(dim=1).std(dim=0)
    vae.set_action_stats(global_mean, global_std)

    optimizer = optim.AdamW(vae.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

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

    print(f"Starting Training for {args.suite}...")
    train_data_iter = iter(train_dataloader)

    with tqdm.tqdm(total=MAX_STEPS) as pbar:
        for step in range(MAX_STEPS):
            try:
                actions, text_features = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_dataloader)
                actions, text_features = next(train_data_iter)

            actions, text_features = actions.to(DEVICE), text_features.to(DEVICE)
            
            current_beta = cyclic_beta_schedule(step, MAX_STEPS, args.beta, args.n_cycles)

            if args.use_cond_prior:
                recon_actions, mu_q, logvar_q, mu_p, logvar_p, z = vae(actions, text_features)
                loss, recon_loss, loss_continuous, loss_gripper, kl_loss = vae.compute_loss(
                    actions, recon_actions, mu_q, logvar_q, mu_p, logvar_p,
                    beta=current_beta, recon_weight=args.recon_weight,
                )
                mu = mu_q
            elif args.use_wae:
                recon_actions, z = vae(actions, text_features)
                loss, recon_loss, loss_continuous, loss_gripper, kl_loss = vae.compute_loss(
                    actions, recon_actions, z,
                    beta=current_beta, recon_weight=args.recon_weight,
                )
                mu = z
            else:
                recon_actions, mu, logvar, z = vae(actions, text_features)
                loss, recon_loss, loss_continuous, loss_gripper, kl_loss = vae.compute_loss(
                    actions, recon_actions, mu, logvar,
                    beta=current_beta, recon_weight=args.recon_weight,
                )

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
                    "diagnostics/n_active_dims": (dim_stds > 0.1).sum().item(),
                    "diagnostics/n_mid_dims":    ((dim_stds >= 0.05) & (dim_stds <= 0.1)).sum().item(),
                    "diagnostics/n_dead_dims":   (dim_stds < 0.05).sum().item(),
                    "diagnostics/dim_std_hist":  wandb.Histogram(dim_stds.cpu().numpy()),
                    "global_step": step,
                })

            pbar.set_description(f"Loss: {loss.item():.2f} (MSE: {pure_mse_metric:.4f})")
            pbar.update(1)

            if (step + 1) % TEST_STEPS == 0:
                vae.eval()

                if test_dataloader is not None:
                    test_losses, test_mses, test_grippers, test_kls = [], [], [], []
                    all_test_mus = []

                    for test_actions, test_text_features in test_dataloader:
                        test_actions = test_actions.to(DEVICE)
                        test_text_features = test_text_features.to(DEVICE)
                        # Use the same beta as the current train step
                        test_beta = current_beta

                        with torch.no_grad():
                            if args.use_cond_prior:
                                test_recon, test_mu, test_logvar, test_mu_p, test_logvar_p, test_z = vae(test_actions, test_text_features)
                                t_loss, _, t_cont, t_grip, t_kl = vae.compute_loss(
                                    test_actions, test_recon, test_mu, test_logvar, test_mu_p, test_logvar_p,
                                    beta=test_beta, recon_weight=args.recon_weight,
                                )
                            elif args.use_wae:
                                test_recon, test_mu = vae(test_actions, test_text_features)
                                t_loss, _, t_cont, t_grip, t_kl = vae.compute_loss(
                                    test_actions, test_recon, test_mu,
                                    beta=test_beta, recon_weight=args.recon_weight,
                                )
                            else:
                                test_recon, test_mu, test_logvar, test_z = vae(test_actions, test_text_features)
                                t_loss, _, t_cont, t_grip, t_kl = vae.compute_loss(
                                    test_actions, test_recon, test_mu, test_logvar,
                                    beta=test_beta, recon_weight=args.recon_weight,
                                )
                            test_losses.append(t_loss.item())
                            test_mses.append(t_cont.item() / (CHUNK_SIZE * 6))
                            test_grippers.append(t_grip.item() / CHUNK_SIZE)
                            test_kls.append(t_kl.item())
                            all_test_mus.append(test_mu.cpu())

                    all_test_mus_cat = torch.cat(all_test_mus, dim=0)
                    mig_score = compute_mig(all_test_mus_cat, test_labels_all[:len(all_test_mus_cat)])

                    if USE_WANDB:
                        wandb.log({
                            "test/total_loss":  sum(test_losses)   / len(test_losses),
                            "test/recon_mse":   sum(test_mses)     / len(test_mses),
                            "test/bce_gripper": sum(test_grippers) / len(test_grippers),
                            "test/kl_loss":     sum(test_kls)      / len(test_kls),
                            "test/mig":         mig_score,
                            "global_step":      step,
                        })
                        print(f"\n📊 Step {step+1} | MIG: {mig_score:.4f}")
                else:
                    print(f"\n📊 Step {step+1} | Protocol A: no test eval (use simulator).")

                if USE_WANDB:
                    log_video_probe(
                        vae=vae, step=step, suite_name=args.suite, stats_path=stats_path,
                        device=DEVICE, probe_task_name=PROBE_VAL_TASK,
                        chunk_size=args.chunk_size, split_name="val", text_backbone=args.text_backbone,
                    )
                    log_video_probe(
                        vae=vae, step=step, suite_name=args.suite, stats_path=stats_path,
                        device=DEVICE, probe_task_name=PROBE_TRAIN_TASK,
                        chunk_size=args.chunk_size, split_name="train", text_backbone=args.text_backbone,
                    )

                torch.save(vae.state_dict(), f"{suite_save_dir}/{RUN_NAME}_seed_{args.seed}_step_{step+1}.pt")
                vae.train()

    if USE_WANDB:
        wandb.finish()
    print(f"✅ Training Complete for {args.suite}!")


if __name__ == "__main__":
    train_conditional_disentangler()