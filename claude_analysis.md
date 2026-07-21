
## DisentangledVLA - Architecture & Methodology Review

### Big-picture opinion

The overall 3-phase strategy is sound and well-scoped: (1) get a near-perfect trajectory VAE, (2) progressively disentangle, (3) train a projector/adapter. The implementation is cleaner than typical research code and the TC decomposition is done correctly. Below I'll go phase by phase with the most impactful issues first.

---

### Phase 1 — Reconstruction quality

**What's working:**
- `recon_weight=100` + `beta=0.1` effectively downweights the TC/KL pressure. The beta warmup (`current_beta = args.beta * min(1.0, step / WARMUP_STEPS)`) is good practice and prevents KL collapse early in training.
- Gradient clipping (`max_norm=1.0`) + AdamW + cosine LR is a solid baseline.
- The masked loss in the full-trajectory variant is the right way to handle variable-length trajectories.

**Issues to fix:**

**1. Per-batch normalization in the kinematic guidance loss is broken** (in both ConvTextActionBetaTCVAE and MLPTextActionBetaTCVAE):

```python
# CURRENT - target shifts every batch:
target_mean = target_kinematics.mean(dim=0, keepdim=True)  # batch stats!
target_std  = target_kinematics.std(dim=0, keepdim=True)
target_kinematics = (target_kinematics - target_mean) / (target_std + 1e-5)
loss_guidance = F.mse_loss(mu[:, :7], target_kinematics)
```
The normalisation statistics change every batch, so `mu[:, :7]` is being pulled toward a different target on every step. This introduces a noisy, inconsistent gradient signal. Replace with **fixed dataset-level statistics** computed once at the start of training:

```python
# Compute ONCE before training, outside the loop:
# action_global_mean, action_global_std = compute_from_full_dataset()

# In compute_loss, pass them as arguments or store in model:
target_kinematics = (target_kinematics - global_mean) / (global_std + 1e-5)
```

**2. `ActionBetaTCVAE` flat size is hardcoded for chunk_size=16.**
`self.fc_mu = nn.Linear(256, latent_dim)` silently breaks for any other chunk size. `ConvTextActionBetaTCVAE` already fixes this with `self.flattened_size = 128 * (chunk_size // 8)` — apply the same pattern to the base class.

**3. Conv1D encoder has almost zero temporal resolution after 3 stride-2 layers.**
For chunk_size=8: `8 → 4 → 2 → 1`. The 1D bottleneck after 3 strides on a sequence of only 8 is essentially a global average pool — you lose all temporal ordering before the linear. This is likely why MLP ≈ Conv1D in your results; they're computing roughly the same thing. Options:
- Use **fewer strides** (stride=1 with dilation, or only 1-2 stride-2 layers).
- Use **causal depthwise-separable Conv1D** to preserve temporal context.
- Or embrace it: switch to a proper TCN (Temporal Convolutional Network) with dilated causal convolutions for better sequential modeling.

**4. Gripper GT is thresholded twice** — minor but confusing to read:
```python
gt_gripper_binary = gt_gripper           # line A (kept)
gt_gripper_binary = (gt_gripper > 0.0).float()  # line B (overwrites A)
```
Line A is dead code. Clean it up.

---

### Phase 2 — Disentanglement

**What's correct:**
- The minibatch-weighted TC estimator (Chen et al. 2018) is implemented correctly: log q(z), log ∏ q(z_d), and log p(z) are all right.
- Alpha/beta/gamma decomposition with α=1, γ=1, β=variable is standard.

**Issues / suggestions:**

**5. TC estimation quality degrades with small z_dim and large batches but improves with z_dim**  
With z_dim=64 and B=128 the intermediate tensor `log_q_z_given_x` is `(128, 128, 64)` ≈ 4MB, fine. But with z_dim=512 it's 32MB per forward, and gradients double that. Keep this in mind when scaling z_dim.

**6. Cyclic beta annealing** is consistently better than single-shot warmup for disentanglement (Li et al., "Don't Blame the ELBO!", 2019). The current linear warmup may settle in a mode that's hard to escape later. Consider:
```python
# Cosine cyclical schedule over n_cycles
cycle_period = MAX_STEPS // n_cycles
current_beta = args.beta * 0.5 * (1 - math.cos(math.pi * (step % cycle_period) / cycle_period))
```

**7. Guidance loss aligns `mu[:, :7]` with mean-position** — this is a creative structural prior but it conflates the "task-specific" and "physics" concepts. A cleaner split would be to reserve specific latent dimensions explicitly for semantics vs. kinematics and use separate projection heads, rather than assuming dims 0–6 are positional.

**8. Text conditioning: encoder is blind, decoder sees text** (in `ConvTextActionBetaTCVAE` / `MLPTextActionBetaTCVAE`). This is actually a reasonable choice for disentanglement — it forces z to be task-agnostic (only motion, not intent). BUT it means the encoder cannot use contextual information to resolve ambiguous trajectories. Consider at least an option to also concatenate the text embedding to the encoder (controlled by a flag), so you can ablate.

**9. Missing quantitative disentanglement metrics.** Without MIG (Mutual Information Gap) or DCI scores, you can't judge whether beta changes are actually helping. Add a `compute_mig()` call in your validation step. It only needs latent codes + ground-truth generative factor labels, which you already have via the task/skill labels in the HDF5.

---

### Phase 3 — Projector / Adapter

**What's conceptually sound:** Using the frozen VAE as a "skills teacher" and training a lightweight adapter on top of a frozen VLA is a strong approach. It avoids catastrophic forgetting and is data-efficient.

**Critical issue:**

**10. The projector only targets `mu`, ignoring `logvar`.** When the projector predicts a Gaussian `(mu_pred, sigma_pred)` but is only trained via NLL against the *mean* target from the VAE (not a full sample `z`), the uncertainty head learns nothing meaningful — `sigma_pred` will collapse or drift arbitrarily. To fix this:
- Use `z = vae.reparameterize(mu, logvar)` as target (a single sample rather than the mean), OR
- Distill the full posterior by minimizing KL(projector_dist || vae_encoder_dist) instead of NLL(z_target | projector_dist).

The KL distillation is more principled:
```python
# In projector training:
with torch.no_grad():
    vae_mu, vae_logvar = vae.encode(gt_actions)
proj_dist, proj_mu, proj_logvar = projector(vla_embedding) 
# Minimise KL(N(vae_mu, exp(vae_logvar)) || N(proj_mu, exp(proj_logvar)))
# KL(p||q) = 0.5 * [log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p-mu_q)^2)/sigma_q^2 - 1]
kl_loss = 0.5 * (proj_logvar - vae_logvar + 
                  (vae_logvar.exp() + (vae_mu - proj_mu).pow(2)) / proj_logvar.exp() - 1).sum(-1).mean()
```

**11. OpenVLA embedding extraction uses only the *last-token* hidden state.** For a generative LLM, the last token before generation is indeed the "action-predicting" state, so it's defensible. But it discards rich contextual information from earlier tokens (object names, spatial prepositions). A more informative alternative is **mean-pool over all visual tokens** or use a learned cross-attention pooler on the full hidden state sequence.

**12. Projector input = VLA(image, instruction) but no proprioception.** At deployment, the same image+instruction yields the same projector output regardless of robot state. This means z (and thus the decoded actions) is fixed for the whole episode. If you want reactive/closed-loop behavior you need to include the robot state (eef position/orientation, gripper) as an additional input to the projector.

---

### Architecture improvements worth trying

| Idea | Why | Complexity |
|---|---|---|
| **Residual blocks in encoder/decoder** | Gradient flow for deeper reconstruction | Low |
| **Transformer encoder** (few attention layers, causal) | Captures temporal dependencies across the full chunk | Medium |
| **Hierarchical latent** (2–3 levels: coarse motion + fine correction) | Better alignment with the structure of manipulation | High |
| **Vector-Quantized bottleneck on top of the continuous VAE** | Enables discrete "skills" for later planning | Medium |
| **Normalizing flow posterior** (IAF/RealNVP) instead of Gaussian | Richer posterior → better reconstruction-disentanglement trade-off | High |
| **Equivariant encoder for SE(3)** | Rotation/translation equivariance in the robot workspace | High |

---

### Summary of priority fixes

1. **Fix per-batch normalization in guidance loss** — precompute global statistics.
2. **Fix Conv hardcoded flat dim in base `ActionBetaTCVAE`** — dynamic like `ConvTextActionBetaTCVAE`.
3. **Switch projector to full posterior distillation (KL)** instead of NLL against mean-only target.
4. **Add MIG to validation loop** — without it you're flying blind on disentanglement quality.
5. **Add proprioception to projector input** for closed-loop behavior.
6. **Try cyclic beta schedule** once you hit good reconstruction — it will give you better disentanglement per reconstruction-unit-of-budget.
7. **Revisit Conv encoder temporal capacity** — 3× stride-2 on chunk=8 effectively destroys temporal ordering.