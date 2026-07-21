import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvTextActionBetaTCVAE(nn.Module):
    def __init__(self, action_dim=7, chunk_size=8, latent_dim=16, text_emb_dim=512, beta=6.0, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.text_emb_dim = text_emb_dim
        self.beta = beta  
        self.dropout = dropout

        self.final_seq_len = chunk_size // 8 
        self.flattened_size = 128 * self.final_seq_len

        # Persistent buffers for guidance loss normalisation.
        # Must be set once before training via set_action_stats().
        # register_buffer ensures they are part of state_dict and move with .to(device).
        self.register_buffer('action_global_mean', torch.zeros(action_dim))
        self.register_buffer('action_global_std', torch.ones(action_dim))
        self._action_stats_set = False

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Use the dynamically calculated flattened size!
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # 2. DECODER
        self.decoder_input = nn.Linear(latent_dim + self.text_emb_dim, self.flattened_size)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, action_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set dataset-level guidance normalisation statistics.
        Call once after building the model, before training starts.
        Args:
            mean: (action_dim,) mean of action.mean(dim=1) over the training split
            std:  (action_dim,) std  of action.mean(dim=1) over the training split
        """
        self.action_global_mean.copy_(mean.to(self.action_global_mean.device))
        self.action_global_std.copy_(std.to(self.action_global_std.device))
        self._action_stats_set = True

    def encode(self, action):
        x = action.float().permute(0, 2, 1).contiguous()
        
        expected_time_steps = self.chunk_size
        actual_time_steps = x.size(2)
        if actual_time_steps != expected_time_steps:
            raise RuntimeError(f"🧨 DATALOADER MISMATCH: VAE expected {expected_time_steps} frames, "
                               f"but received {actual_time_steps} frames. Clear your dataset cache!")

        h = self.encoder_conv(x) 
        
        if h.size(1) != self.flattened_size:
            raise RuntimeError(f"🧨 MATH MISMATCH: Flattened conv output is {h.size(1)}, "
                               f"but Linear layer expects {self.flattened_size}. Check the math!")

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z, text_emb):
        if self.training and torch.rand(1).item() < self.dropout:
            text_emb = torch.zeros_like(text_emb)
        fused_z = torch.cat([z, text_emb], dim=-1) 
        h = self.decoder_input(fused_z)            
        h = h.view(-1, 128, self.final_seq_len)                     
        x_recon = self.decoder_conv(h)             
        
        return x_recon.permute(0, 2, 1).contiguous()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, action, text_emb):
        mu, logvar = self.encode(action)
        z = self.reparameterize(mu, logvar)
        action_recon = self.decode(z, text_emb)
        return action_recon, mu, logvar, z

    def compute_loss(self, action, action_recon, mu, logvar, z, beta=6.0, alpha=1.0, recon_weight=50, vel_weight=0.0):
        batch_size, latent_dim = z.shape

        pred_continuous = action_recon[..., :6]
        gt_continuous = action[..., :6]

        pred_gripper = action_recon[..., 6] 
        gt_gripper = action[..., 6]

        gt_gripper_binary = (gt_gripper > 0.0).float()

        # Map Predicted Gripper from [-1, 1] to [0, 1]
        pred_gripper_prob = (pred_gripper + 1.0) / 2.0
        # Safeguard against log(0) NaN crashes
        pred_gripper_prob = torch.clamp(pred_gripper_prob, min=1e-6, max=1.0-1e-6)

        # 1. Calculate BCE without reducing
        loss_gripper = F.binary_cross_entropy(pred_gripper_prob, gt_gripper_binary, reduction='none')
        
        # 2. Sum across time (dim 1), then mean across batch (dim 0)
        loss_gripper = loss_gripper.sum(dim=1).mean()
        
        loss_continuous = F.mse_loss(pred_continuous, gt_continuous, reduction='none')
        # Sum across time (dim 1) and action features (dim 2), then mean across batch (dim 0)
        loss_continuous = loss_continuous.sum(dim=(1, 2)).mean()

        recon_loss = (loss_continuous + (0.5 * loss_gripper)) * recon_weight
        
        mu_expand = mu.unsqueeze(0)           
        logvar_expand = logvar.unsqueeze(0)   
        z_expand = z.unsqueeze(1)
        
        log_q_z_given_x = -0.5 * (math.log(2 * math.pi) + logvar_expand + 
                                  (z_expand - mu_expand).pow(2) / torch.exp(logvar_expand))
        log_q_z = torch.logsumexp(log_q_z_given_x.sum(dim=2), dim=1) - math.log(batch_size)
        log_prod_q_z_d = torch.logsumexp(log_q_z_given_x, dim=1) - math.log(batch_size)
        log_prod_q_z_d = log_prod_q_z_d.sum(dim=1)
        log_p_z = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)
        
        mi_loss = (log_q_z_given_x.sum(dim=2).diagonal() - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z_d).mean()
        dw_kl_loss = (log_prod_q_z_d - log_p_z).mean()
        
        
        tcvae_loss = recon_loss + (1.0 * mi_loss) + (beta * tc_loss) + (1.0 * dw_kl_loss)

        target_kinematics = action.mean(dim=1)  # (B, action_dim)
        if alpha > 0.0:
            if not self._action_stats_set:
                raise RuntimeError(
                    "Guidance loss is active (alpha > 0) but action stats have not been set. "
                    "Call model.set_action_stats(mean, std) before training."
                )
            target_kinematics = (target_kinematics - self.action_global_mean) / (self.action_global_std + 1e-5)
            loss_guidance = F.mse_loss(mu[:, :self.action_dim], target_kinematics)
        else:
            loss_guidance = torch.tensor(0.0, device=mu.device)

        # --- Velocity Penalty ---
        # Encourage the VAE to preserve temporal smoothness (first-order finite differences
        # on the continuous 6-DOF dims only; gripper is binary and has no meaningful velocity).
        # Skipped for single-step chunks (T=1) or when vel_weight == 0.
        if vel_weight > 0.0 and action.size(1) > 1:
            vel_gt    = action[:, 1:, :6]       - action[:, :-1, :6]   # (B, T-1, 6)
            vel_recon = action_recon[:, 1:, :6] - action_recon[:, :-1, :6]
            loss_vel  = vel_weight * F.mse_loss(vel_recon, vel_gt, reduction='none').sum(dim=(1, 2)).mean()
        else:
            loss_vel = torch.tensor(0.0, device=action.device)

        total_loss = tcvae_loss + alpha * loss_guidance + loss_vel

        return total_loss, recon_loss, loss_continuous, loss_gripper, mi_loss, tc_loss, loss_guidance, loss_vel


class MLPTextActionBetaTCVAE(nn.Module):
    def __init__(self, action_dim=7, chunk_size=8, latent_dim=16, text_emb_dim=512, beta=6.0, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.text_emb_dim = text_emb_dim
        self.beta = beta  
        self.dropout = dropout

        # For an MLP, we flatten the sequence and action dims into one long vector
        self.flat_action_size = chunk_size * action_dim

        # Persistent buffers for guidance loss normalisation.
        # Must be set once before training via set_action_stats().
        self.register_buffer('action_global_mean', torch.zeros(action_dim))
        self.register_buffer('action_global_std', torch.ones(action_dim))
        self._action_stats_set = False

        # 1. ENCODER
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.flat_action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # 2. DECODER
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim + self.text_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.flat_action_size),
            nn.Tanh()  # Keeps outputs bounded strictly between [-1, 1]
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set dataset-level guidance normalisation statistics.
        Call once after building the model, before training starts.
        Args:
            mean: (action_dim,) mean of action.mean(dim=1) over the training split
            std:  (action_dim,) std  of action.mean(dim=1) over the training split
        """
        self.action_global_mean.copy_(mean.to(self.action_global_mean.device))
        self.action_global_std.copy_(std.to(self.action_global_std.device))
        self._action_stats_set = True

    def encode(self, action):
        batch_size = action.size(0)
        expected_time_steps = self.chunk_size
        actual_time_steps = action.size(1)
        
        if actual_time_steps != expected_time_steps:
            raise RuntimeError(f"🧨 DATALOADER MISMATCH: VAE expected {expected_time_steps} frames, "
                               f"but received {actual_time_steps} frames.")

        # Flatten from (B, Chunk, Dim) -> (B, Chunk * Dim)
        x_flat = action.view(batch_size, -1)
        
        h = self.encoder_mlp(x_flat) 
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z, text_emb):
        batch_size = z.size(0)
        
        if self.training and torch.rand(1).item() < self.dropout:
            text_emb = torch.zeros_like(text_emb)
            
        fused_z = torch.cat([z, text_emb], dim=-1) 
        
        # Output is (B, Chunk * Dim)
        x_recon_flat = self.decoder_mlp(fused_z)            
        
        # Reshape back to (B, Chunk, Dim)
        x_recon = x_recon_flat.view(batch_size, self.chunk_size, self.action_dim)
        return x_recon

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, action, text_emb):
        mu, logvar = self.encode(action)
        z = self.reparameterize(mu, logvar)
        action_recon = self.decode(z, text_emb)
        return action_recon, mu, logvar, z

    def compute_loss(self, action, action_recon, mu, logvar, z, beta=6.0, alpha=1.0, recon_weight=50.0, vel_weight=0.0):
        batch_size, latent_dim = z.shape

        pred_continuous = action_recon[..., :6]
        gt_continuous = action[..., :6]

        pred_gripper = action_recon[..., 6] 
        gt_gripper = action[..., 6]

        gt_gripper_binary = (gt_gripper > 0.0).float()
        
        pred_gripper_prob = (pred_gripper + 1.0) / 2.0
        pred_gripper_prob = torch.clamp(pred_gripper_prob, min=1e-6, max=1.0-1e-6)

        loss_gripper = F.binary_cross_entropy(pred_gripper_prob, gt_gripper_binary, reduction='none')
        loss_gripper = loss_gripper.sum(dim=1).mean()
        
        loss_continuous = F.mse_loss(pred_continuous, gt_continuous, reduction='none')
        loss_continuous = loss_continuous.sum(dim=(1, 2)).mean()

        recon_loss = (loss_continuous + (0.5 * loss_gripper)) * recon_weight
        
        mu_expand = mu.unsqueeze(0)           
        logvar_expand = logvar.unsqueeze(0)   
        z_expand = z.unsqueeze(1)
        
        log_q_z_given_x = -0.5 * (math.log(2 * math.pi) + logvar_expand + 
                                  (z_expand - mu_expand).pow(2) / torch.exp(logvar_expand))
        log_q_z = torch.logsumexp(log_q_z_given_x.sum(dim=2), dim=1) - math.log(batch_size)
        log_prod_q_z_d = torch.logsumexp(log_q_z_given_x, dim=1) - math.log(batch_size)
        log_prod_q_z_d = log_prod_q_z_d.sum(dim=1)
        log_p_z = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)
        
        mi_loss = (log_q_z_given_x.sum(dim=2).diagonal() - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z_d).mean()
        dw_kl_loss = (log_prod_q_z_d - log_p_z).mean()
        
        tcvae_loss = recon_loss + (1.0 * mi_loss) + (beta * tc_loss) + (1.0 * dw_kl_loss)

        target_kinematics = action.mean(dim=1)  # (B, action_dim)
        if alpha > 0.0:
            if not self._action_stats_set:
                raise RuntimeError(
                    "Guidance loss is active (alpha > 0) but action stats have not been set. "
                    "Call model.set_action_stats(mean, std) before training."
                )
            target_kinematics = (target_kinematics - self.action_global_mean) / (self.action_global_std + 1e-5)
            loss_guidance = F.mse_loss(mu[:, :self.action_dim], target_kinematics)
        else:
            loss_guidance = torch.tensor(0.0, device=mu.device)

        # --- Velocity Penalty ---
        if vel_weight > 0.0 and action.size(1) > 1:
            vel_gt    = action[:, 1:, :6]       - action[:, :-1, :6]
            vel_recon = action_recon[:, 1:, :6] - action_recon[:, :-1, :6]
            loss_vel  = vel_weight * F.mse_loss(vel_recon, vel_gt, reduction='none').sum(dim=(1, 2)).mean()
        else:
            loss_vel = torch.tensor(0.0, device=action.device)

        total_loss = tcvae_loss + alpha * loss_guidance + loss_vel

        return total_loss, recon_loss, loss_continuous, loss_gripper, mi_loss, tc_loss, loss_guidance, loss_vel


# ─────────────────────────────────────────────────────────────────────────────
# TCN variant
# ─────────────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    """
    Non-causal depthwise-separable dilated residual block (pre-norm).

    Concepts drawn from three papers:
      - WaveNet (van den Oord et al., 2016): exponentially-spaced dilations for
        receptive-field growth without strided downsampling.
      - TCN (Bai et al., 2018): residual connections for sequence modeling;
        showed TCNs match or beat LSTMs on most sequence benchmarks.
      - MobileNet depthwise-separable convs (Howard et al., 2017): each channel
        learns its own temporal dynamics (depthwise), then DOFs are mixed
        (pointwise).

    Non-causal (symmetric) padding: the VAE encoder sees the full chunk at once.

    Receptive field with kernel_size=3, for dilation stack [1, 2, 4]:
        RF_total = 1 + 2*1 + 2*2 + 2*4 = 15  →  covers chunk_size=8 completely.

    Rule of thumb:
        chunk_size=8  → n_blocks=3  (RF=15)
        chunk_size=16 → n_blocks=4  (RF=31)
        chunk_size=32 → n_blocks=5  (RF=63)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same-length, non-causal

        self.dw_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            dilation=dilation, groups=channels,       # depthwise
        )
        self.pw_conv = nn.Conv1d(channels, channels, kernel_size=1)  # pointwise mix

        num_groups = 8
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)     # pre-norm
        h = self.dw_conv(h)  # per-channel temporal filtering
        h = self.act(h)
        h = self.pw_conv(h)  # channel mixing
        h = self.drop(h)
        return residual + h  # residual skip


class TCNTextActionBetaTCVAE(nn.Module):
    """
    TCN (Temporal Convolutional Network) text-conditioned Beta-TCVAE.

    Encoder:  input_proj → [DilatedResBlock × n_blocks] → global_avg_pool → mu / logvar
    Decoder:  Linear(z + text) → reshape → [DilatedResBlock × n_blocks] → out_proj → Tanh

    Advantages over ConvTextActionBetaTCVAE (3× stride-2 which collapses 8→1):
      - No temporal collapse: full-resolution feature maps, pooled exactly once.
      - Full receptive field: every timestep attends to the entire chunk context.
      - Decoupled learning via depthwise-separable structure.
      - Identical API: drop-in replacement in training scripts.

    Args:
        dropout:         classifier-free guidance dropout on text in decode().
        hidden_channels: TCN channel width (64 is a good default for action_dim=7).
        n_blocks:        number of dilated blocks; dilations are 2^0 … 2^(n-1).
        block_dropout:   regularisation dropout inside each DilatedResBlock.
    """

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 8,
        latent_dim: int = 16,
        text_emb_dim: int = 512,
        beta: float = 6.0,
        dropout: float = 0.1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 3,
        block_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim      = latent_dim
        self.chunk_size      = chunk_size
        self.action_dim      = action_dim
        self.text_emb_dim    = text_emb_dim
        self.beta            = beta
        self.dropout         = dropout
        self.hidden_channels = hidden_channels

        self.register_buffer('action_global_mean', torch.zeros(action_dim))
        self.register_buffer('action_global_std',  torch.ones(action_dim))
        self._action_stats_set = False

        num_groups = 8
        while hidden_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        # ── ENCODER ─────────────────────────────────────────────────────────
        self.enc_input_proj = nn.Conv1d(action_dim, hidden_channels, kernel_size=1)
        self.enc_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=block_dropout)
            for i in range(n_blocks)
        ])
        self.enc_norm = nn.Sequential(
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

        # ── DECODER ─────────────────────────────────────────────────────────
        self.dec_input_proj = nn.Sequential(
            nn.Linear(latent_dim + text_emb_dim, hidden_channels * chunk_size),
            nn.GELU(),
        )
        self.dec_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=block_dropout)
            for i in range(n_blocks)
        ])
        self.dec_out_proj = nn.Sequential(
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, action_dim, kernel_size=1),
            nn.Tanh(),
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.action_global_mean.copy_(mean.to(self.action_global_mean.device))
        self.action_global_std.copy_(std.to(self.action_global_std.device))
        self._action_stats_set = True

    def encode(self, action: torch.Tensor):
        x = action.float().permute(0, 2, 1).contiguous()  # (B, action_dim, T)
        h = self.enc_input_proj(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_norm(h).mean(dim=2)                   # global avg pool → (B, hidden)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, text_emb: torch.Tensor):
        B = z.size(0)
        if self.training and self.dropout > 0.0 and torch.rand(1).item() < self.dropout:
            text_emb = torch.zeros_like(text_emb)
        h = self.dec_input_proj(torch.cat([z, text_emb], dim=-1))
        h = h.view(B, self.hidden_channels, self.chunk_size)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out_proj(h).permute(0, 2, 1).contiguous()  # (B, T, action_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, action: torch.Tensor, text_emb: torch.Tensor):
        mu, logvar   = self.encode(action)
        z            = self.reparameterize(mu, logvar)
        action_recon = self.decode(z, text_emb)
        return action_recon, mu, logvar, z

    def compute_loss_tcvae(self, action, action_recon, mu, logvar, z,
                     beta=6.0, alpha=1.0, recon_weight=50.0, vel_weight=0.0):
        """Original Beta-TCVAE loss: MI + beta*TC + DW-KL + guidance + velocity.
        Kept for reference / backwards compatibility."""
        batch_size, latent_dim = z.shape

        pred_continuous   = action_recon[..., :6]
        gt_continuous     = action[..., :6]
        pred_gripper      = action_recon[..., 6]
        gt_gripper        = action[..., 6]

        gt_gripper_binary = (gt_gripper > 0.0).float()
        pred_gripper_prob = torch.clamp((pred_gripper + 1.0) / 2.0, min=1e-6, max=1.0 - 1e-6)
        loss_gripper      = F.binary_cross_entropy(pred_gripper_prob, gt_gripper_binary,
                                                   reduction='none').sum(dim=1).mean()
        loss_continuous   = F.mse_loss(pred_continuous, gt_continuous,
                                       reduction='none').sum(dim=(1, 2)).mean()
        recon_loss        = (loss_continuous + 0.5 * loss_gripper) * recon_weight

        mu_expand     = mu.unsqueeze(0)
        logvar_expand = logvar.unsqueeze(0)
        z_expand      = z.unsqueeze(1)
        log_q_z_given_x = -0.5 * (math.log(2 * math.pi) + logvar_expand +
                                   (z_expand - mu_expand).pow(2) / torch.exp(logvar_expand))
        log_q_z        = torch.logsumexp(log_q_z_given_x.sum(dim=2), dim=1) - math.log(batch_size)
        log_prod_q_z_d = (torch.logsumexp(log_q_z_given_x, dim=1) - math.log(batch_size)).sum(dim=1)
        log_p_z        = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)

        mi_loss    = (log_q_z_given_x.sum(dim=2).diagonal() - log_q_z).mean()
        tc_loss    = (log_q_z - log_prod_q_z_d).mean()
        dw_kl_loss = (log_prod_q_z_d - log_p_z).mean()
        tcvae_loss = recon_loss + mi_loss + beta * tc_loss + dw_kl_loss

        target_kinematics = action.mean(dim=1)
        if alpha > 0.0:
            if not self._action_stats_set:
                raise RuntimeError(
                    "Guidance loss is active (alpha > 0) but action stats have not been set. "
                    "Call model.set_action_stats(mean, std) before training."
                )
            target_kinematics = (target_kinematics - self.action_global_mean) / (self.action_global_std + 1e-5)
            loss_guidance = F.mse_loss(mu[:, :self.action_dim], target_kinematics)
        else:
            loss_guidance = torch.tensor(0.0, device=mu.device)

        if vel_weight > 0.0 and action.size(1) > 1:
            vel_gt    = action[:, 1:, :6]       - action[:, :-1, :6]
            vel_recon = action_recon[:, 1:, :6] - action_recon[:, :-1, :6]
            loss_vel  = vel_weight * F.mse_loss(vel_recon, vel_gt,
                                                reduction='none').sum(dim=(1, 2)).mean()
        else:
            loss_vel = torch.tensor(0.0, device=action.device)

        total_loss = tcvae_loss + alpha * loss_guidance + loss_vel
        return total_loss, recon_loss, loss_continuous, loss_gripper, mi_loss, tc_loss, loss_guidance, loss_vel

    def compute_loss(self, action, action_recon, mu, logvar, beta=None, recon_weight=100.0):
        """Simplified standard beta-CVAE loss: MSE + BCE + beta * KL.

        Drops the O(B^2) TCVAE decomposition, the kinematic guidance term, and
        the velocity penalty — leaving a clean two-term objective that prioritises
        reconstruction while keeping the latent space smooth.

        Args:
            action:       (B, T, action_dim) normalised ground-truth chunk.
            action_recon: (B, T, action_dim) reconstructed chunk.
            mu:           (B, latent_dim) posterior mean.
            logvar:       (B, latent_dim) posterior log-variance.
            beta:         KL weight (defaults to self.beta, typically 0.1).
            recon_weight: scales the reconstruction term (default 100).

        Returns:
            total_loss, recon_loss, loss_continuous, loss_gripper, kl_loss
        """
        if beta is None:
            beta = self.beta

        pred_continuous = action_recon[..., :6]
        gt_continuous   = action[..., :6]
        pred_gripper    = action_recon[..., 6]
        gt_gripper      = action[..., 6]

        gt_gripper_binary = (gt_gripper > 0.0).float()
        pred_gripper_prob = torch.clamp((pred_gripper + 1.0) / 2.0, min=1e-6, max=1.0 - 1e-6)
        loss_gripper    = F.binary_cross_entropy(pred_gripper_prob, gt_gripper_binary,
                                                 reduction='none').sum(dim=1).mean()
        loss_continuous = F.mse_loss(pred_continuous, gt_continuous,
                                     reduction='none').sum(dim=(1, 2)).mean()
        recon_loss      = (loss_continuous + 0.5 * loss_gripper) * recon_weight

        # Closed-form KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, loss_continuous, loss_gripper, kl_loss


class TCNTextActionCVAE(nn.Module):
    """
    Full CVAE: text conditions BOTH encoder and decoder.

    Encoder:  action → TCN → global_avg_pool → fused with text → mu / logvar
    Decoder:  (z + text) → Linear → TCN → action_recon

    Text fusion in the encoder uses an additive projection with a learned soft gate
    (scalar initialised at 0) so the model starts as a decoder-only VAE and
    gradually opens the encoder conditioning as training progresses.  This prevents
    the text from trivially "taking over" the latent code on day one.

    The encoder also applies the same classifier-free guidance dropout to text
    (probability `dropout`) so that z retains marginal trajectory information even
    when text is masked — keeping the latent space useful for the downstream
    projector that has no text at inference.

    Args:
        enc_text_gate_init: initial value of the learnable text gate in the encoder.
                            0.0 = fully closed (pure trajectory VAE at init).
                            Grows during training via gradient.
    """

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 8,
        latent_dim: int = 64,
        text_emb_dim: int = 512,
        beta: float = 0.1,
        dropout: float = 0.0,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 3,
        block_dropout: float = 0.0,
        enc_text_gate_init: float = 0.0,
    ):
        super().__init__()
        self.latent_dim      = latent_dim
        self.chunk_size      = chunk_size
        self.action_dim      = action_dim
        self.text_emb_dim    = text_emb_dim
        self.beta            = beta
        self.dropout         = dropout
        self.hidden_channels = hidden_channels

        self.register_buffer('action_global_mean', torch.zeros(action_dim))
        self.register_buffer('action_global_std',  torch.ones(action_dim))
        self._action_stats_set = False

        num_groups = 8
        while hidden_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        # ── ENCODER ─────────────────────────────────────────────────────────
        self.enc_input_proj = nn.Conv1d(action_dim, hidden_channels, kernel_size=1)
        self.enc_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=block_dropout)
            for i in range(n_blocks)
        ])
        self.enc_norm = nn.Sequential(
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
        )

        # Text conditioning in encoder: project text → hidden_channels, add to pooled features.
        # Soft gate (scalar, init=0) prevents text from dominating at init.
        self.enc_text_proj = nn.Linear(text_emb_dim, hidden_channels)
        self.enc_text_gate = nn.Parameter(torch.tensor(enc_text_gate_init))

        self.fc_mu     = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

        # ── DECODER ─────────────────────────────────────────────────────────
        self.dec_input_proj = nn.Sequential(
            nn.Linear(latent_dim + text_emb_dim, hidden_channels * chunk_size),
            nn.GELU(),
        )
        self.dec_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=block_dropout)
            for i in range(n_blocks)
        ])
        self.dec_out_proj = nn.Sequential(
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, action_dim, kernel_size=1),
            nn.Tanh(),
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.action_global_mean.copy_(mean.to(self.action_global_mean.device))
        self.action_global_std.copy_(std.to(self.action_global_std.device))
        self._action_stats_set = True

    def encode(self, action: torch.Tensor, text_emb: torch.Tensor):
        """Encode action chunk conditioned on text embedding."""
        x = action.float().permute(0, 2, 1).contiguous()  # (B, action_dim, T)
        h = self.enc_input_proj(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_norm(h).mean(dim=2)                   # global avg pool → (B, hidden)

        # Classifier-free guidance dropout: randomly zero text in encoder during training.
        if self.training and self.dropout > 0.0 and torch.rand(1).item() < self.dropout:
            text_emb_enc = torch.zeros_like(text_emb)
        else:
            text_emb_enc = text_emb

        # Soft-gated additive text fusion.
        # gate = sigmoid(scalar) so it stays in [0, 1] and can't blow up.
        gate = torch.sigmoid(self.enc_text_gate)
        h = h + gate * self.enc_text_proj(text_emb_enc)

        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, text_emb: torch.Tensor):
        B = z.size(0)
        if self.training and self.dropout > 0.0 and torch.rand(1).item() < self.dropout:
            text_emb = torch.zeros_like(text_emb)
        h = self.dec_input_proj(torch.cat([z, text_emb], dim=-1))
        h = h.view(B, self.hidden_channels, self.chunk_size)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out_proj(h).permute(0, 2, 1).contiguous()  # (B, T, action_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, action: torch.Tensor, text_emb: torch.Tensor):
        mu, logvar   = self.encode(action, text_emb)
        z            = self.reparameterize(mu, logvar)
        action_recon = self.decode(z, text_emb)
        return action_recon, mu, logvar, z

    def compute_loss_tcvae(self, action, action_recon, mu, logvar, z,
                     beta=0.1, alpha=1.0, recon_weight=100.0, vel_weight=0.0):
        """Original Beta-TCVAE loss: MI + beta*TC + DW-KL + guidance + velocity.
        Kept for reference / backwards compatibility."""
        batch_size, latent_dim = z.shape

        pred_continuous = action_recon[..., :6]
        gt_continuous   = action[..., :6]
        pred_gripper    = action_recon[..., 6]
        gt_gripper      = action[..., 6]

        gt_gripper_binary = (gt_gripper > 0.0).float()
        pred_gripper_prob = torch.clamp((pred_gripper + 1.0) / 2.0, min=1e-6, max=1.0 - 1e-6)
        loss_gripper      = F.binary_cross_entropy(pred_gripper_prob, gt_gripper_binary,
                                                   reduction='none').sum(dim=1).mean()
        loss_continuous   = F.mse_loss(pred_continuous, gt_continuous,
                                       reduction='none').sum(dim=(1, 2)).mean()
        recon_loss        = (loss_continuous + 0.5 * loss_gripper) * recon_weight

        mu_expand     = mu.unsqueeze(0)
        logvar_expand = logvar.unsqueeze(0)
        z_expand      = z.unsqueeze(1)
        log_q_z_given_x = -0.5 * (math.log(2 * math.pi) + logvar_expand +
                                   (z_expand - mu_expand).pow(2) / torch.exp(logvar_expand))
        log_q_z        = torch.logsumexp(log_q_z_given_x.sum(dim=2), dim=1) - math.log(batch_size)
        log_prod_q_z_d = (torch.logsumexp(log_q_z_given_x, dim=1) - math.log(batch_size)).sum(dim=1)
        log_p_z        = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)

        mi_loss    = (log_q_z_given_x.sum(dim=2).diagonal() - log_q_z).mean()
        tc_loss    = (log_q_z - log_prod_q_z_d).mean()
        dw_kl_loss = (log_prod_q_z_d - log_p_z).mean()
        tcvae_loss = recon_loss + mi_loss + beta * tc_loss + dw_kl_loss

        target_kinematics = action.mean(dim=1)
        if alpha > 0.0:
            if not self._action_stats_set:
                raise RuntimeError(
                    "Guidance loss is active (alpha > 0) but action stats have not been set. "
                    "Call model.set_action_stats(mean, std) before training."
                )
            target_kinematics = (target_kinematics - self.action_global_mean) / (self.action_global_std + 1e-5)
            loss_guidance = F.mse_loss(mu[:, :self.action_dim], target_kinematics)
        else:
            loss_guidance = torch.tensor(0.0, device=mu.device)

        if vel_weight > 0.0 and action.size(1) > 1:
            vel_gt    = action[:, 1:, :6]       - action[:, :-1, :6]
            vel_recon = action_recon[:, 1:, :6] - action_recon[:, :-1, :6]
            loss_vel  = vel_weight * F.mse_loss(vel_recon, vel_gt,
                                                reduction='none').sum(dim=(1, 2)).mean()
        else:
            loss_vel = torch.tensor(0.0, device=action.device)

        total_loss = tcvae_loss + alpha * loss_guidance + loss_vel
        return total_loss, recon_loss, loss_continuous, loss_gripper, mi_loss, tc_loss, loss_guidance, loss_vel

    def compute_loss(self, action, action_recon, mu, logvar, beta=None, recon_weight=100.0):
        """Simplified standard beta-CVAE loss: MSE + BCE + beta * KL.

        Drops the O(B^2) TCVAE decomposition, the kinematic guidance term, and
        the velocity penalty — leaving a clean two-term objective that prioritises
        reconstruction while keeping the latent space smooth.

        Args:
            action:       (B, T, action_dim) normalised ground-truth chunk.
            action_recon: (B, T, action_dim) reconstructed chunk.
            mu:           (B, latent_dim) posterior mean.
            logvar:       (B, latent_dim) posterior log-variance.
            beta:         KL weight (defaults to self.beta, typically 0.1).
            recon_weight: scales the reconstruction term (default 100).

        Returns:
            total_loss, recon_loss, loss_continuous, loss_gripper, kl_loss
        """
        if beta is None:
            beta = self.beta

        pred_continuous = action_recon[..., :6]
        gt_continuous   = action[..., :6]
        pred_gripper    = action_recon[..., 6]
        gt_gripper      = action[..., 6]

        gt_gripper_binary = (gt_gripper > 0.0).float()
        pred_gripper_prob = torch.clamp((pred_gripper + 1.0) / 2.0, min=1e-6, max=1.0 - 1e-6)
        loss_gripper    = F.binary_cross_entropy(pred_gripper_prob, gt_gripper_binary,
                                                 reduction='none').sum(dim=1).mean()
        loss_continuous = F.mse_loss(pred_continuous, gt_continuous,
                                     reduction='none').sum(dim=(1, 2)).mean()
        recon_loss      = (loss_continuous + 0.5 * loss_gripper) * recon_weight

        # Closed-form KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, loss_continuous, loss_gripper, kl_loss

