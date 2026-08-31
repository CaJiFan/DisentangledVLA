import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# We can reuse the DilatedResBlock from the original file
from src.disentanglers.TextActionDecOnlyBetaTCVAE import DilatedResBlock

class TCNTextCondPriorCVAE(nn.Module):
    """
    TCN Decoder-Only CVAE but with a Learned Conditional Prior.
    Instead of z ~ N(0, 1), the prior is p(z | text) = N(mu_p, logvar_p).
    The encoder is text-free: q(z | action) = N(mu_q, logvar_q).
    The KL divergence is calculated between q(z | action) and p(z | text).
    """
    def __init__(self, action_dim=7, chunk_size=8, latent_dim=128, text_emb_dim=512,
                 beta=0.1, dropout=0.15, hidden_channels=64, n_blocks=4,
                 prior_hidden_dim=512, use_state=False, state_dim=8):
        super().__init__()
        self.use_state = use_state
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.text_emb_dim = text_emb_dim
        self.beta = beta
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks

        self.register_buffer('action_global_mean', torch.zeros(action_dim))
        self.register_buffer('action_global_std', torch.ones(action_dim))
        self._action_stats_set = False

        # ── ENCODER (Text-Free) ─────────────────────────────────────────────
        self.enc_input_proj = nn.Conv1d(action_dim, hidden_channels, kernel_size=1)
        # RF must cover chunk_size
        kernel_size = 3
        self.enc_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.enc_norm = nn.Sequential(nn.GroupNorm(8, hidden_channels), nn.GELU())
        self.fc_mu_q = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar_q = nn.Linear(hidden_channels, latent_dim)

        # Attention pooling: a learned CLS-like query cross-attends over the T timestep
        # features, replacing global avg pool. Lets the encoder focus on task-critical
        # timesteps (e.g. grasp approach) instead of weighting all frames equally.
        _n_heads = 4
        while hidden_channels % _n_heads != 0:
            _n_heads -= 1
        self.enc_attn_query = nn.Parameter(torch.randn(1, 1, hidden_channels) * 0.02)
        self.enc_pool_attn  = nn.MultiheadAttention(
            hidden_channels, num_heads=_n_heads, dropout=0.0, batch_first=True
        )

        # ── PRIOR NETWORK ───────────────────────────────────────────────────
        # Learns p(z | text, state). Uses prior_hidden_dim (default 512) for 3-layer MLP
        # so it has enough capacity to place 10+ object-type modes in distinct regions.
        prior_in_dim = text_emb_dim + state_dim if use_state else text_emb_dim
        self.prior_net = nn.Sequential(
            nn.Linear(prior_in_dim, prior_hidden_dim),
            nn.GELU(),
            nn.Linear(prior_hidden_dim, prior_hidden_dim),
            nn.GELU(),
            nn.Linear(prior_hidden_dim, hidden_channels),
            nn.GELU(),
        )
        self.fc_mu_p = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar_p = nn.Linear(hidden_channels, latent_dim)

        # ── DECODER ─────────────────────────────────────────────────────────
        dec_in_dim = latent_dim + text_emb_dim + state_dim if use_state else latent_dim + text_emb_dim
        self.dec_input_proj = nn.Sequential(
            nn.Linear(dec_in_dim, hidden_channels * chunk_size),
            nn.GELU(),
        )
        self.dec_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.dec_out_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, action_dim, kernel_size=1),
            nn.Tanh(),
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.action_global_mean.copy_(mean.to(self.action_global_mean.device))
        self.action_global_std.copy_(std.to(self.action_global_std.device))
        self._action_stats_set = True

    def get_prior(self, text_emb: torch.Tensor, state: torch.Tensor = None):
        """Compute the text-conditioned prior p(z|c, s)."""
        if self.use_state:
            assert state is not None, "State must be provided when use_state is True"
            B = text_emb.size(0)
            if len(state.shape) == 1:
                state = state.unsqueeze(0).expand(B, -1)
            elif state.size(0) == 1 and B > 1:
                state = state.expand(B, -1)
            x = torch.cat([text_emb, state], dim=-1)
        else:
            x = text_emb
        h_p = self.prior_net(x)
        mu_p = self.fc_mu_p(h_p)
        logvar_p = self.fc_logvar_p(h_p)
        # Bound logvar to prevent extreme values during early training
        logvar_p = torch.clamp(logvar_p, min=-10.0, max=10.0)
        return mu_p, logvar_p

    def encode(self, action: torch.Tensor):
        x = action.float().permute(0, 2, 1).contiguous()  # (B, action_dim, T)
        h = self.enc_input_proj(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_norm(h)                               # (B, hidden, T)
        # Cross-attention pooling: the learned query attends over all T timesteps.
        # Q = learned CLS token, K = V = temporal feature sequence.
        # This is cross-attention (Q from a different source than K/V).
        h_seq = h.permute(0, 2, 1)                         # (B, T, hidden)
        query = self.enc_attn_query.expand(h_seq.size(0), -1, -1)   # (B, 1, hidden)
        h_pooled, _ = self.enc_pool_attn(query, h_seq, h_seq)        # (B, 1, hidden)
        h_pooled = h_pooled.squeeze(1)                               # (B, hidden)
        mu_q = self.fc_mu_q(h_pooled)
        logvar_q = self.fc_logvar_q(h_pooled)
        return mu_q, logvar_q

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def decode(self, z: torch.Tensor, text_emb: torch.Tensor, state: torch.Tensor = None):
        B = z.size(0)
        if self.training and self.dropout > 0.0 and torch.rand(1).item() < self.dropout:
            text_emb = torch.zeros_like(text_emb)
        
        if self.use_state:
            assert state is not None, "State must be provided when use_state is True"
            if len(state.shape) == 1:
                state = state.unsqueeze(0).expand(B, -1)
            elif state.size(0) == 1 and B > 1:
                state = state.expand(B, -1)
            x = torch.cat([z, text_emb, state], dim=-1)
        else:
            x = torch.cat([z, text_emb], dim=-1)
        h = self.dec_input_proj(x)
        h = h.view(B, self.hidden_channels, self.chunk_size)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out_proj(h).permute(0, 2, 1).contiguous()

    def forward(self, action: torch.Tensor, text_emb: torch.Tensor, state: torch.Tensor = None):
        mu_q, logvar_q = self.encode(action)
        mu_p, logvar_p = self.get_prior(text_emb, state)
        z = self.reparameterize(mu_q, logvar_q)
        action_recon = self.decode(z, text_emb, state)
        return action_recon, mu_q, logvar_q, mu_p, logvar_p, z

    def compute_loss(self, action, action_recon, mu_q, logvar_q, mu_p, logvar_p, beta=None, recon_weight=100.0, gripper_weight=5.0):
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
        recon_loss      = (loss_continuous + gripper_weight * loss_gripper) * recon_weight

        # Closed-form KL(q || p) for diagonal gaussians
        # KL = 0.5 * sum(logvar_p - logvar_q - 1 + (exp(logvar_q) + (mu_q - mu_p)^2) / exp(logvar_p))
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl_loss = 0.5 * (logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p).pow(2)) / var_p).sum(dim=1).mean()

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, loss_continuous, loss_gripper, kl_loss


class TCNTextWAE(nn.Module):
    """
    Wasserstein Autoencoder (WAE) with TCN Decoder-Only conditioning.
    Replaces KL divergence with an MMD (Maximum Mean Discrepancy) penalty.
    The encoder can be deterministic (z = mu).
    """
    def __init__(self, action_dim=7, chunk_size=8, latent_dim=128, text_emb_dim=512, 
                 beta=0.1, dropout=0.15, hidden_channels=64, n_blocks=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.text_emb_dim = text_emb_dim
        self.beta = beta  # Acts as the MMD penalty weight (lambda)
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks

        self.register_buffer('action_global_mean', torch.zeros(action_dim))
        self.register_buffer('action_global_std', torch.ones(action_dim))
        self._action_stats_set = False

        # ── ENCODER ─────────────────────────────────────────────────────────
        self.enc_input_proj = nn.Conv1d(action_dim, hidden_channels, kernel_size=1)
        kernel_size = 3
        self.enc_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.enc_norm = nn.GroupNorm(8, hidden_channels)
        
        # Deterministic encoder, only outputs z (mu)
        self.fc_z = nn.Linear(hidden_channels, latent_dim)

        # ── DECODER ─────────────────────────────────────────────────────────
        self.dec_input_proj = nn.Sequential(
            nn.Linear(latent_dim + text_emb_dim, hidden_channels * chunk_size),
            nn.GELU(),
        )
        self.dec_blocks = nn.ModuleList([
            DilatedResBlock(hidden_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.dec_out_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, action_dim, kernel_size=1),
            nn.Tanh(),
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.action_global_mean.copy_(mean.to(self.action_global_mean.device))
        self.action_global_std.copy_(std.to(self.action_global_std.device))
        self._action_stats_set = True

    def encode(self, action: torch.Tensor):
        x = action.float().permute(0, 2, 1).contiguous()
        h = self.enc_input_proj(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_norm(h).mean(dim=2)
        z = self.fc_z(h)
        # Return dummy logvar to remain drop-in compatible with log_video_probe
        return z, torch.zeros_like(z)

    def decode(self, z: torch.Tensor, text_emb: torch.Tensor):
        B = z.size(0)
        if self.training and self.dropout > 0.0 and torch.rand(1).item() < self.dropout:
            text_emb = torch.zeros_like(text_emb)
            
        h = self.dec_input_proj(torch.cat([z, text_emb], dim=-1))
        h = h.view(B, self.hidden_channels, self.chunk_size)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out_proj(h).permute(0, 2, 1).contiguous()

    def forward(self, action: torch.Tensor, text_emb: torch.Tensor):
        z, _ = self.encode(action)
        # Optional: Add small noise to z during training (makes it stochastic WAE)
        # For pure deterministic WAE, we skip this.
        action_recon = self.decode(z, text_emb)
        return action_recon, z

    def compute_mmd(self, z, z_prior):
        """
        Computes Maximum Mean Discrepancy (MMD) using Inverse Multi-Quadric kernel.
        k(x, y) = C / (C + ||x - y||^2)
        """
        C = 2.0 * self.latent_dim  # Kernel width heuristic
        
        def compute_kernel(x, y):
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
            x = x.unsqueeze(1) # (B, 1, dim)
            y = y.unsqueeze(0) # (1, B, dim)
            dist = torch.sum((x - y) ** 2, dim=2)
            return C / (C + dist)
            
        k_xx = compute_kernel(z, z).mean()
        k_yy = compute_kernel(z_prior, z_prior).mean()
        k_xy = compute_kernel(z, z_prior).mean()
        return k_xx + k_yy - 2 * k_xy

    def compute_loss(self, action, action_recon, z, beta=None, recon_weight=100.0):
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

        # WAE penalty: MMD between z and samples from N(0, 1)
        z_prior = torch.randn_like(z)
        mmd_loss = self.compute_mmd(z, z_prior)
        
        # MMD values are usually smaller than KL sum over dims, so we scale it up
        # to have a similar effect size to KL before applying beta.
        scaled_mmd = mmd_loss * self.latent_dim

        total_loss = recon_loss + beta * scaled_mmd
        return total_loss, recon_loss, loss_continuous, loss_gripper, scaled_mmd
