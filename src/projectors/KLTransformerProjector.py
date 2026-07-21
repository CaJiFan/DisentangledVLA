import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLTransformerProjector(nn.Module):
    """
    Transformer-based projector that maps a flat VLA embedding to a Gaussian
    distribution in the Disentangled Latent Space.

    Key architectural differences vs. ProbabilisticActionProjector (MLP):

    1. **Per-dimension latent queries**: One learnable query vector per latent
       dimension (Z_dim). Each query specialises in attending to the VLA embedding
       to predict *one* kinematic factor — an inductive bias that directly mirrors
       the disentanglement philosophy of the β-TCVAE.

    2. **Cross-attention over VLA tokens**: The flat VLA embedding is projected to
       a single-token "context" sequence [B, 1, d_model]. The Z_dim queries cross-
       attend over this context through `num_layers` stacked blocks. Because the
       input is treated as a length-1 sequence, no cache changes are required.

    3. **Residual + LayerNorm**: Smoother loss landscape than deep MLP funnels,
       which helps with the high-variance regimes observed on SmolVLA / OpenVLA.

    API is identical to ProbabilisticActionProjector:
        forward(x) → (dist, mu, logvar)

    Args:
        input_dim:  VLA embedding dimension (e.g. 768 for Octo, 4096 for OpenVLA).
        latent_dim: Dimensionality of the β-TCVAE latent space (Z_dim).
        d_model:    Internal transformer hidden dimension.
        num_heads:  Number of attention heads in each cross-attention block.
        num_layers: Number of stacked cross-attention + FFN blocks.
        ffn_dim:    Inner dimension of the position-wise FFN.
        dropout:    Dropout rate applied inside FFN and attention.
        min_std:    Numerical stability floor added to predicted std.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 16,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        min_std: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.min_std = min_std

        # Project flat or sequence VLA embedding to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embeddings for spatial tokens (up to 512 length)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Learnable queries — one per latent dimension.
        # Shape: [Z_dim, d_model]; expanded to [B, Z_dim, d_model] at forward time.
        self.latent_queries = nn.Parameter(
            torch.empty(latent_dim, d_model)
        )
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        # Stack of cross-attention blocks
        self.layers = nn.ModuleList([
            _CrossAttentionBlock(d_model=d_model, num_heads=num_heads,
                                 ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output heads: [B, Z_dim, d_model] → [B, Z_dim]
        self.fc_mu     = nn.Linear(d_model, 1)
        self.fc_logvar = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_dim) — VLA embedding (flat vector, same format as MLP).

        Returns:
            dist:   torch.distributions.Independent(Normal) over (B, Z_dim).
            mu:     (B, Z_dim) predicted mean.
            logvar: (B, Z_dim) predicted log-variance, clamped to [-10, 2].
        """
        B = x.size(0)

        # Project input to d_model
        context = self.input_proj(x)  # (B, input_dim) or (B, seq_len, input_dim)

        # If flat (e.g. layer -1), unsqueeze to length 1.
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, d_model)
        
        # Add positional encoding
        seq_len = context.size(1)
        context = context + self.pos_emb[:, :seq_len, :]

        # Queries: expand once to [B, Z_dim, d_model]
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, Z_dim, d_model)

        # Cross-attend: queries attend over the single VLA context token
        for layer in self.layers:
            queries = layer(queries, context)  # (B, Z_dim, d_model)

        # Per-dimension output heads → squeeze last dim
        mu     = self.fc_mu(queries).squeeze(-1)      # (B, Z_dim)
        logvar = self.fc_logvar(queries).squeeze(-1)  # (B, Z_dim)

        # Clamp logvar: range [-10, 2] → std ∈ [0.007, 2.72]
        logvar = torch.clamp(logvar, min=-10, max=2)

        std  = torch.exp(0.5 * logvar) + self.min_std
        dist = torch.distributions.Independent(
            torch.distributions.Normal(mu, std),
            reinterpreted_batch_ndims=1,
        )
        return dist, mu, logvar

    def loss(self, predicted_dist, target_z_from_vae: torch.Tensor) -> torch.Tensor:
        """NLL loss — kept for API parity with ProbabilisticActionProjector."""
        return -predicted_dist.log_prob(target_z_from_vae).mean()


class _CrossAttentionBlock(nn.Module):
    """
    One cross-attention block:
        Q = queries [B, Z_dim, d_model]
        K = V = context [B, T_ctx, d_model]   (T_ctx=1 for the flat-vector case)

        queries ← LayerNorm(queries + CrossAttn(Q, K, V))
        queries ← LayerNorm(queries + FFN(queries))
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention: queries attend over context
        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        queries = self.norm1(queries + attn_out)

        # Position-wise FFN
        queries = self.norm2(queries + self.ffn(queries))
        return queries
