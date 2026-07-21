import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = get_1d_sincos_pos_embed_from_grid(self.frequency_embedding_size, t)
        t_freq = t_freq.to(t.device)
        t_emb = self.mlp(t_freq)
        return t_emb

class FlowTransformerProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 16,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Project flat or sequence VLA embedding to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embeddings for spatial tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Learnable queries — one per latent dimension.
        self.latent_queries = nn.Parameter(
            torch.empty(latent_dim, d_model)
        )
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        # Time conditioning
        self.t_embedder = TimestepEmbedder(d_model)

        # z_t conditioning: project each scalar z_t dimension to d_model
        self.z_proj = nn.Linear(1, d_model)

        # Stack of cross-attention blocks
        self.layers = nn.ModuleList([
            _CrossAttentionBlock(d_model=d_model, num_heads=num_heads,
                                 ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output head: [B, Z_dim, d_model] → [B, Z_dim] predicting the vector field v
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: (B, input_dim) — VLA embedding.
            z_t: (B, Z_dim) — noisy latent at time t.
            t: (B,) — timesteps in [0, 1].

        Returns:
            v: (B, Z_dim) predicted vector field.
        """
        B = x.size(0)

        # Project input to d_model
        context = self.input_proj(x)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        
        seq_len = context.size(1)
        context = context + self.pos_emb[:, :seq_len, :]

        # Base Queries
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, Z_dim, d_model)

        # Time embedding
        t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, d_model)

        # z_t embedding
        z_emb = self.z_proj(z_t.unsqueeze(-1))  # (B, Z_dim, d_model)

        # Add conditionings to queries
        queries = queries + z_emb + t_emb

        # Cross-attend
        for layer in self.layers:
            queries = layer(queries, context)

        # Predict vector field
        v = self.fc_out(queries).squeeze(-1)  # (B, Z_dim)
        return v

class _CrossAttentionBlock(nn.Module):
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
        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries
