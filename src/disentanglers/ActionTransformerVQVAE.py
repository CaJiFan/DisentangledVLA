import torch
import torch.nn as nn
import torch.nn.functional as F


# The Action-Only VQ-VAE Model
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # z: (B, D)
        # 1. Find nearest neighbors
        # Dist = ||z||^2 + ||e||^2 - 2*z*e
        z_flat = z
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flat, self.embedding.weight.t())
            
        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices) # (B, D)

        # 2. Loss
        loss = self.beta * F.mse_loss(z.detach(), z_q) + F.mse_loss(z_q.detach(), z)
        
        # 3. Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, indices

class ActionVQVAE(nn.Module):
    def __init__(self, 
                 action_dim=7, 
                 horizon=32, 
                 embed_dim=256, 
                 num_codes=1024,
                 hidden_dim=512):
        super().__init__()
        self.horizon = horizon
        
        # --- Encoder (Actions -> Latent) ---
        # Projects flattened actions to hidden dim
        self.enc_input_proj = nn.Linear(action_dim, hidden_dim)
        
        # Transformer Encoder to capture temporal dependencies
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.1)
        self.encoder_tf = nn.TransformerEncoder(enc_layer, num_layers=4)
        
        # Project pooled output to latent dim
        self.enc_head = nn.Linear(hidden_dim, embed_dim)

        # --- Quantizer ---
        self.quantizer = VectorQuantizer(num_codes, embed_dim)

        # --- Decoder (Latent -> Actions) ---
        # Project latent back to hidden dim
        self.dec_input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Learnable queries for each timestep (0 to 31)
        self.time_queries = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        
        # Transformer Decoder
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.1)
        self.decoder_tf = nn.TransformerDecoder(dec_layer, num_layers=4)
        
        # Final prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode(self, actions):
        """
        Input: (B, Horizon, Action_Dim)
        Output: (B, Embed_Dim)
        """
        # 1. Project inputs
        x = self.enc_input_proj(actions) # (B, H, Hidden)
        
        # 2. Add Positional Encodings (Simple sinusoidal or learned can be added here)
        # For simplicity, we rely on the Transformer learning relative positions or adding a fixed embedding
        
        # 3. Transformer Pass
        x = self.encoder_tf(x)
        
        # 4. Pooling (Mean pooling over the time dimension to get one vector)
        z = x.mean(dim=1) 
        z = self.enc_head(z)
        return z

    def decode(self, z_q):
        """
        Input: (B, Embed_Dim)
        Output: (B, Horizon, Action_Dim)
        """
        B = z_q.shape[0]
        
        # 1. Prepare Memory (Latent Code)
        memory = self.dec_input_proj(z_q).unsqueeze(1) # (B, 1, Hidden)
        
        # 2. Prepare Target (Time Queries)
        tgt = self.time_queries.repeat(B, 1, 1) # (B, H, Hidden)
        
        # 3. Transformer Pass
        out = self.decoder_tf(tgt, memory)
        
        # 4. Predict
        return self.action_head(out)

    def forward(self, actions):
        z = self.encode(actions)
        z_q, vq_loss, indices = self.quantizer(z)
        pred_actions = self.decode(z_q)
        return pred_actions, vq_loss, indices