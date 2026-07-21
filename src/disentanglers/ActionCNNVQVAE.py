import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # z: (B, Embedding_Dim, T) -> Need (B, T, Embedding_Dim) for quantization usually, 
        # or we flatten. Let's quantize per timestamp or per vector?
        # Standard VQ-VAE for 1D signals usually preserves the temporal dim T.
        
        # Input z is (B, Ch, T). Permute to (B, T, Ch)
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim) # (B*T, Ch)
        
        # Distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Loss
        loss = self.beta * F.mse_loss(z.detach(), z_q) + F.mse_loss(z_q.detach(), z)
        
        # Straight-through
        z_q = z + (z_q - z).detach()
        
        # Permute back to (B, Ch, T)
        z_q = z_q.permute(0, 2, 1).contiguous()
        
        return z_q, loss, min_encoding_indices

class ActionVQVAE(nn.Module):
    def __init__(self, 
                 action_dim=7, 
                 horizon=16, # Unused but kept for API compatibility
                 embed_dim=16, # Match the latent dim of your TCVAE for fair comparison
                 num_codes=1024):
        super().__init__()
        
        # --- Encoder (Same as Beta-TCVAE) ---
        # 7 -> 32 -> 64 -> 128 (embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=4, stride=2, padding=1),
            # No ReLU at the end, let the VQ layer handle the raw space
        )

        # --- Quantizer ---
        self.quantizer = VectorQuantizer(num_codes, embed_dim)

        # --- Decoder (Same as Beta-TCVAE) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, action_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, actions):
        # Input: (B, T, D) -> Permute to (B, D, T) for Conv1d
        x = actions.permute(0, 2, 1)
        
        # Encode -> (B, Embed_Dim, T_Reduced)
        # With 3 stride-2 layers, T=16 becomes T=2.
        z = self.encoder(x)
        
        # Quantize
        z_q, vq_loss, indices = self.quantizer(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # Permute back to (B, T, D)
        x_recon = x_recon.permute(0, 2, 1)
        
        return x_recon, vq_loss, indices