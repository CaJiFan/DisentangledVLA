import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextActionBetaTCVAE(nn.Module):
    def __init__(self, action_dim=7, chunk_size=16, latent_dim=16, text_emb_dim=512, beta=6.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.text_emb_dim = text_emb_dim
        self.beta = beta  

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # After Convs and Flatten: shape is (Batch, 128 * 2) = (Batch, 256)
        # We concatenate the Text Embedding (512) here!
        # 256 + 512 = 768
        self.fc_mu = nn.Linear(256 + self.text_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 + self.text_emb_dim, latent_dim)

        # 2. DECODER
        # Map the concatenated (z + text) back to the spatial conv dimensions
        self.decoder_input = nn.Linear(latent_dim + self.text_emb_dim, 256)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, action_dim, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, action, text_emb):
        # action shape: (B, 16, 7) -> Permute to (B, Channels, Length) -> (B, 7, 16)
        x = action.permute(0, 2, 1)
        
        # Extract temporal features
        h = self.encoder_conv(x) # Output shape: (B, 256)
        
        # FUSE WITH TEXT: Concatenate along feature dimension
        fused_h = torch.cat([h, text_emb], dim=-1) # Output shape: (B, 768)
        
        mu = self.fc_mu(fused_h)
        logvar = self.fc_logvar(fused_h)
        return mu, logvar

    def decode(self, z, text_emb):
        # 1. FUSE WITH TEXT
        fused_z = torch.cat([z, text_emb], dim=-1) # Shape: (B, 16 + 512)
        
        # 2. Project back to convolutional space
        h = self.decoder_input(fused_z)            # Shape: (B, 256)
        h = h.view(-1, 128, 2)                     # Reshape to (B, Channels, Length)
        
        # 3. Transposed Convolutions
        x_recon = self.decoder_conv(h)             # Shape: (B, 7, 16)
        
        # 4. Permute back to standard format
        return x_recon.permute(0, 2, 1)            # Shape: (B, 16, 7)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, action, text_emb):
        mu, logvar = self.encode(action, text_emb)
        z = self.reparameterize(mu, logvar)
        action_recon = self.decode(z, text_emb)
        return action_recon, mu, logvar, z

    def compute_loss(self, action, action_recon, mu, logvar, z):
        batch_size, latent_dim = z.shape
        
        recon_loss = F.mse_loss(action_recon, action, reduction='sum')
        
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
        
        total_loss = recon_loss + (1.0 * mi_loss) + (self.beta * tc_loss) + (1.0 * dw_kl_loss)
                     
        return total_loss, recon_loss, tc_loss