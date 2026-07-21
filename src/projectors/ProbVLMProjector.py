import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbVLMProjector(nn.Module):
    def __init__(
        self, 
        input_dim=4096,   # OpenVLA embedding dimension
        latent_dim=16,    # Action VAE latent dimension
        hidden_dim=1024,  # Shared MLP hidden dimension
        p_drop=0.1
    ):
        super(ProbVLMProjector, self).__init__()
        
        # 1. Shared Base (Compresses 4096 -> 16)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=p_drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=p_drop),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # 2. Mean Head (mu: the actual action to execute)
        self.block_mu = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )

        # 3. Scale Head (1/alpha: confidence/spread)
        # Note: We predict 1/alpha for numerical stability as done in the paper's codebase
        self.block_alpha = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.Softplus() # Enforces strictly positive values
        )

        # 4. Shape Head (beta: the "tail heaviness")
        self.block_beta = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.Softplus() # Enforces strictly positive values
        )
    
    def forward(self, x):
        # Base compression
        x_base = self.shared_net(x) # Shape: (Batch, 16)
        
        # Intrinsic residual connection (like the paper)
        x_intr = x_base 
        
        # Predict the 3 parameters of the Generalized Gaussian
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr) + 1e-4 # Add epsilon to prevent div-by-zero
        x_beta = self.block_beta(x_intr) + 1e-4    # Add epsilon to prevent div-by-zero
        
        return x_mu, x_1alpha, x_beta