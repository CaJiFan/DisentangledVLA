import torch
import torch.nn as nn

class MLPActionProjector(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        
        # Dynamically calculate funnel steps
        h1 = max(input_dim // 2, 256)
        h2 = max(h1 // 2, 128)
        h3 = max(h2 // 2, 64)

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(h2, h3),
            nn.GELU(),
            
            nn.Linear(h3, latent_dim) 
        )

    def forward(self, vla_embedding):
        return self.net(vla_embedding)