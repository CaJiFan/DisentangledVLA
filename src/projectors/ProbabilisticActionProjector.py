import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticActionProjector(nn.Module):
    def __init__(self, input_dim=768, latent_dim=16, hidden_dim=None, dropout=0.3,
                 min_std=1e-4, architecture="mlp"):
        """
        Maps VLA embeddings to a Gaussian distribution in the Disentangled Latent Space.

        Args:
            input_dim:    Output dimension of the VLA backbone.
            latent_dim:   Dimensionality of the Beta-TCVAE latent space.
            hidden_dim:   Override for the first hidden layer width (mlp/bottleneck only).
            dropout:      Dropout rate inside feature_net (mlp/bottleneck only).
            min_std:      Numerical stability floor for predicted standard deviation.
            architecture: One of:
                "mlp"        — 2-layer funnel (4096→512→256→64). High capacity, may overfit
                               when training set has few unique semantic task clusters.
                "bottleneck" — 1-layer compression (4096→128→64). Lower capacity, still
                               non-linear. Good middle ground.
                "linear"     — Direct linear map (4096→64). No hidden layers, no non-linearity.
                               Best extrapolation to unseen task embeddings (new semantic
                               directions not seen during training). Equivalent to linear
                               probing, which generalises best in few-cluster regimes.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = min_std
        self.architecture = architecture

        if architecture == "linear":
            # No feature_net — heads map directly from input space.
            # Forces a linear relationship between VLA embedding and VAE latent space,
            # which extrapolates well to unseen task clusters.
            self.feature_net = nn.Identity()
            self.fc_mu     = nn.Linear(input_dim, latent_dim)
            self.fc_logvar = nn.Linear(input_dim, latent_dim)

        elif architecture == "bottleneck":
            # Single hidden layer with aggressive compression.
            h1 = hidden_dim if hidden_dim is not None else 128
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fc_mu     = nn.Linear(h1, latent_dim)
            self.fc_logvar = nn.Linear(h1, latent_dim)

        else:  # "mlp" — original 2-layer funnel
            # Cap the funnel width to avoid massive over-parameterisation for large VLAs.
            # e.g. OpenVLA: input_dim=4096 → h1=512 (not 2048) → h2=256 (not 1024)
            if hidden_dim is not None:
                h1 = hidden_dim
            else:
                h1 = min(max(input_dim // 2, 256), 512)
            h2 = max(h1 // 2, 128)

            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(h1, h2),
                nn.LayerNorm(h2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fc_mu     = nn.Linear(h2, latent_dim)
            self.fc_logvar = nn.Linear(h2, latent_dim)
        

    def forward(self, x):
        """
        Args:
            x: (Batch, Input_Dim) - The semantic embedding from the VLA
        Returns:
            dist: A torch.distributions.Normal object representing p(z | text/image)
            mu: The predicted mean vector
            logvar: The predicted log-variance vector
        """
        features = self.feature_net(x)
        
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        # Clamp logvar to prevent exploding/vanishing gradients
        # Range [-10, 2] corresponds to std approx [0.006, 2.7]
        logvar = torch.clamp(logvar, min=-10, max=2) 
        
        # Convert to std for the distribution object
        std = torch.exp(0.5 * logvar) + self.min_std
        
        # Create a Differentiable Distribution Object
        # 'Independent' tells PyTorch that the dimensions are uncorrelated (Diagonal Covariance)
        # This explicitly enforces your DISENTANGLEMENT hypothesis.
        dist = torch.distributions.Independent(
            torch.distributions.Normal(mu, std), 
            reinterpreted_batch_ndims=1
        )
        
        return dist, mu, logvar

    def loss(self, predicted_dist, target_z_from_vae):
        """
        Calculates the Negative Log Likelihood (NLL).
        We want to MAXIMIZE the probability of the 'true' VAE latent code 
        under our predicted distribution.
        
        Args:
            predicted_dist: The distribution object from forward()
            target_z_from_vae: The ground truth z vector encoded by your pre-trained VAE
        """
        # NLL = - log( p(z_true | predicted_dist) )
        # This effectively minimizes the distance (MSE) weighted by uncertainty (Sigma)
        return -predicted_dist.log_prob(target_z_from_vae).mean()