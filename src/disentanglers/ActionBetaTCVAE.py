import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ActionBetaTCVAE(nn.Module):
    def __init__(self, action_dim=7, chunk_size=16, latent_dim=16, beta=6.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.beta = beta  # The disentanglement weight

        # --- Encoder (1D Conv) ---
        # Compresses temporal action chunks into a static latent vector
        self.encoder = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flat dimension after convs (depends on chunk_size)
        # For chunk_size=16 -> reduces to 2. 128*2 = 256
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # --- Decoder (Transposed 1D Conv) ---
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, action_dim, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        # x shape: (B, T, D) -> Permute to (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 128, 2) # Reshape back for conv
        x_recon = self.decoder(h)
        return x_recon.permute(0, 2, 1) # Back to (B, T, D)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    # --- The Beta-TC Loss (Minibatch Weighted Sampling) ---
    def compute_loss_old(self, x, x_recon, mu, logvar, z):
        batch_size = x.size(0)
        
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # 2. Total Correlation (TC) Estimation
        # This approximates the dependency between latent dimensions
        log_q_zx = -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()).sum(1)
        
        # Estimate log q(z) using the minibatch as a sample
        # Matrix of all z vs all mu/var: (B, B, Latent)
        z_expand = z.unsqueeze(1) 
        mu_expand = mu.unsqueeze(0)
        logvar_expand = logvar.unsqueeze(0)
        
        # log q(z|x_j) for every j in batch
        log_q_zx_mat = -0.5 * (math.log(2 * math.pi) + logvar_expand + 
                               (z_expand - mu_expand).pow(2) / logvar_expand.exp()).sum(2)
        
        # log q(z) ≈ log (1/N \sum q(z|x_j)) = logsumexp(matrix) - log(N)
        log_q_z = torch.logsumexp(log_q_zx_mat, dim=1) - math.log(batch_size)
        
        # log prod q(z_i) (Independent marginals)
        # We need log q(z_d) for each dimension d
        # This is computationally heavy, standard approx is often used:
        # TC ≈ KL(q(z)||p(z)) - \sum KL(q(z_i)||p(z_i))
        # But for Beta-TCVAE, we focus on the TC term explicitly:
        # TC = E[log q(z) - log prod q(z_i)]
        
        # SIMPLIFIED TC-VAE LOSS (Chen et al. Isolating Sources of Disentanglement):
        # We penalize the Total Correlation heavily (beta * TC)
        # Use standard KL for the other terms (Mutual Info)
        
        # KL Divergence (Standard)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Heuristic TC penalty (from Beta-TCVAE official implementation logic)
        # Real TC calculation requires marginal sampling, but for CoRL 
        # a high beta on standard KL is often "called" Beta-VAE. 
        # To strictly do TC-VAE, we need the decomposition.
        
        # Let's return the robust training objective:
        # Loss = Recon + KL + (beta - 1) * TC_Approx
        
        # A robust approximation for TC in a batch is:
        # TC ≈ (log_q_z - log_prod_q_zi)
        # Here we will stick to a clean Beta-VAE objective which is more stable
        # for robotics unless you have massive batch sizes (needs B > 256 for good TC est).
        
        # If Batch Size is small (e.g., 64 or 128), TC estimation is noisy.
        # Recommendation: Use Beta-VAE objective but validate with TC metrics.
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def compute_loss(self, x, x_recon, mu, logvar, z, dataset_size=None):
        """
        Computes the Decomposed KL Loss (Chen et al., NeurIPS 2018).
        We strictly penalize Total Correlation (TC) to force disentanglement
        without destroying Mutual Information (Reconstruction quality).
        """
        batch_size, latent_dim = z.shape
        
        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # --- The Stratified Sampling Estimator ---
        
        # Compute log(q(z(x_j) | x_i)) for every pair in the batch
        # Result shape: (Batch_Size, Batch_Size, Latent_Dim)
        # This calculates the probability of sample i's latent code under sample j's distribution
        mu_expand = mu.unsqueeze(0)           # (1, B, D)
        logvar_expand = logvar.unsqueeze(0)   # (1, B, D)
        z_expand = z.unsqueeze(1)             # (B, 1, D)
        
        # Gaussian Log Likelihood: -0.5 * (log(2pi) + logvar + (z-mu)^2/var)
        log_q_z_given_x = -0.5 * (math.log(2 * math.pi) + logvar_expand + 
                                  (z_expand - mu_expand).pow(2) / torch.exp(logvar_expand))
        
        # 2. log q(z) (The Aggregate Posterior)
        # We approximate the density of z by averaging over the batch
        # log q(z) ≈ logsumexp_j(q(z|x_j)) - log(N)
        # Sum over the 'j' dimension (dim 1)
        log_q_z = torch.logsumexp(log_q_z_given_x.sum(dim=2), dim=1) - math.log(batch_size)
        
        # 3. log prod q(z_d) (The Product of Marginals)
        # This represents the "Disentangled Prior" assumption
        # log prod q(z_d) = Sum_d [ logsumexp_j(q(z_d|x_j)) - log(N) ]
        log_prod_q_z_d = torch.logsumexp(log_q_z_given_x, dim=1) - math.log(batch_size)
        log_prod_q_z_d = log_prod_q_z_d.sum(dim=1)
        
        # 4. log p(z) (The Standard Gaussian Prior)
        # Standard Normal log prob
        log_p_z = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)
        
        # --- DECOMPOSITION ---
        
        # A. Mutual Information Gap (MIG) Estimate
        # I(z;x) = KL(q(z,x) || q(z)p(x))
        # This measures how much information z has about x.
        # We generally DO NOT want to penalize this heavily (alpha=1).
        mi_loss = (log_q_z_given_x.sum(dim=2).diagonal() - log_q_z).mean()
        
        # B. Total Correlation (TC)
        # TC(z) = KL(q(z) || prod q(z_d))
        # This measures how correlated the dimensions are.
        # WE WANT TO PENALIZE THIS HEAVILY (beta > 1).
        tc_loss = (log_q_z - log_prod_q_z_d).mean()
        
        # C. Dimension-wise KL
        # KL(q(z_d) || p(z_d))
        # This keeps the latent space from drifting too far from a Gaussian.
        dw_kl_loss = (log_prod_q_z_d - log_p_z).mean()
        
        # --- FINAL OBJECTIVE ---
        # L = Recon + alpha*MI + beta*TC + gamma*DW_KL
        # Standard Beta-TCVAE settings: alpha=1, gamma=1, beta=HIGH
        
        # Note: If using dataset_size scaling (optional but theoretically correct):
        # mi_loss *= dataset_size 
        # But for stability in robotics, we usually stick to batch-mean scaling
        
        total_loss = recon_loss + \
                     (1.0 * mi_loss) + \
                     (self.beta * tc_loss) + \
                     (1.0 * dw_kl_loss)
                     
        return total_loss, recon_loss, tc_loss