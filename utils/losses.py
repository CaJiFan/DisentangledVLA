import torch
import torch.nn as nn
import torch.nn.functional as F

class GenGaussLoss(nn.Module):
    def __init__(self, beta_eps=1e-4):
        super(GenGaussLoss, self).__init__()
        self.beta_eps = beta_eps

    def forward(self, mean, one_over_alpha, beta, target):
        # Clamp beta to avoid exploding Gamma functions
        beta = torch.clamp(beta, min=self.beta_eps, max=10.0)
        
        # Absolute residual |z_pred - z_target|
        residual = torch.abs(mean - target)
        
        # Term 1: (|z_pred - z_target| / alpha) ^ beta
        # Since we predict one_over_alpha, it's (residual * one_over_alpha)^beta
        term1 = torch.pow(residual * one_over_alpha + 1e-6, beta)
        
        # Term 2: -log(beta / alpha)  ---> -log(beta * one_over_alpha)
        term2 = -torch.log(beta * one_over_alpha + 1e-6)
        
        # Term 3: log Gamma(1/beta)
        # PyTorch has a built-in log-gamma function!
        term3 = torch.special.gammaln(1.0 / beta)
        
        loss = term1 + term2 + term3
        return loss.mean()

class TempCombLoss(nn.Module):
    def __init__(self):
        super(TempCombLoss, self).__init__()
        self.L_GenGauss = GenGaussLoss()
        self.L_l1 = nn.L1Loss(reduction='mean')
    
    def forward(self, mean, one_over_alpha, beta, target, T1=1.0, T2=0.05):
        """
        T1 scales the L1 Loss (Deterministic Anchor)
        T2 scales the GGD Loss (Probabilistic Shaping)
        """
        l1 = self.L_l1(mean, target)
        l2 = self.L_GenGauss(mean, one_over_alpha, beta, target)
        
        loss = (T1 * l1) + (T2 * l2)
        return loss, l1, l2

class ClosedFormW2Loss(nn.Module):
    def __init__(self, variance_weight=1.0):
        super(ClosedFormW2Loss, self).__init__()
        # We can weight the variance penalty if we want to tune the uncertainty
        self.variance_weight = variance_weight

    def forward(self, pred_mu, pred_logvar, target_mu):
        """
        Calculates the exact Wasserstein-2 distance between a predicted Gaussian
        (with diagonal covariance) and a deterministic target point.
        """
        # 1. Distance between means: ||mu_pred - mu_target||^2
        # We use reduction='none' so we can sum across the 16 latent dimensions first
        mean_penalty = F.mse_loss(pred_mu, target_mu, reduction='none').sum(dim=-1)
        
        # 2. Trace of the predicted covariance matrix: Tr(Sigma_pred)
        # Since logvar is the log of the variance, exp(logvar) gives the variance.
        # The trace of a diagonal matrix is just the sum of its diagonal elements.
        pred_variance = torch.exp(pred_logvar)
        variance_penalty = pred_variance.sum(dim=-1)
        
        # 3. Combine to get W2 squared distance
        w2_squared = mean_penalty + (self.variance_weight * variance_penalty)
        
        # Return the mean loss across the batch
        return w2_squared.mean()


class KLDistillationLoss(nn.Module):
    """
    KL(teacher || projector) for diagonal Gaussians, closed form.

    Minimising this forces the projector to match BOTH the mean AND the
    variance of the teacher VAE posterior — something NLL and W2 fail at:

      NLL  → trains on point target μ_T only; σ_P has no incentive to track σ_T
      W2   → Tr(Σ_P) term acts as a direct variance *penalty*, collapsing σ_P → 0

    KL(N(μ_T,σ_T²) ‖ N(μ_P,σ_P²)) has a stable minimum *only* when
    μ_P = μ_T  AND  σ_P = σ_T.  The σ_T²/σ_P² term prevents collapse;
    the log(σ_P²/σ_T²) term prevents over-spread.
    """

    def forward(self, pred_mu, pred_logvar, target_mu, target_logvar):
        """
        Args:
            pred_mu      : [B, Z] projector predicted mean
            pred_logvar  : [B, Z] projector predicted log-variance
            target_mu    : [B, Z] teacher VAE posterior mean
            target_logvar: [B, Z] teacher VAE posterior log-variance
        Returns:
            scalar mean KL divergence (sum over dims, mean over batch)
        """
        pred_var   = pred_logvar.exp()
        target_var = target_logvar.exp()

        # KL = 0.5 Σ_d [ σ_T² / σ_P²  +  (μ_T - μ_P)² / σ_P²  +  log(σ_P²/σ_T²) - 1 ]
        kl = 0.5 * (
            target_var / pred_var
            + (target_mu - pred_mu).pow(2) / pred_var
            + pred_logvar - target_logvar
            - 1.0
        )
        return kl.sum(dim=-1).mean()