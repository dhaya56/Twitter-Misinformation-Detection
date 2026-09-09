"""
ccm.py
------
Counterfactual Completion Module (CCM).

Generates N stochastic completions of a partial propagation
latent z using the frozen CDP, then computes the completion
variance d_cv as an early detection feature.

Key insight:
    Credible propagation has structurally predictable completions
    -> low variance across N samples.
    Misinformation lies off the credibility manifold
    -> high variance (model generates diverse completions from
       low-density regions of the learned distribution).

All CDP forward passes run under torch.no_grad() since CDP
is frozen during Phase 3. No gradients flow into CDP.

Inputs:
    z        : FloatTensor [B, latent_dim]   current latent
    tau_obs  : float   observation time in minutes
    tau_max  : float   dataset median max timestamp (11786.27 min)

Outputs:
    d_cv     : FloatTensor [B]   completion variance (detection feature)
    z_mean   : FloatTensor [B, latent_dim]  mean completion (for reference)
"""

import torch
import torch.nn as nn


class CCM(nn.Module):
    """
    Counterfactual Completion Module.

    Args:
        cdp          : CDP instance (frozen during Phase 3)
        num_completions : int  number of stochastic samples N (default 8)
        tau_max      : float  dataset median max propagation time in minutes
                              used to compute T_comp from tau_obs
    """

    def __init__(self, cdp, num_completions=8, tau_max=11786.27):
        super().__init__()
        self.cdp             = cdp
        self.num_completions = num_completions
        self.tau_max         = tau_max

    def forward(self, z, tau_obs):
        """
        Generates N completions and computes variance.

        CDP is called under no_grad regardless of training mode
        since it is always frozen when CCM is used (Phase 3+).

        Args:
            z       : FloatTensor [B, latent_dim]
            tau_obs : float  observation cutoff in minutes
                      pass -1.0 to use full graph (tau_obs = tau_max)
                      which gives T_comp = 1 → minimal noise → low variance

        Returns:
            d_cv   : FloatTensor [B]
            z_mean : FloatTensor [B, latent_dim]
        """
        # sentinel: full graph observed → use tau_max so T_comp → 1
        if tau_obs < 0:
            tau_obs = self.tau_max

        completions = self._generate_completions(z, tau_obs)
        # completions: FloatTensor [B, N, latent_dim]

        d_cv   = self._compute_variance(completions)   # [B]
        z_mean = completions.mean(dim=1)               # [B, latent_dim]

        return d_cv, z_mean

    @torch.no_grad()
    def _generate_completions(self, z, tau_obs):
        """
        Runs N stochastic completions via CDP.complete_from_partial.

        Each completion is independent (different random noise).

        Returns:
            FloatTensor [B, N, latent_dim]
        """
        B      = z.shape[0]
        D      = z.shape[1]
        device = z.device

        samples = []
        for _ in range(self.num_completions):
            z_c = self.cdp.complete_from_partial(
                z       = z,
                tau_obs = tau_obs,
                tau_max = self.tau_max,
            )   # [B, D]
            samples.append(z_c)

        completions = torch.stack(samples, dim=1)   # [B, N, D]
        return completions

    def _compute_variance(self, completions):
        """
        Computes scalar variance per graph across N completions.

        Variance = mean over latent dimensions of the
        sample variance across N completions.

        var_per_dim = (1/(N-1)) * sum_n ||z_n - z_mean||^2  per dim
        d_cv = mean over dims of var_per_dim

        Args:
            completions : FloatTensor [B, N, D]

        Returns:
            FloatTensor [B]
        """
        N = completions.shape[1]

        if N == 1:
            # variance undefined with one sample — return zeros
            return torch.zeros(
                completions.shape[0],
                dtype=torch.float32,
                device=completions.device,
            )

        z_mean     = completions.mean(dim=1, keepdim=True)    # [B, 1, D]
        sq_diff    = (completions - z_mean) ** 2              # [B, N, D]
        var_per_dim = sq_diff.sum(dim=1) / (N - 1)           # [B, D]  unbiased
        d_cv        = var_per_dim.mean(dim=1)                 # [B]

        return d_cv