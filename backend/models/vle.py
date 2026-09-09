"""
vle.py
------
Variational Latent Encoder (VLE).

Maps the STE graph embedding h to a latent distribution
N(mu_z, sigma_z^2 * I) and samples z via reparameterization.

Also injects the root text embedding via a learned projection
added to h before the linear heads, so semantic content
influences the latent distribution without polluting the
structural node features used by the GAT.

Outputs:
    z       : FloatTensor [B, latent_dim]   sampled latent vector
    mu_z    : FloatTensor [B, latent_dim]   distribution mean
    sigma_z : FloatTensor [B, latent_dim]   distribution std (> 0)

At inference, use mu_z directly (no sampling noise).
At training, use z = mu_z + sigma_z * eps, eps ~ N(0, I).

sigma2_z (mean diagonal variance) is used as detection feature d4:
    sigma2_z = mean(sigma_z ** 2)  per graph  -> FloatTensor [B]
"""

import torch
import torch.nn as nn


class VLE(nn.Module):
    """
    Variational Latent Encoder.

    Args:
        graph_embed_dim  : int   STE output dimension (128)
        text_embed_dim   : int   sentence embedding dimension (384)
        text_proj_dim    : int   text projection dimension (64)
        latent_dim       : int   latent space dimension (128)
        dropout          : float
    """

    def __init__(
        self,
        graph_embed_dim = 128,
        text_embed_dim  = 384,
        text_proj_dim   = 64,
        latent_dim      = 128,
        dropout         = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # project text embedding to a smaller dimension
        # then add to graph embedding before the latent heads
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # fusion: combine graph embedding and text projection
        fused_dim = graph_embed_dim + text_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, graph_embed_dim),
            nn.LayerNorm(graph_embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # mu head: graph_embed_dim -> latent_dim
        self.mu_head = nn.Linear(graph_embed_dim, latent_dim)

        # log_sigma head: graph_embed_dim -> latent_dim
        # outputs log(sigma) for numerical stability
        self.log_sigma_head = nn.Linear(graph_embed_dim, latent_dim)

        # log_sigma is clamped to [-4, 4] to prevent collapse or explosion
        self.log_sigma_min = -4.0
        self.log_sigma_max =  4.0

        self._init_weights()

    def _init_weights(self):
        """Small initialization for the heads to start near identity mapping."""
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_sigma_head.weight, gain=0.1)
        nn.init.constant_(self.log_sigma_head.bias, -1.0)
        # bias of -1.0 means initial sigma ~ exp(-1) ~ 0.37
        # a reasonable starting spread that is not too wide or collapsed

    def forward(self, h, root_text_emb, training=True):
        """
        Args:
            h              : FloatTensor [B, graph_embed_dim]
                             STE graph embedding
            root_text_emb  : FloatTensor [B, text_embed_dim]
                             sentence embedding of source tweet
            training       : bool
                             if True, sample z with noise (reparameterization)
                             if False, return mu_z directly (deterministic)

        Returns:
            z        : FloatTensor [B, latent_dim]
            mu_z     : FloatTensor [B, latent_dim]
            sigma_z  : FloatTensor [B, latent_dim]
            sigma2_z : FloatTensor [B]   mean diagonal variance per graph
        """
        # ── text injection ────────────────────────────────────────────────
        if root_text_emb.dim() == 3:
            root_text_emb = root_text_emb.squeeze(1)   # [B, 1, 384] → [B, 384]
        t = self.text_proj(root_text_emb)          # [B, text_proj_dim]
        fused = torch.cat([h, t], dim=1)               # [B, graph+text_proj]
        h_f   = self.fusion(fused)                     # [B, graph_embed_dim]

        # ── distribution parameters ───────────────────────────────────────
        mu_z       = self.mu_head(h_f)                 # [B, latent_dim]

        log_sigma  = self.log_sigma_head(h_f)          # [B, latent_dim]
        log_sigma  = log_sigma.clamp(
            self.log_sigma_min, self.log_sigma_max
        )
        sigma_z    = torch.exp(log_sigma)              # [B, latent_dim] > 0

        # ── sampling ──────────────────────────────────────────────────────
        if training:
            eps = torch.randn_like(mu_z)               # [B, latent_dim]
            z   = mu_z + sigma_z * eps
        else:
            z   = mu_z

        # ── detection feature: mean diagonal variance ─────────────────────
        sigma2_z = (sigma_z ** 2).mean(dim=1)          # [B]

        return z, mu_z, sigma_z, sigma2_z

    def kl_loss(self, mu_z, sigma_z):
        """
        Analytical KL divergence from N(mu, sigma^2) to N(0, I).

        KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))

        Used in Phase 1 training only.

        Args:
            mu_z    : FloatTensor [B, latent_dim]
            sigma_z : FloatTensor [B, latent_dim]

        Returns:
            FloatTensor [B]  per-sample KL divergence
        """
        kl = 0.5 * (sigma_z ** 2 + mu_z ** 2 - 1.0 - 2.0 * torch.log(sigma_z))
        return kl.sum(dim=1)   # [B]