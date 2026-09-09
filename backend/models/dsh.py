"""
dsh.py
------
Detection Scoring Head (DSH).

Combines four detection features into a binary classification score.

Detection features (all FloatTensor [B]):
    d_m      : manifold distance from CDP projection
    d_cv     : counterfactual completion variance from CCM
    d_tr     : temporal risk velocity (rate of d_m change over time)
    sigma2_z : mean posterior variance from VLE

These four signals capture complementary aspects:
    d_m      -> absolute deviation from credible manifold
    d_cv     -> structural unpredictability at early time horizons
    d_tr     -> acceleration of divergence (velocity signal)
    sigma2_z -> encoder uncertainty about propagation structure

Architecture:
    Input: [d_m, d_cv, d_tr, sigma2_z] -> log-normalize -> 2-layer MLP -> sigmoid

Log-normalization is applied before the MLP because the four
features operate on very different scales (d_cv can be orders of
magnitude larger than sigma2_z). Log-normalizing stabilizes
gradient flow through the MLP.

Output:
    score : FloatTensor [B]   probability of misinformation in [0, 1]
    logit : FloatTensor [B]   raw logit before sigmoid (for loss computation)
"""

import torch
import torch.nn as nn


class DSH(nn.Module):
    """
    Detection Scoring Head.

    Args:
        input_dim  : int   number of detection features (4)
        hidden_dim : int   MLP hidden dimension (64)
        dropout    : float
    """

    def __init__(self, input_dim=4, hidden_dim=64, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, d_m, d_cv, d_tr, sigma2_z):
        """
        Args:
            d_m      : FloatTensor [B]   manifold distance
            d_cv     : FloatTensor [B]   completion variance
            d_tr     : FloatTensor [B]   temporal risk velocity
            sigma2_z : FloatTensor [B]   posterior variance

        Returns:
            score : FloatTensor [B]   sigmoid probability
            logit : FloatTensor [B]   raw logit
        """
        # ── log-normalize each feature ────────────────────────────────────
        # log1p handles zero values safely: log(1 + x)
        # clamp to avoid numerical issues with very large values
        f_m      = torch.log1p(d_m.clamp(min=0.0))
        f_cv     = torch.log1p(d_cv.clamp(min=0.0))
        f_tr     = torch.log1p(d_tr.clamp(min=0.0))
        f_sigma  = torch.log1p(sigma2_z.clamp(min=0.0))

        # ── concatenate features ──────────────────────────────────────────
        features = torch.stack([f_m, f_cv, f_tr, f_sigma], dim=1)  # [B, 4]

        # ── MLP ───────────────────────────────────────────────────────────
        logit = self.mlp(features).squeeze(1)   # [B]
        score = torch.sigmoid(logit)             # [B]

        return score, logit

    def compute_d_tr(self, d_m_current, d_m_previous):
        """
        Computes temporal risk velocity from two manifold distances.

        d_tr = max(d_m_current - d_m_previous, 0)

        Clamped to zero to count only increasing divergence.
        Decreasing d_m (convergence toward manifold) is not
        penalized — it is a sign of credible propagation.

        Args:
            d_m_current  : FloatTensor [B]
            d_m_previous : FloatTensor [B]

        Returns:
            FloatTensor [B]
        """
        return (d_m_current - d_m_previous).clamp(min=0.0)