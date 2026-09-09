"""
cdp.py
------
Credibility Diffusion Prior (CDP).

A DDPM operating in latent space (z in R^{latent_dim}).
Trained only on credible samples to learn p_theta(z | y=0).
The support of this distribution defines the credibility manifold M_c.

At inference, partial reverse diffusion (DDIM, T_inf steps) from
a test latent z projects it toward M_c. The displacement
||z - z_proj|| is the manifold distance score d_m.

Architecture:
    Denoising network: 4-layer residual MLP with sinusoidal
    time embedding. ~0.7M parameters.

Forward process (training):
    q(z_t | z_0) = N(sqrt(alpha_bar_t) * z_0, (1 - alpha_bar_t) * I)

Reverse process (inference — DDIM, deterministic):
    z_{s-1} = sqrt(alpha_bar_{s-1}) * pred_z0
            + sqrt(1 - alpha_bar_{s-1}) * eps_theta(z_s, s)

    where pred_z0 = (z_s - sqrt(1 - alpha_bar_s) * eps_theta) / sqrt(alpha_bar_s)
"""

import torch
import torch.nn as nn
import math


# ── noise schedule ────────────────────────────────────────────────────────────

def cosine_beta_schedule(T, s=0.008):
    """
    Cosine noise schedule from Improved DDPM (Nichol & Dhariwal 2021).

    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    alpha_bar_t = cos^2( (t/T + s) / (1 + s) * pi/2 )

    Args:
        T : int   total diffusion steps
        s : float offset to prevent beta_0 from being too small

    Returns:
        betas          : FloatTensor [T]
        alphas         : FloatTensor [T]
        alpha_bars     : FloatTensor [T]
    """
    steps     = T + 1
    t         = torch.linspace(0, T, steps)
    f         = torch.cos(((t / T + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]

    betas      = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    betas      = betas.clamp(min=1e-5, max=0.999)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bars


# ── sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timestep t.
    Produces a D-dimensional vector for scalar timestep t.

    Args:
        embed_dim : int  output embedding dimension (must be even)
    """

    def __init__(self, embed_dim=32):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Args:
            t : LongTensor [B]  timestep indices

        Returns:
            FloatTensor [B, embed_dim]
        """
        device = t.device
        half   = self.embed_dim // 2
        freqs  = torch.exp(
            -math.log(10000) *
            torch.arange(half, dtype=torch.float32, device=device) / half
        )
        args   = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb   # [B, embed_dim]


# ── residual MLP block ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    One residual block for the denoising MLP.

    z_in -> Linear -> LayerNorm -> SiLU -> Linear -> LayerNorm -> + z_in

    Time embedding is injected by addition after the first LayerNorm.

    Args:
        dim      : int  feature dimension
        time_dim : int  time embedding dimension
    """

    def __init__(self, dim, time_dim):
        super().__init__()
        self.lin1    = nn.Linear(dim, dim)
        self.lin2    = nn.Linear(dim, dim)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.time_lin = nn.Linear(time_dim, dim)
        self.act      = nn.SiLU()

    def forward(self, x, t_emb):
        """
        Args:
            x     : FloatTensor [B, dim]
            t_emb : FloatTensor [B, time_dim]

        Returns:
            FloatTensor [B, dim]
        """
        h = self.lin1(x)
        h = self.norm1(h)
        h = h + self.time_lin(t_emb)   # time injection
        h = self.act(h)
        h = self.lin2(h)
        h = self.norm2(h)
        return x + h   # residual


# ── denoising network ─────────────────────────────────────────────────────────

class DenoisingMLP(nn.Module):
    """
    Denoising network eps_theta(z_t, t).

    Predicts the noise eps added to z_0 to produce z_t.

    Architecture:
        Input projection: latent_dim -> hidden_dim
        3 residual blocks: hidden_dim -> hidden_dim
        Output projection: hidden_dim -> latent_dim

    Args:
        latent_dim  : int  latent space dimension (128)
        hidden_dim  : int  MLP hidden dimension (256)
        time_dim    : int  sinusoidal time embedding dim (32)
        num_blocks  : int  number of residual blocks (3)
    """

    def __init__(self, latent_dim=128, hidden_dim=256,
                 time_dim=32, num_blocks=3):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )

        # input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        # output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_output_small()

    def _init_output_small(self):
        """Initialize output projection to near-zero for stable training start."""
        nn.init.xavier_uniform_(
            self.output_proj[-1].weight, gain=0.01
        )
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, z_t, t):
        """
        Args:
            z_t : FloatTensor [B, latent_dim]  noisy latent
            t   : LongTensor  [B]              timestep indices

        Returns:
            FloatTensor [B, latent_dim]  predicted noise eps
        """
        t_emb = self.time_embed(t)      # [B, time_dim]
        t_emb = self.time_mlp(t_emb)   # [B, hidden_dim]

        h = self.input_proj(z_t)       # [B, hidden_dim]

        for block in self.blocks:
            h = block(h, t_emb)

        return self.output_proj(h)     # [B, latent_dim]


# ── main CDP module ───────────────────────────────────────────────────────────

class CDP(nn.Module):
    """
    Credibility Diffusion Prior.

    Wraps the denoising network with the noise schedule
    and provides methods for:
        - training loss (DDPM objective)
        - manifold projection (DDIM inference)
        - counterfactual completion noise (for CCM)

    Args:
        latent_dim      : int   latent space dimension (128)
        hidden_dim      : int   denoising MLP hidden dim (256)
        time_dim        : int   time embedding dim (32)
        num_blocks      : int   residual blocks in MLP (3)
        T               : int   total diffusion steps (200)
        T_inf           : int   DDIM inference steps (20)
        t_star          : int   noise level for manifold projection (40)
                                z is noised to t_star then projected back
    """

    def __init__(
        self,
        latent_dim = 128,
        hidden_dim = 256,
        time_dim   = 32,
        num_blocks = 3,
        T          = 200,
        T_inf      = 20,
        t_star     = 40,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.T          = T
        self.T_inf      = T_inf
        self.t_star     = t_star

        self.denoiser = DenoisingMLP(
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            time_dim   = time_dim,
            num_blocks = num_blocks,
        )

        # precompute and register noise schedule buffers
        betas, alphas, alpha_bars = cosine_beta_schedule(T)
        self.register_buffer("betas",       betas)
        self.register_buffer("alphas",      alphas)
        self.register_buffer("alpha_bars",  alpha_bars)
        self.register_buffer(
            "sqrt_alpha_bars",      torch.sqrt(alpha_bars)
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )

    # ── training ──────────────────────────────────────────────────────────────

    def training_loss(self, z_0):
        """
        DDPM training objective.
        Samples random timestep t, adds noise, predicts noise.

        Args:
            z_0 : FloatTensor [B, latent_dim]  clean latent vectors
                  (only credible samples should be passed here)

        Returns:
            loss : FloatTensor scalar  mean squared error on noise prediction
        """
        B      = z_0.shape[0]
        device = z_0.device

        # sample random timesteps
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # sample noise
        eps = torch.randn_like(z_0)

        # forward process: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1-alpha_bar_t) * eps
        sqrt_ab     = self.sqrt_alpha_bars[t].unsqueeze(1)       # [B, 1]
        sqrt_one_ab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)

        z_t = sqrt_ab * z_0 + sqrt_one_ab * eps                  # [B, D]

        # predict noise
        eps_pred = self.denoiser(z_t, t)                         # [B, D]

        return nn.functional.mse_loss(eps_pred, eps)

    # ── manifold projection (DDIM inference) ──────────────────────────────────

    @torch.no_grad()
    def project_to_manifold(self, z, return_trajectory=False):
        """
        Projects z onto the credibility manifold M_c using
        partial DDIM reverse diffusion.

        Steps:
            1. Noise z to t_star: z_tstar = sqrt(ab_tstar)*z + sqrt(1-ab_tstar)*eps
            2. Run T_inf DDIM reverse steps from t_star to 0
            3. Return z_proj = z_0  (the projected point)

        Args:
            z                  : FloatTensor [B, latent_dim]
            return_trajectory  : bool  if True, return all intermediate states

        Returns:
            z_proj      : FloatTensor [B, latent_dim]
            trajectory  : list of FloatTensors (only if return_trajectory=True)
            d_m         : FloatTensor [B]  manifold distance ||z - z_proj||_2
        """
        device = z.device
        B      = z.shape[0]

        # step 1: noise z to t_star
        t_star_tensor = torch.tensor(
            self.t_star - 1, dtype=torch.long, device=device
        )
        sqrt_ab     = self.sqrt_alpha_bars[t_star_tensor]
        sqrt_one_ab = self.sqrt_one_minus_alpha_bars[t_star_tensor]
        eps_init    = torch.randn_like(z)
        z_t         = sqrt_ab * z + sqrt_one_ab * eps_init    # [B, D]

        # step 2: DDIM reverse from t_star to 0
        # equispaced timestep sequence from t_star down to 0
        timesteps = torch.linspace(
            float(self.t_star - 1), 0.0, self.T_inf,
            device=device
        ).long().unique(sorted=False)

        trajectory  = [z_t.clone()] if return_trajectory else None
        z_current   = z_t

        for i, s in enumerate(timesteps):
            s_batch = s.expand(B)                              # [B]

            # predict noise at current step
            eps_pred = self.denoiser(z_current, s_batch)      # [B, D]

            ab_s  = self.alpha_bars[s]
            # previous timestep (one step earlier, toward 0)
            if i + 1 < len(timesteps):
                s_prev  = timesteps[i + 1]
                ab_prev = self.alpha_bars[s_prev]
            else:
                ab_prev = torch.tensor(1.0, device=device)

            # DDIM update (deterministic)
            pred_z0    = (
                (z_current - torch.sqrt(1 - ab_s) * eps_pred)
                / torch.sqrt(ab_s)
            ).clamp(-5.0, 5.0)

            z_current = (
                torch.sqrt(ab_prev) * pred_z0
                + torch.sqrt(1 - ab_prev) * eps_pred
            )

            if return_trajectory:
                trajectory.append(z_current.clone())

        z_proj = z_current                                     # [B, D]
        d_m    = torch.norm(z - z_proj, dim=1)                 # [B]

        if return_trajectory:
            return z_proj, trajectory, d_m
        return z_proj, d_m

    # ── counterfactual completion (for CCM) ───────────────────────────────────

    @torch.no_grad()
    def complete_from_partial(self, z, tau_obs, tau_max):
        """
        Generates one stochastic completion of a partial propagation state.

        The partial observation at tau_obs produces a latent z.
        T_comp = T * (1 - tau_obs / tau_max) is the noise level
        reflecting how much of the cascade is unobserved.

        Steps:
            1. Noise z to T_comp (stochastic)
            2. Run stochastic DDPM reverse from T_comp to 0
            3. Return z_complete (one sample)

        Args:
            z        : FloatTensor [B, latent_dim]
            tau_obs  : float  observation cutoff in minutes
            tau_max  : float  typical max propagation time in minutes
                              use dataset median: 11786.27

        Returns:
            z_complete : FloatTensor [B, latent_dim]
        """
        device = z.device
        B      = z.shape[0]

        # compute completion noise level
        frac   = min(tau_obs / tau_max, 1.0)
        T_comp = max(int(self.T * (1.0 - frac)), 1)
        T_comp = min(T_comp, self.T - 1)

        t_comp_tensor = torch.tensor(T_comp - 1, dtype=torch.long, device=device)

        # noise z to T_comp
        sqrt_ab     = self.sqrt_alpha_bars[t_comp_tensor]
        sqrt_one_ab = self.sqrt_one_minus_alpha_bars[t_comp_tensor]
        eps_init    = torch.randn_like(z)
        z_t         = sqrt_ab * z + sqrt_one_ab * eps_init       # [B, D]

        # stochastic DDPM reverse from T_comp to 0
        for s in range(T_comp - 1, -1, -1):
            s_batch  = torch.full((B,), s, dtype=torch.long, device=device)
            eps_pred = self.denoiser(z_t, s_batch)               # [B, D]

            alpha_t    = self.alphas[s]
            ab_t       = self.alpha_bars[s]
            sqrt_recip = 1.0 / torch.sqrt(alpha_t)
            beta_t     = self.betas[s]

            # DDPM reverse mean
            z_mean = sqrt_recip * (
                z_t - (beta_t / torch.sqrt(1 - ab_t)) * eps_pred
            )

            if s > 0:
                noise = torch.randn_like(z_t)
                z_t   = z_mean + torch.sqrt(beta_t) * noise
            else:
                z_t   = z_mean

        return z_t   # z_complete