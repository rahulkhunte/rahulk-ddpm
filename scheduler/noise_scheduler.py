import torch
import math


class CosineNoiseScheduler:
    """
    Cosine beta schedule — Nichol & Dhariwal (2021) "Improved DDPMs".

    Motivation over linear schedule:
      Linear β rises too fast at low t → destroys structure early in forward process.
      Cosine ᾱ_t = cos²( (t/T + s) / (1+s) · π/2 ) decays more smoothly,
      keeping signal at small t and ensuring ᾱ_T ≈ 0.

    ELBO objective (full variational form):
      L = E[ ||ε - ε_θ(x_t, t)||² · w(t) ]
      where w(t) = β_t² / (2σ_t² α_t (1-ᾱ_t))
      In practice we drop w(t) and optimize the simple objective:
      L_simple = E_t,x0,ε [ ||ε - ε_θ(√ᾱ_t·x0 + √(1-ᾱ_t)·ε, t)||² ]

    Forward:  q(x_t | x_0) = N(√ᾱ_t · x_0,  (1 - ᾱ_t) · I)
    Reverse:  p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t² · I)
              μ_θ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
    """

    def __init__(self, T: int = 1000, s: float = 0.008, device: str = 'cpu'):
        self.T      = T
        self.device = device

        # ᾱ_t = cos²((t/T + s)/(1+s) · π/2)  clipped so β_t ≤ 0.999
        steps = torch.arange(T + 1, dtype=torch.float64)
        f     = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        f     = f / f[0]                                # normalise f(0)=1

        betas = torch.clamp(1.0 - (f[1:] / f[:-1]), min=0.0, max=0.999)

        self.betas              = betas.float().to(device)
        self.alphas             = (1.0 - self.betas)
        self.alpha_bar          = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar     = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_ab  = torch.sqrt(1.0 - self.alpha_bar)

    # ── forward process ──────────────────────────────────────────────────────
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        """x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε"""
        s_ab  = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        s_1ab = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return s_ab * x0 + s_1ab * noise

    # ── reverse step ─────────────────────────────────────────────────────────
    def sample_prev_timestep(self, xt: torch.Tensor, noise_pred: torch.Tensor,
                              t: int, clip_x0: bool = True) -> torch.Tensor:
        """
        Posterior reverse step p_θ(x_{t-1} | x_t).

        Default path (clip_x0=True) reconstructs x̂_0 from ε, clamps it to the
        data range [-1, 1] (static thresholding), then uses the true forward
        posterior q(x_{t-1} | x_t, x̂_0):
            x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t,            clamped to [-1, 1]
            μ   = (√ᾱ_{t-1}·β_t)/(1-ᾱ_t)·x̂_0
                + (√α_t·(1-ᾱ_{t-1}))/(1-ᾱ_t)·x_t
            σ²  = (1-ᾱ_{t-1})/(1-ᾱ_t)·β_t                  (β̃_t, the posterior var)

        Why the clamp matters: at the top of the chain β_t is clamped to 0.999,
        so α_t≈1e-3 and 1/√α_t≈31×. In the unclamped ε-form mean below, any small
        DC bias in ε_θ is amplified ~31× per step and, over the ~50 highest-t
        steps, drives the whole field to a saturated constant (all-black/all-white
        collapse) instead of nucleating structure from pure noise. Clamping x̂_0
        each step removes that runaway. See Nichol & Dhariwal (2021), Ho et al.
        Algorithm 2; static thresholding is standard in production samplers.

        clip_x0=False keeps the original ε-form mean with σ_t=√β_t, retained for
        reference / ablation.
        """
        alpha_t = self.alphas[t]
        beta_t  = self.betas[t]

        if clip_x0:
            ab_t    = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones_like(ab_t)
            x0_hat  = (xt - self.sqrt_one_minus_ab[t] * noise_pred) / self.sqrt_alpha_bar[t]
            x0_hat  = x0_hat.clamp(-1.0, 1.0)
            coef_x0 = torch.sqrt(ab_prev) * beta_t / (1.0 - ab_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t)
            mean    = coef_x0 * x0_hat + coef_xt * xt
            if t == 0:
                return mean
            var     = (1.0 - ab_prev) / (1.0 - ab_t) * beta_t
            return mean + torch.sqrt(var) * torch.randn_like(xt)

        # Original ε-form (σ_t = √β_t), no x̂_0 clamp — kept for ablation.
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / self.sqrt_one_minus_ab[t]) * noise_pred
        )
        if t == 0:
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(xt)
