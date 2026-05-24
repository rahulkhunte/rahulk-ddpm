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
                              t: int) -> torch.Tensor:
        """
        μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ)
        Sample: x_{t-1} = μ_θ + σ_t · z,  z ~ N(0,I)  (σ_t = √β_t)
        """
        alpha_t = self.alphas[t]
        beta_t  = self.betas[t]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / self.sqrt_one_minus_ab[t]) * noise_pred
        )
        if t == 0:
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(xt)
