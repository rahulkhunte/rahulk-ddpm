import torch

class LinearNoiseScheduler:
    """
    Linear beta schedule from Ho et al. (2020).
    Forward:  q(x_t | x_0) = N(sqrt(ā_t) * x_0, (1 - ā_t) * I)
    Reverse:  p_θ(x_{t-1} | x_t) via predicted noise
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = 'cpu'):
        self.T      = T
        self.device = device
        self.betas  = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar          = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar     = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_ab  = torch.sqrt(1.0 - self.alpha_bar)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        s_ab  = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        s_1ab = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return s_ab * x0 + s_1ab * noise

    def sample_prev_timestep(self, xt: torch.Tensor, noise_pred: torch.Tensor,
                              t: int) -> torch.Tensor:
        alpha_t     = self.alphas[t]
        beta_t      = self.betas[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / self.sqrt_one_minus_ab[t]) * noise_pred
        )
        if t == 0:
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(xt)
