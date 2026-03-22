import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timestep t.
    Identical in form to transformer positional encodings (Vaswani et al., 2017).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)
