import torch
import torch.nn as nn

def norm_layer(channels: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)

class ResBlock(nn.Module):
    """
    ResNet block with time embedding injection.
    Time conditioning: h = h + MLP(t_emb) broadcast over spatial dims.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1    = norm_layer(in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2    = norm_layer(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act      = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)
