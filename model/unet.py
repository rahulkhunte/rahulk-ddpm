import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbedding
from .resblock       import ResBlock
from .attention      import SelfAttention

class UNet(nn.Module):
    """
    U-Net noise predictor ε_θ(x_t, t) for DDPM.

    Architecture:
        Encoder  : ResBlock → Downsample (stride-2 conv) × 2
        Bottleneck: ResBlock → SelfAttention → ResBlock
        Decoder  : Upsample (ConvTranspose) → ResBlock with skip connections × 2
        Output   : 1×1 conv → same shape as input

    Time conditioning injected at every ResBlock via sinusoidal embeddings.
    """
    def __init__(self, in_channels: int = 1, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        self.enc1  = ResBlock(in_channels,  base_ch,     time_dim)
        self.down1 = nn.Conv2d(base_ch,     base_ch,     4, 2, 1)
        self.enc2  = ResBlock(base_ch,      base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)

        self.mid1  = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.attn  = SelfAttention(base_ch * 4)
        self.mid2  = ResBlock(base_ch * 4, base_ch * 2, time_dim)

        self.up2   = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.dec2  = ResBlock(base_ch * 4, base_ch,     time_dim)
        self.up1   = nn.ConvTranspose2d(base_ch,     base_ch,     4, 2, 1)
        self.dec1  = ResBlock(base_ch * 2, base_ch,     time_dim)

        self.out   = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        e1 = self.enc1(x,              t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        m  = self.mid1(self.down2(e2), t_emb)
        m  = self.attn(m)
        m  = self.mid2(m,              t_emb)
        d2 = self.dec2(torch.cat([self.up2(m),  e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out(d1)
