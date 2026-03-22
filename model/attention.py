import torch
import torch.nn as nn
from .resblock import norm_layer

class SelfAttention(nn.Module):
    """
    Single-head self-attention over spatial feature maps.
    Used in UNet bottleneck to capture global structure.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm = norm_layer(channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1,2), k) * (C ** -0.5), dim=-1)
        out  = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)
