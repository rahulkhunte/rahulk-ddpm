import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Norm — conditions scale/shift on timestep embedding.
    Used in DiT instead of standard LN to inject diffusion conditioning.

    DiT paper (Peebles & Xie, 2023):
      γ, β = Linear(c)  where c = timestep + class embedding
      AdaLN(x) = γ * LayerNorm(x) + β
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm    = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj    = nn.Linear(cond_dim, dim * 2)    # → (γ, β)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    One DiT Transformer block.

    Architecture (Peebles & Xie, 2023 — "Scalable Diffusion Models with Transformers"):
      x → AdaLN → MultiheadAttention → residual
        → AdaLN → MLP (GELU, 4× expand) → residual

    Key insight vs UNet-DDPM:
      - Operates on flattened image patches (like ViT), not spatial feature maps
      - Scales as O(n²) in sequence length (patch count), not O(h·w) in resolution
      - Conditioning via AdaLN, NOT via cross-attention (cheaper + equally effective)
      - Enables class-conditional generation natively (add class embed to t embed)

    This block sits between VAE latent encoding and VAE decoding in an LDM pipeline:
      Image → VAE encoder → latent z → patchify → DiT blocks → unpatchify
           → VAE decoder → Image
    """

    def __init__(self, hidden_dim: int = 384, num_heads: int = 6,
                 cond_dim: int = 256, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(hidden_dim, cond_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, N, D) — B=batch, N=num_patches, D=hidden_dim
        cond: (B, cond_dim) — timestep + optional class embedding
        """
        # Self-attention with AdaLN conditioning
        normed = self.norm1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP with AdaLN conditioning
        x = x + self.mlp(self.norm2(x, cond))
        return x


class DiTStub(nn.Module):
    """
    Minimal DiT (Diffusion Transformer) — full pipeline stub.

    Replaces the UNet noise predictor in DDPM with a Transformer that:
      1. Patchifies the (latent) image
      2. Adds positional embeddings
      3. Passes through N DiT blocks conditioned on (t, class)
      4. Unpatchifies back to image shape

    Training objective stays identical to DDPM:
      L = ||ε - ε_θ(x_t, t)||²    (simple MSE on predicted noise)

    Scale variants from paper (on ImageNet 256×256):
      DiT-S: depth=12, hidden=384, heads=6   → ~33M params
      DiT-B: depth=12, hidden=768, heads=12  → ~130M params
      DiT-L: depth=24, hidden=1024, heads=16 → ~458M params
      DiT-XL: depth=28, hidden=1152, heads=16 → ~675M params  ← SOTA
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 1, hidden_dim: int = 384,
                 depth: int = 6, num_heads: int = 6,
                 time_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim        = in_channels * patch_size * patch_size

        # Patch embedding (same idea as ViT)
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_dim) for _ in range(depth)
        ])

        self.norm   = nn.LayerNorm(hidden_dim)
        self.head   = nn.Linear(hidden_dim, patch_dim)   # predict noise per patch

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def unpatchify(self, x: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)"""
        B, N, _ = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: noisy image/latent (B, C, H, W)
        t: diffusion timestep  (B,)
        Returns: predicted noise ε_θ  (B, C, H, W)
        """
        B, C, H, W = x.shape
        cond = self.time_embed(t)              # (B, time_dim)

        tokens = self.patchify(x)              # (B, N, patch_dim)
        tokens = self.patch_embed(tokens)      # (B, N, hidden_dim)
        tokens = tokens + self.pos_embed       # add positional encoding

        for block in self.blocks:
            tokens = block(tokens, cond)

        tokens = self.head(self.norm(tokens))  # (B, N, patch_dim)
        return self.unpatchify(tokens, C, H, W)
