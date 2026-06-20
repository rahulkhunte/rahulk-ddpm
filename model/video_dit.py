import torch
import torch.nn as nn

from .time_embedding import SinusoidalTimeEmbedding
from .dit_block      import DiTBlock


class VideoDiT(nn.Module):
    """
    Minimal spatiotemporal Diffusion Transformer ε_θ(x_t, t) for video.

    This is the video analogue of the image `DiTStub` in `dit_block.py`.
    It reuses the existing building blocks instead of reimplementing them:
      - `SinusoidalTimeEmbedding` (model/time_embedding.py) → timestep cond
      - `DiTBlock` (model/dit_block.py)                     → AdaLN + MHA + MLP

    Pipeline (identical idea to image DiT, extended to time):
        video (B, C, T, H, W)
          → 3-D patchify (split T into temporal patches, H/W into spatial)
          → Linear patch embed + learnable positional embedding
          → N DiT blocks conditioned on timestep t (via AdaLN)
          → Linear head → 3-D unpatchify
          → predicted noise ε_θ (B, C, T, H, W)

    Training objective is unchanged from the image DDPM:
        L_simple = E || ε - ε_θ(x_t, t) ||²
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_frames:  int = 16,
                 image_size:  int = 32,
                 patch_size:  int = 4,
                 patch_t:     int = 2,
                 hidden_dim:  int = 256,
                 depth:       int = 4,
                 num_heads:   int = 4,
                 time_dim:    int = 256,
                 cond_features: int = 0):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert num_frames % patch_t     == 0, "num_frames must be divisible by patch_t"
        assert hidden_dim % num_heads    == 0, "hidden_dim must be divisible by num_heads"

        self.in_channels = in_channels
        self.patch_size  = patch_size
        self.patch_t     = patch_t

        n_t = num_frames // patch_t
        n_h = image_size  // patch_size
        n_w = image_size  // patch_size
        self.num_patches = n_t * n_h * n_w
        patch_dim = in_channels * patch_t * patch_size * patch_size

        # Patch embedding (spatiotemporal tube → token)
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        # Learnable positional embedding over all space-time tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep conditioning (reused sinusoidal embedding + MLP, → time_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Optional label conditioning (DiT-style): embed a continuous attribute
        # vector y (e.g. the square's start position / velocity / size) and ADD
        # it to the timestep embedding, exactly as DiT adds a class embedding.
        # This is what lets the model nucleate structure from pure noise: at the
        # top of the chain (t≈T) the timestep signal carries no spatial info, so
        # the label tells the model WHERE to place the square — breaking the
        # symmetry an unconditional model cannot. The second Linear is zero-init
        # so training starts identical to the unconditional baseline (stable).
        self.cond_features = cond_features
        if cond_features > 0:
            self.label_embed = nn.Sequential(
                nn.Linear(cond_features, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            nn.init.zeros_(self.label_embed[-1].weight)
            nn.init.zeros_(self.label_embed[-1].bias)

        # Transformer trunk — reuse the image DiT block, conditioned on t
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, cond_dim=time_dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, patch_dim)   # predict noise per tube

    # ── space-time (un)patchify ────────────────────────────────────────────────
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) → (B, N, patch_dim)"""
        B, C, T, H, W = x.shape
        pt, p = self.patch_t, self.patch_size
        x = x.reshape(B, C, T // pt, pt, H // p, p, W // p, p)
        # → B, Tn, Hn, Wn, C, pt, p, p
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, (T // pt) * (H // p) * (W // p), C * pt * p * p)
        return x

    def unpatchify(self, x: torch.Tensor, C: int, T: int, H: int, W: int) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, T, H, W)"""
        B = x.shape[0]
        pt, p = self.patch_t, self.patch_size
        n_t, n_h, n_w = T // pt, H // p, W // p
        x = x.reshape(B, n_t, n_h, n_w, C, pt, p, p)
        # → B, C, Tn, pt, Hn, p, Wn, p
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(B, C, T, H, W)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        """
        x: noisy video (B, C, T, H, W)
        t: diffusion timestep (B,)
        y: optional conditioning attributes (B, cond_features); ignored if the
           model was built without conditioning.
        Returns: predicted noise ε_θ (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        cond = self.time_embed(t)              # (B, time_dim)
        if self.cond_features > 0 and y is not None:
            cond = cond + self.label_embed(y)  # DiT class-conditioning: t + label

        tokens = self.patchify(x)              # (B, N, patch_dim)
        tokens = self.patch_embed(tokens)      # (B, N, hidden_dim)
        tokens = tokens + self.pos_embed       # add positional encoding

        for block in self.blocks:
            tokens = block(tokens, cond)

        tokens = self.head(self.norm(tokens))  # (B, N, patch_dim)
        return self.unpatchify(tokens, C, T, H, W)
