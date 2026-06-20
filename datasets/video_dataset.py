"""
video_dataset.py — lightweight video datasets for the video diffusion prototype.

Every dataset yields a (video, label) pair where:
    video : FloatTensor of shape (C, T, H, W), pixel values in [-1, 1]
    label : int (dummy 0 — kept for API parity with torchvision (img, label))

Two sources are provided:
  1. SyntheticMovingShapes — fully synthetic moving square/blob, no download.
     Default choice: tiny, deterministic, perfect for smoke tests + first runs.
  2. MovingMNISTVideo     — wraps torchvision.datasets.MovingMNIST (downloads
     ~0.8 GB on first use). Use only when you actually want MovingMNIST.

Normalization matches the image pipeline: x → x*2 - 1 maps [0,1] → [-1,1],
so the existing CosineNoiseScheduler / DDPM objective applies unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SyntheticMovingShapes(Dataset):
    """
    Synthetic videos of a bright square drifting across a black canvas and
    bouncing off the borders. Deterministic per-index (seeded), so the dataset
    is reproducible and needs no download — ideal for a first training run.
    """

    def __init__(self,
                 num_samples: int = 512,
                 num_frames:  int = 16,
                 image_size:  int = 32,
                 in_channels: int = 1,
                 seed:        int = 0):
        self.num_samples = num_samples
        self.num_frames  = num_frames
        self.image_size  = image_size
        self.in_channels = in_channels
        self.seed        = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(self.seed + idx)
        S   = self.image_size
        T   = self.num_frames

        # Square side = 40–50% of the frame (≈16–25% of pixels). The earlier
        # 1/8–1/3 range (~1.6–8% of pixels) made the foreground so rare that an
        # unconditional VideoDiT minimised the loss by predicting a constant
        # background and never nucleated a square from pure noise. A prominent
        # foreground gives the generative prior a strong, learnable mode.
        lo   = max(4, int(round(0.40 * S)))
        hi   = max(lo + 1, int(round(0.50 * S)) + 1)
        size = rng.randint(lo, hi)                           # square side
        pos  = rng.uniform(0, S - size, size=2)              # (y, x)
        vel  = rng.uniform(-2.5, 2.5, size=2)                # (vy, vx)

        if self.in_channels == 3:
            color = rng.uniform(0.6, 1.0, size=3).astype(np.float32)
        else:
            color = np.array([rng.uniform(0.7, 1.0)], dtype=np.float32)

        # Conditioning vector (6 scalars, ~[-1,1]): start position, velocity,
        # size and brightness. The bounce dynamics make the whole clip a
        # deterministic function of these, so feeding them to the model (DiT
        # AdaLN) tells it WHERE/HOW the square moves — exactly the information an
        # unconditional model lacks at the top of the diffusion chain, where it
        # otherwise collapses to a constant background instead of nucleating a
        # square. Normalisation must match between training and sampling.
        cond = np.array([
            2.0 * pos[0] / S - 1.0,
            2.0 * pos[1] / S - 1.0,
            vel[0] / 2.5,
            vel[1] / 2.5,
            2.0 * size / S - 1.0,
            2.0 * float(color.mean()) - 1.0,
        ], dtype=np.float32)

        frames = np.zeros((T, self.in_channels, S, S), dtype=np.float32)
        for f in range(T):
            y, x = pos
            y0, x0 = int(round(y)), int(round(x))
            y1, x1 = min(S, y0 + size), min(S, x0 + size)
            y0, x0 = max(0, y0), max(0, x0)
            for c in range(self.in_channels):
                frames[f, c, y0:y1, x0:x1] = color[c]

            pos = pos + vel
            # bounce off the walls
            for d in range(2):
                if pos[d] < 0:
                    pos[d] = -pos[d];                 vel[d] = -vel[d]
                if pos[d] > S - size:
                    pos[d] = 2 * (S - size) - pos[d]; vel[d] = -vel[d]

        video = torch.from_numpy(frames)            # (T, C, H, W) in [0,1]
        video = video.permute(1, 0, 2, 3).contiguous()   # (C, T, H, W)
        video = video * 2.0 - 1.0                   # → [-1, 1]
        return video, torch.from_numpy(cond)


class MovingMNISTVideo(Dataset):
    """
    Thin wrapper over torchvision.datasets.MovingMNIST that returns clips as
    (C, T, H, W) in [-1, 1], subsampled to `num_frames` and resized to
    `image_size`. Downloads on first use.
    """

    def __init__(self,
                 root:        str = 'data/',
                 num_frames:  int = 16,
                 image_size:  int = 32,
                 download:    bool = True,
                 num_samples: int = None):
        from torchvision.datasets import MovingMNIST
        self.base       = MovingMNIST(root=root, download=download)
        self.num_frames = num_frames
        self.image_size = image_size
        # Optional cap so a "decent subset" can be used for a bounded-cost run
        # instead of all 10k sequences (keeps epochs/steps within the GPU budget).
        length = len(self.base) if num_samples is None else min(num_samples, len(self.base))

        # Pre-process every clip ONCE into an in-memory cache (C,T,H,W in [-1,1]).
        # The frame-subsample + bilinear resize (64→32) is expensive; doing it
        # per __getitem__ ran ~2.4M times over a 600-epoch run and starved the
        # GPU (the run hit the wall-clock timeout). 4096 clips ≈ 268 MB — fine.
        clips = [self._process(self.base[i]) for i in range(length)]
        self.clips = torch.stack(clips)          # (N, C, T, H, W)

    def _process(self, clip) -> torch.Tensor:
        if isinstance(clip, (tuple, list)):
            clip = clip[0]
        clip = torch.as_tensor(clip).float()

        # Normalize to (T, C, H, W)
        if clip.dim() == 3:                      # (T, H, W)
            clip = clip.unsqueeze(1)             # (T, 1, H, W)
        elif clip.dim() == 4 and clip.shape[1] not in (1, 3) and clip.shape[-3] not in (1, 3):
            clip = clip.unsqueeze(1)

        clip = clip.permute(1, 0, 2, 3).contiguous()   # (C, T, H, W)
        C, T, H, W = clip.shape

        # Subsample / pad frames to num_frames (take the leading frames)
        if T >= self.num_frames:
            clip = clip[:, :self.num_frames]
        else:
            pad = clip[:, -1:].repeat(1, self.num_frames - T, 1, 1)
            clip = torch.cat([clip, pad], dim=1)

        # Spatial resize if needed
        if H != self.image_size or W != self.image_size:
            c, t = clip.shape[0], clip.shape[1]
            clip = clip.reshape(c * t, 1, H, W)
            clip = F.interpolate(clip, size=(self.image_size, self.image_size),
                                 mode='bilinear', align_corners=False)
            clip = clip.reshape(c, t, self.image_size, self.image_size)

        return clip / 255.0 * 2.0 - 1.0          # uint8 range → [-1, 1]

    def __len__(self) -> int:
        return self.clips.shape[0]

    def __getitem__(self, idx: int):
        return self.clips[idx], 0


def build_video_dataset(cfg: dict) -> Dataset:
    """
    Factory driven by cfg['video']. Defaults to the synthetic dataset so the
    prototype runs end-to-end with zero downloads.
    """
    v    = cfg.get('video', {})
    name = str(v.get('dataset', 'synthetic')).lower()

    if name == 'movingmnist':
        return MovingMNISTVideo(
            root=v.get('data_root', 'data/'),
            num_frames=v.get('num_frames', 16),
            image_size=v.get('frame_size', 32),
            download=v.get('download', True),
            num_samples=v.get('num_samples', None),
        )

    return SyntheticMovingShapes(
        num_samples=v.get('num_samples', 512),
        num_frames=v.get('num_frames', 16),
        image_size=v.get('frame_size', 32),
        in_channels=v.get('in_channels', 1),
        seed=v.get('seed', 0),
    )
