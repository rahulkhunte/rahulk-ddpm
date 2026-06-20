"""
sample_video.py — reverse diffusion sampling for the video DDPM prototype.

Loads a VideoDiT checkpoint and runs the full reverse process
    x_T ~ N(0, I)  →  x_0   over T steps,
operating directly on 5-D video tensors (B, C, T, H, W). The scheduler's
`sample_prev_timestep` uses scalar coefficients, so no changes are needed.

Outputs:
  - assets/video_samples/final_grid.png : (clip × frame) montage
  - assets/video_samples/sample.gif     : animated frames of clip 0
"""

import os, yaml, argparse
import torch
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.video_dit import VideoDiT
from scheduler       import CosineNoiseScheduler
from train_video     import build_model
from datasets.video_dataset import build_video_dataset


@torch.no_grad()
def sample(ckpt: str, cfg_path: str = 'config.yaml', n: int = 4,
           save_gif: bool = True, overrides: dict = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    vcfg = dict(cfg.get('video', {}))
    if overrides:
        vcfg.update({k: v for k, v in overrides.items() if v is not None})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(vcfg, cfg.get('time_dim', 256), device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    scheduler = CosineNoiseScheduler(cfg['T'], s=cfg.get('cosine_s', 0.008), device=device)

    C = vcfg.get('in_channels', 1)
    T = vcfg.get('num_frames', 16)
    S = vcfg.get('frame_size', 32)
    cmap = 'gray' if C == 1 else None

    sample_dir = vcfg.get('sample_dir', 'assets/video_samples/')
    os.makedirs(sample_dir, exist_ok=True)

    # Conditional sampling: pull in-distribution attribute vectors from the
    # dataset so the model knows which square trajectory to nucleate.
    y_cond = None
    if getattr(model, 'cond_features', 0) > 0:
        ds = build_video_dataset({'video': vcfg})
        y_cond = torch.stack([ds[i][1] for i in range(n)]).to(device).float()

    x = torch.randn(n, C, T, S, S, device=device)
    for t_val in reversed(range(cfg['T'])):
        t_t = torch.full((n,), t_val, device=device, dtype=torch.long)
        x   = scheduler.sample_prev_timestep(x, model(x, t_t, y_cond), t_val)

    vids = ((x.clamp(-1, 1) + 1) / 2).cpu()              # (n, C, T, S, S) in [0,1]

    # ── montage: rows = clips, cols = frames ─────────────────────────────────────
    fig, axes = plt.subplots(n, T, figsize=(T * 1.0, n * 1.0))
    axes = axes.reshape(n, T)
    for i in range(n):
        for f in range(T):
            frame = vids[i, :, f]
            frame = frame[0].numpy() if C == 1 else frame.permute(1, 2, 0).numpy()
            axes[i, f].imshow(frame, cmap=cmap)
            axes[i, f].axis('off')
    plt.suptitle('VideoDiT — generated clips', fontsize=12)
    plt.tight_layout()
    grid_path = f"{sample_dir}final_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close('all')
    print(f"Saved {grid_path}")

    # ── animated gif of clip 0 ───────────────────────────────────────────────────
    if save_gif:
        frames = []
        for f in range(T):
            arr = vids[0, :, f]
            if C == 1:
                arr = (arr[0].numpy() * 255).astype('uint8')
                im  = Image.fromarray(arr, mode='L')
            else:
                arr = (arr.permute(1, 2, 0).numpy() * 255).astype('uint8')
                im  = Image.fromarray(arr, mode='RGB')
            frames.append(im.resize((128, 128), Image.NEAREST))
        gif_path = f"{sample_dir}sample.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=120, loop=0)
        print(f"Saved {gif_path} ({len(frames)} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   default='checkpoints/video/final_video_dit_ema.pth')
    parser.add_argument('--cfg',    default='config.yaml')
    parser.add_argument('--n',      type=int, default=4)
    parser.add_argument('--no-gif', action='store_true')
    # Optional architecture overrides — must match the trained checkpoint
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--frame_size', type=int, default=None)
    parser.add_argument('--in_channels', type=int, default=None)
    args = parser.parse_args()
    overrides = {
        'num_frames':  args.num_frames,
        'frame_size':  args.frame_size,
        'in_channels': args.in_channels,
    }
    sample(args.ckpt, args.cfg, args.n, not args.no_gif, overrides)
