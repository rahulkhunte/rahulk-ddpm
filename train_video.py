"""
train_video.py — minimal video DDPM training loop.

Reuses the existing image pipeline pieces unchanged:
  - EMA                  (train.py)              — shadow weights
  - CosineNoiseScheduler (scheduler/)            — forward / reverse diffusion
  - VideoDiT             (model/video_dit.py)    — spatiotemporal noise predictor

Objective is identical to the image DDPM (Ho et al., Eq.14):
    L_simple = E || ε - ε_θ(x_t, t) ||²

Video tensors are shaped (B, C, T, H, W). The scheduler's `add_noise` is
written for 4-D tensors, so we fold (B, C, T, H, W) → (B, C*T, H, W) before
calling it (the same timestep t applies to every frame of a sample), then
reshape back. The reverse step `sample_prev_timestep` uses scalar coefficients
and already works on 5-D tensors directly.
"""

import os, gc, copy, yaml, argparse
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model.video_dit       import VideoDiT
from scheduler             import CosineNoiseScheduler
from datasets.video_dataset import build_video_dataset
from train                 import EMA          # reuse existing EMA implementation


def add_video_noise(scheduler, x0, noise, t):
    """Forward diffusion for video by folding time into channels (4-D add_noise)."""
    B, C, T, H, W = x0.shape
    xt = scheduler.add_noise(x0.reshape(B, C * T, H, W),
                             noise.reshape(B, C * T, H, W), t)
    return xt.reshape(B, C, T, H, W)


def build_model(vcfg, time_dim, device):
    return VideoDiT(
        in_channels=vcfg.get('in_channels', 1),
        num_frames=vcfg.get('num_frames', 16),
        image_size=vcfg.get('frame_size', 32),
        patch_size=vcfg.get('patch_size', 4),
        patch_t=vcfg.get('patch_t', 2),
        hidden_dim=vcfg.get('hidden_dim', 256),
        depth=vcfg.get('depth', 4),
        num_heads=vcfg.get('num_heads', 4),
        time_dim=time_dim,
        cond_features=vcfg.get('cond_features', 0),
    ).to(device)


def train(cfg_path: str = 'config.yaml', overrides: dict = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    vcfg = dict(cfg.get('video', {}))
    if overrides:
        vcfg.update({k: v for k, v in overrides.items() if v is not None})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    ckpt_dir   = vcfg.get('checkpoint_dir', 'checkpoints/video/')
    sample_dir = vcfg.get('sample_dir',     'assets/video_samples/')
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    dataset = build_video_dataset({'video': vcfg})
    dataloader = DataLoader(dataset, batch_size=vcfg.get('batch_size', 8),
                            shuffle=True, num_workers=cfg.get('num_workers', 4),
                            pin_memory=cfg.get('pin_memory', False) and device == 'cuda')
    print(f"Dataset: {vcfg.get('dataset', 'synthetic')}  |  {len(dataset)} clips  |  "
          f"shape (C={vcfg.get('in_channels',1)}, T={vcfg.get('num_frames',16)}, "
          f"{vcfg.get('frame_size',32)}x{vcfg.get('frame_size',32)})", flush=True)

    # ── Model / scheduler / optim ───────────────────────────────────────────────
    model     = build_model(vcfg, cfg.get('time_dim', 256), device)
    # EMA decay: video runs are short (few k steps), so 0.9999 leaves the shadow
    # weights dominated by the random init (0.9999^9600 ≈ 0.38). Allow the video
    # config to lower it (e.g. 0.999) so EMA actually tracks the trained weights.
    ema_decay = float(vcfg.get('ema_decay', cfg.get('ema_decay', 0.9999)))
    ema       = EMA(model, decay=ema_decay)
    print(f"EMA decay: {ema_decay}", flush=True)
    scheduler = CosineNoiseScheduler(cfg['T'], s=cfg.get('cosine_s', 0.008), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=vcfg.get('learning_rate', 1e-4))
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VideoDiT params: {n_params/1e6:.2f}M", flush=True)

    # ── Training loop ────────────────────────────────────────────────────────────
    epochs    = vcfg.get('epochs', 5)
    save_every = vcfg.get('save_every', 1)
    # Low-t loss weighting: emphasise the low/mid-noise steps where structure is
    # reconstructed. w(t) = 1 + λ·(1 - t/T), mean-normalised so the LR scale is
    # unchanged. High-t steps keep weight 1 (no starvation of the prior).
    t_weighting = vcfg.get('loss_t_weighting', False)
    t_weight_lambda = float(vcfg.get('loss_t_weight_lambda', 1.0))
    # min-SNR-gamma weighting (Hang et al. 2023): for eps-prediction, scales the
    # per-sample loss by min(SNR,gamma)/SNR. Down-weights easy low-noise steps and
    # keeps full weight on high-noise steps — the regime where structure must be
    # nucleated from pure noise. Takes precedence over low-t weighting if set.
    min_snr_gamma = vcfg.get('min_snr_gamma', None)
    if min_snr_gamma is not None:
        min_snr_gamma = float(min_snr_gamma)
        print(f"Loss weighting: min-SNR-gamma (gamma={min_snr_gamma})", flush=True)
    elif t_weighting:
        print(f"Loss t-weighting ON  (lambda={t_weight_lambda}, low-t emphasised)", flush=True)
    losses    = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for videos, y in dataloader:
            videos = videos.to(device)                       # (B, C, T, H, W)
            t      = torch.randint(0, cfg['T'], (videos.size(0),), device=device)
            noise  = torch.randn_like(videos)
            # Conditioning vector (B, cond_features) when the synthetic dataset
            # provides one and the model was built conditional; else unconditional.
            y_cond = y.to(device).float() if (model.cond_features > 0
                                              and torch.is_tensor(y) and y.dim() == 2) else None

            xt   = add_video_noise(scheduler, videos, noise, t)
            pred = model(xt, t, y_cond)
            if min_snr_gamma is not None:
                per_sample = ((pred - noise) ** 2).flatten(1).mean(dim=1)   # (B,)
                snr = scheduler.alpha_bar[t] / (1.0 - scheduler.alpha_bar[t])
                w = torch.clamp(snr, max=min_snr_gamma) / snr               # min-SNR
                w = w / w.mean()
                loss = (w * per_sample).mean()
            elif t_weighting:
                per_sample = ((pred - noise) ** 2).flatten(1).mean(dim=1)   # (B,)
                w = 1.0 + t_weight_lambda * (1.0 - t.float() / cfg['T'])
                w = w / w.mean()                                            # keep scale
                loss = (w * per_sample).mean()
            else:
                loss = criterion(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {avg:.4f}  (EMA active)", flush=True)

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(),           f"{ckpt_dir}video_dit_epoch_{epoch+1}.pth")
            torch.save(ema.get_model().state_dict(), f"{ckpt_dir}video_dit_ema_epoch_{epoch+1}.pth")
            ema.get_model().eval()
            _save_video_samples(ema.get_model(), scheduler, device, cfg, vcfg,
                                sample_dir, epoch + 1)
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    torch.save(model.state_dict(),           f"{ckpt_dir}final_video_dit.pth")
    torch.save(ema.get_model().state_dict(), f"{ckpt_dir}final_video_dit_ema.pth")
    if losses:
        print(f"Done! Final loss: {losses[-1]:.4f}  Best loss: {min(losses):.4f}", flush=True)


@torch.no_grad()
def _save_video_samples(model, scheduler, device, cfg, vcfg, sample_dir, epoch, n: int = 2):
    """Generate a couple of clips and save a (clip × frame) montage."""
    C = vcfg.get('in_channels', 1)
    T = vcfg.get('num_frames', 16)
    S = vcfg.get('frame_size', 32)
    cmap = 'gray' if C == 1 else None

    # When conditional, sample with in-distribution attribute vectors pulled
    # straight from the dataset (guaranteed valid trajectories).
    y_cond = None
    if getattr(model, 'cond_features', 0) > 0:
        ds = build_video_dataset({'video': vcfg})
        y_cond = torch.stack([ds[i][1] for i in range(n)]).to(device).float()

    x = torch.randn(n, C, T, S, S, device=device)
    for t_val in reversed(range(cfg['T'])):
        t_t = torch.full((n,), t_val, device=device, dtype=torch.long)
        x   = scheduler.sample_prev_timestep(x, model(x, t_t, y_cond), t_val)

    vids = ((x.clamp(-1, 1) + 1) / 2).cpu()              # (n, C, T, S, S) in [0,1]

    fig, axes = plt.subplots(n, T, figsize=(T * 1.0, n * 1.0))
    axes = axes.reshape(n, T)
    for i in range(n):
        for f in range(T):
            frame = vids[i, :, f]
            frame = frame[0].numpy() if C == 1 else frame.permute(1, 2, 0).numpy()
            axes[i, f].imshow(frame, cmap=cmap)
            axes[i, f].axis('off')
    plt.suptitle(f'Epoch {epoch} — EMA video samples')
    plt.tight_layout()
    plt.savefig(f"{sample_dir}epoch_{epoch:03d}.png", dpi=100)
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',         default='config.yaml')
    parser.add_argument('--dataset',     default=None, help='synthetic | movingmnist')
    parser.add_argument('--epochs',      type=int, default=None)
    parser.add_argument('--batch_size',  type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_frames',  type=int, default=None)
    parser.add_argument('--frame_size',  type=int, default=None)
    parser.add_argument('--save_every',  type=int, default=None)
    parser.add_argument('--ema_decay',   type=float, default=None)
    parser.add_argument('--cond_features', type=int, default=None,
                        help='label-conditioning dim; 0 = unconditional')
    parser.add_argument('--data_root',   default=None, help='dataset root (e.g. a persisted volume path)')
    parser.add_argument('--min_snr_gamma', type=float, default=None,
                        help='min-SNR-gamma loss weighting (e.g. 5); emphasises high-t')
    parser.add_argument('--loss_t_weighting', dest='loss_t_weighting',
                        action='store_true', default=None,
                        help='emphasise low/mid-t steps in the loss')
    args = parser.parse_args()

    overrides = {
        'dataset':     args.dataset,
        'epochs':      args.epochs,
        'batch_size':  args.batch_size,
        'num_samples': args.num_samples,
        'num_frames':  args.num_frames,
        'frame_size':  args.frame_size,
        'save_every':  args.save_every,
        'ema_decay':   args.ema_decay,
        'min_snr_gamma': args.min_snr_gamma,
        'loss_t_weighting': args.loss_t_weighting,
        'cond_features': args.cond_features,
        'data_root':   args.data_root,
    }
    train(args.cfg, overrides)
