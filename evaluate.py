"""
evaluate.py — Sample quality evaluation for trained DDPM checkpoint.

Metrics implemented:
  1. FID (Fréchet Inception Distance) — gold standard for generative models
     Measures distributional similarity between real and generated images.
     Lower is better. State-of-the-art on MNIST: FID < 2.0

  2. Sampling diversity — visual grid of 64 generated samples

Usage:
  python evaluate.py --ckpt checkpoints/final_ema_model.pth --n_samples 1000

FID math:
  FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r·Σ_g)^½)
  where (μ_r, Σ_r) and (μ_g, Σ_g) are mean/cov of Inception features
  of real and generated images respectively.
"""

import argparse, yaml, torch
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm

from model     import UNet
from scheduler import CosineNoiseScheduler


# ── Minimal Inception feature extractor (uses torchvision) ────────────────────
def get_inception_features(images: torch.Tensor, device: str) -> np.ndarray:
    """Extract 2048-d Inception-v3 features from a batch of RGB images."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    inc = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    inc.fc = torch.nn.Identity()           # remove classifier head
    inc = inc.to(device).eval()

    # Inception expects 3-channel 299×299
    resize = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x)
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            batch = resize(images[i:i+64].to(device))
            feats.append(inc(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·√(Σ_r·Σ_g))"""
    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sig_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)
    diff    = mu_r - mu_g
    cov_sqrt, _ = sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid = diff @ diff + np.trace(sig_r + sig_g - 2 * cov_sqrt)
    return float(fid)


@torch.no_grad()
def generate_samples(model, scheduler, cfg, n_samples: int, device: str) -> torch.Tensor:
    """Full reverse diffusion: x_T ~ N(0,I) → x_0"""
    model.eval()
    all_samples = []
    batch = 64
    for start in range(0, n_samples, batch):
        bs = min(batch, n_samples - start)
        x  = torch.randn(bs, cfg['in_channels'],
                         cfg['image_size'], cfg['image_size'], device=device)
        for t_val in reversed(range(cfg['T'])):
            t_t = torch.full((bs,), t_val, device=device, dtype=torch.long)
            x   = scheduler.sample_prev_timestep(x, model(x, t_t), t_val)
        all_samples.append(x.cpu())
        print(f"  Generated {min(start+batch, n_samples)}/{n_samples}", end='\r')
    return ((torch.cat(all_samples, 0).clamp(-1, 1) + 1) / 2)   # [0,1]


def save_sample_grid(samples: torch.Tensor, path: str = 'assets/eval_samples.png'):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze().numpy(), cmap='gray'); ax.axis('off')
    plt.suptitle('Generated Samples (EMA model)', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close('all')
    print(f"\n  Sample grid saved → {path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DDPM checkpoint')
    parser.add_argument('--ckpt',      default='checkpoints/final_ema_model.pth')
    parser.add_argument('--cfg',       default='config.yaml')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--fid',       action='store_true', help='Compute FID score')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Generating {args.n_samples} samples...")

    model = UNet(cfg['in_channels'], cfg['base_channels'], cfg['time_dim']).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    scheduler = CosineNoiseScheduler(cfg['T'], device=device)
    samples   = generate_samples(model, scheduler, cfg, args.n_samples, device)

    save_sample_grid(samples[:64])

    if args.fid:
        print("\nComputing FID (loading real MNIST samples)...")
        transform = transforms.Compose([
            transforms.Resize(cfg['image_size']),
            transforms.ToTensor(),
        ])
        real_data   = datasets.MNIST(root='data/', train=False,
                                     download=True, transform=transform)
        real_loader = DataLoader(real_data, batch_size=args.n_samples, shuffle=True)
        real_imgs   = next(iter(real_loader))[0]

        print("  Extracting Inception features (real)...")
        real_feats = get_inception_features(real_imgs, device)
        print("  Extracting Inception features (generated)...")
        gen_feats  = get_inception_features(samples, device)

        fid = compute_fid(real_feats, gen_feats)
        print(f"\n{'='*40}")
        print(f"  FID Score: {fid:.2f}")
        print(f"  (Lower is better — SOTA MNIST ≈ 0.5-2.0)")
        print(f"{'='*40}")
    else:
        print(f"\n✅ Sample grid saved. Run with --fid flag to compute FID score.")


if __name__ == '__main__':
    main()
