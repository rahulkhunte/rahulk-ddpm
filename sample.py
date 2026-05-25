import yaml, torch, argparse
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model     import UNet
from scheduler import CosineNoiseScheduler

def sample(ckpt: str, cfg_path: str = 'config.yaml',
           n: int = 16, save_gif: bool = True):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = UNet(cfg['in_channels'], cfg['base_channels'], cfg['time_dim']).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    scheduler = CosineNoiseScheduler(cfg['T'], device=device)
    C    = cfg['in_channels']
    cmap = 'gray' if C == 1 else None
    frames = []

    with torch.no_grad():
        x = torch.randn(n, C, cfg['image_size'], cfg['image_size'], device=device)
        for t_val in reversed(range(cfg['T'])):
            t_t = torch.full((n,), t_val, device=device, dtype=torch.long)
            x   = scheduler.sample_prev_timestep(x, model(x, t_t), t_val)
            if save_gif and t_val % 40 == 0:
                img = ((x[0, 0].cpu().clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
                frames.append(Image.fromarray(img).resize((128, 128), Image.NEAREST))

    # Save grid
    imgs = ((x.clamp(-1, 1) + 1) / 2).cpu()
    if C == 1:
        grid = imgs.view(n, cfg['image_size'], cfg['image_size']).numpy()
    else:
        grid = imgs.permute(0, 2, 3, 1).numpy()  # (N, H, W, C) for RGB

    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid[i], cmap=cmap); ax.axis('off')
    plt.suptitle('DDPM — Generated Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig('assets/final_samples.png', dpi=150)
    plt.close('all')
    print("✅ Saved assets/final_samples.png")

    # Save GIF
    if save_gif and frames:
        frames[0].save('assets/denoising.gif', save_all=True,
                       append_images=frames[1:], duration=60, loop=0)
        print(f"✅ Saved assets/denoising.gif ({len(frames)} frames)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   default='checkpoints/final_model.pth')
    parser.add_argument('--cfg',    default='config.yaml')
    parser.add_argument('--n',      type=int, default=16)
    parser.add_argument('--no-gif', action='store_true')
    args = parser.parse_args()
    sample(args.ckpt, args.cfg, args.n, not args.no_gif)
