import os, gc, copy, yaml, torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model     import UNet
from scheduler import CosineNoiseScheduler


class EMA:
    """
    Exponential Moving Average of model weights.

    Standard practice in all production diffusion models
    (DDPM, DALL-E 2, Stable Diffusion, DiT).

    EMA maintains a shadow copy of weights updated as:
        θ_ema ← decay · θ_ema + (1 - decay) · θ

    Why it matters:
      - Training weights oscillate around optima due to SGD noise
      - EMA weights are a smoothed average → more stable samples
      - At inference we always use EMA weights, not raw model weights
      - Nichol & Dhariwal (2021) show EMA is critical for sample quality
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay      = decay
        self.shadow     = copy.deepcopy(model).eval()
        # EMA model never needs gradients
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_model(self) -> nn.Module:
        return self.shadow


def train(cfg_path: str = 'config.yaml'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['sample_dir'],     exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(cfg['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset    = datasets.MNIST(root='data/', train=True,
                                download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'],
                            shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=cfg['pin_memory'])

    model     = UNet(cfg['in_channels'], cfg['base_channels'], cfg['time_dim']).to(device)
    ema       = EMA(model, decay=cfg.get('ema_decay', 0.9999))

    # ── Cosine schedule (Nichol & Dhariwal, 2021) ───────────────────────────
    # Replaces linear β schedule — decays ᾱ_t smoothly so signal is preserved
    # at small t, avoiding the abrupt information destruction of linear schedule.
    scheduler = CosineNoiseScheduler(cfg['T'], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()

    # ── ELBO note ────────────────────────────────────────────────────────────
    # Full ELBO: L = L_T + Σ L_{t-1} + L_0
    # L_{t-1} = KL( q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) )
    # Simplified (Ho et al.): L_simple = E||ε - ε_θ(x_t, t)||²
    # We optimise L_simple — equivalent to reweighted ELBO (Eq.14 in paper)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    losses, ema_losses = [], []

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0

        for imgs, _ in dataloader:
            imgs  = imgs.to(device)
            t     = torch.randint(0, cfg['T'], (imgs.size(0),), device=device)
            noise = torch.randn_like(imgs)

            # Forward diffusion: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
            xt   = scheduler.add_noise(imgs, noise, t)
            pred = model(xt, t)
            loss = criterion(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clip
            optimizer.step()

            # Update EMA after every step
            m = model.module if isinstance(model, nn.DataParallel) else model
            ema.update(m)

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"Epoch [{epoch+1:3d}/{cfg['epochs']}] Loss: {avg:.4f}  (EMA model active)")

        if (epoch + 1) % cfg['save_every'] == 0:
            m = model.module if isinstance(model, nn.DataParallel) else model
            # Save both raw model and EMA model
            torch.save(m.state_dict(),
                       f"{cfg['checkpoint_dir']}ddpm_epoch_{epoch+1}.pth")
            torch.save(ema.get_model().state_dict(),
                       f"{cfg['checkpoint_dir']}ddpm_ema_epoch_{epoch+1}.pth")

            # Always sample from EMA model — smoother outputs
            ema.get_model().eval()
            _save_samples(ema.get_model(), scheduler, device, cfg, epoch + 1)
            torch.cuda.empty_cache(); gc.collect()

    # Final saves
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), f"{cfg['checkpoint_dir']}final_model.pth")
    torch.save(ema.get_model().state_dict(), f"{cfg['checkpoint_dir']}final_ema_model.pth")
    _save_loss_plot(losses)
    print(f"✅ Done! Final loss: {losses[-1]:.4f}  Best loss: {min(losses):.4f}")


def _save_samples(model, scheduler, device, cfg, epoch):
    with torch.no_grad():
        x = torch.randn(16, cfg['in_channels'],
                        cfg['image_size'], cfg['image_size'], device=device)
        for t_val in reversed(range(cfg['T'])):
            t_t = torch.full((16,), t_val, device=device, dtype=torch.long)
            x   = scheduler.sample_prev_timestep(x, model(x, t_t), t_val)
    grid = ((x.clamp(-1, 1) + 1) / 2).cpu().view(
        16, cfg['image_size'], cfg['image_size']).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid[i], cmap='gray'); ax.axis('off')
    plt.suptitle(f'Epoch {epoch} — EMA samples'); plt.tight_layout()
    plt.savefig(f"{cfg['sample_dir']}epoch_{epoch:03d}.png", dpi=100)
    plt.close('all')


def _save_loss_plot(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='royalblue', linewidth=2, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('DDPM Training Loss (Cosine Schedule + EMA)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('assets/training_loss.png', dpi=150)
    plt.close('all')


if __name__ == '__main__':
    train()
