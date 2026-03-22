import os, gc, yaml, torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model     import UNet
from scheduler import LinearNoiseScheduler

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
    scheduler = LinearNoiseScheduler(cfg['T'], cfg['beta_start'],
                                     cfg['beta_end'], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    losses = []
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs  = imgs.to(device)
            t     = torch.randint(0, cfg['T'], (imgs.size(0),), device=device)
            noise = torch.randn_like(imgs)
            pred  = model(scheduler.add_noise(imgs, noise, t), t)
            loss  = criterion(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"Epoch [{epoch+1:3d}/{cfg['epochs']}] Loss: {avg:.4f}")

        if (epoch + 1) % cfg['save_every'] == 0:
            m = model.module if isinstance(model, nn.DataParallel) else model
            m.eval()
            torch.save(m.state_dict(),
                       f"{cfg['checkpoint_dir']}ddpm_epoch_{epoch+1}.pth")
            _save_samples(m, scheduler, device, cfg, epoch + 1)
            torch.cuda.empty_cache(); gc.collect()
            m.train()

    # Final saves
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), f"{cfg['checkpoint_dir']}final_model.pth")
    _save_loss_plot(losses)
    print(f"✅ Done! Best loss: {min(losses):.4f}")

def _save_samples(model, scheduler, device, cfg, epoch):
    with torch.no_grad():
        x = torch.randn(16, cfg['in_channels'],
                        cfg['image_size'], cfg['image_size'], device=device)
        for t_val in reversed(range(cfg['T'])):
            t_t = torch.full((16,), t_val, device=device, dtype=torch.long)
            x   = scheduler.sample_prev_timestep(x, model(x, t_t), t_val)
    grid = ((x.clamp(-1,1)+1)/2).cpu().view(16, cfg['image_size'],
                                              cfg['image_size']).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid[i], cmap='gray'); ax.axis('off')
    plt.suptitle(f'Epoch {epoch}'); plt.tight_layout()
    plt.savefig(f"{cfg['sample_dir']}epoch_{epoch:03d}.png", dpi=100)
    plt.close('all')

def _save_loss_plot(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='royalblue', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('DDPM Training Loss'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/training_loss.png', dpi=150)
    plt.close('all')

if __name__ == '__main__':
    train()
