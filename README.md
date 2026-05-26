# rahulk-ddpm

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Denoising Diffusion Probabilistic Models (Ho et al., 2020) implemented from scratch in PyTorch.  
No diffusers library. No pretrained weights. Just math → code → results.

> 🔥 Generating handwritten digits from pure Gaussian noise — trained on MNIST in ~1 hour on free Kaggle GPUs.  
> ✅ Upgraded: **Cosine noise schedule** + **EMA weights** + **DiT block stub** + **FID evaluation**

---

## Results

### Denoising Process — xT → x₀
![denoise](assets/denoising.gif)

### Forward Process — x₀ → xT (pure noise)
![forward](assets/forward_diffusion.png)

### Training Progression
| Epoch 10 | Epoch 20 | Epoch 30 | Epoch 40 |
|:--------:|:--------:|:--------:|:--------:|
| ![e10](assets/samples/epoch_010.png) | ![e20](assets/samples/epoch_020.png) | ![e30](assets/samples/epoch_030.png) | ![e40](assets/samples/epoch_040.png) |

### Final Generated Samples (Epoch 40)
![final](assets/final_samples.png)

---

## How DDPM Works

**Forward process** — gradually destroy an image with Gaussian noise over T=1000 steps

**Reverse process** — UNet learns to predict and remove noise step by step

**Key insight** — we predict noise ε (not the image directly) because we know the exact noise added at each step. This makes training stable with a simple objective derived from the ELBO:

```
# Full ELBO: L = L_T + Σ L_{t-1} + L_0
# L_{t-1} = KL( q(x_{t-1}|x_t,x0) || p_θ(x_{t-1}|x_t) )
# Simplified (Ho et al., Eq.14): L_simple = E||ε - ε_θ(x_t, t)||²
Loss = MSE(predicted_noise, actual_noise)
```

---

## Noise Schedule: Linear → Cosine

The original DDPM uses a **linear β schedule** which increases β uniformly.  
This implementation upgrades to the **cosine schedule** (Nichol & Dhariwal, 2021):

```
ā_t = cos²( (t/T + s) / (1+s) · π/2 )    s = 0.008
```

**Why cosine is better:**
| | Linear β | Cosine ā_t |
|---|---|---|
| Signal at low t | Destroyed too fast | Preserved longer |
| Noise at high t | Sometimes too noisy | Smooth decay to ~0 |
| Sample quality | Baseline | +10-15% FID improvement |

---

## EMA (Exponential Moving Average)

Training uses EMA shadow weights updated every step:

```
θ_ema ← 0.9999 · θ_ema + 0.0001 · θ_train
```

**Inference always uses EMA weights** — smoother, more stable samples.  
Standard in DDPM, DALL-E 2, Stable Diffusion, and DiT.

---

## Architecture

```
Input (xₜ, t)
│
├── SinusoidalTimeEmbedding(t) → injected at every ResBlock
│
[Encoder]
  ResBlock(1→64)    ──────────────────────── skip_1
  ResBlock(64→128)  ──────────────────────── skip_2
[Bottleneck]
  ResBlock(128→256)
  SelfAttention(256)
  ResBlock(256→128)
[Decoder]
  ConvTranspose + skip_2 → ResBlock(256→64)
  ConvTranspose + skip_1 → ResBlock(128→64)
│
Conv1x1 → ε_pred
```

**Parameters: 3.6M**

---

## DDPM → LDM → DiT: The Full Progression

This repo implements **pixel-space DDPM** as the foundation. The modern pipeline extends it:

```
DDPM (this repo)
  └── Noise predictor = UNet operating on raw pixels
  └── Forward/reverse diffusion in pixel space

LDM — Latent Diffusion Models (Rombach et al., 2022 — Stable Diffusion)
  └── VAE encoder compresses image → latent z (8-16× smaller)
  └── DDPM runs in latent space, not pixel space
  └── VAE decoder reconstructs image from denoised latent
  └── 4-8× faster training, same quality

DiT — Diffusion Transformer (Peebles & Xie, 2023)
  └── Replaces UNet noise predictor with Vision Transformer
  └── Image split into patches (like ViT), processed as token sequence
  └── Conditioning via AdaLayerNorm (timestep + class embedding)
  └── Scales better than UNet: DiT-XL achieves FID 2.27 on ImageNet 256×256
  └── Full pipeline: Image → VAE → latent → patchify → DiT blocks → unpatchify → VAE⁻¹ → Image
```

See [`model/dit_block.py`](model/dit_block.py) for a fully annotated DiT block implementation.

---

## Project Structure

```
rahulk-ddpm/
├── model/
│   ├── __init__.py
│   ├── time_embedding.py     # Sinusoidal embeddings
│   ├── resblock.py           # ResNet blocks + time conditioning
│   ├── attention.py          # Self-attention at bottleneck
│   ├── unet.py               # Full UNet noise predictor
│   └── dit_block.py          # DiT block stub (AdaLN + Attention + MLP)
├── scheduler/
│   ├── __init__.py
│   └── noise_scheduler.py    # Cosine β schedule, forward + reverse
├── train.py                  # Training loop + EMA
├── sample.py                 # Reverse diffusion + GIF export
├── evaluate.py               # FID evaluation + sample grid
├── config.yaml               # Hyperparameters
└── assets/
    ├── denoising.gif
    ├── forward_diffusion.png
    ├── final_samples.png
    └── samples/
```

---

## Quickstart

```bash
git clone https://github.com/rahulkhunte/rahulk-ddpm.git
cd rahulk-ddpm
pip install torch torchvision pyyaml pillow matplotlib scipy
```

**Train from scratch:**
```bash
python train.py
```

**Generate samples + GIF from checkpoint:**
```bash
python sample.py --ckpt checkpoints/final_ema_model.pth
```

**Evaluate — sample grid:**
```bash
python evaluate.py --ckpt checkpoints/final_ema_model.pth
```

**Evaluate — with FID score:**
```bash
python evaluate.py --ckpt checkpoints/final_ema_model.pth --fid --n_samples 1000
```

---

## Training Details

| Config | Value |
|--------|-------|
| Dataset | MNIST 32×32 |
| Training samples | 60,000 |
| Diffusion steps T | 1,000 |
| β schedule | **Cosine** (Nichol & Dhariwal, 2021) |
| EMA decay | 0.9999 |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | Adam + grad clip (max norm 1.0) |
| Hardware | Kaggle 2×T4 GPU |
| Final loss | ~0.0155 MSE |

---

## References

Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. NeurIPS 2020.  
https://arxiv.org/abs/2006.11239

Nichol, A., & Dhariwal, P. (2021). **Improved Denoising Diffusion Probabilistic Models**. ICML 2021.  
https://arxiv.org/abs/2102.09672

Peebles, W., & Xie, S. (2023). **Scalable Diffusion Models with Transformers (DiT)**. ICCV 2023.  
https://arxiv.org/abs/2212.09748

Rombach, R. et al. (2022). **High-Resolution Image Synthesis with Latent Diffusion Models**. CVPR 2022.  
https://arxiv.org/abs/2112.10752

---

**Rahul Khunte** — [github.com/rahulkhunte](https://github.com/rahulkhunte)
