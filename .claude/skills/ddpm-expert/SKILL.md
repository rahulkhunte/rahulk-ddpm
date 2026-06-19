---
name: DDPM Expert
description: >
  Use this skill whenever the task involves diffusion models — DDPM, DDIM, latent
  diffusion (LDM), Diffusion Transformers (DiT), or video diffusion. Triggers
  include: implementing or debugging forward/reverse diffusion, noise schedulers
  (linear, cosine), UNet or DiT noise predictors, sinusoidal time embeddings, EMA,
  the reparameterization trick, ELBO / Lsimple loss, sampling loops (Algorithm 2),
  FID evaluation, temporal attention for video, or extending image diffusion to
  video for a pedagogy/text-to-video model. Also use when working on the
  rahulk-ddpm repository or the Zulense Z1 Imagination Engine. This skill encodes
  the exact conventions, file structure, and architectural decisions used in
  rahulk-ddpm so generated code matches the existing codebase.
---

# DDPM Expert

This skill makes Claude a precise collaborator on diffusion-model work, matched to
the conventions of the `rahulk-ddpm` repository and the Z1 video-pedagogy goal.

## Operating principles

1. **Math grounds every code change.** Before writing or editing a diffusion
   component, state which equation it implements (cite by number, e.g. "Eq. 4 —
   closed-form forward"). Never write a scheduler or loss without naming the math.
2. **Connect theory to the existing files.** When explaining a concept, point to
   the file in the repo that implements it (e.g. "this is `noise_scheduler.py`").
3. **Predict noise, not images.** The training target is always ε. Loss is MSE on
   predicted noise (`Lsimple`). Flag any deviation from this as a deliberate choice.
4. **EMA at inference, always.** Sampling and evaluation use EMA weights, never raw
   training weights. If a sampling script loads non-EMA weights, flag it.
5. **Be honest about hardware limits.** CPU (Oracle A1) is for data prep, light
   training, hosting. GPU (Modal credits, Kaggle T4) is for real training runs.
   Don't suggest full video-diffusion training on CPU.

## Core math reference (use exact notation)

```
Forward one step (Eq. 2):
  q(xₜ|xₜ₋₁) = N(xₜ ; √(1-βₜ)·xₜ₋₁ , βₜI)
  code form: xₜ = √αₜ·xₜ₋₁ + √(1-αₜ)·ε ,  αₜ = 1-βₜ

Closed form / reparameterization (Eq. 4):
  xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε ,  ᾱₜ = Π αₛ ,  ε ~ N(0,I)
  - ε sampled OUTSIDE computation → gradients flow → training possible
  - ground-truth noise known → loss is simple MSE

Reverse mean (Eq. 11):
  μθ = (1/√αₜ)·(xₜ - (βₜ/√(1-ᾱₜ))·εθ(xₜ,t))

Sampling step (Algorithm 2):
  xₜ₋₁ = μθ + σₜ·z ,  z ~ N(0,I) if t>1 else z=0   (z=0 at last step → clean image)

Loss (Eq. 14, Lsimple):
  L = E_{x₀,ε,t} [ ‖ε - εθ(xₜ,t)‖² ]

ELBO three terms (Eq. 5):
  L_T   = DKL(q(xT|x₀)‖N(0,I))   → ~0 because β fixed → ignored in training
  L_{t-1} = DKL(true posterior ‖ pθ)  → main term → reduces to MSE on noise
  L_0   = -log pθ(x₀|x₁)          → small reconstruction term
  Lsimple drops the weighting in L_{t-1}; empirically better samples (FID 3.17 vs 13.51)
```

## Schedulers

```
Linear (Ho et al. 2020):
  βₜ = linspace(1e-4, 0.02, T)

Cosine (Nichol & Dhariwal 2021) — preferred for complex / RGB data:
  ᾱₜ = cos²( ((t/T + s)/(1+s)) · π/2 ) ,  s = 0.008
  derive βₜ from ᾱₜ ratios, clamp to (1e-4, 0.999)
  reason: linear destroys signal too fast early; cosine decays smoothly to ~0
```

## EMA (always use at inference)

```
θ_ema ← decay·θ_ema + (1-decay)·θ   ,  decay = 0.9999
- shadow copy, updated every step, no grad
- save both raw and EMA checkpoints
- sample.py / evaluate.py must load the EMA weights
```

## Architectures

```
UNet noise predictor (pixel space, ~3.6M params):
  SinusoidalTimeEmbedding(t) injected at EVERY ResBlock
  Encoder ResBlocks (skip connections) → bottleneck ResBlock + SelfAttention
  → Decoder ConvTranspose + skips → Conv1x1 → ε_pred

DiT block (latent space, scales to 675M — Peebles & Xie 2023):
  patchify (B,C,H,W)→(B,N,D)
  AdaLayerNorm(x, cond)  where cond = time (+ optional class) embedding
    γ,β = Linear(cond); AdaLN(x) = (1+γ)·LayerNorm(x) + β ; zero-init the proj
  block: x → AdaLN → MHA → +res → AdaLN → MLP(GELU,4×) → +res
  conditioning via AdaLN, NOT cross-attention (cheaper, equally effective)

Full modern pipeline (Z1 target):
  Image/frame → VAE encoder → latent z → patchify → DiT blocks → unpatchify
              → VAE decoder → output
```

## Video diffusion extension (current frontier for Z1)

```
Goal: extend image diffusion → temporally consistent video.
Same forward/reverse math, same Lsimple. New piece = TEMPORAL modeling.

Tensor shape: image (B,C,H,W) → video (B,T,C,H,W) , T = frames.

Temporal attention (the key new block):
  After spatial attention, reshape so each patch attends across frames at the
  same spatial position:
    (B,T,N,D) → (B·N, T, D) → MultiheadAttention over T → reshape back
  This enforces frame-to-frame coherence (prevents flicker / text drift).

Attention variants (cite when relevant):
  - full spatio-temporal: attend all patches in all frames (expensive, best motion)
  - causal: attend only to current + previous frames (no future leak)
  - sparse causal: attend to first + immediately-preceding frame (cheap)

Key papers to ground video work:
  Ho et al. 2022 — Video Diffusion Models (3D UNet, temporal attention)  arxiv 2204.03458
  Blattmann et al. 2023 — Align Your Latents (Video LDM)                arxiv 2304.08818
  Peebles & Xie 2023 — DiT (transformer backbone)                       arxiv 2212.09748
  Sora technical report 2024 — spacetime patches, DiT at scale

Z1-specific framing (pedagogy / NCERT text-to-video):
  - text/content hallucination is solved at the CONTENT layer (retrieval/grounding),
    stroke/visual hallucination at the RENDERING layer (the diffusion model)
  - AdaLN can inject stroke/glyph conditioning the same way it injects timestep
  - keep these two concerns separate when reasoning about the system
```

## rahulk-ddpm repo conventions

```
model/
  time_embedding.py   # SinusoidalTimeEmbedding
  resblock.py         # ResNet block + time injection (h = h + time_mlp(t_emb))
  attention.py        # self-attention bottleneck
  unet.py             # full UNet predictor
  dit_block.py        # AdaLayerNorm + DiTBlock + DiTStub
scheduler/
  noise_scheduler.py  # cosine + linear, forward + reverse
train.py              # loop + EMA + grad clip(max_norm=1.0) + resume checkpoints
sample.py             # reverse diffusion + GIF export (loads EMA)
evaluate.py           # FID + sample grid (loads EMA)
config.yaml           # all hyperparameters
checkpoints/          # gitignored (large .pth) — use HF Hub / Modal volume instead
assets/               # gifs, sample grids, loss curves
```

Conventions to preserve when generating code:
- Adam optimizer, lr 1e-4, gradient clip max_norm 1.0
- T = 1000, batch sized to hardware, `pin_memory` only when CUDA present
- resume checkpoint saves: model + ema + optimizer + loss history + config
- never push large `.pth` to git; keep `checkpoints/` in `.gitignore`
- new architecture files go in `model/`, new schedulers in `scheduler/`

## Hardware playbook

```
Oracle A1 (4 OCPU, 24GB, 150GB) — CPU only:
  data prep, dataset caching, light/CPU training, hosting samples, repo home
Oracle micro (1 OCPU, 1GB, 50GB):
  small services, monitoring, cron jobs — not training
Modal ($30 GPU credits):
  real video-diffusion training runs; A100/H100 on demand; per-second billing
  store checkpoints on a Modal volume, not git
Kaggle 2×T4 (free):
  image-diffusion training, quick experiments, resume from Oracle CPU checkpoints
```

When asked "where should this run", match the workload: GPU-heavy training → Modal
or Kaggle; everything else → Oracle.

## Response style for this skill

- Lead with the math/equation, then the code, then the file it belongs in.
- When a request is ambiguous, ask one targeted question (dataset? frames? schedule?).
- Keep checkpoints, EMA, and "predict ε" invariant unless told otherwise.
- For video work, always separate the spatial part (already built) from the new
  temporal part (the actual task).
- Be honest when something won't fit the hardware or timeline; suggest the right split.
