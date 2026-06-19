import modal

app = modal.App("rahulk-ddpm-video-run")

# Persisted volume so checkpoints + sample montages survive the container.
outputs = modal.Volume.from_name("rahulk-ddpm-video-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "pyyaml",
        "matplotlib",
        "pillow",
        "numpy",
    )
    .add_local_dir("/home/ubuntu/rahulk-ddpm", remote_path="/root/rahulk-ddpm")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/outputs": outputs},
)
def run_train():
    import os
    import shutil
    import subprocess

    os.chdir("/root/rahulk-ddpm")

    # Real prototype run (SETUP.md Phase 2): coherent moving square.
    # 150 epochs so the generative prior forms (the 5-epoch run learned to refine
    # existing structure but never nucleated a square from pure noise). Low-t loss
    # weighting accelerates that. save_every=30 keeps cost down: each save runs a
    # full T=1000 reverse sampling, so we sample only ~5 times, not every epoch.
    cmd = [
        "python",
        "train_video.py",
        "--epochs", "400",
        "--num_samples", "512",
        "--batch_size", "8",
        "--num_frames", "16",
        "--frame_size", "32",
        "--save_every", "50",
        "--ema_decay", "0.999",      # 0.9999 leaves EMA ~38% random init at this length
        "--min_snr_gamma", "5.0",    # emphasise high-t (nucleation-from-noise) steps
    ]

    result = subprocess.run(cmd, check=True, text=True, capture_output=True)

    # Persist artifacts into the volume so they can be pulled back to A1.
    for src, dst in [
        ("checkpoints/video", "/outputs/checkpoints/video"),
        ("assets/video_samples", "/outputs/assets/video_samples"),
    ]:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
    outputs.commit()

    return result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")

@app.local_entrypoint()
def main():
    out = run_train.remote()
    print(out)
