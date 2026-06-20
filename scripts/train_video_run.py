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
    # Data fix: the synthetic square is now 40–50% of the frame (was ~1.6–8% of
    # pixels). Loss reweighting (low-t, uniform, min-SNR-γ) all collapsed to a
    # constant background because the foreground was too rare to model — proven
    # not to be the lever. With a prominent foreground we go back to UNIFORM t
    # (no weighting) to isolate whether the rarity was the cause.
    # save_every=50 keeps cost down: each save runs a full T=1000 reverse sample.
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
