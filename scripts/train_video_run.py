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
    timeout=60 * 120,         # MovingMNIST run headroom (download + 38k steps)
    volumes={"/outputs": outputs},
)
def run_train():
    import os
    import shutil
    import subprocess

    os.chdir("/root/rahulk-ddpm")

    # MovingMNIST run (UNCONDITIONAL): can a 7M DiT generate recognizable moving
    # digits from pure noise now that the x0-clamp sampler is fixed (commit
    # 45b0204)? cond_features=0 forces unconditional (config default is 6).
    #
    # Compute is sized to the EXACT profile that completed for the synthetic run
    # (2048 clips x batch 32 x 600 epochs = 38,400 steps = 1.23M sample-forwards).
    # Earlier MovingMNIST attempts at 4096 clips / batch 64 = 2.46M forwards were
    # ~2x that compute and overran the timeout. depth 6 (config). data_root on
    # the persisted volume caches the ~0.8GB download. Foreground (digits) is ~5%
    # of pixels — if it collapses to a constant field, that's the signal to add
    # trajectory conditioning next.
    cmd = [
        "python",
        "train_video.py",
        "--dataset", "movingmnist",
        "--data_root", "/outputs/data",
        "--cond_features", "0",      # unconditional
        "--epochs", "600",
        "--num_samples", "2048",
        "--batch_size", "32",
        "--num_frames", "16",
        "--frame_size", "32",
        "--save_every", "100",
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
