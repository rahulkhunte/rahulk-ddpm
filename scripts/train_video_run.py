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
    timeout=60 * 90,          # longer/bigger run needs more than the 30-min default
    volumes={"/outputs": outputs},
)
def run_train():
    import os
    import shutil
    import subprocess

    os.chdir("/root/rahulk-ddpm")

    # Quality run (SETUP.md Phase 2): SHARPEN the moving square.
    # The collapse blocker is fixed (x0-clamp sampler, commit 45b0204) — the
    # earlier model already nucleates a MOVING foreground from pure noise, but
    # the shapes were blobby clusters, not crisp squares. That's capacity/length,
    # not the blocker. So: deeper (depth 4->6 in config), more data (512->2048),
    # bigger batch (8->32) and longer (400->600 epochs). Conditioning (cond_
    # features=6) + uniform t + ema 0.999 retained. save_every=100 keeps cost
    # down: each save runs a full T=1000 reverse sample (now the fixed sampler).
    cmd = [
        "python",
        "train_video.py",
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
