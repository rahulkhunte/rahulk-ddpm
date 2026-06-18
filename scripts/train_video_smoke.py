import modal

app = modal.App("rahulk-ddpm-video-smoke")

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
)
def run_smoke_train():
    import os
    import subprocess

    os.chdir("/root/rahulk-ddpm")

    cmd = [
        "python",
        "train_video.py",
        "--epochs", "1",
        "--num_samples", "16",
        "--batch_size", "4",
        "--num_frames", "8",
        "--frame_size", "16",
    ]

    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")

@app.local_entrypoint()
def main():
    out = run_smoke_train.remote()
    print(out)
