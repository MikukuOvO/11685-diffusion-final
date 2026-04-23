from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "diffusion-vae-trainer"
PYTHON_VERSION = "3.10"
SRC_DIR = Path(__file__).resolve().parent
REMOTE_SRC_DIR = PurePosixPath("/root/project/src")
DATA_MOUNT_PATH = PurePosixPath("/mnt/data")
OUTPUT_MOUNT_PATH = PurePosixPath("/mnt/output")


def _env_flag(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value or default


GPU_CONFIG = _env_flag("MODAL_GPU", "A100")
CPU_COUNT = float(_env_flag("MODAL_CPU", "8"))
MEMORY_MB = int(_env_flag("MODAL_MEMORY_MB", "32768"))
TIMEOUT_SECONDS = int(_env_flag("MODAL_TIMEOUT_SECONDS", str(12 * 60 * 60)))
DATA_VOLUME_NAME = _env_flag("MODAL_DATA_VOLUME", "imagenet100-data")
OUTPUT_VOLUME_NAME = _env_flag("MODAL_OUTPUT_VOLUME", "diffusion-train-output")
WANDB_SECRET_NAME = _env_flag("MODAL_WANDB_SECRET", "wandb")

IGNORE_PATTERNS = ["__pycache__", "*.pyc", ".pytest_cache", "wandb", "output", "experiments*"]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install_from_requirements(SRC_DIR / "requirements.txt")
    .add_local_dir(SRC_DIR, remote_path=str(REMOTE_SRC_DIR), ignore=IGNORE_PATTERNS)
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)


def _resolve_remote_path(base_dir: PurePosixPath, value: str) -> str:
    raw_path = PurePosixPath(value)
    if raw_path.is_absolute():
        return str(raw_path)

    normalized = PurePosixPath(os.path.normpath(value))
    if normalized.parts and normalized.parts[0] == "..":
        raise ValueError(f"Path '{value}' escapes the expected mount root.")
    return str(base_dir / normalized)


def _stream_subprocess(command: list[str], env: dict[str, str], cwd: str) -> None:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


@app.function(
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    secrets=[wandb_secret],
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(OUTPUT_MOUNT_PATH): output_volume,
    },
)
def train_vae_remote(
    config: str = "configs/vae_modal_10k.yaml",
    run_name: str = "",
    data_subdir: str = "imagenet100_128x128/train",
    output_subdir: str = "vae_runs",
    extra_args: str = "",
) -> dict[str, str]:
    config_path = _resolve_remote_path(REMOTE_SRC_DIR, config)
    data_dir = _resolve_remote_path(DATA_MOUNT_PATH, data_subdir)
    output_dir = _resolve_remote_path(OUTPUT_MOUNT_PATH, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "vae")
    env["WANDB_DIR"] = os.path.join(output_dir, "wandb")
    env["WANDB_CACHE_DIR"] = os.path.join(output_dir, ".cache", "wandb")
    if os.environ.get("WANDB_ENTITY"):
        env["WANDB_ENTITY"] = os.environ["WANDB_ENTITY"]
    os.makedirs(env["WANDB_DIR"], exist_ok=True)
    os.makedirs(env["WANDB_CACHE_DIR"], exist_ok=True)

    command = [
        sys.executable,
        "train_vae.py",
        "--config",
        config_path,
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
    ]
    if run_name:
        command.extend(["--run_name", run_name])
    if extra_args:
        command.extend(shlex.split(extra_args))

    print("Launching VAE training command:")
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Using Modal GPU config: {GPU_CONFIG}")
    print(f"Training data dir: {data_dir}")
    print(f"Training output dir: {output_dir}")

    _stream_subprocess(command, env=env, cwd=str(REMOTE_SRC_DIR))
    output_volume.commit()

    resolved_run_name = run_name or "vae_baseline_10k"
    return {
        "config": config_path,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "run_dir": os.path.join(output_dir, resolved_run_name),
    }


@app.local_entrypoint()
def main(
    config: str = "configs/vae_modal_10k.yaml",
    run_name: str = "",
    data_subdir: str = "imagenet100_128x128/train",
    output_subdir: str = "vae_runs",
    extra_args: str = "",
) -> None:
    result = train_vae_remote.remote(
        config=config,
        run_name=run_name,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        extra_args=extra_args,
    )
    print("Modal VAE training finished.")
    for key, value in result.items():
        print(f"{key}: {value}")
