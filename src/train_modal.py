from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "diffusion-trainer"
PYTHON_VERSION = "3.10"
SRC_DIR = Path(__file__).resolve().parent
REMOTE_SRC_DIR = PurePosixPath("/root/project/src")
DATA_MOUNT_PATH = PurePosixPath("/mnt/data")
OUTPUT_MOUNT_PATH = PurePosixPath("/mnt/output")


def _env_flag(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value or default


GPU_CONFIG = _env_flag("MODAL_GPU", "H100")
CPU_COUNT = float(_env_flag("MODAL_CPU", "24"))
MEMORY_MB = int(_env_flag("MODAL_MEMORY_MB", "32768"))
TIMEOUT_SECONDS = int(_env_flag("MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60)))
DATA_VOLUME_NAME = _env_flag("MODAL_DATA_VOLUME", "imagenet100-data")
OUTPUT_VOLUME_NAME = _env_flag("MODAL_OUTPUT_VOLUME", "diffusion-train-output")
WANDB_SECRET_NAME = _env_flag("MODAL_WANDB_SECRET", "wandb")

IGNORE_PATTERNS = ["__pycache__", "*.pyc", ".pytest_cache", "wandb", "output", "experiments*"]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install("pytorch-lightning")
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


def _stream_subprocess(command: list[str], env: dict[str, str], cwd: str, on_line=None) -> None:
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
        if on_line is not None:
            on_line(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _ensure_cuda_ready(env: dict[str, str], cwd: str, attempts: int = 3, sleep_seconds: int = 10) -> None:
    preflight_command = [
        sys.executable,
        "-c",
        (
            "import subprocess, torch; "
            "subprocess.run(['nvidia-smi'], check=True); "
            "assert torch.cuda.is_available(), 'CUDA is not available yet'"
        ),
    ]
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Running CUDA preflight check (attempt {attempt}/{attempts})")
            _stream_subprocess(preflight_command, env=env, cwd=cwd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(f"CUDA preflight failed; retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def _looks_like_archive(path: str) -> bool:
    return path.endswith((".tar.gz", ".tgz", ".tar"))


def _normalize_gdrive_file_url(url: str) -> str:
    file_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if file_match:
        file_id = file_match.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return url


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
def train_remote(
    config: str = "configs/ddpm_a100_40gb.yaml",
    run_name: str = "",
    data_subdir: str = "train",
    output_subdir: str = "train_runs",
    vae_ckpt_path: str = "",
    extra_args: str = "",
    commit_every_steps: int = 100000,
    wandb_mode: str = "online",
    wandb_project: str = "ddpm",
    wandb_entity: str = "",
) -> dict[str, str]:
    config_path = _resolve_remote_path(REMOTE_SRC_DIR, config)
    data_dir = _resolve_remote_path(DATA_MOUNT_PATH, data_subdir)
    output_dir = _resolve_remote_path(OUTPUT_MOUNT_PATH, output_subdir)

    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = wandb_mode
    env["WANDB_PROJECT"] = wandb_project
    env["WANDB_DIR"] = os.path.join(output_dir, "wandb")
    env["WANDB_CACHE_DIR"] = os.path.join(output_dir, ".cache", "wandb")
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity

    os.makedirs(env["WANDB_DIR"], exist_ok=True)
    os.makedirs(env["WANDB_CACHE_DIR"], exist_ok=True)

    command = [
        sys.executable,
        "train.py",
        "--config",
        config_path,
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
    ]
    if run_name:
        command.extend(["--run_name", run_name])
    if vae_ckpt_path:
        resolved_vae_ckpt = _resolve_remote_path(OUTPUT_MOUNT_PATH, vae_ckpt_path)
        command.extend(["--vae_ckpt", resolved_vae_ckpt])
    if extra_args:
        command.extend(shlex.split(extra_args))

    print("Launching training command:")
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Using Modal GPU config: {GPU_CONFIG}")
    print(f"Training data dir: {data_dir}")
    print(f"Training output dir: {output_dir}")

    _ensure_cuda_ready(env=env, cwd=str(REMOTE_SRC_DIR))
    latest_global_step = 0
    next_commit_step = commit_every_steps if commit_every_steps > 0 else None

    def on_training_line(line: str) -> None:
        nonlocal latest_global_step, next_commit_step
        step_match = re.search(r"Global Step (\d+)", line)
        if step_match:
            latest_global_step = int(step_match.group(1))

        if (
            next_commit_step is not None
            and latest_global_step >= next_commit_step
            and "Checkpoint saved at" in line
        ):
            print(
                f"Committing output volume after checkpoint near global step "
                f"{latest_global_step} (threshold {next_commit_step})"
            )
            output_volume.commit()
            next_commit_step += commit_every_steps

    _stream_subprocess(command, env=env, cwd=str(REMOTE_SRC_DIR), on_line=on_training_line)
    output_volume.commit()

    return {
        "config": config_path,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "wandb_dir": env["WANDB_DIR"],
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=60 * 60,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
    },
)
def download_data_remote(
    url: str,
    target_subdir: str = "",
    extract: bool = True,
    filename: str = "",
    is_folder: bool = False,
) -> dict[str, str]:
    import gdown

    target_dir = _resolve_remote_path(DATA_MOUNT_PATH, target_subdir or ".")
    os.makedirs(target_dir, exist_ok=True)

    if is_folder:
        output_path = target_dir
        print(f"Downloading Google Drive folder into {output_path}")
        downloaded = gdown.download_folder(url=url, output=output_path, quiet=False)
        data_volume.commit()
        return {
            "target_dir": target_dir,
            "downloaded": str(downloaded),
        }

    resolved_filename = filename or os.path.basename(url.rstrip("/")) or "downloaded_file"
    archive_path = os.path.join(target_dir, resolved_filename)
    print(f"Downloading file to {archive_path}")
    normalized_url = _normalize_gdrive_file_url(url)
    downloaded_path = gdown.download(url=normalized_url, output=archive_path, quiet=False)
    if downloaded_path is None:
        raise RuntimeError(f"Failed to download from {url}")

    extracted_to = ""
    if extract and _looks_like_archive(downloaded_path):
        print(f"Extracting {downloaded_path} into {target_dir}")
        with tarfile.open(downloaded_path, "r:*") as archive:
            archive.extractall(target_dir)
        extracted_to = target_dir

    data_volume.commit()
    return {
        "downloaded_path": downloaded_path,
        "target_dir": target_dir,
        "extracted_to": extracted_to,
    }


@app.local_entrypoint()
def main(
    config: str = "configs/ddpm_a100_40gb.yaml",
    run_name: str = "",
    data_subdir: str = "train",
    output_subdir: str = "train_runs",
    vae_ckpt_path: str = "",
    extra_args: str = "",
    wandb_mode: str = "online",
    wandb_project: str = "ddpm",
    wandb_entity: str = "",
) -> None:
    result = train_remote.remote(
        config=config,
        run_name=run_name,
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        vae_ckpt_path=vae_ckpt_path,
        extra_args=extra_args,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    print("Modal training finished.")
    for key, value in result.items():
        print(f"{key}: {value}")


@app.local_entrypoint()
def download_data(
    url: str,
    target_subdir: str = "",
    extract: bool = True,
    filename: str = "",
    is_folder: bool = False,
) -> None:
    result = download_data_remote.remote(
        url=url,
        target_subdir=target_subdir,
        extract=extract,
        filename=filename,
        is_folder=is_folder,
    )
    print("Modal data download finished.")
    for key, value in result.items():
        print(f"{key}: {value}")
