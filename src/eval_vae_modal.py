from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "diffusion-vae-eval"
PYTHON_VERSION = "3.10"
SRC_DIR = Path(__file__).resolve().parent
REMOTE_SRC_DIR = PurePosixPath("/root/project/src")
OUTPUT_MOUNT_PATH = PurePosixPath("/mnt/output")


def _env_flag(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value or default


GPU_CONFIG = _env_flag("MODAL_GPU", "H100")
CPU_COUNT = float(_env_flag("MODAL_CPU", "8"))
MEMORY_MB = int(_env_flag("MODAL_MEMORY_MB", "32768"))
TIMEOUT_SECONDS = int(_env_flag("MODAL_TIMEOUT_SECONDS", str(6 * 60 * 60)))
OUTPUT_VOLUME_NAME = _env_flag("MODAL_OUTPUT_VOLUME", "diffusion-train-output")

IGNORE_PATTERNS = ["__pycache__", "*.pyc", ".pytest_cache", "wandb", "output", "experiments*"]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install("pytorch-lightning")
    .pip_install_from_requirements(SRC_DIR / "requirements.txt")
    .add_local_dir(SRC_DIR, remote_path=str(REMOTE_SRC_DIR), ignore=IGNORE_PATTERNS)
)

app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


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
    volumes={
        str(OUTPUT_MOUNT_PATH): output_volume,
    },
)
def eval_vae_remote(
    config: str = "configs/vae_modal_10k_normkl_beta1_bs64.yaml",
    ckpt_subpath: str = "vae_runs/vae_normkl_beta1_10k_bs64/checkpoints/checkpoint_epoch_4.pth",
    total_images: int = 1000,
    batch_size: int = 64,
    eval_run_name: str = "vae_normkl_beta1_10k_bs64_1000",
    reference_subpath: str = "report_eval/val_stats.npz",
) -> dict[str, str]:
    config_path = _resolve_remote_path(REMOTE_SRC_DIR, config)
    ckpt_path = _resolve_remote_path(OUTPUT_MOUNT_PATH, ckpt_subpath)
    reference_path = _resolve_remote_path(OUTPUT_MOUNT_PATH, reference_subpath)
    generated_dir = _resolve_remote_path(OUTPUT_MOUNT_PATH, f"report_eval/{eval_run_name}")
    output_csv = _resolve_remote_path(OUTPUT_MOUNT_PATH, f"report_eval/{eval_run_name}.csv")
    output_npz = _resolve_remote_path(OUTPUT_MOUNT_PATH, f"report_eval/{eval_run_name}.npz")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(
            f"Reference stats not found: {reference_path}. "
            "Upload val_stats.npz to the output volume first."
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    inference_command = [
        sys.executable,
        "inference_vae.py",
        "--config",
        config_path,
        "--ckpt",
        ckpt_path,
        "--total_images",
        str(total_images),
        "--batch_size",
        str(batch_size),
        "--output_dir",
        generated_dir,
    ]
    fid_command = [
        sys.executable,
        "generate_submission.py",
        "--image_dir",
        generated_dir,
        "--output",
        output_csv,
        "--reference",
        reference_path,
        "--save_npz",
        output_npz,
        "--batch_size",
        str(batch_size),
    ]

    print("Launching VAE image generation:")
    print(" ".join(shlex.quote(part) for part in inference_command))
    _stream_subprocess(inference_command, env=env, cwd=str(REMOTE_SRC_DIR))

    print("Launching local FID evaluation:")
    print(" ".join(shlex.quote(part) for part in fid_command))
    _stream_subprocess(fid_command, env=env, cwd=str(REMOTE_SRC_DIR))

    output_volume.commit()
    return {
        "checkpoint": ckpt_path,
        "generated_dir": generated_dir,
        "reference": reference_path,
        "output_csv": output_csv,
        "output_npz": output_npz,
    }


@app.local_entrypoint()
def main(
    config: str = "configs/vae_modal_10k_normkl_beta1_bs64.yaml",
    ckpt_subpath: str = "vae_runs/vae_normkl_beta1_10k_bs64/checkpoints/checkpoint_epoch_4.pth",
    total_images: int = 1000,
    batch_size: int = 64,
    eval_run_name: str = "vae_normkl_beta1_10k_bs64_1000",
    reference_subpath: str = "report_eval/val_stats.npz",
) -> None:
    result = eval_vae_remote.remote(
        config=config,
        ckpt_subpath=ckpt_subpath,
        total_images=total_images,
        batch_size=batch_size,
        eval_run_name=eval_run_name,
        reference_subpath=reference_subpath,
    )
    print("Modal VAE evaluation finished.")
    for key, value in result.items():
        print(f"{key}: {value}")
