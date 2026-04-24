from __future__ import annotations

import netrc
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = os.environ.get("MODAL_APP_NAME", "11685-diffusion")
PROJECT_VOLUME_NAME = os.environ.get("MODAL_PROJECT_VOLUME", "11685-diffusion-project")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

REMOTE_ROOT = Path("/root/project")
REMOTE_SRC_DIR = REMOTE_ROOT / "src"
VOLUME_ROOT = Path("/vol")
DEFAULT_DATA_DIR = VOLUME_ROOT / "data" / "imagenet100_128x128" / "train"
DEFAULT_VAE_CKPT = VOLUME_ROOT / "data" / "pretrained" / "official_vae.ckpt"
DEFAULT_OUTPUT_ROOT = VOLUME_ROOT / "output"
DEFAULT_DATA_ARCHIVE = VOLUME_ROOT / "data" / "imagenet100_128x128.tar"
LOCAL_STAGE_ROOT = Path("/tmp/modal_stage")
LOCAL_DATA_ROOT = LOCAL_STAGE_ROOT / "imagenet100_128x128"
LOCAL_DATA_DIR = LOCAL_DATA_ROOT / "train"
LOCAL_VAE_CKPT = LOCAL_STAGE_ROOT / "pretrained" / "official_vae.ckpt"
FASTEST_GPU_FALLBACKS = ["H200", "H100"]

app = modal.App(APP_NAME)
project_volume = modal.Volume.from_name(PROJECT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "pillow",
        "numpy",
        "pytorch-lightning",
    )
    .pip_install_from_requirements(str(SRC_DIR / "requirements.txt"))
    .add_local_dir(
        str(SRC_DIR),
        remote_path=str(REMOTE_SRC_DIR),
        ignore=[
            "pretrained/**",
            "tests/**",
            ".pytest_cache/**",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            "tea_debug.log",
        ],
    )
)

BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "WANDB_SILENT": "true",
}


def _load_wandb_api_key() -> str | None:
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return api_key.strip()

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return None

    try:
        auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        return None

    if auth and auth[2]:
        return auth[2].strip()
    return None


_WANDB_API_KEY = _load_wandb_api_key()
FUNCTION_SECRETS = []
DEFAULT_WANDB_MODE = "offline"
if _WANDB_API_KEY:
    FUNCTION_SECRETS.append(modal.Secret.from_dict({"WANDB_API_KEY": _WANDB_API_KEY}))
    DEFAULT_WANDB_MODE = "online"


def _checkpoint_epoch(path: Path) -> int:
    match = re.match(r"checkpoint_epoch_(\d+)\.pth$", path.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint filename: {path}")
    return int(match.group(1))


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = [
        path for path in checkpoint_dir.glob("checkpoint_epoch_*.pth") if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_epoch)


def _resolve_checkpoint(checkpoint_dir: Path, checkpoint_epoch: int) -> Path:
    if checkpoint_epoch < 0:
        checkpoint_path = _latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
        return checkpoint_path

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{checkpoint_epoch}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"Running: {rendered}", flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _extract_archive(archive_path: Path, target_dir: Path, strip_components: int = 0) -> None:
    command = ["tar", "xf", str(archive_path), "-C", str(target_dir)]
    if strip_components > 0:
        command.extend(["--strip-components", str(strip_components)])
    _run_command(command, cwd=target_dir, env=os.environ.copy())


def _copy_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_command(["cp", "-a", str(src), str(dst)], cwd=dst.parent, env=os.environ.copy())


def _config_path(config: str) -> Path:
    config_path = Path(config)
    if config_path.is_absolute():
        return config_path
    return REMOTE_SRC_DIR / config_path


def _config_uses_latent_ddpm(config: str) -> bool:
    config_path = _config_path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("latent_ddpm:"):
            continue
        value = stripped.split(":", 1)[1].strip().lower()
        return value in {"1", "true", "yes", "on"}
    return False


def _stage_training_assets(data_dir: str, vae_ckpt: str | None) -> tuple[str, str | None]:
    staged_data_dir = Path(data_dir)
    staged_vae_ckpt = Path(vae_ckpt) if vae_ckpt else None

    if str(staged_data_dir).startswith(str(VOLUME_ROOT)):
        if DEFAULT_DATA_ARCHIVE.exists():
            if not LOCAL_DATA_DIR.exists():
                LOCAL_STAGE_ROOT.mkdir(parents=True, exist_ok=True)
                _extract_archive(DEFAULT_DATA_ARCHIVE, LOCAL_STAGE_ROOT)
        else:
            dataset_root = staged_data_dir.parent
            if not LOCAL_DATA_ROOT.exists():
                _copy_tree(dataset_root, LOCAL_DATA_ROOT)
        staged_data_dir = LOCAL_DATA_DIR

    if staged_vae_ckpt is not None and str(staged_vae_ckpt).startswith(str(VOLUME_ROOT)):
        if not LOCAL_VAE_CKPT.exists():
            LOCAL_VAE_CKPT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_vae_ckpt, LOCAL_VAE_CKPT)
        staged_vae_ckpt = LOCAL_VAE_CKPT

    return str(staged_data_dir), str(staged_vae_ckpt) if staged_vae_ckpt is not None else None


def _train_command(
    *,
    config: str,
    run_dir: str,
    data_dir: str,
    vae_ckpt: str | None,
    batch_size: int,
    num_workers: int,
    max_train_steps: int,
    num_epochs: int,
    resume: bool,
    extra_args: str,
) -> tuple[list[str], Path]:
    run_root = DEFAULT_OUTPUT_ROOT / "train_runs" / run_dir
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "train.py",
        "--config",
        config,
        "--data_dir",
        data_dir,
        "--output_dir",
        str(run_root),
        "--exact_output_dir",
        "true",
        "--run_name",
        Path(run_dir).name,
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--max_train_steps",
        str(max_train_steps),
        "--num_epochs",
        str(num_epochs),
    ]
    if vae_ckpt:
        command.extend(["--vae_ckpt", vae_ckpt])

    resume_ckpt = _latest_checkpoint(checkpoint_dir) if resume else None
    if resume_ckpt is not None:
        print(f"Resuming from {resume_ckpt}", flush=True)
        command.extend(["--resume_ckpt", str(resume_ckpt)])
    else:
        print("Starting a fresh training run", flush=True)

    if extra_args.strip():
        command.extend(shlex.split(extra_args))

    return command, run_root


@app.function(
    image=image,
    gpu=FASTEST_GPU_FALLBACKS,
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 24,
    startup_timeout=60 * 20,
    retries=modal.Retries(max_retries=3, initial_delay=0.0, backoff_coefficient=1.0),
    single_use_containers=True,
    volumes={str(VOLUME_ROOT): project_volume},
    env=BASE_ENV,
    secrets=FUNCTION_SECRETS,
)
def run_training(
    config: str = "configs/latent_cfg_official_vae_ema_cosine_a100_40gb.yaml",
    run_dir: str = "latent_cfg_official_vae_modal",
    data_dir: str = str(DEFAULT_DATA_DIR),
    vae_ckpt: str = str(DEFAULT_VAE_CKPT),
    batch_size: int = 64,
    num_workers: int = 4,
    max_train_steps: int = 150000,
    num_epochs: int = 20,
    resume: bool = True,
    wandb_mode: str = DEFAULT_WANDB_MODE,
    extra_args: str = "",
) -> str:
    project_volume.reload()
    uses_latent_ddpm = _config_uses_latent_ddpm(config)

    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Dataset path not found in volume: {data_dir}")
    if uses_latent_ddpm and not Path(vae_ckpt).exists():
        raise FileNotFoundError(f"VAE checkpoint not found in volume: {vae_ckpt}")

    staged_data_dir, staged_vae_ckpt = _stage_training_assets(
        data_dir,
        vae_ckpt if uses_latent_ddpm else None,
    )

    env = os.environ.copy()
    env["WANDB_MODE"] = wandb_mode
    env["WANDB_DIR"] = str(DEFAULT_OUTPUT_ROOT / "wandb")

    command, run_root = _train_command(
        config=config,
        run_dir=run_dir,
        data_dir=staged_data_dir,
        vae_ckpt=staged_vae_ckpt,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_steps=max_train_steps,
        num_epochs=num_epochs,
        resume=resume,
        extra_args=extra_args,
    )
    _run_command(command, cwd=REMOTE_SRC_DIR, env=env)
    project_volume.commit()
    return str(run_root)


@app.function(
    image=modal.Image.debian_slim(python_version="3.10").apt_install("tar"),
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 4,
    volumes={str(VOLUME_ROOT): project_volume},
    env=BASE_ENV,
)
def prepare_dataset(
    archive_path: str = str(DEFAULT_DATA_ARCHIVE),
    target_dir: str = str(VOLUME_ROOT / "data"),
) -> str:
    project_volume.reload()
    archive = Path(archive_path)
    target = Path(target_dir)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found in volume: {archive}")

    dataset_root = target / "imagenet100_128x128"
    if dataset_root.exists():
        print(f"Dataset already present at {dataset_root}", flush=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive, target)
        print(f"Extracted dataset to {dataset_root}", flush=True)

    project_volume.commit()
    return str(dataset_root)


@app.function(
    image=image,
    gpu=FASTEST_GPU_FALLBACKS,
    cpu=4,
    memory=24576,
    timeout=60 * 60 * 8,
    startup_timeout=60 * 20,
    volumes={str(VOLUME_ROOT): project_volume},
    env=BASE_ENV,
    secrets=FUNCTION_SECRETS,
)
def run_submission(
    config: str = "configs/latent_cfg_official_vae_ema_cosine_a100_40gb.yaml",
    run_dir: str = "latent_cfg_official_vae_modal",
    checkpoint_epoch: int = -1,
    batch_size: int = 64,
    total_images: int = 5000,
    num_inference_steps: int = 50,
    use_ema: bool = True,
    use_ddim: bool = True,
    guidance_scale: float = 3.0,
    wandb_mode: str = DEFAULT_WANDB_MODE,
    reference_npz: str = "",
    extra_args: str = "",
) -> str:
    project_volume.reload()
    uses_latent_ddpm = _config_uses_latent_ddpm(config)

    run_root = DEFAULT_OUTPUT_ROOT / "train_runs" / run_dir
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_path = _resolve_checkpoint(checkpoint_dir, checkpoint_epoch)
    generated_dir = checkpoint_dir / "generated_images"
    if generated_dir.exists():
        shutil.rmtree(generated_dir)

    env = os.environ.copy()
    env["WANDB_MODE"] = wandb_mode

    inference_cmd = [
        sys.executable,
        "inference.py",
        "--config",
        config,
        "--ckpt",
        str(checkpoint_path),
    ]
    if uses_latent_ddpm:
        _, staged_vae_ckpt = _stage_training_assets(str(DEFAULT_DATA_DIR), str(DEFAULT_VAE_CKPT))
        if staged_vae_ckpt is None:
            raise RuntimeError("Expected a staged VAE checkpoint for latent DDPM inference.")
        inference_cmd.extend(["--vae_ckpt", staged_vae_ckpt])
    inference_cmd.extend(
        [
            "--batch_size",
            str(batch_size),
            "--total_images",
            str(total_images),
            "--num_inference_steps",
            str(num_inference_steps),
            "--use_ema",
            str(use_ema).lower(),
            "--use_ddim",
            str(use_ddim).lower(),
            "--cfg_guidance_scale",
            str(guidance_scale),
        ]
    )
    if extra_args.strip():
        inference_cmd.extend(shlex.split(extra_args))
    _run_command(inference_cmd, cwd=REMOTE_SRC_DIR, env=env)

    submissions_dir = DEFAULT_OUTPUT_ROOT / "submissions" / run_dir
    submissions_dir.mkdir(parents=True, exist_ok=True)
    sampler_tag = "ddim" if use_ddim else "ddpm"
    csv_path = submissions_dir / (
        f"{checkpoint_path.stem}_{sampler_tag}{num_inference_steps}_ema{int(use_ema)}.csv"
    )

    submission_cmd = [
        sys.executable,
        "generate_submission.py",
        "--image_dir",
        str(generated_dir),
        "--output",
        str(csv_path),
        "--batch_size",
        str(batch_size),
    ]
    if reference_npz.strip():
        submission_cmd.extend(["--reference", reference_npz])
    _run_command(submission_cmd, cwd=REMOTE_SRC_DIR, env=env)

    project_volume.commit()
    return str(csv_path)


@app.local_entrypoint()
def smoke(
    config: str = "configs/latent_cfg_official_vae_ema_cosine_a100_40gb.yaml",
    run_dir: str = "smoke/latent_cfg_official_vae",
    batch_size: int = 32,
    wandb_mode: str = DEFAULT_WANDB_MODE,
) -> None:
    result = run_training.remote(
        config=config,
        run_dir=run_dir,
        batch_size=batch_size,
        num_workers=2,
        max_train_steps=1,
        num_epochs=1,
        resume=False,
        wandb_mode=wandb_mode,
        extra_args="--log_every 1 --eval_num_images 4 --num_inference_steps 10",
    )
    print(result)


@app.local_entrypoint()
def stage_data(
    archive_path: str = str(DEFAULT_DATA_ARCHIVE),
    target_dir: str = str(VOLUME_ROOT / "data"),
) -> None:
    print(prepare_dataset.remote(archive_path=archive_path, target_dir=target_dir))


@app.local_entrypoint()
def train(
    config: str = "configs/latent_cfg_official_vae_ema_cosine_a100_40gb.yaml",
    run_dir: str = "latent_cfg_official_vae_modal_300k",
    batch_size: int = 64,
    num_workers: int = 4,
    max_train_steps: int = 300000,
    num_epochs: int = 20,
    wait: bool = True,
    wandb_mode: str = DEFAULT_WANDB_MODE,
    extra_args: str = "",
) -> None:
    call = run_training.spawn(
        config=config,
        run_dir=run_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_steps=max_train_steps,
        num_epochs=num_epochs,
        resume=True,
        wandb_mode=wandb_mode,
        extra_args=extra_args,
    )
    print(f"Started Modal training call: {call.object_id}")
    print(f"Volume run directory: {DEFAULT_OUTPUT_ROOT / 'train_runs' / run_dir}")
    if wait:
        print(call.get())


@app.local_entrypoint()
def submit(
    config: str = "configs/latent_cfg_official_vae_ema_cosine_a100_40gb.yaml",
    run_dir: str = "latent_cfg_official_vae_modal",
    checkpoint_epoch: int = -1,
    batch_size: int = 64,
    total_images: int = 5000,
    num_inference_steps: int = 50,
    use_ema: bool = True,
    use_ddim: bool = True,
    guidance_scale: float = 3.0,
    wait: bool = True,
    reference_npz: str = "",
    extra_args: str = "",
) -> None:
    call = run_submission.spawn(
        config=config,
        run_dir=run_dir,
        checkpoint_epoch=checkpoint_epoch,
        batch_size=batch_size,
        total_images=total_images,
        num_inference_steps=num_inference_steps,
        use_ema=use_ema,
        use_ddim=use_ddim,
        guidance_scale=guidance_scale,
        reference_npz=reference_npz,
        extra_args=extra_args,
    )
    print(f"Started Modal submission call: {call.object_id}")
    if wait:
        print(call.get())
