# Diffusion Final Project Quick Pickup

This document is for quickly taking over the current codebase and running the final-project experiments. The repo currently supports:

- Pixel-space DDPM training.
- DDIM sampling from a DDPM checkpoint.
- VAE training and VAE image generation.
- Latent DDPM training with a frozen VAE.
- Classifier-free guidance (CFG) through class conditioning and class dropout.
- Kaggle submission CSV generation from generated images.
- W&B logging plus local JSONL metrics.

Current repo:

```bash
/media/volume/mmci/bozhu/11685/Project
branch: final-fenglin
remote: git@github.com:MikukuOvO/11685-diffusion-final.git
```

## Code Structure

```text
Project/
├── instruction.md                 # this quick pickup guide
├── Diffusion_writeup_final.pdf     # final report / assignment handout
├── output/                         # training logs, checkpoints, generated samples, eval outputs
└── src/
    ├── train.py                    # DDPM / latent DDPM / CFG training entrypoint
    ├── inference.py                # image generation from DDPM or latent CFG checkpoints
    ├── train_vae.py                # VAE baseline training
    ├── inference_vae.py            # VAE sampling from a trained VAE checkpoint
    ├── generate_submission.py      # Inception features, local FID, Kaggle CSV
    ├── fid_utils.py                # shared FID/Inception utilities
    ├── configs/
    │   ├── ddpm.yaml               # starter/default DDPM config
    │   ├── ddpm_a100_40gb.yaml     # current pixel DDPM long-run config
    │   ├── latent_cfg_a100_40gb.yaml # current latent DDPM + CFG long-run config
    │   └── vae.yaml                # VAE baseline config
    ├── models/
    │   ├── unet.py                 # U-Net denoiser
    │   ├── unet_modules.py         # U-Net blocks and attention
    │   ├── class_embedder.py       # class embedding + unconditional token for CFG
    │   ├── vae.py                  # VAE wrapper: encode/decode/reconstruct/sample
    │   ├── vae_modules.py          # VAE encoder/decoder modules
    │   └── vae_distributions.py    # diagonal Gaussian posterior
    ├── schedulers/
    │   ├── scheduling_ddpm.py      # DDPM forward/reverse scheduler
    │   └── scheduling_ddim.py      # DDIM scheduler
    ├── pipelines/
    │   └── ddpm.py                 # sampling pipeline; supports VAE decode and CFG
    ├── utils/
    │   ├── checkpoint.py           # save/load UNet, scheduler, VAE, class embedder
    │   ├── dist.py                 # distributed-device helpers
    │   ├── metric.py               # AverageMeter
    │   └── misc.py                 # seeds, bool parsing, randn helper
    └── tests/
        ├── test_inference.py
        ├── test_pipeline.py
        ├── test_schedulers.py
        ├── test_train_utils.py
        └── test_vae.py
```

Generated outputs are not source code. The important generated locations are:

```text
output/train_runs/<run_name>/checkpoints/       # training checkpoints
output/train_runs/<run_name>/metrics.jsonl      # structured training metrics
output/train_runs/<run_name>/samples/           # sample grids saved at epoch end
output/logs/                                    # tmux shell logs
output/report_eval/                             # FID summaries, grids, CSV/NPZ eval artifacts
```

## Environment Setup

Miniconda has already been installed locally:

```bash
/media/volume/mmci/bozhu/11685/tools/miniconda3
```

The project environment is:

```bash
intro2dl
```

Activate it:

```bash
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl
```

If the environment needs to be recreated from scratch:

```bash
cd /media/volume/mmci/bozhu/11685/Project

# Install Miniconda first if missing, then:
/media/volume/mmci/bozhu/11685/tools/miniconda3/bin/conda create -y -n intro2dl python=3.10
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl

conda install -y -c pytorch -c nvidia \
  pytorch=2.5.1 torchvision=0.20.1 torchaudio pytorch-cuda=12.1

# Avoid MKL 2025 iJIT symbol issues seen on this machine.
conda install -y 'mkl<2025' 'intel-openmp<2025'

pip install -r src/requirements.txt pytest
```

Verify the environment:

```bash
cd /media/volume/mmci/bozhu/11685/Project
PYTHONPATH=src python -m pytest -q src/tests
```

Expected result at the time of writing:

```text
17 passed
```

Check CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Expected GPU on the current machine:

```text
NVIDIA A100-SXM4-40GB
```

## Data Layout

The training configs expect ImageFolder-style data:

```text
imagenet100_128x128/train/<class_name>/*.png
```

Current long-run configs use:

```bash
../imagenet100_128x128/train
```

relative to `src/`, so from repo root this is:

```bash
/media/volume/mmci/bozhu/11685/imagenet100_128x128/train
```

## Experiment Design

The final report should cover both pixel-space and latent-space diffusion:

1. **Pixel DDPM baseline**
   - Train U-Net directly on normalized 128x128 RGB images.
   - This remains the pixel-space baseline and a useful Kaggle backup because it does not depend on VAE quality.

2. **DDIM sampling**
   - Use the same pixel DDPM checkpoint.
   - Compare DDPM/DDIM at different sampling steps, e.g. DDPM 200, DDIM 100, DDIM 50.
   - Report the quality/speed trade-off.

3. **VAE baseline**
   - Train/evaluate the VAE separately.
   - The current VAE is functional but previous visual/FID quality was weak, so do not assume it is the best submission model.

4. **Latent DDPM**
   - Freeze the VAE encoder/decoder.
   - Encode images to latents, scale by the config's `vae_scale_factor`, and train diffusion in latent space.
   - The final official-VAE configs use `vae_scale_factor=0.1912`; keep training and inference on the same value.
   - Decode generated latents back to image space during sampling.

5. **Latent DDPM + CFG**
   - Add class conditioning with `ClassEmbedder`.
   - During training, class labels are randomly replaced with an unconditional token with `cond_drop_rate=0.1`.
   - During sampling, CFG uses:

```text
eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

6. **Evaluation**
   - Use generated grids for qualitative progress.
   - Use denoising MSE for quick training sanity checks.
   - Use local FID against validation statistics for checkpoint/model selection.
   - Generate exactly 5000 images for Kaggle final submission, 50 per class for CFG runs.

Recommended run order:

```text
1. Pixel DDPM long run
2. Evaluate first/medium/late pixel checkpoints with 1000-image local FID
3. Use best pixel checkpoint for DDPM vs DDIM comparison
4. Run latent DDPM + CFG after confirming VAE quality is acceptable
5. Compare best pixel and best latent/CFG checkpoints
6. Generate final 5000-image Kaggle CSV from the best model
```

## Current Long Training Run

At the time this guide was updated, the active training run was:

```text
tmux session: pixel_ddpm_online
W&B: https://wandb.ai/fenglin02/ddpm/runs/cnqmsiyp
run dir: output/train_runs/exp-9-ddpm_a100_40gb_online_20260418_165147
shell log: output/logs/pixel_ddpm_online_20260418_165147.log
```

Check whether it is still running:

```bash
tmux has-session -t pixel_ddpm_online && echo running
```

Follow the shell log without attaching to tmux:

```bash
tail -f /media/volume/mmci/bozhu/11685/Project/output/logs/pixel_ddpm_online_20260418_165147.log
```

Follow structured metrics:

```bash
tail -f /media/volume/mmci/bozhu/11685/Project/output/train_runs/exp-9-ddpm_a100_40gb_online_20260418_165147/metrics.jsonl
```

Monitor GPU:

```bash
watch -n 5 nvidia-smi
```

The A100 run was around:

```text
batch size: 16
speed: about 0.22 sec/step
steps per epoch: 8125
one epoch: about 30 minutes
150000 steps: about 9.3-9.7 hours including checkpoint/sample overhead
```

## W&B

W&B is installed in the conda env. A wrapper also exists at:

```bash
/home/exouser/.local/bin/wandb
```

Verify login:

```bash
wandb login --verify
```

For live online logging, use:

```bash
WANDB_MODE=online python train.py --config configs/ddpm_a100_40gb.yaml
```

For offline logging:

```bash
WANDB_MODE=offline python train.py --config configs/ddpm_a100_40gb.yaml
```

Sync an offline run later:

```bash
wandb sync /media/volume/mmci/bozhu/11685/Project/src/wandb/offline-run-YYYYMMDD_HHMMSS-RUNID
```

## Run Pixel DDPM Training

Use this as the main long-run baseline:

```bash
cd /media/volume/mmci/bozhu/11685/Project/src
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl
WANDB_MODE=online python train.py --config configs/ddpm_a100_40gb.yaml
```

Detached tmux version:

```bash
tmux new-session -d -s pixel_ddpm_online \
  "cd /media/volume/mmci/bozhu/11685/Project/src && \
   source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl && \
   mkdir -p ../output/logs && \
   export WANDB_MODE=online && \
   export PYTHONUNBUFFERED=1 && \
   python train.py --config configs/ddpm_a100_40gb.yaml --run_name ddpm_a100_40gb_online_$(date -u +%Y%m%d_%H%M%S) \
     2>&1 | tee ../output/logs/pixel_ddpm_online_$(date -u +%Y%m%d_%H%M%S).log"
```

Important config values:

```text
config: src/configs/ddpm_a100_40gb.yaml
batch_size: 16
max_train_steps: 150000
learning_rate: 2e-4
num_train_timesteps: 1000
num_inference_steps: 100
use_ddim: true for validation sampling
```

Training writes:

```text
output/train_runs/<run>/config.yaml
output/train_runs/<run>/metrics.jsonl
output/train_runs/<run>/samples/epoch_XXXX.png
output/train_runs/<run>/checkpoints/checkpoint_epoch_X.pth
```

## Run VAE Training

Use only if improving latent DDPM quality:

```bash
cd /media/volume/mmci/bozhu/11685/Project/src
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl
python train_vae.py --config configs/vae.yaml --output_dir ../output/vae_runs --run_name vae_baseline
```

Generate images from a VAE checkpoint:

```bash
python inference_vae.py \
  --config configs/vae.yaml \
  --ckpt ../output/vae_runs/vae_baseline/checkpoints/checkpoint_epoch_0.pth \
  --total_images 1000 \
  --output_dir ../output/report_eval/vae_1000
```

Known caveat: earlier VAE results looked weak and had poor local FID. Treat VAE quality as a risk for latent DDPM.

## Run Latent DDPM + CFG Training

This trains diffusion in VAE latent space with class conditioning:

```bash
cd /media/volume/mmci/bozhu/11685/Project/src
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl
WANDB_MODE=online python train.py --config configs/latent_cfg_a100_40gb.yaml
```

Detached tmux version:

```bash
tmux new-session -d -s latent_cfg_train \
  "cd /media/volume/mmci/bozhu/11685/Project/src && \
   source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl && \
   mkdir -p ../output/logs && \
   export WANDB_MODE=online && \
   export PYTHONUNBUFFERED=1 && \
   python train.py --config configs/latent_cfg_a100_40gb.yaml --run_name latent_cfg_a100_40gb_online \
     2>&1 | tee ../output/logs/latent_cfg_train.log"
```

Important config values:

```text
config: src/configs/latent_cfg_a100_40gb.yaml
batch_size: 64
max_train_steps: 150000
vae_ckpt: ../output/vae_runs/vae_baseline_10k_bs16/checkpoints/checkpoint_epoch_1.pth
latent_ddpm: true
use_cfg: true
cond_drop_rate: 0.1
cfg_guidance_scale: 2.0
```

Do not run pixel DDPM and latent CFG long runs on the same A100 unless you intentionally monitor memory and throughput.

## Inference From a Checkpoint

Pixel DDPM / DDIM generation:

```bash
cd /media/volume/mmci/bozhu/11685/Project/src
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl

python inference.py \
  --config configs/ddpm_a100_40gb.yaml \
  --ckpt ../output/train_runs/<run>/checkpoints/checkpoint_epoch_0.pth \
  --total_images 1000 \
  --batch_size 16 \
  --use_ddim true \
  --num_inference_steps 100
```

Generated images are saved under:

```text
output/train_runs/<run>/checkpoints/generated_images/
```

Latent CFG generation:

```bash
python inference.py \
  --config configs/latent_cfg_a100_40gb.yaml \
  --ckpt ../output/train_runs/<latent_run>/checkpoints/checkpoint_epoch_0.pth \
  --total_images 1000 \
  --batch_size 64 \
  --use_ddim true \
  --num_inference_steps 100 \
  --cfg_guidance_scale 2.0
```

For CFG, `inference.py` assigns classes in Kaggle order:

```text
5000 images total = 100 classes * 50 images per class
class = image_index // 50
```

## Local FID, IS and Kaggle CSV

Reference validation stats are currently stored at:

```bash
output/report_eval/val_stats.npz
```

After generating images, compute local FID and a CSV:

```bash
cd /media/volume/mmci/bozhu/11685/Project/src
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl

python generate_submission.py \
  --image_dir ../output/train_runs/<run>/checkpoints/generated_images \
  --output ../output/report_eval/<run>_1000.csv \
  --reference ../output/report_eval/val_stats.npz \
  --save_npz ../output/report_eval/<run>_1000.npz \
  --batch_size 64
```

Compute Inception Score from the same generated images:

```bash
python compute_is.py \
  --image_dir ../output/train_runs/<run>/checkpoints/generated_images \
  --output ../output/report_eval/<run>_is.json \
  --batch_size 64
```

Use FID as the primary model-selection metric for Kaggle. Report IS as a secondary metric for image sharpness/classifiability.

For Kaggle final submission, generate exactly 5000 images:

```bash
python inference.py \
  --config configs/ddpm_a100_40gb.yaml \
  --ckpt ../output/train_runs/<best_run>/checkpoints/<best_checkpoint>.pth \
  --total_images 5000 \
  --batch_size 16 \
  --use_ddim true \
  --num_inference_steps 100

python generate_submission.py \
  --image_dir ../output/train_runs/<best_run>/checkpoints/generated_images \
  --output ../output/report_eval/kaggle_submission.csv \
  --batch_size 64
```

Upload `kaggle_submission.csv` to the Kaggle/InClass competition.

## DDPM vs DDIM Comparison

Use the same checkpoint and vary only sampler/steps.

DDPM baseline:

```bash
python inference.py \
  --config configs/ddpm_a100_40gb.yaml \
  --ckpt ../output/train_runs/<run>/checkpoints/<ckpt>.pth \
  --total_images 1000 \
  --use_ddim false \
  --num_inference_steps 200
```

DDIM 100:

```bash
python inference.py \
  --config configs/ddpm_a100_40gb.yaml \
  --ckpt ../output/train_runs/<run>/checkpoints/<ckpt>.pth \
  --total_images 1000 \
  --use_ddim true \
  --num_inference_steps 100
```

DDIM 50:

```bash
python inference.py \
  --config configs/ddpm_a100_40gb.yaml \
  --ckpt ../output/train_runs/<run>/checkpoints/<ckpt>.pth \
  --total_images 1000 \
  --use_ddim true \
  --num_inference_steps 50
```

Record for the report:

```text
sampler
inference steps
wall-clock generation time
local FID@1000
representative image grid
notes on visual quality/artifacts
```

## Metrics and Logs

`train.py` writes W&B metrics and local JSONL. The JSONL records include:

```text
loss_step
loss_avg
lr
grad_norm
batch_mean
batch_std
timestep_mean
timestep_min
timestep_max
seconds_per_step
global_step
epoch
```

Inspect latest metrics:

```bash
tail -n 20 output/train_runs/<run>/metrics.jsonl
```

Quick estimate of remaining time:

```bash
/media/volume/mmci/bozhu/11685/tools/miniconda3/envs/intro2dl/bin/python - <<'PY'
import json, statistics
from pathlib import Path

run = Path("output/train_runs/<run>/metrics.jsonl")
rows = [json.loads(x) for x in run.read_text().splitlines() if x.strip()]
rows = [x for x in rows if x.get("type") == "train"]
last = rows[-1]
recent = rows[-20:]
sps = statistics.mean(x["seconds_per_step"] for x in recent if x["global_step"] > 1)
total_steps = 150000
steps_per_epoch = 8125

print("current step:", last["global_step"])
print("sec/step:", sps)
print("one epoch min:", steps_per_epoch * sps / 60)
print("remaining hours:", (total_steps - last["global_step"]) * sps / 3600)
PY
```

## Smoke Tests

Pixel DDPM one-step smoke:

```bash
cd /media/volume/mmci/bozhu/11685/Project/src
source /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/activate intro2dl
WANDB_MODE=offline python train.py \
  --config configs/ddpm_a100_40gb.yaml \
  --run_name ddpm_smoke \
  --max_train_steps 1 \
  --batch_size 2 \
  --num_inference_steps 2 \
  --log_every 1
```

Latent CFG one-step smoke:

```bash
WANDB_MODE=offline python train.py \
  --config configs/latent_cfg_a100_40gb.yaml \
  --run_name latent_cfg_smoke \
  --max_train_steps 1 \
  --batch_size 2 \
  --num_inference_steps 2 \
  --log_every 1
```

Full tests:

```bash
cd /media/volume/mmci/bozhu/11685/Project
PYTHONPATH=src /media/volume/mmci/bozhu/11685/tools/miniconda3/bin/conda run -n intro2dl python -m pytest -q src/tests
```

## Git Notes

The repo is pushed to GitHub from `main`. On this machine, use the repo-local SSH key:

```bash
git config core.sshCommand 'ssh -i ~/.ssh/github_bozhu -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
```

Normal push:

```bash
git status -sb
git add <source files only>
git commit -m "your message"
git push origin main
```

Do not commit generated training outputs unless explicitly needed:

```text
output/
src/wandb/
wandb/
__pycache__/
*.log
```

`src/tea_debug.log` is a local debug log and should stay uncommitted unless there is a specific reason.

## Known Caveats

- Pixel DDPM remains the clean baseline and Kaggle backup.
- Latent DDPM + CFG is the current strongest path, with final quality still dependent on VAE decode quality.
- Previous VAE local FID was poor (`fid_1000_vs_validation = 412.6947`), so do not assume latent results will beat pixel DDPM.
- The current training loop does not compute FID or IS automatically. Run evaluation separately after checkpoints are saved.
- `num_epochs` can be larger than the actual run length because `max_train_steps` is the real stopping condition.
