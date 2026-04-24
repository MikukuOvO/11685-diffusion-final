# Modal Setup

This repo now includes a Modal app at [modal_app.py](modal_app.py) for remote training and submission generation.

## 1. Authenticate this machine

```bash
python3 -m modal token new
```

If you already have a token ID and secret, you can also use:

```bash
python3 -m modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

## 2. Create the project volume

The default volume name is `11685-diffusion-project`.

```bash
python3 -m modal volume create 11685-diffusion-project
```

If you want a different volume name, set `MODAL_PROJECT_VOLUME` before every `modal run` command.

## 3. Upload dataset and official VAE

```bash
python3 -m modal volume put 11685-diffusion-project ./imagenet100_128x128 /data/imagenet100_128x128
python3 -m modal volume put 11685-diffusion-project ./src/pretrained/official_vae.ckpt /data/pretrained/official_vae.ckpt
```

Optional check:

```bash
python3 -m modal volume ls 11685-diffusion-project /data
```

## 4. Run a smoke test

```bash
python3 -m modal run modal_app.py::smoke
```

This launches a 1-step latent DDPM + CFG run on the fastest available GPU pool and writes outputs under:

```text
/vol/output/train_runs/smoke/latent_cfg_official_vae
```

## 5. Start full training

```bash
python3 -m modal run modal_app.py::train
```

Background mode:

```bash
python3 -m modal run -d modal_app.py::train
```

Useful overrides:

```bash
python3 -m modal run modal_app.py::train \
  --run-dir latent_cfg_official_vae_modal \
  --batch-size 64 \
  --num-workers 4 \
  --max-train-steps 150000 \
  --num-epochs 20
```

The training function:

- requests GPUs in this order: `H200`, `H100`
- uses online W&B automatically when a local `WANDB_API_KEY` or `~/.netrc` entry for `api.wandb.ai` is available
- writes to a fixed run directory
- auto-resumes from the latest checkpoint in that run directory
- uses a 24 hour function timeout with retries

## 6. Watch running jobs

```bash
python3 -m modal app list
python3 -m modal app logs <app-id>
```

## 7. Generate a submission remotely

After training finishes:

```bash
python3 -m modal run modal_app.py::submit \
  --run-dir latent_cfg_official_vae_modal \
  --checkpoint-epoch -1 \
  --num-inference-steps 50 \
  --batch-size 64 \
  --total-images 5000
```

`--checkpoint-epoch -1` means "use the latest checkpoint".

The generated CSV is written under:

```text
/vol/output/submissions/<run-dir>/
```

## 8. Download results back to local disk

```bash
python3 -m modal volume get 11685-diffusion-project /output ./modal_output
```

## Notes

- `modal_app.py` intentionally does not upload `src/pretrained/official_vae.ckpt` with the code bundle. Large assets stay in the Volume.
- `train.py` now supports `--exact_output_dir true`, which is what makes Modal retries resume cleanly in the same run directory.
- Default training config is [latent_cfg_official_vae_ema_cosine_a100_40gb.yaml](src/configs/latent_cfg_official_vae_ema_cosine_a100_40gb.yaml).
