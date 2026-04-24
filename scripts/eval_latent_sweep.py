#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from ruamel import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import ClassEmbedder, UNet  # noqa: E402
from pipelines import DDPMPipeline  # noqa: E402
from schedulers import DDIMScheduler  # noqa: E402
from train import build_vae_from_args, infer_latent_shape, load_vae_weights  # noqa: E402
from utils import load_checkpoint, seed_everything  # noqa: E402


def load_config(path):
    y = yaml.YAML()
    with open(path, "r", encoding="utf-8") as f:
        return y.load(f)


def make_grid(images, image_size, cols):
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * image_size, rows * image_size))
    for i, image in enumerate(images):
        grid.paste(image, ((i % cols) * image_size, (i // cols) * image_size))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clip_sample", action="store_true")
    parser.add_argument("--use_raw_weights", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vae_ckpt is not None:
        cfg["vae_ckpt"] = args.vae_ckpt
    cfg["clip_sample"] = bool(args.clip_sample)
    cfg["use_ddim"] = True
    cfg["use_cfg"] = True
    cfg["ckpt"] = args.ckpt
    cfg["eval_num_images"] = args.num_images
    cfg.setdefault("cond_drop_rate", 0.1)
    cfg.setdefault("vae_scale_factor", 0.1912)
    run_args = SimpleNamespace(**cfg)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    vae = build_vae_from_args(run_args)
    load_vae_weights(vae, run_args.vae_ckpt)
    vae.eval().to(device)
    run_args.unet_in_ch, run_args.unet_in_size = infer_latent_shape(
        vae, run_args.image_size, device
    )

    unet = UNet(
        input_size=run_args.unet_in_size,
        input_ch=run_args.unet_in_ch,
        T=run_args.num_train_timesteps,
        ch=run_args.unet_ch,
        ch_mult=run_args.unet_ch_mult,
        attn=run_args.unet_attn,
        num_res_blocks=run_args.unet_num_res_blocks,
        dropout=run_args.unet_dropout,
        conditional=True,
        c_dim=run_args.unet_ch,
    ).to(device)

    class_embedder = ClassEmbedder(
        run_args.unet_ch,
        n_classes=run_args.num_classes,
        cond_drop_rate=run_args.cond_drop_rate,
    ).to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=run_args.num_train_timesteps,
        num_inference_steps=max(args.steps),
        beta_start=run_args.beta_start,
        beta_end=run_args.beta_end,
        beta_schedule=run_args.beta_schedule,
        variance_type=run_args.variance_type,
        prediction_type=run_args.prediction_type,
        clip_sample=bool(args.clip_sample),
        clip_sample_range=run_args.clip_sample_range,
    ).to(device)

    load_checkpoint(
        unet,
        scheduler,
        vae=vae,
        class_embedder=class_embedder,
        checkpoint_path=args.ckpt,
        use_ema=not args.use_raw_weights,
    )
    unet.eval()
    class_embedder.eval()

    pipeline = DDPMPipeline(
        unet,
        scheduler,
        vae=vae,
        class_embedder=class_embedder,
        vae_scale_factor=run_args.vae_scale_factor,
    )
    pipeline._progress_bar_config = {"disable": True}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = int(args.num_images**0.5)
    while args.num_images % cols != 0 and cols > 1:
        cols -= 1
    classes = [idx % run_args.num_classes for idx in range(args.num_images)]

    for steps in args.steps:
        for guidance in args.guidance:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            with torch.no_grad():
                images = pipeline(
                    batch_size=args.num_images,
                    num_inference_steps=steps,
                    classes=classes,
                    guidance_scale=guidance,
                    generator=generator,
                    device=device,
                )
            grid = make_grid(images, run_args.image_size, cols)
            guidance_tag = str(guidance).replace(".", "p")
            clip_tag = "cliptrue" if args.clip_sample else "clipfalse"
            filename = f"ddim{steps}_cfg{guidance_tag}_{clip_tag}_ema.png"
            grid.save(out_dir / filename)
            print(out_dir / filename)


if __name__ == "__main__":
    main()
