import argparse
import os

import ruamel.yaml as yaml
import torch
from PIL import Image

from models import VAE
from utils import seed_everything, str2bool


def resolve_path(path):
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base_dir in (script_dir, os.path.dirname(script_dir)):
        resolved = os.path.abspath(os.path.join(base_dir, path))
        if os.path.exists(resolved):
            return resolved
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with a trained VAE baseline.")

    parser.add_argument("--config", type=str, default="configs/vae.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--total_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=128)

    parser.add_argument("--vae_double_z", type=str2bool, default=True)
    parser.add_argument("--vae_z_channels", type=int, default=3)
    parser.add_argument("--vae_embed_dim", type=int, default=3)
    parser.add_argument("--vae_in_channels", type=int, default=3)
    parser.add_argument("--vae_out_ch", type=int, default=3)
    parser.add_argument("--vae_ch", type=int, default=128)
    parser.add_argument("--vae_ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--vae_num_res_blocks", type=int, default=2)

    args = parser.parse_args()
    if args.config is not None:
        config_path = resolve_path(args.config)
        with open(config_path, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    return parser.parse_args()


def build_vae(args):
    return VAE(
        double_z=args.vae_double_z,
        z_channels=args.vae_z_channels,
        embed_dim=args.vae_embed_dim,
        resolution=args.image_size,
        in_channels=args.vae_in_channels,
        out_ch=args.vae_out_ch,
        ch=args.vae_ch,
        ch_mult=args.vae_ch_mult,
        num_res_blocks=args.vae_num_res_blocks,
    )


def load_vae_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["vae_state_dict"])
    return checkpoint


def save_images(images, save_dir, start_index=0):
    os.makedirs(save_dir, exist_ok=True)
    for offset, image in enumerate(images):
        array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        Image.fromarray(array).save(os.path.join(save_dir, f"{start_index + offset:05d}.png"))


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vae(args).to(device)
    load_vae_checkpoint(model, resolve_path(args.ckpt))
    model.eval()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.ckpt), "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    generated = 0
    while generated < args.total_images:
        current_batch_size = min(args.batch_size, args.total_images - generated)
        with torch.no_grad():
            images = model.sample(current_batch_size, device=device, generator=generator)
            images = (images / 2 + 0.5).clamp(0.0, 1.0)
        save_images(images, output_dir, start_index=generated)
        generated += current_batch_size

    print(f"Generated {generated} images to {output_dir}")


if __name__ == "__main__":
    main()
