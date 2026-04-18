import argparse
import os
from pathlib import Path

import ruamel.yaml as yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import VAE
from utils import AverageMeter, seed_everything, str2bool


def resolve_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    for base_dir in (script_dir, script_dir.parent):
        resolved = (base_dir / path).resolve()
        if resolved.exists():
            return resolved
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE baseline.")

    parser.add_argument("--config", type=str, default="configs/vae.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/imagenet100_128x128/train")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments_vae")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=1)

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


def save_vae_checkpoint(model, optimizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "vae_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved at {checkpoint_path}")


def tensor_to_pil_batch(images):
    images = (images / 2 + 0.5).clamp(0.0, 1.0)
    images = images.detach().cpu()
    pil_images = []
    for image in images:
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        pil_images.append(Image.fromarray(array))
    return pil_images


def save_image_row(images, output_path):
    width, height = images[0].size
    canvas = Image.new("RGB", (width * len(images), height))
    for idx, image in enumerate(images):
        canvas.paste(image, (idx * width, 0))
    canvas.save(output_path)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_path(args.data_dir)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_vae(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or "vae_baseline"
    run_dir = output_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    fixed_images, _ = next(iter(loader))
    fixed_images = fixed_images[:args.sample_batch_size].to(device)

    print("***** Running VAE training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num epochs = {args.num_epochs}")
    print(f"  Batch size = {args.batch_size}")

    completed_steps = 0
    stop_training = False
    for epoch in range(args.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        rec_meter = AverageMeter()
        kl_meter = AverageMeter()
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for images, _ in progress_bar:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            reconstructions, posterior = model(images, sample_posterior=True)

            rec_loss = F.mse_loss(reconstructions, images)
            kl_loss = posterior.kl().mean()
            loss = rec_loss + args.beta_kl * kl_loss

            loss.backward()
            if args.grad_clip:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            rec_meter.update(rec_loss.item(), batch_size)
            kl_meter.update(kl_loss.item(), batch_size)
            progress_bar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                rec=f"{rec_meter.avg:.4f}",
                kl=f"{kl_meter.avg:.4f}",
            )

            completed_steps += 1
            if args.max_train_steps is not None and completed_steps >= args.max_train_steps:
                stop_training = True
                break

        print(
            f"Epoch {epoch + 1}: "
            f"loss={loss_meter.avg:.4f}, "
            f"rec={rec_meter.avg:.4f}, "
            f"kl={kl_meter.avg:.4f}"
        )

        with torch.no_grad():
            model.eval()
            reconstructions = model.reconstruct(fixed_images, sample_posterior=False)
            recon_row = tensor_to_pil_batch(reconstructions)
            sample_generator = torch.Generator(device=device)
            sample_generator.manual_seed(args.seed + epoch)
            sampled = model.sample(args.sample_batch_size, device=device, generator=sample_generator)
            sample_row = tensor_to_pil_batch(sampled)

        save_image_row(recon_row, sample_dir / f"reconstruction_epoch_{epoch}.png")
        save_image_row(sample_row, sample_dir / f"sample_epoch_{epoch}.png")

        if (epoch + 1) % args.save_every == 0:
            save_vae_checkpoint(model, optimizer, epoch, checkpoint_dir)

        if stop_training:
            print(f"Reached max_train_steps={args.max_train_steps}. Stopping training early.")
            break


if __name__ == "__main__":
    main()
