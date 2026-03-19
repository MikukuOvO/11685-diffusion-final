import os
import logging
from logging import getLogger as get_logger

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint


logger = get_logger(__name__)


def build_scheduler(args, device):
    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    return scheduler.to(device)


def save_images(images, save_dir, start_index=0):
    os.makedirs(save_dir, exist_ok=True)
    for offset, image in enumerate(images):
        image.save(os.path.join(save_dir, f"{start_index + offset:05d}.png"))


def generate_unconditional_batches(
    pipeline,
    total_images,
    batch_size,
    num_inference_steps,
    generator,
    device,
    save_dir=None,
):
    all_images = []

    for start in tqdm(range(0, total_images, batch_size), desc="Generating images"):
        current_batch_size = min(batch_size, total_images - start)
        gen_images = pipeline(
            batch_size=current_batch_size,
            num_inference_steps=num_inference_steps,
            generator=generator,
            device=device,
        )

        if save_dir is not None:
            save_images(gen_images, save_dir, start_index=start)

        gen_tensors = [pil_to_tensor(image).float() / 255.0 for image in gen_images]
        all_images.append(torch.stack(gen_tensors, dim=0))

    return torch.cat(all_images, dim=0)


def main():
    from train import parse_args

    # parse arguments
    args = parse_args()
    if args.ckpt is None:
        raise ValueError("Please provide `--ckpt` for inference.")
    if args.latent_ddpm:
        raise NotImplementedError("This inference script currently supports only pixel-space DDPM/DDIM.")
    if args.use_cfg:
        raise NotImplementedError("This inference script currently supports only unconditional DDPM/DDIM.")

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # seed everything
    seed_everything(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # setup model
    logger.info("Creating model")
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.unet_ch,
    )
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    scheduler = build_scheduler(args, device)

    vae = None
    if args.latent_ddpm and VAE is not None:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()

    class_embedder = None
    if args.use_cfg and ClassEmbedder is not None:
        class_embedder = ClassEmbedder(None)

    # send to device
    unet = unet.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    unet.eval()
    if class_embedder is not None:
        class_embedder.eval()

    pipeline = DDPMPipeline(unet, scheduler, vae=vae, class_embedder=class_embedder)

    logger.info("***** Running Inference *****")
    total_images = 1000
    save_dir = os.path.join(os.path.dirname(args.ckpt), "generated_images")
    all_images = generate_unconditional_batches(
        pipeline=pipeline,
        total_images=total_images,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        device=device,
        save_dir=save_dir,
    )

    logger.info(f"Generated {all_images.shape[0]} images")
    logger.info(f"Saved generated images to {save_dir}")


if __name__ == '__main__':
    main()
