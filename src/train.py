import os 
import sys 
import argparse
import json
import math
import time
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint


logger = get_logger(__name__)


def resolve_training_schedule(num_epochs, num_update_steps_per_epoch, max_train_steps=None):
    if num_update_steps_per_epoch <= 0:
        raise ValueError("`num_update_steps_per_epoch` must be positive.")

    if max_train_steps is None:
        return num_epochs, num_epochs * num_update_steps_per_epoch
    if max_train_steps <= 0:
        raise ValueError("`max_train_steps` must be positive when provided.")

    resolved_num_epochs = max(num_epochs, math.ceil(max_train_steps / num_update_steps_per_epoch))
    return resolved_num_epochs, max_train_steps


def build_lr_scheduler(optimizer, args):
    scheduler_name = str(args.lr_scheduler).lower()
    if scheduler_name in ("none", "constant"):
        return None
    if scheduler_name != "cosine":
        raise NotImplementedError(f"LR scheduler {args.lr_scheduler} not implemented.")

    warmup_steps = max(0, int(args.lr_warmup_steps))
    max_train_steps = max(1, int(args.max_train_steps))
    min_lr_ratio = float(args.min_lr) / float(args.learning_rate)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-8, float(current_step + 1) / float(warmup_steps))
        progress = (current_step - warmup_steps) / max(1, max_train_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    @torch.no_grad()
    def update(self, model):
        state_dict = model.state_dict()
        for name, shadow_param in self.shadow.items():
            shadow_param.mul_(self.decay).add_(state_dict[name].detach(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        state_dict = model.state_dict()
        for name, shadow_param in self.shadow.items():
            state_dict[name].copy_(shadow_param)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class use_ema_weights:
    def __init__(self, model, ema):
        self.model = model
        self.ema = ema
        self.backup = None

    def __enter__(self):
        if self.ema is None:
            return self.model
        self.backup = {
            name: param.detach().clone()
            for name, param in self.model.state_dict().items()
            if torch.is_floating_point(param)
        }
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        if self.backup is None:
            return False
        state_dict = self.model.state_dict()
        for name, backup_param in self.backup.items():
            state_dict[name].copy_(backup_param)
        return False


def sample_grid(
    pipeline,
    image_size,
    args,
    generator,
    device,
):
    gen_images = pipeline(
        batch_size=4,
        num_inference_steps=args.num_inference_steps,
        classes=[0, 1, 2, 3] if args.use_cfg else None,
        guidance_scale=args.cfg_guidance_scale if args.use_cfg else None,
        generator=generator,
        device=device,
    )

    grid_image = Image.new('RGB', (4 * image_size, image_size))
    for i, image in enumerate(gen_images):
        x = (i % 4) * image_size
        grid_image.paste(image, (x, 0))
    return grid_image


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="config file used to specify parameters")

    # data 
    parser.add_argument("--data_dir", type=str, default='./data/imagenet100_128x128/train', help="data folder") 
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes in dataset")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None, help="optional override for total training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="none", help="learning rate scheduler: none or cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="linear warmup steps for lr scheduler")
    parser.add_argument("--min_lr", type=float, default=0.0, help="minimum lr for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--log_every", type=int, default=100, help="training metric logging frequency in steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')
    parser.add_argument("--use_ema", type=str2bool, default=False, help="maintain EMA copy of UNet weights")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay for UNet weights")
    parser.add_argument("--ema_start_step", type=int, default=0, help="first optimizer step to update EMA")
    parser.add_argument("--ema_update_every", type=int, default=1, help="EMA update interval in optimizer steps")
    
    # ddpm
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="ddpm variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="clip sample range")
    
    # unet
    parser.add_argument("--unet_in_size", type=int, default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")
    
    # vae
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="checkpoint path for the VAE used by latent DDPM")
    parser.add_argument("--vae_scale_factor", type=float, default=0.1845, help="latent scale factor")
    parser.add_argument("--vae_double_z", type=str2bool, default=True)
    parser.add_argument("--vae_z_channels", type=int, default=3)
    parser.add_argument("--vae_embed_dim", type=int, default=3)
    parser.add_argument("--vae_in_channels", type=int, default=3)
    parser.add_argument("--vae_out_ch", type=int, default=3)
    parser.add_argument("--vae_ch", type=int, default=128)
    parser.add_argument("--vae_ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--vae_num_res_blocks", type=int, default=2)
    
    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference")
    parser.add_argument("--cond_drop_rate", type=float, default=0.1, help="class dropout probability for CFG training")
    
    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference")
    
    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path for inference")
    parser.add_argument("--total_images", type=int, default=5000, help="number of images to generate during inference")
    
    # first parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args


def build_vae_from_args(args):
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


def load_vae_weights(vae, checkpoint_path):
    if checkpoint_path is None:
        default_path = "pretrained/model.ckpt"
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            raise ValueError("`--vae_ckpt` is required for latent DDPM when pretrained/model.ckpt is unavailable.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "vae_state_dict" in checkpoint:
        vae.load_state_dict(checkpoint["vae_state_dict"])
    elif "state_dict" in checkpoint:
        vae.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        vae.load_state_dict(checkpoint)


def infer_latent_shape(vae, image_size, device):
    was_training = vae.training
    vae.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        latent = vae.encode(dummy)
    if was_training:
        vae.train()
    return latent.shape[1], latent.shape[2]


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    
    
def main():
    
    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup distributed initialize and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    
    # setup dataset
    logger.info("Creating dataset")
    # TODO: use transform to normalize your images to [-1, 1]
    # TODO: you can also use horizontal flip
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    # TOOD: use image folder for your train dataset
    train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    # TODO: setup dataloader
    sampler = None 
    if args.distributed:
        # TODO: distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
    # TODO: shuffle
    shuffle = False if sampler else True
    # TODO dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    
    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size 
    args.total_batch_size = total_batch_size
    
    # setup experiment folder
    os.makedirs(args.output_dir, exist_ok=True)
    num_existing_runs = len(os.listdir(args.output_dir))
    if args.run_name is None:
        args.run_name = f'exp-{num_existing_runs}'
    else:
        args.run_name = f'exp-{num_existing_runs}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    vae = None
    if args.latent_ddpm:
        vae = build_vae_from_args(args)
        load_vae_weights(vae, args.vae_ckpt)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        vae = vae.to(device)
        args.unet_in_ch, args.unet_in_size = infer_latent_shape(vae, args.image_size, device)

    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    diffusion_scheduler = DDPMScheduler(
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
    
    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            args.unet_ch,
            n_classes=args.num_classes,
            cond_drop_rate=args.cond_drop_rate,
        )
        
    # send to device
    unet = unet.to(device)
    diffusion_scheduler = diffusion_scheduler.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
    
    # TODO: setup optimizer
    trainable_params = list(unet.parameters())
    if class_embedder is not None:
        trainable_params += list(class_embedder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    if num_update_steps_per_epoch == 0:
        raise ValueError("Training dataloader is empty. Check `data_dir`, `batch_size`, and dataset structure.")
    args.num_epochs, args.max_train_steps = resolve_training_schedule(
        args.num_epochs,
        num_update_steps_per_epoch,
        args.max_train_steps,
    )
    lr_scheduler = build_lr_scheduler(optimizer, args)
    ema_unet = EMAModel(unet, decay=args.ema_decay) if args.use_ema else None
    
    #  setup distributed training
    class_embedder_wo_ddp = class_embedder
    if args.distributed:
        ddp_kwargs = dict(find_unused_parameters=False)
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            **ddp_kwargs,
        )
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder,
                **ddp_kwargs,
            )
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
    vae_wo_ddp = vae
    scheduler_wo_ddp = diffusion_scheduler
    if args.use_ddim:
        eval_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        ).to(device)
    else:
        eval_scheduler = diffusion_scheduler
    
    # TODO: setup evaluation pipeline
    # NOTE: this pipeline is not differentiable and only for evaluatin
    pipeline = DDPMPipeline(
        unet_wo_ddp,
        eval_scheduler,
        vae=vae_wo_ddp,
        class_embedder=class_embedder_wo_ddp,
        vae_scale_factor=args.vae_scale_factor,
    )
    
    
    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        metrics_path = os.path.join(output_dir, "metrics.jsonl")
    else:
        metrics_path = None
    
    # start tracker
    if is_primary(args):
        wandb_logger = wandb.init(
            project='ddpm', 
            name=args.run_name, 
            config=vars(args))
    
    # Start training    
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))
    completed_steps = 0
    stop_training = False
    last_log_time = time.perf_counter()

    # training
    for epoch in range(args.num_epochs):
        
        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        
        loss_m = AverageMeter()
        
        # TODO: set unet and scheduelr to train
        unet.train()
        diffusion_scheduler.train()
        if class_embedder is not None:
            class_embedder.train()
        
        
        # TODO: finish this
        for step, (images, labels) in enumerate(train_loader):
            
            batch_size = images.size(0)
            
            # TODO: send to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            
            # NOTE: this is for latent DDPM 
            if vae is not None:
                # use vae to encode images as latents
                with torch.no_grad():
                    images = vae.encode(images)
                # NOTE: do not change  this line, this is to ensure the latent has unit std
                images = images * args.vae_scale_factor
            batch_mean = images.detach().mean()
            batch_std = images.detach().std()
            
            # TODO: zero grad optimizer
            optimizer.zero_grad(set_to_none=True)
            
            
            # NOTE: this is for CFG
            if class_embedder is not None:
                # TODO: use class embedder to get class embeddings
                class_emb = class_embedder(labels)
            else:
                # NOTE: if not cfg, set class_emb to None
                class_emb = None
            
            # TODO: sample noise 
            noise = torch.randn_like(images)
            
            # TODO: sample timestep t
            timesteps = torch.randint(
                0,
                diffusion_scheduler.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            
            # TODO: add noise to images using scheduler
            noisy_images = diffusion_scheduler.add_noise(images, noise, timesteps)
            
            # TODO: model prediction
            model_pred = unet(noisy_images, timesteps, class_emb)
            
            if args.prediction_type == 'epsilon':
                target = noise 
            else:
                raise NotImplementedError(f"Prediction type {args.prediction_type} not implemented.")
            
            # TODO: calculate loss
            loss = F.mse_loss(model_pred, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Encountered non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
            
            # record loss
            loss_m.update(loss.item(), batch_size)
            
            # backward and step 
            loss.backward()
            # TODO: grad clip
            if args.grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            else:
                grad_norm = torch.norm(
                    torch.stack([
                        p.grad.detach().norm(2)
                        for p in trainable_params
                        if p.grad is not None
                    ]),
                    2,
                )
            
            # TODO: step your optimizer
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            next_completed_steps = completed_steps + 1
            if (
                ema_unet is not None
                and next_completed_steps >= args.ema_start_step
                and next_completed_steps % args.ema_update_every == 0
            ):
                ema_unet.update(unet_wo_ddp)
            
            progress_bar.update(1)
            completed_steps += 1
            
            # logger
            if step % args.log_every == 0 and is_primary(args):
                now = time.perf_counter()
                seconds_per_step = (now - last_log_time) / max(1, args.log_every)
                last_log_time = now
                lr = optimizer.param_groups[0]["lr"]
                grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                metrics = {
                    "epoch": epoch,
                    "step_in_epoch": step,
                    "global_step": completed_steps,
                    "loss_step": float(loss.detach().cpu()),
                    "loss_avg": float(loss_m.avg),
                    "lr": float(lr),
                    "grad_norm": grad_norm_value,
                    "batch_mean": float(batch_mean.detach().cpu()),
                    "batch_std": float(batch_std.detach().cpu()),
                    "timestep_mean": float(timesteps.float().mean().detach().cpu()),
                    "timestep_min": int(timesteps.min().detach().cpu()),
                    "timestep_max": int(timesteps.max().detach().cpu()),
                    "seconds_per_step": float(seconds_per_step),
                }
                if ema_unet is not None:
                    metrics["ema_decay"] = float(ema_unet.decay)
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, "
                    f"Global Step {completed_steps}, Loss {loss.item():.6f} ({loss_m.avg:.6f}), "
                    f"LR {lr:.2e}, GradNorm {grad_norm_value:.4f}, Sec/Step {seconds_per_step:.4f}"
                )
                wandb_logger.log({
                    "train/loss_step": metrics["loss_step"],
                    "train/loss_avg": metrics["loss_avg"],
                    "train/lr": metrics["lr"],
                    "train/grad_norm": metrics["grad_norm"],
                    "train/batch_mean": metrics["batch_mean"],
                    "train/batch_std": metrics["batch_std"],
                    "train/timestep_mean": metrics["timestep_mean"],
                    "train/seconds_per_step": metrics["seconds_per_step"],
                    "global_step": metrics["global_step"],
                    "epoch": metrics["epoch"],
                }, step=completed_steps)
                append_jsonl(metrics_path, {"type": "train", **metrics})

            if completed_steps >= args.max_train_steps:
                stop_training = True
                break

        # validation
        # send unet to evaluation mode
        unet.eval()
        generator = torch.Generator(device=device).manual_seed(epoch + args.seed)
        raw_grid_image = sample_grid(
            pipeline,
            args.image_size,
            args,
            generator,
            device,
        )
        ema_grid_image = None
        if ema_unet is not None:
            generator = torch.Generator(device=device).manual_seed(epoch + args.seed)
            with use_ema_weights(unet_wo_ddp, ema_unet):
                ema_grid_image = sample_grid(
                    pipeline,
                    args.image_size,
                    args,
                    generator,
                    device,
                )
        
        # Send to wandb
        if is_primary(args):
            sample_path_raw = os.path.join(output_dir, "samples", f"epoch_{epoch:04d}_raw.png")
            raw_grid_image.save(sample_path_raw)
            preferred_grid_image = ema_grid_image if ema_grid_image is not None else raw_grid_image
            sample_path = os.path.join(output_dir, "samples", f"epoch_{epoch:04d}.png")
            preferred_grid_image.save(sample_path)
            sample_path_ema = None
            if ema_grid_image is not None:
                sample_path_ema = os.path.join(output_dir, "samples", f"epoch_{epoch:04d}_ema.png")
                ema_grid_image.save(sample_path_ema)
            epoch_metrics = {
                "epoch": epoch,
                "global_step": completed_steps,
                "loss_avg": float(loss_m.avg),
                "sample_path": sample_path,
                "sample_path_raw": sample_path_raw,
            }
            if sample_path_ema is not None:
                epoch_metrics["sample_path_ema"] = sample_path_ema
            wandb_payload = {
                "eval/gen_images": wandb.Image(preferred_grid_image),
                "eval/gen_images_raw": wandb.Image(raw_grid_image),
                "eval/loss_avg_epoch": epoch_metrics["loss_avg"],
                "global_step": completed_steps,
                "epoch": epoch,
            }
            if ema_grid_image is not None:
                wandb_payload["eval/gen_images_ema"] = wandb.Image(ema_grid_image)
            wandb_logger.log(wandb_payload, step=completed_steps)
            append_jsonl(metrics_path, {"type": "epoch", **epoch_metrics})
            
        # save checkpoint
        if is_primary(args):
            save_checkpoint(
                unet_wo_ddp,
                scheduler_wo_ddp,
                vae_wo_ddp,
                class_embedder_wo_ddp,
                optimizer,
                epoch,
                save_dir=save_dir,
                ema_unet=ema_unet,
            )

        if stop_training:
            if is_primary(args):
                logger.info(f"Reached max_train_steps={args.max_train_steps}. Stopping training early.")
            break


if __name__ == '__main__':
    main()
