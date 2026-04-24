
from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch 
from utils import randn_tensor



class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None, vae_scale_factor=0.1845):
        self.unet = unet
        self.scheduler = scheduler
        self.class_embedder = None
        self.vae_scale_factor = vae_scale_factor
        
        # NOTE: this is for latent DDPM
        self.vae = None
        if vae is not None:
            self.vae = vae
            
        # NOTE: this is for CFG
        if class_embedder is not None:
            self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    
    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale : Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device = None,
    ):
        if self.unet is None or self.scheduler is None:
            raise ValueError("Both `unet` and `scheduler` must be provided to the pipeline.")

        image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)
        if device is None:
            try:
                device = next(self.unet.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        
        # NOTE: this is for CFG
        class_emb = None
        uncond_emb = None
        if self.class_embedder is not None and classes is None:
            raise ValueError("A class-conditional pipeline requires `classes` for sampling.")
        if classes is not None:
            if self.class_embedder is None:
                raise ValueError("`classes` were provided, but no class embedder is attached to the pipeline.")
            if isinstance(classes, int):
                classes = [classes] * batch_size
            if len(classes) != batch_size:
                raise ValueError("`classes` must be an int or a list with length equal to `batch_size`.")
            class_labels = torch.tensor(classes, dtype=torch.long, device=device)
            class_emb = self.class_embedder(class_labels)
            if guidance_scale not in (None, 1.0):
                uncond_emb = self.class_embedder.unconditional_embedding(batch_size, device)
        
        image = randn_tensor(image_shape, generator=generator, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            model_input = image

            if uncond_emb is not None:
                noise_uncond = self.unet(model_input, t, c=uncond_emb)
                noise_cond = self.unet(model_input, t, c=class_emb)
                model_output = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                model_output = self.unet(model_input, t, c=class_emb)
            
            image = self.scheduler.step(model_output, int(t), image, generator=generator)
            
        
        if self.vae is not None:
            image = image / self.vae_scale_factor
            image = self.vae.decode(image)
            image = image.clamp(-1.0, 1.0)
        
        image = (image / 2 + 0.5).clamp(0.0, 1.0)
        
        # convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        
        return image
        
