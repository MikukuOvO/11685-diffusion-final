from typing import List, Optional, Tuple, Union

import torch 
import torch.nn as nn 
import numpy as np

from utils import randn_tensor


def betas_for_alpha_bar(num_train_timesteps, max_beta=0.999):
    """Cosine schedule from Improved DDPM, discretized through alpha_bar(t)."""
    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        alpha_bar_t1 = np.cos((t1 + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar_t2 = np.cos((t2 + 0.008) / 1.008 * np.pi / 2) ** 2
        betas.append(min(1 - alpha_bar_t2 / alpha_bar_t1, max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(nn.Module):
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        variance_type: str = "fixed_small",
        prediction_type: str = 'epsilon',
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        """
        Args:
            num_train_timesteps (`int`): 
            
        """
        super(DDPMScheduler, self).__init__()
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.beta = beta_start
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
    
        if self.beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif self.beta_schedule == 'cosine':
            betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"Beta schedule {self.beta_schedule} not implemented.")
        self.register_buffer("betas", betas)
         
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)
        self.register_buffer("timesteps", timesteps)
        

    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
            
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(
            0,
            self.num_train_timesteps - 1,
            num_inference_steps,
            dtype=np.int64,
        )[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)


    def __len__(self):
        return self.num_train_timesteps


    def previous_timestep(self, timestep):
        """
        Get the previous timestep for a given timestep.
        
        Args:
            timestep (`int`): The current timestep.
        
        Return: 
            prev_t (`int`): The previous timestep.
        """
        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        )
        timestep = int(timestep)
        if num_inference_steps == self.num_train_timesteps:
            prev_t = timestep - 1
        else:
            timestep_matches = (self.timesteps == timestep).nonzero(as_tuple=False)
            if timestep_matches.numel() == 0:
                raise ValueError(f"Timestep {timestep} is not in the current inference schedule.")
            timestep_index = timestep_matches[0].item()
            if timestep_index == len(self.timesteps) - 1:
                prev_t = -1
            else:
                prev_t = int(self.timesteps[timestep_index + 1].item())
        return prev_t

    
    def _get_variance(self, t):
        """
        This is one of the most important functions in the DDPM. It calculates the variance $sigma_t$ for a given timestep.
        
        Args:
            t (`int`): The current timestep.
        
        Return:
            variance (`torch.Tensor`): The variance $sigma_t$ for the given timestep.
        """
        
        
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.alphas_cumprod.new_tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
    
        # Compute the posterior variance used when sampling x_{t-1}.
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Clamp to avoid numerical issues at very small variance.
        variance = torch.clamp(variance, min=1e-20)

        if self.variance_type == "fixed_small":
            variance = variance
        elif self.variance_type == "fixed_large":
            variance = current_beta_t
        else:
            raise NotImplementedError(f"Variance type {self.variance_type} not implemented.")

        return variance
    
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor: 
        """
        Add noise to the original samples. This function is used to add noise to the original samples at the beginning of each training iteration.
        
        
        Args:
            original_samples (`torch.Tensor`): 
                The original samples.
            noise (`torch.Tensor`): 
                The noise tensor.
            timesteps (`torch.IntTensor`): 
                The timesteps.
        
        Return:
            noisy_samples (`torch.Tensor`): 
                The noisy samples.
        """
        
        # make sure alphas the on the same device as samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps]
        sqrt_alpha_prod = sqrt_alpha_prod.sqrt()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = 1 - alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.sqrt()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """
        
        
        t = int(timestep)
        prev_t = self.previous_timestep(t)
        
        alpha_prod_t = self.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t].to(device=sample.device, dtype=sample.dtype)
            if prev_t >= 0
            else sample.new_tensor(1.0)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # Predict x_0 from the model's epsilon prediction.
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample


        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            variance = self._get_variance(t).to(device=device, dtype=model_output.dtype).sqrt() * variance_noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample
