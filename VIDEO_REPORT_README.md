# Video Report Script

This README contains the final narration draft for the project video report. Replace the `[TODO]` placeholders before recording if those results are available.

## Suggested Timing

| Section | Target Time |
| --- | ---: |
| Opening | 10 seconds |
| Problem Description | 35-45 seconds |
| Task | 30 seconds |
| Approach / Methods | 1.5-2 minutes |
| Results & Discussion | 1 minute |
| Conclusions | 25-30 seconds |

## 0. Opening

Hi everyone. Our project is Latent Denoising Diffusion Probabilistic Models for ImageNet-100 generation. We study how to train diffusion models on 128 by 128 ImageNet-100 images, and how different sampling methods trade off generation quality and inference speed.

## 1. Problem Description

Diffusion models are a powerful class of generative models. The basic idea is to gradually add noise to real images, then train a model to reverse this process and generate images from pure noise. Classic DDPMs can produce high-quality samples, but a major limitation is slow sampling, because the reverse denoising process often requires many steps.

Prior work has addressed this in several ways. DDPM introduced the probabilistic denoising framework. DDIM showed that we can accelerate inference with fewer sampling steps without retraining the model. Latent diffusion reduces computation by running diffusion in a compressed VAE latent space. Classifier-free guidance, or CFG, uses conditional and unconditional predictions to improve class-conditional generation.

In this project, we ask three main questions. First, can we implement and train a usable diffusion model for ImageNet-100 from scratch? Second, can DDIM significantly reduce sampling time while preserving quality? Third, how do VAE and latent diffusion compare against pixel-space diffusion?

This task is challenging because ImageNet-100 contains 100 semantic classes and about 130,000 images. For unconditional generation, the model receives no explicit class label, so it must model many object categories and visual modes at once.

## 2. Task

Our work is mainly a re-implementation and empirical evaluation project. Following the handout, we implemented a DDPM scheduler, a DDIM scheduler, a VAE baseline, latent DDPM, classifier-free guidance, and an evaluation pipeline for FID and Kaggle submission.

The dataset is ImageNet-100 at 128 by 128 resolution. We resize each image to 128 by 128, apply random horizontal flip, convert it to a tensor, and normalize pixel values from `[0, 1]` to `[-1, 1]`.

Our main model is an unconditional pixel-space DDPM. This means the diffusion model is trained directly in RGB image space, without using VAE latents and without using class labels. We also train and evaluate a VAE baseline, and we implement latent DDPM with CFG as the extension required by the handout.

Visual cue: show several representative ImageNet-100 training images here.

## 3. Approach / Methods

Our implementation has three main paths.

The first and most important path is the unconditional pixel-space DDPM. During training, we start from a real image `x0`, randomly sample a timestep `t`, add Gaussian noise to obtain `xt`, and train a UNet to predict the added noise. The UNet input is the noisy image and the timestep, and the output is the predicted epsilon noise. The loss is mean squared error between predicted noise and true noise.

Our current v0 pixel DDPM uses 1000 diffusion training timesteps with a linear beta schedule from `0.0001` to `0.02`. The UNet directly maps 3-channel 128 by 128 RGB tensors to predicted 3-channel noise tensors. It uses timestep embeddings, residual blocks, skip connections, and self-attention at lower spatial resolutions. The model has about 169.5 million parameters.

For optimization, we use AdamW with an initial learning rate of `1e-4`, warmup, and cosine decay. We also maintain an exponential moving average, or EMA, of the UNet weights. During evaluation, we usually prefer EMA weights because they produce more stable samples.

The second path is DDIM sampling. DDIM is not a separately trained model. Instead, it changes the reverse sampling process for the same DDPM checkpoint. We compare DDPM-200, DDIM-100, and DDIM-50, where the number indicates the number of reverse denoising steps. This directly evaluates the handout's requirement to study inference speedup and quality-speed trade-off.

The third path is VAE, latent DDPM, and CFG. The VAE baseline learns an encoder-decoder representation for images. Latent DDPM uses the VAE encoder to map images into latent space, scales the latents by `0.1845`, trains diffusion in that latent space, and uses the VAE decoder to convert generated latents back into RGB images. The motivation is that latent space can be computationally cheaper than pixel space.

CFG, or classifier-free guidance, supports class-conditional generation. During training, class conditions are randomly dropped so the same model learns both conditional and unconditional denoising. During sampling, we compute both predictions and combine them as:

```text
epsilon = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```

For ImageNet-100, CFG can generate 50 images per class, giving the 5000 images required by Kaggle.

For evaluation, we mainly use Frechet Inception Distance, or FID. Lower FID means the generated image feature distribution is closer to the real validation distribution. We also report seconds per image to measure sampling speed. Kaggle submission requires 5000 generated images and a CSV containing Inception feature statistics.

Visual cue: show a pipeline diagram with real image, forward noise, UNet denoising, reverse sampler, and generated image. Also show that DDPM and DDIM share the same checkpoint but use different samplers.

## 4. Results & Discussion

Our experiments show that pixel-space DDPM is the most reliable generation path so far.

For speed, DDIM provides a clear improvement. In early experiments, DDPM-200 took about `1.2499` seconds per image, DDIM-100 took about `0.6153` seconds per image, and DDIM-50 took about `0.3076` seconds per image. This means DDIM-50 is roughly four times faster than DDPM-200.

For quality, the VAE baseline was much weaker. Direct VAE sampling reached FID@1000 of `412.6947`, and the samples were blurry. Pixel DDPM performed substantially better. A previous pixel DDPM checkpoint at epoch 14 reached FID@1000 of `145.1464`, and its Kaggle submission was valid with score `-126.40334`.

With EMA and learning-rate scheduling, local FID improved further. In the current v0 run, epoch 25 EMA with DDIM-50 reached local FID@1000 of `122.4505`.

[TODO: current v0 final checkpoint FID@1000.]

[TODO: final selected best checkpoint, sampler, and FID@1000.]

[TODO: final 5000-image local FID and Kaggle score.]

We also found that training loss is not enough for model selection. The denoising loss reaches a plateau relatively early, but FID and visual quality still vary across checkpoints. Therefore, we select checkpoints using generated-image validation rather than only using the final checkpoint.

For latent DDPM and CFG, the code path is implemented, including VAE encode/decode, latent scaling, class embedding, condition dropout, and CFG sampling. However, it has not become our strongest result so far. A likely reason is that the VAE baseline itself is weak, so latent representation and decoder quality can limit the final image quality.

[TODO: latent DDPM FID@1000, if available.]

[TODO: latent DDPM + CFG FID@1000, if available.]

Visually, pixel DDPM can generate natural textures and some local object structure, such as bird-like, animal-like, or natural-scene-like patterns. The main failure cases are blurry boundaries, unstable object identity, and samples that look more like abstract textures than clear objects. A likely cause is that our strongest pixel model is unconditional, so it must model all 100 classes without explicit semantic guidance.

## 5. Conclusions

In summary, we implemented DDPM, DDIM, VAE, latent DDPM, and CFG, and evaluated them on ImageNet-100 generation. Our main finding is that pixel-space DDPM is the most reliable path so far, DDIM significantly speeds up sampling, VAE direct sampling is a weak baseline, and EMA plus checkpoint selection are important for FID.

For final submission, we will use the best local-FID pixel DDPM EMA checkpoint with DDIM sampling to generate 5000 images for Kaggle. The next steps are to complete the DDPM-200, DDIM-100, and DDIM-50 comparison on the same checkpoint, and to further explore class-conditional pixel diffusion or CFG to improve semantic consistency.

