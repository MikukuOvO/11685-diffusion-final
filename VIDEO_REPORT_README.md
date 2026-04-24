# Video Report Script

This file is the current 5-minute video-report plan for the ImageNet-100 diffusion project.

## Timing

| Section | Time | Content |
| --- | ---: | --- |
| Title | 0:00-0:10 | Project title and team |
| Problem and motivation | 0:10-0:40 | Why 128x128 ImageNet-100 generation is hard |
| Task, data, baselines | 0:40-1:10 | Dataset, evaluation, pixel DDPM baseline |
| Methodology | 1:10-2:40 | Pixel DDPM, DDIM, latent DDPM, CFG |
| Results | 2:40-3:40 | FID/Kaggle table and sample grids |
| Limitations | 3:40-4:05 | Failure cases and local-hidden gap |
| Future work | 4:05-4:30 | Clean ablations and validation automation |
| Conclusion | 4:30-4:50 | Main takeaways |
| Bibliography | 4:50-4:55 | References on screen |

## Core Story

We first implemented a pixel-space DDPM baseline directly on 128x128 RGB images. Then we added DDIM sampling, a frozen official VAE, latent DDPM, class conditioning, classifier-free guidance, EMA, and cosine learning-rate scheduling. The strongest result so far comes from the latent DDPM + CFG path. The key result story is not only that later checkpoints improve, but that a fixed-checkpoint ablation shows how CFG scale and DDIM step count determine the final submission setting. The current best setting is epoch 280 with DDIM-250 and CFG 2.75, which reached local FID 26.5575 and Kaggle score -25.83352.

## 0. Opening

Hi everyone. Our project is diffusion models for class-conditional image generation on ImageNet-100. We generate 128 by 128 images from 100 classes, evaluate with FID, and study the quality-speed trade-off between DDPM and DDIM sampling. We start from a pixel-space DDPM baseline and then extend the system with latent diffusion and classifier-free guidance.

## 1. Problem and Motivation

Image synthesis is a canonical generative-modeling task. The challenge in our setting is that ImageNet-100 has 100 semantic classes and high visual diversity, while 128 by 128 RGB images still have a large pixel space. A good model needs both fidelity, meaning realistic local details, and diversity, meaning coverage of many object categories and backgrounds.

Prior work motivates our design. DDPM introduced denoising diffusion as a strong but slow generative model. DDIM showed that the same trained checkpoint can be sampled with fewer reverse steps. Latent diffusion reduces computation by moving the diffusion process into a VAE latent space. Classifier-free guidance improves class-conditional generation by combining conditional and unconditional predictions.

Our goal is not only to reproduce a diffusion model, but also to understand which design choices matter under a fixed ImageNet-100 Kaggle evaluation.

## 2. Task, Data, and Baselines

This is a re-implementation and application project. We implement the training and sampling pipeline, apply it to ImageNet-100 at 128 by 128 resolution, and evaluate generated-image quality with local FID and the Kaggle hidden test score.

The dataset contains roughly 130 thousand training images across 100 classes. We resize images to 128 by 128, apply random horizontal flip during training, convert images to tensors, and normalize pixel values to the range from -1 to 1.

The baseline is a pixel-space DDPM trained directly on RGB images. We also keep a VAE baseline for comparison, and build latent DDPM plus CFG as the stronger extension. For Kaggle, we generate 5000 images, 50 per class, extract Inception-v3 features, and submit the mean and covariance statistics.

## 3. Methodology

Our first path is pixel-space DDPM. During training, we sample a clean image x0, choose a random timestep t, add Gaussian noise to produce xt, and train a UNet to predict the added noise epsilon. The loss is mean squared error between predicted and true noise. The model operates directly on 3-channel 128 by 128 RGB tensors.

Our pixel run uses 1000 training timesteps, a linear beta schedule, AdamW, warmup plus cosine learning-rate decay, and EMA weights for evaluation. This gives a fair baseline for the original handout objective: train a usable pixel-space DDPM and test accelerated sampling with DDIM.

The second path is DDIM sampling. DDIM is not a new trained model. It reuses the same DDPM checkpoint and changes the reverse denoising trajectory. We use DDIM-50 as the fast evaluation sampler and DDIM-250 as the higher-quality submission sampler. In our experiments, DDIM-250 gave better local and Kaggle results than DDIM-50, while DDIM-500 was slower and did not improve the hidden score.

The third path is latent DDPM with classifier-free guidance. We use the official frozen VAE to encode images into 32 by 32 latents, train the diffusion model in latent space, and decode generated latents back to RGB. This reduces the spatial cost of diffusion substantially. For conditioning, we add a class embedder and use condition dropout during training. At sampling time, CFG combines unconditional and conditional noise predictions:

```text
epsilon = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```

We use DDIM-50 to sweep guidance scales cheaply, then run DDIM-250 with the best guidance value for full FID and Kaggle submission. For latent inference we set `clip_sample=false`; for pixel inference we keep `clip_sample=true`.

## 4. Results and Discussion

The pixel DDPM baseline establishes the gap we need to close. With EMA and DDIM-50, the pixel model at epoch 14, or 60K steps, reached local FID@5000 of 92.3896. The samples have natural colors and local textures, but many images still look like texture collages rather than stable semantic objects.

For the latent model, we selected the final setting in two stages. First, we fixed the checkpoint and used DDIM-50 to sweep CFG cheaply. On the epoch 221 checkpoint, CFG 2.75 was the best local setting, and CFG 3.0 was almost tied but slightly worse:

| Checkpoint | Sampler | CFG | Local FID@5000 |
| --- | --- | ---: | ---: |
| Latent epoch 221 | DDIM-50 | 2.00 | 33.2586 |
| Latent epoch 221 | DDIM-50 | 2.25 | 30.6560 |
| Latent epoch 221 | DDIM-50 | 2.50 | 29.3073 |
| Latent epoch 221 | DDIM-50 | 2.75 | **28.7364** |
| Latent epoch 221 | DDIM-50 | 3.00 | 28.7687 |

Second, after continuing to epoch 280, we kept CFG 2.75 and compared DDIM step counts. DDIM-250 was better than DDIM-50 on local FID and became the final Kaggle setting:

| Checkpoint | Sampler | CFG | Local FID@5000 | Kaggle |
| --- | --- | ---: | ---: | ---: |
| Latent epoch 280 | DDIM-50 | 2.75 | 28.2489 | not submitted |
| Latent epoch 280 | DDIM-250 | 2.75 | **26.5575** | **-25.83352** |

We also saw the same DDIM-step trend earlier at epoch 140 with CFG 3.0:

| Checkpoint | Sampler | CFG | Local FID@5000 | Kaggle |
| --- | --- | ---: | ---: | ---: |
| Latent epoch 140 | DDIM-50 | 3.0 | 31.3175 | -31.06839 |
| Latent epoch 140 | DDIM-100 | 3.0 | 30.2484 | not submitted |
| Latent epoch 140 | DDIM-250 | 3.0 | 29.6562 | -29.36503 |

The main result is that latent DDPM + CFG is substantially stronger than the pixel DDPM baseline, and DDIM-250 gives a better quality-speed point than DDIM-50 for final submission. We used DDIM-50 for fast sweeps, then DDIM-250 for the selected checkpoint. DDIM-500 was tested in an earlier run, but it was slower and did not improve the hidden Kaggle score.

We also observed that checkpoint selection matters. Training loss alone was not sufficient for model choice: FID continued changing after the loss had mostly plateaued. Continuing latent training from 150K to 190K steps improved both local FID and Kaggle score. We are also running a later continuation to test whether this trend keeps improving.

## 5. Limitations and Failure Cases

The pixel-space DDPM is expensive because it denoises full 128 by 128 RGB images. Its samples often show texture-like patterns instead of clean semantic objects, especially because this baseline is not class-conditioned.

The latent model depends on the frozen VAE. The official VAE makes latent training feasible, but decoder bottlenecks can still introduce blur or local artifacts. Early latent checkpoints especially showed texture smoothing and repeated visual patterns.

Our local FID does not perfectly match the hidden Kaggle reference. The overall trend is useful, but it is not exact; for example, DDIM-500 did not transfer to a better hidden score even when local FID looked competitive.

Finally, some improvements are bundled together, including EMA, cosine learning-rate decay, latent-space training, CFG, and sampler changes. This means the final model is strong, but the causal ablation is not fully isolated yet.

## 6. Future Work

First, we should run clean ablations: EMA on versus off, learning-rate schedule changes, DDIM step count, CFG scale, and `clip_sample` should be varied one at a time.

Second, we should automate checkpoint evaluation. A reliable pipeline should run DDIM-50 every fixed number of epochs, compute local FID, and only run DDIM-250 on promising checkpoints.

Third, we should train a class-conditioned pixel DDPM baseline. That would separate the benefit of class conditioning and CFG from the benefit of latent compression.

Fourth, we should improve validation reliability by evaluating multiple random seeds or using a local reference split closer to the Kaggle hidden distribution.

## 7. Conclusion

In this project, we built a complete DDPM system for ImageNet-100 generation, including pixel-space DDPM, DDIM sampling, VAE support, latent DDPM, CFG, EMA, Modal training, and Kaggle submission generation. The best current model is latent DDPM + CFG with DDIM-250 and CFG 2.75, reaching local FID 26.5575 and Kaggle -25.83352. The main takeaway is that latent diffusion, guidance-scale selection, sampler choice, and checkpoint selection are more important than simply training longer or relying only on denoising loss.

## 8. Bibliography

```text
[1] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS, 2020.
[2] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," ICLR, 2021.
[3] R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR, 2022.
[4] J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," NeurIPS Workshop, 2021.
[5] M. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," NeurIPS, 2017.
[6] J. Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database," CVPR, 2009.
```
