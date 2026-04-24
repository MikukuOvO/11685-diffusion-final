# Experiment Insights and Failure Analysis

> Status: historical experiment log. The pixel-era recommendations in this file are superseded by the final latent DDPM + CFG results in `VIDEO_REPORT_README.md`. Keep this file as source material for earlier experiments and failure analysis, not as the current best-model summary.

This note summarizes the main results produced by the current project code and records the likely causes of the observed shortcomings. It is intended as source material for the final experimental report.

## Historical Pixel-Era Best Result

The best pixel diffusion checkpoint observed so far is:

| Model / Checkpoint | Sampler | Eval Images | Local FID |
| --- | --- | ---: | ---: |
| Pixel DDPM v3, EMA, epoch 6 | DDIM-50 | 1000 | **134.9025** |
| Pixel DDPM v2, EMA, epoch 14 | DDIM-50 | 1000 | 139.7261 |
| Pixel DDPM, epoch 14 | DDIM-50 | 1000 | 145.1464 |
| Pixel DDPM, epoch 1 | DDIM-50 | 1000 | 150.6496 |
| Pixel DDPM, epoch 18 | DDIM-50 | 1000 | 153.0033 |
| Pixel DDPM, epoch 12 | DDIM-50 | 1000 | 157.9977 |
| Pixel DDPM, epoch 13 | DDIM-50 | 1000 | 162.6118 |
| Pixel DDPM, epoch 0 | DDIM-50 | 1000 | 213.5419 |
| VAE direct generation, epoch 1 | VAE sampling | 1000 | 412.6947 |

The current recommendation is to treat Pixel DDPM v3 as the best local-validation model, but keep the original Pixel DDPM epoch 14 submission as the best confirmed Kaggle result until v3 has a full 5000-image submission. Pixel DDPM v2 EMA epoch 14 improved local validation FID, but its hidden-test Kaggle score was worse.

Kaggle submission result:

| Submission | Source Checkpoint | Sampler | Images | Kaggle Score | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `kaggle_submission_v2_epoch14_ema_ddim50.csv` | Pixel DDPM v2 EMA, epoch 14 | DDIM-50 | 5000 | -119.39619 | Submitted successfully on 2026-04-19. Local FID against validation reference with 5000 generated images: `115.0269`, but hidden-test Kaggle score was worse than v1. |
| `kaggle_submission_epoch14_ddim50.csv` | Pixel DDPM, epoch 14 | DDIM-50 | 5000 | -126.40334 | Submitted successfully on 2026-04-19. The competition appears to report a transformed score, likely negative FID, because conventional FID is non-negative. |

## Code and Experiment Versions

### 1. Initial DDPM Baseline and Short Training Checks

Relevant outputs:

- `output/report_eval/summary.json`
- `output/smoke_run_ddpm/summary.json`

Observed results:

| Checkpoint / Setting | Metric | Result |
| --- | --- | ---: |
| DDPM checkpoint, 200 train steps | denoising MSE, 8 batches | 0.04748 |
| DDPM checkpoint, 3000 train steps | denoising MSE, 8 batches | 0.01818 |
| DDPM checkpoint, 10000 train steps | denoising MSE, 8 batches | 0.01621 |
| DDPM-200 sampling | seconds per image, batch 4 | 1.2499 |
| DDIM-100 sampling | seconds per image, batch 4 | 0.6153 |
| DDIM-50 sampling | seconds per image, batch 4 | 0.3076 |

Interpretation:

- The model learned a denoising objective quickly: the MSE dropped sharply from the 200-step checkpoint to the 3000-step checkpoint.
- The improvement from 3000 to 10000 steps was much smaller, suggesting that the training loss enters a plateau early.
- DDIM sampling is much faster than full DDPM sampling, and DDIM-50 became the practical quick-eval setting.

Shortcomings:

- Early experiments measured denoising MSE and sampling speed, but did not establish a complete FID curve.
- MSE is not a reliable substitute for image distribution quality. It can improve or plateau while FID still fluctuates.

Likely causes:

- The epsilon-prediction MSE objective is a local denoising objective, while FID measures generated image distribution statistics.
- The initial evaluation protocol was too sparse to identify the best checkpoint reliably.

### 2. VAE Baseline

Relevant outputs:

- `output/report_eval/vae_summary.json`
- `src/configs/vae.yaml`
- `src/train_vae.py`
- `src/inference_vae.py`

Observed results:

| Metric | Result |
| --- | ---: |
| Parameters | 55.31M |
| Eval loss, 8 batches | 0.4745 |
| Reconstruction MSE, 8 batches | 0.2862 |
| KL, 8 batches | 0.1883 |
| 1000-image generation time | 9.54 sec |
| Local FID@1000 | 412.6947 |

Interpretation:

- The VAE is very fast at sampling, but direct VAE samples are far below the pixel DDPM results.
- The poor FID indicates that the VAE alone is not suitable as a final generative model.

Shortcomings:

- Reconstruction quality and latent distribution quality are not strong enough for high-quality direct image generation.
- The VAE checkpoint used for downstream latent diffusion is therefore a risk factor.

Likely causes:

- The VAE was trained for limited capacity/limited duration relative to the complexity of ImageNet-100 at 128x128.
- The latent bottleneck and KL regularization can blur reconstructions and weaken sample fidelity if not tuned carefully.
- A weak VAE decoder can cap the quality of latent diffusion even if the latent denoiser trains well.

### 3. Pixel DDPM Long Training on A100 40GB

Relevant config:

- `src/configs/ddpm_a100_40gb.yaml`

Important settings:

| Setting | Value |
| --- | --- |
| Image size | 128 |
| Batch size | 16 |
| Max train steps | 150000 |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Prediction type | epsilon |
| Noise schedule | linear beta, 1000 train timesteps |
| Model width | UNet base channels 128 |
| Sampler for quick eval | DDIM-50 |
| W&B run | `fenglin02/ddpm/runs/cnqmsiyp` |

Training outcome:

- Training completed cleanly at `max_train_steps=150000`.
- Final saved checkpoint: `checkpoint_epoch_18.pth`.
- Final logged train loss average: about `0.02019`.
- Normal training speed after eval jobs ended: about `0.220 sec/step`.

Observed FID curve:

| Checkpoint | Local FID@1000 | Comment |
| --- | ---: | --- |
| epoch 0 | 213.5419 | Poor but improved from random-like behavior |
| epoch 1 | 150.6496 | Large early improvement |
| epoch 12 | 157.9977 | Worse than epoch 1 |
| epoch 13 | 162.6118 | Worse than epoch 12 |
| epoch 14 | **145.1464** | Best measured checkpoint |
| epoch 18 | 153.0033 | Final checkpoint worse than epoch 14 |

Kaggle result:

- A 5000-image submission generated from epoch 14 with DDIM-50 sampling was accepted by Kaggle.
- Leaderboard score: `-126.40334`.
- Since FID is conventionally non-negative, this is most likely a competition-specific transformed score such as negative FID. The important operational result is that the submission format was valid and the hidden-test score was competitive relative to our local validation results.

Interpretation:

- The largest gain happened early.
- After the early phase, training loss remained near a stable level for most of the run.
- FID did not monotonically improve with training time.
- The final checkpoint is not the best checkpoint.
- Checkpoint selection by validation FID is necessary for this experiment.

Shortcomings:

- The long training run did not produce a clear, monotonic improvement in FID.
- The model spent most of the run near a loss plateau.
- The checkpoint with best FID occurred before the end of training.
- The evaluation schedule was sparse, so we know epoch 14 is the best among measured checkpoints, but we have not exhaustively checked every nearby checkpoint.

Likely causes:

- Constant learning rate likely kept the model oscillating around a plateau instead of refining late-stage sample quality.
- No EMA sampling checkpoint was used. Diffusion models often benefit from EMA weights, which can reduce checkpoint-to-checkpoint noise and improve FID.
- Training objective and evaluation metric are mismatched: epsilon MSE can stay flat while FID fluctuates.
- DDIM-50 is a fast, consistent quick-eval sampler, but may not be the best final sampler for the best checkpoint.
- FID@1000 has sampling noise. It is useful for rapid comparison but less stable than a larger evaluation set.

### 4. Pixel DDPM v2 with EMA, Cosine LR, and Cosine Beta Schedule

Relevant config:

- `src/configs/ddpm_v2_ema_cosine_a100_40gb.yaml`

Important settings:

| Setting | Value |
| --- | --- |
| Max train steps | 120000 |
| Learning rate schedule | warmup + cosine decay |
| Minimum LR | 2e-5 |
| EMA decay | 0.9999 |
| Noise schedule | cosine beta, 1000 train timesteps |
| Sampler for quick eval | DDIM-50 |
| W&B run | `fenglin02/ddpm/runs/oy0k2j6f` |

Observed result:

| Checkpoint / Setting | Local FID@1000 | Comment |
| --- | ---: | --- |
| v2 EMA epoch 14 | **139.7261** | Best local FID measured so far |

Submission result:

| Checkpoint / Setting | Eval Images | Local FID | Kaggle Status |
| --- | ---: | ---: | --- |
| v2 EMA epoch 14, DDIM-50 | 5000 | **115.0269** | Complete, Kaggle score `-119.39619` |

Interpretation:

- After fixing EMA checkpoint loading for the existing `shadow` format, the final v2 EMA checkpoint generated structured images rather than pure noise.
- Local FID improved from the v1 best measured value of `145.1464` to `139.7261`.
- Visually, samples still show blur and artifacts, but the distribution statistics improved under the local FID protocol.
- The local FID improvement did not transfer to the hidden Kaggle reference: v2 scored `-119.39619`, worse than v1's `-126.40334` under the leaderboard's apparent negative-FID scoring.

Shortcomings:

- The improvement is measured on 1000 images, so it should be treated as a quick-eval result rather than a final 5000-image estimate.
- The run changed several variables at once: EMA, cosine LR, and cosine beta schedule.
- Early EMA sample grids were misleading because the checkpoint stored EMA as a wrapper state containing `shadow`, while inference initially expected a direct UNet state dict.
- The local validation reference was not fully predictive of hidden-test ranking for this run.

Likely causes:

- EMA likely reduced checkpoint noise and improved sample distribution statistics.
- Cosine LR decay may have helped late-stage refinement compared with the v1 constant learning rate.
- The effect of cosine beta schedule is not isolated, so this run does not prove that cosine beta alone helped.
- The v2 run may have overfit or shifted toward the local validation feature statistics while not improving the hidden test distribution. Another possibility is that changing the beta schedule altered sample texture/color statistics in a way that local FID rewarded but hidden FID penalized.

### 5. Latent DDPM + CFG Implementation

Relevant files:

- `src/configs/latent_cfg_a100_40gb.yaml`
- `src/models/class_embedder.py`
- `src/pipelines/ddpm.py`
- `src/train.py`
- `src/inference.py`
- `src/utils/checkpoint.py`

Implemented features:

- Latent DDPM mode.
- VAE encode/decode path.
- Classifier-free guidance through conditional dropout.
- Unconditional class embedding token.
- CFG sampling support during inference.
- Checkpoint save/load support for class embedder and VAE-related components.

Historical status at the time of this note:

- The code path existed and tests passed, but a full latent DDPM + CFG long training run had not yet produced a competitive reported FID at that point.
- The VAE baseline FID is weak, so latent diffusion quality may be limited by the VAE checkpoint.

Shortcomings:

- Superseded: later Modal training validated latent DDPM + CFG as the stronger final path.
- The current VAE is likely not strong enough to support high-quality latent generation.
- CFG guidance scale has not been swept.

Likely causes:

- Latent diffusion depends on both latent denoising quality and decoder quality.
- The current VAE reconstruction/sample quality is poor relative to pixel DDPM.
- CFG introduces additional hyperparameters, especially `cond_drop_rate` and guidance scale, that need validation.

### 6. Pixel DDPM v3 with Linear Beta, EMA, and Cosine LR

Relevant config:

- `src/configs/ddpm_v3_linear_ema_cosinelr_a100_40gb.yaml`

Important settings:

| Setting | Value |
| --- | --- |
| Max train steps | 50000 |
| Learning rate schedule | warmup + cosine decay |
| Minimum LR | 2e-5 |
| EMA decay | 0.9995 |
| Noise schedule | linear beta, 1000 train timesteps |
| Sampler for quick eval | DDIM-50 |
| W&B run | `fenglin02/ddpm/runs/5ou8091r` |

Observed result:

| Checkpoint / Setting | Eval Images | Local FID | Comment |
| --- | ---: | ---: | --- |
| v3 EMA epoch 6, DDIM-50 | 1000 | **134.9025** | Best quick local FID so far |

Training outcome:

- Training completed cleanly at `max_train_steps=50000`.
- Final saved checkpoint: `checkpoint_epoch_6.pth`.
- Final epoch-level loss average: `0.02052`.
- Final local quick FID@1000: `134.9025`.

Epoch-level loss curve:

| Epoch record | Global Step | Loss Avg |
| ---: | ---: | ---: |
| 0 | 8125 | 0.237797 |
| 1 | 16250 | 0.042213 |
| 2 | 24375 | 0.021238 |
| 3 | 32500 | 0.020462 |
| 4 | 40625 | 0.020313 |
| 5 | 48750 | 0.019712 |
| 6 | 50000 | 0.020520 |

Interpretation:

- v3 gives the best local quick FID so far, improving over v2 by about `4.82` FID points and over v1 by about `10.24` FID points.
- The absolute gain is useful but not large enough to declare a decisive modeling breakthrough.
- Most of the denoising-loss improvement still happens very early: by epoch 2 / `24375` steps, the loss has already reached about `0.0212`.
- From epoch 2 to the final checkpoint, the loss remains close to `0.020`, but the sampled images continue to show clearer object outlines and more coherent large-scale color regions.
- This creates an important diagnostic: denoising MSE is not sensitive enough to capture the visual improvements that still happen after the loss plateaus.

Shortcomings:

- The model still produces many abstract or painterly samples rather than clearly recognizable ImageNet-100 objects.
- The final FID improvement is incremental, not transformative.
- We only have a 1000-image local FID for v3 so far. A full 5000-image FID/submission is still needed before treating it as the best competition checkpoint.
- Local FID has already failed to fully predict Kaggle hidden-test ranking for v2, so v3 must be validated on the leaderboard before replacing the v1 submission.

Likely causes:

- EMA and cosine LR help stabilize and refine samples, but they do not solve the core difficulty of unconditional ImageNet-100 generation at 128x128.
- The epsilon-MSE loss saturates quickly because the model becomes competent at the average denoising task, while semantic object quality and class-level structure remain much harder.
- The visual improvements after the loss plateau likely come from better allocation of probability mass and cleaner sampling trajectories, not from a large reduction in average per-pixel denoising error.
- The current model remains unconditional, so generated samples do not use class labels at sampling time even though the competition data has explicit class structure.

## Main Lessons for the Final Report

### Loss Plateau Does Not Mean Best Samples

The pixel DDPM loss dropped quickly and then remained around `0.020-0.021` for much of training. This suggests the model reached a stable denoising-loss regime early. However, FID still varied across checkpoints, and v3 shows that visual outlines can keep improving even after the scalar loss has mostly plateaued.

Report framing:

> The denoising MSE stabilized early, but FID and visible sample quality continued to change. Therefore, checkpoint selection must use generated-image validation rather than training loss alone.

### Longer Training Was Not Automatically Better

Epoch 18 completed the full 150k-step training run, but its FID@1000 was `153.0033`, worse than epoch 14's `145.1464`.

Report framing:

> Continuing training past the best checkpoint did not improve FID, suggesting either mild overtraining, checkpoint noise, or insufficient late-stage optimization control.

### Root Cause Analysis from the Recent Long Runs

The recent long runs were:

| Run | Steps | Main Changes | Final / Best Local FID | Kaggle Score |
| --- | ---: | --- | ---: | ---: |
| v1 pixel DDPM | 150000 | linear beta, constant LR, raw weights | best measured FID@1000 `145.1464` at epoch 14 | **-126.40334** |
| v2 pixel DDPM EMA | 120000 | cosine beta, warmup+cosine LR, EMA sampling | FID@5000 `115.0269` at epoch 14 | -119.39619 |
| v3 pixel DDPM EMA | 50000 | linear beta, warmup+cosine LR, EMA sampling | FID@1000 `134.9025` at epoch 6 | not submitted yet |

Training-duration evidence:

- v1 reached most of its denoising-loss improvement by epoch 1 / about `16250` steps. After that, epoch-level loss stayed around `0.0201-0.0207` through `150000` steps.
- v2 similarly reached its main loss drop by epoch 1 / about `16250` steps. After that, epoch-level loss moved only gradually from about `0.0397` to `0.0362` by `120000` steps.
- v3 reached `0.0212` loss by epoch 2 / `24375` steps. From there to `50000` steps, loss moved only slightly, ending at `0.0205`.
- v1's final checkpoint was not best: epoch 18 FID@1000 was `153.0033`, worse than epoch 14's `145.1464`.
- These curves argue against "not enough epochs" as the primary failure mode. More steps under the same objective and architecture are unlikely to produce a large quality jump without additional changes.

Training-methodology evidence:

- Both runs optimize epsilon-prediction MSE uniformly over timesteps. This objective can plateau while perceptual quality and FID still move unpredictably.
- The model is unconditional pixel diffusion over 100 ImageNet classes. Without class conditioning at sampling time, the generator must model a broad multimodal distribution, which makes sample quality harder at 128x128.
- The UNet architecture is moderate in capacity for ImageNet-100 128x128. The generated images show texture and color structure but remain blurry with artifacts, which points to a model/data/objective quality ceiling rather than a simple undertraining issue.
- v2 improved local FID but hurt hidden Kaggle score. That suggests the local validation reference is noisy or distribution-shifted relative to hidden test, and it also means we should not tune solely to one local FID number.
- v2 changed EMA, LR schedule, and beta schedule together. Because the ablation is not isolated, we cannot attribute the local improvement cleanly to one factor. The hidden-score regression makes the cosine beta change especially suspect.
- v3 isolates the safer part of v2 by keeping v1's linear beta schedule and adding EMA plus cosine LR. This improved quick local FID to `134.9025`, but the gain is still moderate and does not remove the semantic weakness visible in samples.
- The v3 sample grids indicate that outlines and coarse structure can improve while loss is flat. This means the loss curve alone underestimates late-stage perceptual changes, but it also means the loss curve cannot tell us when to stop.

Most likely root cause:

> The main issue is not simply insufficient training duration. The stronger explanation is a methodology and model-selection problem: the current unconditional epsilon-MSE pixel DDPM reaches a denoising-loss plateau early, while semantic sample quality continues to change in ways that loss does not capture. EMA and cosine LR improve local FID, but the improvement is incremental. To get a larger gain, the next run should improve validation-driven model selection, sampling, and class-conditional generation rather than only extending training.

Recommended next experiment if more GPU time is available:

- Keep the v3 recipe as the current pixel baseline: linear beta, EMA, and warmup+cosine LR.
- First run a full 5000-image v3 evaluation and Kaggle submission to test whether the local FID improvement transfers.
- Add automated FID checkpoint tracking so training does not rely on the final checkpoint or visual inspection.
- Run a sampler sweep on the best v3 checkpoint: DDIM-50, DDIM-100, and possibly DDPM/high-step sampling.
- For the next modeling change, prioritize class-conditional or CFG pixel diffusion over longer unconditional training, because the competition data has class structure and the current unconditional model struggles with recognizable object semantics.

### Pixel DDPM Was the Safer Submission Path at This Stage

The VAE direct-generation FID was `412.6947`, and latent DDPM + CFG had not yet been fully validated at this stage. This conclusion is superseded by the later latent DDPM + CFG runs summarized in `VIDEO_REPORT_README.md`.

Report framing:

> Although latent diffusion and CFG were implemented, the pixel-space DDPM currently gives the most reliable measured sample quality.

## Concrete Improvement Points

### 1. Validate v3 with Full 5000-Image FID and Kaggle Submission

Recommended action:

- Generate 5000 images from `checkpoint_epoch_6.pth` using EMA + DDIM-50.
- Compute full local FID against `val_stats.npz`.
- Submit the 5000-image CSV to Kaggle/Cargo if formatting is valid.

Why it addresses the observed issue:

- v3 improved quick FID@1000 to `134.9025`, but v2 already showed that local validation improvement may not transfer to the hidden leaderboard.
- A full submission is the only way to decide whether v3 should replace the current v1 Kaggle-validated result.

### 2. Automate FID Checkpoint Tracking

Recommended protocol:

- Every epoch or every fixed step interval, generate 1000 validation images with DDIM-50.
- Compute local FID and log it to W&B.
- Save a copy or symlink for the best-FID checkpoint.
- Keep both raw and EMA metrics so we can see whether EMA consistently helps.

Why it addresses the observed issue:

- Loss reaches a plateau early, but visible sample quality and FID continue to change.
- The final checkpoint is not guaranteed to be best, and visual inspection alone is too subjective.

### 3. Sweep Final Sampling Settings

Recommended sweep on the best v3 checkpoint:

- DDIM-50.
- DDIM-100.
- DDIM-200 if time permits.
- Full DDPM or higher-step stochastic sampling if generation time is acceptable.

Why it addresses the observed issue:

- The current FID is measured only with DDIM-50.
- If outlines are improving while loss is flat, sampling trajectory quality may be a meaningful bottleneck.
- A better sampler may improve final images without another long training run.

### 4. Add Class-Conditional Pixel Diffusion or Pixel CFG

Recommended direction:

- Extend the pixel DDPM path to use class labels during training and sampling.
- Start with explicit class conditioning before relying on latent CFG.
- Sweep guidance scale if classifier-free guidance is used.

Why it addresses the observed issue:

- The generated images now have outlines, but object identity remains weak.
- The dataset and competition are class-structured, while the current pixel model is effectively unconditional.
- Conditioning should directly target the semantic weakness that EMA/LR scheduling did not solve.

### 5. Consider a Plateau-Aware Fine-Tune Instead of Longer Training

Recommended options:

- Resume from the best v3 checkpoint with a very low LR.
- Try a short fine-tune with a lower `min_lr`, such as `5e-6` or `1e-5`.
- Increase EMA decay slightly only after confirming sample diversity does not collapse.

Why it addresses the observed issue:

- The model has already reached the low-loss regime.
- More training at the same settings is unlikely to produce a large jump, but a lower-LR refinement run may improve outlines and texture without destabilizing the learned denoiser.

### 6. Improve the VAE Before Relying on Latent Diffusion

Recommended improvements:

- Train the VAE longer.
- Tune KL weight and latent dimensionality.
- Evaluate reconstruction quality visually and numerically before latent DDPM training.

Why it addresses the observed issue:

- The VAE direct FID is very poor. A weak decoder can limit latent diffusion output quality regardless of CFG.

### 7. Sweep CFG Hyperparameters

Recommended sweep:

- Guidance scale: 1.0, 1.5, 2.0, 3.0.
- Conditional dropout: 0.05, 0.1, 0.2.

Why it addresses the observed issue:

- CFG can improve class conditioning but can also reduce diversity or create artifacts if guidance is too strong.

## Report-Ready Conclusion

The experiments show that the model learns the denoising objective quickly, but training loss is not a sufficient proxy for sample quality. In v3, the loss reached about `0.021` by `24375` steps and then stayed close to `0.020`, yet generated samples continued to develop clearer outlines and the final quick local FID improved to `134.9025`. This indicates that EMA and cosine LR are useful, but the improvement is incremental rather than a full solution. The next revision should treat v3 as the current local-FID baseline, validate it with a full 5000-image submission, automate FID-based checkpoint selection, sweep sampling settings, and then prioritize class-conditioned pixel diffusion or pixel CFG to address the remaining semantic weakness in generated images.
