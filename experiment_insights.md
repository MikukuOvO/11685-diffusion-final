# Experiment Insights and Failure Analysis

This note summarizes the main results produced by the current project code and records the likely causes of the observed shortcomings. It is intended as source material for the final experimental report.

## Current Best Result

The best pixel diffusion checkpoint observed so far is:

| Model / Checkpoint | Sampler | Eval Images | Local FID |
| --- | --- | ---: | ---: |
| Pixel DDPM v2, EMA, epoch 14 | DDIM-50 | 1000 | **139.7261** |
| Pixel DDPM, epoch 14 | DDIM-50 | 1000 | 145.1464 |
| Pixel DDPM, epoch 1 | DDIM-50 | 1000 | 150.6496 |
| Pixel DDPM, epoch 18 | DDIM-50 | 1000 | 153.0033 |
| Pixel DDPM, epoch 12 | DDIM-50 | 1000 | 157.9977 |
| Pixel DDPM, epoch 13 | DDIM-50 | 1000 | 162.6118 |
| Pixel DDPM, epoch 0 | DDIM-50 | 1000 | 213.5419 |
| VAE direct generation, epoch 1 | VAE sampling | 1000 | 412.6947 |

The current recommendation is to keep the original Pixel DDPM epoch 14 submission as the best confirmed Kaggle result. Pixel DDPM v2 EMA epoch 14 improved local validation FID, but its hidden-test Kaggle score was worse.

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

Current status:

- The code path exists and tests pass, but a full latent DDPM + CFG long training run has not yet produced a competitive reported FID.
- The VAE baseline FID is weak, so latent diffusion quality may be limited by the VAE checkpoint.

Shortcomings:

- Latent DDPM + CFG is not yet validated as a better final submission path.
- The current VAE is likely not strong enough to support high-quality latent generation.
- CFG guidance scale has not been swept.

Likely causes:

- Latent diffusion depends on both latent denoising quality and decoder quality.
- The current VAE reconstruction/sample quality is poor relative to pixel DDPM.
- CFG introduces additional hyperparameters, especially `cond_drop_rate` and guidance scale, that need validation.

## Main Lessons for the Final Report

### Loss Plateau Does Not Mean Best Samples

The pixel DDPM loss dropped quickly and then remained around `0.020-0.021` for much of training. This suggests the model reached a stable denoising-loss regime early. However, FID still varied across checkpoints, and the final checkpoint was not the best.

Report framing:

> The denoising MSE stabilized early, but FID was not monotonic across checkpoints. Therefore, checkpoint selection using a generated-image validation metric was necessary.

### Longer Training Was Not Automatically Better

Epoch 18 completed the full 150k-step training run, but its FID@1000 was `153.0033`, worse than epoch 14's `145.1464`.

Report framing:

> Continuing training past the best checkpoint did not improve FID, suggesting either mild overtraining, checkpoint noise, or insufficient late-stage optimization control.

### Root Cause Analysis from the Last Two Long Runs

The last two full runs were:

| Run | Steps | Main Changes | Final / Best Local FID | Kaggle Score |
| --- | ---: | --- | ---: | ---: |
| v1 pixel DDPM | 150000 | linear beta, constant LR, raw weights | best measured FID@1000 `145.1464` at epoch 14 | **-126.40334** |
| v2 pixel DDPM EMA | 120000 | cosine beta, warmup+cosine LR, EMA sampling | FID@5000 `115.0269` at epoch 14 | -119.39619 |

Training-duration evidence:

- v1 reached most of its denoising-loss improvement by epoch 1 / about `16250` steps. After that, epoch-level loss stayed around `0.0201-0.0207` through `150000` steps.
- v2 similarly reached its main loss drop by epoch 1 / about `16250` steps. After that, epoch-level loss moved only gradually from about `0.0397` to `0.0362` by `120000` steps.
- v1's final checkpoint was not best: epoch 18 FID@1000 was `153.0033`, worse than epoch 14's `145.1464`.
- These curves argue against "not enough epochs" as the primary failure mode. More steps under the same objective and architecture are unlikely to produce a large quality jump without additional changes.

Training-methodology evidence:

- Both runs optimize epsilon-prediction MSE uniformly over timesteps. This objective can plateau while perceptual quality and FID still move unpredictably.
- The model is unconditional pixel diffusion over 100 ImageNet classes. Without class conditioning at sampling time, the generator must model a broad multimodal distribution, which makes sample quality harder at 128x128.
- The UNet architecture is moderate in capacity for ImageNet-100 128x128. The generated images show texture and color structure but remain blurry with artifacts, which points to a model/data/objective quality ceiling rather than a simple undertraining issue.
- v2 improved local FID but hurt hidden Kaggle score. That suggests the local validation reference is noisy or distribution-shifted relative to hidden test, and it also means we should not tune solely to one local FID number.
- v2 changed EMA, LR schedule, and beta schedule together. Because the ablation is not isolated, we cannot attribute the local improvement cleanly to one factor. The hidden-score regression makes the cosine beta change especially suspect.

Most likely root cause:

> The main issue is not insufficient training duration. The stronger explanation is a methodology and model-selection problem: the current unconditional epsilon-MSE pixel DDPM reaches a denoising-loss plateau early, and additional training mostly changes checkpoint/sample statistics rather than steadily improving distribution quality. Local FID is useful but imperfect, and the v2 run shows that optimizing local FID can fail to transfer to the hidden Kaggle reference.

Recommended next experiment if more GPU time is available:

- Keep the v1 linear beta schedule as the safer baseline.
- Add EMA and cosine LR as isolated changes, but do not change the beta schedule at the same time.
- Evaluate raw and EMA checkpoints separately every few epochs.
- Run a sampler sweep on the best checkpoint: DDIM-50, DDIM-100, and possibly DDPM/high-step sampling.
- If time allows, prioritize class-conditional/CFG pixel diffusion over longer unconditional training, because the competition data has class structure.

### Pixel DDPM Is Currently the Safer Submission Path

The VAE direct-generation FID was `412.6947`, and latent DDPM + CFG has not yet been fully validated. Pixel DDPM epoch 14 is currently the strongest Kaggle-validated model.

Report framing:

> Although latent diffusion and CFG were implemented, the pixel-space DDPM currently gives the most reliable measured sample quality.

## Concrete Improvement Points

### 1. Add EMA Weights

Expected benefit:

- More stable sampling checkpoints.
- Potentially lower FID.
- Less sensitivity to a single checkpoint's raw weights.

Why it addresses the observed issue:

- Current checkpoint FID fluctuates even when training loss is stable. EMA often smooths those fluctuations.

### 2. Use a Learning Rate Schedule

Recommended options:

- Warmup + cosine decay.
- Step decay after the early sharp loss drop.
- Short low-LR fine-tune from the best checkpoint.

Why it addresses the observed issue:

- The constant `2e-4` learning rate may be too aggressive after the model enters the plateau. A decayed learning rate may help refine samples instead of oscillating around similar loss values.

### 3. Automate FID Checkpoint Tracking

Recommended protocol:

- Every few epochs, generate 1000 validation images with DDIM-50.
- Record local FID in a table and W&B.
- Keep the best-FID checkpoint separately from the latest checkpoint.

Why it addresses the observed issue:

- The final checkpoint was not the best. Automatic FID tracking would make model selection more reliable.

### 4. Sweep Final Sampling Settings

Recommended sweep on the best checkpoint:

- DDIM-50.
- DDIM-100.
- Full DDPM or a higher-step DDPM/DDIM variant if time permits.

Why it addresses the observed issue:

- DDIM-50 is efficient and comparable, but the final Kaggle submission may benefit from more sampling steps.

### 5. Improve the VAE Before Relying on Latent Diffusion

Recommended improvements:

- Train the VAE longer.
- Tune KL weight and latent dimensionality.
- Evaluate reconstruction quality visually and numerically before latent DDPM training.

Why it addresses the observed issue:

- The VAE direct FID is very poor. A weak decoder can limit latent diffusion output quality regardless of CFG.

### 6. Sweep CFG Hyperparameters

Recommended sweep:

- Guidance scale: 1.0, 1.5, 2.0, 3.0.
- Conditional dropout: 0.05, 0.1, 0.2.

Why it addresses the observed issue:

- CFG can improve class conditioning but can also reduce diversity or create artifacts if guidance is too strong.

## Report-Ready Conclusion

This trial produced a usable pixel DDPM baseline, but the training dynamics were not ideal. The denoising loss improved rapidly early in training and then remained nearly flat for most of the run. Local FID did not improve monotonically: the best measured checkpoint was epoch 14 with FID@1000 of `145.1464`, while the final epoch 18 checkpoint worsened to `153.0033`. This suggests that the current setup benefits from explicit checkpoint selection and that future runs should include EMA weights, a decaying learning rate schedule, more systematic FID tracking, and a final sampling-step sweep. The VAE and latent DDPM + CFG path should be treated as an implemented but not yet competitive extension until the VAE quality and CFG hyperparameters are improved.
