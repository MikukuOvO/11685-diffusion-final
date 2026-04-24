<div align="center">

<br>

<h1 style="font-size: 2.8em; margin-bottom: 0.2em; font-weight: 700; letter-spacing: -0.02em;">
  Diffusion Models for<br>Class-Conditional Image Generation
</h1>

<p style="font-size: 1.2em; color: #6b7280; margin-top: 0; font-style: italic;">
  ImageNet-100 · 128×128 · 100 classes
</p>

<p>
  <img src="https://img.shields.io/badge/CMU-11--785-C41E3A?style=flat-square&labelColor=2d3748" alt="CMU 11-785">
  <img src="https://img.shields.io/badge/Spring-2026-4a5568?style=flat-square" alt="Spring 2026">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Modal-Training-7C3AED?style=flat-square" alt="Modal">
</p>

</div>

---

<div align="center">

<table>
<tr>
<td align="center" width="200">
<sub>🏆 BEST MODEL</sub><br>
<b style="font-size: 1.1em;">Latent DDPM + CFG</b><br>
<sub>DDIM-250 · w=2.75 · ep280</sub>
</td>
<td align="center" width="140">
<sub>LOCAL FID</sub><br>
<b style="font-size: 2em; color: #059669;">26.56</b>
</td>
<td align="center" width="140">
<sub>INCEPTION SCORE</sub><br>
<b style="font-size: 2em; color: #2563eb;">33.91</b><br>
<sub>± 1.59</sub>
</td>
<td align="center" width="140">
<sub>KAGGLE SCORE</sub><br>
<b style="font-size: 2em; color: #dc2626;">−25.83</b>
</td>
</tr>
</table>

<br>

<blockquote>
<b>Core takeaway —</b> <i>sampler choice, guidance scale, and checkpoint selection<br>matter more than simply training longer.</i>
</blockquote>

</div>

---

## 🎯 Problem

Generate **128×128** images across **100 ImageNet classes** that balance two competing objectives:

<table>
<tr>
<td width="50%" align="center">
<h3>🔬 Fidelity</h3>
<i>Realistic local detail</i>
</td>
<td width="50%" align="center">
<h3>🌈 Diversity</h3>
<i>Coverage across categories</i>
</td>
</tr>
</table>

### Prior work that shaped my design

<table>
<thead>
<tr>
<th>Paper</th>
<th>Contribution</th>
<th>What I use it for</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>DDPM</b> <sub>Ho et al. '20</sub></td>
<td>Denoising diffusion — strong quality, slow sampling</td>
<td>Pixel baseline + training objective</td>
</tr>
<tr>
<td><b>DDIM</b> <sub>Song et al. '21</sub></td>
<td>Same checkpoint, fewer sampling steps</td>
<td>Accelerated, deterministic inference</td>
</tr>
<tr>
<td><b>Latent Diffusion</b> <sub>Rombach et al. '22</sub></td>
<td>Diffuse in VAE latent space</td>
<td>Training at 32² instead of 128²</td>
</tr>
<tr>
<td><b>CFG</b> <sub>Ho & Salimans '21</sub></td>
<td>Class-conditional generation, no classifier</td>
<td>Controllable, class-faithful samples</td>
</tr>
</tbody>
</table>

> **My question —** which of these choices actually move the needle once Kaggle evaluation is fixed?

---

## 📦 Task · Data · Baselines

<table>
<tr>
<td width="50%" valign="top">

<h4>📊 Dataset</h4>

<b>ImageNet-100</b> · 128×128 RGB<br>
~<b>130K</b> training images · <b>100</b> classes

<br><br>

<b>Preprocessing pipeline</b>

```
resize → hflip → normalize [−1, 1]
```

</td>
<td width="50%" valign="top">

<h4>📤 Kaggle submission</h4>

<b>5,000</b> images (50 per class)<br>
Inception-v3 features extracted<br>
Submit <b>(μ, Σ)</b> statistics

<br><br>

Evaluation: hidden FID reference

</td>
</tr>
</table>

### Three paths kept in parallel

<table>
<tr>
<td width="33%" align="center" valign="top">
<br>
<b>① Pixel DDPM</b><br>
<sub>BASELINE</sub><br>
<br>
Denoise in pixel space<br>
<sub>Slow · unconditional</sub>
</td>
<td width="33%" align="center" valign="top">
<br>
<b>② Direct VAE</b><br>
<sub>REFERENCE</sub><br>
<br>
VAE reconstruction only<br>
<sub>Much weaker</sub>
</td>
<td width="33%" align="center" valign="top">
<br>
<b>③ Latent + CFG</b> ⭐<br>
<sub>STRONG EXTENSION</sub><br>
<br>
VAE latents + guidance<br>
<sub>Final submission</sub>
</td>
</tr>
</table>

---

## 🏗️ Methodology

<table>
<thead>
<tr>
<th width="25%">Component</th>
<th>What happens</th>
<th width="30%">Key details</th>
</tr>
</thead>
<tbody>
<tr>
<td valign="top">
<b>① Pixel DDPM</b><br>
<sub>🔵 Baseline</sub>
</td>
<td valign="top">

```
x₀ ──[add noise, T=1000]──► x_t ──► U-Net ──► ε̂    (MSE)
```

Clean image noised over 1,000 linear-β timesteps; U-Net predicts the added noise.

</td>
<td valign="top">
<sub>
• AdamW<br>
• Warmup + cosine LR<br>
• EMA weights<br>
• 1,000 timesteps
</sub>
</td>
</tr>
<tr>
<td valign="top">
<b>② DDIM Sampling</b><br>
<sub>🔴 Same checkpoint</sub>
</td>
<td valign="top">

```
ε ~ 𝒩(0,I) ──► DDIM-50   (fast sweeps)
             ──► DDIM-250  (final submission)
```

Non-Markovian, deterministic reverse trajectory — **not a new model**, just a different sampler on the same weights.

</td>
<td valign="top">
<sub>
• DDIM-50: sweep<br>
• DDIM-250: submit<br>
• DDIM-500: no gain<br>
• Deterministic
</sub>
</td>
</tr>
<tr>
<td valign="top">
<b>③ Latent DDPM</b><br>
<sub>⚪ Frozen VAE</sub>
</td>
<td valign="top">

```
x₀(128²) ──► VAE.enc ──► z₀(32²) ──► diffusion ──► ẑ₀ ──► VAE.dec ──► x̂₀
```

Diffusion runs in 32×32 latent space, not 128×128 pixels. Cuts spatial cost substantially.

</td>
<td valign="top">
<sub>
• Frozen official VAE<br>
• 128² → 32² latent<br>
• <code>clip_sample=false</code><br>
• Feasibility unlock
</sub>
</td>
</tr>
<tr>
<td valign="top">
<b>④ CFG</b><br>
<sub>🟢 Class-conditional</sub>
</td>
<td valign="top">

```
Training:    class embedder + drop label (p=0.1)
Inference:   ε̂ = ε_uncond + w · (ε_cond − ε_uncond)
```

At inference, blend conditional and unconditional predictions using guidance scale *w*.

</td>
<td valign="top">
<sub>
• Dropout p=0.1<br>
• Guidance scale <i>w</i><br>
• Higher w → stronger class<br>
• Best: w=2.75
</sub>
</td>
</tr>
</tbody>
</table>

### 🛠 Advanced training techniques

<table>
<tr>
<td align="center" width="25%">
<b>EMA weights</b><br>
<sub>Stable evaluation</sub>
</td>
<td align="center" width="25%">
<b>Mixed precision</b><br>
<sub>AMP fp16</sub>
</td>
<td align="center" width="25%">
<b>Cosine LR + warmup</b><br>
<sub>Smooth decay</sub>
</td>
<td align="center" width="25%">
<b>FID-driven ckpt sel</b><br>
<sub>Beat loss plateau</sub>
</td>
</tr>
</table>

---

## 📈 Results

### ① Baselines

<table>
<thead>
<tr>
<th>Model</th>
<th>Sampler</th>
<th align="right">Local FID</th>
<th align="right">IS</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pixel DDPM <sub>(ep14, ~60K steps)</sub></td>
<td>DDIM-50</td>
<td align="right"><code>92.39</code></td>
<td align="right"><code>7.69 ± 0.23</code></td>
</tr>
<tr>
<td>Direct VAE <sub>(no diffusion)</sub></td>
<td>—</td>
<td align="right"><code>412.69</code> ❌</td>
<td align="right">—</td>
</tr>
</tbody>
</table>

> Direct VAE is far too weak to generate — so I use it strictly as encoder/decoder for the latent path.

---

### ② CFG guidance-scale sweep

<sub>Fixed checkpoint: latent DDPM at epoch 221 · Fixed sampler: DDIM-50</sub>

<table>
<thead>
<tr>
<th align="right">CFG <i>w</i></th>
<th align="right">Local FID</th>
<th align="right">Inception Score</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td align="right">2.00</td>
<td align="right">33.26</td>
<td align="right">25.05 ± 0.92</td>
<td></td>
</tr>
<tr>
<td align="right">2.25</td>
<td align="right">30.66</td>
<td align="right">28.20 ± 1.65</td>
<td></td>
</tr>
<tr>
<td align="right">2.50</td>
<td align="right">29.31</td>
<td align="right">30.65 ± 1.19</td>
<td></td>
</tr>
<tr style="background-color: #fef3c7;">
<td align="right"><b>2.75</b></td>
<td align="right"><b>28.74</b> ⭐</td>
<td align="right">33.01 ± 1.44</td>
<td><b>Winner</b></td>
</tr>
<tr>
<td align="right">3.00</td>
<td align="right">28.77</td>
<td align="right">34.61 ± 1.45</td>
<td>Highest IS, FID ≈ tied</td>
</tr>
</tbody>
</table>

---

### ③ DDIM step-count sweep

<sub>Fixed checkpoint: latent DDPM at epoch 280 · Fixed CFG: w = 2.75</sub>

<table>
<thead>
<tr>
<th>Sampler</th>
<th align="right">Local FID</th>
<th align="right">Inception Score</th>
<th align="right">Kaggle</th>
</tr>
</thead>
<tbody>
<tr>
<td>DDIM-50</td>
<td align="right">28.25</td>
<td align="right">33.90 ± 1.25</td>
<td align="right">—</td>
</tr>
<tr style="background-color: #fef3c7;">
<td><b>DDIM-250</b></td>
<td align="right"><b>26.56</b> 🏆</td>
<td align="right"><b>33.91 ± 1.59</b></td>
<td align="right"><b>−25.83</b> 🏆</td>
</tr>
</tbody>
</table>

### Consistency check — same trend at epoch 140

<details>
<summary>Earlier checkpoint confirms: more DDIM steps → better FID and Kaggle (<i>click to expand</i>)</summary>

<br>

<sub>Fixed checkpoint: latent DDPM at epoch 140 · Fixed CFG: w = 3.0</sub>

<table>
<thead>
<tr>
<th>Sampler</th>
<th align="right">Local FID</th>
<th align="right">Kaggle</th>
</tr>
</thead>
<tbody>
<tr><td>DDIM-50</td>   <td align="right">31.32</td> <td align="right">−31.07</td></tr>
<tr><td>DDIM-100</td>  <td align="right">30.25</td> <td align="right">—</td></tr>
<tr><td>DDIM-250</td>  <td align="right">29.66</td> <td align="right">−29.37</td></tr>
</tbody>
</table>

Same pattern, different epoch — not a one-off.

</details>

---

## ⚠️ Limitations

<table>
<tr>
<td width="33%" valign="top" align="center">
<h4>🎨 Unconditional baseline</h4>
Pixel DDPM has no class conditioning — samples read as texture collages, not objects.
</td>
<td width="33%" valign="top" align="center">
<h4>🔒 Frozen-VAE ceiling</h4>
Latent quality is bounded by decoder capacity — early checkpoints show blur and repetition.
</td>
<td width="33%" valign="top" align="center">
<h4>📊 Local ≠ Kaggle</h4>
Local FID isn't a perfect proxy — DDIM-500 looked competitive locally but didn't transfer.
</td>
</tr>
</table>

---

## 🔭 Future Work

<table>
<tr>
<td width="50%" valign="top">

<h4>🧪 1. Clean ablations</h4>

Vary each one at a time: <b>EMA · LR schedule · DDIM steps · CFG · <code>clip_sample</code></b> — isolate what actually matters.

</td>
<td width="50%" valign="top">

<h4>🔄 2. Automated eval</h4>

DDIM-50 every N epochs → auto-promote promising checkpoints to DDIM-250 full FID.

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h4>🎯 3. Class-conditioned pixel baseline</h4>

Separate the benefit of <b>class conditioning</b> from <b>latent compression</b> — a cleaner apples-to-apples comparison.

</td>
<td width="50%" valign="top">

<h4>🌱 4. Multi-seed validation</h4>

Reference split closer to the hidden Kaggle distribution — more reliable local signal.

</td>
</tr>
</table>

---

## 📚 References

<sub>

1. J. Ho, A. Jain, P. Abbeel. **Denoising Diffusion Probabilistic Models.** *NeurIPS* 2020.
2. J. Song, C. Meng, S. Ermon. **Denoising Diffusion Implicit Models.** *ICLR* 2021.
3. R. Rombach et al. **High-Resolution Image Synthesis with Latent Diffusion Models.** *CVPR* 2022.
4. J. Ho, T. Salimans. **Classifier-Free Diffusion Guidance.** *NeurIPS Workshop* 2021.
5. M. Heusel et al. **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.** *NeurIPS* 2017.
6. J. Deng et al. **ImageNet: A Large-Scale Hierarchical Image Database.** *CVPR* 2009.

</sub>

---

<div align="center">
<sub>
Built on PyTorch 2.x · Trained on Modal · Submitted to Kaggle InClass
</sub>
</div>