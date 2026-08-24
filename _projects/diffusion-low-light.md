---
layout: page
title: Diffusion-Based Low-Light Image Enhancement
description: Conditional diffusion models for low-light image enhancement — anchoring and back-projection strategies for statistical fidelity
img: /assets/img/siu1ab-3486610-large.gif
importance: 2
---

Low-light image enhancement (LLIE) is the task of recovering a well-exposed, faithful image from a capture taken under poor illumination. The difficulty is that darkness is not a simple global dimming: real low-light frames carry heavy sensor noise, color bias, low signal-to-noise ratio, and blur, and any attempt to brighten them naively tends to either over-smooth the result or amplify the very noise it should suppress. Supervised learning compounds the problem — paired low/normal-light captures are scarce, so models are starved of the data they need to learn the true statistics of darkness.

A line of work reframes LLIE as a **conditional generative problem**. Instead of regressing a single deterministic output from the low-light input, a diffusion probabilistic model iteratively refines a prediction, which makes it possible to preserve both the diversity of plausible restorations and their fidelity to the measured input. The work spans two complementary directions: enhancing low-light images into normal-light ones, and — to relieve the data-scarcity bottleneck — synthesizing realistic low-light images from normal-light counterparts for augmentation.

## Conditional diffusion for low-light enhancement

A diffusion model defines a forward process that gradually corrupts data with Gaussian noise and a reverse process that learns to undo it. For image enhancement the model is conditioned on the low-light measurement \(\mathbf{x}_L\), so the reverse step becomes

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_L) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_\theta(\mathbf{x}_t, t, \mathbf{x}_L),\, \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right),
$$

with the forward corruption given by the standard DDPM schedule

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right).
$$

Casting LLIE as conditional generation rather than end-to-end regression matters because it lets the model (i) **preserve diversity** — multiple plausible restorations instead of one averaged output, (ii) **stay faithful to the input** through explicit anchoring/back-projection constraints, and (iii) **improve perceptual quality** by matching statistics instead of minimizing a pixel-wise loss that encourages over-smoothing.

### Anchoring the generative process

When left to its own devices, a diffusion model can wander away from the actual low-light photo it is supposed to fix. The anchoring mechanism prevents this: it nudges the noise trajectory so the model keeps paying attention to the original image. The anchored forward step becomes

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_t^\star,
\qquad
\boldsymbol{\epsilon}_t^\star \sim \mathcal{N}(\mathbf{m}_t,\, \tilde{\beta}_t \mathbf{I}),
$$

where the center \(\mathbf{m}_t\) injects domain knowledge about the geometry of the low-/normal-light data. The corresponding reverse sampler adds a matching center term \(\boldsymbol{\phi}\) to the posterior mean, giving the model more flexibility to explore complex target distributions while remaining anchored.

Supervising such a model with a plain noise-prediction loss is not enough — the predictions live in noise space, far from the image domain where artifacts are visible. A diffusion-feature perceptual loss closes this gap by reconstructing the predicted noisy image from the predicted noise and comparing it, at the image level, against the ground-truth noisy image:

$$
\mathcal{L}_{\text{DFPL}}(\mathbf{x}_0, \boldsymbol{\epsilon}_t, \boldsymbol{\epsilon}_t^\theta)
= \mathcal{L}_{\text{Image}}\!\left(
\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_t,\;
\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_t^\theta
\right).
$$

In short, anchoring keeps the model honest to the input instead of guessing. The figure below shows this directly: on the left, enhancement *without* anchoring loses detail and washes the colors toward a flat white filter; on the right, the anchored method preserves both fine detail and natural color mapping, with clearly better lighting. The enhancement models are trained and evaluated on the standard LLIE benchmarks (LOL, VE-LOL, and LOL-v2).

{% include figure.liquid path="/assets/img/siu12abcd-3486610-large.gif" alt="Before and after the dynamically regulated diffusion anchoring" caption="Before and after of the diffusion anchoring (DRDA). With anchoring (right), the enhanced image keeps fine detail and natural color with better lighting; without it (left), the result loses detail and color and tends toward a flat white-filter look." %}

### Back-projection for low-light synthesis

The inverse direction attacks the data-scarcity problem directly. Because paired captures are hard to obtain, a generative model can instead *synthesize* realistic low-light images from abundant normal-light images, producing training data that captures the authentic noise, blur, and color distortions of real low-light photography. A normal-to-low diffusion model does this by injecting noise perturbations across multiple timesteps, so that a single normal-light input yields diverse, statistically faithful low-light variants.

To model the narrow dynamic range and nuanced noise of real low-light sensors, the synthesis backbone introduces back-projection-aware building blocks — a back-projection attention, a back-projection feed-forward (BP\({}^2\)) module, and BP Transformer blocks — that fold the low-light measurement back into the generative process. The same generative machinery also seeds two large-scale synthetic datasets that downstream LLIE models can train and evaluate on.

## Qualitative results

The figure below illustrates the normal-to-low synthesis direction: how the back-projection generative model turns a single normal-light image into diverse, physically plausible low-light samples, and how the synthesized data compares with simpler augmentation techniques.

{% include figure.liquid path="/assets/img/n2ldiff-bp-3.png" alt="Qualitative results of the back-projection generative model" caption="Qualitative results of the back-projection generative model: synthesized low-light images capture authentic noise, blur, and color distortion, and outperform simpler augmentation baselines." %}

On the enhancement side, the anchoring formulation consistently improves perceptual quality over regression baselines: by constraining the reverse process to the input distribution, it suppresses the color bias and over-smoothing that plague classical methods, while the diffusion-feature perceptual loss reduces residual artifacts. Quantitative gains appear on the standard LLIE benchmarks across PSNR, SSIM, and perceptual (LPIPS) metrics, and the synthesized data from the inverse direction measurably improves the robustness of downstream enhancement models.

## Related Repo

| Repo | Description |
|------|-------------|
| [AnlightenDiff](https://github.com/allanchan339/AnlightenDiff) | Anchoring diffusion probabilistic model for low-light image enhancement (IEEE TIP'24) |
| [N2LDiff-BP](https://github.com/allanchan339/N2LDiff-BP) | Back-projection generative strategy for low/normal-light pairs with enhanced statistical fidelity and diversity (IEEE TCE'24) |
| [N2LDiff](https://github.com/allanchan339/N2LDiff) | Generative strategy for low- and normal-light image pairs with enhanced statistical fidelity (ICCE'24) |
