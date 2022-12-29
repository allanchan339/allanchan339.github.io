---
layout:       post
title:        "Review: High-Resolution Image Synthesis with Latent Diffusion Models (LDM)"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
tags:
    - AI
    - Review
    - Diffusion
    - Generation
---
# Latent diffusion model (LDM)
Latent diffusion model(LDM) ([Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. 

It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulate/generate semantic concepts with diffusion process on learned latent.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/image-distortion-rate.png)

# Latent representation from VAE
The perceptual compression process relies on an autoencoder model. An encoder $\mathcal{E}$ is used to compress the input image $x \in \mathbb{R}^{H\times W \times 3}$ to a smaller 2D latent vector $z = \mathcal{E}(x) \in \mathbb{R}^{h\times w \times c}$, where downsampling rate $f=H/h = W/w = 2^m, m \in \mathbb{N}$.

Then an decoder $\mathcal{D}$ reconstructs the images from the latent vector $\tilde{x} = \mathcal{D}(z)$.

Here the paper explored two types of regularization. 
- KL-reg: A small KL penalty towards a standard normal distribution over the learned latent, similar to [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
- VQ-reg: Uses a vector quantization layer within the decoder, like [VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2) but the quantization layer is absorbed by the decoder.

# Diffusion in Latent Space
![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png)
The diffusion and denoising processes happen on the latent vector $z$. The denoising model is a time-conditioned U-Net, augmented with the cross-attention mechanism to handle flexible conditioning information for image generation (e.g. class labels, semantic maps, blurred variants of an image).

The design is equivalent to fuse representation of different modality into the model with cross-attention mechanism. 

Each type of conditioning information is paired with a domain-specific encoder $\tau_\theta$ to project the conditioning input $y$ to an intermediate representation that can be mapped into cross-attention component.

# Conditioned Generation
While training generative models on images with conditioning information such as ImageNet dataset, it is common to generate samples conditioned on class labels or a piece of descriptive text.


## Classifier Guided Diffusion 
To explicit incorporate class information into the diffusion process, Dhariwal & Nichol (2021) trained a classifier 
$$p_\phi(y\vert x_t,t)$$
on noisy image $x_t$ and use gradients 
$$\nabla_x log p_\phi (y\vert x_t)$$
to guide the diffusion sampling process toward the conditioning information $y$
e.g. a target class label) by altering the noise prediction.

Recall

$$
\mu(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t) \Big)
$$

Authors indicate that the following equation can be used to replace $\epsilon_\theta(x_t,t)$ to provide conditioning.

$$
\tilde{\epsilon}(x_t) := \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} log p_\phi (y\vert x_t)
$$

Lastly, one extra hyper-parameter **gradient scale** is introduced to form the samplng equation.
![圖 2](https://s2.loli.net/2022/12/28/zO6QtxBvU1LJSbi.png)  

Result is improved due to guided classifer 
![圖 1](https://s2.loli.net/2022/12/28/2TFX9ci7qL3lvon.png)  

## Classifier-Free Guidence
Without an independent classifier $p_\phi$, it is still possible to run conditional diffusion steps by incorporating the scores from a conditional and an unconditional diffusion model ([Ho & Salimans, 2021](https://openreview.net/forum?id=qw8AKxfYbI)). Let unconditional denoising diffusion model 
$$p_\theta(\mathbf{x})$$
parameterized through a score estimator 
$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$
and the conditional model 
$$p_\theta (x\vert y)$$
parameterized through 
$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y)$$
.  These two models can be learned via a single neural network. Precisely, a conditional diffusion model 
$$p_\theta(\mathbf{x} \vert y)$$
 is trained on paired data
$$(x,y)$$
, where the conditioning information $y$ gets discarded periodically at random such that the model knows how to generate images unconditionally as well, i.e.

$$ \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y=\varnothing) $$
.


