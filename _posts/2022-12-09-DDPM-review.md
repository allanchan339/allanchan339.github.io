---
layout:       post
title:        "Review: Denoising Diffusion Probabilistic Models (DDPM)"
author:       "Allan"
header-style: text
catalog:      true
tags:
    - AI
    - Review
    - Diffusion
    - Generation
---
# What are Diffusion Models?
Denoising Diffusion Probabilistic Models (DDPM) is introduced in [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239). The maths background is discussed [(Sohl-Dickstein et al., 2015)](https://arxiv.org/abs/1503.03585). The essential idea is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process which is fixed.

# Some maths on DDPM
![figure1](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)
>The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise.

we can let a image smaple(from real data distribution)
$$ \mathbf{x}_0 \sim q(\mathbf{x}) $$

Usually we will also define steps
$$T$$
, where step size is controlled by 
$$ \{\beta_t \in (0, 1)\}_{t=1}^T $$

With steps 
$$T$$
, we can produce a sequence of noisy samples

$$ x_1, x_2, ..., x_T = z$$

However, it is difficult to rebuild the image from $$x_t$$ to $$x_0$$ directly, we need the model to learn the rebuilt process piece by piece. 
i.e.

$$ x_T \rarr x_{T-1} = u(x_T) \rarr x_{T-2} = u(x_{T-1}) \rarr ... \rarr x_1 = u(x_2) \rarr x_0 = u(x_1) $$

## Destruction (Forward Process)
I'd like to use destruction instead of forward process. Basically we want to make a image (with pattern) to a pure gaussian noise by putting more gaussian noise recursively (with a fixed number of steps). 

Each step can be defined as following formulas 
$$
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t , \text{where } \epsilon\sim \boldsymbol{N}(0, \boldsymbol{I}) 
$$

where
$$ \alpha_t + \beta_t = 1 \text{ and } \beta \approx 0$$
and let 
$$ \bar{\alpha}_t = \prod^t_{i=1}\alpha_i$$
, we have 

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t}()
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{(a_t\dots a_1)} \mathbf{x}_0 + \sqrt{1 - (a_t\dots a_1)}\boldsymbol{\epsilon}\\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

where 
$$ \boldsymbol{\epsilon} $$
is a sum of i.i.d gaussian noise

Therefore, we can observe by more steps iterated, the more image will be converted to pure noise. 

### Schedule

The formula 
$$ \bar{\alpha}_t = \prod^t_{i=1}\alpha_i$$
is following a schedule. The schedle is responsilbe to how the way is to destruct an image to pure noise. 

#### Linear Schedule
The DDPM adopt linear schedule as follows:
```python
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
```
![linear schedule](https://i.imgur.com/Y5HARtf.png)

#### Cosine Schedule
![cosine schedule](https://i.imgur.com/dj9bcqr.png)
Later cosine schedule is proponsed. It replace the linear schedule as:
- In linear schedule, the last couple of timesteps already seems like complete noise 
- and might be redundent. 
- Therefore, Information is destroyed too fast.

Cosine schedule can solve the problem mentioned above.

### Merging of Gaussian Noise
Two Gaussian ,e.g. 
$$ \boldsymbol{N}(0,\sigma^2_1 \boldsymbol{I}) \And  \boldsymbol{N}(0,\sigma^2_2 \boldsymbol{I})$$
with different variance can be merged to 
$$ \boldsymbol{N}(0,(\sigma^2_1+\sigma^2_2) \boldsymbol{I}) $$

