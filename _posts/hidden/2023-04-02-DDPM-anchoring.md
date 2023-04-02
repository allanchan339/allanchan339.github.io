---
layout:       post
title:        "Dynamic Regulated Diffusion
Anchoring on DDPM"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
comments: true
tags:
    - AI
    - Diffusion
    - Generation
    - Maths
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---

# Introduction

As discussed before, DDPM has 3 important component, including the forward process, the reverse process and the diffusion model (noise prediction model). In this post, we will focus on Dynamic Regulated Diffusion
Anchoring on DDPM forward and reverse process.

# Forward Process
## DDPM Forward Process on Dynamic Regulated Diffusion Anchoring
$$
\begin{aligned}
    x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t; \quad \epsilon_t \sim \mathcal{N}(\mu_t, I) \\
    \text{Solving gives }x_t &= \sqrt{\bar{\alpha}_t} x_0 + \sum^t_{j=1} \sqrt{\frac{\bar{\alpha}_t}{\bar{\alpha}_j}}\sqrt{1-\alpha_j} \epsilon_j; \quad \epsilon_j \sim \mathcal{N}(\mu_j, I) \\
\end{aligned}
$$
Let $$\mu_t$$ take the form of $$\mu_t \stackrel{\Delta}{=} \frac{1-\sqrt{\alpha_t}}{\sqrt{1-\alpha_t}} g(x_l)$$

$$
\begin{aligned}
\mathbb{E}\left[x_t \mid x_0\right] 
& =\sqrt{\bar{\alpha}_t} x_0+\sum_{j=1}^t \sqrt{\frac{\bar{\alpha}_t}{\bar{\alpha}_j}} \sqrt{1-\alpha_j} \cdot \frac{1-\sqrt{\alpha_j}}{\sqrt{1-\alpha_j}} g\left(x_l\right) \\
& =\sqrt{\bar{\alpha}_t} x_0+\sum_{j=1}^t\left(\sqrt{\frac{\alpha_t}{\bar{\alpha}_j}}-\sqrt{\frac{\alpha_t}{\alpha_{j-1}}}\right) g\left(x_l\right) \\
& =\sqrt{\bar{\alpha}_t} x_0+\left(1-\sqrt{\bar{\alpha}_t}\right) g\left(x_l\right) \rightarrow g\left(x_l\right) \text { as } t \rightarrow+\infty \\
& \operatorname{Var}\left(x_t \mid x_0\right)=\sum_{j=1}^t\left(\frac{\bar{\alpha}_t}{\bar{\alpha}_j}-\frac{\bar{\alpha}_t}{\bar{\alpha}_{j-1}}\right) I=\left(1-\bar{\alpha}_t\right) I \\
& \therefore x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \bar{\epsilon}_t, \bar{\epsilon}_t \sim N\left(\bar{\mu}_t, I\right); \text{where } \bar{\mu}_t=\frac{1-\sqrt{\bar{\alpha}}_t}{\sqrt{1-\bar{\alpha}_t}} g\left(x_l\right)

\end{aligned}$$

## DDPM Reverse Process on Dynamic Regulated Diffusion Anchoring

$$
\begin{aligned}
& p\left(x_{t-1} \mid x_t, x_0\right) \propto \frac{p\left(x_t \mid x_{t-1}, x_0\right) p\left(x_{t-1} \mid x_0\right)}{p\left(x_t \mid x_0\right)}
\end{aligned}
$$
Suppose $$x_{t-1} \mid x_t, x_0 \sim N\left(\tilde{\mu}_t\left(x_t, x_0\right), \tilde{\beta}_t\left(x_t, x_0\right) I\right)$$
$$
\begin{split}
\frac{1}{2\tilde{\beta}_t}\Vert x_{t-1} - \sqrt{\alpha_t}x_{t-1} - \tilde{\mu}_t(x_t, x_0) \Vert^2 &=  \frac{1}{2\tilde{\beta}_t}\Vert x_{t-1} \Vert^2 - \langle \frac{\tilde{\mu}_t(x_t, x_0)}{\tilde{\beta}_t}, x_{t-1} \rangle\\
% &=\frac{1}{2\left(1-\bar{\alpha}_l\right)}\left\|x_t-\sqrt{\bar{\alpha}_t} x_0-\left(1-\sqrt{\bar{\alpha}_l}\right) g\left(x_l\right)\right\|^2+\text { const. } \\
\end{split}
$$
