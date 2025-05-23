---
layout:       post
title:        "Dynamic Regulated Diffusion
Anchoring on DDPM"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
comments: true
hidden: true
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
\frac{1}{2\tilde{\beta}_t}\Vert x_{t-1} - \sqrt{\alpha_t}x_{t-1} - \tilde{\mu}_t(x_t, x_0) \Vert^2 &=  \frac{1}{2\tilde{\beta}_t}\Vert x_{t-1} \Vert^2 - \langle \frac{\tilde{\mu}_t(x_t, x_0)}{\tilde{\beta}_t}, x_{t-1} \rangle + \cdots\\
& = \frac{1}{2\left(1-\alpha_t\right)}\left\|x_t-\sqrt{\alpha_t} x_{t-1}-\left(1-\sqrt{\alpha_t}\right) g\left(x_l\right)\right\|^2 \\
& +\frac{1}{2\left(1-\bar{\alpha}_{t-1}\right)}\left\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}} x_0-\left(1-\sqrt{\bar{\alpha}_{t-1}}\right) g\left(x_l\right)\right\|^2 \\
& -\frac{1}{2\left(1-\bar{\alpha}_t\right)}\left\|x_t-\sqrt{\bar{\alpha}_t} x_0-\left(1-\sqrt{\bar{\alpha}_t}\right) g\left(x_l\right)\right\|^2+\text { const. } \\
= & \frac{1}{2\left(1-\alpha_t\right)}\left(\alpha_t\left\|x_{t-1}\right\|^2-2\left\langle x_t-\left(1-\sqrt{\alpha_t}\right) g\left(x_l\right), \sqrt{\alpha_t} x_{t-1}\right\rangle+\cdots\right) \\
& +\frac{1}{2\left(1-\bar{\alpha}_{t-1}\right)}\left(\left\|x_{t-1}\right\|^2-2\left\langle\sqrt{\bar{\alpha}_{t-1}} x_0+\left(1-\sqrt{\bar{\alpha}_{t-1}}\right) g\left(x_l\right), x_{t-1}\right\rangle+\cdots\right) \\
= & \frac{\alpha_t\left(1-\bar{\alpha}_{t-1}\right)+\left(1-\alpha_t\right)}{2\left(1-\alpha_t\right)\left(1-\bar{\alpha}_{t-1}\right)}\left\|x_{t-1}\right\|^2  \\
&-\left\langle\frac{\sqrt{\alpha_t} x_t-\sqrt{\alpha_t}\left(1-\sqrt{\alpha_t}\right) g\left(x_t\right)}{1-\alpha_t}+\frac{\sqrt{\bar{\alpha}_{t-1}} x_0+\left(1-\sqrt{\bar{\alpha}_{t-1}}\right) g\left(x_l\right)}{1-\bar{\alpha}_{t-1}}, x_{t-1}\right\rangle+\cdots
\end{split}
$$

Therefore, for
$$
\widetilde{\beta}_t\left(x_t, x_0\right)=\frac{\beta_t\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t}
$$

We have 
$$
\begin{aligned}
& \tilde{\mu}_t\left(x_t, x_0\right) \\
& =\left(\frac{\sqrt{\alpha_t} x_t-\sqrt{\alpha_t}\left(1-\sqrt{\alpha_t}\right) g\left(x_l\right)}{1-\alpha_t}+\frac{\sqrt{\bar{\alpha}_{t-1}} x_0+\left(1-\sqrt{\bar{\alpha}_{t-1}}\right) g\left(x_l\right)}{1-\bar{\alpha}_{t-1}}\right) \tilde{\beta}_t\left(x_t, x_0\right) \\
& =\frac{\left(\sqrt{\alpha_t} x_t-\sqrt{\alpha_t}\left(1-\sqrt{\alpha_t}\right) g\left(x_l\right)\right)\left(1-\bar{\alpha}_{t-1}\right)+\left(\sqrt{\bar{\alpha}_{t-1}} x_0+\left(1-\sqrt{\bar{\alpha}_{t-1}}\right) g\left(x_l\right)\right)\left(1-\alpha_t\right)}{1-\bar{\alpha}_t} \\
& =\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0+\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} x_t \\
& \color{red}{+\frac{\left(1-\sqrt{\bar{\alpha}_{t-1}}\right)\left(1-\alpha_t\right)-\sqrt{\alpha_t}\left(1-\sqrt{\alpha_t}\right)\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} g\left(x_l\right)} \\
&
\end{aligned}
$$

## Summary
### For training
$$
\begin{aligned}
&\text{repeat:} \\
&&x_0 \sim g(x_0); t \sim U(0,T) \\
&&\epsilon_t \sim  N(\frac{1-\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}}g(x_l), I) \\
&& \text{gradient descent step on } \nabla_{\theta} \Vert \epsilon_t - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t), t) \Vert^2 \\
& \text{until convergence}
\end{aligned}
$$

### For inference
$$
\begin{aligned}
& x^{pred}_0 = \frac{1}{\sqrt{\alpha_t}}\left(x_t-\sqrt{1-\bar{\alpha}_t} \varepsilon_\theta\left(x_t, t\right)\right) \\
& \tilde{u}_\theta(x_t, t)=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t}(\frac{1}{\sqrt{\alpha_t}}\left(x_t-\sqrt{1-\bar{\alpha}_t} \varepsilon_\theta\left(x_t, t\right)\right)) +\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} x_t \\
& \color{red}{+\frac{\left(1-\sqrt{\bar{\alpha}_{t-1}}\right)\left(1-\alpha_t\right)-\sqrt{\alpha_t}\left(1-\sqrt{\alpha_t}\right)\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} g\left(x_l\right)} \\
\\
& x_{t-1} = \tilde{u}_\theta(x_t, t) + z_t; z_t \sim N(0, I)
\end{aligned}
$$