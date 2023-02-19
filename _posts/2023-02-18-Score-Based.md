---
layout:       post
title:        "Review: Score-Based Generative Modeling through Stochastic Differential Equations"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
publish:      false
tags:
    - AI
    - Code
    - Diffusion
    - Enlightening
---

# Introduction
[Yang et.al ](https://arxiv.org/abs/2011.13456) introduced Score-Based Generative Modeling through Stochastic Differential Equations that can generate high resolution images as diffusion model did. 

In the following paragraph, we discuss 
1. How to approximate **Score** in image generation 
2. How to use the **Score**

Let say we have a dataset with sample $x_1, x_2, \dotsc, x_N$. Each sample $x_i, i \in 1, \dotsc, N$ is drawn from an unknown distribution $$p_{data}(x)$$. We want to find out this distribution (hopeful) and make use of this distribution. 

We should find  the distribution $$p_{data}(x)$$ (a closed form). However, it is impossible as $$p_{data}(x)$$ is impractical to obtain. By dataset, we can only know the sampling results of $$p_{data}(x)$$. 

Can we use a score function to replace finding the probability density function p.d.f or probability mass function p.m.f? Yes

>A p.d.f should fulfill two things:
>1. $$p(x) \ge 0 for all x \in \mathbb{R}$$
>2. $$\int^{\infty}_{-\infty} p(x) dx = 1$$ 

This lead to $$s_\theta(x) = \nabla_x log p_\theta (x)$$. Let us forget score function for seconds.

# Why $\nabla_x log p (x)$ ?
Taking log provide a good property, we use log to escape normalization. Consider a p.d.f $q(x)$ which is unnormalized, we should perform an operation s.t. $$p(x) = \frac{e^{-q(x)}}{Z}$$, $$Z$$ is a normalizing constant to make $$q(x)$$ fulfill condition 2. 

Second, consider $$\frac{dlog(P)}{dx} = \frac{1}{P}\frac{dP}{dx}$$, it is actually calculating the rate of change of function $P$.

Combining two points above, we can have 

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x})=-\nabla_{\mathbf{x}} p(\mathbf{x})+\underbrace{\nabla_{\mathbf{x}} \log Z}_{=0}=-\nabla_{\mathbf{x}} p(\mathbf{x})
$$

This score function can 
1. escape normalization term
2. consider the rate of change of captioned distribution

This property is illustrated as gif as well:
>![](https://img-blog.csdnimg.cn/img_convert/e0d0e5217901b754cbafd2e6c699a30c.gif)
> Parameterizing probability density functions(pdfs). No matter how you change the model family and parameters, it has to be normalized (area under the curve (AUC) must integrate to one).

>![](https://img-blog.csdnimg.cn/img_convert/c7136dd4f83a72da0cefc5726148c51d.gif)
> Parameterizing score functions. No need to worry about normalization.


