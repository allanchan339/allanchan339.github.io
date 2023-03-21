---
layout:       post
title:        "Introduction of Probability : Basic"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
publish:      false
tags:
    - Maths
    - Foundation
    - Review
---

# Basic of Probability 

## Important Definition 

1. Random Experiments
   1. An experiment is said to be random if its results cannot be determined beforehand. 

2. Sample Space
   1. The set $$\Omega$$ of all possible results of a random experiment is called sample space
   2. A space that contains all possible outcome / sample
   3. Sample may be numerical or non-numerical

   - Example
     - E: Tossing a coin
       -  Omega = \{H, T\}$$
     - E: Rolling a die
       - : $$\Omega = \{1,2,3,4,5,6\}$$
     - E: # of calls ongoing in telephone exchange
       - : $$\Omega = \{ 0, 1, 2, \cdots, \infty \}$$
     - E: Temperature of a particular city
       - : $$\Omega = \{x \vert 0 < x < 50^o C \}$$

3. The $$\sigma$$ field
   1. A collection $$F$$ of subsets of $$\Omega$$ is called a $$\sigma$$ field over $$\Omega$$
   - Condition:
     - (1) $$\Omega \in F$$
     - (2) If $$A \in F$$, then $$A^c \in F$$, where $$c$$ means complement
   - (3) If $$A_1, A_2 , cdots, \in F$$, then $$U^\infty_{i=1} A_i \in F$$
   - Example
     - (1) $$\Omega = \{H, T \}$$
       - : $$F = \{ \emptyset, \{H\}, \{T\}, \Omega\}$$
       - : $$F_0 = \{\emptyset, \Omega\}$$, which is called as trival $$\sigma$$ field
     - (2) $$\Omega = \{a, b, c\}$$
       - : $$F_0 = \{\emptyset, \Omega\}$$
       - : $$F= \{\emptyset, \{a\}, \{b,c\}, \Omega \}$$
       - : $$F = \{ \emptyset, \{a\}, \{b\} \{c\}, \{a,b\}, \{b,c\}, \{c,a\}, \Omega \}$$
     - (3) $$\Omega = \mathbb{R} = \{x \vert -\infty < x < \infty \}$$
       - : $$F_0 = \{\emptyset, \Omega\}$$
       - : $$\vdots$$