---
layout:       post
title:        "Introduction of Probability : Basic"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
comments: true
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
   2. A space that contains all possible outcomes / samples
   3. Sample may be numerical or non-numerical

   - Example
     - E: Tossing a coin
       - : $$\Omega = \{H, T\}$$
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
     - (1) 
     $$\begin{aligned}
     \Omega &= \{H, T \} \\
      F &= \{ \emptyset, \{H\}, \{T\}, \Omega\} \\
      F_0 &= \{\emptyset, \Omega\}, \text{which is also called the trivial } \sigma \text{ field}
     \end{aligned}
     $$
     - (2) $$\Omega = \{a, b, c\}$$
       - : $$F_0 = \{\emptyset, \Omega\}$$
       - : $$F= \{\emptyset, \{a\}, \{b,c\}, \Omega \}$$
       - : $$F = \{ \emptyset, \{a\}, \{b\} \{c\}, \{a,b\}, \{b,c\}, \{c,a\}, \Omega \}$$
     - (3) $$\Omega = \mathbb{R} = \{x \vert -\infty < x < \infty \}$$
       - : $$F_0 = \{\emptyset, \Omega\}$$
       - : $$F = \{ \emptyset, \{a\}, (a,b), (a,b], [a,b), [a,b], (-\infty, a),[a, \infty), (a, \infty), \Omega\}$$, which is also called Borel $$\sigma$$ field on the real line.

4. Probability
   1. Let $$\Omega$$ be a sample space
   2. Let $$F$$ be a $$\sigma$$ field over $$\Omega$$
   3. A real value set function $$P$$ defined on $$F$$ is called a probabilty if satisfying
      1. $$P(A) \geq 0$$ for all $$A \in F$$
      2. $$P(\Omega) = 1$$
      3. If $$A_1, A_2, \cdots$$ are mutually disjoint events inf $$F$$, then $$P(U_{i=1} A_i) = \sum_i P(A_i)$$ 
   4. The triplet $$(\Omega, F, P)$$ is called a probability space
   5. Elements of $$\Omega$$ is called sample; Elements of $$\F$$ is called events