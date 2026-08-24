---
layout: page
title: WorldQuant International Quant Championship 2026
description: Building an LLM-as-researcher alpha discovery engine (AlphaMine) and reaching the National Finals of the WorldQuant International Quant Championship 2026
img: /assets/img/IQC.jpg
importance: 1
---

I reached the National Finals of the WorldQuant International Quant Championship 2026, placing 82nd of 22,089 globally and 1st of 170 in Hong Kong out of more than 156,000 participants. I competed as Team ForgeEntropy at The Hong Kong Polytechnic University, and over two months this turned into 796+ git commits, 10,000+ simulation trials, and 173 active alphas spread across different datasets. At its core, I developed AlphaMine, an agent-native research loop that runs LLM as researchers to discover and validate trading alphas on the WorldQuant BRAIN platform.

{% include figure.liquid path="/assets/img/IQC_in_stage.jpeg" alt="Presenting at the IQC National Finals" caption="Presenting at the National Finals of the WorldQuant International Quant Championship." %}

## The Competition

The championship runs on the WorldQuant BRAIN platform, which hosts tens of thousands of data fields and hundreds of operator functions alongside a backtesting simulator. An alpha is a portfolio-construction algorithm that reweights stocks every day using related data, e.g. price, volume, fundamentals, and news. Each alpha is scored by a formula that rewards a high Sharpe ratio and high returns with low turnover, and alphas with strong, uncorrelated returns are aggregated into high-scoring baskets. The field is pared down round by round: more than 156,000 participants entered, and I advanced through Stage 1 (ranked 6th of 810 in Hong Kong and 842nd of 152,452 globally) into Stage 2, where I placed 82nd of 22,089 worldwide and 1st of 170 in Hong Kong — securing a place among the National Finalists (Top 8 across Mainland China and Hong Kong combined).

{% include figure.liquid path="/assets/img/stage1.jpg" alt="WorldQuant IQC Stage 1 result" caption="Stage 1 result: ranked 842nd of 152,452 participants globally." %}

{% include figure.liquid path="/assets/img/stage2.jpg" alt="WorldQuant IQC Stage 2 result" caption="Stage 2 result: ranked 82nd of 22,089 globally and 1st of 170 in Hong Kong — securing the National Finalist entry from more than 156,000 entrants." %}

## Conclusion

Reaching the National Finals of the WorldQuant International Quant Championship came down to building AlphaMine — an agent-native research loop that could research at scale — rather than any single clever expression. The run reshaped how I think about alpha research. Neutralization turned out to be foundational — it absorbs sector-level noise but also removes sector-level signal, so the choice has to match the alpha's economic horizon. Data proved to be king: a single well-chosen datafield, backed by at most a few operators, is worth more than elaborate expression gymnastics, which is exactly what drove the urge to keep collecting data. And the path from alpha data to PnL is deeply non-linear: hand-crafted operators only explore the space we can imagine, whereas machine learning captures far richer mappings — which is why ML is so heavily adopted in quant. On the systems side, the lasting lesson is that an agentic researcher is only as good as its guardrails: model-agnostic design, parallel exploration, and a fully auditable decision log are what turned 10000+ trials into 173 alphas that actually held up.

AlphaMine's behavior in this competition is studied in depth in our paper, *Misaligned Success: Adaptive Skew of an Autonomous LLM Agent under Sparse-Reward Multi-Gate Incentives*, submitted to ACM ICAIF 2026 (under review). It marked a decisive step from retail curiosity toward professional quant research, and the questions it raised about how autonomous agents are evaluated are exactly what I'm taking forward next.
