---
layout: post
title: "Findings: Karpathy-style autoresearch on a crypto backtester (local LLM)"
date: 2026-04-24 00:00:00 +0800
description: "Local Qwen 3.5 autoresearch on my crypto DB + Nautilus-style backtester (~2h, 30+ iter, $0 API): tool-calling blocker, run observations, human-in-the-loop steering, GA contrast, diversity, gates."
tags: [llm, quant, backtesting, vllm, qwen, automation, crypto, local-inference]
categories: [research]
---

The thread I started from was whether **Karpathy-style autoresearch**—an LLM in a tight **read / act / evaluate** loop—could apply to **quant mining** on my own stack: **local** inference, **crypto** data, a **custom backtester**, no paid API for the search itself. I connected a **Qwen 3.5** agent to that backtester and DB, let it run for **~2 hours** and **30+** strategy cycles with **no per-step prompting**, and watched it **self-learn** the repo, burn through bad ideas, and land **one** configuration that cleared my gates (**$0** inference bill for that grind). It was **not** a fully closed loop: when it stalled in weak local optima, I used **human-in-the-loop** nudges (short instructions, ideas from notes or papers) and it folded those into the next iterations without restarting the harness.

The point of this note is the **trail**: what blocked, what the loop did, and what I take away for the next run. Nothing here claims the strategy generalizes out-of-sample.

**Recording** ([YouTube](https://youtu.be/aEvj0SiU6WI))—captures how the autonomous loop behaved on my stack:

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:1rem 0;">
  <iframe
    style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
    src="https://www.youtube.com/embed/aEvj0SiU6WI"
    title="Karpathy-style autoresearch — local LLM on crypto backtester"
    loading="lazy"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    referrerpolicy="strict-origin-when-cross-origin"
    allowfullscreen></iframe>
</div>

## What I was testing

**Hypothesis:** The same **generate → backtest → gate → decide** skeleton Karpathy describes for research automation is enough here for the model to learn my strategy API from the codebase, iterate without per-step prompting, and eventually hit **strict** metrics (Sharpe, max drawdown, profit factor, minimum trades).

**What I wanted out of the experiment:** A **working skeleton**—stable tools, objective discard, bounded wall clock—not a proof that the first passing parameter set is economic edge. I also wanted to see whether **human-in-the-loop** could stay **lightweight**: occasional steering instead of babysitting every iteration.

## Setup (fixed for the run)

| Component      | Detail                                                   |
| -------------- | -------------------------------------------------------- |
| **Model**      | Qwen3.5-27B via vLLM                                     |
| **GPUs**       | Mixed setup (RTX 4090 + 3090)                            |
| **Inference**  | vLLM with FP8, custom Jinja template, `qwen3_xml` parser |
| **Runtime**    | ~2 hours; autonomous iterations, **human-in-the-loop** when stuck |
| **Iterations** | ~30+ strategy cycles                                     |
| **API cost**   | $0                                                       |

- **Data:** Crypto spot, 1m bars; TimescaleDB-HA for continuous aggregation.
- **Backtester:** Custom DB + engine (Nautilus-based).
- **Harness:** Ralph loop + Claude Code as execution harness.

## Finding: tool calling was the real prerequisite

Before any autoresearch, Qwen 3.5 in agent mode was **unreliable** for me: premature stops, mid-thought tool calls, format drift—unacceptable for multi-hour loops.

**What eventually worked:** A **custom M2.5-style Jinja template**, the **`qwen3_xml` parser**, and **precision alignment** across the mixed GPU pair. Calendar time on this was on the order of **weeks**.

Reference I kept for myself while debugging: [Qwen 3.5 27B/35B tool calling on vLLM](https://www.reddit.com/r/Vllm/comments/1skks8n/qwen_35_27b35ba3b_tool_calling_issues_why_it/). Without that stack, I would not rerun the experiment and expect the same stability.

## Procedure (what I actually ran)

Each iteration:

1. **Generate** strategy code + YAML config.
2. **Backtest** immediately.
3. **Score** against the gates.
4. **Decide:** abandon, fix error, or change approach.

**Observation:** The useful part is not the four bullets in isolation—it is that the same interface and discard semantics repeat every cycle so the run has a **trajectory** instead of isolated guesses.

## Observations from the long run

**Bootstrap.** The agent searched the repo, read existing strategies, matched the template, and produced a `run_backtest()`-compatible scaffold before leaning on novel ideas.

**Regime coverage.** Under my high-level brief it systematically tried **momentum** (EMA crosses, breakouts, ATR stops) and **mean reversion** (RSI, Bollinger, range fades). Most candidates failed fast on the gates; WandB showed **DISCARD** with clear numeric causes (Sharpe, drawdown, trade count).

**Adaptation.** When returns were weak but non-terrible, it **tuned parameters** (EMA lengths, sizing, vol filters). It also **moved the backtest start date** on its own to get more history.

**Finding on the date move:** That behavior is a **leak / overfitting lever** if left unrestricted. My takeaway is to **freeze** evaluation windows and walk-forward rules in the harness so the model cannot silently widen the training corridor.

**Grind.** Over ~2 hours it cycled EMA stacks (10/20, 5/15, 50/200), RSI + confirmation, MACD, 4h Bollinger reversion, vol-adjusted trend with jump detection, grid-style scalps, fixed R/R templates, long/short EMA divergence—**dozens** of variants, all discarded on the numbers until late in the run.

**Eligibility.** After **~30+** iterations, **one** configuration cleared all gates; I stopped there for this MVP.

## Human-in-the-loop (what actually happened)

The loop was **mostly hands-off**: I did not craft each strategy or click through each backtest. The agent ran the **generate → backtest → gate → decide** cycle on its own until the trajectory looked **stuck**—same family of tweaks, marginal metrics, no path to the gates.

When I intervened, the input was **small and textual**: one-line briefs like *volatility-adjusted position sizing* or *dual trend filters*, or a concept lifted from a paper or my own notes. The agent **ingested** that and **reprioritized** the next hypotheses without me editing the harness or resetting state.

**Finding:** For this run, **human-in-the-loop** was the escape hatch from **local optima**, not a substitute for the loop. The economics still felt like automation: hours of machine time between nudges, and **$0** marginal API cost for the search itself.

## Comparative note: GA (why I contrast it)

I keep **genetic algorithms** in mind as a baseline: genome encode, **mutation / crossover**, batch backtest, selection—**no** explicit operator that reads a failure and chooses among *abandon*, *patch*, or *new hypothesis*; steering is emergent from selection.

**Finding for my setting (multi-constraint gates, not one fitness scalar):**

- **Cost:** Generations imply **batches** of backtests; many individuals are obviously bad but consume slots until selection removes them.
- **Completion:** A run often ends without any individual that **simultaneously** satisfies Sharpe, drawdown, trade count, etc.; “best so far” still fails a gate.

**Contrast I recorded:** The LLM loop ends each step with a **decision conditioned on the backtest and repo context**. That does not remove overfitting, but in this run it gave a **bounded** path to one gate-passing config without a large per-generation fan-out.

## Findings I am carrying forward

1. **Integration over single-strategy hype.** The durable artifact is **tool-stable loop + backtester + gates + harness**, not the first passing parameter set.
2. **Autonomous date-window edits are a policy bug** unless intentionally allowed; they read as implicit curve fitting in my book.
3. **Discard speed matters.** The model moved on immediately on bad metrics; that matched what I want from mining hygiene.
4. **Reuse worked.** It recombined ideas already present in the repo and notes (e.g. vol-adjusted trend with jump suppression).
5. **Economics.** ~2h local inference, **$0** API line item—material for how long I am willing to let a search run.
6. **Human-in-the-loop scales the search.** Mid-run nudges broke local optima; the agent absorbed paper- or note-level hints without harness churn. I am keeping that pattern: **autonomous bulk + sparse human steering**, not full manual mining.
7. **Diversity risk.** With the **same** harness, prompts, and data, an LLM driver can still **converge** to the same or nearly the same strategy across runs—inductive bias and “default” repairs narrow the trajectory. I am treating **perturbation** (sampling, seeds, varied briefs / inits, parallel nudges, small prompt jitter) as **part of the method**, not an optional polish.

## Limitations and planned follow-ups

- **Eligibility is not alpha:** Clearing the backtest gates is not, by itself, a claim of edge out-of-sample.
- **Next checks:** Lock windows, walk-forward or holdout, and use “passes gates once” as a **regression signal for the harness**, not as validation of edge.
