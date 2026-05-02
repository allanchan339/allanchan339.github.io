---
layout: post
title: "Stable tool calling for Qwen 3.5 27B/35B on vLLM: template, parser, and mixed-GPU fixes"
date: 2026-04-13 00:00:00 +0800
description: "Debugging notes on Jinja chat templates, qwen3_xml vs qwen3_coder parsers, mixed-GPU FP8 drift, and SFT-distilled checkpoints when running Qwen 3.5 27B/35B-class models for long agentic sessions on vLLM."
tags: [vllm, qwen, tool-calling, llm, inference, gpu]
categories: [bug-fixes]
---

Public write-ups on Qwen 3.5 often emphasize reasoning quality and slow time-to-first-token (TTFT). The reasoning capability on **Qwen 3.5-27B** is genuinely strong, and yes, TTFT is slow—but for **agentic** workloads **tool calling** is frequently what actually breaks: malformed XML, mid-stream stops, and long-context format drift are not always visible in short demos. This note comes from roughly **a month** of running that model on a **mixed-GPU** workstation (**RTX 4090 + 3090**), plus **many hours** debugging failed runs and reading **vLLM** source. The same patterns apply to **27B/35B-class** checkpoints (including **A3B**-style variants where instruction-following is comparable). The resulting configuration has stayed stable in production (**weeks** of use after the initial fix pass). Official model cards describe the happy path; they understate **edge cases that smaller models** hit more often than **122B+** checkpoints.

# 1. Chat template: `qwen3.5_official.jinja` and smaller models

## Symptoms

The run started from the official `qwen3.5_official.jinja` template. For the first handful of tool turns, output looked fine. Then failures clustered:

- Tool calls appeared **mid-thought**, including closing **`</think>`** without having opened a matching `<think>` tag.
- **Premature stops** in the middle of XML tool calls—for example, the model produced a line like “Let me do that for you:” and then **stopped** without finishing the tool payload.
- **Historical thinking blocks leaked into context**, so later turns saw polluted reasoning boundaries and inconsistent tool formatting.

At first the usual suspects were operator error, a possible **vLLM** bug, or **heterogeneous GPUs**. Instrumentation and template experiments showed the **chat template** was the actual root cause.

## Cause

The official template contains **edge cases that 122B+ models tend to absorb but 27B/35B models do not**. Smaller checkpoints have **less robust instruction following**; the same ambiguity around where “thinking” ends and tool XML begins produces **silent parser-level failures** rather than self-correction.

## Fix

The working approach was a custom **M2.5-style interleaved thinking** template (`qwen3.5-enhanced.jinja`) that:

- Closes **`</thinking>`** (the structured thinking segment) **before** tool calls, not after—so tool XML is not interleaved in a way that confuses the runtime.
- **Hides historical reasoning** from the context the model sees on subsequent turns while keeping **current** reasoning visible where needed.
- Uses XML-shaped tool output that **does not accidentally trigger** `<stop>`-style termination patterns.
- **Handles the edge cases** that smaller models struggle with when the stock template leaves boundaries implicit.

**vLLM does not auto-detect** the right chat template for this stack. You must pass the Jinja file explicitly; otherwise the default template remains in force and instability tends to persist **regardless of other optimizations**:

```bash
--chat-template qwen3.5-enhanced.jinja
```

# 2. Tool-call parser: `qwen3_coder` vs `qwen3_xml`

## Official guidance

The [Qwen3.5-27B-FP8 Hugging Face page](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) recommends:

```bash
--tool-call-parser qwen3_coder
```

## What went wrong in practice

For **complex tool calls** and **long-context agentic** runs (on the order of **50K+ tokens** in a trace), `qwen3_coder` was a primary source of breakage. A lot of wall-clock time went into proving that the parser—not only the model—was responsible.

From reading **vLLM’s implementation**, the distinction is structural:

| Parser        | How it works                         | Special characters (`<`, `>`, `&`) | Nested JSON while streaming | Malformed XML |
| ------------- | ------------------------------------ | ----------------------------------- | --------------------------- | ------------- |
| `qwen3_coder` | Regex string extraction              | Breaks pattern matching             | Often corrupts mid-stream   | Fails hard    |
| `qwen3_xml`   | C-based parser (`xml.parsers.expat`) | Auto-sanitizes                    | Deferred / safer parsing    | Often auto-heals |

**Concrete example:** a tool argument that contains code such as `if (a < b)` breaks `qwen3_coder` because `<` and `>` interfere with regex-based extraction. `qwen3_xml` treats the stream as XML-shaped text under a real parser and **does not depend on that fragile string match**.

## Fix

```bash
--tool-call-parser qwen3_xml
```

This **contradicts the one-line official recommendation** on the model card. The preference for `qwen3_xml` here is grounded in **vLLM source inspection** (not only empirical trial-and-error): the **C-based XML path** is fundamentally more robust than regex extraction for messy, nested, or streaming tool payloads.

# 3. Mixed-GPU precision drift (4090 + 3090)

## Problem

Tensor parallelism splits matrix multiplications across devices. In this setup:

- **RTX 4090 (SM89)** exposes **native FP8** **W8A8** tensor-core paths.
- **RTX 3090 (SM80)** has **no native FP8** and falls back to **W8A16**.

So **different ranks use different precision**: **W8A8** on one GPU and **W8A16** on the other → **mismatched partial products** → **error accumulation** over depth and sequence length.

## Symptoms

Beyond roughly **30–40K tokens**, conversations **drifted**: tool calls grew inconsistent and reasoning quality degraded, consistent with numerical divergence rather than a single bad sampling draw.

## Fix

```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1
```

This forces the 4090 onto **W8A16** (via the Marlin path) so it **matches the 3090** instead of using native **W8A8** alone. Both ranks then share the same effective precision, which removed the long-run drift in this configuration.

**NCCL tuning** for stability on this **mixed consumer topology** (helpful in practice alongside the precision alignment):

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
```

# 4. Checkpoint choice: SFT-distilled variants (e.g. Qwopus3.5) vs official weights

## The failure mode

Checkpoints such as **`QuantTrio/Qwopus3.5-27B-v3-AWQ`** are **SFT-distilled from Claude 4.6 Opus**. They can look excellent initially:

- For the **first ~65K tokens**, tool calling stayed stable.
- **After ~65K+ tokens**, output began **mixing XML tool format with JSON-style** tool messages.

This branch of debugging **cost the most calendar time** before the hypothesis “wrong checkpoint for the protocol” was confirmed.

## Cause

**SFT** shifted the **surface** tool format toward a **Hermes-style JSON** tool protocol to align with **Claude-like** training targets, but it does **not fully realign** the underlying token distribution with the base **Qwen XML** tool priors. In long context, the model **drifts between** the original **Qwen `qwen3_xml` shape** and the SFT’d **JSON** shape—something post-processing cannot reliably paper over.

## What to run instead

**If you have about 48 GB VRAM** (best quality in this comparison):

```text
Qwen/Qwen3.5-27B-FP8
```

- **Near-lossless** accuracy relative to denser formats.
- **Full ~219K context** support as advertised for the stack used here.
- **Stable tool calling** when paired with the **custom template** above.

**If VRAM is below ~48 GB** (accept some accuracy loss):

```text
Intel/Qwen3.5-27B-int4-AutoRound
```

- Saves on the order of **~4 GB VRAM** versus FP8 in this class of setup.
- Remains **stable with the same custom template** in testing.
- **Higher perplexity than FP8**; **INT4 is not lossless**.

**FP8 quantization is near-lossless** in practice for this model line: avoid dropping to INT4 **unless** VRAM truly forces it.

# Reference `vllm serve` configuration

After consolidation in an **independent repo** and roughly **three days** of **production-style** use, the following **environment** and **serve** command matched the stable behavior described above:

```bash
# Environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export VLLM_TEST_FORCE_FP8_MARLIN=1

# vLLM serve command
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --served-model-name Qwen3.5-27B \
  --chat-template qwen3.5-enhanced.jinja \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 219520 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only
```

# Validation

One **continuous agentic session** lasted about **1h 9m** on this configuration:

- **138.2K tokens** generated.
- **Stable tool calling** throughout—**no XML/JSON format drift** tied to parser or template collapse.
- **M2.5-style interleaved thinking** remained coherent across the run.
- The model **autonomously** implemented a **production-oriented knowledge-graph platform** (**FastAPI + React**); **18 minutes** of that session were **uninterrupted** end-to-end work **without tool-calling failures**—the sort of reliability that matters for real agents rather than toy demos.

The stack has also remained **stable for weeks** after that validation period in the author’s deployment. As always, numbers are **tied to one hardware topology** and one workload mix; they are included as **evidence**, not a universal benchmark.

# Summary

1. **Jinja template**: For **27B/35B-class** Qwen 3.5, the **custom interleaved-thinking template** is **critical**; the **official template** leaves **edge cases** that **smaller models** hit routinely.
2. **Parser**: Do not treat **`qwen3_coder`** as mandatory for agentic work; **`qwen3_xml`**’s **Expat-based** path is **more robust** than **regex** for long, messy tool traces—and that conclusion is supported by **reading vLLM**, not only by trial runs.
3. **Mixed GPU**: When **FP8 paths differ by generation**, **`VLLM_TEST_FORCE_FP8_MARLIN=1`** (or an equivalent precision-alignment strategy) is **effectively required** to stop **long-context drift**.
4. **Weights**: **SFT-distilled “Claude-shaped”** forks (e.g. **Qwopus3.5**) can **mix formats after ~65K tokens**; for **long** tool-heavy jobs, prefer **official Qwen FP8** (or the INT4 variant if VRAM demands it).
5. **Quantization**: **FP8 is near-lossless** here; downgrade formats **only** when VRAM leaves no alternative.

# Resources

- **Working setup (templates, env, notes):** [GitHub — vLLM Qwen 3.5 27B config](https://github.com/allanchan339/vLLM-Qwen3.5-27B) — includes **`qwen3.5-enhanced.jinja`**.
- **Concrete long-session example:** [qwen_own_project](https://github.com/allanchan339/qwen_own_project).
- **Earlier discussion:** [Reddit — tool calling fixes thread](https://www.reddit.com/r/LocalLLaMA/comments/1sdhvc5/qwen_35_tool_calling_fixes_for_agentic_use_whats/).

If you run **Qwen 3.5 27B/35B-class** models for **agents** and see **silent** tool failures—**truncation**, **wrong boundaries**, or **thinking leakage**—inspect the **Jinja chat template** first. In this experience it was **almost always** the dominant factor: the **stock template** does not cover the **failure modes smaller checkpoints** actually hit.
