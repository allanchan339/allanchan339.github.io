---
layout: post
title: "Qwen 3.6-27B-FP8 on vLLM: enhanced.jinja, qwen3_coder, and fixing NCCL after Studio Driver 595.79"
date: 2026-04-29 00:00:00 +0800
description: "Same qwen3.5-enhanced.jinja and mixed-GPU stack as earlier Qwen 3.5 notes; switching to qwen3_coder for 3.6, mandatory preserve_thinking=false, and NCCL overrides that stopped deadlocks on NVIDIA Studio 595.79—plus a 180k-token agentic run."
tags: [vllm, qwen, tool-calling, llm, agent, inference]
categories: [bug-fixes]
---

This post continues from [my earlier notes on Qwen 3.5 tool-calling](https://www.reddit.com/r/Vllm/comments/1skks8n/qwen_35_27b35ba3b_tool_calling_issues_why_it/) and [the Qwen 3.6-35B-A3B follow-up](https://www.reddit.com/r/LocalLLM/comments/1sqpsut/qwen_3635ba3b_reddit_asked_so_i_tested_if_the_35/). I reused the same `enhanced.jinja` stack and ran **Qwen 3.6-27B-FP8** in a long unsupervised agentic session. **NVIDIA Studio Driver 595.79** introduced **NCCL deadlocks** until I added the environment and flag overrides in the sections below. After that, the run reached about **180 000 tokens** with no malformed tool calls. The resulting project is [on GitHub](https://github.com/allanchan339/qwen36_27B_own_project).

The `qwen3.5-enhanced.jinja` template **requires** `preserve_thinking=false` under Qwen 3.6 (a new surface-area flag). With `preserve_thinking=true`, that template breaks and tool calls fail. The rest of this note assumes the flag is set to `false`.

## Background: what already worked on Qwen 3.5

Earlier work on **Qwen 3.5-27B** and **35B-A3B** used an **RTX 4090** and **RTX 3090** together. The configuration that made long agentic runs reliable included:

- **`qwen3.5-enhanced.jinja`** — interleaved-thinking template that treats an **unclosed `<thinking>` block as plain content**, not reasoning-only text, so the harness still sees tool output when the model omits the closing tag (“CoT leakage”). **`preserve_thinking=false`** is required (and was the workable default for this path).
- **Streaming tool-call parsing** — the template assumes tokens are parsed as they arrive so **`<tool_calling>`** can be recognized while `<thinking>` is still open. On **Qwen 3.5-27B**, **`qwen3_xml`** behaved well and was the more robust option. On **Qwen 3.6**, **`qwen3_xml` did not emit tool calls** in that unclosed-thinking situation; **`qwen3_coder`** did.
- **`VLLM_TEST_FORCE_FP8_MARLIN=1`** — keeps the **4090 (SM89)** on **W8A16** instead of native **W8A8**, avoiding precision drift across the two GPUs.
- **NCCL tuning** (`P2P_DISABLE`, `IB_DISABLE`, `Ring`) for stability on PCIe topologies.

With that setup, **Qwen 3.5-27B** completed a **1h 9m** agentic session at **138K** tokens and built a **FastAPI + React** application without tool-calling failures.

## Moving to Qwen 3.6-27B and changing the parser

I pointed the same server at **`Qwen/Qwen3.6-27B-FP8`**, still using **`enhanced.jinja`** and **`preserve_thinking=false`**. **`qwen3_xml`**, which was preferable on 3.5, **did not trigger tool calls** when `<thinking>` stayed open—the case the template is designed for—so I moved the **`--tool-call-parser`** to **`qwen3_coder`**, which streams more aggressively and still picks up the tool call inside the unclosed block. A related fix is discussed in [this vLLM thread](https://www.reddit.com/r/Vllm/comments/1suasv2/comment/oi02krw/?context=1); a future **vLLM 0.20.1** release might let me revisit **`qwen3_xml`** for this workload.

On **driver 591.86** the stack was stable. After moving to **Studio Driver 595.79**, the server began hitting **NCCL deadlocks**: hard freezes mid-generation that required a restart. In logs the failure showed up as **NCCL timeouts**, not parser errors, so it was easy to confuse with tool-calling regressions.

## Why `qwen3_coder` pairs with this template on 3.6

Two separate behaviors interact usefully here:

- **Model output:** Qwen 3.6 sometimes emits a **`<tool_call>`** before closing **`</thinking>`**. The enhanced template leaves that tool call in **plain content** so downstream code can still parse it.
- **Parser behavior:** **`qwen3_coder`** is tuned for code-like streams and detects tool-call patterns **mid-stream** even when the XML framing is incomplete: it does not need a fully closed `<thinking>` section and will fire on **`<tool_call>`**. **`qwen3_xml`** behaves more like strict XML; an unclosed `<thinking>` can block it from surfacing the nested **`<tool_call>`**.

Together, that yields **more resilient extraction** for this template on 3.6 than either piece alone. Neither behavior is ideal in isolation; combined they amount to a **production-viable** setup. I therefore keep **`qwen3_coder`** for **Qwen 3.6** with **`enhanced.jinja`**, while **`qwen3_xml`** remains the better fit for **3.5-27B** on my hardware. If this interpretation disagrees with how the maintainers model the parsers, I am glad to be corrected.

## Driver 595.79, NCCL, and vLLM all-reduce

I had been on **591.86** with acceptable behavior. **595.79** introduced **NCCL deadlocks** that froze generation. My working hypothesis is that the newer driver tightens **NCCL** behavior on **mixed-GPU PCIe** topologies enough to break vLLM’s **custom all-reduce** path. I did not roll back the driver; instead I applied:

1. **Additional environment variables:**
   ```bash
   export NCCL_SHM_DISABLE=0
   export NCCL_P2P_LEVEL=LOC          # restrict P2P to local GPUs only
   export VLLM_RPC_TIMEOUT=180        # prevent premature RPC timeouts
   export VLLM_WORKER_MULTIPROC_METHOD=spawn  # more robust worker lifecycle
   ```
2. **`--disable-custom-all-reduce`** on **`vllm serve`**, forcing **native NCCL** all-reduce instead of vLLM’s custom path on this **PCIe-only** topology.

Without these settings on **595.79**, I saw **intermittent deadlocks** that resembled tool-calling failures in the UI but were not parser issues.

## Long run: 180K tokens

With the driver and NCCL changes in place, I gave **Qwen 3.6-27B** ownership of a directory and a **10 000-token** budget per step, without manual steering.

| Prompt | Wall time | Accumulated tokens |
| --- | --- | --- |
| “Welcome to life, you are Qwen 3.6-27B. Full leadership. What project do you want to build?” | 0s | 0k |
| “Don't ask me – you have full leadership. 10k token budget.” *(model used a Question tool to clarify, then proceeded)* | 31s | 14.0k |
| “Did you check if this is bug-free? It's your own project.” | 17m 13s | 63.3k |
| “Deliver the first possible functional upgrade. Do it nicely.” | 11m 35s | 126.7k |
| *(session ended naturally)* | 10m 46s | **180.0k** |

The model built a **React + Vite + TypeScript** front end with a **FastAPI** backend, revised it after critical feedback, and shipped a further upgrade. I did not observe a **malformed tool call** in that trace. Code: [qwen36_27B_own_project](https://github.com/allanchan339/qwen36_27B_own_project).

## Launch script

The script below is the same one I published in the discussion thread. Lines that end with `\` followed by an inline `#` comment can confuse some shells; if paste fails, drop the comments after `\` and keep the flags.

```bash
#!/bin/bash
# -------------------------------------------------
# Qwen 3.6-27B-FP8 – Agentic-Ready vLLM Launch Script
# Tested: 180K tokens, zero tool-calling failures
# Driver: NVIDIA Studio 595.79
# -------------------------------------------------

# ---- Safe, Speed-Focused Env Vars ----
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1

export OMP_NUM_THREADS=8

# ---- NCCL Tuning for SYS/PCIe Topology ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0          # NEW for driver 595.79
export NCCL_ALGO=Ring
export NCCL_P2P_LEVEL=LOC          # NEW for driver 595.79

# ---- vLLM Stability (Driver-Dependent) ----
export VLLM_RPC_TIMEOUT=180                  # NEW
export VLLM_WORKER_MULTIPROC_METHOD=spawn    # NEW

# ---- FP8 & Memory ----
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SLEEP_WHEN_IDLE=1

# Clean stale FlashInfer cache
rm -rf ~/.cache/flashinfer

# Activate environment
source /home/cychan/vLLM/.venv/bin/activate

vllm serve Qwen/Qwen3.6-27B-FP8 \
  --served-model-name Qwen3.5-27B \
  --chat-template qwen3.5-enhanced.jinja \
  --default-chat-template-kwargs '{"preserve_thinking": false}' \   # MANDATORY: the enhanced jinja will break if this is true
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 219520 \
  --gpu-memory-utilization 0.91 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 12288 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder \          # REQUIRED for Qwen 3.6 with enhanced.jinja; for Qwen 3.5 27B, qwen3_xml also works (see https://www.reddit.com/r/Vllm/comments/1suasv2/)
  --reasoning-parser qwen3 \
  --no-use-tqdm-on-load \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only \
  --disable-custom-all-reduce            # CRITICAL for driver 595.79
```

## Summary

The **`enhanced.jinja`** path needs a **streaming** tool parser that still fires when **`<thinking>`** is left open. On **Qwen 3.5-27B**, **`qwen3_xml`** met that requirement and stayed the more robust option in my tests ([detail](https://www.reddit.com/r/Vllm/comments/1skks8n/qwen_35_27b35ba3b_tool_calling_issues_why_it/)). On **Qwen 3.6**, **`qwen3_xml`** missed tool calls in that pattern, so **`qwen3_coder`** was necessary.

**`preserve_thinking`** must stay **`false`** for **`qwen3.5-enhanced.jinja`** on Qwen 3.6; **`true`** is incompatible with that template in my setup.

**591.86 → 595.79** introduced **NCCL deadlocks** on the mixed **4090/3090** box. Mitigations were **`NCCL_SHM_DISABLE=0`**, **`NCCL_P2P_LEVEL=LOC`**, **`VLLM_RPC_TIMEOUT=180`**, **`VLLM_WORKER_MULTIPROC_METHOD=spawn`**, and **`--disable-custom-all-reduce`**. Without them, deadlocks can look like tool failures.

**`VLLM_TEST_FORCE_FP8_MARLIN=1`** and **NCCL** tuning remain mandatory for mixed FP8 ranks; the driver upgrade added the extra variables above.

**Qwen 3.6-27B** is a dense **27B** checkpoint that, on the public agentic-coding figures I cite, beats **Qwen 3.5-397B-A17B** (**SWE-bench Verified** 77.2 vs 76.2, **Pro** 53.5 vs 50.9, **SkillsBench** 48.2 vs 30.0)—a larger step than a minor revision.

In one continuous session the stack sustained about **180K** tokens with **no tool-calling errors** and roughly **10 minutes** of uninterrupted agentic use on consumer GPUs, which matches what I wanted from a production-minded local setup.

The same template works on **Qwen 3.5** and **3.6** if **`preserve_thinking=false`** and the parser matches the model generation. With the **595.79** workarounds, I get stable **180K-token** agentic runs. Step-by-step background and earlier tuning notes are in the [Qwen 3.5 deep-dive](https://www.reddit.com/r/Vllm/comments/1skks8n/qwen_35_27b35ba3b_tool_calling_issues_why_it/).

## Resources

- [Qwen 3.5 tool-calling deep-dive](https://www.reddit.com/r/Vllm/comments/1skks8n/qwen_35_27b35ba3b_tool_calling_issues_why_it/)
- [Parser behaviour: Qwen 3.5 vs 3.6](https://www.reddit.com/r/Vllm/comments/1suasv2/)
- [Qwen 3.6-27B project repository](https://github.com/allanchan339/qwen36_27B_own_project)
- [vLLM / Qwen config repository](https://github.com/allanchan339/vLLM-Qwen3.5-27B)
