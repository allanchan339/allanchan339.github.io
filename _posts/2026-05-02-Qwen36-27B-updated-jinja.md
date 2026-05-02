---
layout: post
title: "qwen3.6-enhanced.jinja: CoT leakage into tool turns and why preserve_thinking works now"
date: 2026-05-02 00:00:00 +0800
description: "Why Qwen 3.6 with qwen3.5-enhanced.jinja forced preserve_thinking=false, and how qwen3.6-enhanced.jinja restores full Qwen 3.6-series capability—self-healing think/tool boundaries, safe preserve_thinking."
tags: [vllm, qwen, tool-calling, llm, jinja, agent, inference]
categories: [bug-fixes]
---

In [my April note on Qwen 3.6-27B]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) I described a stack that survived a long agentic trace: **`qwen3.5-enhanced.jinja`** on the 3.6 checkpoint, **`qwen3_coder`** for streaming extraction, **`preserve_thinking=false`**, and NCCL tweaks after **Studio Driver 595.79**.

**That is the same cluster of reasons **`preserve_thinking`** had to stay off:** **Qwen 3.6** **sustains** interleaved **thinking** in a way **3.5** largely does not; **`qwen3.5-enhanced.jinja`** **does not repair** missing `` `</think>` `` and can **double-wrap** assistant turns on **3.6**; with **`preserve_thinking=true`** the template **keeps** more of that **broken** structure in **rendered history**, so **prefix pollution**, **CoT bleed**, and **ignored `tool_call`** **get worse**. **`preserve_thinking=false`** was the **pressure-release**—**stripping** much **think** from **earlier** turns so agent runs could finish—not a statement that **3.6** “should not” expose reasoning. I dug in when **reasoning still leaked into `tool_response`** and **tools stopped firing** even with the flag off.

I developed **[`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** so the **Qwen 3.6 family** can use an enhanced chat template **without** that compromise: **multimodal** paths, **interleaved** thinking aligned to how **3.6** actually behaves, **self-healing** before the reasoning split, and **`preserve_thinking` supported** (`true` or `false`)—i.e. the **full surface** the **3.6 series** is meant to expose, instead of **turning off** **`preserve_thinking`** to paper over **3.5-enhanced-on-3.6** bugs. [Raw file](https://raw.githubusercontent.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/main/chat-template/qwen3.6-enhanced.jinja) for `vllm serve --chat-template`. **Working proof:** **[qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)**.

This post is the template-side story: **why** pointing raw **`qwen3.5-enhanced.jinja`** at 3.6 could corner the runtime, why that file **never inserts** a missing `` `</think>` `` (it **leaves the broken assistant text in the prompt**—**causal** models still **condition on it**), and the **minimal self‑healing** step I put in the **`assistant`** branch of **`qwen3.6-enhanced.jinja`** before the reasoning split.

## What broke in plain terms

Sometimes the assistant emitted something shaped like:

- an opening think marker (**`qwen3.6-enhanced.jinja`** and **`qwen3.5-enhanced.jinja`** share the same literal `` `<think>` `` family; the April post used **`thinking`** casually for readability),

- **no closing tag** before a raw `` `<tool_call>` `` block.

Training and runtime prompts encourage **closed** think sections. Reality is messier: the model can wedge a tool payload **inside** what is effectively still “thinking.”

Separately, using **`qwen3.5-enhanced.jinja`** on **Qwen 3.6** used assistant logic equivalent to wrapping **every** qualifying turn in a synthetic think sandwich **even when `reasoning_content` stayed empty**. That interacted badly with malformed history: after rendering, it could look like the model was **still inside** an outer think envelope when `` `<tool_call>` `` appeared. Downstream behaviour matches what I observed as **CoT leakage across turn boundaries** and **tool instructions that never get scheduled**.

None of this negates **`qwen3_coder`** on 3.6—the parser lane still matters—but fixing the template removes a **structural** failure mode rather than leaning only on parsing heuristics.

## Why reasoning extraction silently failed

The template extracts `reasoning_content` by looking for `` `</think>` `` in the **message body**. When the assistant never emits that closing tag, the splitter never runs, `reasoning_content` stays **empty**, and the **remainder** stays the full raw string—**including** the unclosed opening think tag ahead of `` `<tool_call>` ``.

A **3.6-style** handler that unconditionally wrapped “post–last-user” assistant text in opening and closing redacted-thinking fences, **plus** the recombined body, then effectively produced stacked think markup: a **vacant** fenced block followed by thought text that **still began with** another dangling `` `<think>` `` ahead of `` `<tool_call>` ``.

From the model’s point of view that is dangerously close to “tool call emitted while still reasoning,” which rationalizes **ignored tool XML** and **follow-up prose that belongs in Think leaking into structured tool payloads**.

### What `qwen3.5-enhanced.jinja` actually does (and does not do)

**`qwen3.5-enhanced.jinja`** does not repair a missing `` `</think>` ``: there is no pass that closes a dangling opener or strips half-open think markup. Whatever the assistant emitted—including **`<think>`** with no matching close before **`tool_call`**—can still **show up** in the **serialized prompt** the **causal** model conditions on next step; “letting it be” is **input-side pollution** in principle whenever that text **stays** in prefix.

**Why the same no-fix workaround looked “fine” on Qwen 3.5:** in my runs **Qwen 3.5 does not really sustain** a long-lived **interleaved thinking** block the way **Qwen 3.6** does—it **lacks** that **stickier** “keep thinking open across turns” behaviour. **Interleaved** chat templating also **discards** many **think** segments for assistant turns **before** the last real user message, so **most** of the half-open scaffold **never re-enters** the prefix the model sees. **3.6** is where that stops being a sufficient safety net, so the **same** “don’t repair the close” policy starts to **hurt** visibly (**CoT bleed**, ignored tools) and **self-healing** in **`qwen3.6-enhanced.jinja`** becomes worth the complexity.

Earlier **3.5** assistant logic only wrapped output in an explicit Think block when `reasoning_content` was **non-empty** after splitting. With **no close tag**, `reasoning_content` stayed blank, the template **skipped an extra synthetic think envelope**, and the **same dirty assistant string** (still containing the unclosed opener) was emitted as **bare assistant content**. That sometimes kept **`tool_call`** **outside** a **second** layer of scaffolding the template would have invented—helping **scheduling**—but it **did not** make the **token history** structurally clean. On the faulty **3.6-on-3.5-enhanced** path, the unconditional wrapper **added** that outer layer on top of the still-unclosed inner block, which made tool behaviour worse **without** fixing the underlying transcript hygiene problem.

## The fix I settled on

I wanted **deterministic repair**, not another special case that might leave historic turns ending in `` `<think>` `` without a sibling close before `` `<tool_call>` ``:

1. **Self-healing (before splitting):**  
   When both `` `<tool_call>` `` and `` `<think>` `` appear and the **last** `` `</think>` `` sits **before** the **last** `` `<think>` `` (including the `-1 / missing` cases), inject `` `</think>` `` immediately **before** the first `` `<tool_call>` `` when that tool call sits after the dangling opener; otherwise append `` `</think>` `` at the end.

2. **Keep the outer think wrapper unchanged** afterward: splitting now sees balanced markers, extracts `reasoning_content` cleanly, and the tool payload never sits upstream of **two** contradictory think layers.

Roughly—the snippet lives today in **`qwen3.6-enhanced.jinja`**; the operative structure is:

{% raw %}
```jinja
{%- elif message.role == "assistant" -%}
    {%- set content = render_content(message.content, true)|trim -%}

    {# Ensure </think> exists before tool XML when opener was left dangling #}
    {%- if '<tool_call>' in content and '<think>' in content -%}
        {%- set last_think = content.rfind('<think>') -%}
        {%- set last_close = content.rfind('</think>') -%}
        {%- set tool_pos = content.find('<tool_call>') -%}
        {%- if last_close < last_think or last_close == -1 -%}
            {%- if tool_pos > last_think -%}
                {%- set content = content[:tool_pos] ~ '</think>' ~ content[tool_pos:] -%}
            {%- else -%}
                {%- set content = content ~ '</think>' -%}
            {%- endif -%}
        {%- endif -%}
    {%- endif -%}

    {%- set reasoning_content = '' -%}
    {# … existing reasoning extraction + interleaved-thinking render … #}
{%- endif -%}
```
{% endraw %}

Above, tags match **[`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** as checked in; if you merge this into another fork, substitute your **literal** open/close think strings verbatim.

**Branches I deliberately did not adopt:** emitting a **trail-only** `` `<think>\n` `` wrapper for assistant history when reasoning is blank but the turn qualifies for preservation. That mirrors the **`add_generation_prompt`** tail—which is appropriate at **generation start**—but is **incorrect** mid-conversation because it nests the next `` `<tool_call>` `` beneath an unfinished think scaffold.

## Practical scope

- **Surface area:** the `assistant` message branch through the unchanged `tool` message handler—nothing else needed in my audits (system preamble, structured `` `tool_calls` `` serialization, trailing generation prompt untouched).
- **Interaction with knobs:** with **`qwen3.6-enhanced.jinja`**, **`preserve_thinking=true`** is a safe option again—histories carry **balanced** fences after self-healing, so interleaved-thinking strip/keep semantics stay predictable. On bare **`qwen3.5-enhanced.jinja`** against 3.6 I still recommend **`preserve_thinking=false`** until you migrate.

## What stays the same in the April stack

[**The April launcher**]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) remains the blueprint for parsers, GPUs, MARLIN-aligned FP8, NCCL tweaks, **`--disable-custom-all-reduce`** on **595.79**, and **`qwen3_coder`** on 3.6. Point **`--chat-template`** at the local path of **[`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** (clone or copy from [the `chat-template/` folder](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/tree/main/chat-template)); **`--default-chat-template-kwargs`** can then set **`preserve_thinking`** to **`true`** or **`false`** as you prefer (April’s **`preserve_thinking=false`** was keyed to **`qwen3.5-enhanced.jinja`** on 3.6, not to vLLM itself).

Where I reran transcripts that previously reproduced leakage, executions **scheduled** reliably again and stray reasoning stopped surfacing downstream of repaired `` `<tool_call>` `` markers; the public trace and code live in **[qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)**. Others’ mileage will vary by checkpoint and client parsing, which is exactly why **I publish both halves**: parser ergonomics plus **truthful templating**, plus a **repo you can clone** when a blog post is not enough.

## vLLM launch recipe (`qwen3.6-enhanced.jinja`, `preserve_thinking=true`)

Below is the **vLLM** recipe I use with **`qwen3.6-enhanced.jinja`** and **`preserve_thinking: true`** (the pairing this post is about). Point **`--chat-template`** at your local copy—e.g. from [`chat-template/qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja). Adjust **`source …/activate`**, **GPU** indices, and paths for your box. Lines that end with `\` plus an inline `# …` can trip some shells; drop those comments after `\` if paste fails.

On **NVIDIA Studio 595.79** with **mixed GPUs** I still needed **`--disable-custom-all-reduce`** for stability ([April note]({% post_url 2026-04-29-Qwen36-27B-tool-calling %})); it is commented here so you can enable it without hunting the flag.

```bash
#!/bin/bash
# ------------------------------
# Safe, Speed-Focused Env Vars
# ------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # mixed-GPU safeguard
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1

export OMP_NUM_THREADS=8

# NCCL tuning for SYS/PCIe topology
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export MODEL_NAME="Qwen/Qwen3.6-27B-FP8"
export NCCL_P2P_LEVEL=LOC
export VLLM_RPC_TIMEOUT=180
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --------------------------
# Clean stale FlashInfer cache
# --------------------------
rm -rf ~/.cache/flashinfer

# Activate virtual environment (change to your path)
source /home/cychan/vLLM/.venv/bin/activate

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SLEEP_WHEN_IDLE=1

vllm serve $MODEL_NAME \
  --served-model-name Qwen3.5-27B \
  --chat-template qwen3.6-enhanced.jinja \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
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
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --no-use-tqdm-on-load \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only
#  --disable-custom-all-reduce   # uncomment on Studio 595.79 + mixed GPU if you hit NCCL deadlocks (see April post)

# Optional: Qwen3 MTP speculative decoding (needs headroom; 80B-A3B speculator not on current hardware)
#  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}' \
```

## Summary

The flawed **`qwen3.5-enhanced.jinja`** assistant branch aimed at **3.6**, combined with **sometimes-unclosed `` `<think>` `` markers**, yielded **double layering** after rendering: vacant synthetic think blocks atop still-open reasoning. Downstream failures looked like ignored tools and polluted tool responses—not always mistakable for NCCL deadlocks.

**`qwen3.5-enhanced.jinja`** could **look** less explosive on **3.5** partly because an **empty** `reasoning_content` **skipped** an extra synthetic wrapper—**not** because it **healed** think markup—and partly because **Qwen 3.5** in my experience **does not keep** a **thinking** block **alive** the way **3.6** does, so **prefix pollution** rarely **compounds**. **That skip disappeared on the faulty 3.6-on-3.5 path**, and **3.6** **does** sustain interleaved thinking, so **`preserve_thinking=false`** on the old file masked **double-layer** tool failures while **dirty prefixes** became a **first-class** problem.

**`qwen3.6-enhanced.jinja`** uses **pre-split self-healing** to **insert** the missing close where needed so **`tool_call`** is not trapped inside an unterminated think region **and** the serialized history is **not** stuck carrying an endless “still thinking” span before the tool payload. That is what lets **`preserve_thinking`** work without the old trade-off between **trace fidelity** and **clean conditioning**. **Operationally**, keep parser and GPU settings from April; swap **`--chat-template`** and **revisit** **`preserve_thinking`** as intended rather than forced off. **[qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)** is the end-to-end proof repository for this template path.

## Resources

- **[`qwen3.6-enhanced.jinja` (source)](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** — [raw](https://raw.githubusercontent.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/main/chat-template/qwen3.6-enhanced.jinja)
- **[`qwen3.5-enhanced.jinja` (source)](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.5-enhanced.jinja)**
- **[Repository: vLLM Qwen 3 / 3.5 / 3.6 chat-template fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)** — `chat-template/`
- **[Proof: `qwen3.6-enhanced.jinja` agentic run — qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)**
- [Prior field note: Qwen 3.6-27B on vLLM with `qwen3.5-enhanced.jinja`]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) — [Reddit discussion](https://www.reddit.com/r/LocalLLM/comments/1sv6cqk/follow_up_tested_tool_calling_fixes_for_qwen/)
- [April demo project (`qwen3.5-enhanced.jinja` on 3.6)](https://github.com/allanchan339/qwen36_27B_own_project)
