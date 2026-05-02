---
layout: post
title: "qwen3.6-enhanced.jinja: CoT leakage into tool turns and why preserve_thinking works now"
date: 2026-05-02 00:00:00 +0800
description: "I ship qwen3.6-enhanced.jinja for Qwen 3.6 (built on the 3.5 enhanced design): diagnosing double-wrapped thinking, the self-healing assistant branch, and how that restores safe preserve_thinking—after the earlier qwen3.5-enhanced-on-3.6 workaround that required preserve_thinking=false."
tags: [vllm, qwen, tool-calling, llm, jinja, agent, inference]
categories: [bug-fixes]
---

In [my April note on Qwen 3.6-27B]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) I described a stack that survived a long agentic trace: **`qwen3.5-enhanced.jinja`** on the 3.6 checkpoint, **`qwen3_coder`** for streaming extraction, **`preserve_thinking=false`** (mandatory there), and NCCL tweaks after **Studio Driver 595.79**. That combination worked in practice, but **`preserve_thinking=false` was compensating for a template bug**, not declaring a philosophical preference—I only chased it seriously once **reasoning started bleeding into `tool_response` turns** and **tool calls that the runtime never acted on**.

I now ship **[`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** as my Qwen 3.6-specific enhanced template (**I developed it**—multimodal and interleaved-thinking behaviour aligned to 3.6, plus the fixes below; [raw file](https://raw.githubusercontent.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/main/chat-template/qwen3.6-enhanced.jinja) for drop-in `vllm serve --chat-template`). With this file, **`preserve_thinking` is supported** (`true` or `false` per your tracing needs) because malformed assistant turns no longer pile an extra vacant think scaffold on top of an unclosed inner block. **Working proof** of the stack with this template (not the older **`qwen3.5-enhanced.jinja`** path) is **[qwen36_27B_36jinja_project on GitHub](https://github.com/allanchan339/qwen36_27B_36jinja_project)**—the artifact I point to when someone asks for a reproducible agentic trace.

This post is the template-side story: **why** pointing raw **`qwen3.5-enhanced.jinja`** at 3.6 could corner the runtime, why **Qwen 3.5** mostly avoided it, and the **minimal self‑healing** step I put in the **`assistant`** branch of **`qwen3.6-enhanced.jinja`** before the reasoning split.

## What broke in plain terms

Sometimes the assistant emitted something shaped like:

- an opening think marker (**`qwen3.6-enhanced.jinja`** and **`qwen3.5-enhanced.jinja`** share the same literal `` `<think>` `` family; the April post used **`thinking`** casually for readability),

- **no closing tag** before a raw `` `<tool_call>` `` block.

Training and runtime prompts encourage **closed** think sections. Reality is messier: the model can wedge a tool payload **inside** what is effectively still “thinking.”

Separately, the flawed **`qwen3.5-enhanced.jinja`**-on-Qwen 3.6 path used assistant logic equivalent to wrapping **every** qualifying turn in a synthetic think sandwich **even when `reasoning_content` stayed empty**. That interacted badly with malformed history: after rendering, it could look like the model was **still inside** an outer think envelope when `` `<tool_call>` `` appeared. Downstream behaviour matches what I observed as **CoT leakage across turn boundaries** and **tool instructions that never get scheduled**.

None of this negates **`qwen3_coder`** on 3.6—the parser lane still matters—but fixing the template removes a **structural** failure mode rather than leaning only on parsing heuristics.

## Why reasoning extraction silently failed

The template extracts `reasoning_content` by looking for `` `</think>` `` in the **message body**. When the assistant never emits that closing tag, the splitter never runs, `reasoning_content` stays **empty**, and the **remainder** stays the full raw string—**including** the unclosed opening think tag ahead of `` `<tool_call>` ``.

A **3.6-style** handler that unconditionally wrapped “post–last-user” assistant text in opening and closing redacted-thinking fences, **plus** the recombined body, then effectively produced stacked think markup: a **vacant** fenced block followed by thought text that **still began with** another dangling `` `<think>` `` ahead of `` `<tool_call>` ``.

From the model’s point of view that is dangerously close to “tool call emitted while still reasoning,” which rationalizes **ignored tool XML** and **follow-up prose that belongs in Think leaking into structured tool payloads**.

### Why Qwen 3.5’s enhanced path looked immune

Earlier **3.5** logic only wrapped assistant output in an explicit Think block when `reasoning_content` was **non-empty** after splitting. Malformed transcripts with **no close tag** left `reasoning_content` blank, skipped the synthetic wrapper entirely, and dropped the messy body through as **bare assistant content**. Tool XML was ugly but **outside** any outer think scaffold the template fabricated. That is a **lucky default**, not proof that unfinished think sections are harmless—only proof that **the older condition failed open** toward plain content instead of layering new fences.

## The fix I settled on

I wanted **deterministic repair**, not another special case that might leave historic turns ending in `` `<think>` `` without a sibling close before `` `<tool_call>` ``:

1. **Self-healing (before splitting):**  
   When both `` `<tool_call>` `` and `` `<think>` `` appear and the **last** `` `</think>` `` sits **before** the **last** `` `<think>` `` (including the `-1 / missing` cases), inject `` `</think>` `` immediately **before** the first `` `<tool_call>` `` when that tool call sits after the dangling opener; otherwise append `` `</think>` `` at the end.

2. **Keep the outer think wrapper unchanged** afterward: splitting now sees balanced markers, extracts `reasoning_content` cleanly, and the tool payload never sits upstream of **two** contradictory think layers.

Roughly—the snippet lives today in **`qwen3.6-enhanced.jinja`**; the operative structure is:

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

Above, tags match **[`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** as checked in; if you merge this into another fork, substitute your **literal** open/close think strings verbatim.

**Branches I deliberately did not adopt:** emitting a **trail-only** `` `<think>\n` `` wrapper for assistant history when reasoning is blank but the turn qualifies for preservation. That mirrors the **`add_generation_prompt`** tail—which is appropriate at **generation start**—but is **incorrect** mid-conversation because it nests the next `` `<tool_call>` `` beneath an unfinished think scaffold.

## Practical scope

- **Surface area:** the `assistant` message branch through the unchanged `tool` message handler—nothing else needed in my audits (system preamble, structured `` `tool_calls` `` serialization, trailing generation prompt untouched).
- **Interaction with knobs:** with **`qwen3.6-enhanced.jinja`**, **`preserve_thinking=true`** is a safe option again—histories carry **balanced** fences after self-healing, so interleaved-thinking strip/keep semantics stay predictable. On bare **`qwen3.5-enhanced.jinja`** against 3.6 I still recommend **`preserve_thinking=false`** until you migrate.

## What stays the same in the April stack

[**The April launcher**]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) remains the blueprint for parsers, GPUs, MARLIN-aligned FP8, NCCL tweaks, **`--disable-custom-all-reduce`** on **595.79**, and **`qwen3_coder`** on 3.6. Point **`--chat-template`** at the local path of **[`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** (clone or copy from [the `chat-template/` folder](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/tree/main/chat-template)); **`--default-chat-template-kwargs`** can then set **`preserve_thinking`** to **`true`** or **`false`** as you prefer (April’s **`preserve_thinking=false`** was keyed to **`qwen3.5-enhanced.jinja`** on 3.6, not to vLLM itself).

Where I reran transcripts that previously reproduced leakage, executions **scheduled** reliably again and stray reasoning stopped surfacing downstream of repaired `` `<tool_call>` `` markers; the public trace and code live in **[qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)**. Others’ mileage will vary by checkpoint and client parsing, which is exactly why **I publish both halves**: parser ergonomics plus **truthful templating**, plus a **repo you can clone** when a blog post is not enough.

## Summary

The flawed **`qwen3.5-enhanced.jinja`** assistant branch aimed at **3.6**, combined with **sometimes-unclosed `` `<think>` `` markers**, yielded **double layering** after rendering: vacant synthetic think blocks atop still-open reasoning. Downstream failures looked like ignored tools and polluted tool responses—not always mistakable for NCCL deadlocks.

**Qwen 3.5** mostly dodged explosion because wrappers were **skipped** whenever split reasoning was empty. **That conditional went away on the faulty 3.6-on-3.5 path**, so **`preserve_thinking=false`** ended up masking the breakage.

That repair lives in **`qwen3.6-enhanced.jinja`** (my template): **pre-split self-healing** guarantees a closing think tag precedes structured tool XML whenever both appear, so **`preserve_thinking`** can expose full traces without resurrecting ignored tools. **Operationally**, keep parser and GPU settings from April; swap **`--chat-template`** and **revisit** **`preserve_thinking`** as intended rather than forced off. **[qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)** is the end-to-end proof repository for this template path.

## Resources

- **[`qwen3.6-enhanced.jinja` (source)](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)** — [raw](https://raw.githubusercontent.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/main/chat-template/qwen3.6-enhanced.jinja)
- **[`qwen3.5-enhanced.jinja` (same repo, 3.5 baseline)](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.5-enhanced.jinja)**
- **[Repository: vLLM Qwen 3 / 3.5 / 3.6 chat-template fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)** — templates under `chat-template/`
- **[Proof: `qwen3.6-enhanced.jinja` agentic run — qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)**
- [Prior field note: Qwen 3.6-27B on vLLM with `qwen3.5-enhanced.jinja`]({% post_url 2026-04-29-Qwen36-27B-tool-calling %})
- [Older Qwen 3.6 35B-A3B comparison]({% post_url 2026-04-20-Qwen36-35B-A3B-tool-calling %})
- [April demo project (`qwen3.5-enhanced.jinja` on 3.6)](https://github.com/allanchan339/qwen36_27B_own_project)
