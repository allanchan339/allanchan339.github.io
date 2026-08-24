---
layout: page
title: Qwen 3.X Tool-Calling Silent Failure in vLLM
description: Fixing tool-calling reliability for Qwen 3.5/3.6 on mixed-GPU consumer hardware
img: assets/img/vLLM.jpg
importance: 1
---

I developed a chat template fix and configuration stack that makes Qwen 3.5/3.6 tool-calling reliable on vLLM with mixed consumer GPUs (RTX 4090 + RTX 3090). The core artifact is [`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja), a self-healing chat template that restores full Qwen 3.6-series capability without the `preserve_thinking=false` compromise.

This project spans three blog posts documenting the debugging journey: [Qwen 3.6-35B-A3B]({% post_url 2026-04-20-Qwen36-35B-A3B-tool-calling %}), [Qwen 3.6-27B-FP8]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}), and [qwen3.6-enhanced.jinja]({% post_url 2026-05-02-Qwen36-27B-updated-jinja %}).

## The Problem

When running Qwen 3.5/3.6 on vLLM for agentic work (tool calling across long sessions), three failure modes converge:

1. **CoT leakage into tool turns.** Qwen 3.6 sustains interleaved `<thinking>` blocks across turns in a way 3.5 does not. The model sometimes emits `<tool_call>` before closing `</thinking>`, causing chain-of-thought reasoning to bleed into tool payloads.

2. **Double-wrapping with `qwen3.5-enhanced.jinja` on 3.6.** The 3.5 template unconditionally wraps qualifying assistant turns in a synthetic think envelope — even when `reasoning_content` is empty. On 3.6 this stacks on top of the model's own unclosed `<thinking>` block, producing two layers of think markup that trap `tool_call` inside an unterminated reasoning region.

3. **`preserve_thinking` forced off.** With `qwen3.5-enhanced.jinja` on 3.6, `preserve_thinking=true` makes things worse because it keeps the broken double-layer structure in rendered history. Setting it to `false` was the workaround — but it strips reasoning traces that 3.6 is meant to expose.

The net result: tools get ignored, `tool_response` contains stray reasoning text, and long agentic sessions terminate prematurely.

## Why `qwen3.5-enhanced.jinja` Fails on 3.6

The 3.5 template does **not** repair a missing `</redacted_thinking>`: there is no pass that closes a dangling opener or strips half-open think markup. Whatever the assistant emitted — including `<redacted_thinking>` with no matching close before `tool_call` — shows up in the serialized prompt the causal model conditions on next step.

On **Qwen 3.5** this was mostly invisible because:
- Qwen 3.5 does not sustain long-lived interleaved thinking the way 3.6 does
- Interleaved chat templating discards many think segments for assistant turns before the last real user message, so most of the half-open scaffold never re-enters the prefix

On **Qwen 3.6** that safety net disappears. The model keeps thinking blocks alive across turns, so the same "don't repair the close" policy produces visible CoT bleed, ignored tools, and prefix pollution that compounds over long sessions.

The double-wrapping happens because the 3.5 template's assistant handler unconditionally wraps post–last-user assistant text in opening and closing redacted-thinking fences. When the raw string already contains an unclosed `<redacted_thinking>` ahead of `<tool_call>`, the template adds a second outer layer. From the model's perspective, that looks like "tool call emitted while still reasoning" — rationalizing ignored tool XML and follow-up prose leaking into structured tool payloads.

## The Fix: `qwen3.6-enhanced.jinja`

I developed [`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja) with deterministic pre-split self-healing:

1. **Self-healing (before splitting):** When both `<tool_call>` and `<redacted_thinking>` appear and the last `</redacted_thinking>` sits before the last `<redacted_thinking>` (including the −1 / missing cases), inject `</redacted_thinking>` immediately before the first `<tool_call>` when that tool call sits after the dangling opener; otherwise append `</redacted_thinking>` at the end.

2. **Balanced markers for splitting:** After self-healing, the splitter sees balanced markers, extracts `reasoning_content` cleanly, and the tool payload never sits upstream of two contradictory think layers.

The operative structure in the `assistant` branch:

{% raw %}
{::nomarkdown}
<div class="language-jinja highlighter-rouge"><div class="highlight"><pre class="highlight"><code>{%- elif message.role == "assistant" -%}
    {%- set content = render_content(message.content, true)|trim -%}

    {# Ensure &lt;/redacted_thinking&gt; exists before tool XML when opener was left dangling #}
    {%- if '&lt;tool_call&gt;' in content and '&lt;redacted_thinking&gt;' in content -%}
        {%- set last_think = content.rfind('&lt;redacted_thinking&gt;') -%}
        {%- set last_close = content.rfind('&lt;/redacted_thinking&gt;') -%}
        {%- set tool_pos = content.find('&lt;tool_call&gt;') -%}
        {%- if last_close &lt; last_think or last_close == -1 -%}
            {%- if tool_pos &gt; last_think -%}
                {%- set content = content[:tool_pos] ~ '&lt;/redacted_thinking&gt;' ~ content[tool_pos:] -%}
            {%- else -%}
                {%- set content = content ~ '&lt;/redacted_thinking&gt;' -%}
            {%- endif -%}
        {%- endif -%}
    {%- endif -%}

    {%- set reasoning_content = '' -%}
    {# … existing reasoning extraction + interleaved-thinking render … #}
{%- endif -%}
</code></pre></div></div>
{:/nomarkdown}
{% endraw %}

With this template, `preserve_thinking` becomes a genuine choice again — `true` or `false` — instead of a forced-off workaround. Histories carry balanced fences after self-healing, so interleaved-thinking strip/keep semantics stay predictable.

## Key Configuration

### Parser selection

| Model | Parser | Why |
|-------|--------|-----|
| Qwen 3.5-27B | `qwen3_xml` | Robust on 3.5; handles unclosed thinking |
| Qwen 3.6 (any) | `qwen3_coder` | Streams more aggressively; picks up `tool_call` inside unclosed `<thinking>` |

`qwen3_xml` does not emit tool calls when `<thinking>` stays open on 3.6 — the case the template is designed for. `qwen3_coder` fires on `<tool_call>` even with incomplete XML framing.

### NCCL tuning (mixed 4090/3090 PCIe)

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
```

### Driver 595.79 additions

```bash
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=LOC
export VLLM_RPC_TIMEOUT=180
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Plus `--disable-custom-all-reduce` on `vllm serve` to force native NCCL all-reduce instead of vLLM's custom path on PCIe-only topologies.

### FP8 precision alignment

```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1
```

Keeps the 4090 (SM89) on W8A16 instead of native W8A8, avoiding precision drift across mixed GPU ranks.

### Launch script (`qwen3.6-enhanced.jinja`, `preserve_thinking=true`)

```bash
#!/bin/bash
# vLLM v0.19.0 (tested on this version)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export OMP_NUM_THREADS=8

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_P2P_LEVEL=LOC
export VLLM_RPC_TIMEOUT=180
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SLEEP_WHEN_IDLE=1

rm -rf ~/.cache/flashinfer
source /home/cychan/vLLM/.venv/bin/activate

vllm serve Qwen/Qwen3.6-27B-FP8 \
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
#  --disable-custom-all-reduce   # uncomment on Studio 595.79 + mixed GPU if you hit NCCL deadlocks
```

## Results

### 180K-token agentic run (Qwen 3.6-27B-FP8, `qwen3.5-enhanced.jinja` + `qwen3_coder`)

| Prompt | Wall time | Accumulated tokens |
|--------|-----------|-------------------|
| "Welcome to life, you are Qwen 3.6-27B. Full leadership. What project do you want to build?" | 0s | 0k |
| "Don't ask me – you have full leadership. 10k token budget." | 31s | 14.0k |
| "Did you check if this is bug-free? It's your own project." | 17m 13s | 63.3k |
| "Deliver the first possible functional upgrade. Do it nicely." | 11m 35s | 126.7k |
| *(session ended naturally)* | 10m 46s | **180.0k** |

Zero malformed tool calls. The model built a React + Vite + TypeScript frontend with a FastAPI backend, revised it, and shipped an upgrade. Code: [qwen36_27B_own_project](https://github.com/allanchan339/qwen36_27B_own_project).

### 128K-token agentic run (Qwen 3.6-27B-FP8, `qwen3.6-enhanced.jinja` + `preserve_thinking=true`)

Validated the new template end-to-end with `preserve_thinking` enabled — the configuration that was previously broken on `qwen3.5-enhanced.jinja`. Code: [qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project).

### Cross-model comparison (Qwen 3.6-35B-A3B)

| Configuration | Survival | Failure mode |
|---------------|----------|--------------|
| `enhanced.jinja` + `qwen3_xml` | ~111k tokens (~13m 20s) | Improper tool calling |
| `official.jinja` + `qwen3_coder` | 6m 32s | Improper tool calling |
| `official.jinja` + `qwen3_xml` | ~1m 16s | Malformed tool calls inside thinking |

Even the best 3.6-35B-A3B configuration fails more frequently than Qwen 3.5-27B under the same harness. Qwen 3.5-27B remains more reliable for long-horizon agentic work.

## Repos

| Repo | What it is |
|------|-----------|
| [vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) | `chat-template/` — the jinja templates (`qwen3.5-enhanced.jinja`, `qwen3.6-enhanced.jinja`) |
| [qwen36_27B_own_project](https://github.com/allanchan339/qwen36_27B_own_project) | 180K-token proof run with `qwen3.5-enhanced.jinja` |
| [qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project) | 128K-token proof run with `qwen3.6-enhanced.jinja` + `preserve_thinking=true` |

## Impacts

This work has been picked up by the local-LLM community beyond the original blog posts:

- **NVIDIA Developer Forums** — A user referenced the improved Qwen jinja templates on the thread [*"What's the best speed we can get with Qwen 3.6 27B without quantizing?"*](https://forums.developer.nvidia.com/t/whats-the-best-speed-we-can-get-with-qwen-3-6-27b-without-quantizing/367561/32) (June 2026), pointing others to `vLLM-Qwen3-3.5-3.6-chat-template-fix` as the fix for agentic use.
- **Community fork / merge** — A [GitHub Gist](https://gist.github.com/fakezeta/9e8e039c60332fcb143c6e805558afe0) merged `qwen3.6-enhanced.jinja` with froggeric's multimodal template (`qwen3.6_merged_template.jinja`), extending the self-healing logic to vision inputs.
- **Claude Code + Qwen guides** — External write-ups such as [*"Quick Tip :: Fix for Claude Code using Qwen 3.6"*](https://polarsparc.github.io/Claude/Claude-Qwen3-Jinja.html) (June 2026) cite the template approach for getting tool calling to work with local Qwen 3.6 backends.
- **Reddit traction** — The debugging notes on [r/vLLM](https://www.reddit.com/r/Vllm/comments/1skks8n/) and [r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1sqpsut/) sparked follow-up testing from other users running mixed-GPU consumer setups, which directly motivated the `qwen3.6-enhanced.jinja` release.

## Blog Posts

- [Qwen 3.6-35B-A3B on vLLM: do the Qwen 3.5 tool-calling fixes carry over?]({% post_url 2026-04-20-Qwen36-35B-A3B-tool-calling %}) — first test of 3.6 with the 3.5 stack
- [Qwen 3.6-27B-FP8 on vLLM: enhanced.jinja, qwen3_coder, and fixing NCCL after Studio Driver 595.79]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) — parser switch, NCCL fixes, 180K-token run
- [qwen3.6-enhanced.jinja: CoT leakage into tool turns and why preserve_thinking works now]({% post_url 2026-05-02-Qwen36-27B-updated-jinja %}) — the self-healing template fix
