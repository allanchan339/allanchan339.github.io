---
layout: page
title: Qwen 3.X Tool-Calling Silent Failure in vLLM
description: A root-cause fix for Qwen3.x tool-calling silent failures on vLLM via a self-healing chat template
img: assets/img/vLLM.jpg
importance: 1
---

I developed a self-healing chat template that fixes a structural tool-calling failure affecting the Qwen3.x series on vLLM. The core artifact is [`qwen3.6-enhanced.jinja`](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja), which repairs unclosed thinking markers before the reasoning split so `preserve_thinking` becomes a free choice instead of a forced-off workaround.

This project spans three blog posts documenting the debugging journey: [Qwen 3.6-35B-A3B]({% post_url 2026-04-20-Qwen36-35B-A3B-tool-calling %}), [Qwen 3.6-27B-FP8]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}), and [qwen3.6-enhanced.jinja]({% post_url 2026-05-02-Qwen36-27B-updated-jinja %}).

## Root Cause

The Qwen3.x family sustains interleaved `<thinking>` blocks across agentic turns and sometimes emits `<tool_call>` before closing `</thinking>`. Stock and "enhanced" chat templates do **not** repair the missing close — they leave the broken assistant string in the serialized prompt, so the causal model conditions on unterminated reasoning. Downstream effects:

- **CoT leakage** — chain-of-thought bleeds into structured tool payloads
- **Ignored tools** — `tool_call` XML is dropped or never scheduled
- **Session death** — long agentic runs terminate prematurely on a single bad turn

This is a **template-structural issue, not a model or parser defect**. Because Qwen3.x keeps thinking blocks alive across turns, any template that doesn't heal the close produces prefix pollution that compounds over the session. It affects the 3.x series broadly, not just one checkpoint.

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

## Configuration

Use `qwen3_coder` as the tool-call parser on Qwen3.x: it fires on `<tool_call>` even with incomplete XML framing, whereas `qwen3_xml` can miss tool calls when `<thinking>` stays open. The full launch recipe, mixed-GPU/NCCL tuning, and FP8 alignment live in the repos below.

## Results

Validated end-to-end with long unsupervised agentic runs — a 180K-token session with `qwen3.5-enhanced.jinja` ([qwen36_27B_own_project](https://github.com/allanchan339/qwen36_27B_own_project)) and a 128K-token session with `qwen3.6-enhanced.jinja` + `preserve_thinking=true` ([qwen36_27B_36jinja_project](https://github.com/allanchan339/qwen36_27B_36jinja_project)), both with zero malformed tool calls.

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
- **Hugging Face collection** — [`nrrso/Qwen3-Chat-Template-vLLM-Fixes`](https://huggingface.co/nrrso/Qwen3-Chat-Template-vLLM-Fixes) republishes the fixed chat templates on the Hub, making them easy to pull for vLLM deployments without cloning the GitHub repo.

## Blog Posts

- [Qwen 3.6-35B-A3B on vLLM: do the Qwen 3.5 tool-calling fixes carry over?]({% post_url 2026-04-20-Qwen36-35B-A3B-tool-calling %}) — first test of 3.6 with the 3.5 stack
- [Qwen 3.6-27B-FP8 on vLLM: enhanced.jinja, qwen3_coder, and fixing NCCL after Studio Driver 595.79]({% post_url 2026-04-29-Qwen36-27B-tool-calling %}) — parser switch, NCCL fixes, 180K-token run
- [qwen3.6-enhanced.jinja: CoT leakage into tool turns and why preserve_thinking works now]({% post_url 2026-05-02-Qwen36-27B-updated-jinja %}) — the self-healing template fix
