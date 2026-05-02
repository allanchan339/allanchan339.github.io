---
layout: post
title: "Claude Code with local vLLM: client validation, model aliases, and a working settings.json"
date: 2026-04-19 00:00:00 +0800
description: "Run Claude Code against local vLLM without Anthropic API access: why common env-only recipes fail, the alias + settings.json pattern that works, and when this matters if you cannot register or use the Claude API."
tags: [claude-code, vllm, llm, local-inference, anthropic-api]
categories: [bug-fixes]
---

**Straight story:** I wanted Claude Code to talk to **my own** model on **vLLM**, not to Anthropic’s hosted API. Tutorials usually say: set `ANTHROPIC_CUSTOM_MODEL_OPTION` and `ANTHROPIC_BASE_URL`. **That was not enough.** The CLI applies **its own checks** and can fail with “issue with the selected model” **before** meaningful traffic hits your server. The fix is a small set of aligned settings: **tier aliases** (`"model": "sonnet"` + `ANTHROPIC_DEFAULT_*_MODEL`), a **root** base URL (no extra `/v1`), **`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`**, and a **dummy** `ANTHROPIC_AUTH_TOKEN` so vLLM still gets a header.

**Why I care (and maybe you do too):** with **`ANTHROPIC_BASE_URL`** pointed at **local vLLM** and a **placeholder** token, **model traffic does not use the real Claude API**—no Anthropic API key or inference billing for that path. That matters if you **cannot register**, are **out of region**, or **cannot obtain API access**, but still want the **Claude Code** loop against a model you control. (You still install and run **Claude Code**; this is about **where completions are served**, not a different product.)

**vLLM / Qwen:** Tooling and template notes for Qwen 3.5 on vLLM are in this [vLLM / Qwen 3.5 thread](https://www.reddit.com/r/Vllm/comments/1skks8n/qwen_35_27b35ba3b_tool_calling_issues_why_it/). Below assumes **vLLM is already up** and passes a simple `curl` check.

# Baseline: vLLM responds, Claude Code does not (yet)

I run **Qwen 3.5-27B** behind vLLM. Direct HTTP calls succeed:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -X POST \
  -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"test"}]}'
# Works
```

So I expected a quick env change. Instead I iterated through docs and issues, then grepped **`cli.js`** to see why validation fired.

# The trap: `ANTHROPIC_CUSTOM_MODEL_OPTION`

## What the official docs suggest

The [Claude Code model configuration docs](https://docs.anthropic.com/en/docs/claude-code/model-config) describe `ANTHROPIC_CUSTOM_MODEL_OPTION` as a way to add a custom entry to the `/model` picker and imply relaxed handling for that id.

I tried:

```json
{
  "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-27B",
  "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000"
}
```

**Observed error:** `There's an issue with the selected model (Qwen3.5-27B). It may not exist or you may not have access to it.`

## What actually happens

The variable **does** add a picker entry, but it **does not** reliably bypass validation when you drive the CLI via **`--model`**, **`settings.json`**, or similar. In practice you still hit the same guardrails unless you adopt the alias + env pattern later in this note.

This behavior shows up in community threads—for example GitHub issues **#18025**, **#23266**, and **#34821**—while the product docs have not caught up.

**Takeaway:** when the documented env var does not match runtime behavior, the implementation (not the blog post) is the source of truth.

# What I learned from `cli.js`

I stopped relying on tutorials and searched the installed **`cli.js`** (on the order of **~50k** lines, minified) for the error string:

```bash
grep -n "There's an issue with the selected model" ~/.nvm/versions/node/*/lib/node_modules/@anthropic-ai/claude-code/cli.js
```

The hit landed near **line 5146**. The logic, paraphrased from the minified source, is:

```javascript
if (q instanceof AnthropicError && q.status === 404) {
  // Reject custom models on 404
  return {
    content: `There's an issue with the selected model (${K}). 
              It may not exist or you may not have access to it.`,
    error: "invalid_request"
  }
}
```

So the CLI issues **validation-style requests**, gets **404** responses when the id is not on Anthropic’s expected list, and **returns the “selected model” error before** the path you care about (your vLLM **`/v1/messages`** traffic) is exercised normally.

That is **client-side validation**, not “your server returned 404 on chat.”

## The undocumented lever that matters

Experimenting with env vars, **`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`** consistently reduced the failure mode where the client keeps probing endpoints that will never acknowledge a local model id. I did not find this called out in the same place as the high-level “custom model” docs; it is nonetheless **necessary** for a stable loop in my setup.

# Working `~/.claude/settings.json` (tested here, not copy-pasted blind)

```json
{
  "model": "sonnet",
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "Qwen3.5-27B",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "Qwen3.5-27B",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Qwen3.5-27B",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  }
}
```

## The settings that must agree (if any drift, you get confusing errors)

| Setting | Why it matters | Typical failure if wrong |
| ------- | -------------- | ------------------------- |
| `"model": "sonnet"` **and** `ANTHROPIC_DEFAULT_SONNET_MODEL` | Claude resolves the **alias** “sonnet” to your real vLLM id; putting the **custom id** directly in `"model"` triggers list validation | “Issue with the selected model” |
| `ANTHROPIC_BASE_URL` is **`http://127.0.0.1:8000`** (no `/v1`) | The client appends **`/v1/messages`** itself; a base URL that already ends in `/v1` becomes **`/v1/v1/messages`** | **404** on API calls |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`: `"1"` | Cuts non-essential / validation traffic that assumes Anthropic-hosted models | **Intermittent** validation failures |
| `ANTHROPIC_AUTH_TOKEN` (e.g. `"dummy"`) plus aligned **`ANTHROPIC_DEFAULT_*_MODEL`** for **Opus / Sonnet / Haiku** | vLLM still expects an Authorization-shaped header; mapping **all three** tiers to the same served name avoids internal tier switches pointing at invalid ids | Auth or “wrong model” surprises when the CLI switches tier |

## vLLM side (must match the JSON exactly)

- **`--served-model-name Qwen3.5-27B`** must match the strings in **`ANTHROPIC_DEFAULT_*_MODEL`** **character for character**.
- Avoid **`/`** in the served name if your settings use a flat id (a **`Qwen/...`** vs **`Qwen3.5-27B`** mismatch broke one of my attempts).
- Server should listen where **`ANTHROPIC_BASE_URL`** points (here **`8000`**).

## Smoke test

```bash
claude "test"
# Expect a normal assistant reply, e.g. readiness to help.
```

If this fails, reconcile the table above **in order** before chasing unrelated flags.

# Debugging sequence (short)

If something below matches your error, fix that first; the full **`settings.json`** block is the target state.

**Attempt 1 — vLLM-style base URL with `/v1`**

```json
"ANTHROPIC_BASE_URL": "http://127.0.0.1:8000/v1"
```

**Error:** `API Error: 404` — the client adds `/v1/messages` again.

---

**Attempt 2 — custom id in `"model"` (GitHub #18025-style reports)**

```json
"model": "Qwen3.5-27B"
```

**Error:** `There's an issue with the selected model` — no alias mapping; Anthropic list validation wins.

---

**Attempt 3 — slash in `--served-model-name` vs settings**

```bash
--served-model-name Qwen/Qwen3.5-27B
```

vs settings expecting `Qwen3.5-27B` without `/`.

**Error:** model not found / mismatch.

---

**Attempt 4 — `ANTHROPIC_CUSTOM_MODEL_OPTION` only (official wording)**

```json
{
  "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-27B",
  "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000"
}
```

**Error:** still validation errors — picker entry ≠ full bypass for **`settings.json`** flows.

---

**Attempt 5 — `ANTHROPIC_API_KEY` instead of token**

```json
"ANTHROPIC_API_KEY": "dummy"
```

**Error:** authentication friction — **`ANTHROPIC_AUTH_TOKEN`** behaved better with vLLM in my tests.

---

**Attempt 6 — correct URL and aliases but no traffic / validation flag**

```json
// CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC omitted
```

**Error:** **intermittent** validation failures — sometimes works, sometimes not.

---

**Attempt 7 — minimal working core (before I added timeout / attribution / all three tiers)**

```json
{
  "model": "sonnet",
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "Qwen3.5-27B",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

**Result:** **stable enough to proceed**; I then expanded to the full block at the top (Opus/Haiku defaults, long **`API_TIMEOUT_MS`**, **`CLAUDE_CODE_ATTRIBUTION_HEADER`**) for day-to-day use.

# Why the alias + base URL + flag pattern works

## Model tiers and aliases

Claude Code still thinks in **Opus / Sonnet / Haiku** tiers. If **`"model": "sonnet"`**, the runtime resolves that label via **`ANTHROPIC_DEFAULT_SONNET_MODEL`**. If you put **`"model": "Qwen3.5-27B"`** directly, the CLI tries to treat it like an Anthropic-hosted id and **fails validation**.

```json
"model": "sonnet"
"ANTHROPIC_DEFAULT_SONNET_MODEL": "Qwen3.5-27B"
```

## URL construction

The client builds:

```text
{ANTHROPIC_BASE_URL}/v1/messages
```

So **`ANTHROPIC_BASE_URL=http://127.0.0.1:8000/v1`** becomes:

```text
http://127.0.0.1:8000/v1/v1/messages
```

which **404s**. The base should stop at the host (and port), e.g. **`http://127.0.0.1:8000`**.

## `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`

With **`=1`**, the CLI skips **some** checks and ancillary calls that assume Anthropic’s catalog. For **local** ids, those checks are exactly where **404 → “invalid model”** loops come from. **Without** the flag I still saw **sporadic** failures even when aliases and URLs were otherwise correct.

# Common errors (quick map)

| Symptom | Likely cause |
| ------- | ------------ |
| “There's an issue with the selected model” | Custom string in **`"model"`** without alias mapping |
| `API Error: 404` | **`ANTHROPIC_BASE_URL`** includes **`/v1`** |
| Model not found | **`--served-model-name`** does not match JSON, or contains **`/`** when settings do not |
| Intermittent validation | Missing **`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`** |
| Relying on **`ANTHROPIC_CUSTOM_MODEL_OPTION` alone** | Docs oversell bypass; **picker ≠ full CLI bypass** |

# Pre-flight checklist

Before invoking `claude`:

- **`"model"`** is an **alias** such as **`sonnet`**, not the vLLM id.
- **`ANTHROPIC_DEFAULT_SONNET_MODEL`** (and siblings if you use tier changes) points at the served name.
- **`ANTHROPIC_BASE_URL`** ends at **`...:8000`** with **no** trailing **`/v1`**.
- **`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`** is **`"1"`**.
- **`--served-model-name`** matches the JSON **exactly** (no stray **`/`**).
- vLLM is up and reachable at that host/port.
- Do **not** treat **`ANTHROPIC_CUSTOM_MODEL_OPTION`** as sufficient on its own.

# Summary

1. **Accessibility:** Pointing Claude Code at **local vLLM** means **Anthropic API access is not required for the model layer**—useful when you **cannot register**, **cannot get API keys**, or want **zero** hosted inference spend. You still use the CLI; completions hit **your** server.
2. **`ANTHROPIC_CUSTOM_MODEL_OPTION`** alone did **not** match what I needed; treat tier **aliases** + env as the real fix.
3. **`"model": "sonnet"`** (or another tier label) plus **`ANTHROPIC_DEFAULT_*_MODEL`** → your **`Qwen3.5-27B`** (or served name).
4. **`ANTHROPIC_BASE_URL`** stops at **`http://host:port`**; the client adds **`/v1/messages`**.
5. **`--served-model-name`** matches those env strings **exactly** (watch **`/`** in ids).
6. **`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`** avoids **intermittent** validation against Anthropic’s catalog.
7. **`ANTHROPIC_AUTH_TOKEN`** (e.g. **`dummy`**) worked better than **`ANTHROPIC_API_KEY`** with my vLLM.
8. When docs and **`cli.js`** disagree, the bundle wins; most bad copy-pastes omit **one** of alias mapping, base URL shape, or the traffic flag.

# Resources

- [ForgeBookAuto — Claude Code third-party models (quick reference)](https://github.com/allanchan339/ForgeBookAuto/blob/main/docs/claude-code-third-party-models.md)
- [BigModel docs — coding plan / Claude (working third-party pattern)](https://docs.bigmodel.cn/cn/coding-plan/tool/claude)
- [vLLM docs — Claude Code integration](https://docs.vllm.ai/en/latest/serving/integrations/claude_code/) (useful but incomplete versus real client behavior)
- Related GitHub issues: **#18025**, **#23266**, **#34821**

If you want **Claude Code’s workflow** without **Claude API** inference, start from the **`settings.json`** block and checklist: **aliases**, **root base URL**, **`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`**, then align vLLM’s served name. That order saves time versus chasing **`ANTHROPIC_CUSTOM_MODEL_OPTION`** alone.
