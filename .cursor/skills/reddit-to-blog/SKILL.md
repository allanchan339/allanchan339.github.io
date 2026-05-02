---
name: reddit-to-blog
description: >-
  Converts Reddit-style or thread-style drafts into long-form blog posts: voice,
  structure, front matter, and information preservation. Use when the user wants
  to rewrite a Reddit post, thread notes, or "TL;DR" style content into a blog
  article; when moving content from r/ or social threads to _posts/; or when
  they mention reddit-to-blog, thread-to-blog, or de-Reddit a draft for the blog.
---

# Reddit → blog post

## Goal

Turn **thread-native** prose (Reddit, forum, or chat-export style) into a **cohesive blog article** without dropping facts, numbers, links, commands, or tables. The author speaks as **themselves** (**I / me / my**), not as “the author.”

When the output lives in this repo as Jekyll markdown, **also follow** [.cursor/skills/posts_quality/SKILL.md](../posts_quality/SKILL.md) (front matter, `categories`, build).

Information and fact in original reddit thread must be preserved in the blog post. What is expected to change is tone, style, and structure to make it a coherent blog article.

## Voice and diction

- **First person** for the person publishing the blog (they wrote the original): prefer *I found*, *my setup*, *in my runs* — never *the author*.
- Prefer **connected paragraphs** over bullet-only sections when the user wants a less “checklist” feel; keep bullets for real lists (steps, comparisons, configs).
- Avoid thread idioms unless quoting: **TL;DR** (fold into a short intro), **Edit:** / **Edit 2:** (merge into the narrative), **Spoiler:**, engagement closers (**Happy … 🚀**), **this** (overused emphasis).
- Replace vague collectives with concrete phrasing: *most tutorials*, *what you often see online*, *search results*, or naming the source.
- **Do not** use the word *write-ups* to describe this blog or its posts; prefer *posts*, *articles*, or *notes* (or neutral phrases like *this piece*) when a collective noun is needed.

## Structure

1. **Title & hook**: Clear blog title; opening paragraph states scope and outcome (what broke, what worked).
2. **Body**: Use headings that match the site’s tone (technical posts: *Problem / Cause / Fix* or *Symptoms / What happened / Mitigation*; narrative posts: chronological or thematic sections).
3. **Evidence**: Preserve **all** metrics, tables, code fences, file names, version pins, and URLs. **Link** resources with markdown `[text](url)`; keep full URLs when the user cares about verbatim citation.
4. **Reddit-specific**: A **Resources** or **Links** section may keep the original thread URL for provenance; the post should stand alone without requiring readers to open Reddit.

## Remove or rewrite (typical)

| Thread pattern | Blog-style approach |
| -------------- | ------------------- |
| `---` rule separators between every block | Section headings or short transitions |
| “Am I the only one?” / “Anyone else?” | Delete or one neutral sentence on prevalence |
| Upvote bait / “thoughts?” | Delete |
| Inline “sorry for English” | Delete unless material |
| Screenshots described as “see img” | If no image asset, describe in text or omit |

## Preserve (non-negotiable)

- Every **command**, **env var**, **config JSON**, **error string**, **issue #**, **model id**, **hardware detail**, and **benchmark number** from the source unless the user asks to trim.
- **Hypotheses and caveats** (*maybe X*, *I did not verify Y*) — rephrase clearly; do not silently delete uncertainty.
- **Failures and dead ends** that teach (Attempt 1…7): keep as a structured section when they carry information density.

## Jekyll alignment (this repo)

If saving under `_posts/`:

- Add or normalize YAML: `layout`, `title`, `date` (from filename if standard), `description`, `tags`, `categories`.
- Pick **one** `categories` value from the project taxonomy (`research`, `journey`, `bug-fixes`, `reflection`); use **`bug-fixes`** for debugging / workaround posts unless the user specifies otherwise.
- After edits, run **`bundle exec jekyll build`** as in `posts_quality`; fix YAML or markdown that breaks the build.

## Markdown gotchas

- Angle-bracket tags in prose (e.g. XML-like names) can be eaten by Markdown/HTML parsers: wrap in **inline code** or split rare tag names with concatenation in source if the build strips substrings.
- Watch for `$...$` inline math: this repo expects `\\( ... \\)` per `posts_quality`.

## Quality check before handoff

- [ ] No third-person **author** distancing.
- [ ] No dropped numbers, links, or code from the source (unless user asked to shorten).
- [ ] Front matter complete for Jekyll posts; category matches intent.
- [ ] Build succeeds if `posts_quality` applies.
