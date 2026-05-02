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

**Content preservation (substance):** Claims, metrics, hypotheses, caveats, appeals for correction, and narrative beats from the thread stay in the post unless the user asks to trim. **Presentation (form):** Retitle sections, merge blocks, and rewrite sentences so the piece reads like a technical blog—not like a forum thread.

## Voice and diction

- **First person** for the person publishing the blog (they wrote the original): prefer *I found*, *my setup*, *in my runs* — never *the author*.
- Prefer **connected paragraphs** over bullet-only sections when a less “checklist” feel is appropriate; keep bullets for real lists (steps, comparisons, configs).
- Replace vague collectives with concrete phrasing: *most tutorials*, *what you often see online*, *search results*, or naming the source.
- **Do not** use the word *write-ups* to describe this blog or its posts; prefer *posts*, *articles*, or *notes* (or neutral phrases like *this piece*) when a collective noun is needed.

### Thread idioms and engagement (usually remove or fold in)

Unless the user asks to keep a verbatim voice, **strip or neutralize** patterns that read as forum-native:

- **TL;DR** — fold into the opening paragraph; do not leave a labeled TL;DR block.
- **Edit:** / **Edit 2:** / **Update:** — merge into chronological or thematic flow.
- **Spoiler:**, emoji-heavy sign-offs, upvote bait, **“thoughts?”**, **“Anyone else?”**
- Rhetorical hooks used as paragraph openers: **“The catch?”**, **“Here’s the twist:”**, **“Instead, everything broke.”** — replace with direct statements of what happened.
- Direct-address memes to maintainers (**“Dear X dev…”**) — replace with a calm line that preserves the intent, e.g. *If this disagrees with upstream’s model of the component, I welcome a correction.*
- Thread-style closers (**“go build”**, **“Bottom line:”** as a punchy sign-off) — fold the substance into a short **Summary** or final paragraph without engagement-bait tone.
- Overuse of **scare quotes** and cute section titles (**“bug + bug = feature”**) — keep the technical explanation; rename the heading to something descriptive unless the phrase is genuinely useful as jargon.

### Target tone: blog, not Reddit

Aim for **neutral technical prose**: clear section headings (*Background*, *Mitigation*, *Results*), declarative sentences, and hypotheses labeled as such (**working hypothesis**, **in my runs**). The post should not sound like a scored comment; it should stand alone for readers who never open the thread.

## Structure

1. **Title & opening**: Clear blog `title` in front matter; first paragraph states scope and outcome (what broke, what worked) without thread labels.
2. **Body**: Headings that match a technical article (*Background*, *Cause*, *Fix*, *Results*, *Configuration*) rather than `# 1.` forum numbering or clicky phrasing.
3. **Evidence**: Preserve **all** metrics, tables, code fences, file names, version pins, and URLs. **Link** resources with markdown `[text](url)`; keep full URLs when the user cares about verbatim citation.
4. **Summary**: Prefer a short closing section that restates the numbered takeaways as **paragraphs** (or tight bullets) instead of a Reddit-style **Bottom line** + numbered list combo—unless the user wants that list format.
5. **Reddit-specific**: A **Resources** section may list the original thread(s) for provenance; the post should still read completely on its own.

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
- **Requests for fact-checking** — keep the substance in professional wording (see *Thread idioms* above).
- **Failures and dead ends** that teach (Attempt 1…7): keep as a structured section when they carry information density.

## Jekyll alignment (this repo)

If saving under `_posts/`:

- Add or normalize YAML: `layout`, `title`, `date` (from filename if standard), `description`, `tags`, `categories`.
- Pick **one** `categories` value from the project taxonomy (`research`, `journey`, `bug-fixes`, `reflection`); use **`bug-fixes`** for debugging / workaround posts unless the user specifies otherwise.
- After edits, run **`bundle exec jekyll build`** as in `posts_quality`; fix YAML or markdown that breaks the build.

## Markdown gotchas

- Angle-bracket tags in prose (e.g. XML-like names) can be eaten by Markdown/HTML parsers: wrap in **inline code** or split rare tag names with concatenation in source if the build strips substrings.
- Watch for `$...$` inline math: this repo expects `\\( ... \\)` per `posts_quality`. Prefer spelling out amounts or using **10k tokens** instead of `$10k` when `$` is currency, to avoid math-parser collisions.

## Quality check before handoff

- [ ] No third-person **author** distancing.
- [ ] No dropped numbers, links, code, or substantive claims from the source (unless user asked to shorten).
- [ ] Thread engagement patterns removed or neutralized; prose reads like a blog article, not a scored post.
- [ ] Front matter complete for Jekyll posts; category matches intent.
- [ ] Build succeeds if `posts_quality` applies.
