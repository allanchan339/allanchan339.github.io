---
name: fix-jekyll-inline-math
description: Fix inline LaTeX rendering and front matter consistency in Jekyll/al-folio markdown posts. Use when inline math appears as raw text or when post metadata differs from the repo's post format.
disable-model-invocation: true
---

# Fix Jekyll Inline Math

## Goal
Ensure inline math renders correctly in this repo and keep post front matter consistent with current post conventions.

## Project Rules

- **Critical safety checks (must not break):**
  - Keep valid YAML front matter delimiters exactly:
    - first line must be `---`
    - closing delimiter `---` must remain after metadata
  - Never remove or alter opening/closing front matter delimiters when editing metadata
- Inline math delimiter format for this repo is escaped:
  - use `\\( ... \\)` (double-backslash form in markdown source)
  - do **not** use `\(...\)` single-backslash form
- Inline math: use `\\( ... \\)` (not \( \) as markdown will not render correctly)
- Display math: keep existing format (`$$ ... $$` or `\[ ... \]`) if already rendering correctly
- Do not mass-convert display blocks
- Front matter should follow the current post format:
  - keep: `layout`, `title`, `date`, `description`, `tags`, `categories`
  - remove legacy/non-standard fields when aligning format (e.g. `mathjax: true` for this repo's current post style)

## Workflow

1. Open target post in `_posts/*.md`.
2. Check front matter format first:
   - align keys with current post convention used in recent posts
   - remove `mathjax: true` if present and format alignment is requested
   - verify both front matter delimiters `---` still exist after edits
3. Find inline formulas written as `$...$`.
4. Convert only inline delimiters:
   - `$...$` -> `\\(...\\)`
   - confirm converted text is literally `\\(...\\)` in file content
5. Do not modify display math blocks unless user asks.
6. Run pipeline validation (must complete end-to-end):
   - Build command:
     - `eval "$(rbenv init - zsh)" && rbenv local 3.4.9 && bundle exec jekyll build`
   - If build fails:
     - read the error carefully
     - fix the reported issue (YAML front matter, math delimiter typo, missing quote, etc.)
     - run the same build command again
     - repeat until build succeeds
7. After build succeeds, verify page rendering for each edited post:
   - derive the local page URL from filename date + slug, e.g. `_posts/2022-12-21-DDPM-Bayes.md` -> `http://127.0.0.1:4000/blog/2022/12/21/ddpm-bayes.html`
   - open/fetch the page and confirm:
     - inline math is rendered (not raw LaTeX text)
     - display math blocks render correctly
     - no broken formatting around equations
8. If rendering is still wrong:
   - fix markdown/math delimiters in source
   - rerun pipeline validation from step 6 until both build and render checks pass

## Validation Checklist

- [ ] Front matter matches project post style
- [ ] Opening and closing front matter `---` delimiters are intact
- [ ] `mathjax: true` removed when doing format-alignment cleanup
- [ ] No inline `$...$` remains in edited sections
- [ ] Inline math uses escaped delimiters `\\(...\\)` (double-backslash in source)
- [ ] Existing display math blocks are preserved
- [ ] Site build succeeds
- [ ] Corresponding local page(s) checked for render/display issues after successful build
- [ ] No raw LaTeX appears on checked page(s)

## Common Fix Patterns

### Front matter alignment
**Before**
```yaml
---
layout:       post
title:        "XXXX"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
comments: true
tags:
    - AI
    - Review
---
```

**After**
```yaml
---
layout: post
title: ...
date: 2022-12-21 00:00:00 +0800
description: ...
tags: [ai, review]
categories: [blog]
---
```

### Inline math conversion
**Before**
```markdown
We minimize $L(\theta)$ and estimate $p(x_t|x_0)$.
```

**After**
```markdown
We minimize \\(L(\theta)\\) and estimate \\(p(x_t|x_0)\\).
```

## Scope

Apply this skill to technical posts under `_posts/` when inline math is not rendering or when post metadata needs to match the repository's current post format.
