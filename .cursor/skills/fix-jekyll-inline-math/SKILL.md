---
name: jekyll-post-quality-workflow
description: End-to-end Jekyll post quality workflow: align front matter format, normalize math notation, run build, verify render, and iteratively fix issues until the page looks correct.
disable-model-invocation: true
---

# Jekyll Post Quality Workflow

## Goal
Ensure each edited post follows current metadata conventions, renders math correctly, builds successfully, and visually looks correct in the generated page.

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
- Front matter must be aligned before math checks and should follow the current post format:
  - keep: `layout`, `title`, `date`, `description`, `tags`, `categories`
  - expected style example:
    - `layout: post`
    - `title: "Code Review: Denoising Diffusion Probabilistic Models (DDPM)"`
    - `date: 2023-01-10 00:00:00 +0800`
    - `description: ...`
    - `tags: [ai, review, diffusion, generation]`
    - `categories: [blog]`
  - remove legacy/non-standard fields when aligning format (e.g. `mathjax: true` for this repo's current post style)

## Workflow

1. Open target post in `_posts/*.md`.
2. **Align front matter first**:
   - align keys with current post convention used in recent posts
   - remove `mathjax: true` if present and format alignment is requested
   - remove legacy fields not used in current style (e.g. `author`, `header-style`, `catalog`, `comments`) when doing alignment cleanup
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
9. Continue iterating until the page is visually correct and stable (no raw LaTeX, no broken equations, no malformed front matter).

## Validation Checklist

- [ ] Front matter matches project post style
- [ ] Opening and closing front matter `---` delimiters are intact
- [ ] `mathjax: true` removed when doing format-alignment cleanup
- [ ] Legacy metadata fields removed when aligning to current post style
- [ ] No inline `$...$` remains in edited sections
- [ ] Inline math uses escaped delimiters `\\(...\\)` (double-backslash in source)
- [ ] Existing display math blocks are preserved
- [ ] Site build succeeds
- [ ] Corresponding local page(s) checked for render/display issues after successful build
- [ ] No raw LaTeX appears on checked page(s)
- [ ] Repeat fixes/build/render checks until page quality is acceptable

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

Apply this skill to technical posts under `_posts/` whenever post quality updates are needed, including front matter alignment, math-notation normalization, build validation, render checks, and iterative bug fixing.
