---
name: fix-jekyll-inline-math
description: Fix inline LaTeX rendering and front matter consistency in Jekyll/al-folio markdown posts. Use when inline math appears as raw text or when post metadata differs from the repo's post format.
disable-model-invocation: true
---

# Fix Jekyll Inline Math

## Goal
Ensure inline math renders correctly in this repo and keep post front matter consistent with current post conventions.

## Project Rules

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
3. Find inline formulas written as `$...$`.
4. Convert only inline delimiters:
   - `$...$` -> `\\(...\\)`
5. Do not modify display math blocks unless user asks.
6. Rebuild and verify:
   - `bundle exec jekyll build`
   - confirm build succeeds and inline formulas render as math.

## Validation Checklist

- [ ] Front matter matches project post style
- [ ] `mathjax: true` removed when doing format-alignment cleanup
- [ ] No inline `$...$` remains in edited sections
- [ ] Inline math uses `\\(...\\)`
- [ ] Existing display math blocks are preserved
- [ ] Site build succeeds

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
