---
name: pre-commit-quality
description: >-
  Pre-commit quality checks: Prettier formatting, Jekyll build, and
  local verification to prevent CI failures on push.
---

# Pre-Commit Quality Checks

Run these checks **before every commit and push** to avoid CI failures.

## Checklist

- [ ] Prettier formatting
- [ ] Jekyll build (no YAML/Liquid/SCSS errors)
- [ ] (Optional) `git status` to confirm no stray files

## 1. Prettier

```bash
npx prettier . --write
```

Check for any issues:

```bash
npx prettier . --check
```

Note: `.gitignore` has no parser — exclude it if it triggers a warning.

## 2. Jekyll Build

```bash
bundle exec jekyll build 2>&1 | tail -5
```

Look for:
- `done in X seconds` → build succeeded
- `Error:` or `warning:` → fix before committing

## 3. Last-Minute Review

```bash
git status
git diff --stat
```

Ensure only intended files are staged.

## What CI Runs on Push (for reference)

| Check | How to verify locally |
|---|---|
| Prettier | `npx prettier . --check` |
| Jekyll build + deploy | `bundle exec jekyll build` |
| CodeQL (JS/TS, Ruby) | Cannot run locally; CI-only |
| Broken links | Cannot run locally; CI-only |
