# Blog Migration TODO

## 1) Identity and Core Config
- [x] Update `_config.yml` name/email/url/social handles.
- [x] Extract and set `scholar_userid` correctly.
- [x] Rename `projects` page title/permalink to `blog` (`/blog/`).
- [ ] Optional: set `title` in `_config.yml` (currently `blank`).
- [ ] Optional: set `keywords` in `_config.yml`.

## 2) About + CV (owner edits)
- [ ] Rewrite `_pages/about.md` with personal bio and links.
- [ ] Update `_pages/cv.md` to your own CV filename.
- [ ] Replace old CV PDF in `assets/pdf/` with your file.
- [ ] Replace profile image in `assets/img/prof_pic.jpg` (if needed).

## 3) Content Cleanup (template -> yours)
- [ ] Review and replace/remove all `_news/*.md` sample announcements.
- [ ] Review and replace/remove all `_projects/*.md` old content.
- [ ] Remove placeholder/sample images not used anymore in `assets/img/`.

## 4) Decide Blog Structure
- [ ] Decide whether `/blog/` should use:
  - [ ] Real posts in `_posts/` (recommended for blog flow), or
  - [ ] Existing card-style entries from `_projects/`.
- [ ] If using `_posts/`, create first 1-2 posts and adjust nav/page wiring.

## 5) Publications (next round)
- [ ] Replace `_bibliography/papers.bib` sample entries.
- [ ] Update `_data/coauthors.yml` (if publication linking is needed).
- [ ] Update `_data/venues.yml` (if venue badges/colors are used).
- [ ] Update Scholar-related display details as needed.

## 6) Git + Publish
- [ ] Add your new remote:
  - `git remote add origin https://github.com/<your-username>/<your-repo>.git`
- [ ] Ensure `.gitignore` excludes build output and cache (especially `_site/`).
- [ ] Create first clean commit.
- [ ] Push to GitHub.
- [ ] Enable GitHub Pages and verify site loads at your URL.

## 7) Final Quality Check
- [ ] Run local preview and click through all nav pages.
- [ ] Check for old owner name/email/links still visible.
- [ ] Check broken images and broken links.
- [ ] Verify mobile layout and dark mode.

