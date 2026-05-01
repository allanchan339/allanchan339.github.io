---
layout: page
title: blogs
permalink: /blogs/
description:
nav: true
nav_order: 4
---

<div class="post">
{%- assign posts_count = site.posts | size -%}
{%- if posts_count > 0 -%}
  <ul class="post-list">
  {%- for post in site.posts -%}
    <li>
      <h3>
        <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h3>
      <p class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</p>
      {%- if post.description -%}
      <p>{{ post.description }}</p>
      {%- endif -%}
    </li>
  {%- endfor -%}
  </ul>
{%- else -%}
  <p>No posts yet. Add your first post in <code>_posts/</code>.</p>
{%- endif -%}
</div>
