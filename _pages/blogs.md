---
layout: page
title: blogs
permalink: /blogs/
description:
nav: true
nav_order: 3
display_categories: [research, journey, bug-fixes, reflection]
---

<div class="post">
  <style>
    .blogs-layout {
      display: flex;
      gap: 2rem;
    }

    .blogs-sidebar {
      flex: 0 0 260px;
      max-width: 260px;
    }

    .blogs-sidebar .sticky-wrap {
      position: sticky;
      top: 96px;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
      border: 1px solid var(--global-divider-color);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      background-color: var(--global-bg-color);
    }

    .blogs-sidebar h4 {
      margin: 0 0 0.5rem 0;
      font-size: 1rem;
    }

    .blogs-sidebar ul {
      list-style: none;
      padding-left: 0;
      margin-bottom: 0;
    }

    .blogs-sidebar li {
      margin: 0.25rem 0;
    }

    .blogs-sidebar .post-links {
      padding-left: 0.8rem;
      border-left: 1px solid var(--global-divider-color);
      margin-left: 0.25rem;
      margin-top: 0.25rem;
    }

    .blogs-content {
      flex: 1 1 auto;
      min-width: 0;
    }

    @media (max-width: 992px) {
      .blogs-layout {
        flex-direction: column;
      }

      .blogs-sidebar {
        max-width: 100%;
      }

      .blogs-sidebar .sticky-wrap {
        position: static;
        max-height: none;
      }
    }
  </style>
{%- assign posts_count = site.posts | size -%}
{%- if posts_count > 0 -%}
  <div class="blogs-layout">
    {%- if page.display_categories %}
    <aside class="blogs-sidebar" aria-label="Blogs navigation">
      <nav class="sticky-wrap">
        <h4>Quick Navigation</h4>
        <ul>
          {%- for category in page.display_categories %}
          {%- assign categorized_posts = site.posts | where_exp: "post", "post.categories contains category" -%}
          {%- assign category_words = category | replace: "-", " " | split: " " -%}
          {%- capture category_label -%}{%- for word in category_words -%}{{ word | capitalize }}{% unless forloop.last %} {% endunless %}{%- endfor -%}{%- endcapture -%}
          {%- if categorized_posts.size > 0 -%}
          <li>
            <a href="#category-{{ category | slugify }}">{{ category_label | strip }}</a>
            <ul class="post-links">
              {%- for post in categorized_posts -%}
              <li><a href="#post-{{ post.title | slugify }}">{{ post.title }}</a></li>
              {%- endfor -%}
            </ul>
          </li>
          {%- endif -%}
          {%- endfor -%}
        </ul>
      </nav>
    </aside>
    {%- endif -%}
    <section class="blogs-content">
  {%- if page.display_categories %}
  {%- for category in page.display_categories %}
  {%- assign categorized_posts = site.posts | where_exp: "post", "post.categories contains category" -%}
  {%- assign category_words = category | replace: "-", " " | split: " " -%}
  {%- capture category_label -%}{%- for word in category_words -%}{{ word | capitalize }}{% unless forloop.last %} {% endunless %}{%- endfor -%}{%- endcapture -%}
  {%- if categorized_posts.size > 0 -%}
  <h2 id="category-{{ category | slugify }}" class="category">{{ category_label | strip }}</h2>
  <ul class="post-list">
    {%- for post in categorized_posts -%}
      <li id="post-{{ post.title | slugify }}">
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
  {%- endif -%}
  {%- endfor -%}
  {%- else -%}
  <ul class="post-list">
    {%- for post in site.posts -%}
      <li id="post-{{ post.title | slugify }}">
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
  {%- endif -%}
    </section>
  </div>
{%- else -%}
  <p>No posts yet. Add your first post in <code>_posts/</code>.</p>
{%- endif -%}
</div>
