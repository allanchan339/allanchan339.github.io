<!-- this file compiles my notes about the installation and maintenance processes -->

# Installation

I ran into an issue with the ruby json package during the `bundle install` step. 
Fixed by installing ruby with the devkit following [these](https://stackoverflow.com/questions/64070241/install-ruby-development-tools-for-gem-installation-on-osx-catalina) instructions. 


# Maintenance

There are a few branches in this GitHub repo. The important one is `source`, which is the website served to the Internet.

Preview changes with `bundle exec jekyll serve`. This generates a local web server to live-preview changes to the source code.


After finished previewing, CLOSE THE SESSION. Then make any final changes on `source` and commit. 
Push to GitHub with `git push -u origin source` (not `origin main`!).


Then update page with `./bin/deploy --user`
The template will update the front-facing website using a GitHub Pages action.


# Content
Include teaching materials like Mathematica scripts for HS kids / old tutees.
link to CCFs published online
