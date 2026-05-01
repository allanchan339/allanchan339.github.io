# Personal website
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[maintainers]: https://img.shields.io/badge/maintainers-3-success.svg 'Number of maintainers'
<!-- ALL-CONTRIBUTORS-BADGE:END -->


I obtained this template from the [al-folio](https://github.com/alshedivat/al-folio) project.
Big thanks to the maintainers listed below for developing this beautiful site template and keeping it open-source.

## Build locally

This site is built with Jekyll and Bundler.

### Prerequisites

- `rbenv` installed
- Ruby `3.4.9` installed in `rbenv`
- Bundler gems installed from `Gemfile.lock`

### Setup (first time only)

```bash
eval "$(rbenv init - zsh)"
rbenv local 3.4.9
bundle install
```

### Build command

Use this command to build the static site into `_site/`:

```bash
eval "$(rbenv init - zsh)" && rbenv local 3.4.9 && bundle exec jekyll build
```

### Run local preview server

Use this command to run a local preview server:

```bash
eval "$(rbenv init - zsh)" && rbenv local 3.4.9 && bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

Then open [http://127.0.0.1:4000](http://127.0.0.1:4000).

### Maintainers

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://maruan.alshedivat.com"><br /><sub><b>Maruan</b></sub></a></td>
    <td align="center"><a href="http://rohandebsarkar.github.io"><br /><sub><b>Rohan Deb Sarkar</b></sub></a></td>
    <td align="center"><a href="https://amirpourmand.ir"><br /><sub><b>Amir Pourmand</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## License

The theme is available as open source under the terms of the [MIT License](https://github.com/alshedivat/al-folio/blob/master/LICENSE).

Originally, **al-folio** was based on the [\*folio theme](https://github.com/bogoli/-folio) (published by [Lia Bogoev](https://liabogoev.com) and under the MIT license).
Since then, it got a full re-write of the styles and many additional cool features.
