# MONAI Physio Documentation

Sphinx source for the MONAI Physio documentation, published to GitHub Pages at
<https://project-monai.github.io/monai-physio/> by
`.github/workflows/docs.yml`. This file is excluded from the build; it is a
note for contributors editing the docs.

## Building locally

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. A clean build ends with
`build succeeded` and one pre-existing docutils warning from the
`RegisterImagesICON` docstring.

## Layout

| Path | Contents |
| --- | --- |
| `index.rst` | Landing page: hero, tutorial cards, topic grid, toctrees |
| `tutorials.rst` | The primary entry point for new users — one section per tutorial |
| `quickstart.rst`, `installation.rst` | Getting started |
| `cli_scripts/` | Task-oriented guides for the installed `monai-physio-*` commands |
| `api/` | Class and module reference, grouped by functionality |
| `developer/` | Architecture, extension points, conventions |
| `assets/` | Screenshots and GIFs referenced by `tutorials.rst` |
| `_static/custom.css` | NVIDIA-styled hero, cards and figure styling |

## Conventions

- New modules do **not** appear automatically. `conf.py` uses `autodoc` with
  hand-written `.rst` wrappers, so adding a class means adding or extending a
  page under `api/` and wiring it into the nearest `index.rst` toctree.
- Class references use `autoclass` under `.. currentmodule:: monai_physio`;
  module-level pages use `automodule`.
- `{{ mphysio_project_version }}` is substituted with the release at build time by
  the `source-read` hook in `conf.py`.
- Tutorial numbers appear in `index.rst` card anchors, `tutorials.rst` section
  titles, `quickstart.rst`, and `tutorials/README.md`. When a tutorial is
  renumbered, the card `href` values must be regenerated from the new section
  titles — Sphinx derives anchors from the heading text, and `linkcheck` does
  not catch same-page fragments.
