# Valtron Docs

Sphinx documentation source for Valtron, deployed to
[valtron.ai/docs](https://valtron.ai/docs). Self-contained to this folder: it
manages its own `uv` environment here and points straight at `../src`, with
no `pip install` of `valtron_core` itself, so it always reflects the current
working tree.

## Build

```bash
./build.sh
```

Then open `_build/html/index.html`.

## Live-reloading dev server

```bash
./build.sh -b live
```

Watches for file changes and rebuilds automatically in the browser.

## Adding a new API page

API pages under `api/` are thin stubs — Sphinx's `autodoc` renders the actual
class documentation live from docstrings in `../src/valtron_core/` at build
time, so editing a class's fields or docstring needs no action here. Adding a
**new** public class isn't automatic, though: it needs a small `.rst` stub
and a toctree entry.

1. Add a stub file, e.g. `api/my_new_class.rst`, modeled on an existing one
   such as [`api/classification_config.rst`](api/classification_config.rst):

   ```rst
   MyNewClass
   ==========

   .. autoclass:: valtron_core.some.module.MyNewClass
      :members:
      :show-inheritance:
   ```

2. Add one line linking that stub to the relevant module's toctree page
   (e.g. `api/evaluation.md`) and, if useful, a row in the searchable table
   in `api/index.md`.
