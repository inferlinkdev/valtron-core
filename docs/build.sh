#!/usr/bin/env bash
# Builds the Sphinx docs. Self-contained to this folder: manages its own uv
# environment here, resolving only what's declared in pyproject.toml, and
# points straight at ../src, with no `pip install` of valtron_core itself, no
# dependency on any other environment.
#
# Usage: ./build.sh [-b live]
#   ./build.sh          one-shot build -> _build/html/index.html
#   ./build.sh -b live  live-reloading dev server (sphinx-autobuild)
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "-b" ] && [ "${2:-}" = "live" ]; then
  uv sync --group dev -q
  uv run --group dev sphinx-autobuild . _build/html
else
  uv sync -q
  uv run sphinx-build -b html . _build/html
  echo ""
  echo "Docs built: $(pwd)/_build/html/index.html"
fi
