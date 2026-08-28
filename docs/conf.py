import os
import sys

# Points straight at the sibling src/ tree in this same repo, meaning no pip
# install of valtron_core itself, so this always reflects the current
# working tree, not whatever happens to be installed somewhere else.
sys.path.insert(0, os.path.abspath("../src"))

project = "Valtron"
copyright = "2026 InferLink. ALL SYSTEMS NOMINAL."
author = "InferLink"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

# Copy button strips shell prompts and output markers so users only copy the
# runnable command, not the leading "$" or a ">>> " REPL prompt.
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
# Auto-generates a #slug anchor for every heading up to this depth, matching
# how Docusaurus auto-anchors headings; without this, cross-file links to
# `#some-heading` (used throughout these pages) don't resolve.
myst_heading_anchors = 4
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# TransformerClassifier/BERTTrainer/BERTEvaluator import torch/transformers/
# sklearn/datasets. Mocking keeps this self-contained: autodoc still imports
# and introspects the real class, it just doesn't need the multi-GB ML
# stack installed purely to render docs.
autodoc_mock_imports = ["torch", "transformers", "datasets", "sklearn"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", ".venv", "README.md", "Thumbs.db", ".DS_Store"]

# Install and Configuration Wizard are single flat pages with no children, so
# their "Section Navigation" primary sidebar entry is always empty, so hide
# the sidebar on them entirely.
html_sidebars = {
    "getting-started/installation": [],
    "getting-started/configuration-wizard": [],
}

html_theme = "pydata_sphinx_theme"
html_title = "Valtron"
html_static_path = ["_static"]
html_css_files = ["valtron.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "logo": {
        "text": "VALTRON",
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": [],
    # Real nesting (module landing pages with their own sub-toctree) makes the
    # API sidebar sections collapsible; this starts them collapsed except the
    # active branch instead of showing every class at once.
    "collapse_navigation": True,
    # No "On this page" secondary sidebar on the landing page, since it's a set of
    # cards, not a long article, so the content should be centered instead.
    # Order matters: the theme applies whichever pattern matches *last*, so
    # the specific "index" override must come after the "**" wildcard.
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
        "index": [],
    },
    # Closer to the old Docusaurus site's Prism themes (github light / vs dark).
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
}
