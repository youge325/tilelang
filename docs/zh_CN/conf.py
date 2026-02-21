# Chinese (zh_CN) documentation configuration
# This is a self-contained Sphinx conf.py for the Chinese translation.

import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_docs_root = os.path.dirname(_this_dir)
_repo_root = os.path.dirname(_docs_root)

# ── Project ──────────────────────────────────────────────────────────────
project = "TileLang <br>"
author = "Tile Lang Contributors"
copyright = f"2025-2025, {author}"

with open(os.path.join(_repo_root, "VERSION")) as f:
    version = f.read().strip()
release = version

# ── Extensions ───────────────────────────────────────────────────────────
extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_reredirects",
    "sphinx.ext.mathjax",
    "myst_parser",
    "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = [os.path.join(_repo_root, "tilelang")]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_keep_files = False
autoapi_generate_api_docs = True
autodoc_typehints = "description"
autoapi_ignore = ["*language/ast*", "*version*", "*libinfo*", "*parser*"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["colon_fence", "deflist"]

redirects = {"get_started/try_out": "../index.html#getting-started"}

language = "zh_CN"

exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "README.md",
    "**/*libinfo*", "**/*version*",
]

pygments_style = "sphinx"
todo_include_todos = False

# ── HTML Output ──────────────────────────────────────────────────────────
html_theme = "furo"
templates_path = [os.path.join(_docs_root, "_templates")]
html_static_path = [os.path.join(_docs_root, "_static")]
html_css_files = ["custom.css"]
html_js_files = ["lang_switcher.js"]

footer_copyright = "© 2025-2026 TileLang"
footer_note = " "

html_theme_options = {
    "light_logo": "img/logo-v2.png",
    "dark_logo": "img/logo-v2.png",
    "announcement": (
        '<div id="lang-switcher">\U0001f310 '
        '<a data-lang="en" href="#">English</a> | '
        '<strong>\u4e2d\u6587</strong></div>'
    ),
}

header_links = [
    ("Home", "https://github.com/tile-ai/tilelang"),
    ("Github", "https://github.com/tile-ai/tilelang"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "tile-ai",
    "github_repo": "tilelang",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
    "current_language": "zh_CN",
}
