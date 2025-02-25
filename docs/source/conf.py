"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the
documentation:

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

# Add the project root directory to the system path.
sys.path.insert(0, os.path.abspath("../../"))

# Set project information.
project = "PynaMIT"
copyright = "2024, PynaMIT Developers"
author = "PynaMIT Developers"

# Set general configuration options.
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.viewcode", "myst_parser"]

templates_path = ["_templates"]
exclude_patterns = []

nitpicky = True

# Set options for HTML output.
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Set options for autodoc.
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# Set options for napoleon.
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False

# Set otions for myst_parser.
# https://myst-parser.readthedocs.io/en/latest/index.html

source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}
