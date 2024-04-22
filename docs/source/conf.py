# Sphinx configuration file
# See: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root directory to the system path
sys.path.insert(0, os.path.abspath('../..'))

# Project information

project = 'PynaMIT'
author = 'PynaMIT Developers'
copyright = '2024, PynaMIT Developers'

# General configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    ]

templates_path = ['_templates']
exclude_patterns = []

# Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Options for autodoc extension

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
    }

# Options for napoleon extension

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

# Options for viewcode extension

viewcode_follow_imported_members = True
viewcode_imported_members = True
viewcode_include_stdlib = False
viewcode_include_local = True
viewcode_include_modules = True

# Options for HTML output

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    }
