"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the
documentation:

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

# Import the checkout being documented without requiring an editable
# install.
sys.path.insert(0, os.path.abspath("../../src"))

# Set project information.
project = "PynaMIT"
copyright = "2024-2026, PynaMIT Developers"
author = "PynaMIT Developers"

# Set general configuration options.
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.viewcode", "myst_parser"]

templates_path = ["_templates"]
exclude_patterns = []

nitpicky = True
nitpick_ignore = [
    ("py:class", "InputPreparation"),
    ("py:class", "SimulationData"),
    ("py:class", "SimulationGeometry"),
    ("py:class", "ndarray"),
    ("py:class", "optional"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "kompe.SphericalGrid"),
    ("py:class", "kompe.basis.ScalarBasis"),
    ("py:class", "kompe.math.LeastSquaresProblem"),
    ("py:class", "kompe.SurfaceDifferentialBasis"),
    ("py:class", "kompe.spherical_transform.SphericalTransform"),
    ("py:class", "pynamit.storage.PersistentArrayCache"),
    ("py:class", "pynamit.storage.persistent_array_cache.PersistentArrayCache"),
    ("py:class", "pynamit.storage.artifact_store.ArtifactStore"),
    ("py:class", "pynamit.storage.field_time_series.FieldTimeSeries"),
    ("py:class", "pynamit.geomagnetism.MainField"),
    ("py:class", "pynamit.simulation.geometry.SimulationGeometry"),
    ("py:class", "pynamit.simulation.schema.SimulationSchema"),
    ("py:class", "xarray.core.dataarray.DataArray"),
    ("py:class", "xarray.core.dataset.Dataset"),
]

# Set options for HTML output.
html_theme = "sphinx_rtd_theme"

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
napoleon_use_param = False
napoleon_use_rtype = False

# Set options for myst_parser.
# https://myst-parser.readthedocs.io/en/latest/index.html

source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}
