"""
PynaMIT: A Python package for dynamic MIT coupling simulations.

This package includes modules for simulation, visualization, and various
utilities.

Attributes
----------
BasisEvaluator : class
    Class for evaluating basis functions.
CSBasis : class
    Class for cubed sphere projections.
Dynamics : class
    Class for simulating ionospheric dynamics.
FieldEvaluator : class
    Class for evaluating fields.
Grid : class
    Class for grid management.
Mainfield : class
    Class for main field evaluation.
PynamEye : class
    Class for visualization.
SHBasis : class
    Class for spherical harmonics basis functions.
FieldExpansion : class
    Class for storing vector data and defining vector operations.
debugplot : function
    Function for debug plotting.
globalplot : function
    Function for global plotting.
"""

from .sphere import (
    Basis,
    BasisView,
    CSBasis,
    Grid,
    GridBasis,
    RadialLaplaceContinuation,
    SHBasis,
    SurfaceOperators,
    basis_kind,
    is_basis_kind,
    is_cs_basis,
    is_grid_basis,
    is_sh_basis,
    normalize_horizontal_basis_kind,
)
from .primitives.basis_evaluator import BasisEvaluator
from .primitives.field_evaluator import FieldEvaluator
from .primitives.field_expansion import FieldExpansion
from .simulation.dynamics import Dynamics
from .simulation.input_vs_interpolated import plot_input_vs_interpolated
from .simulation.mainfield import Mainfield
from .simulation.pynameye import PynamEye
from .simulation.visualization import debugplot, globalplot
from .math import set_backend, use_jax
from .external_inputs import set_input_source, get_input_source

__all__ = [
    "Basis",
    "BasisView",
    "BasisEvaluator",
    "CSBasis",
    "Dynamics",
    "FieldEvaluator",
    "FieldExpansion",
    "Grid",
    "GridBasis",
    "RadialLaplaceContinuation",
    "Mainfield",
    "PynamEye",
    "SHBasis",
    "SurfaceOperators",
    "basis_kind",
    "debugplot",
    "globalplot",
    "is_basis_kind",
    "is_cs_basis",
    "is_grid_basis",
    "is_sh_basis",
    "normalize_horizontal_basis_kind",
    "plot_input_vs_interpolated",
    "set_backend",
    "use_jax",
    "set_input_source",
    "get_input_source",
]
