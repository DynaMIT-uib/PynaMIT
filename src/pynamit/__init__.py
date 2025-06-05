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

from .cubed_sphere.cs_basis import CSBasis
from .primitives.basis_evaluator import BasisEvaluator
from .primitives.field_evaluator import FieldEvaluator
from .primitives.field_expansion import FieldExpansion
from .primitives.grid import Grid
from .simulation.dynamics import Dynamics
from .simulation.mainfield import Mainfield
from .simulation.pynameye import PynamEye
from .simulation.visualization import debugplot, globalplot
from .simulation.input_vs_interpolated import plot_input_vs_interpolated
from .spherical_harmonics.sh_basis import SHBasis
from .simulation.debug import debug_br_plot

__all__ = [
    "BasisEvaluator",
    "CSBasis",
    "Dynamics",
    "FieldEvaluator",
    "FieldExpansion",
    "Grid",
    "Mainfield",
    "PynamEye",
    "SHBasis",
    "debugplot",
    "globalplot",
    "plot_input_vs_interpolated",
    "debug_br_plot"
]
