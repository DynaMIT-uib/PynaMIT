"""
PynaMIT: A Python package for numerical simulations of ionospheric dynamics
induced by field-aligned currents and horizontal neutral winds.

This package includes modules for simulation, visualization, and various utilities.

Attributes
----------
BasisEvaluator : class
    Class for evaluating basis functions.
CSProjection : class
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
Vector : class
    Class for storing vector data and defining vector operations.
debugplot : function
    Function for debug plotting.
globalplot : function
    Function for global plotting.
"""

from .cubed_sphere.cubed_sphere import CSProjection
from .primitives.basis_evaluator import BasisEvaluator
from .primitives.field_evaluator import FieldEvaluator
from .primitives.grid import Grid
from .primitives.vector import Vector
from .simulation.dynamics import Dynamics
from .simulation.mainfield import Mainfield
from .simulation.pynameye import PynamEye
from .simulation.visualization import debugplot, globalplot
from .spherical_harmonics.sh_basis import SHBasis

__all__ = [
    "BasisEvaluator",
    "CSProjection",
    "Dynamics",
    "FieldEvaluator",
    "Grid",
    "Mainfield",
    "PynamEye",
    "SHBasis",
    "Vector",
    "debugplot",
    "globalplot",
]
