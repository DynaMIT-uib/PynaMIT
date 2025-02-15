"""
PynaMIT: A Python package for numerical simulations of the ionosphere,
with dynamics induced by field-aligned currents and horizontal neutral winds.

This package includes modules for simulation, visualization, and various utilities.
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
    "globalplot"
]