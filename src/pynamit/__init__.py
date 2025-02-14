"""
PynaMIT: A Python package for numerical simulations of the ionosphere,
with dynamics induced by field-aligned currents and horizontal neutral winds.

This package includes modules for simulation, visualization, and various utilities.
"""

from .simulation.dynamics import Dynamics
from .simulation.visualization import globalplot, debugplot
from .simulation.pynameye import PynamEye
from .cubed_sphere.cubed_sphere import CSProjection
from .spherical_harmonics.sh_basis import SHBasis
from .simulation.mainfield import Mainfield
from .primitives.field_evaluator import FieldEvaluator
from .primitives.basis_evaluator import BasisEvaluator
from .primitives.grid import Grid
from .primitives.vector import Vector