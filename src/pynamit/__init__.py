"""
PynaMIT: A Python package for dynamic MIT coupling simulations.

This package includes modules for simulation, visualization, and various
utilities.

Attributes
----------
CSBasis : class
    Class for cubed sphere projections.
Dynamics : class
    Class for simulating ionospheric dynamics.
Field : class
    Unified class for scalar/vector fields.
Grid : class
    Class for grid management.
Mainfield : class
    Class for main field evaluation.
PynamEye : class
    Class for visualization.
debugplot : function
    Function for debug plotting.
globalplot : function
    Function for global plotting.
SHBasis : class
    Class for spherical harmonics basis functions.
"""

from .cubed_sphere import CSBasis
from .primitives import (
    Field,
    Grid,
    IO,
    Mainfield,
    Timeseries,
    interpolation,  # Re-export for backwards compatibility
)
from .simulation import Dynamics, State, run_pynamit
from .visualization import PynamEye, debugplot, globalplot
from .spherical_harmonics import SHBasis
from .utils import set_backend

__all__ = [
    "CSBasis",
    "Dynamics",
    "Field",
    "Grid",
    "IO",
    "Mainfield",
    "PynamEye",
    "debugplot",
    "globalplot",
    "SHBasis",
    "State",
    "Timeseries",
    "run_pynamit",
    "set_backend",
]
