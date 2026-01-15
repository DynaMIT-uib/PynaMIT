"""Primitives module.

Core data structures and fundamental types for PynaMIT:
- Field: Unified field abstraction (scalar/vector, discrete/expanded)
- Grid: 2D coordinate grids (lat/lon or theta/phi)
- IO: I/O utilities for saving/loading data
- Mainfield: Magnetic field models (Dipole, IGRF, Radial)
- Timeseries: Time series data management
- interpolation: Grid interpolation strategies
"""

from .field import Field
from .grid import Grid
from .io import IO
from .mainfield import Mainfield
from .timeseries import Timeseries
from . import interpolation
from .interpolation import (
    Interpolator,
    CachedDelaunayInterpolator,
    UnstructuredInterpolator,
    CSInterpolator,
    create_interpolator,
)

__all__ = [
    "Field",
    "Grid",
    "IO",
    "Mainfield",
    "Timeseries",
    "interpolation",
    "Interpolator",
    "CachedDelaunayInterpolator",
    "UnstructuredInterpolator",
    "CSInterpolator",
    "create_interpolator",
]
