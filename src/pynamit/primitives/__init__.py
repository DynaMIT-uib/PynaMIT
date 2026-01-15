"""Primitives module.

Core data structures and fundamental types for PynaMIT:
- Field: Unified field abstraction (scalar/vector, discrete/expanded)
- Grid: 2D coordinate grids (lat/lon or theta/phi)
- GridBasis: Abstract base for grid-based representations
- IO: I/O utilities for saving/loading data
- Mainfield: Magnetic field models (Dipole, IGRF, Radial)
- Timeseries: Time series data management
- interpolation: Grid interpolation strategies
"""

from .field import Field
from .grid import Grid, GridBasis
from .grid import interpolation
from .grid.interpolation import (
    Interpolator,
    CachedDelaunayInterpolator,
    UnstructuredInterpolator,
    CSInterpolator,
    create_interpolator,
)
from .io import IO
from .mainfield import Mainfield
from .timeseries import Timeseries

__all__ = [
    "Field",
    "Grid",
    "GridBasis",
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
