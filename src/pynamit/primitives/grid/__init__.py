"""Grid module for primitives.

This module contains grid-related primitives:
- Grid: Container for spatial coordinates (theta, phi)
- GridBasis: Abstract base class for grid-based field representations
- Interpolator: Abstract base class and implementations for grid interpolation
- Grid utilities: Helper functions for grid operations
"""

from .coordinates import Grid
from .grid_basis import GridBasis
from .interpolation import (
    Interpolator,
    CachedDelaunayInterpolator,
    UnstructuredInterpolator,
    CSInterpolator,
    create_interpolator,
)
from .grid_utils import (
    get_3D_determinants,
    invert_3D_matrices,
    constrain_values,
)

__all__ = [
    "Grid",
    "GridBasis",
    "Interpolator",
    "CachedDelaunayInterpolator",
    "UnstructuredInterpolator",
    "CSInterpolator",
    "create_interpolator",
    "get_3D_determinants",
    "invert_3D_matrices",
    "constrain_values",
]
