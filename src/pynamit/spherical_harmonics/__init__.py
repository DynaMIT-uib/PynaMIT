"""Spherical harmonics module.

This module provides spherical harmonic basis functions and related utilities:
- SHBasis: Main spherical harmonic basis class
- GauntEngine: Gaunt coefficient computation for product operations
- Wigner symbols: 3j, 6j, 9j, and small-d rotation matrices
- legendre: Legendre polynomial backends
- sh_operators: Spectral space operators
- sh_transforms: Fast transform methods
"""

from .sh_basis import SHBasis
from .gaunt import GauntEngine
from .wigner import wigner_3j, wigner_6j, wigner_9j, wigner_small_d
from . import legendre
from . import sh_operators
from . import sh_transforms

__all__ = [
    "SHBasis",
    "GauntEngine",
    "wigner_3j",
    "wigner_6j",
    "wigner_9j",
    "wigner_small_d",
    "legendre",
    "sh_operators",
    "sh_transforms",
]
