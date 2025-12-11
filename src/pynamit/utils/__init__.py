"""
Utility helpers for working with optional JAX acceleration.
"""

from .backend import (
    JAX_AVAILABLE,
    get_array_module,
    jit,
    set_backend,
    to_jax,
    to_numpy,
    use_jax,
    vmap,
    xp,
    asarray,
)
from .linalg import tensor_pinv

__all__ = [
    "JAX_AVAILABLE",
    "get_array_module",
    "jit",
    "set_backend",
    "tensor_pinv",
    "to_jax",
    "to_numpy",
    "use_jax",
    "vmap",
    "xp",
    "asarray",
]
