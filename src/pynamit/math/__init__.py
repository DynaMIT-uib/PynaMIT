"""Mathematical helpers for PynaMIT."""

from pynamit.math.backend import (
    JAX_AVAILABLE,
    asarray,
    block_until_ready,
    get_array_module,
    jit,
    set_backend,
    to_jax,
    to_numpy,
    use_jax,
    vmap,
    xp,
)
from pynamit.math.linear_map import (
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
)
from pynamit.math._einsum_linear_map import einsum_linear_map

__all__ = [
    "JAX_AVAILABLE",
    "LinearMap",
    "as_linear_map",
    "asarray",
    "block_until_ready",
    "diagonal_linear_map",
    "einsum_linear_map",
    "get_array_module",
    "jit",
    "set_backend",
    "to_jax",
    "to_numpy",
    "use_jax",
    "vmap",
    "xp",
]
