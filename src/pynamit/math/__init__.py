"""Mathematical helpers for PynaMIT."""

from pynamit.math._einsum_linear_map import einsum_linear_map, einsum_linear_map_from_matvec
from pynamit.math.backend import (
    JAX_AVAILABLE,
    asarray,
    backend_context,
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
from pynamit.math.fingerprints import array_fingerprint, content_fingerprint
from pynamit.math.linear_map import (
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    identity_linear_map,
    is_noop_linear_map,
    pointwise_matrix_linear_map,
    take_linear_map,
    vstack_linear_maps,
)

__all__ = [
    "JAX_AVAILABLE",
    "LinearMap",
    "as_linear_map",
    "asarray",
    "array_fingerprint",
    "backend_context",
    "block_until_ready",
    "content_fingerprint",
    "diagonal_linear_map",
    "einsum_linear_map",
    "einsum_linear_map_from_matvec",
    "get_array_module",
    "identity_linear_map",
    "is_noop_linear_map",
    "jit",
    "pointwise_matrix_linear_map",
    "set_backend",
    "take_linear_map",
    "to_jax",
    "to_numpy",
    "use_jax",
    "vstack_linear_maps",
    "vmap",
    "xp",
]
