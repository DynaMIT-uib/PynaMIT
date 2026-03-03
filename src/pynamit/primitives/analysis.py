"""Cached analysis operators for coefficient field spaces.

This module owns the backward / fitted side of basis usage:
- scalar projection matrices
- Helmholtz projection matrices
- least-squares problem setup

Forward / exact operator caching remains on the basis objects themselves.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.utils import tensor_pinv

_SCALAR_PROJECTION_CACHE: dict[tuple[Any, ...], Any] = {}
_HELMHOLTZ_PROJECTION_CACHE: dict[tuple[Any, ...], Any] = {}
_SCALAR_PROBLEM_CACHE: dict[tuple[Any, ...], LeastSquaresProblem] = {}
_HELMHOLTZ_PROBLEM_CACHE: dict[tuple[Any, ...], LeastSquaresProblem] = {}


def _array_signature(values: Any) -> Any:
    """Return a stable signature for dense/sparse numeric arrays."""
    if values is None:
        return None

    try:
        import scipy.sparse

        if scipy.sparse.issparse(values):
            arr = values.tocsr()
            payload = arr.data.tobytes() + arr.indices.tobytes() + arr.indptr.tobytes()
            digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
            return ("sparse", arr.shape, arr.dtype.str, digest)
    except ImportError:
        pass

    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return ("dense", arr.shape, arr.dtype.str, digest)


def _grid_signature(grid: Any) -> tuple[Any, Any]:
    """Return a stable signature for grid-dependent analysis objects."""
    return (
        getattr(grid, "hash", id(grid)),
        _array_signature(getattr(grid, "weights", None)),
    )


def _spec_signature(spec: Any) -> Any:
    """Return a stable signature for a field specification."""
    return getattr(spec, "signature", id(spec))


def get_scalar_least_squares_problem(
    spec: Any,
    grid: Any,
    sqrt_weights: Optional[np.ndarray] = None,
    reg_lambda: Optional[float] = None,
) -> LeastSquaresProblem:
    """Return cached least-squares setup for scalar analysis."""
    cache_key = (
        "scalar_problem",
        _spec_signature(spec),
        _grid_signature(grid),
        _array_signature(sqrt_weights),
        None if reg_lambda is None else float(reg_lambda),
    )
    cached = _SCALAR_PROBLEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    G = spec.get_evaluation_matrix(grid)
    L = spec.get_regularization_matrix(scalar=True, reg_lambda=reg_lambda)
    reg_matrices = [L] if L is not None else []
    reg_weights = [reg_lambda] if reg_lambda is not None else []

    problem = LeastSquaresProblem(
        A=[G],
        solution_shape=spec.index_length,
        data_shapes=[grid.size],
        sqrt_weights=[sqrt_weights],
        regularization_weights=reg_weights,
        regularization_matrices=reg_matrices,
    )
    _SCALAR_PROBLEM_CACHE[cache_key] = problem
    return problem


def get_helmholtz_least_squares_problem(
    spec: Any,
    grid: Any,
    sqrt_weights: Optional[np.ndarray] = None,
    reg_lambda: Optional[float] = None,
) -> LeastSquaresProblem:
    """Return cached least-squares setup for Helmholtz analysis."""
    cache_key = (
        "helmholtz_problem",
        _spec_signature(spec),
        _grid_signature(grid),
        _array_signature(sqrt_weights),
        None if reg_lambda is None else float(reg_lambda),
    )
    cached = _HELMHOLTZ_PROBLEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    G_h = spec.get_vector_basis_matrix(grid)
    L_h = spec.get_regularization_matrix(scalar=False, reg_lambda=reg_lambda)
    reg_matrices = [L_h] if L_h is not None else []
    reg_weights = [reg_lambda] if reg_lambda is not None else []

    problem = LeastSquaresProblem(
        A=[G_h],
        solution_shape=(2, spec.index_length),
        data_shapes=[(2, grid.size)],
        sqrt_weights=[sqrt_weights],
        regularization_weights=reg_weights,
        regularization_matrices=reg_matrices,
    )
    _HELMHOLTZ_PROBLEM_CACHE[cache_key] = problem
    return problem


def get_scalar_projection_matrix(spec: Any, grid: Any) -> Any:
    """Return cached scalar analysis matrix for one field space and grid."""
    cache_key = ("scalar_projection", _spec_signature(spec), _grid_signature(grid))
    cached = _SCALAR_PROJECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import scipy.sparse

    G = spec.get_evaluation_matrix(grid)
    weights = getattr(grid, "weights", None)
    if weights is not None:
        is_sparse = scipy.sparse.issparse(G)
        if is_sparse:
            w_diag = scipy.sparse.diags(weights)
            gt_w = G.T @ w_diag
        else:
            gt_w = G.T * weights
        mass = gt_w @ G
        if is_sparse:
            mass = mass.toarray()
            gt_w = gt_w.toarray()
        projection = np.linalg.solve(mass, gt_w)
    else:
        g_dense = G.toarray() if scipy.sparse.issparse(G) else G
        projection = tensor_pinv(g_dense, n_leading_flattened=1)

    _SCALAR_PROJECTION_CACHE[cache_key] = projection
    return projection


def get_helmholtz_projection_matrix(spec: Any, grid: Any) -> Any:
    """Return cached Helmholtz analysis tensor for one field space and grid."""
    cache_key = ("helmholtz_projection", _spec_signature(spec), _grid_signature(grid))
    cached = _HELMHOLTZ_PROJECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if getattr(spec, "kind", "") == "SH" and spec.field_type in ("scalar", "tangential"):
        g_th = spec.get_evaluation_matrix(grid, derivative="theta")
        g_ph = spec.get_evaluation_matrix(grid, derivative="phi")
        g_th = g_th.toarray() if hasattr(g_th, "toarray") else g_th
        g_ph = g_ph.toarray() if hasattr(g_ph, "toarray") else g_ph
        g_grad = np.array([g_th, g_ph])
        g_rxgrad = np.array([g_ph, -g_th])
        g_helmholtz = np.stack([-g_grad, g_rxgrad], axis=2)
        projection = tensor_pinv(g_helmholtz, n_leading_flattened=2)
    else:
        # Non-SH bases may need basis-specific gauge/constraint handling.
        projection = spec.basis.construct_projection_matrix(grid)

    _HELMHOLTZ_PROJECTION_CACHE[cache_key] = projection
    return projection
