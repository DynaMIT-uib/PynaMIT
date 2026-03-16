"""Shared plumbing helpers for solver-facing operator objects.

These helpers keep branch-specific solver modules focused on physics and
constraint semantics, while centralizing the small amount of reusable linear
operator exposure code.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from pynamit.math.linear_map import LinearMap
from pynamit.utils import asarray, to_numpy

DenseBuilder = Callable[[], np.ndarray]
VectorMap = Callable[[np.ndarray], np.ndarray]


def cached_dense_builder(builder: DenseBuilder) -> DenseBuilder:
    """Return a cached dense builder."""
    cache: dict[str, Optional[np.ndarray]] = {"value": None}

    def _get_dense() -> np.ndarray:
        if cache["value"] is None:
            cache["value"] = asarray(builder())
        return asarray(cache["value"])

    return _get_dense


def coerce_dense_operator_matrix(
    operator: Any,
    *,
    n_component_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> np.ndarray:
    """Return a dense 2D matrix for an operator-like object.

    If the dense representation is stored with explicit component rows, e.g.
    shape ``(n_components, n_cols, n_cols)``, it is flattened to
    ``(n_components * n_cols, n_cols)``.
    """
    dense = operator.to_dense() if hasattr(operator, "to_dense") else to_numpy(operator)
    arr = asarray(dense)
    if arr.ndim == 3:
        if n_component_rows is None or n_cols is None:
            raise ValueError(
                "n_component_rows and n_cols are required to flatten a 3D dense operator."
            )
        return asarray(arr.reshape(n_component_rows * n_cols, n_cols))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D dense operator, got ndim={arr.ndim}.")
    return asarray(arr)


def build_linear_map(
    *,
    shape: tuple[int, int],
    matvec: VectorMap,
    rmatvec: Optional[VectorMap] = None,
    dense_builder: Optional[DenseBuilder] = None,
    dtype: Any = np.float64,
    domain_space: Optional[str] = None,
    codomain_space: Optional[str] = None,
) -> LinearMap:
    """Build a `LinearMap` with optional cached dense fallback."""
    cached_dense = cached_dense_builder(dense_builder) if dense_builder is not None else None

    def _matvec(x: np.ndarray) -> np.ndarray:
        return asarray(matvec(x))

    def _rmatvec(y: np.ndarray) -> np.ndarray:
        if rmatvec is not None:
            return asarray(rmatvec(y))
        if cached_dense is None:
            raise RuntimeError("rmatvec requested without custom adjoint or dense builder.")
        return asarray(cached_dense().T @ asarray(y).reshape(-1))

    return LinearMap(
        shape=shape,
        dtype=np.dtype(dtype),
        domain_space=domain_space,
        codomain_space=codomain_space,
        _matvec=_matvec,
        _rmatvec=_rmatvec,
        _to_dense=cached_dense,
        source=None,
    )
