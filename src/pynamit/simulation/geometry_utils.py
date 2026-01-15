"""Utility functions for geometry module.

This module provides helper functions for common operations in the geometry
module, reducing code duplication and ensuring consistent handling of
sparse matrices and operator extraction.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.grid import Grid
    from pynamit.math.linear_map import LinearMap


def to_dense(matrix: Union[np.ndarray, "LinearMap", scipy.sparse.spmatrix]) -> np.ndarray:
    """Convert matrix to dense numpy array.

    Handles scipy sparse matrices, LinearMap objects, and regular arrays
    with a unified interface.

    Parameters
    ----------
    matrix : array-like, LinearMap, or sparse matrix
        The matrix to convert.

    Returns
    -------
    np.ndarray
        Dense numpy array.
    """
    if scipy.sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_dense"):
        return matrix.to_dense()
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def get_radial_shift_diagonal(
    basis: "Basis",
    start_r: float,
    end_r: float,
    kind: str = "external"
) -> np.ndarray:
    """Extract diagonal of radial shift operator.

    This is a common operation for computing propagation factors
    in magnetospheric coupling calculations.

    Parameters
    ----------
    basis : Basis
        The basis providing the radial shift operator.
    start_r : float
        Starting radius.
    end_r : float
        Ending radius.
    kind : str, optional
        Type of radial shift ("external" or "internal"). Default "external".

    Returns
    -------
    np.ndarray
        Diagonal elements of the radial shift operator.
    """
    operator = basis.get_radial_shift_operator(start_r, end_r, kind=kind)
    return np.diag(to_dense(operator))


def get_evaluation_matrix_dense(basis: "Basis", grid: "Grid") -> np.ndarray:
    """Get evaluation matrix as dense array.

    Parameters
    ----------
    basis : Basis
        The basis to evaluate.
    grid : Grid
        The grid on which to evaluate.

    Returns
    -------
    np.ndarray
        Dense evaluation matrix.
    """
    matrix = basis.get_evaluation_matrix(grid)
    return to_dense(matrix)
