"""Array utilities.

This module provides utility functions for performing array operations
such as computing determinants and inverses of 3D matrices, as well as
constraining array values within specified bounds.
"""

import numpy as np


def get_3D_determinants(M):
    """Calculate determinants of 3D matrices.

    Parameters
    ----------
    M : array
        Array with shape ``(N, 3, 3)``, corresponding to ``N`` 3D
        matrices.

    Returns
    -------
    det : array
        Array with determinants, shape ``(N)``.

    Raises
    ------
    ValueError
        If the input array is not 3D or if the last two axes are not
        3 x 3.
    """
    if (M.shape[1:] != (3, 3)) | (M.ndim != 3):
        raise ValueError("Input array must have shape (N, 3, 3).")

    det = (
        M[:, 0, 0] * M[:, 1, 1] * M[:, 2, 2]
        - M[:, 0, 0] * M[:, 1, 2] * M[:, 2, 1]
        - M[:, 0, 1] * M[:, 1, 0] * M[:, 2, 2]
        + M[:, 0, 1] * M[:, 1, 2] * M[:, 2, 0]
        + M[:, 0, 2] * M[:, 1, 0] * M[:, 2, 1]
        - M[:, 0, 2] * M[:, 1, 1] * M[:, 2, 0]
    )

    return det


def invert_3D_matrices(M):
    """Calculate inverse of 3D matrices.

    Parameters
    ----------
    M : array
        Array with shape ``(N, 3, 3)``, corresponding to ``N`` 3D
        invertible matrices.

    Returns
    -------
    Minv : array
        Array with inverse matrices, shape ``(N, 3, 3)``.

    Raises
    ------
    ValueError
        If the input array is not 3D, if the last two axes are not
        3 x 3, or if any of the matrices are not invertible.
    """
    if (M.shape[1:] != (3, 3)) | (M.ndim != 3):
        raise ValueError("Input array must have shape (N, 3, 3).")
    det = get_3D_determinants(M)

    if np.any(np.isclose(det, 0)):
        raise ValueError(
            "The following matrices are not invertible: "
            f"{np.where(np.isclose(det, 0))[0]}."
        )

    Minv = np.empty(M.shape)
    Minv[:, 0, 0] = M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1]
    Minv[:, 0, 1] = -M[:, 0, 1] * M[:, 2, 2] + M[:, 0, 2] * M[:, 2, 1]
    Minv[:, 0, 2] = M[:, 0, 1] * M[:, 1, 2] - M[:, 0, 2] * M[:, 1, 1]
    Minv[:, 1, 0] = -M[:, 1, 0] * M[:, 2, 2] + M[:, 1, 2] * M[:, 2, 0]
    Minv[:, 1, 1] = M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0]
    Minv[:, 1, 2] = -M[:, 0, 0] * M[:, 1, 2] + M[:, 0, 2] * M[:, 1, 0]
    Minv[:, 2, 0] = M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0]
    Minv[:, 2, 1] = -M[:, 0, 0] * M[:, 2, 1] + M[:, 0, 1] * M[:, 2, 0]
    Minv[:, 2, 2] = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]

    return Minv / det.reshape((M.shape[0], 1, 1))


def constrain_values(arr, vmin, vmax, axis):
    """Constrain values of an array.

    Constrains the values of `arr` to be between `vmin` and `vmax` by
    adding a constant along a given axis.

    Parameters
    ----------
    arr : array
        Array to be clipped.
    vmin : scalar
        Minimum allowed value in result array `a_shifted`.
    vmax : scalar
        Maximum allowed value in result array `a_shifted`.
    axis : integer
        Axis along which to add a constant.

    Returns
    -------
    a_shifted : array
        ``a + constant``, where ``constant`` is chosen so that all
        elements of `a_shifted` is ``>= vmin`` and ``<= vmax`` (if
        possible).

    Raises
    ------
    ValueError
        If the range of `arr` is too large compared to `vmin` and
        `vmax`.
    """
    amin = arr.min(axis=axis, keepdims=True)
    amax = arr.max(axis=axis, keepdims=True)

    if np.any(amax - amin > vmax - vmin):
        raise ValueError(
            "Range of array values is too large compared to vmin and vmax."
        )

    a_shifted = (
        arr - np.minimum(amin, vmin) + vmin - np.maximum(amax, vmax) + vmax
    )

    return a_shifted
