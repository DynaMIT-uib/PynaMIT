"""Generic grid utilities.

This module contains utility functions for grid-related operations
such as computing determinants and inverses of 3D Jacobian matrices,
and constraining array values within specified bounds.

These are generic routines that don't depend on any specific grid type.
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
            f"The following matrices are not invertible: {np.where(np.isclose(det, 0))[0]}."
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
        raise ValueError("Range of array values is too large compared to vmin and vmax.")

    a_shifted = arr - np.minimum(amin, vmin) + vmin - np.maximum(amax, vmax) + vmax

    return a_shifted


def _unit_vectors_from_latlon(lat_deg, lon_deg):
    """Convert latitude/longitude arrays (degrees) to unit vectors."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat_rad)
    return np.stack(
        [
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        axis=-1,
    )


def _spherical_triangle_area(a, b, c):
    """Area of spherical triangles on the unit sphere.

    Parameters
    ----------
    a, b, c : ndarray
        Arrays of unit vectors with common trailing dimension 3.

    Returns
    -------
    ndarray
        Triangle areas on the unit sphere.
    """
    cross_bc = np.cross(b, c)
    numer = np.abs(np.einsum("...i,...i->...", a, cross_bc))
    denom = (
        1.0
        + np.einsum("...i,...i->...", a, b)
        + np.einsum("...i,...i->...", b, c)
        + np.einsum("...i,...i->...", c, a)
    )
    return 2.0 * np.arctan2(numer, denom)


def compute_structured_spherical_point_areas(
    lat,
    lon,
    *,
    periodic_lon=True,
    normalize_mean=True,
):
    """Estimate point-area weights for a structured curvilinear spherical grid.

    The input grid is assumed to be structured in array index space (2D arrays),
    but not necessarily separable in latitude/longitude. Cell areas are computed
    on the unit sphere by splitting each quadrilateral into two spherical
    triangles. Cell areas are then distributed equally to the four corner points.

    Parameters
    ----------
    lat, lon : array-like
        2D latitude/longitude arrays in degrees with identical shape.
    periodic_lon : bool, optional
        Whether the last longitude column wraps to the first. Defaults to True.
    normalize_mean : bool, optional
        If True, normalize point areas to have mean 1. This keeps weighting
        scale comparable to unweighted least-squares when regularization is used.

    Returns
    -------
    ndarray
        Point-area weights with the same 2D shape as ``lat``/``lon``.

    Raises
    ------
    ValueError
        If inputs are not matching 2D arrays, or if the computed weights are
        non-finite/non-positive.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError("lat and lon must be matching 2D arrays.")
    if min(lat.shape) < 2:
        raise ValueError("Structured spherical point areas require at least a 2x2 grid.")

    v = _unit_vectors_from_latlon(lat, lon)
    n_row, n_col = lat.shape

    if periodic_lon:
        v_right = np.roll(v, -1, axis=1)
        v00 = v[:-1, :]
        v10 = v[1:, :]
        v01 = v_right[:-1, :]
        v11 = v_right[1:, :]
    else:
        v00 = v[:-1, :-1]
        v10 = v[1:, :-1]
        v01 = v[:-1, 1:]
        v11 = v[1:, 1:]

    cell_area = _spherical_triangle_area(v00, v10, v11) + _spherical_triangle_area(v00, v11, v01)
    point_area = np.zeros((n_row, n_col), dtype=float)

    if periodic_lon:
        point_area[:-1, :] += 0.25 * cell_area  # top-left
        point_area[1:, :] += 0.25 * cell_area   # bottom-left
        cell_area_prev = np.roll(cell_area, 1, axis=1)
        point_area[:-1, :] += 0.25 * cell_area_prev  # top-right
        point_area[1:, :] += 0.25 * cell_area_prev   # bottom-right
    else:
        point_area[:-1, :-1] += 0.25 * cell_area
        point_area[1:, :-1] += 0.25 * cell_area
        point_area[:-1, 1:] += 0.25 * cell_area
        point_area[1:, 1:] += 0.25 * cell_area

    if normalize_mean:
        mean_area = np.mean(point_area)
        if mean_area > 0:
            point_area = point_area / mean_area

    if not np.all(np.isfinite(point_area)):
        raise ValueError("Computed point-area weights contain non-finite values.")
    if np.any(point_area <= 0):
        raise ValueError("Computed point-area weights must be strictly positive.")

    return point_area
