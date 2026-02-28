"""Helpers for constructing input-fit weights on spherical sample grids.

These utilities provide a small set of named weighting policies that can be
reused in scripts and tests instead of manually assembling ``sqrt_weights``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pynamit.primitives.grid import compute_structured_spherical_point_areas
from pynamit.spherical_harmonics.sh_basis import SHBasis


_MW_ALIASES = {"mw"}
_UNIT_ALIASES = {None, "unit"}


def _as_matching_latlon_arrays(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    if lat_arr.shape != lon_arr.shape:
        raise ValueError("lat and lon must have matching shapes.")
    if lat_arr.ndim != 2:
        raise ValueError(
            "Expected 2D structured lat/lon arrays. "
            "Use reshaped arrays for weight-policy helpers."
        )
    return lat_arr, lon_arr


def _extract_regular_latlon_axes(lat_2d: np.ndarray, lon_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract separable 1D lat/lon axes from a regular geographic grid."""
    n_lat, n_lon = lat_2d.shape
    try:
        lat_rows = lat_2d.reshape(n_lat, n_lon)
        lon_rows = lon_2d.reshape(n_lat, n_lon)
    except ValueError as exc:
        raise ValueError("Could not reshape lat/lon to regular 2D grid.") from exc

    if not (
        np.allclose(lat_rows[:, 0:1], lat_rows)
        and np.allclose(lon_rows[0:1, :], lon_rows)
    ):
        raise ValueError(
            "Grid is not separable in geographic lat/lon; exact SH theta weights "
            "are not valid for this grid."
        )
    return lat_rows[:, 0].copy(), lon_rows[0, :].copy()


def compute_spherical_input_point_weights(
    lat,
    lon,
    *,
    weighting: Optional[str],
    nmax: Optional[int] = None,
    periodic_lon: bool = True,
    normalize_geom_area_mean: bool = True,
) -> np.ndarray:
    """Compute point weights for a structured spherical sample grid.

    Parameters
    ----------
    lat, lon : array-like
        Matching 2D latitude/longitude arrays in degrees.
    weighting : str or None
        Weighting policy:
        - ``None`` / ``"unit"``: unit weights
        - ``"sin_theta"``: pointwise ``sin(theta)`` weights
        - ``"mw"``: exact SH theta quadrature weights on regular lat/lon grids only
        - ``"geom_area"``: geometry-based point areas on curvilinear grids
    nmax : int, optional
        Required for the exact SH theta-weight policies.
    periodic_lon : bool, optional
        Only used for ``"geom_area"``.
    normalize_geom_area_mean : bool, optional
        If True, normalize geometry-based point areas to mean 1.

    Returns
    -------
    np.ndarray
        Point weights with the same 2D shape as ``lat``/``lon``.
    """
    lat_2d, lon_2d = _as_matching_latlon_arrays(lat, lon)

    if weighting in _UNIT_ALIASES:
        return np.ones_like(lat_2d, dtype=float)

    if weighting == "sin_theta":
        theta = np.deg2rad(90.0 - lat_2d)
        return np.sin(theta).astype(float, copy=False)

    if weighting in _MW_ALIASES:
        if nmax is None:
            raise ValueError(f"weighting={weighting!r} requires nmax.")
        lat_1d, _ = _extract_regular_latlon_axes(lat_2d, lon_2d)
        theta_1d = np.deg2rad(90.0 - lat_1d)
        w_theta = np.asarray(SHBasis.compute_exact_weights(theta_1d, int(nmax)), dtype=float)
        if w_theta.shape[0] != lat_2d.shape[0]:
            raise ValueError("Exact SH theta weights do not match latitude dimension.")
        return np.broadcast_to(w_theta[:, None], lat_2d.shape).copy()

    if weighting == "geom_area":
        return compute_structured_spherical_point_areas(
            lat_2d,
            lon_2d,
            periodic_lon=periodic_lon,
            normalize_mean=normalize_geom_area_mean,
        )

    raise ValueError(
        f"Unknown weighting policy {weighting!r}. "
        "Valid values: None/'unit', 'sin_theta', "
            "'mw', 'geom_area'."
    )


def compute_spherical_input_sqrt_weights(
    lat,
    lon,
    *,
    weighting: Optional[str],
    nmax: Optional[int] = None,
    vector: bool = False,
    periodic_lon: bool = True,
    normalize_geom_area_mean: bool = True,
) -> np.ndarray:
    """Compute ``sqrt_weights`` arrays for InputManager-based projections.

    Returns flattened scalar weights by default. For vector/tangential inputs,
    set ``vector=True`` to return a stacked ``(2, N)`` layout.
    """
    w = compute_spherical_input_point_weights(
        lat,
        lon,
        weighting=weighting,
        nmax=nmax,
        periodic_lon=periodic_lon,
        normalize_geom_area_mean=normalize_geom_area_mean,
    )
    sqrt_w = np.sqrt(np.asarray(w, dtype=float).reshape(-1))
    if vector:
        return np.vstack([sqrt_w, sqrt_w])
    return sqrt_w
