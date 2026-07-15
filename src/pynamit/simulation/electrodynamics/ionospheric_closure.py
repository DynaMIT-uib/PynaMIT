"""Ionospheric constitutive closure between motion, current, and E."""

from __future__ import annotations

import numpy as np

from pynamit.math.backend import get_array_module
from pynamit.math.linear_map import as_linear_map, pointwise_matrix_linear_map


def _invert_pedersen_hall_pair(pedersen, hall):
    """Invert a Pedersen/Hall tensor pair pointwise."""
    pedersen, hall = np.broadcast_arrays(
        np.asarray(pedersen, dtype=float), np.asarray(hall, dtype=float)
    )
    denominator = pedersen**2 + hall**2
    valid = np.isfinite(denominator) & (denominator > np.finfo(float).tiny)
    inverse_pedersen = np.full_like(pedersen, np.nan, dtype=float)
    inverse_hall = np.full_like(hall, np.nan, dtype=float)
    np.divide(pedersen, denominator, out=inverse_pedersen, where=valid)
    np.divide(hall, denominator, out=inverse_hall, where=valid)
    return inverse_pedersen, inverse_hall


def conductance_to_resistance(sigmaP, sigmaH):
    """Convert Pedersen/Hall conductance to resistance values."""
    return _invert_pedersen_hall_pair(sigmaP, sigmaH)


def resistance_to_conductance(etaP, etaH):
    """Convert Pedersen/Hall resistance to physical conductance."""
    return _invert_pedersen_hall_pair(etaP, etaH)


def pedersen_geometry_tensor(btheta, bphi, br):
    """Return the horizontal Pedersen geometry tensor."""
    xp = get_array_module(btheta, bphi, br)
    btheta = xp.asarray(btheta)
    bphi = xp.asarray(bphi)
    br = xp.asarray(br)
    return xp.stack(
        [
            xp.stack([bphi**2 + br**2, -btheta * bphi], axis=0),
            xp.stack([-btheta * bphi, btheta**2 + br**2], axis=0),
        ],
        axis=0,
    )


def hall_geometry_tensor(br):
    """Return the antisymmetric horizontal Hall geometry tensor."""
    xp = get_array_module(br)
    br = xp.asarray(br)
    zeros = xp.zeros_like(br)
    return xp.stack([xp.stack([zeros, br], axis=0), xp.stack([-br, zeros], axis=0)], axis=0)


def wind_motional_E_tensor(Br):
    """Map neutral wind to ``-u x B`` using radial field ``Br``."""
    xp = get_array_module(Br)
    Br = xp.asarray(Br)
    zeros = xp.zeros_like(Br)
    return xp.stack([xp.stack([zeros, -Br], axis=0), xp.stack([Br, zeros], axis=0)], axis=0)


def resistance_tensor_on_grid(etaP, etaH, pedersen_geometry, hall_geometry):
    """Return the horizontal resistance tensor on the model grid."""
    xp = get_array_module(etaP, etaH, pedersen_geometry, hall_geometry)
    resistance = xp.stack([xp.asarray(etaP), xp.asarray(etaH)], axis=0)
    geometry = xp.stack([xp.asarray(pedersen_geometry), xp.asarray(hall_geometry)], axis=0)
    return xp.einsum("sijk,sk->ijk", geometry, resistance, optimize=True)


def electric_field_on_grid(sheet_current, resistance_tensor, *, wind=None, wind_to_E=None):
    """Apply the ionospheric closure directly on one grid."""
    if (wind is None) != (wind_to_E is None):
        raise ValueError("wind and wind_to_E must be provided together.")
    xp = get_array_module(sheet_current, resistance_tensor, wind, wind_to_E)
    electric_field = xp.einsum(
        "ijg,jg->ig", xp.asarray(resistance_tensor), xp.asarray(sheet_current), optimize=True
    )
    if wind is not None:
        electric_field += xp.einsum(
            "ijg,jg->ig", xp.asarray(wind_to_E), xp.asarray(wind), optimize=True
        )
    return electric_field


def joule_heating_from_current(sheet_current, etaP, pedersen_geometry):
    """Return collisional Joule heating ``etaP * J.T @ P @ J``."""
    xp = get_array_module(sheet_current, etaP, pedersen_geometry)
    sheet_current = xp.asarray(sheet_current)
    etaP = xp.asarray(etaP)
    pedersen_geometry = xp.asarray(pedersen_geometry)
    return etaP * xp.einsum(
        "ig,ijg,jg->g", sheet_current, pedersen_geometry, sheet_current, optimize=True
    )


def wind_to_E_coeffs_operator(helmholtz_analysis, wind_to_E_grid, wind_synthesis):
    """Return the operator mapping neutral-wind coefficients to E."""
    n_grid = int(wind_to_E_grid.shape[-1])
    n_coefficients = int(
        helmholtz_analysis.output_shape[1]
        if hasattr(helmholtz_analysis, "output_shape")
        else helmholtz_analysis.shape[1]
    )
    grid_to_coefficients = as_linear_map(
        helmholtz_analysis,
        input_shape=(2, n_grid),
        output_shape=(2, n_coefficients),
    )
    n_wind_coefficients = int(
        wind_synthesis.input_shape[1]
        if hasattr(wind_synthesis, "input_shape")
        else wind_synthesis.shape[-1]
    )
    wind_to_grid = as_linear_map(
        wind_synthesis,
        input_shape=(2, n_wind_coefficients),
        output_shape=(2, n_grid),
    )
    return (
        grid_to_coefficients
        @ pointwise_matrix_linear_map(wind_to_E_grid)
        @ wind_to_grid
    )


def tangential_current_to_E_coeffs_operator(
    helmholtz_analysis, resistance_tensor, sheet_current_synthesis
):
    """Map sheet-current coefficients to E coefficients."""
    n_coefficients = int(
        helmholtz_analysis.output_shape[1]
        if hasattr(helmholtz_analysis, "output_shape")
        else helmholtz_analysis.shape[1]
    )
    grid_to_coefficients = as_linear_map(
        helmholtz_analysis,
        input_shape=(2, resistance_tensor.shape[-1]),
        output_shape=(2, n_coefficients),
    )
    current_to_E_grid = pointwise_matrix_linear_map(resistance_tensor)
    return grid_to_coefficients @ current_to_E_grid @ sheet_current_synthesis


def Q_eff_on_grid_from_wind(wind_on_grid, wind_to_E_grid, resistance_tensor):
    """Return the effective sheet current equivalent to neutral wind."""
    E_wind_on_grid = np.einsum(
        "abg,bg->ag", np.asarray(wind_to_E_grid), np.asarray(wind_on_grid), optimize=True
    )
    point_resistance = np.moveaxis(np.asarray(resistance_tensor), -1, 0)
    return np.linalg.solve(point_resistance, E_wind_on_grid.T[..., np.newaxis])[..., 0].T


def solve_Q_eff_coefficients(Q_eff_to_E, E_wind_coeffs, *, reg_lambda=None, pinv_rtol=1e-15):
    """Solve for Q_eff coefficients matching wind-driven E."""
    matrix = np.asarray(Q_eff_to_E.to_matrix(backend="numpy"))
    rhs = np.asarray(E_wind_coeffs).reshape(-1)
    if reg_lambda is not None and float(reg_lambda) > 0.0:
        weight = float(reg_lambda)
        regularization = weight * np.eye(matrix.shape[1], dtype=matrix.dtype)
        matrix = np.vstack([matrix, regularization])
        rhs = np.concatenate([rhs, np.zeros(matrix.shape[1], dtype=rhs.dtype)])
    coefficients, *_ = np.linalg.lstsq(matrix, rhs, rcond=pinv_rtol)
    return coefficients


__all__ = [
    "Q_eff_on_grid_from_wind",
    "conductance_to_resistance",
    "electric_field_on_grid",
    "hall_geometry_tensor",
    "joule_heating_from_current",
    "pedersen_geometry_tensor",
    "resistance_to_conductance",
    "resistance_tensor_on_grid",
    "solve_Q_eff_coefficients",
    "tangential_current_to_E_coeffs_operator",
    "wind_motional_E_tensor",
    "wind_to_E_coeffs_operator",
]
