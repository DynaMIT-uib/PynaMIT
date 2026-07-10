"""Ionospheric constitutive closure between motion, current, and E."""

from __future__ import annotations

import numpy as np

from pynamit.math import einsum_linear_map_from_matvec
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


def conductance_to_resistance(hall, pedersen):
    """Convert Hall and Pedersen conductance to resistance values."""
    return _invert_pedersen_hall_pair(
        np.atleast_2d(pedersen), np.atleast_2d(hall)
    )


def resistance_to_conductance(etaP, etaH):
    """Convert Pedersen/Hall resistance to physical conductance."""
    return _invert_pedersen_hall_pair(etaP, etaH)


def resistance_tensor_on_grid(etaP, etaH, bP, bH):
    """Return the horizontal resistance tensor on the model grid."""
    xp = get_array_module(etaP, etaH, bP, bH)
    resistance = xp.stack([xp.asarray(etaP), xp.asarray(etaH)], axis=0)
    geometry = xp.stack([xp.asarray(bP), xp.asarray(bH)], axis=0)
    return xp.einsum("sijk,sk->ijk", geometry, resistance, optimize=True)


def wind_to_E_coeffs_operator(helmholtz_analysis, wind_to_E_grid, wind_synthesis):
    """Return the operator mapping neutral-wind coefficients to E."""
    xp = get_array_module(helmholtz_analysis, wind_to_E_grid, wind_synthesis)
    n_coefficients = int(helmholtz_analysis.shape[1])
    return einsum_linear_map_from_matvec(
        component_tensors=[
            xp.asarray(helmholtz_analysis),
            xp.asarray(wind_to_E_grid),
            xp.asarray(wind_synthesis),
        ],
        einsum_string_matvec="cmpg,pqg,qgrs,rs->cm",
        output_shape=(2, n_coefficients),
        input_shape=wind_synthesis.shape[2:],
    )


def tangential_current_to_E_coeffs_operator(
    helmholtz_analysis, resistance_tensor, current_synthesis
):
    """Map tangential-current coefficients to E coefficients."""
    n_coefficients = int(helmholtz_analysis.shape[1])
    grid_to_coefficients = as_linear_map(
        helmholtz_analysis,
        input_shape=(2, resistance_tensor.shape[-1]),
        output_shape=(2, n_coefficients),
    )
    current_to_E_grid = pointwise_matrix_linear_map(resistance_tensor)
    return grid_to_coefficients @ current_to_E_grid @ current_synthesis


def source_to_E_coeffs_operator(
    helmholtz_analysis, resistance_tensor, source_to_JS, coefficient_length
):
    """Compose a source-to-JS map with the resistance closure."""
    if source_to_JS is None:
        return None
    xp = get_array_module(helmholtz_analysis, resistance_tensor, source_to_JS)
    return einsum_linear_map_from_matvec(
        component_tensors=[
            xp.asarray(helmholtz_analysis),
            xp.asarray(resistance_tensor),
            xp.asarray(source_to_JS),
        ],
        einsum_string_matvec="cmpg,pqg,qgl,l->cm",
        output_shape=(2, int(coefficient_length)),
        input_shape=source_to_JS.shape[2:],
    )


def Q_eff_from_neutral_wind(
    state, input_timeseries, wind_representation, input_time, wind_coeff_rows
):
    """Return effective current equivalent to neutral wind."""
    wind_synthesis = wind_representation.get_helmholtz_synthesis_operator(
        state.geometry.grid
    )
    Q_eff_values = []
    for time_value, wind_coeffs in zip(input_time, wind_coeff_rows):
        state.update(input_timeseries, time_value)
        wind_on_grid = np.asarray(wind_synthesis.matvec(wind_coeffs)).reshape(
            (2, state.geometry.grid.size)
        )
        E_wind_on_grid = np.einsum(
            "abg,bg->ag", np.asarray(state.geometry.bu), wind_on_grid, optimize=True
        )
        point_resistance = np.moveaxis(
            np.asarray(state.resistance_tensor_on_grid), -1, 0
        )
        Q_eff_on_grid = np.linalg.solve(
            point_resistance, E_wind_on_grid.T[..., np.newaxis]
        )[..., 0].T
        Q_eff_values.append(Q_eff_on_grid)

    values = np.asarray(Q_eff_values)
    grid = state.geometry.grid
    return values[:, 0, :], values[:, 1, :], grid.lat, grid.lon


def fit_Q_eff_coefficients(
    state,
    input_timeseries,
    q_field_space,
    input_time,
    wind_coeff_rows,
    *,
    reg_lambda=None,
    pinv_rtol=1e-15,
):
    """Fit effective-current coefficients for wind forcing."""
    q_coeff_rows = []
    q_synthesis = q_field_space.representation.get_helmholtz_synthesis_operator(
        state.geometry.grid
    )
    for time_value, wind_coeffs in zip(input_time, wind_coeff_rows):
        state.update(input_timeseries, time_value)
        E_wind_coeffs = state.u_coeffs_to_E_coeffs.matvec(wind_coeffs)
        q_to_E = tangential_current_to_E_coeffs_operator(
            state.geometry.helmholtz_analysis_matrix,
            state.resistance_tensor_on_grid,
            q_synthesis,
        )
        matrix = np.asarray(q_to_E.to_matrix(backend="numpy"))
        rhs = np.asarray(E_wind_coeffs).reshape(-1)
        if reg_lambda is not None and float(reg_lambda) > 0.0:
            weight = float(reg_lambda)
            regularization = weight * np.eye(matrix.shape[1], dtype=matrix.dtype)
            matrix = np.vstack([matrix, regularization])
            rhs = np.concatenate([rhs, np.zeros(matrix.shape[1], dtype=rhs.dtype)])
        q_coeffs, *_ = np.linalg.lstsq(matrix, rhs, rcond=pinv_rtol)
        q_coeff_rows.append(
            q_field_space.validate_coefficients(q_coeffs, name="Q_eff coefficients")
        )
    return np.asarray(q_coeff_rows)


__all__ = [
    "Q_eff_from_neutral_wind",
    "conductance_to_resistance",
    "fit_Q_eff_coefficients",
    "resistance_to_conductance",
    "resistance_tensor_on_grid",
    "source_to_E_coeffs_operator",
    "tangential_current_to_E_coeffs_operator",
    "wind_to_E_coeffs_operator",
]
