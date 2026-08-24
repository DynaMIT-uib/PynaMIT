"""Ionospheric constitutive closure between motion, current, and E."""

from __future__ import annotations

import numpy as np
from kompe.math import LinearMap, get_array_module, pointwise_matrix_linear_map

CONDUCTANCE_REFERENCE_S = 1.0


def _validate_reference_conductance(reference_conductance):
    """Return one finite, strictly positive conductance reference."""
    reference_conductance = float(reference_conductance)
    if not np.isfinite(reference_conductance) or reference_conductance <= 0.0:
        raise ValueError("reference_conductance must be finite and strictly positive.")
    return reference_conductance


def _invert_pedersen_hall_pair(pedersen, hall):
    """Invert a Pedersen/Hall tensor pair pointwise."""
    xp = get_array_module(pedersen, hall)
    pedersen, hall = xp.broadcast_arrays(
        xp.asarray(pedersen, dtype=float), xp.asarray(hall, dtype=float)
    )
    scale = xp.maximum(xp.abs(pedersen), xp.abs(hall))
    valid = xp.isfinite(scale) & (scale > 0.0)
    safe_scale = xp.where(valid, scale, xp.ones_like(scale))
    scaled_pedersen = pedersen / safe_scale
    scaled_hall = hall / safe_scale
    denominator = safe_scale * (scaled_pedersen**2 + scaled_hall**2)
    safe_denominator = xp.where(valid, denominator, xp.ones_like(denominator))
    inverse_pedersen = xp.where(valid, scaled_pedersen / safe_denominator, xp.nan)
    inverse_hall = xp.where(valid, scaled_hall / safe_denominator, xp.nan)
    return inverse_pedersen, inverse_hall


def conductance_to_resistance(sigmaP, sigmaH):
    """Convert Pedersen/Hall conductance to resistance values."""
    return _invert_pedersen_hall_pair(sigmaP, sigmaH)


def resistance_to_conductance(etaP, etaH):
    """Convert Pedersen/Hall resistance to physical conductance."""
    return _invert_pedersen_hall_pair(etaP, etaH)


def conductance_to_log_coordinates(
    sigmaP, sigmaH, *, reference_conductance=CONDUCTANCE_REFERENCE_S
):
    """Return dimensionless log-magnitude and log Hall/Pedersen ratio.

    Both conductance components must be strictly positive. The fixed
    reference only makes the magnitude logarithm dimensionless; with
    the default one-siemens reference it does not alter numeric input
    values before taking the logarithm.
    """
    xp = get_array_module(sigmaP, sigmaH)
    sigmaP, sigmaH = xp.broadcast_arrays(
        xp.asarray(sigmaP, dtype=float), xp.asarray(sigmaH, dtype=float)
    )
    reference_conductance = _validate_reference_conductance(reference_conductance)
    if bool(xp.any(~xp.isfinite(sigmaP))) or bool(xp.any(sigmaP <= 0.0)):
        raise ValueError("Pedersen conductance must be finite and strictly positive.")
    if bool(xp.any(~xp.isfinite(sigmaH))) or bool(xp.any(sigmaH <= 0.0)):
        raise ValueError("Hall conductance must be finite and strictly positive.")

    magnitude = xp.hypot(sigmaP, sigmaH)
    return (xp.log(magnitude) - xp.log(reference_conductance), xp.log(sigmaH) - xp.log(sigmaP))


def resistance_to_log_conductance_coordinates(
    etaP, etaH, *, reference_conductance=CONDUCTANCE_REFERENCE_S
):
    """Map positive resistance directly to log-conductance coordinates.

    Pedersen/Hall conductance and resistance have reciprocal magnitudes
    and the same Hall/Pedersen ratio. This direct mapping avoids
    constructing the intermediate conductance components.
    """
    xp = get_array_module(etaP, etaH)
    etaP, etaH = xp.broadcast_arrays(xp.asarray(etaP, dtype=float), xp.asarray(etaH, dtype=float))
    reference_conductance = _validate_reference_conductance(reference_conductance)
    if bool(xp.any(~xp.isfinite(etaP))) or bool(xp.any(etaP <= 0.0)):
        raise ValueError("Pedersen resistance must be finite and strictly positive.")
    if bool(xp.any(~xp.isfinite(etaH))) or bool(xp.any(etaH <= 0.0)):
        raise ValueError("Hall resistance must be finite and strictly positive.")

    magnitude = xp.hypot(etaP, etaH)
    return (-xp.log(magnitude) - xp.log(reference_conductance), xp.log(etaH) - xp.log(etaP))


def conductance_from_log_coordinates(
    log_magnitude, log_ratio, *, reference_conductance=CONDUCTANCE_REFERENCE_S
):
    """Reconstruct positive conductance from log coordinates."""
    reference_conductance = _validate_reference_conductance(reference_conductance)
    xp = get_array_module(log_magnitude, log_ratio)
    log_magnitude = xp.asarray(log_magnitude)
    log_ratio = xp.asarray(log_ratio)
    log_pedersen = (
        xp.log(reference_conductance) + log_magnitude - 0.5 * xp.logaddexp(0.0, 2.0 * log_ratio)
    )
    sigmaP = xp.exp(log_pedersen)
    sigmaH = xp.exp(log_pedersen + log_ratio)
    return sigmaP, sigmaH


def resistance_from_log_conductance_coordinates(
    log_magnitude, log_ratio, *, reference_conductance=CONDUCTANCE_REFERENCE_S
):
    """Reconstruct positive resistance from log conductance.

    Conductance and resistance have reciprocal magnitudes and the same
    Hall/Pedersen ratio. Working directly in log coordinates avoids a
    second pointwise inversion and remains stable for large ratios.
    """
    reference_conductance = _validate_reference_conductance(reference_conductance)
    xp = get_array_module(log_magnitude, log_ratio)
    log_magnitude = xp.asarray(log_magnitude)
    log_ratio = xp.asarray(log_ratio)
    log_pedersen = (
        -xp.log(reference_conductance) - log_magnitude - 0.5 * xp.logaddexp(0.0, 2.0 * log_ratio)
    )
    etaP = xp.exp(log_pedersen)
    etaH = xp.exp(log_pedersen + log_ratio)
    return etaP, etaH


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
    etaP = xp.asarray(etaP)
    etaH = xp.asarray(etaH)
    pedersen_geometry = xp.asarray(pedersen_geometry)
    hall_geometry = xp.asarray(hall_geometry)
    return etaP * pedersen_geometry + etaH * hall_geometry


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


def _cross_spherical(a_r, a_theta, a_phi, b_r, b_theta, b_phi):
    """Return a cross product in the local spherical basis."""
    return (
        a_theta * b_phi - a_phi * b_theta,
        a_phi * b_r - a_r * b_phi,
        a_r * b_theta - a_theta * b_r,
    )


def _current_from_weighted_winds(
    *, sigma_p, sigma_h, u_p_theta, u_p_phi, u_h_theta, u_h_phi, field
):
    """Return the height-integrated 3D wind-current source.

    ``u_p`` and ``u_h`` are the separate conductivity-weighted column
    means. With the thin-sheet approximation that the main field is
    constant through the dynamo region, multiplying them by ``sigma_p``
    and ``sigma_h`` reconstructs the wind moments in Appendix A,
    Eqs. (A3)-(A4), of Laundal et al. (2025).
    """
    xp = get_array_module(
        sigma_p,
        sigma_h,
        u_p_theta,
        u_p_phi,
        u_h_theta,
        u_h_phi,
        field.unit_br,
        field.unit_btheta,
        field.unit_bphi,
        field.Br,
        field.Btheta,
        field.Bphi,
    )
    sigma_p = xp.asarray(sigma_p, dtype=float).reshape(-1)
    sigma_h = xp.asarray(sigma_h, dtype=float).reshape(-1)
    u_p_theta = xp.asarray(u_p_theta, dtype=float).reshape(-1)
    u_p_phi = xp.asarray(u_p_phi, dtype=float).reshape(-1)
    u_h_theta = xp.asarray(u_h_theta, dtype=float).reshape(-1)
    u_h_phi = xp.asarray(u_h_phi, dtype=float).reshape(-1)

    b_r = xp.asarray(field.unit_br, dtype=float).reshape(-1)
    b_theta = xp.asarray(field.unit_btheta, dtype=float).reshape(-1)
    b_phi = xp.asarray(field.unit_bphi, dtype=float).reshape(-1)
    B_r = xp.asarray(field.Br, dtype=float).reshape(-1)
    B_theta = xp.asarray(field.Btheta, dtype=float).reshape(-1)
    B_phi = xp.asarray(field.Bphi, dtype=float).reshape(-1)

    zero = xp.zeros_like(u_p_theta)
    u_p_cross_B = _cross_spherical(zero, u_p_theta, u_p_phi, B_r, B_theta, B_phi)
    u_h_cross_B = _cross_spherical(zero, u_h_theta, u_h_phi, B_r, B_theta, B_phi)
    hall_current = _cross_spherical(b_r, b_theta, b_phi, *u_h_cross_B)
    return (
        sigma_p * u_p_cross_B[0] + sigma_h * hall_current[0],
        sigma_p * u_p_cross_B[1] + sigma_h * hall_current[1],
        sigma_p * u_p_cross_B[2] + sigma_h * hall_current[2],
    )


def electric_field_from_weighted_winds(
    *, sigma_p, sigma_h, u_p_theta, u_p_phi, u_h_theta, u_h_phi, field, eta_p, eta_h
):
    """Return equivalent E from height-integrated neutral-wind current.

    ``u_p`` and ``u_h`` are winds averaged separately with Pedersen and
    Hall conductivity. Together with their conductances, they represent
    the two height integrals of the neutral-wind current in Appendix A
    of Laundal et al. (2025). The resulting three-dimensional source is
    inverted in the plane perpendicular to the main field and only then
    projected onto the spherical sheet.

    This is algebraically equivalent to applying the thin-sheet
    resistance tensor to ``-Q_eff`` away from the dip equator. It avoids
    explicitly forming ``Q_eff``, whose infinite-parallel-conductance
    expression divides by the radial direction cosine and is singular
    at the dip equator. Returned components follow PynaMIT's
    ``E_neutral_wind`` convention.
    """
    q_r, q_theta, q_phi = _current_from_weighted_winds(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
    )
    xp = get_array_module(q_r, q_theta, q_phi, eta_p, eta_h)
    eta_p = xp.asarray(eta_p, dtype=float).reshape(-1)
    eta_h = xp.asarray(eta_h, dtype=float).reshape(-1)
    b_r = xp.asarray(field.unit_br, dtype=float).reshape(-1)
    b_theta = xp.asarray(field.unit_btheta, dtype=float).reshape(-1)
    b_phi = xp.asarray(field.unit_bphi, dtype=float).reshape(-1)

    q_dot_b = q_r * b_r + q_theta * b_theta + q_phi * b_phi
    q_perp_theta = q_theta - q_dot_b * b_theta
    q_perp_phi = q_phi - q_dot_b * b_phi
    q_cross_b = _cross_spherical(q_r, q_theta, q_phi, b_r, b_theta, b_phi)
    return (
        -(eta_p * q_perp_theta + eta_h * q_cross_b[1]),
        -(eta_p * q_perp_phi + eta_h * q_cross_b[2]),
    )


def joule_heating_from_current(sheet_current, etaP, pedersen_geometry):
    """Return collisional Joule heating ``etaP * J.T @ P @ J``."""
    xp = get_array_module(sheet_current, etaP, pedersen_geometry)
    sheet_current = xp.asarray(sheet_current)
    etaP = xp.asarray(etaP)
    pedersen_geometry = xp.asarray(pedersen_geometry)
    return etaP * xp.einsum(
        "ig,ijg,jg->g", sheet_current, pedersen_geometry, sheet_current, optimize=True
    )


def wind_to_E_coeffs_operator(
    helmholtz_analysis_operator: LinearMap, wind_to_E_grid, wind_synthesis_operator: LinearMap
) -> LinearMap:
    """Return the operator mapping neutral-wind coefficients to E."""
    return (
        helmholtz_analysis_operator
        @ pointwise_matrix_linear_map(wind_to_E_grid)
        @ wind_synthesis_operator
    )


def tangential_current_to_E_coeffs_operator(
    helmholtz_analysis_operator: LinearMap,
    resistance_tensor,
    sheet_current_synthesis_operator: LinearMap,
) -> LinearMap:
    """Map sheet-current coefficients to E coefficients."""
    return (
        helmholtz_analysis_operator
        @ pointwise_matrix_linear_map(resistance_tensor)
        @ sheet_current_synthesis_operator
    )


def Q_eff_on_grid_from_wind(wind_on_grid, wind_to_E_grid, resistance_tensor):
    """Return the effective sheet current equivalent to neutral wind."""
    xp = get_array_module(wind_on_grid, wind_to_E_grid, resistance_tensor)
    E_wind_on_grid = xp.einsum(
        "abg,bg->ag", xp.asarray(wind_to_E_grid), xp.asarray(wind_on_grid), optimize=True
    )
    point_resistance = xp.moveaxis(xp.asarray(resistance_tensor), -1, 0)
    return xp.linalg.solve(point_resistance, E_wind_on_grid.T[..., None])[..., 0].T


def solve_Q_eff_coefficients(
    Q_eff_to_E_operator: LinearMap, E_wind_coeffs, *, reg_lambda=None, pinv_rtol=1e-15
):
    """Fit Q_eff, adding ``reg_lambda * ||Q_eff||²`` when requested."""
    xp = get_array_module(E_wind_coeffs, *Q_eff_to_E_operator.backend_operands)
    backend = "numpy" if xp is np else "jax"
    matrix = xp.asarray(Q_eff_to_E_operator.to_matrix(backend=backend))
    rhs = xp.asarray(E_wind_coeffs).reshape(-1)
    tolerance = float(pinv_rtol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("pinv_rtol must be finite and non-negative.")
    weight = 0.0 if reg_lambda is None else float(reg_lambda)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("reg_lambda must be finite and non-negative.")
    if weight > 0.0:
        regularization = weight**0.5 * xp.eye(matrix.shape[1], dtype=matrix.dtype)
        matrix = xp.vstack([matrix, regularization])
        rhs = xp.concatenate([rhs, xp.zeros(matrix.shape[1], dtype=rhs.dtype)])
    coefficients, *_ = xp.linalg.lstsq(matrix, rhs, rcond=tolerance)
    return coefficients


__all__ = [
    "CONDUCTANCE_REFERENCE_S",
    "Q_eff_on_grid_from_wind",
    "conductance_from_log_coordinates",
    "conductance_to_log_coordinates",
    "conductance_to_resistance",
    "electric_field_from_weighted_winds",
    "electric_field_on_grid",
    "hall_geometry_tensor",
    "joule_heating_from_current",
    "pedersen_geometry_tensor",
    "resistance_from_log_conductance_coordinates",
    "resistance_tensor_on_grid",
    "resistance_to_conductance",
    "resistance_to_log_conductance_coordinates",
    "solve_Q_eff_coefficients",
    "tangential_current_to_E_coeffs_operator",
    "wind_motional_E_tensor",
    "wind_to_E_coeffs_operator",
]
