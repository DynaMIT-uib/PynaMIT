"""Evaluate coefficient fields on spherical grids."""

from __future__ import annotations

from kompe import SolidHarmonicOperators
from kompe.constants import MU0
from kompe.math import as_linear_map, get_array_module

from pynamit.simulation.config import setting_value
from pynamit.simulation.electrodynamics import magnetic_boundary
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_from_log_coordinates,
    conductance_to_resistance,
)


def apply_coefficient_operator(operator, coefficients):
    """Apply a linear operator to one flattened coefficient field."""
    xp = get_array_module(coefficients)
    return as_linear_map(operator).matvec(xp.asarray(coefficients).reshape(-1))


def evaluate_conductance_values(log_magnitude, log_ratio):
    """Return canonical and physical closure values on one grid."""
    xp = get_array_module(log_magnitude, log_ratio)
    log_magnitude = xp.asarray(log_magnitude, dtype=float)
    log_ratio = xp.asarray(log_ratio, dtype=float)
    sigmaP, sigmaH = conductance_from_log_coordinates(log_magnitude, log_ratio)
    etaP, etaH = conductance_to_resistance(sigmaP, sigmaH)
    return {
        "log_conductance_magnitude": log_magnitude,
        "log_hall_to_pedersen_ratio": log_ratio,
        "etaP": etaP,
        "etaH": etaH,
        "SigmaP": sigmaP,
        "SigmaH": sigmaH,
    }


def evaluate_conductance_coefficients(transform, log_magnitude_coeffs, log_ratio_coeffs):
    """Evaluate canonical coordinates and physical conductance."""
    log_magnitude = transform.synthesize_scalar(log_magnitude_coeffs)
    log_ratio = transform.synthesize_scalar(log_ratio_coeffs)
    return evaluate_conductance_values(log_magnitude, log_ratio)


def evaluate_tangential_coefficients(transform, coeffs, *, include_magnitude=True):
    """Evaluate Helmholtz tangential-field coefficients."""
    theta_component, phi_component = transform.synthesize_helmholtz(coeffs)
    values = {"theta": theta_component, "phi": phi_component}
    if include_magnitude:
        xp = get_array_module(theta_component, phi_component)
        values["magnitude"] = xp.hypot(theta_component, phi_component)
    return values


def evaluate_wind_coefficients(transform, coeffs, *, include_magnitude=True):
    """Evaluate wind coefficients with plotting-friendly directions."""
    components = evaluate_tangential_coefficients(
        transform, coeffs, include_magnitude=include_magnitude
    )
    values = {
        "u_theta": components["theta"],
        "u_phi": components["phi"],
        "u_north": -components["theta"],
        "u_east": components["phi"],
    }
    if include_magnitude:
        values["u_mag"] = components["magnitude"]
    return values


def evaluate_sheet_current_from_operators(
    boundary_jr,
    induced_Br,
    *,
    boundary_jr_to_JS,
    induced_Br_to_JS,
    boundary_Br=None,
    boundary_Br_to_JS=None,
):
    """Evaluate sheet current from physical magnetic quantities."""
    current = apply_coefficient_operator(
        boundary_jr_to_JS, boundary_jr
    ) + apply_coefficient_operator(induced_Br_to_JS, induced_Br)
    if boundary_Br is not None:
        if boundary_Br_to_JS is None:
            raise ValueError("boundary_Br_to_JS is required when boundary_Br is provided.")
        current += apply_coefficient_operator(boundary_Br_to_JS, boundary_Br)
    return current.reshape(2, -1)


def build_sheet_current_matrices(settings, sh_basis, transform, boundary_jr_to_gap_Br_matrix=None):
    """Build coefficient-to-JS matrices for direct array workflows."""
    rm = setting_value(settings, "RM", None)
    rm = None if rm in (None, 0, 0.0) else float(rm)
    solid_harmonics = SolidHarmonicOperators(sh_basis)
    radius = float(setting_value(settings, "RI"))
    induced_Br_to_JS_matrix = magnetic_boundary.induced_Br_to_gridded_JS_operator(
        solid_harmonics,
        transform,
        radius=radius,
        boundary_radius=rm,
        boundary_shielding=bool(setting_value(settings, "magnetic_boundary_shielding", False)),
    ).array
    boundary_jr_to_toroidal_potential = (
        MU0 / radius * sh_basis.mean_free_surface_poisson_operator(radius)
    )
    boundary_jr_to_gap_Br = (
        None
        if boundary_jr_to_gap_Br_matrix is None
        else as_linear_map(boundary_jr_to_gap_Br_matrix)
    )
    boundary_jr_to_JS_matrix = magnetic_boundary.boundary_jr_to_gridded_JS_operator(
        solid_harmonics,
        transform,
        poloidal_transform=transform,
        boundary_jr_to_toroidal_potential=boundary_jr_to_toroidal_potential,
        boundary_jr_to_gap_Br=boundary_jr_to_gap_Br,
    ).array
    boundary_Br_to_JS_matrix = (
        None
        if rm is None
        else magnetic_boundary.boundary_Br_to_gridded_JS_operator(
            solid_harmonics, transform, radius=radius, boundary_radius=rm
        ).array
    )
    return {
        "induced_Br_to_JS": induced_Br_to_JS_matrix,
        "boundary_jr_to_JS": boundary_jr_to_JS_matrix,
        "boundary_Br_to_JS": boundary_Br_to_JS_matrix,
    }


__all__ = [
    "apply_coefficient_operator",
    "build_sheet_current_matrices",
    "evaluate_sheet_current_from_operators",
    "evaluate_conductance_coefficients",
    "evaluate_conductance_values",
    "evaluate_tangential_coefficients",
    "evaluate_wind_coefficients",
]
