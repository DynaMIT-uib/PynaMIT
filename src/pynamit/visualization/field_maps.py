"""Reusable field maps for visualization frontends."""

from __future__ import annotations

import numpy as np
from kompe.math import as_linear_map

from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_from_log_coordinates,
    conductance_to_resistance,
)


def _coefficient_array(coeffs):
    """Return coefficient values from an array-like or field object."""
    return np.asarray(getattr(coeffs, "array", coeffs))


def _apply_linear_map(linear_map, coeffs):
    """Apply any supported LinearMap input to coefficients."""
    coeffs = _coefficient_array(coeffs).reshape(-1)
    return as_linear_map(linear_map).matvec(coeffs)


def evaluate_conductance_values(log_magnitude, log_ratio):
    """Return canonical and physical closure values on one grid."""
    log_magnitude = np.asarray(log_magnitude, dtype=float)
    log_ratio = np.asarray(log_ratio, dtype=float)
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
        values["magnitude"] = np.sqrt(theta_component**2 + phi_component**2)
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


def evaluate_JS_from_maps(
    boundary_jr,
    induced_Br,
    *,
    boundary_jr_to_JS,
    induced_Br_to_JS,
    boundary_Br=None,
    boundary_Br_to_JS=None,
):
    """Evaluate sheet current from physical magnetic quantities."""
    current = _apply_linear_map(boundary_jr_to_JS, boundary_jr) + _apply_linear_map(
        induced_Br_to_JS, induced_Br
    )
    if boundary_Br is not None:
        if boundary_Br_to_JS is None:
            raise ValueError("boundary_Br_to_JS is required when boundary_Br is provided.")
        current += _apply_linear_map(boundary_Br_to_JS, boundary_Br)
    return current.reshape(2, -1)


__all__ = [
    "evaluate_JS_from_maps",
    "evaluate_conductance_coefficients",
    "evaluate_conductance_values",
    "evaluate_tangential_coefficients",
    "evaluate_wind_coefficients",
]
