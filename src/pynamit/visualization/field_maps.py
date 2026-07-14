"""Reusable field maps for visualization frontends."""

from __future__ import annotations

import numpy as np

from pynamit.math.linear_map import as_linear_map
from pynamit.simulation.electrodynamics.ionospheric_closure import resistance_to_conductance


def _coefficient_array(coeffs):
    """Return coefficient values from an array-like or field object."""
    return np.asarray(getattr(coeffs, "array", coeffs))


def _apply_linear_map(linear_map, coeffs):
    """Apply any supported LinearMap input to coefficients."""
    coeffs = _coefficient_array(coeffs).reshape(-1)
    return as_linear_map(linear_map).matvec(coeffs)


def evaluate_conductance_values(etaP, etaH):
    """Return resistance and physical conductance values on one grid."""
    etaP = np.asarray(etaP, dtype=float)
    etaH = np.asarray(etaH, dtype=float)
    sigmaP, sigmaH = resistance_to_conductance(etaP, etaH)
    return {"etaP": etaP, "etaH": etaH, "SigmaP": sigmaP, "SigmaH": sigmaH}


def evaluate_conductance_coefficients(transform, etaP_coeffs, etaH_coeffs):
    """Evaluate resistance coefficients and derived conductance."""
    etaP = transform.synthesize_scalar(etaP_coeffs)
    etaH = transform.synthesize_scalar(etaH_coeffs)
    return evaluate_conductance_values(etaP, etaH)


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


def evaluate_JS_from_maps(m_imp, m_ind, *, m_imp_to_JS, m_ind_to_JS, Br=None, Br_to_JS=None):
    """Evaluate total JS from magnetic and boundary-field maps."""
    current = _apply_linear_map(m_imp_to_JS, m_imp) + _apply_linear_map(m_ind_to_JS, m_ind)
    if Br is not None:
        if Br_to_JS is None:
            raise ValueError("Br_to_JS is required when Br coefficients are provided.")
        current += _apply_linear_map(Br_to_JS, Br)
    return current.reshape(2, -1)


__all__ = [
    "evaluate_conductance_coefficients",
    "evaluate_conductance_values",
    "evaluate_JS_from_maps",
    "evaluate_tangential_coefficients",
    "evaluate_wind_coefficients",
]
