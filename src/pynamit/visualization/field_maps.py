"""Reusable field maps for visualization frontends."""

from __future__ import annotations

import numpy as np

from pynamit.math.linear_map import as_linear_map
from pynamit.visualization.grid_evaluation import resistance_to_conductance


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
    return {
        "etaP": etaP,
        "etaH": etaH,
        "SigmaP": sigmaP,
        "SigmaH": sigmaH,
    }


def evaluate_conductance_coefficients(transform, etaP_coeffs, etaH_coeffs):
    """Evaluate resistance coefficients and derived conductance."""
    etaP = transform.synthesize_scalar(etaP_coeffs)
    etaH = transform.synthesize_scalar(etaH_coeffs)
    return evaluate_conductance_values(etaP, etaH)


def evaluate_tangential_coefficients(transform, coeffs, *, include_magnitude=True):
    """Evaluate Helmholtz tangential-field coefficients."""
    theta_component, phi_component = transform.synthesize_helmholtz(coeffs)
    values = {
        "theta": theta_component,
        "phi": phi_component,
    }
    if include_magnitude:
        values["magnitude"] = np.sqrt(theta_component**2 + phi_component**2)
    return values


def evaluate_wind_coefficients(transform, coeffs, *, include_magnitude=True):
    """Evaluate wind coefficients with plotting-friendly directions."""
    components = evaluate_tangential_coefficients(
        transform,
        coeffs,
        include_magnitude=include_magnitude,
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


def evaluate_electric_field_coefficients(transform, Phi, W, radius):
    """Evaluate E from physical potential coefficients.

    ``Phi`` and ``W`` must be volt-scaled potential coefficients.
    Saved state output stores Helmholtz E-field coefficients in the
    unit-sphere gradient convention; multiply those coefficients by
    ``radius`` before using this helper.
    """
    coeffs = np.stack(
        [
            _coefficient_array(Phi).reshape(-1),
            _coefficient_array(W).reshape(-1),
        ],
    )
    return transform.synthesize_helmholtz(coeffs) / float(radius)


def evaluate_joule_from_fields(sheet_current, electric_field):
    """Evaluate Joule heating from sheet and electric field values."""
    sheet_current = np.asarray(sheet_current).reshape(2, -1)
    electric_field = np.asarray(electric_field).reshape(2, -1)
    return (
        sheet_current[0] * electric_field[0]
        + sheet_current[1] * electric_field[1]
    )


def evaluate_sheet_current_from_maps(
    m_imp,
    m_ind,
    *,
    m_imp_to_sheet,
    m_ind_to_sheet,
):
    """Evaluate total sheet current from coefficient maps."""
    return (
        _apply_linear_map(m_imp_to_sheet, m_imp)
        + _apply_linear_map(m_ind_to_sheet, m_ind)
    ).reshape(2, -1)


def evaluate_joule_from_coefficients(
    transform,
    m_imp,
    m_ind,
    Phi,
    W,
    radius,
    *,
    m_imp_to_sheet,
    m_ind_to_sheet,
):
    """Evaluate Joule heating from source and E coefficients."""
    electric_field = evaluate_electric_field_coefficients(transform, Phi, W, radius)
    sheet_current = evaluate_sheet_current_from_maps(
        m_imp,
        m_ind,
        m_imp_to_sheet=m_imp_to_sheet,
        m_ind_to_sheet=m_ind_to_sheet,
    )
    joule = evaluate_joule_from_fields(sheet_current, electric_field)
    return joule, electric_field, sheet_current


__all__ = [
    "evaluate_conductance_coefficients",
    "evaluate_conductance_values",
    "evaluate_electric_field_coefficients",
    "evaluate_joule_from_coefficients",
    "evaluate_joule_from_fields",
    "evaluate_sheet_current_from_maps",
    "evaluate_tangential_coefficients",
    "evaluate_wind_coefficients",
]
