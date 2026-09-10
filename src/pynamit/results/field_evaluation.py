"""Evaluate coefficient fields on spherical grids."""

from __future__ import annotations

import numpy as np
from kompe import SphericalGrid
from kompe.math import get_array_module

from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_from_log_coordinates,
    resistance_from_log_conductance_coordinates,
)


def model_grid_from_geographic(main_field, latitude, longitude):
    """Convert geographic degrees to model-frame sample points."""
    latitude, longitude = main_field.geo_to_model_coordinates(latitude, longitude)
    return SphericalGrid(lat=latitude, lon=longitude)


def model_to_geographic_tangential_array(main_field, grid):
    """Return pointwise model-to-geographic theta/phi rotations.

    Provider transformations run on the CPU once. The returned array
    rotates arbitrary field/time blocks on their backend.
    """
    ones, zeros = np.ones(grid.size), np.zeros(grid.size)
    _, _, east_from_theta, north_from_theta = main_field.model_to_geo_coordinates(
        grid.lat, grid.lon, east=zeros, north=-ones
    )
    _, _, east_from_phi, north_from_phi = main_field.model_to_geo_coordinates(
        grid.lat, grid.lon, east=ones, north=zeros
    )
    return np.array([[-north_from_theta, -north_from_phi], [east_from_theta, east_from_phi]])


def evaluate_conductance_values(log_magnitude, log_ratio):
    """Return canonical and physical closure values on one grid."""
    xp = get_array_module(log_magnitude, log_ratio)
    log_magnitude = xp.asarray(log_magnitude, dtype=float)
    log_ratio = xp.asarray(log_ratio, dtype=float)
    SigmaP, SigmaH = conductance_from_log_coordinates(log_magnitude, log_ratio)
    etaP, etaH = resistance_from_log_conductance_coordinates(log_magnitude, log_ratio)
    return {
        "log_conductance_magnitude": log_magnitude,
        "log_hall_to_pedersen_ratio": log_ratio,
        "etaP": etaP,
        "etaH": etaH,
        "SigmaP": SigmaP,
        "SigmaH": SigmaH,
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


__all__ = [
    "evaluate_conductance_coefficients",
    "evaluate_conductance_values",
    "evaluate_tangential_coefficients",
    "evaluate_wind_coefficients",
]
