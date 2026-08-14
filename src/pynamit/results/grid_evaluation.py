"""Spherical-grid and coefficient-evaluation helpers."""

import numpy as np
from kompe import SolidHarmonicOperators, SphericalGrid, SphericalTransform
from kompe.constants import MU0
from kompe.math import as_linear_map

from pynamit.simulation.config import setting_value
from pynamit.simulation.electrodynamics import magnetic_boundary


def build_plot_grid(nlat=60, nlon=100, lat_range=(-89.9, 89.9), lon_range=(-180.0, 180.0)):
    """Build a regular latitude/longitude plotting grid."""
    lat_1d = np.linspace(lat_range[0], lat_range[1], int(nlat))
    lon_1d = np.linspace(lon_range[0], lon_range[1], int(nlon))
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    return lat_2d, lon_2d, SphericalGrid(lat=lat_2d, lon=lon_2d)


def model_grid_for_geographic_display(main_field, lat, lon, *, event_time=None):
    """Return the model-coordinate grid underlying a geographic map."""
    model_lat, model_lon = main_field.geo_to_model_coordinates(lat, lon, event_time=event_time)
    return SphericalGrid(lat=model_lat, lon=model_lon)


def transform_for_basis(basis, transform):
    """Return ``transform`` or an equivalent one for ``basis``."""
    if transform.basis.coefficients_are_compatible_with(basis):
        return transform
    return SphericalTransform(
        basis,
        transform.grid,
        sqrt_weights=(transform.sqrt_weights if transform.explicit_sqrt_weights else None),
        reg_lambda=transform.reg_lambda,
        pinv_rtol=transform.pinv_rtol,
        area_weighted=transform.area_weighted,
    )


def build_JS_matrices(settings, sh_basis, transform, boundary_jr_to_gap_Br_matrix=None):
    """Build common coefficient-to-JS matrices.

    This is the low-level matrix bundle used by notebook and script
    visualizations that operate directly on saved coefficient arrays.
    """
    rm = setting_value(settings, "RM", None)
    if rm not in (None, 0, 0.0):
        rm = float(rm)
    else:
        rm = None
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
    if boundary_jr_to_gap_Br_matrix is None:
        boundary_jr_to_gap_Br_matrix = np.zeros((sh_basis.index_length, sh_basis.index_length))
    boundary_jr_to_JS_matrix = magnetic_boundary.boundary_jr_to_gridded_JS_operator(
        solid_harmonics,
        transform,
        poloidal_transform=transform,
        boundary_jr_to_toroidal_potential=boundary_jr_to_toroidal_potential,
        boundary_jr_to_gap_Br=as_linear_map(boundary_jr_to_gap_Br_matrix),
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
    "build_JS_matrices",
    "build_plot_grid",
    "model_grid_for_geographic_display",
    "transform_for_basis",
]
