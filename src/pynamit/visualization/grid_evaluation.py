"""Grid and coefficient-evaluation helpers for visualization."""

import numpy as np

from pynamit.simulation.config import setting_value
from pynamit.simulation.magnetic_boundary import JS_operator_bundle
from pynamit.sphere import Grid, SHBasis, SolidHarmonics, SphericalTransform
from pynamit.visualization.artifacts import load_dataset_artifact


def load_settings_and_basis(settings_path):
    """Load run settings and construct the associated SH basis."""
    settings = load_dataset_artifact(settings_path)
    nmax = int(setting_value(settings, "Nmax"))
    mmax = int(setting_value(settings, "Mmax"))
    return settings, SHBasis(nmax, mmax)


def build_plot_grid(nlat=60, nlon=100, lat_range=(-89.9, 89.9), lon_range=(-180.0, 180.0)):
    """Build a regular latitude/longitude plotting grid."""
    lat_1d = np.linspace(lat_range[0], lat_range[1], int(nlat))
    lon_1d = np.linspace(lon_range[0], lon_range[1], int(nlon))
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    return lat_2d, lon_2d, Grid(lat=lat_2d, lon=lon_2d)


def build_evaluator(basis, grid, **kwargs):
    """Build a spherical transform for a plotting grid."""
    return SphericalTransform(basis, grid, **kwargs)


def transform_for_source(source, transform):
    """Return ``transform`` or an equivalent one for ``source``."""
    if transform.source.coefficients_are_compatible_with(source):
        return transform
    return SphericalTransform(
        source,
        transform.target,
        sqrt_weights=(transform.sqrt_weights if transform.explicit_sqrt_weights else None),
        reg_lambda=transform.reg_lambda,
        pinv_rtol=transform.pinv_rtol,
        area_weighted=transform.area_weighted,
    )


def build_JS_operators(settings, sh_basis, transform, T_to_Ve=None):
    """Build common coefficient-to-JS matrices.

    This is the low-level matrix bundle used by notebook and script
    visualizations that operate directly on saved coefficient arrays.
    """
    rm = setting_value(settings, "RM", None)
    if rm not in (None, 0, 0.0):
        rm = float(rm)
    else:
        rm = None
    return JS_operator_bundle(
        SolidHarmonics(sh_basis),
        transform,
        radius=float(setting_value(settings, "RI")),
        boundary_radius=rm,
        boundary_shielding=bool(setting_value(settings, "RM_shielding", False)),
        T_to_Ve=T_to_Ve,
    )
__all__ = [
    "build_evaluator",
    "build_plot_grid",
    "build_JS_operators",
    "load_settings_and_basis",
    "transform_for_source",
]
