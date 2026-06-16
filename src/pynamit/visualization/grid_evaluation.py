"""Grid and coefficient-evaluation helpers for visualization."""

import numpy as np

from pynamit.math.constants import mu0
from pynamit.simulation.config import setting_value
from pynamit.simulation.sheet_current import sheet_current_operator_bundle
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


def compute_conversion_factors(settings, sh_basis):
    """Compute common SH coefficient conversion factors."""
    ri = float(setting_value(settings, "RI"))
    solid_harmonics = SolidHarmonics(sh_basis)
    return {
        "RI": ri,
        "m_ind_to_Br": -(ri**2) * sh_basis.laplacian(ri),
        "m_imp_to_jr": ri / mu0 * sh_basis.laplacian(ri),
        "m_ind_to_Jeq": (-ri / mu0 * solid_harmonics.poloidal_to_boundary_potential_jump_factor),
    }


def build_sheet_current_operators(settings, sh_basis, transform, T_to_Ve=None):
    """Build common coefficient-to-sheet-current matrices.

    This is the low-level matrix bundle used by notebook and script
    visualizations that operate directly on saved coefficient arrays.
    """
    rm = setting_value(settings, "RM", None)
    if rm not in (None, 0, 0.0):
        rm = float(rm)
    else:
        rm = None
    return sheet_current_operator_bundle(
        SolidHarmonics(sh_basis),
        transform,
        radius=float(setting_value(settings, "RI")),
        boundary_radius=rm,
        boundary_shielding=bool(setting_value(settings, "RM_shielding", False)),
        T_to_Ve=T_to_Ve,
    )


def resistance_to_conductance(etaP, etaH):
    """Convert Pedersen/Hall resistance coefficients to conductance."""
    etaP = np.asarray(etaP, dtype=float)
    etaH = np.asarray(etaH, dtype=float)
    den = etaP**2 + etaH**2
    valid = np.isfinite(den) & (den > np.finfo(float).tiny)
    sigmaP = np.full_like(etaP, np.nan, dtype=float)
    sigmaH = np.full_like(etaH, np.nan, dtype=float)
    sigmaP[valid] = etaP[valid] / den[valid]
    sigmaH[valid] = etaH[valid] / den[valid]
    return sigmaP, sigmaH


__all__ = [
    "build_evaluator",
    "build_plot_grid",
    "build_sheet_current_operators",
    "compute_conversion_factors",
    "load_settings_and_basis",
    "resistance_to_conductance",
    "transform_for_source",
]
