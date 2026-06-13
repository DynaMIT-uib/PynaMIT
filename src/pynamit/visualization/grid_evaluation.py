"""Grid and coefficient-evaluation helpers for visualization."""

import numpy as np

from pynamit.math.constants import mu0
from pynamit.simulation.schema import setting_value
from pynamit.sphere import Grid, SHBasis, SolidHarmonics, SphericalTransform
from pynamit.visualization.artifacts import load_dataset_artifact


def load_settings_and_basis(settings_path):
    """Load run settings and construct the associated SH basis."""
    settings = load_dataset_artifact(settings_path)
    nmax = int(setting_value(settings, "Nmax"))
    mmax = int(setting_value(settings, "Mmax"))
    return settings, SHBasis(nmax, mmax)


def build_plot_grid(
    nlat=60,
    nlon=100,
    lat_range=(-89.9, 89.9),
    lon_range=(-180.0, 180.0),
):
    """Build a regular latitude/longitude plotting grid."""
    lat_1d = np.linspace(lat_range[0], lat_range[1], int(nlat))
    lon_1d = np.linspace(lon_range[0], lon_range[1], int(nlon))
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    return lat_2d, lon_2d, Grid(lat=lat_2d, lon=lon_2d)


def build_evaluator(basis, grid, **kwargs):
    """Build a spherical transform for a plotting grid."""
    return SphericalTransform(basis, grid, **kwargs)


def compute_conversion_factors(settings, sh_basis):
    """Compute common SH coefficient conversion factors."""
    ri = float(setting_value(settings, "RI"))
    solid_harmonics = SolidHarmonics(sh_basis)
    return {
        "RI": ri,
        "m_ind_to_Br": -(ri**2) * sh_basis.laplacian(ri),
        "m_imp_to_jr": ri / mu0 * sh_basis.laplacian(ri),
        "m_ind_to_Jeq": (
            -ri / mu0 * solid_harmonics.poloidal_to_boundary_potential_jump_factor
        ),
    }


def build_sheet_current_operators(settings, sh_basis, transform, T_to_Ve=None):
    """Build common coefficient-to-sheet-current matrices.

    This is the low-level matrix bundle used by notebook and script
    visualizations that operate directly on saved coefficient arrays.
    """
    ri = float(setting_value(settings, "RI"))
    solid_harmonics = SolidHarmonics(sh_basis)
    ve_to_j_df_coeffs = (
        -ri / mu0 * solid_harmonics.poloidal_to_boundary_potential_jump_factor
    )
    poloidal_to_sheet = (
        transform.scalar_coeffs_to_gridded_rhat_cross_gradient
        * (ve_to_j_df_coeffs / ri)
    )
    toroidal_to_sheet = -transform.scalar_coeffs_to_gridded_gradient * (1.0 / mu0)

    m_imp_to_sheet = toroidal_to_sheet
    if T_to_Ve is not None:
        m_imp_to_sheet = m_imp_to_sheet + np.tensordot(
            poloidal_to_sheet,
            np.asarray(T_to_Ve),
            axes=([2], [0]),
        )

    m_ind_to_sheet = poloidal_to_sheet
    Br_to_sheet = np.zeros_like(m_ind_to_sheet)
    rm = setting_value(settings, "RM", None)
    if rm not in (None, 0, 0.0):
        br_shift = solid_harmonics.regular_reference_shift(rm, ri)
        vi_shift = solid_harmonics.irregular_reference_shift(ri, rm)
        denominator = 1.0 - br_shift * vi_shift
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
        m_ind_to_Br = -(ri**2) * sh_basis.laplacian(ri)
        Br_to_sheet = poloidal_to_sheet * (
            -br_shift / (safe_denominator * m_ind_to_Br)
        )
        m_ind_to_sheet = poloidal_to_sheet * (
            1.0 + br_shift * vi_shift / safe_denominator
        )

    return {
        "G_m_ind_to_JS": m_ind_to_sheet,
        "G_m_imp_to_JS": m_imp_to_sheet,
        "G_Br_to_JS": Br_to_sheet,
    }


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
]
