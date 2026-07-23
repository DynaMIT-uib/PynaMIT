"""Derived sheet-current operators from magnetic-boundary fields.

In PynaMIT's B, v formulation, JS is a derived horizontal surface
current rather than an independent state.  These functions map magnetic
potential coefficients and optional boundary field data to JS on a grid.
SimulationGeometry supplies the run-specific radii, transforms, and
shielding.
"""

from __future__ import annotations

import numpy as np

from pynamit.math import as_linear_map, diagonal_linear_map
from pynamit.math.constants import mu0


def _coefficient_scale(values):
    """Return a one-dimensional coefficient-space scale."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"coefficient scale must be one-dimensional; got shape {array.shape}.")
    return array.copy()


def poloidal_to_gridded_JS_operator(solid_harmonics, transform, *, poloidal_scale=None):
    """Return the map from poloidal coefficients to gridded JS."""
    scale = _coefficient_scale(solid_harmonics.poloidal_to_boundary_potential_jump_factor)
    if poloidal_scale is not None:
        scale = scale * np.asarray(poloidal_scale)
    return (
        (-1.0 / mu0)
        * transform.scalar_coeffs_to_gridded_rhat_cross_gradient_operator
        @ diagonal_linear_map(scale)
    )


def poloidal_to_gridded_JS(solid_harmonics, transform, *, poloidal_scale=None):
    """Map poloidal coefficients to gridded JS."""
    return poloidal_to_gridded_JS_operator(
        solid_harmonics, transform, poloidal_scale=poloidal_scale
    ).array


def shielded_m_ind_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Return the optional zero-Br image response for ``m_ind``."""
    regular_shift = _coefficient_scale(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = _coefficient_scale(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    denominator = 1.0 - regular_shift * irregular_shift
    return 1.0 / denominator


def boundary_Br_to_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Continue prescribed ``Br(RM)`` to its ionospheric potential jump.

    The regular source field and its irregular response sum to the
    supplied radial field at ``boundary_radius``. This continuation is
    intrinsic to the boundary input and is separate from optional
    shielding of the evolving ``m_ind`` field.
    """
    regular_shift = _coefficient_scale(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = _coefficient_scale(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    denominator = 1.0 - regular_shift * irregular_shift
    m_ind_to_Br = _coefficient_scale(-(radius**2) * solid_harmonics.basis.laplacian(radius))
    return -regular_shift / denominator / m_ind_to_Br


def m_ind_to_gridded_JS(
    solid_harmonics, transform, *, radius, boundary_radius=None, boundary_shielding=False
):
    """Map induced-potential coefficients to gridded JS."""
    return m_ind_to_gridded_JS_operator(
        solid_harmonics,
        transform,
        radius=radius,
        boundary_radius=boundary_radius,
        boundary_shielding=boundary_shielding,
    ).array


def m_ind_to_gridded_JS_operator(
    solid_harmonics, transform, *, radius, boundary_radius=None, boundary_shielding=False
):
    """Return the map from induced-potential coefficients to JS."""
    poloidal_scale = None
    if boundary_radius is not None and boundary_shielding:
        poloidal_scale = shielded_m_ind_poloidal_scale(solid_harmonics, boundary_radius, radius)
    return poloidal_to_gridded_JS_operator(
        solid_harmonics, transform, poloidal_scale=poloidal_scale
    )


def Br_to_gridded_JS(solid_harmonics, transform, *, radius, boundary_radius):
    """Map boundary-Br coefficients to gridded JS."""
    return Br_to_gridded_JS_operator(
        solid_harmonics, transform, radius=radius, boundary_radius=boundary_radius
    ).array


def Br_to_gridded_JS_operator(solid_harmonics, transform, *, radius, boundary_radius):
    """Return the map from boundary-Br coefficients to gridded JS."""
    poloidal_scale = boundary_Br_to_poloidal_scale(solid_harmonics, boundary_radius, radius)
    return poloidal_to_gridded_JS_operator(
        solid_harmonics, transform, poloidal_scale=poloidal_scale
    )


def m_imp_to_gridded_JS(
    solid_harmonics, horizontal_transform, *, poloidal_transform=None, pfac_coupling_matrix=None
):
    """Map imposed-potential coefficients to their total gridded JS."""
    return m_imp_to_gridded_JS_operator(
        solid_harmonics,
        horizontal_transform,
        poloidal_transform=poloidal_transform,
        pfac_coupling_matrix=pfac_coupling_matrix,
    ).array


def m_imp_to_gridded_JS_operator(
    solid_harmonics, horizontal_transform, *, poloidal_transform=None, pfac_coupling_matrix=None
):
    """Return the map from imposed-potential coefficients to total JS.

    The direct term is the sheet current represented by ``m_imp`` on
    the ionosphere. With PFAC coupling, field-aligned current above the
    ionosphere also creates a poloidal magnetic contribution and an
    additional sheet current through the magnetic-potential jump.
    """
    poloidal_transform = horizontal_transform if poloidal_transform is None else poloidal_transform
    direct_sheet_current = (
        -1.0 / mu0
    ) * horizontal_transform.scalar_coeffs_to_gridded_gradient_operator
    if pfac_coupling_matrix is None:
        return direct_sheet_current
    m_imp_to_poloidal = as_linear_map(
        pfac_coupling_matrix,
        input_shape=(horizontal_transform.basis.index_length,),
        output_shape=(solid_harmonics.basis.index_length,),
    )
    pfac_sheet_current = (
        poloidal_to_gridded_JS_operator(solid_harmonics, poloidal_transform) @ m_imp_to_poloidal
    )
    return direct_sheet_current + pfac_sheet_current


__all__ = [
    "Br_to_gridded_JS",
    "Br_to_gridded_JS_operator",
    "boundary_Br_to_poloidal_scale",
    "m_imp_to_gridded_JS",
    "m_imp_to_gridded_JS_operator",
    "m_ind_to_gridded_JS",
    "m_ind_to_gridded_JS_operator",
    "poloidal_to_gridded_JS",
    "poloidal_to_gridded_JS_operator",
    "shielded_m_ind_poloidal_scale",
]
