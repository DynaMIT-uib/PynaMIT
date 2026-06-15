"""Sheet-current operator construction."""

from __future__ import annotations

import numpy as np

from pynamit.math.constants import mu0
from pynamit.math.backend import get_array_module


def coefficient_scale_values(values):
    """Return a one-dimensional coefficient-space scale."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"coefficient scale must be one-dimensional; got shape {array.shape}."
        )
    return array.copy()


def solid_poloidal_to_gridded_sheet_current(
    solid_harmonics,
    transform,
    *,
    solid_scale=None,
):
    """Map solid-harmonic poloidal coefficients to sheet current."""
    scale = coefficient_scale_values(
        solid_harmonics.poloidal_to_boundary_potential_jump_factor
    )
    if solid_scale is not None:
        scale = scale * np.asarray(solid_scale)
    return (
        -transform.scalar_coeffs_to_gridded_rhat_cross_gradient
        * scale.reshape(1, 1, -1)
        / mu0
    )


def horizontal_poloidal_to_gridded_sheet_current(
    solid_harmonics,
    transform,
    *,
    horizontal_to_solid_harmonic=None,
    solid_scale=None,
):
    """Map horizontal poloidal coefficients to gridded sheet current."""
    sheet_current = solid_poloidal_to_gridded_sheet_current(
        solid_harmonics,
        transform,
        solid_scale=solid_scale,
    )
    if horizontal_to_solid_harmonic is None:
        return sheet_current.copy()
    xp = get_array_module(sheet_current, horizontal_to_solid_harmonic)
    return xp.tensordot(
        sheet_current,
        xp.asarray(horizontal_to_solid_harmonic),
        axes=([2], [0]),
    )


def reference_boundary_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Return the m_ind-to-poloidal scale for a reference boundary."""
    regular_shift = coefficient_scale_values(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = coefficient_scale_values(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    denominator = 1.0 - regular_shift * irregular_shift
    return 1.0 + regular_shift * irregular_shift / denominator


def reference_boundary_br_to_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Return the boundary-Br-to-poloidal reference scale."""
    regular_shift = coefficient_scale_values(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = coefficient_scale_values(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    denominator = 1.0 - regular_shift * irregular_shift
    solid_harmonic_m_ind_to_Br = coefficient_scale_values(
        -(radius**2) * solid_harmonics.basis.laplacian(radius)
    )
    return -regular_shift / denominator / solid_harmonic_m_ind_to_Br


def m_ind_to_gridded_sheet_current(
    solid_harmonics,
    transform,
    *,
    radius,
    boundary_radius=None,
    horizontal_to_solid_harmonic=None,
):
    """Map induced-potential coefficients to gridded sheet current."""
    solid_scale = None
    if boundary_radius is not None:
        solid_scale = reference_boundary_poloidal_scale(
            solid_harmonics,
            boundary_radius,
            radius,
        )
        # Hack, remove assumption of shielding magnetosphere boundary.
        solid_scale = 1.0
    return horizontal_poloidal_to_gridded_sheet_current(
        solid_harmonics,
        transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        solid_scale=solid_scale,
    )


def Br_to_gridded_sheet_current(
    solid_harmonics,
    transform,
    *,
    radius,
    boundary_radius,
    horizontal_to_solid_harmonic=None,
):
    """Map boundary-Br coefficients to gridded sheet current."""
    solid_scale = reference_boundary_br_to_poloidal_scale(
        solid_harmonics,
        boundary_radius,
        radius,
    )
    return horizontal_poloidal_to_gridded_sheet_current(
        solid_harmonics,
        transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        solid_scale=solid_scale,
    )


def m_imp_to_gridded_sheet_current(
    solid_harmonics,
    horizontal_transform,
    *,
    solid_transform=None,
    horizontal_to_solid_harmonic=None,
    T_to_Ve=None,
):
    """Map imposed-potential coefficients to gridded sheet current."""
    solid_transform = horizontal_transform if solid_transform is None else solid_transform
    toroidal = -horizontal_transform.scalar_coeffs_to_gridded_gradient / mu0
    if T_to_Ve is None:
        return toroidal
    poloidal = horizontal_poloidal_to_gridded_sheet_current(
        solid_harmonics,
        solid_transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
    )
    xp = get_array_module(toroidal, poloidal, T_to_Ve)
    return toroidal + xp.tensordot(
        poloidal,
        xp.asarray(T_to_Ve),
        axes=([2], [0]),
    )


def sheet_current_operator_bundle(
    solid_harmonics,
    horizontal_transform,
    *,
    radius,
    boundary_radius=None,
    solid_transform=None,
    horizontal_to_solid_harmonic=None,
    T_to_Ve=None,
):
    """Return m_ind, m_imp, and boundary-Br sheet-current operators."""
    solid_transform = horizontal_transform if solid_transform is None else solid_transform
    m_ind_to_sheet = m_ind_to_gridded_sheet_current(
        solid_harmonics,
        solid_transform,
        radius=radius,
        boundary_radius=boundary_radius,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
    )
    m_imp_to_sheet = m_imp_to_gridded_sheet_current(
        solid_harmonics,
        horizontal_transform,
        solid_transform=solid_transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        T_to_Ve=T_to_Ve,
    )
    if boundary_radius is None:
        xp = get_array_module(m_ind_to_sheet)
        Br_to_sheet = xp.zeros_like(m_ind_to_sheet)
    else:
        Br_to_sheet = Br_to_gridded_sheet_current(
            solid_harmonics,
            solid_transform,
            radius=radius,
            boundary_radius=boundary_radius,
            horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        )
    return {
        "G_m_ind_to_JS": m_ind_to_sheet,
        "G_m_imp_to_JS": m_imp_to_sheet,
        "G_Br_to_JS": Br_to_sheet,
    }


__all__ = [
    "Br_to_gridded_sheet_current",
    "coefficient_scale_values",
    "horizontal_poloidal_to_gridded_sheet_current",
    "m_imp_to_gridded_sheet_current",
    "m_ind_to_gridded_sheet_current",
    "reference_boundary_br_to_poloidal_scale",
    "reference_boundary_poloidal_scale",
    "sheet_current_operator_bundle",
    "solid_poloidal_to_gridded_sheet_current",
]
