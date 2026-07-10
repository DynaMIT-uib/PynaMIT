"""Derived sheet-current operators from magnetic-boundary fields.

In PynaMIT's B, v formulation, JS is a derived horizontal surface
current rather than an independent state.  These functions map magnetic
potential coefficients and optional boundary field data to JS on a grid.
Geometry supplies the run-specific radii, transforms, and shielding.
"""

from __future__ import annotations

import numpy as np

from pynamit.math.constants import mu0
from pynamit.math.backend import get_array_module


def solid_harmonic_scale_values(values):
    """Return a one-dimensional coefficient-space scale."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"coefficient scale must be one-dimensional; got shape {array.shape}.")
    return array.copy()


def poloidal_to_gridded_JS(
    solid_harmonics, transform, *, horizontal_to_solid_harmonic=None, solid_scale=None
):
    """Map poloidal coefficients to gridded JS."""
    scale = solid_harmonic_scale_values(
        solid_harmonics.poloidal_to_boundary_potential_jump_factor
    )
    if solid_scale is not None:
        scale = scale * np.asarray(solid_scale)
    JS = (
        -transform.scalar_coeffs_to_gridded_rhat_cross_gradient
        * scale.reshape(1, 1, -1)
        / mu0
    )
    if horizontal_to_solid_harmonic is None:
        return JS.copy()
    xp = get_array_module(JS, horizontal_to_solid_harmonic)
    return xp.tensordot(JS, xp.asarray(horizontal_to_solid_harmonic), axes=([2], [0]))


def shielded_m_ind_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Return the outer-boundary shielding response for m_ind."""
    regular_shift = solid_harmonic_scale_values(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = solid_harmonic_scale_values(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    denominator = 1.0 - regular_shift * irregular_shift
    return 1.0 + regular_shift * irregular_shift / denominator


def boundary_Br_to_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Return the prescribed boundary-Br-to-poloidal scale."""
    regular_shift = solid_harmonic_scale_values(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = solid_harmonic_scale_values(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    denominator = 1.0 - regular_shift * irregular_shift
    solid_harmonic_m_ind_to_Br = solid_harmonic_scale_values(
        -(radius**2) * solid_harmonics.basis.laplacian(radius)
    )
    return -regular_shift / denominator / solid_harmonic_m_ind_to_Br


def m_ind_to_gridded_JS(
    solid_harmonics,
    transform,
    *,
    radius,
    boundary_radius=None,
    boundary_shielding=False,
    horizontal_to_solid_harmonic=None,
):
    """Map induced-potential coefficients to gridded JS."""
    solid_scale = None
    if boundary_radius is not None:
        if boundary_shielding:
            solid_scale = shielded_m_ind_poloidal_scale(
                solid_harmonics, boundary_radius, radius
            )
        else:
            solid_scale = 1.0
    return poloidal_to_gridded_JS(
        solid_harmonics,
        transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        solid_scale=solid_scale,
    )


def Br_to_gridded_JS(
    solid_harmonics,
    transform,
    *,
    radius,
    boundary_radius,
    horizontal_to_solid_harmonic=None,
):
    """Map boundary-Br coefficients to gridded JS."""
    solid_scale = boundary_Br_to_poloidal_scale(
        solid_harmonics, boundary_radius, radius
    )
    return poloidal_to_gridded_JS(
        solid_harmonics,
        transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        solid_scale=solid_scale,
    )


def m_imp_to_gridded_JS(
    solid_harmonics,
    horizontal_transform,
    *,
    solid_transform=None,
    horizontal_to_solid_harmonic=None,
    T_to_Ve=None,
):
    """Map imposed-potential coefficients to gridded JS."""
    solid_transform = horizontal_transform if solid_transform is None else solid_transform
    toroidal = -horizontal_transform.scalar_coeffs_to_gridded_gradient / mu0
    if T_to_Ve is None:
        return toroidal
    poloidal = poloidal_to_gridded_JS(
        solid_harmonics,
        solid_transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
    )
    xp = get_array_module(toroidal, poloidal, T_to_Ve)
    return toroidal + xp.tensordot(poloidal, xp.asarray(T_to_Ve), axes=([2], [0]))


def JS_operator_bundle(
    solid_harmonics,
    horizontal_transform,
    *,
    radius,
    boundary_radius=None,
    boundary_shielding=False,
    solid_transform=None,
    horizontal_to_solid_harmonic=None,
    T_to_Ve=None,
):
    """Return m_ind, m_imp, and boundary-Br JS operators."""
    solid_transform = horizontal_transform if solid_transform is None else solid_transform
    m_ind_to_JS = m_ind_to_gridded_JS(
        solid_harmonics,
        solid_transform,
        radius=radius,
        boundary_radius=boundary_radius,
        boundary_shielding=boundary_shielding,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
    )
    m_imp_to_JS = m_imp_to_gridded_JS(
        solid_harmonics,
        horizontal_transform,
        solid_transform=solid_transform,
        horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        T_to_Ve=T_to_Ve,
    )
    if boundary_radius is None:
        xp = get_array_module(m_ind_to_JS)
        Br_to_JS = xp.zeros_like(m_ind_to_JS)
    else:
        Br_to_JS = Br_to_gridded_JS(
            solid_harmonics,
            solid_transform,
            radius=radius,
            boundary_radius=boundary_radius,
            horizontal_to_solid_harmonic=horizontal_to_solid_harmonic,
        )
    return {
        "G_m_ind_to_JS": m_ind_to_JS,
        "G_m_imp_to_JS": m_imp_to_JS,
        "G_Br_to_JS": Br_to_JS,
    }


__all__ = [
    "Br_to_gridded_JS",
    "JS_operator_bundle",
    "boundary_Br_to_poloidal_scale",
    "m_imp_to_gridded_JS",
    "m_ind_to_gridded_JS",
    "poloidal_to_gridded_JS",
    "shielded_m_ind_poloidal_scale",
    "solid_harmonic_scale_values",
]
