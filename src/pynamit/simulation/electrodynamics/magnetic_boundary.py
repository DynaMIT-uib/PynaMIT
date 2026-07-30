"""Sheet-current response to physical magnetic boundary quantities.

The ionosphere sees two external-source radial fields: the field
continued inward from the magnetospheric boundary and the poloidal
field created by field-aligned current in the gap. Both are shielded
by the same divergence-free ionospheric sheet-current response.

The evolving induced field is continuous across the ionosphere. Its
stored coordinate is radial magnetic field at the ionosphere; private
poloidal and toroidal potentials are used only to build well-conditioned
operators.
"""

from __future__ import annotations

import numpy as np

from pynamit.math import diagonal_linear_map
from pynamit.math.constants import mu0


def _coefficient_scale(values):
    """Return a one-dimensional coefficient-space scale."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"coefficient scale must be one-dimensional; got shape {array.shape}.")
    return array.astype(np.result_type(array.dtype, np.float64), copy=True)


def _poloidal_degree_factor(solid_harmonics):
    """Return ``n(n+1)`` in the poloidal coefficient ordering."""
    n = np.asarray(solid_harmonics.basis.n)
    return n * (n + 1)


def poloidal_potential_to_gridded_JS_operator(solid_harmonics, transform, *, poloidal_scale=None):
    """Map private poloidal-potential coefficients to sheet current."""
    scale = _coefficient_scale(solid_harmonics.poloidal_to_boundary_potential_jump_factor)
    if poloidal_scale is not None:
        scale *= np.asarray(poloidal_scale)
    return (
        (-1.0 / mu0)
        * transform.scalar_coeffs_to_gridded_rhat_cross_gradient_operator
        @ diagonal_linear_map(scale)
    )


def external_Br_to_gridded_JS_operator(solid_harmonics, transform):
    """Shield external-source ``Br(RI)`` with ionospheric current."""
    external_Br_to_shielding_potential = diagonal_linear_map(
        -1.0 / _poloidal_degree_factor(solid_harmonics)
    )
    return (
        poloidal_potential_to_gridded_JS_operator(solid_harmonics, transform)
        @ external_Br_to_shielding_potential
    )


def shielded_induced_poloidal_scale(solid_harmonics, boundary_radius, radius):
    """Return the optional zero-Br outer image-response scale."""
    regular_shift = _coefficient_scale(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = _coefficient_scale(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    return 1.0 / (1.0 - regular_shift * irregular_shift)


def induced_Br_to_gridded_JS_operator(
    solid_harmonics, transform, *, radius, boundary_radius=None, boundary_shielding=False
):
    """Map continuous induced ``Br(RI)`` to ionospheric current."""
    poloidal_scale = None
    if boundary_radius is not None and boundary_shielding:
        poloidal_scale = shielded_induced_poloidal_scale(solid_harmonics, boundary_radius, radius)
    induced_Br_to_poloidal_potential = diagonal_linear_map(
        1.0 / _poloidal_degree_factor(solid_harmonics)
    )
    return (
        poloidal_potential_to_gridded_JS_operator(
            solid_harmonics, transform, poloidal_scale=poloidal_scale
        )
        @ induced_Br_to_poloidal_potential
    )


def boundary_Br_to_ionosphere_external_Br_scale(solid_harmonics, boundary_radius, radius):
    """Continue prescribed outer-boundary Br to external ``Br(RI)``.

    The source is represented by a regular field in the gap together
    with the irregular ionospheric shielding response required to
    reproduce the prescribed field at ``boundary_radius``.
    """
    regular_shift = _coefficient_scale(
        solid_harmonics.regular_reference_shift(boundary_radius, radius)
    )
    irregular_shift = _coefficient_scale(
        solid_harmonics.irregular_reference_shift(radius, boundary_radius)
    )
    return regular_shift / (1.0 - regular_shift * irregular_shift)


def boundary_Br_to_gridded_JS_operator(solid_harmonics, transform, *, radius, boundary_radius):
    """Map outer-boundary Br to ionospheric shielding current."""
    continued_Br = diagonal_linear_map(
        boundary_Br_to_ionosphere_external_Br_scale(solid_harmonics, boundary_radius, radius)
    )
    return external_Br_to_gridded_JS_operator(solid_harmonics, transform) @ continued_Br


def toroidal_potential_to_gridded_JS_operator(
    solid_harmonics,
    horizontal_transform,
    *,
    poloidal_transform,
    toroidal_potential_to_boundary_jr,
    boundary_jr_to_gap_Br=None,
):
    """Map private toroidal potential to total sheet current."""
    direct_sheet_current = (
        -1.0 / mu0
    ) * horizontal_transform.scalar_coeffs_to_gridded_gradient_operator
    if boundary_jr_to_gap_Br is None:
        return direct_sheet_current
    gap_shielding_current = (
        external_Br_to_gridded_JS_operator(solid_harmonics, poloidal_transform)
        @ boundary_jr_to_gap_Br
        @ toroidal_potential_to_boundary_jr
    )
    return direct_sheet_current + gap_shielding_current


def boundary_jr_to_gridded_JS_operator(
    solid_harmonics,
    horizontal_transform,
    *,
    poloidal_transform,
    boundary_jr_to_toroidal_potential,
    boundary_jr_to_gap_Br=None,
):
    """Map upper-boundary radial current to total sheet current.

    The curl-free term closes ``boundary_jr`` through current
    continuity. The divergence-free term shields the poloidal radial
    field created by the continuation of that current through the gap.
    """
    direct_sheet_current = (
        (-1.0 / mu0)
        * horizontal_transform.scalar_coeffs_to_gridded_gradient_operator
        @ boundary_jr_to_toroidal_potential
    )
    if boundary_jr_to_gap_Br is None:
        return direct_sheet_current
    gap_shielding_current = (
        external_Br_to_gridded_JS_operator(solid_harmonics, poloidal_transform)
        @ boundary_jr_to_gap_Br
    )
    return direct_sheet_current + gap_shielding_current


__all__ = [
    "boundary_Br_to_gridded_JS_operator",
    "boundary_Br_to_ionosphere_external_Br_scale",
    "boundary_jr_to_gridded_JS_operator",
    "external_Br_to_gridded_JS_operator",
    "induced_Br_to_gridded_JS_operator",
    "poloidal_potential_to_gridded_JS_operator",
    "shielded_induced_poloidal_scale",
    "toroidal_potential_to_gridded_JS_operator",
]
