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

import logging

import numpy as np
from kompe import SphericalGrid, SphericalTransform
from kompe.constants import MU0
from kompe.math import diagonal_linear_map, get_array_module, pointwise_matrix_linear_map

logger = logging.getLogger(__name__)


def _coefficient_scale(values):
    """Return a one-dimensional coefficient-space scale."""
    xp = get_array_module(values)
    array = xp.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"coefficient scale must be one-dimensional; got shape {array.shape}.")
    return xp.asarray(array, dtype=np.result_type(array.dtype, np.float64))


def _poloidal_degree_factor(solid_harmonics):
    """Return ``n(n+1)`` in the poloidal coefficient ordering."""
    n_values = solid_harmonics.basis.n
    xp = get_array_module(n_values)
    n = xp.asarray(n_values)
    return n * (n + 1)


def poloidal_potential_to_gridded_JS_operator(solid_harmonics, transform, *, poloidal_scale=None):
    """Map private poloidal-potential coefficients to sheet current."""
    scale = _coefficient_scale(solid_harmonics.poloidal_to_normalized_potential_jump_factors)
    if poloidal_scale is not None:
        xp = get_array_module(scale, poloidal_scale)
        scale = xp.asarray(scale) * xp.asarray(poloidal_scale)
    return (-1.0 / MU0) * transform.rhat_cross_gradient_operator @ diagonal_linear_map(scale)


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
        solid_harmonics.regular_reference_shift_factors(boundary_radius, radius)
    )
    irregular_shift = _coefficient_scale(
        solid_harmonics.irregular_reference_shift_factors(radius, boundary_radius)
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
        solid_harmonics.regular_reference_shift_factors(boundary_radius, radius)
    )
    irregular_shift = _coefficient_scale(
        solid_harmonics.irregular_reference_shift_factors(radius, boundary_radius)
    )
    return regular_shift / (1.0 - regular_shift * irregular_shift)


def boundary_Br_to_gridded_JS_operator(solid_harmonics, transform, *, radius, boundary_radius):
    """Map outer-boundary Br to ionospheric shielding current."""
    continued_Br = diagonal_linear_map(
        boundary_Br_to_ionosphere_external_Br_scale(solid_harmonics, boundary_radius, radius)
    )
    return external_Br_to_gridded_JS_operator(solid_harmonics, transform) @ continued_Br


def boundary_jr_to_gap_Br_matrix(
    main_field,
    horizontal_basis,
    poloidal_transform,
    solid_harmonics,
    *,
    ionosphere_radius,
    integration_radii,
    boundary_radius=None,
):
    """Integrate the gap field produced by ionospheric radial current.

    Map horizontal ``jr(RI)`` coefficients to poloidal ``Br(RI)``
    coefficients using midpoint shell quadrature (radii in metres).
    Field-aligned continuation gives each shell's horizontal current;
    its shielding potential determines the incident gap field.

    This one-time construction returns a NumPy matrix for reuse and
    storage. Intermediate shell maps are materialized on the CPU and
    are excluded from the persistent basis-evaluation cache.
    """
    model_grid = poloidal_transform.grid
    poloidal_basis = solid_harmonics.basis
    shielding_potential_response = np.zeros(
        (poloidal_basis.index_length, horizontal_basis.index_length)
    )
    integration_radii = np.asarray(integration_radii)
    radial_step_widths = np.diff(integration_radii)
    radial_midpoints = integration_radii[:-1] + 0.5 * radial_step_widths
    gridded_JS_to_poloidal = poloidal_transform.rhat_cross_gradient_analysis_operator(
        coefficient_scale=(
            -np.asarray(solid_harmonics.poloidal_to_normalized_potential_jump_factors) / MU0
        )
    )
    if boundary_radius is None:
        boundary_response_factor = -1.0
    else:
        # Regular/irregular images enforce Br(boundary_radius) = 0.
        outer_regular_to_ionosphere = np.asarray(
            solid_harmonics.regular_reference_shift_factors(boundary_radius, ionosphere_radius)
        )
        boundary_response_factor = -np.asarray(
            shielded_induced_poloidal_scale(solid_harmonics, boundary_radius, ionosphere_radius)
        )

    for i, (radius, width) in enumerate(zip(radial_midpoints, radial_step_widths, strict=True)):
        logger.debug(
            "Gap-field integration step %d/%d (r=%s)", i + 1, radial_midpoints.size, radius
        )
        theta_footpoint, phi_footpoint = main_field.map_along_field_lines(
            r_dest=ionosphere_radius, r=radius, theta=model_grid.theta, phi=model_grid.phi
        )
        footpoint_grid = SphericalGrid(theta=theta_footpoint, phi=phi_footpoint)
        _, shell_Btheta, shell_Bphi = main_field.evaluate(model_grid, radius)
        footpoint_Br = main_field.evaluate(footpoint_grid, ionosphere_radius)[0]
        footpoint_transform = SphericalTransform(
            horizontal_basis, footpoint_grid, use_persistent_evaluation_cache=False
        )
        jr_to_gridded_JS = pointwise_matrix_linear_map(
            np.array([shell_Btheta / footpoint_Br, shell_Bphi / footpoint_Br]).reshape(
                2, 1, model_grid.size
            )
        )

        poloidal_scale = np.array(
            solid_harmonics.regular_reference_shift_factors(radius, ionosphere_radius), copy=True
        )
        if boundary_radius is not None:
            poloidal_scale -= outer_regular_to_ionosphere * np.asarray(
                solid_harmonics.irregular_reference_shift_factors(radius, boundary_radius)
            )
        poloidal_scale *= boundary_response_factor
        shell_response = (
            diagonal_linear_map(poloidal_scale)
            @ gridded_JS_to_poloidal
            @ jr_to_gridded_JS
            @ footpoint_transform.scalar_synthesis_operator
        )
        shielding_potential_response += width * np.asarray(
            shell_response.to_matrix(backend="numpy")
        )

    # Shielding gives kappa = -D^-1 b_gap, hence b_gap = -D kappa.
    degree_factor = np.asarray(poloidal_basis.n * (poloidal_basis.n + 1), dtype=float)
    return -degree_factor[:, None] * shielding_potential_response


def toroidal_potential_to_gridded_JS_operator(
    solid_harmonics,
    horizontal_transform,
    *,
    poloidal_transform,
    toroidal_potential_to_boundary_jr,
    boundary_jr_to_gap_Br=None,
):
    """Map private toroidal potential to total sheet current."""
    direct_sheet_current = (-1.0 / MU0) * horizontal_transform.surface_gradient_operator
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
        (-1.0 / MU0)
        * horizontal_transform.surface_gradient_operator
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
    "boundary_jr_to_gap_Br_matrix",
    "external_Br_to_gridded_JS_operator",
    "induced_Br_to_gridded_JS_operator",
    "poloidal_potential_to_gridded_JS_operator",
    "shielded_induced_poloidal_scale",
    "toroidal_potential_to_gridded_JS_operator",
]
