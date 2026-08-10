"""Magnetic outer-boundary shielding test."""

import numpy as np
import pytest
from kompe import SHBasis, SolidHarmonicOperators
from kompe.constants import EARTH_RADIUS_M

from pynamit.simulation.electrodynamics.magnetic_boundary import (
    boundary_Br_to_ionosphere_external_Br_scale,
    shielded_induced_poloidal_scale,
)
from pynamit.simulation.workflows.standard import run_pynamit
from tests import magnetic_potential_coordinate_array


def test_boundary_br_continuation_reproduces_prescribed_outer_field():
    """Regular source plus irregular response must equal Br at RM."""
    solid_harmonics = SolidHarmonicOperators(SHBasis(max_degree=4, max_order=3))
    inner_radius = 6.5e6
    boundary_radius = 10.0e6
    regular_to_inner = np.asarray(
        solid_harmonics.regular_reference_shift(boundary_radius, inner_radius)
    )
    irregular_to_boundary = np.asarray(
        solid_harmonics.irregular_reference_shift(inner_radius, boundary_radius)
    )
    denominator = 1.0 - regular_to_inner * irregular_to_boundary
    inner_br_per_poloidal = -(inner_radius**2) * np.asarray(
        solid_harmonics.basis.surface_laplacian_operator(inner_radius).matvec(
            np.ones(solid_harmonics.basis.index_length)
        )
    )
    continued_Br = boundary_Br_to_ionosphere_external_Br_scale(
        solid_harmonics, boundary_radius, inner_radius
    )
    degree_factor = solid_harmonics.basis.n * (solid_harmonics.basis.n + 1)
    boundary_to_inner_poloidal = -continued_Br / degree_factor

    regular_source_at_boundary = 1.0 / denominator
    irregular_response_at_boundary = (
        inner_br_per_poloidal * irregular_to_boundary * boundary_to_inner_poloidal
    )

    np.testing.assert_allclose(
        regular_source_at_boundary + irregular_response_at_boundary, 1.0, atol=1e-14
    )


def test_optional_induced_Br_shielding_cancels_field_at_outer_boundary():
    """The optional image response must cancel induced Br at RM."""
    solid_harmonics = SolidHarmonicOperators(SHBasis(max_degree=4, max_order=3))
    inner_radius = 6.5e6
    boundary_radius = 10.0e6
    regular_to_inner = np.asarray(
        solid_harmonics.regular_reference_shift(boundary_radius, inner_radius)
    )
    irregular_to_boundary = np.asarray(
        solid_harmonics.irregular_reference_shift(inner_radius, boundary_radius)
    )
    shielded_scale = shielded_induced_poloidal_scale(
        solid_harmonics, boundary_radius, inner_radius
    )

    unshielded_part = irregular_to_boundary * shielded_scale
    image_at_inner = -regular_to_inner * irregular_to_boundary * shielded_scale
    image_part = image_at_inner * np.asarray(
        solid_harmonics.regular_reference_shift(inner_radius, boundary_radius)
    )
    np.testing.assert_allclose(unshielded_part + image_part, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        shielded_scale, 1.0 / (1.0 - regular_to_inner * irregular_to_boundary)
    )


def test_magnetic_boundary_shielding():
    """Test 2D simulation with magnetosphere boundary currents."""
    # Arrange.
    expected_coeff_norm = 8.926152422900569e-09
    expected_coeff_max = 1.7074729543055983e-09
    expected_coeff_min = -3.7995225478677866e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        RM=4 * EARTH_RADIUS_M,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        equilibrium_initialization=False,
        magnetic_boundary_shielding=True,
    )

    # Assert.
    coeff_array = magnetic_potential_coordinate_array(simulation)

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)
