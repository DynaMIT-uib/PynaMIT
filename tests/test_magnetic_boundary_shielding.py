"""Magnetic outer-boundary shielding test."""

import numpy as np
import pytest

from pynamit.math.constants import RE
from pynamit.simulation.electrodynamics.magnetic_boundary import (
    boundary_Br_to_poloidal_scale,
    shielded_m_ind_poloidal_scale,
)
from pynamit.simulation.workflows.standard import run_pynamit
from pynamit.sphere import SHBasis, SolidHarmonics


def test_boundary_br_continuation_reproduces_prescribed_outer_field():
    """Regular source plus irregular response must equal Br at RM."""
    solid_harmonics = SolidHarmonics(SHBasis(Nmax=4, Mmax=3))
    inner_radius = 6.5e6
    boundary_radius = 10.0e6
    regular_to_inner = np.asarray(
        solid_harmonics.regular_reference_shift(boundary_radius, inner_radius)
    )
    irregular_to_boundary = np.asarray(
        solid_harmonics.irregular_reference_shift(inner_radius, boundary_radius)
    )
    denominator = 1.0 - regular_to_inner * irregular_to_boundary
    inner_br_per_poloidal = np.asarray(
        -(inner_radius**2) * solid_harmonics.basis.laplacian(inner_radius)
    )
    boundary_to_inner_poloidal = boundary_Br_to_poloidal_scale(
        solid_harmonics, boundary_radius, inner_radius
    )

    regular_source_at_boundary = 1.0 / denominator
    irregular_response_at_boundary = (
        inner_br_per_poloidal * irregular_to_boundary * boundary_to_inner_poloidal
    )

    np.testing.assert_allclose(
        regular_source_at_boundary + irregular_response_at_boundary, 1.0, atol=1e-14
    )


def test_optional_m_ind_shielding_cancels_field_at_outer_boundary():
    """The optional image response must cancel m_ind Br at RM."""
    solid_harmonics = SolidHarmonics(SHBasis(Nmax=4, Mmax=3))
    inner_radius = 6.5e6
    boundary_radius = 10.0e6
    regular_to_inner = np.asarray(
        solid_harmonics.regular_reference_shift(boundary_radius, inner_radius)
    )
    irregular_to_boundary = np.asarray(
        solid_harmonics.irregular_reference_shift(inner_radius, boundary_radius)
    )
    shielded_scale = shielded_m_ind_poloidal_scale(
        solid_harmonics, boundary_radius, inner_radius
    )

    unshielded_part = irregular_to_boundary * shielded_scale
    image_at_inner = -regular_to_inner * irregular_to_boundary * shielded_scale
    image_part = image_at_inner * np.asarray(
        solid_harmonics.regular_reference_shift(inner_radius, boundary_radius)
    )
    np.testing.assert_allclose(unshielded_part + image_part, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        shielded_scale,
        1.0 / (1.0 - regular_to_inner * irregular_to_boundary),
    )


def test_magnetic_boundary_shielding():
    """Test 2D simulation with magnetosphere boundary currents."""
    # Arrange.
    expected_coeff_norm = 9.215777844127273e-09
    expected_coeff_max = 1.5650387484970313e-09
    expected_coeff_min = -3.860308906912494e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        RM=4 * RE,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        steady_state_initialization=False,
        magnetic_boundary_shielding=True,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            simulation.run_data.output_series.datasets["state"]["SH_m_ind"].values[-1],
            simulation.run_data.output_series.datasets["state"]["SH_m_imp"].values[-1],
        )
    )

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
