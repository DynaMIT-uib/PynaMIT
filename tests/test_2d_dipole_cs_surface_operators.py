"""Dipole test using CS surface operators."""

import numpy as np
import pytest

from pynamit.sphere import CSBasis, SHBasis
from pynamit.default_run import run_pynamit


def test_2d_dipole_cs_surface_operators(tmp_path):
    """Run a 2D dipole case with CS state and Helmholtz operators."""
    expected_coeff_norm = 3.21758062211637e-07
    expected_coeff_max = 1.0044485660736859e-07
    expected_coeff_min = -8.091092606324309e-08
    expected_n_coeffs = 768

    dynamics = run_pynamit(
        final_time=0.01,
        dt=0.01,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=True,
        steady_state_initialization=False,
        use_wind=False,
        run_directory=str(tmp_path / "run"),
        vector_jr=False,
        vector_conductance=False,
        vector_u=False,
        least_squares_solver="normal_pinv",
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    assert isinstance(dynamics.horizontal_basis, CSBasis)
    assert isinstance(dynamics.solid_harmonics.basis.root_basis, SHBasis)
    assert dynamics.horizontal_basis is dynamics.state.basis
    assert dynamics.solid_harmonics.basis is not dynamics.horizontal_basis
    assert dynamics.horizontal_spherical_transform is dynamics.state.geometry.spherical_transform
    assert (
        dynamics.state.geometry.spherical_transform_zero_added.source
        is dynamics.horizontal_basis
    )
    assert dynamics.output_field_spaces["state"].representation is dynamics.horizontal_basis

    geometry = dynamics.state.geometry
    spherical_transform = geometry.spherical_transform
    expected_helmholtz = dynamics.horizontal_basis.get_helmholtz_synthesis_matrix(
        geometry.grid
    )
    np.testing.assert_allclose(
        spherical_transform.helmholtz_coeffs_to_gridded_vector,
        expected_helmholtz,
    )
    np.testing.assert_allclose(
        geometry.surface_laplacian_operator.to_matrix(backend="numpy"),
        dynamics.horizontal_basis.get_surface_laplacian_matrix(
            geometry.RI
        ),
    )
    assert geometry._poloidal_to_boundary_potential_jump_factor is None
    assert geometry._horizontal_to_boundary_potential_jump_factor is None
    expected_boundary_potential_jump_factor = (
        np.diag(dynamics.solid_harmonics.poloidal_to_boundary_potential_jump_factor)
        @ geometry.horizontal_to_solid_harmonic
    )
    np.testing.assert_allclose(
        geometry.horizontal_to_boundary_potential_jump_factor_operator.to_matrix(backend="numpy"),
        expected_boundary_potential_jump_factor,
    )
    np.testing.assert_allclose(
        geometry.horizontal_to_boundary_potential_jump_factor,
        expected_boundary_potential_jump_factor,
    )
    np.testing.assert_allclose(
        geometry.poloidal_to_boundary_potential_jump_factor,
        np.diag(dynamics.solid_harmonics.poloidal_to_boundary_potential_jump_factor),
    )

    state = dynamics.output_timeseries.datasets["state"]
    assert "CS_m_ind" in state
    assert "CS_m_imp" in state

    coeff_array = np.hstack(
        (
            state["CS_m_ind"].values[-1],
            state["CS_m_imp"].values[-1],
        )
    )

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    assert actual_n_coeffs == 2 * dynamics.horizontal_basis.index_length
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == expected_n_coeffs
    assert np.all(np.isfinite(coeff_array))

    for name in ("m_ind", "m_imp", "Phi", "W"):
        assert dynamics.horizontal_basis.scalar_mean(state[f"CS_{name}"].values[-1]) == (
            pytest.approx(0.0, abs=1e-18)
        )
    assert dynamics.horizontal_basis.scalar_mean(dynamics.state.jr.coeffs) == (
        pytest.approx(0.0, abs=1e-18)
    )
