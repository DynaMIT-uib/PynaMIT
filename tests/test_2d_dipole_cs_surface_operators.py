"""Dipole test using CS surface operators."""

import numpy as np
import pytest

from pynamit.sphere import CSBasis, Grid, SHBasis, SphericalTransform
from pynamit.default_run import run_pynamit
from pynamit.visualization.figure_specs import PynamitFigureSpec
from pynamit.visualization.ground_figures import GroundFigureRenderer
from pynamit.visualization.run_fields import SavedCoefficientFieldView


def test_2d_dipole_cs_surface_operators(tmp_path):
    """Run a 2D dipole case with CS state and saved-run views."""
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
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
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
        dynamics.state.geometry.spherical_transform_zero_added.source is dynamics.horizontal_basis
    )
    assert dynamics.output_field_spaces["state"].representation is dynamics.horizontal_basis

    geometry = dynamics.state.geometry
    spherical_transform = geometry.spherical_transform
    expected_helmholtz = dynamics.horizontal_basis.get_helmholtz_synthesis_matrix(geometry.grid)
    np.testing.assert_allclose(
        spherical_transform.helmholtz_coeffs_to_gridded_vector, expected_helmholtz
    )
    np.testing.assert_allclose(
        geometry.surface_laplacian_operator.to_matrix(backend="numpy"),
        dynamics.horizontal_basis.get_surface_laplacian_matrix(geometry.RI),
    )
    assert "poloidal_to_boundary_potential_jump_factor" not in geometry.__dict__
    assert "horizontal_to_boundary_potential_jump_factor" not in geometry.__dict__
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
    assert geometry.m_ind_to_gridded_JS().shape == (
        2,
        geometry.grid.size,
        dynamics.horizontal_basis.index_length,
    )
    assert geometry.m_imp_to_gridded_JS().shape == (
        2,
        geometry.grid.size,
        dynamics.horizontal_basis.index_length,
    )

    plot_grid = Grid(theta=geometry.grid.theta[:10], phi=geometry.grid.phi[:10])
    plot_transform = SphericalTransform(dynamics.horizontal_basis, plot_grid)
    assert geometry.solid_transform_for(plot_transform) is geometry.solid_transform_for(
        plot_transform
    )
    assert geometry.m_ind_to_gridded_JS(plot_transform).shape == (
        2,
        plot_grid.size,
        dynamics.horizontal_basis.index_length,
    )

    state = dynamics.output_timeseries.datasets["state"]
    assert "CS_m_ind" in state
    assert "CS_m_imp" in state

    coeff_array = np.hstack((state["CS_m_ind"].values[-1], state["CS_m_imp"].values[-1]))

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
    assert dynamics.horizontal_basis.scalar_mean(dynamics.state.jr.array) == (
        pytest.approx(0.0, abs=1e-18)
    )

    view = SavedCoefficientFieldView.from_directory(
        dynamics.run_directory, nlat=8, nlon=12, wind_nlat=5, wind_nlon=7
    )
    fields = view.state_comparison_grid_fields(0)
    assert isinstance(view.evaluator.source, CSBasis)
    assert fields["Br_state"].shape == view.lat.shape
    assert fields["jr_state"].shape == view.lat.shape
    assert np.all(np.isfinite(fields["Br_state"]))
    assert np.all(np.isfinite(fields["jr_state"]))

    renderer = GroundFigureRenderer(
        PynamitFigureSpec(run_directory=dynamics.run_directory, include_station_data=False),
        view=view,
    )
    br_ind, bh_ind, _, _ = renderer.ground_field_matrices([65.0], [0.0])
    assert br_ind.shape == (1, view.n_time)
    assert bh_ind.shape == (2, 1, view.n_time)
    assert np.all(np.isfinite(br_ind))
    assert np.all(np.isfinite(bh_ind))
