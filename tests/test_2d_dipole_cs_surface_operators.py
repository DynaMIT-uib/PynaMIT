"""Dipole test using CS surface operators."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit
from pynamit.sphere import CSBasis, Grid, SHBasis, SphericalTransform
from pynamit.visualization.figure_specs import PynamitFigureSpec
from pynamit.visualization.ground_figures import GroundFigureRenderer
from pynamit.visualization.run_fields import SavedCoefficientFieldView


def test_2d_dipole_cs_surface_operators(tmp_path):
    """Run a 2D dipole case with CS state and saved-run views."""
    simulation = run_pynamit(
        final_time=0.01,
        dt=0.01,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        steady_state_initialization=False,
        use_wind=False,
        run_directory=str(tmp_path / "run"),
        jr_projection_basis="CS",
        resistance_projection_basis="CS",
        u_projection_basis="CS",
        least_squares_solver="normal_pinv",
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    assert isinstance(simulation.geometry.horizontal_basis, CSBasis)
    assert isinstance(simulation.geometry.solid_harmonics.basis.root_basis, SHBasis)
    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert not simulation.run_data.schema.input_field_spaces["resistance"].mean_free
    state_spaces = simulation.run_data.schema.output_field_spaces["state"]
    assert state_spaces["m_ind"].representation is simulation.geometry.poloidal_basis
    assert state_spaces["m_imp"].representation is simulation.geometry.horizontal_basis

    geometry = simulation.geometry
    spherical_transform = geometry.horizontal_transform
    expected_helmholtz = simulation.geometry.horizontal_basis.get_helmholtz_synthesis_matrix(
        geometry.model_grid
    )
    np.testing.assert_allclose(
        spherical_transform.helmholtz_coeffs_to_gridded_vector, expected_helmholtz
    )
    np.testing.assert_allclose(
        geometry.surface_laplacian_operator.to_matrix(backend="numpy"),
        simulation.geometry.horizontal_basis.get_surface_laplacian_matrix(geometry.RI),
    )
    expected_boundary_potential_jump_factor = np.diag(
        simulation.geometry.solid_harmonics.poloidal_to_boundary_potential_jump_factor
    )
    np.testing.assert_allclose(
        geometry.poloidal_to_boundary_potential_jump_factor_operator.to_matrix(backend="numpy"),
        expected_boundary_potential_jump_factor,
    )
    assert geometry.m_ind_to_gridded_JS().shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )
    assert geometry.m_imp_to_gridded_JS().shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.horizontal_basis.index_length,
    )

    plot_grid = Grid(theta=geometry.model_grid.theta[:10], phi=geometry.model_grid.phi[:10])
    plot_transform = SphericalTransform(simulation.geometry.horizontal_basis, plot_grid)
    assert geometry.poloidal_transform_for(
        plot_transform
    ) is geometry.poloidal_transform_for(plot_transform)
    assert geometry.m_ind_to_gridded_JS(plot_transform).shape == (
        2,
        plot_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )

    state = simulation.run_data.output_series.datasets["state"]
    assert "SH_m_ind" in state
    assert "CS_m_imp" in state

    coeff_array = np.hstack((state["SH_m_ind"].values[-1], state["CS_m_imp"].values[-1]))

    actual_n_coeffs = coeff_array.shape[0]

    assert actual_n_coeffs == (
        simulation.geometry.poloidal_basis.index_length
        + simulation.geometry.horizontal_basis.index_length
    )
    assert np.all(np.isfinite(coeff_array))

    assert np.all(simulation.geometry.poloidal_basis.n > 0)
    for name in ("m_imp", "Phi", "W"):
        assert simulation.geometry.horizontal_basis.scalar_mean(
            state[f"CS_{name}"].values[-1]
        ) == (pytest.approx(0.0, abs=1e-18))
    assert simulation.geometry.horizontal_basis.scalar_mean(simulation.response.jr.array) == (
        pytest.approx(0.0, abs=1e-18)
    )

    view = SavedCoefficientFieldView.from_directory(
        simulation.run_data.run_directory, nlat=8, nlon=12, wind_nlat=5, wind_nlon=7
    )
    fields = view.state_comparison_grid_fields(0)
    assert isinstance(view.state_evaluator.basis, CSBasis)
    assert fields["Br_state"].shape == view.lat.shape
    assert fields["jr_state"].shape == view.lat.shape
    assert np.all(np.isfinite(fields["Br_state"]))
    assert np.all(np.isfinite(fields["jr_state"]))

    renderer = GroundFigureRenderer(
        PynamitFigureSpec(
            run_directory=simulation.run_data.run_directory, include_station_data=False
        ),
        view=view,
    )
    br_ind, bh_ind, _, _ = renderer._ground_field_matrices([65.0], [0.0])
    assert br_ind.shape == (1, view.n_time)
    assert bh_ind.shape == (2, 1, view.n_time)
    assert np.all(np.isfinite(br_ind))
    assert np.all(np.isfinite(bh_ind))
