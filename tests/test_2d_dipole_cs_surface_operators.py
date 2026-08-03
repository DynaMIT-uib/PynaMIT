"""Dipole test using CS surface operators."""

import numpy as np
import pytest
from kompe import GlobalCSBasis, Grid, SHBasis, SphericalTransform

from pynamit.simulation.workflows.standard import run_pynamit
from pynamit.visualization.figure_specs import PynamitFigureSpec
from pynamit.visualization.ground_figures import GroundFigureRenderer
from pynamit.visualization.run_fields import SavedCoefficientFieldView


def test_2d_dipole_cs_surface_operators(tmp_path):
    """Run a dipole case with CS fields and saved-run views."""
    simulation = run_pynamit(
        final_time=0.01,
        dt=0.01,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        equilibrium_initialization=False,
        use_wind=False,
        run_directory=str(tmp_path / "run"),
        boundary_jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        least_squares_solver="normal_pinv",
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    assert isinstance(simulation.geometry.horizontal_basis, GlobalCSBasis)
    assert isinstance(simulation.geometry.solid_harmonics.basis.root_basis, SHBasis)
    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert not simulation.run_data.schema.input_field_spaces["conductance"].mean_free
    output_spaces = simulation.run_data.schema.output_field_spaces["dynamic"]
    assert output_spaces["induced_Br"].representation is simulation.geometry.poloidal_basis
    assert output_spaces["boundary_jr"].representation is simulation.geometry.horizontal_basis

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
    assert geometry.induced_Br_to_gridded_JS().shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )
    assert geometry.boundary_jr_to_gridded_JS().shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.horizontal_basis.index_length,
    )

    plot_grid = Grid(theta=geometry.model_grid.theta[:10], phi=geometry.model_grid.phi[:10])
    plot_transform = SphericalTransform(simulation.geometry.horizontal_basis, plot_grid)
    assert geometry.poloidal_transform_for(plot_transform) is geometry.poloidal_transform_for(
        plot_transform
    )
    assert geometry.induced_Br_to_gridded_JS(plot_transform).shape == (
        2,
        plot_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )

    output = simulation.run_data.output_series.datasets["dynamic"]
    assert "SH_induced_Br" in output
    assert "CS_boundary_jr" in output

    induced_Br = output["SH_induced_Br"].values[-1]
    boundary_jr = output["CS_boundary_jr"].values[-1]

    actual_n_coeffs = induced_Br.size + boundary_jr.size

    assert actual_n_coeffs == (
        simulation.geometry.poloidal_basis.index_length
        + simulation.geometry.horizontal_basis.index_length
    )
    assert np.all(np.isfinite(induced_Br))
    assert np.all(np.isfinite(boundary_jr))

    assert np.all(simulation.geometry.poloidal_basis.n > 0)
    for name in ("Phi", "W"):
        assert simulation.geometry.horizontal_basis.scalar_mean(
            output[f"CS_{name}"].values[-1]
        ) == (pytest.approx(0.0, abs=1e-18))
    reconstructed_potential = (
        simulation.geometry.boundary_jr_to_toroidal_potential_operator.matvec(boundary_jr)
    )
    assert simulation.geometry.horizontal_basis.scalar_mean(reconstructed_potential) == (
        pytest.approx(0.0, abs=1e-18)
    )
    assert simulation.geometry.horizontal_basis.scalar_mean(
        simulation.response.boundary_jr.array
    ) == (pytest.approx(0.0, abs=1e-18))

    view = SavedCoefficientFieldView.from_directory(
        simulation.run_data.run_directory, nlat=8, nlon=12, wind_nlat=5, wind_nlon=7
    )
    fields = view.solution_comparison_grid_fields(0)
    assert isinstance(view.output_evaluator.basis, GlobalCSBasis)
    assert fields["Br_dynamic"].shape == view.lat.shape
    assert fields["jr_dynamic"].shape == view.lat.shape
    assert np.all(np.isfinite(fields["Br_dynamic"]))
    assert np.all(np.isfinite(fields["jr_dynamic"]))

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
