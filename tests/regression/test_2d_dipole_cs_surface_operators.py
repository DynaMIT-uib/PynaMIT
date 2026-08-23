"""Dipole test using CS surface operators."""

import numpy as np
import pytest
from kompe import GlobalCSBasis, SHBasis, SphericalGrid, SphericalTransform
from tests.example_scenario import run_example

from pynamit.plotting.figure_settings import FigureSettings
from pynamit.plotting.ground_figures import GroundFigureRenderer
from pynamit.plotting.plot_data import PlotData


def test_2d_dipole_cs_surface_operators(tmp_path):
    """Run a dipole case with CS fields and evaluate saved fields."""
    simulation = run_example(
        final_time=0.01,
        dt=0.01,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        initialize_from_equilibrium=False,
        use_wind=False,
        simulation_directory=str(tmp_path / "run"),
        boundary_jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        least_squares_solver="normal_pinv",
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    assert isinstance(simulation.geometry.horizontal_basis, GlobalCSBasis)
    assert isinstance(simulation.geometry.solid_harmonics.basis.root_basis, SHBasis)
    assert simulation.data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert not simulation.data.schema.input_field_spaces["conductance"].mean_free
    output_spaces = simulation.data.schema.output_field_spaces["dynamic"]
    assert output_spaces["induced_Br"].basis is simulation.geometry.poloidal_basis
    assert output_spaces["boundary_jr"].basis is simulation.geometry.horizontal_basis

    geometry = simulation.geometry
    spherical_transform = geometry.horizontal_transform
    expected_helmholtz = simulation.geometry.horizontal_basis.helmholtz_synthesis_matrix(
        geometry.model_grid
    )
    np.testing.assert_allclose(spherical_transform.helmholtz_synthesis_matrix, expected_helmholtz)
    np.testing.assert_allclose(
        geometry.surface_laplacian_operator.to_matrix(backend="numpy"),
        simulation.geometry.horizontal_basis.surface_laplacian_matrix(geometry.RI),
    )
    expected_boundary_potential_jump_factor = np.diag(
        simulation.geometry.solid_harmonics.poloidal_to_boundary_potential_jump_factor
    )
    np.testing.assert_allclose(
        geometry.poloidal_to_boundary_potential_jump_factor_operator.to_matrix(backend="numpy"),
        expected_boundary_potential_jump_factor,
    )
    assert geometry.induced_Br_to_gridded_JS_operator().array.shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )
    assert geometry.boundary_jr_to_gridded_JS_operator().array.shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.horizontal_basis.index_length,
    )

    plot_grid = SphericalGrid(
        theta=geometry.model_grid.theta[:10], phi=geometry.model_grid.phi[:10]
    )
    plot_transform = SphericalTransform(simulation.geometry.horizontal_basis, plot_grid)
    assert geometry.poloidal_transform_for(plot_transform) is geometry.poloidal_transform_for(
        plot_transform
    )
    assert geometry.induced_Br_to_gridded_JS_operator(plot_transform).array.shape == (
        2,
        plot_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )

    output = simulation.data.output_series.datasets["dynamic"]
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

    view = PlotData.from_directory(
        simulation.data.simulation_directory, nlat=8, nlon=12, wind_nlat=5, wind_nlon=7
    )
    fields = view.output_plot_data(0)
    assert isinstance(view.output_transform.basis, GlobalCSBasis)
    assert fields["Br_dynamic"].shape == view.lat.shape
    assert fields["jr_dynamic"].shape == view.lat.shape
    assert np.all(np.isfinite(fields["Br_dynamic"]))
    assert np.all(np.isfinite(fields["jr_dynamic"]))

    renderer = GroundFigureRenderer(
        FigureSettings(
            simulation_directory=simulation.data.simulation_directory, include_station_data=False
        ),
        plot_data=view,
    )
    br_dynamic, bh_dynamic, _, _ = renderer._ground_field_matrices([65.0], [0.0])
    assert br_dynamic.shape == (1, view.n_time)
    assert bh_dynamic.shape == (2, 1, view.n_time)
    assert np.all(np.isfinite(br_dynamic))
    assert np.all(np.isfinite(bh_dynamic))
