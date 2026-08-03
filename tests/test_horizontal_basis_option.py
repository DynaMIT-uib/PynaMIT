"""Tests for selecting the horizontal calculation basis."""

import numpy as np
import pytest
from kompe.constants import EARTH_RADIUS_M
from kompe.math import tensor_pinv, weighted_tensor_pinv

from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.standard import run_pynamit


def test_default_horizontal_basis_is_sh(tmp_path):
    """Default horizontal basis is SH with radial continuation."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    assert simulation.run_data.config.horizontal_basis_kind == "SH"
    assert simulation.geometry.horizontal_basis is simulation.geometry.solid_harmonics.basis
    geometry = simulation.geometry
    schema = simulation.run_data.schema
    assert geometry.horizontal_basis.mean_free
    assert not schema.sh_basis.mean_free
    assert schema.sh_basis.index_length == geometry.horizontal_basis.index_length + 1
    assert geometry.surface_gauge_operator is None
    np.testing.assert_allclose(
        geometry.surface_to_poloidal_operator.to_matrix(backend="numpy"),
        np.eye(geometry.poloidal_basis.index_length),
    )


def test_horizontal_basis_kind_is_persisted(tmp_path):
    """Explicit horizontal basis choice keeps SH radial continuation."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="cs",
        artifact_storage="netcdf",
    )

    assert simulation.run_data.config.horizontal_basis_kind == "CS"
    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert simulation.run_data.schema.input_field_spaces["boundary_jr"].mean_free
    assert simulation.run_data.schema.input_field_spaces["boundary_Br"].mean_free
    assert simulation.run_data.schema.input_field_spaces["u"].mean_free
    assert not simulation.run_data.schema.input_field_spaces["conductance"].mean_free
    output_spaces = simulation.run_data.schema.output_field_spaces["dynamic"]
    assert output_spaces["induced_Br"].mean_free
    assert not output_spaces["boundary_jr"].mean_free
    assert output_spaces["Phi"].mean_free
    assert output_spaces["W"].mean_free


def test_cs_surface_gauge_makes_toroidal_potential_system_unique(tmp_path):
    """The CS constant gauge is constrained without regularization."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        toroidal_potential_regularization_lambda=0.0,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    gauge = geometry.surface_gauge_operator
    assert gauge is not None
    np.testing.assert_allclose(
        gauge.matvec(np.ones(geometry.horizontal_basis.index_length)),
        np.sqrt(geometry.horizontal_basis.index_length),
    )

    system = simulation.response._toroidal_potential_problem.data_operator.to_matrix(
        backend="numpy"
    )
    assert np.linalg.matrix_rank(system) == geometry.horizontal_basis.index_length


def test_cs_runtime_toroidal_solve_does_not_build_dense_response_matrix(tmp_path):
    """A single current input should remain a single toroidal solve."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        least_squares_solver="normal_pinv",
        artifact_storage="netcdf",
        backend="numpy",
    )
    n = simulation.geometry.horizontal_basis.index_length
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(n), log_ratio_coefficients=np.zeros(n), time=0.0
    )
    simulation.set_boundary_jr(boundary_jr_coefficients=np.linspace(-1.0, 1.0, n), time=0.0)
    response = simulation.response
    response.activate_inputs_at_time(simulation.run_data.input_series, 0.0)

    _, solved_boundary_jr = response.calculate_noninductive_response()

    assert response._boundary_jr_to_toroidal_potential_matrix is None
    assert response._runtime_toroidal_potential_to_E_coeffs._cached_dense(np) is None
    explicit_toroidal_potential = response.boundary_jr_to_toroidal_potential_operator.matvec(
        response.boundary_jr.array
    )
    expected_boundary_jr = simulation.geometry.toroidal_potential_to_boundary_jr_operator.matvec(
        explicit_toroidal_potential
    )
    np.testing.assert_allclose(solved_boundary_jr, expected_boundary_jr, atol=1e-12)


def test_cs_reduced_induction_response_matches_full_E_response(tmp_path):
    """Reduced poloidal columns preserve interhemispheric feedback."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        horizontal_basis_kind="CS",
        least_squares_solver="normal_pinv",
        artifact_storage="netcdf",
        backend="numpy",
    )
    grid = simulation.geometry.model_grid
    phase = np.linspace(0.0, 2.0 * np.pi, grid.size, endpoint=False)
    simulation.set_conductance(
        1.0 + 0.1 * np.sin(2.0 * phase),
        2.0 + 0.2 * np.cos(phase),
        lat=grid.lat,
        lon=grid.lon,
        time=0.0,
    )
    response = simulation.response
    response.activate_inputs_at_time(simulation.run_data.input_series, 0.0)

    reduced = response.induced_Br_to_E_df_operator.to_matrix(backend="numpy")
    full = (response.driving_E_to_E_df_operator @ response.induced_Br_to_E_coeffs).to_matrix(
        backend="numpy"
    )

    assert reduced.shape == (
        simulation.geometry.horizontal_basis.index_length,
        simulation.geometry.poloidal_basis.index_length,
    )
    np.testing.assert_allclose(reduced, full, rtol=1e-10, atol=1e-12)


def test_area_weighted_least_squares_option_is_persisted(tmp_path):
    """Area-weighted fits are a persisted global option."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    assert simulation.run_data.config.area_weighted_least_squares
    assert geometry.area_weighted_least_squares
    np.testing.assert_allclose(
        geometry.model_grid_sqrt_weights(), np.sqrt(simulation.run_data.schema.cs_basis.unit_area)
    )
    np.testing.assert_allclose(
        geometry.model_grid_sqrt_weights(vector=True),
        np.tile(np.sqrt(simulation.run_data.schema.cs_basis.unit_area), (2, 1)),
    )


def test_cs_horizontal_basis_runs_with_split_output_spaces(tmp_path):
    """CS surface fields coexist with the poloidal SH output."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        use_wind=False,
        boundary_jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    output = simulation.run_data.output_series.datasets["dynamic"]
    assert "SH_induced_Br" in output
    assert "CS_boundary_jr" in output
    assert output["SH_induced_Br"].shape[-1] == simulation.geometry.poloidal_basis.index_length
    assert output["CS_boundary_jr"].shape[-1] == simulation.geometry.horizontal_basis.index_length
    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis


def test_cs_horizontal_basis_runs_with_pfac(tmp_path):
    """CS horizontal basis can use SH radial continuation for PFAC."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=True,
        use_wind=False,
        boundary_jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry
    gap_Br_response = geometry.boundary_jr_to_gap_Br_matrix
    assert isinstance(gap_Br_response, np.ndarray)
    assert not gap_Br_response.flags.writeable

    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert gap_Br_response.shape == (
        simulation.geometry.poloidal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.linalg.norm(gap_Br_response) > 0.0
    assert np.all(np.isfinite(gap_Br_response))
    assert np.all(np.isfinite(geometry.boundary_jr_to_gridded_JS()))


def test_cs_horizontal_basis_supports_rm_solid_harmonics(tmp_path):
    """CS horizontal basis can use solid harmonics for RM terms."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * EARTH_RADIUS_M,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    induced_Br_to_JS = geometry.induced_Br_to_gridded_JS()
    boundary_Br_to_JS = geometry.boundary_Br_to_gridded_JS()

    assert induced_Br_to_JS.shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )
    assert boundary_Br_to_JS.shape == induced_Br_to_JS.shape
    assert np.all(np.isfinite(induced_Br_to_JS))
    assert np.all(np.isfinite(boundary_Br_to_JS))


def test_cs_horizontal_basis_supports_connected_hemispheres(tmp_path):
    """CS horizontal basis can evaluate conjugate Helmholtz terms."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        boundary_jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.conjugate_horizontal_transform.helmholtz_coeffs_to_gridded_vector.shape == (
        2,
        geometry.conjugate_grid.size,
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert geometry.interhemispheric_electric_field_difference_matrix.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.all(
        np.isfinite(geometry.conjugate_horizontal_transform.helmholtz_coeffs_to_gridded_vector)
    )
    assert np.all(np.isfinite(geometry.interhemispheric_electric_field_difference_matrix))


def test_connected_E_apex_constraint_operator_is_lazy(tmp_path):
    """Connected E-apex constraint stays operator-backed."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    operator = geometry.interhemispheric_electric_field_difference_operator
    assert operator is not None
    assert "interhemispheric_electric_field_difference_matrix" not in geometry.__dict__

    rng = np.random.default_rng(20260612)
    coeffs = rng.standard_normal(operator.input_shape)

    actual = operator.matvec(coeffs).reshape(operator.output_shape)
    explicit = geometry.interhemispheric_electric_field_difference_matrix
    expected = np.tensordot(explicit, coeffs, axes=([2, 3], [0, 1]))

    np.testing.assert_allclose(actual, expected)


def test_cs_horizontal_basis_combines_pfac_rm_and_connected_terms(tmp_path):
    """CS horizontal basis supports the combined radial/coupled path."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * EARTH_RADIUS_M,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        boundary_jr_projection_basis="CS",
        boundary_Br_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.boundary_jr_to_gap_Br_matrix.shape == (
        simulation.geometry.poloidal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert geometry.boundary_Br_to_gridded_JS().shape == (
        geometry.induced_Br_to_gridded_JS().shape
    )
    assert geometry.interhemispheric_electric_field_difference_matrix.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.linalg.norm(geometry.boundary_jr_to_gap_Br_matrix) > 0.0
    assert np.all(np.isfinite(geometry.boundary_jr_to_gap_Br_matrix))
    assert np.all(np.isfinite(geometry.boundary_Br_to_gridded_JS()))


def test_surface_to_poloidal_projection_matches_grid_least_squares(tmp_path):
    """The CS-surface to magnetic-SH bridge uses grid least squares."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    assert not hasattr(geometry.horizontal_transform, "_scalar_coeffs_to_grid")
    expected = tensor_pinv(
        geometry.poloidal_transform.scalar_coeffs_to_grid, n_leading_flattened=1
    )

    surface_to_poloidal = geometry.surface_to_poloidal_operator.to_matrix(backend="numpy")
    np.testing.assert_allclose(surface_to_poloidal, expected)

    rng = np.random.default_rng(20260520)
    radial_coeffs = rng.standard_normal(simulation.geometry.solid_harmonics.basis.index_length)
    cs_coeffs = geometry.poloidal_transform.scalar_coeffs_to_grid @ radial_coeffs

    np.testing.assert_allclose(surface_to_poloidal @ cs_coeffs, radial_coeffs, atol=1e-10)


def test_surface_to_poloidal_supports_area_weighted_projection(tmp_path):
    """The surface-to-magnetic bridge can use CS cell-area weighting."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    assert not hasattr(geometry.horizontal_transform, "_scalar_coeffs_to_grid")
    expected = weighted_tensor_pinv(
        geometry.poloidal_transform.scalar_coeffs_to_grid,
        sqrt_weights=np.sqrt(simulation.run_data.schema.cs_basis.unit_area),
        n_leading_flattened=1,
    )

    np.testing.assert_allclose(
        geometry.surface_to_poloidal_operator.to_matrix(backend="numpy"), expected
    )


def test_invalid_horizontal_basis_kind_is_rejected(tmp_path):
    """Unknown horizontal basis names fail early."""
    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        Simulation(
            run_directory=str(tmp_path / "run"),
            Nmax=2,
            Mmax=1,
            Ncs=4,
            enable_pfac_coupling=False,
            horizontal_basis_kind="spectral",
            artifact_storage="netcdf",
        )
