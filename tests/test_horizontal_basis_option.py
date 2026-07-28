"""Tests for selecting the horizontal calculation basis."""

import numpy as np
import pytest

from pynamit.math.constants import RE
from pynamit.math.tensor_operations import tensor_pinv, weighted_tensor_pinv
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
    assert simulation.run_data.schema.input_field_spaces["jr"].mean_free
    assert simulation.run_data.schema.input_field_spaces["Br"].mean_free
    assert simulation.run_data.schema.input_field_spaces["u"].mean_free
    assert not simulation.run_data.schema.input_field_spaces["conductance"].mean_free
    assert all(
        field_space.mean_free
        for field_space in simulation.run_data.schema.output_field_spaces["state"].values()
    )


def test_cs_surface_gauge_makes_m_imp_system_unique(tmp_path):
    """The CS constant gauge is constrained without regularization."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        m_imp_regularization_lambda=0.0,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    gauge = geometry.surface_gauge_operator
    assert gauge is not None
    np.testing.assert_allclose(
        gauge.matvec(np.ones(geometry.horizontal_basis.index_length)),
        np.sqrt(geometry.horizontal_basis.index_length),
    )

    system = simulation.response._m_imp_problem.data_operator.to_matrix(backend="numpy")
    assert np.linalg.matrix_rank(system) == geometry.horizontal_basis.index_length


def test_cs_runtime_m_imp_solve_does_not_build_dense_response_matrix(tmp_path):
    """A single current input should remain a single m_imp solve."""
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
    simulation.set_jr(jr_coefficients=np.linspace(-1.0, 1.0, n), time=0.0)
    response = simulation.response
    response.activate_inputs_at_time(simulation.run_data.input_series, 0.0)

    _, direct_m_imp = response.calculate_noninductive_response()

    assert response._jr_to_m_imp_matrix is None
    assert response._runtime_m_imp_to_E_coeffs._cached_dense(np) is None
    explicit_m_imp = response.jr_to_m_imp_operator.matvec(response.jr.array)
    np.testing.assert_allclose(direct_m_imp, explicit_m_imp, atol=1e-12)


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

    reduced = response.m_ind_to_E_df_operator.to_matrix(backend="numpy")
    full = (response.driving_E_to_E_df_operator @ response.m_ind_to_E_coeffs).to_matrix(
        backend="numpy"
    )

    assert reduced.shape == (
        simulation.geometry.horizontal_basis.index_length,
        simulation.geometry.poloidal_basis.index_length,
    )
    np.testing.assert_allclose(reduced, full, rtol=1e-11, atol=1e-12)


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


def test_cs_horizontal_basis_runs_with_split_state_spaces(tmp_path):
    """CS surface fields coexist with the poloidal SH state."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        use_wind=False,
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    state = simulation.run_data.output_series.datasets["state"]
    assert "SH_m_ind" in state
    assert "CS_m_imp" in state
    assert state["SH_m_ind"].shape[-1] == simulation.geometry.poloidal_basis.index_length
    assert state["CS_m_imp"].shape[-1] == simulation.geometry.horizontal_basis.index_length
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
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry
    pfac = geometry.pfac_coupling_matrix
    assert isinstance(pfac, np.ndarray)
    assert not pfac.flags.writeable

    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert pfac.shape == (
        simulation.geometry.poloidal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.linalg.norm(pfac) > 0.0
    assert np.all(np.isfinite(pfac))
    assert np.all(np.isfinite(geometry.m_imp_to_gridded_JS()))


def test_cs_horizontal_basis_supports_rm_solid_harmonics(tmp_path):
    """CS horizontal basis can use solid harmonics for RM terms."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * RE,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    m_ind_to_JS = geometry.m_ind_to_gridded_JS()
    Br_to_JS = geometry.Br_to_gridded_JS()

    assert m_ind_to_JS.shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.poloidal_basis.index_length,
    )
    assert Br_to_JS.shape == m_ind_to_JS.shape
    assert np.all(np.isfinite(m_ind_to_JS))
    assert np.all(np.isfinite(Br_to_JS))


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
        jr_projection_basis="CS",
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
        RM=4 * RE,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        jr_projection_basis="CS",
        Br_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.pfac_coupling_matrix.shape == (
        simulation.geometry.poloidal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert geometry.Br_to_gridded_JS().shape == (geometry.m_ind_to_gridded_JS().shape)
    assert geometry.interhemispheric_electric_field_difference_matrix.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.linalg.norm(geometry.pfac_coupling_matrix) > 0.0
    assert np.all(np.isfinite(geometry.pfac_coupling_matrix))
    assert np.all(np.isfinite(geometry.Br_to_gridded_JS()))


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
