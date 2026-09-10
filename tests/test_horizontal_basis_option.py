"""Tests for selecting the horizontal calculation basis."""

import numpy as np
import pytest
from kompe import SphericalGrid, SphericalTransform
from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.math import LeastSquaresSolver

from pynamit.results.output_fields import OutputEvaluation
from pynamit.simulation.electrodynamics import magnetic_boundary
from pynamit.simulation.simulation import Simulation
from tests.example_scenario import run_example


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
@pytest.mark.parametrize("enable_pfac_coupling", [False, True])
def test_sheet_current_maps_share_grid_and_resolve_bases(
    tmp_path, monkeypatch, horizontal_basis_kind, enable_pfac_coupling
):
    """One transform suffices for all physical current sources."""
    RI = 6.5e6
    simulation = Simulation(
        tmp_path / "run",
        Nmax=2,
        Mmax=2,
        Ncs=4,
        RI=RI,
        RM=2 * RI,
        main_field_kind="dipole",
        horizontal_basis_kind=horizontal_basis_kind,
        enable_pfac_coupling=enable_pfac_coupling,
        fac_integration_radii=[RI, 1.1 * RI],
        enable_interhemispheric_coupling=False,
    )
    geometry = simulation.geometry
    assert (
        geometry.horizontal_transform.with_basis(geometry.poloidal_basis)
        is geometry.poloidal_transform
    )
    grid = SphericalGrid(theta=[0.4, 0.8, 1.2, 2.5], phi=[0.1, 0.7, 1.9, 3.4])
    # This full SH plotting basis differs from the CS horizontal basis
    # and the mean-free SH basis used for radial magnetic fields.
    transform = SphericalTransform(simulation.results.geometry.sh_basis, grid)
    horizontal = transform.with_basis(geometry.horizontal_basis)
    poloidal = horizontal.with_basis(geometry.poloidal_basis)
    direct_current = (-1 / MU0) * horizontal.surface_gradient_operator
    expected_jr = direct_current @ geometry.boundary_jr_to_toroidal_potential_operator
    expected_toroidal = direct_current
    if enable_pfac_coupling:
        gap_current = (
            magnetic_boundary.external_Br_to_gridded_JS_operator(
                geometry.solid_harmonics, poloidal
            )
            @ geometry.boundary_jr_to_gap_Br_operator
        )
        expected_jr = expected_jr + gap_current
        expected_toroidal = (
            expected_toroidal + gap_current @ geometry.toroidal_potential_to_boundary_jr_operator
        )
    expected = {
        "boundary_jr_to_JS": expected_jr,
        "induced_Br_to_JS": magnetic_boundary.induced_Br_to_gridded_JS_operator(
            geometry.solid_harmonics, poloidal, radius=RI
        ),
        "boundary_Br_to_JS": magnetic_boundary.boundary_Br_to_gridded_JS_operator(
            geometry.solid_harmonics, poloidal, radius=RI, boundary_radius=2 * RI
        ),
    }

    def unexpected_transform(*args, **kwargs):
        pytest.fail("Sheet-current construction should reuse the cached basis transforms.")

    monkeypatch.setattr(SphericalTransform, "__init__", unexpected_transform)
    for _ in range(2):
        operators = OutputEvaluation(geometry, transform=transform).sheet_current_operators
        for name, operator in operators.items():
            assert operator.output_shape == (2, grid.size)
            np.testing.assert_allclose(
                operator.to_array(), expected[name].to_array(), rtol=1e-12, atol=1e-12
            )
    np.testing.assert_allclose(
        geometry.toroidal_potential_to_gridded_JS_operator(transform).to_array(),
        expected_toroidal.to_array(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_default_horizontal_basis_is_sh(tmp_path):
    """Default horizontal basis is SH with radial continuation."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    assert simulation.results.config.horizontal_basis_kind == "SH"
    assert simulation.geometry.horizontal_basis is simulation.geometry.solid_harmonics.basis
    geometry = simulation.geometry
    assert geometry.horizontal_basis.mean_free
    assert not geometry.sh_basis.mean_free
    assert geometry.sh_basis.coefficient_count == geometry.horizontal_basis.coefficient_count + 1
    assert simulation.response._toroidal_potential_problem.constraints is None
    np.testing.assert_allclose(
        geometry.surface_to_poloidal_operator.to_matrix(backend="numpy"),
        np.eye(geometry.poloidal_basis.coefficient_count),
    )


def test_horizontal_basis_kind_is_persisted(tmp_path):
    """Explicit horizontal basis choice keeps SH radial continuation."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="cs",
        artifact_storage="netcdf",
    )

    assert simulation.results.config.horizontal_basis_kind == "CS"
    assert simulation.results.geometry.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert simulation.results.schema.input_field_spaces["boundary_jr"].mean_free
    assert simulation.results.schema.input_field_spaces["boundary_Br"].mean_free
    assert simulation.results.schema.input_field_spaces["u"].mean_free
    assert not simulation.results.schema.input_field_spaces["conductance"].mean_free
    output_spaces = simulation.results.schema.output_field_spaces["dynamic"]
    assert output_spaces["induced_Br"].mean_free
    assert not output_spaces["boundary_jr"].mean_free
    assert output_spaces["Phi"].mean_free
    assert output_spaces["W"].mean_free


def test_cs_surface_gauge_makes_toroidal_potential_system_unique(tmp_path):
    """The CS constant gauge is constrained without regularization."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        toroidal_potential_regularization_lambda=0.0,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    problem = simulation.response._toroidal_potential_problem
    gauge = problem.solution_basis
    assert gauge is not None
    np.testing.assert_allclose(
        geometry.horizontal_basis.scalar_mean_weights
        @ gauge.matmat(np.eye(geometry.horizontal_basis.coefficient_count - 1)),
        0.0,
        atol=1e-14,
    )

    system = simulation.response._toroidal_potential_problem.data_operator.to_matrix(
        backend="numpy"
    )
    assert np.linalg.matrix_rank(system) == geometry.horizontal_basis.coefficient_count - 1


def test_cs_runtime_toroidal_solve_does_not_build_dense_response_matrix(tmp_path):
    """A single current input should remain a single toroidal solve."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        least_squares_solver="normal_pinv",
        artifact_storage="netcdf",
    )
    n = simulation.geometry.horizontal_basis.coefficient_count
    simulation.inputs.set_coefficients(
        "conductance",
        {"log_conductance_magnitude": np.zeros(n), "log_hall_to_pedersen_ratio": np.zeros(n)},
        time=0.0,
    )
    simulation.inputs.set_coefficients("boundary_jr", np.linspace(-1.0, 1.0, n), time=0.0)
    response = simulation.response_at_time(0.0)

    forcing = simulation.results.input_series.get_entry("boundary_jr", 0.0)
    _, solved_boundary_jr = response.solve_noninductive_response(**forcing)

    assert "boundary_jr_to_toroidal_potential_operator" not in response.__dict__
    assert np not in response.toroidal_potential_to_E_coeffs_operator._dense_cache
    explicit_toroidal_potential = response.boundary_jr_to_toroidal_potential_operator.matvec(
        forcing["boundary_jr"]
    )
    expected_boundary_jr = simulation.geometry.toroidal_potential_to_boundary_jr_operator.matvec(
        explicit_toroidal_potential
    )
    np.testing.assert_allclose(solved_boundary_jr, expected_boundary_jr, atol=1e-12)


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_toroidal_gauge_does_not_depend_on_current_residual_units(algorithm, scale):
    """Residual units do not change the zero-mean potential."""
    simulation = Simulation(
        Nmax=2,
        Mmax=1,
        Ncs=4,
        horizontal_basis_kind="CS",
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=False,
        toroidal_potential_regularization_lambda=0.0,
        least_squares_solver=algorithm,
        least_squares_tolerance=1e-12,
    )
    geometry = simulation.geometry
    geometry.radial_current_constraint_operator = (
        scale * geometry.radial_current_constraint_operator
    )
    n = geometry.horizontal_basis.coefficient_count
    potential = geometry.horizontal_basis.project_scalar_mean_free(
        np.random.default_rng(87).normal(size=n)
    )
    current = geometry.toroidal_potential_to_boundary_jr_operator(potential)
    actual = simulation.response._solve_toroidal_potential(current, np.zeros((2, n)))
    np.testing.assert_allclose(actual, potential, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(geometry.horizontal_basis.scalar_mean(actual), 0.0, atol=1e-12)


def test_cs_reduced_induction_response_matches_full_E_response(tmp_path):
    """Reduced poloidal columns preserve interhemispheric feedback."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        horizontal_basis_kind="CS",
        least_squares_solver="normal_pinv",
        artifact_storage="netcdf",
    )
    grid = simulation.geometry.model_grid
    phase = np.linspace(0.0, 2.0 * np.pi, grid.size, endpoint=False)
    simulation.inputs.set_conductance(
        pedersen=2.0 + 0.2 * np.cos(phase),
        hall=1.0 + 0.1 * np.sin(2.0 * phase),
        time=0.0,
        grid=grid,
    )
    response = simulation.response_at_time(0.0)

    reduced = response.induced_Br_to_W_operator.to_matrix(backend="numpy")
    full = (response.driving_E_to_W_operator @ response.induced_Br_to_E_coeffs_operator).to_matrix(
        backend="numpy"
    )

    assert reduced.shape == (
        simulation.geometry.horizontal_basis.coefficient_count,
        simulation.geometry.poloidal_basis.coefficient_count,
    )
    np.testing.assert_allclose(reduced, full, rtol=1e-10, atol=1e-12)


def test_area_weighted_least_squares_option_is_persisted(tmp_path):
    """Area-weighted fits are a persisted global option."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    assert simulation.results.config.area_weighted_least_squares
    assert geometry.area_weighted_least_squares
    np.testing.assert_allclose(
        geometry.model_grid_sqrt_weights(),
        np.sqrt(simulation.results.geometry.cs_basis.mesh.cell_areas.reshape(-1)),
    )
    np.testing.assert_allclose(
        geometry.model_grid_sqrt_weights(vector=True),
        np.tile(np.sqrt(simulation.results.geometry.cs_basis.mesh.cell_areas.reshape(-1)), (2, 1)),
    )


def test_cs_horizontal_basis_runs_with_split_output_spaces(tmp_path):
    """CS surface fields coexist with the poloidal SH output."""
    simulation = run_example(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        use_wind=False,
        boundary_jr_remapping="CS",
        conductance_basis="CS",
        u_remapping="CS",
        simulation_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    output = simulation.results.output_series.datasets["dynamic"]
    assert "SH_induced_Br" in output
    assert "CS_boundary_jr" in output
    assert (
        output["SH_induced_Br"].shape[-1] == simulation.geometry.poloidal_basis.coefficient_count
    )
    assert (
        output["CS_boundary_jr"].shape[-1]
        == simulation.geometry.horizontal_basis.coefficient_count
    )
    assert simulation.results.geometry.horizontal_basis is simulation.geometry.horizontal_basis


def test_cs_horizontal_basis_runs_with_pfac(tmp_path):
    """CS horizontal basis can use SH radial continuation for PFAC."""
    simulation = run_example(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=True,
        use_wind=False,
        boundary_jr_remapping="CS",
        conductance_basis="CS",
        u_remapping="CS",
        simulation_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry
    response_matrix = geometry.boundary_jr_to_gap_Br_matrix
    assert isinstance(response_matrix, np.ndarray)
    assert not response_matrix.flags.writeable

    assert simulation.results.geometry.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert response_matrix.shape == (
        simulation.geometry.poloidal_basis.coefficient_count,
        simulation.geometry.horizontal_basis.coefficient_count,
    )
    assert np.linalg.norm(response_matrix) > 0.0
    assert np.all(np.isfinite(response_matrix))
    assert np.all(np.isfinite(geometry.boundary_jr_to_gridded_JS_operator().to_array()))


def test_cs_horizontal_basis_supports_rm_solid_harmonics(tmp_path):
    """CS horizontal basis can use solid harmonics for RM terms."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * EARTH_RADIUS_M,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    induced_Br_to_JS = geometry.induced_Br_to_gridded_JS_operator().to_array()
    boundary_Br_to_JS = geometry.boundary_Br_to_gridded_JS_operator().to_array()

    assert induced_Br_to_JS.shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.poloidal_basis.coefficient_count,
    )
    assert boundary_Br_to_JS.shape == induced_Br_to_JS.shape
    assert np.all(np.isfinite(induced_Br_to_JS))
    assert np.all(np.isfinite(boundary_Br_to_JS))


def test_cs_horizontal_basis_supports_connected_hemispheres(tmp_path):
    """CS horizontal basis can evaluate conjugate Helmholtz terms."""
    simulation = run_example(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        boundary_jr_remapping="CS",
        conductance_basis="CS",
        u_remapping="CS",
        simulation_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.conjugate_horizontal_transform.helmholtz_synthesis_array.shape == (
        2,
        geometry.conjugate_grid.size,
        2,
        simulation.geometry.horizontal_basis.coefficient_count,
    )
    assert geometry.interhemispheric_electric_field_difference_array.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.coefficient_count,
    )
    assert np.all(np.isfinite(geometry.conjugate_horizontal_transform.helmholtz_synthesis_array))
    assert np.all(np.isfinite(geometry.interhemispheric_electric_field_difference_array))


def test_connected_E_apex_constraint_operator_is_lazy(tmp_path):
    """Connected E-apex constraint stays operator-backed."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
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
    assert "interhemispheric_electric_field_difference_array" not in geometry.__dict__

    rng = np.random.default_rng(20260612)
    coeffs = rng.standard_normal(operator.input_shape)

    actual = operator.matvec(coeffs).reshape(operator.output_shape)
    explicit = geometry.interhemispheric_electric_field_difference_array
    expected = np.tensordot(explicit, coeffs, axes=([2, 3], [0, 1]))

    np.testing.assert_allclose(actual, expected)


def test_cs_horizontal_basis_combines_pfac_rm_and_connected_terms(tmp_path):
    """CS horizontal basis supports the combined radial/coupled path."""
    simulation = run_example(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * EARTH_RADIUS_M,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        boundary_jr_remapping="CS",
        boundary_Br_remapping="CS",
        conductance_basis="CS",
        u_remapping="CS",
        simulation_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.boundary_jr_to_gap_Br_matrix.shape == (
        simulation.geometry.poloidal_basis.coefficient_count,
        simulation.geometry.horizontal_basis.coefficient_count,
    )
    assert geometry.boundary_Br_to_gridded_JS_operator().to_array().shape == (
        geometry.induced_Br_to_gridded_JS_operator().to_array().shape
    )
    assert geometry.interhemispheric_electric_field_difference_array.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.coefficient_count,
    )
    assert np.linalg.norm(geometry.boundary_jr_to_gap_Br_matrix) > 0.0
    assert np.all(np.isfinite(geometry.boundary_jr_to_gap_Br_matrix))
    assert np.all(np.isfinite(geometry.boundary_Br_to_gridded_JS_operator().to_array()))


def test_surface_to_poloidal_projection_matches_grid_least_squares(tmp_path):
    """The CS-surface to magnetic-SH bridge uses grid least squares."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    synthesis = np.asarray(geometry.poloidal_transform.scalar_synthesis_array)
    expected = np.linalg.pinv(synthesis)

    surface_to_poloidal = geometry.surface_to_poloidal_operator.to_matrix(backend="numpy")
    np.testing.assert_allclose(surface_to_poloidal, expected)

    rng = np.random.default_rng(20260520)
    radial_coeffs = rng.standard_normal(
        simulation.geometry.solid_harmonics.basis.coefficient_count
    )
    cs_coeffs = geometry.poloidal_transform.scalar_synthesis_array @ radial_coeffs

    np.testing.assert_allclose(surface_to_poloidal @ cs_coeffs, radial_coeffs, atol=1e-10)


def test_surface_to_poloidal_supports_area_weighted_projection(tmp_path):
    """The surface-to-magnetic bridge can use CS cell-area weighting."""
    simulation = Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    synthesis = np.asarray(geometry.poloidal_transform.scalar_synthesis_array)
    sqrt_weights = np.sqrt(simulation.results.geometry.cs_basis.mesh.cell_areas.reshape(-1))
    expected = np.linalg.pinv(sqrt_weights[:, None] * synthesis) * sqrt_weights

    np.testing.assert_allclose(
        geometry.surface_to_poloidal_operator.to_matrix(backend="numpy"), expected
    )


def test_invalid_horizontal_basis_kind_is_rejected(tmp_path):
    """Unknown horizontal basis names fail early."""
    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        Simulation(
            simulation_directory=str(tmp_path / "run"),
            Nmax=2,
            Mmax=1,
            Ncs=4,
            enable_pfac_coupling=False,
            horizontal_basis_kind="spectral",
            artifact_storage="netcdf",
        )
