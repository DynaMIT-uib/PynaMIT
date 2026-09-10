"""Tests for electrodynamic response construction and application."""

from types import SimpleNamespace

import numpy as np
import pytest
from kompe.constants import EARTH_RADIUS_M
from kompe.math import (
    JAX_AVAILABLE,
    LeastSquaresSolver,
    LinearMap,
    as_linear_map,
    einsum_linear_map,
    get_array_module,
    get_backend,
    set_backend,
)

from pynamit.simulation.electrodynamics.ionospheric_closure import (
    resistance_from_log_conductance_coordinates,
)
from pynamit.simulation.geometry import SimulationGeometry
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.simulation.simulation import Simulation


def _dummy_constraint_map():
    return as_linear_map(np.eye(1), input_shape=(1,), output_shape=(1,))


@pytest.mark.parametrize("horizontal", ["SH", "CS"])
@pytest.mark.parametrize("coupling", [False, True])
@pytest.mark.parametrize("wind", ["u", "Q_eff", "E_neutral_wind"])
def test_forcing_batches_broadcast_and_match_independent_closures(horizontal, coupling, wind):
    """Two scientific component axes are not mistaken for batch axes."""
    simulation = Simulation(
        Nmax=2,
        Mmax=1,
        Ncs=4,
        RM=2 * EARTH_RADIUS_M,
        main_field_kind="dipole",
        horizontal_basis_kind=horizontal,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=coupling,
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=np.full(grid.size, 5.0), hall=np.full(grid.size, 3.0), grid=grid
    )
    response = simulation.response
    xp = get_array_module()
    n = simulation.geometry.horizontal_basis.coefficient_count
    wind_values = (
        xp.arange(2 * n, dtype=float).reshape(2, n, 1, 1)
        * xp.array([1.0, 2.0, 3.0])[None, None, None, :]
        * 1e-5
    )
    current = xp.linspace(-1e-9, 1e-9, n)[:, None, None] * xp.array([1.0, 2.0])[None, :, None]
    magnetic = xp.linspace(-1e-9, 1e-9, simulation.geometry.poloidal_basis.coefficient_count)
    magnetic = magnetic[:, None] * xp.array([1.0, 2.0, 3.0])[None, :]
    E, jr = response.solve_noninductive_response(
        **{wind: wind_values}, boundary_jr=current, boundary_Br=magnetic
    )
    assert E.shape == (2, n, 2, 3) and jr.shape == (n, 2, 3)
    for i in range(2):
        for j in range(3):
            expected = response.solve_noninductive_response(
                **{wind: wind_values[..., 0, j]},
                boundary_jr=current[..., i, 0],
                boundary_Br=magnetic[..., j],
            )
            for actual, scalar in zip((E[..., i, j], jr[..., i, j]), expected, strict=True):
                np.testing.assert_allclose(actual, scalar, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("horizontal", ["SH", "CS"])
def test_response_owns_single_conductance_fields(backend, horizontal):
    """Closure arrays are fixed single fields, not time histories."""
    simulation = Simulation(
        Nmax=2, Mmax=1, Ncs=4, horizontal_basis_kind=horizontal, enable_pfac_coupling=False
    )
    space = simulation.results.schema.input_field_spaces["conductance"]
    values = {
        "log_conductance_magnitude": np.ones((space.size, 1)),
        "log_hall_to_pedersen_ratio": np.zeros(space.shape),
    }
    response = ElectrodynamicResponse(
        simulation.geometry, simulation.config, conductance_space=space, conductance_values=values
    )
    for name, original in values.items():
        stored = getattr(response, name)
        assert stored.shape == space.shape
        np.testing.assert_array_equal(stored, original.reshape(space.shape))
        expected = np.asarray(stored).copy()
        original[:] = 42.0
        np.testing.assert_array_equal(stored, expected)
        if isinstance(stored, np.ndarray):
            assert not stored.flags.writeable

    values["log_conductance_magnitude"] = np.ones((space.size, 2))
    with pytest.raises(ValueError, match="one field"):
        ElectrodynamicResponse(
            simulation.geometry,
            simulation.config,
            conductance_space=space,
            conductance_values=values,
        )


@pytest.mark.parametrize("horizontal", ["SH", "CS"])
@pytest.mark.parametrize("solver", LeastSquaresSolver.VALID_SOLVERS)
def test_induced_outputs_reuse_the_compact_feedback_fit(backend, horizontal, solver, monkeypatch):
    """Evolution and batched outputs share one fixed closure map."""
    simulation = Simulation(
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="dipole",
        horizontal_basis_kind=horizontal,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        least_squares_solver=solver,
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=5 + np.cos(np.deg2rad(grid.theta)),
        hall=3 + np.sin(np.deg2rad(grid.theta)),
        grid=grid,
    )
    response = simulation.response
    geometry = simulation.geometry
    xp = get_array_module()
    Br = xp.linspace(-1e-9, 2e-9, geometry.poloidal_basis.coefficient_count)
    expected_E, expected_jr = response._solve_electric_closure(
        response.induced_Br_to_E_coeffs_operator(Br), None
    )
    original = response._toroidal_potential_response_solver
    calls = []

    def solve(rhs):
        calls.append(rhs[1].shape[-1])
        return original(rhs)

    monkeypatch.setattr(response, "_toroidal_potential_response_solver", solve)
    feedback = response.induced_poloidal_potential_feedback_operator
    assert calls == [geometry.poloidal_basis.coefficient_count]
    E, jr = response.solve_induced_response(Br)
    np.testing.assert_allclose(E, expected_E, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(jr, expected_jr, rtol=1e-7, atol=1e-18)
    batched_E, batched_jr = response.solve_induced_response(xp.stack([Br, 2 * Br], axis=-1))
    np.testing.assert_allclose(batched_E[..., 1], 2 * E, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(batched_jr[..., 1], 2 * jr, rtol=1e-12, atol=1e-20)
    actual_W = geometry.surface_to_poloidal_operator(
        geometry.helmholtz_divergence_free_potential_operator(E)
    )
    potential = geometry.induced_Br_to_poloidal_potential_operator(Br)
    np.testing.assert_allclose(feedback(potential), actual_W, rtol=1e-11, atol=1e-12)
    assert calls == [geometry.poloidal_basis.coefficient_count]


@pytest.mark.parametrize("solver_name", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("reg_lambda", [0.0, 0.03])
def test_toroidal_potential_solvers_match_the_direct_physical_solution(
    tmp_path, backend, solver_name, reg_lambda
):
    """Every solver recovers the same constrained boundary current."""
    simulation = Simulation(
        simulation_directory=tmp_path / solver_name,
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="radial",
        horizontal_basis_kind="CS",
        enable_pfac_coupling=False,
        least_squares_solver=solver_name,
        least_squares_preconditioner="jacobi" if solver_name in {"lsmr", "cgls"} else None,
        toroidal_potential_regularization_lambda=reg_lambda,
        artifact_storage="netcdf",
    )
    response = simulation.response
    geometry = simulation.geometry
    boundary_jr = geometry.horizontal_basis.project_scalar_mean_free(
        np.linspace(-1.0, 1.0, geometry.horizontal_basis.coefficient_count)
    )

    problem = response._toroidal_potential_problem
    assert problem.solution_shape == (geometry.horizontal_basis.coefficient_count,)
    rhs_entries = [None] * len(problem.data_operators)
    rhs_entries[0] = geometry.radial_current_constraint_operator.matvec(boundary_jr)
    rhs, _, _ = problem.assemble_rhs_block(rhs_entries)
    system = problem.system_operator.to_matrix(backend="numpy")
    from scipy.linalg import null_space

    Z = null_space(problem.constraints)
    expected_potential = Z @ np.linalg.lstsq(system @ Z, np.asarray(rhs), rcond=None)[0]

    potential = response._toroidal_potential_response_solver(rhs_entries)
    expected_boundary_jr = geometry.toroidal_potential_to_boundary_jr_operator.matvec(
        expected_potential
    )
    solved_boundary_jr = geometry.toroidal_potential_to_boundary_jr_operator.matvec(potential)

    assert response._toroidal_potential_solver.method == solver_name
    np.testing.assert_allclose(
        np.asarray(potential).reshape(-1), expected_potential.reshape(-1), rtol=1e-8, atol=1e-11
    )
    np.testing.assert_allclose(geometry.horizontal_basis.scalar_mean(potential), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(solved_boundary_jr), np.asarray(expected_boundary_jr), rtol=1e-8, atol=1e-11
    )


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
def test_response_prepares_one_canonical_operator(
    tmp_path, monkeypatch, backend, horizontal_basis_kind
):
    """Compile compact maps once; keep CS maps structured."""
    simulation = Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        horizontal_basis_kind=horizontal_basis_kind,
        artifact_storage="netcdf",
    )
    space = simulation.results.schema.input_field_spaces["conductance"]
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(space.shape),
            "log_hall_to_pedersen_ratio": np.zeros(space.shape),
        },
        time=0.0,
    )
    response = simulation.response_at_time(0.0)
    materialized = []
    original_to_matrix = LinearMap.to_matrix

    def to_matrix(operator, **kwargs):
        materialized.append(operator)
        return original_to_matrix(operator, **kwargs)

    monkeypatch.setattr(LinearMap, "to_matrix", to_matrix)
    induced = response.induced_Br_to_E_coeffs_operator
    toroidal = response.toroidal_potential_to_E_coeffs_operator
    xp = get_array_module()
    induced(xp.ones(induced.input_shape))
    toroidal(xp.ones(toroidal.input_shape))

    assert response.induced_Br_to_E_coeffs_operator is induced
    assert response.toroidal_potential_to_E_coeffs_operator is toroidal
    assert sum(operator is induced for operator in materialized) == 1
    assert sum(operator is toroidal for operator in materialized) == (
        horizontal_basis_kind == "SH"
    )
    assert bool(toroidal._dense_cache) == (horizontal_basis_kind == "SH")


def test_response_caches_absent_optional_operators_and_preconditioners():
    """None is a reusable result, not a second readiness flag."""
    response = object.__new__(ElectrodynamicResponse)
    calls = []

    def no_boundary_field():
        calls.append("boundary")
        return None

    def no_preconditioner(*, problem):
        calls.append("preconditioner")
        return None

    response.geometry = SimpleNamespace(boundary_Br_to_gridded_JS_operator=no_boundary_field)
    response._toroidal_potential_solver = SimpleNamespace(
        method="lsmr", build_preconditioner=no_preconditioner
    )
    response._toroidal_potential_problem = object()
    for _ in range(2):
        assert response.boundary_Br_to_E_coeffs_operator is None
        assert response._toroidal_potential_preconditioner is None
    assert calls == ["boundary", "preconditioner"]


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_u_coeffs_to_E_coeffs_is_linear_map_on_jax():
    """Wind-to-E is exposed as a shaped LinearMap."""
    import jax.numpy as jnp

    previous_backend = get_backend()
    n = 3
    helmholtz_analysis = np.arange(2 * n * 2 * 4, dtype=float).reshape(2, n, 2, 4) / 10.0
    bu = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4) / 20.0
    helmholtz_synthesis = np.arange(2 * 4 * 2 * n, dtype=float).reshape(2, 4, 2, n) / 30.0
    coeffs = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0

    u_to_uxB_grid = np.einsum("pqg,qgrs->pgrs", bu, helmholtz_synthesis, optimize=True)
    expected = np.tensordot(helmholtz_analysis, u_to_uxB_grid, axes=([2, 3], [0, 1]))
    expected = np.tensordot(expected, coeffs, axes=([2, 3], [0, 1]))

    geometry = object.__new__(SimulationGeometry)
    geometry.__dict__.update(
        horizontal_basis=SimpleNamespace(
            coefficient_count=n, project_scalar_mean_free=lambda coeffs: coeffs
        ),
        wind_motional_E_tensor=jnp.asarray(bu),
        horizontal_transform=SimpleNamespace(
            helmholtz_analysis_operator=as_linear_map(
                jnp.asarray(helmholtz_analysis), input_shape=(2, 4), output_shape=(2, n)
            ),
            helmholtz_synthesis_operator=as_linear_map(
                jnp.asarray(helmholtz_synthesis), input_shape=(2, n), output_shape=(2, 4)
            ),
        ),
    )

    try:
        set_backend("jax")
        operator = geometry.u_coeffs_to_E_coeffs_operator
        result = operator(jnp.asarray(coeffs))
    finally:
        set_backend(previous_backend)

    assert isinstance(operator, LinearMap)
    assert operator.output_shape == (2, n)
    assert operator.input_shape == (2, n)
    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), expected)


def test_Q_eff_coeffs_to_E_coeffs_uses_resistance_tensor_operator():
    """Q_eff maps through the resistance tensor before E analysis."""
    n = 3
    n_grid = 4
    helmholtz_analysis = np.arange(2 * n * 2 * n_grid, dtype=float).reshape(2, n, 2, n_grid) / 10.0
    M_total = np.arange(2 * 2 * n_grid, dtype=float).reshape(2, 2, n_grid) / 20.0
    M_total += np.array([[[2.0], [0.0]], [[0.0], [3.0]]])
    synthesis = np.arange(2 * n_grid * 2 * n, dtype=float).reshape(2, n_grid, 2, n) / 30.0
    coeffs = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0

    q_on_grid = np.einsum("qgrs,rs->qg", synthesis, coeffs, optimize=True)
    E_on_grid = np.einsum("pqg,qg->pg", M_total, q_on_grid, optimize=True)
    expected = np.einsum("cmpg,pg->cm", helmholtz_analysis, E_on_grid, optimize=True)

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(coefficient_count=n),
        poloidal_basis=None,
        model_grid=SimpleNamespace(size=n_grid),
        helmholtz_analysis_operator=as_linear_map(
            helmholtz_analysis, input_shape=(2, n_grid), output_shape=(2, n)
        ),
        horizontal_transform=SimpleNamespace(
            helmholtz_synthesis_operator=as_linear_map(
                synthesis, input_shape=(2, n), output_shape=(2, n_grid)
            )
        ),
    )
    response.resistance_tensor_on_grid = M_total

    operator = response.Q_eff_to_E_coeffs_operator
    result = operator(coeffs)

    assert isinstance(operator, LinearMap)
    assert operator.output_shape == (2, n)
    assert operator.input_shape == (2, n)
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_induction_matrix_assembly_stays_on_jax():
    """Dense induction assembly should not bounce through NumPy."""
    import jax.numpy as jnp

    previous_backend = get_backend()
    n = 3
    divergence_free_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 10.0
    driving_E_matrix = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 20.0
    driving_E_to_toroidal_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 30.0
    E_imp_matrix = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 40.0

    expected = np.tensordot(divergence_free_potential, driving_E_matrix, axes=([1, 2], [0, 1]))
    toroidal_potential_matrix = np.tensordot(
        driving_E_to_toroidal_potential, driving_E_matrix, axes=([1, 2], [0, 1])
    )
    E_imp_to_df = np.tensordot(divergence_free_potential, E_imp_matrix, axes=([1, 2], [0, 1]))
    expected = expected + E_imp_to_df @ toroidal_potential_matrix

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(coefficient_count=n),
        surface_to_poloidal_operator=as_linear_map(jnp.eye(n)),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            jnp.asarray(divergence_free_potential), input_shape=(2, n), output_shape=(n,)
        ),
    )
    response.induced_poloidal_potential_to_E_coeffs_operator = einsum_linear_map(
        component_tensors=[jnp.asarray(driving_E_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response.toroidal_potential_to_E_coeffs_operator = einsum_linear_map(
        component_tensors=[jnp.asarray(E_imp_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response.driving_E_to_toroidal_potential_operator = as_linear_map(
        jnp.asarray(driving_E_to_toroidal_potential), input_shape=(2, n), output_shape=(n,)
    )
    response.induced_poloidal_potential_to_W_operator = as_linear_map(jnp.asarray(expected))
    response.config = SimpleNamespace(enable_interhemispheric_coupling=True)
    response._interhemispheric_electric_field_constraint = _dummy_constraint_map()

    try:
        set_backend("jax")
        feedback_matrix = response.induced_poloidal_potential_feedback_operator.to_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(feedback_matrix).__module__
    np.testing.assert_allclose(np.asarray(feedback_matrix), expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_equilibrium_operator_preserves_jax_matrix():
    """Equilibrium map should use LinearMap without forcing NumPy."""
    import jax.numpy as jnp

    previous_backend = get_backend()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    coeffs = np.array([7.0, 11.0])

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        poloidal_basis=SimpleNamespace(coefficient_count=2),
        surface_to_poloidal_operator=as_linear_map(jnp.eye(2)),
    )
    response.induced_poloidal_potential_feedback_operator = as_linear_map(-jnp.linalg.inv(matrix))

    try:
        set_backend("jax")
        result = response.noninductive_W_to_equilibrium_induced_poloidal_potential_operator.matvec(
            jnp.asarray(coeffs)
        )
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ coeffs)


def test_equilibrium_operator_keeps_cross_space_bridge_structured():
    """Ordinary equilibrium application avoids a dense surface map."""
    surface_matrix = np.arange(10, dtype=float).reshape(2, 5) / 10.0
    feedback_matrix = np.array([[2.0, 0.25], [-0.5, 1.5]])

    def apply_surface(values):
        xp = get_array_module(values)
        return xp.asarray(surface_matrix) @ xp.asarray(values)

    def apply_surface_adjoint(values):
        xp = get_array_module(values)
        return xp.asarray(surface_matrix.T) @ xp.asarray(values)

    surface_operator = LinearMap(
        shape=surface_matrix.shape,
        dtype=surface_matrix.dtype,
        matvec=apply_surface,
        rmatvec=apply_surface_adjoint,
        matmat=apply_surface,
        rmatmat=apply_surface_adjoint,
        input_shape=(5,),
        output_shape=(2,),
    )
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        poloidal_basis=SimpleNamespace(coefficient_count=2),
        surface_to_poloidal_operator=surface_operator,
    )
    response.induced_poloidal_potential_feedback_operator = as_linear_map(feedback_matrix)

    probe = np.linspace(-1.0, 1.0, 5)
    operator = response.noninductive_W_to_equilibrium_induced_poloidal_potential_operator
    actual = operator.matvec(probe)
    expected = -np.linalg.pinv(feedback_matrix, rtol=1e-15) @ surface_matrix @ probe

    assert np not in surface_operator._dense_cache
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    explicit = (
        response.noninductive_W_to_equilibrium_induced_poloidal_potential_operator.to_matrix()
    )
    expected_matrix = -np.linalg.pinv(feedback_matrix, rtol=1e-15) @ surface_matrix
    np.testing.assert_allclose(explicit, expected_matrix, rtol=1e-13, atol=1e-13)


def test_toroidal_potential_runtime_solve_uses_one_physical_rhs():
    """Runtime toroidal-potential application solves one RHS."""
    n = 3
    radial_current_constraint = np.arange(n * n, dtype=float).reshape(n, n) / 10.0
    electric_field_difference = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 20.0
    jr_coeffs = np.arange(n, dtype=float) / 30.0
    driving_E = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0
    weight = 0.25

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(
            coefficient_count=n, project_scalar_mean_free=lambda values: values
        ),
        radial_current_constraint_operator=as_linear_map(radial_current_constraint),
        interhemispheric_electric_field_difference_operator=as_linear_map(
            electric_field_difference, input_shape=(2, n), output_shape=(n,)
        ),
    )
    response._toroidal_potential_problem = SimpleNamespace(data_operators=[None, None])
    response._interhemispheric_electric_field_constraint = _dummy_constraint_map()
    response.config = SimpleNamespace(
        enable_interhemispheric_coupling=True, interhemispheric_electric_field_weight=weight
    )

    captured_rhs = None

    def solve_response(rhs_entries):
        nonlocal captured_rhs
        captured_rhs = rhs_entries
        return rhs_entries[0] + rhs_entries[1]

    response._toroidal_potential_response_solver = solve_response

    expected_jr_rhs = radial_current_constraint @ jr_coeffs
    expected_E_rhs = (
        -weight * electric_field_difference.reshape(n, 2 * n) @ driving_E.reshape(2 * n)
    )

    np.testing.assert_allclose(
        response._solve_toroidal_potential(jr_coeffs, driving_E), expected_jr_rhs + expected_E_rhs
    )
    np.testing.assert_allclose(captured_rhs[0], expected_jr_rhs)
    np.testing.assert_allclose(captured_rhs[1], expected_E_rhs)


def test_induced_poloidal_potential_E_response_solves_only_poloidal_source_columns():
    """Interhemispheric feedback omits unrelated E columns."""
    n_surface = 4
    n_poloidal = 2
    n_constraint = 3
    source = (
        np.arange(2 * n_surface * n_poloidal, dtype=float).reshape(2 * n_surface, n_poloidal)
        / 10.0
    )
    difference = (
        np.arange(n_constraint * 2 * n_surface, dtype=float).reshape(n_constraint, 2 * n_surface)
        / 20.0
    )
    toroidal_potential_to_E = (
        np.arange(2 * n_surface * n_surface, dtype=float).reshape(2 * n_surface, n_surface) / 30.0
    )
    divergence_free = (
        np.arange(n_surface * 2 * n_surface, dtype=float).reshape(n_surface, 2 * n_surface) / 40.0
    )
    solved_toroidal_potential = (
        np.arange(n_surface * n_poloidal, dtype=float).reshape(n_surface, n_poloidal) / 50.0
    )
    weight = 0.25

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(coefficient_count=n_surface),
        interhemispheric_electric_field_difference_operator=as_linear_map(
            difference, input_shape=(2, n_surface), output_shape=(n_constraint,)
        ),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            divergence_free, input_shape=(2, n_surface), output_shape=(n_surface,)
        ),
        toroidal_potential_to_boundary_jr_operator=as_linear_map(np.eye(n_surface)),
    )
    response._interhemispheric_electric_field_constraint = _dummy_constraint_map()
    response._toroidal_potential_problem = SimpleNamespace(
        data_operators=[SimpleNamespace(), SimpleNamespace(output_shape=(n_constraint,))]
    )
    response.toroidal_potential_to_E_coeffs_operator = as_linear_map(
        toroidal_potential_to_E, input_shape=(n_surface,), output_shape=(2, n_surface)
    )
    response.config = SimpleNamespace(
        enable_interhemispheric_coupling=True, interhemispheric_electric_field_weight=weight
    )

    captured_rhs = None

    def solve_response(rhs_entries):
        nonlocal captured_rhs
        captured_rhs = rhs_entries
        return solved_toroidal_potential

    response._toroidal_potential_response_solver = solve_response
    response.induced_poloidal_potential_to_E_coeffs_operator = as_linear_map(
        source, input_shape=(n_poloidal,), output_shape=(2, n_surface)
    )
    operator = response.induced_poloidal_potential_to_W_operator

    expected_rhs = -weight * difference @ source
    expected = divergence_free @ (source + toroidal_potential_to_E @ solved_toroidal_potential)
    np.testing.assert_allclose(captured_rhs[1], expected_rhs)
    assert captured_rhs[1].shape == (n_constraint, n_poloidal)
    np.testing.assert_allclose(operator.to_matrix(backend="numpy"), expected)


def test_toroidal_potential_problem_uses_radial_current_constraint_operator_directly():
    """The radial-current constraint should retain its LinearMap."""
    n = 3
    radial_current_constraint = np.arange(n * n, dtype=float).reshape(n, n) / 10.0
    toroidal_potential_to_boundary_jr = np.diag(np.array([2.0, 3.0, 5.0]))

    class GeometryStub:
        horizontal_basis = SimpleNamespace(coefficient_count=n, omits_constant_mode=lambda: True)
        radial_current_constraint_operator = as_linear_map(
            radial_current_constraint, input_shape=(n,), output_shape=(n,)
        )
        toroidal_potential_to_boundary_jr_operator = as_linear_map(
            toroidal_potential_to_boundary_jr
        )

        @property
        def radial_current_constraint_matrix(self):
            raise AssertionError("toroidal_potential problem should use the LinearMap operator")

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = GeometryStub()
    response._interhemispheric_electric_field_constraint = None
    response.config = SimpleNamespace(
        enable_interhemispheric_coupling=False, toroidal_potential_regularization_lambda=0.0
    )

    problem = response._toroidal_potential_problem

    np.testing.assert_allclose(
        problem.system_operator.to_matrix(backend="numpy"),
        radial_current_constraint @ toroidal_potential_to_boundary_jr,
    )


def test_interhemispheric_constraint_uses_geometry_operator_without_dense_property():
    """The IH E constraint should compose LinearMaps directly."""
    n = 2
    n_ll = 3
    E_outer = np.arange(2 * n_ll * 2 * n, dtype=float).reshape(2, n_ll, 2, n) / 10.0
    toroidal_potential_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 20.0

    class GeometryStub:
        horizontal_basis = SimpleNamespace(coefficient_count=n)
        interhemispheric_electric_field_difference_operator = as_linear_map(
            E_outer, input_shape=(2, n), output_shape=(2, n_ll)
        )

        @property
        def interhemispheric_electric_field_difference_array(self):
            raise AssertionError("constraint should use the LinearMap operator")

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = GeometryStub()
    response.toroidal_potential_to_E_coeffs_operator = as_linear_map(
        toroidal_potential_to_E, input_shape=(n,), output_shape=(2, n)
    )

    constraint = response._interhemispheric_electric_field_constraint

    expected = E_outer.reshape(2 * n_ll, 2 * n) @ toroidal_potential_to_E.reshape(2 * n, n)
    np.testing.assert_allclose(constraint.to_matrix(backend="numpy"), expected)


def test_resistance_tensor_uses_conductance_synthesis_operator_without_matrix():
    """Avoid grid-evaluation matrices during log synthesis."""
    n_grid = 4
    n_coeffs = 3
    synthesis = np.arange(n_grid * n_coeffs, dtype=float).reshape(n_grid, n_coeffs) / 10.0
    log_magnitude = np.array([0.1, 0.2, 0.3])
    log_ratio = np.array([-0.4, 0.5, 0.6])
    bP = np.arange(2 * 2 * n_grid, dtype=float).reshape(2, 2, n_grid) / 20.0
    bH = np.arange(2 * 2 * n_grid, dtype=float).reshape(2, 2, n_grid) / 30.0
    model_grid = object()

    class ConductanceBasis:
        def coefficients_are_compatible_with(self, _basis):
            return False

        def scalar_evaluation_operator(self, grid):
            assert grid is model_grid
            return as_linear_map(synthesis, input_shape=(n_coeffs,), output_shape=(n_grid,))

        def scalar_evaluation_array(self, _grid):
            raise AssertionError("conductance synthesis should use the operator API")

    conductance_basis = ConductanceBasis()
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        model_grid=model_grid, pedersen_geometry_tensor=bP, hall_geometry_tensor=bH
    )
    field_space = SimpleNamespace(basis=conductance_basis)
    response._conductance_space = field_space
    response.log_conductance_magnitude = log_magnitude
    response.log_hall_to_pedersen_ratio = log_ratio

    log_coordinates_on_grid = synthesis @ np.stack([log_magnitude, log_ratio], axis=1)
    resistance_on_grid = np.stack(
        resistance_from_log_conductance_coordinates(
            log_coordinates_on_grid[:, 0], log_coordinates_on_grid[:, 1]
        ),
        axis=1,
    )
    expected = np.einsum(
        "sijk,sk->ijk", np.stack([bP, bH], axis=0), resistance_on_grid.T, optimize=True
    )

    np.testing.assert_allclose(response.resistance_tensor_on_grid, expected)


def test_model_operator_accessors_match_runtime_operator_chain():
    """Dense accessors should expose the same W/rate operators."""
    n = 3
    divergence_free_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 10.0
    u_to_E = np.arange(2 * n * 2 * n, dtype=float).reshape(2, n, 2, n) / 20.0
    toroidal_potential_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 30.0
    driving_E_to_toroidal_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 40.0
    boundary_jr_to_toroidal_potential = np.arange(n * n, dtype=float).reshape(n, n) / 50.0
    induced_poloidal_potential_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 60.0
    scale = 2.5

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(coefficient_count=n),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            divergence_free_potential, input_shape=(2, n), output_shape=(n,)
        ),
        surface_to_poloidal_operator=as_linear_map(np.eye(n)),
        induced_poloidal_potential_to_Br_operator=as_linear_map(np.eye(n)),
        induced_poloidal_potential_faraday_rate_scale=scale,
    )
    response.geometry.u_coeffs_to_E_coeffs_operator = einsum_linear_map(
        component_tensors=[u_to_E],
        einsum_string_dense="cmrs->cmrs",
        einsum_string_matvec="cmrs,rs->cm",
        einsum_string_rmatvec="cm,cmrs->rs",
        output_shape=(2, n),
        input_shape=(2, n),
    )
    response.toroidal_potential_to_E_coeffs_operator = einsum_linear_map(
        component_tensors=[toroidal_potential_to_E],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response.boundary_Br_to_E_coeffs_operator = None
    response.boundary_jr_to_toroidal_potential_operator = as_linear_map(
        boundary_jr_to_toroidal_potential, input_shape=(n,), output_shape=(n,)
    )
    response.driving_E_to_toroidal_potential_operator = as_linear_map(
        driving_E_to_toroidal_potential, input_shape=(2, n), output_shape=(n,)
    )
    response.induced_poloidal_potential_to_E_coeffs_operator = as_linear_map(
        induced_poloidal_potential_to_E, input_shape=(n,), output_shape=(2, n)
    )
    response.config = SimpleNamespace(enable_interhemispheric_coupling=True)
    response._interhemispheric_electric_field_constraint = _dummy_constraint_map()

    D = divergence_free_potential.reshape(n, 2 * n)
    U = u_to_E.reshape(2 * n, 2 * n)
    toroidal_potential_to_E_matrix = toroidal_potential_to_E.reshape(2 * n, n)
    driving_E_feedback = driving_E_to_toroidal_potential.reshape(n, 2 * n)
    driving_E_to_total_E = np.eye(2 * n) + toroidal_potential_to_E_matrix @ driving_E_feedback
    induced_poloidal_potential_to_E_matrix = induced_poloidal_potential_to_E.reshape(2 * n, n)

    expected_W = {
        "u": D @ driving_E_to_total_E @ U,
        "boundary_jr": (D @ toroidal_potential_to_E_matrix @ boundary_jr_to_toroidal_potential),
        "induced_Br": (D @ driving_E_to_total_E @ induced_poloidal_potential_to_E_matrix),
    }
    expected_rates = {key: scale * value for key, value in expected_W.items()}
    response.induced_Br_to_W_operator = as_linear_map(expected_W["induced_Br"])

    response.geometry.poloidal_basis = response.geometry.horizontal_basis
    runtime_toroidal_potential_to_E = response.toroidal_potential_to_E_coeffs_operator
    assert isinstance(runtime_toroidal_potential_to_E, LinearMap)
    assert runtime_toroidal_potential_to_E is response.toroidal_potential_to_E_coeffs_operator
    np.testing.assert_allclose(
        runtime_toroidal_potential_to_E.matvec(np.arange(n, dtype=float)),
        toroidal_potential_to_E_matrix @ np.arange(n, dtype=float),
    )

    W_operators = response.source_to_W_operators(
        include_boundary_Br=False, include_Q_eff=False, include_E_neutral_wind=False
    )
    rate_operators = response.source_to_induced_Br_rate_operators(
        include_boundary_Br=False, include_Q_eff=False, include_E_neutral_wind=False
    )

    assert isinstance(response.driving_E_to_total_E_operator, LinearMap)
    assert set(W_operators) == set(expected_W)
    assert set(rate_operators) == set(expected_rates)
    for key, expected in expected_W.items():
        np.testing.assert_allclose(W_operators[key].to_matrix(), expected)
    for key, expected in expected_rates.items():
        np.testing.assert_allclose(rate_operators[key].to_matrix(), expected)

    sample = np.arange(2 * n, dtype=float)
    operators = response.source_to_W_operators(
        include_boundary_Br=False, include_Q_eff=False, include_E_neutral_wind=False
    )
    np.testing.assert_allclose(operators["u"].matvec(sample), expected_W["u"] @ sample)

    scipy_operator = operators["u"].as_linear_operator()
    np.testing.assert_allclose(scipy_operator.matvec(sample), expected_W["u"] @ sample)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_model_operators_materialize_on_explicit_jax_backend():
    """Materialization retains Kompe's explicit backend selection."""
    previous_backend = get_backend()
    n = 2
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(coefficient_count=n),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            np.arange(n * 2 * n, dtype=float).reshape(n, 2, n),
            input_shape=(2, n),
            output_shape=(n,),
        ),
        surface_to_poloidal_operator=as_linear_map(np.eye(n)),
        induced_poloidal_potential_to_Br_operator=as_linear_map(np.eye(n)),
        induced_Br_to_poloidal_potential_operator=as_linear_map(np.eye(n)),
        induced_poloidal_potential_faraday_rate_scale=1.0,
    )
    response.geometry.u_coeffs_to_E_coeffs_operator = einsum_linear_map(
        component_tensors=[np.ones((2, n, 2, n))],
        einsum_string_dense="cmrs->cmrs",
        einsum_string_matvec="cmrs,rs->cm",
        einsum_string_rmatvec="cm,cmrs->rs",
        output_shape=(2, n),
        input_shape=(2, n),
    )
    response.toroidal_potential_to_E_coeffs_operator = einsum_linear_map(
        component_tensors=[np.ones((2, n, n))],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response.boundary_Br_to_E_coeffs_operator = None
    response.boundary_jr_to_toroidal_potential_operator = as_linear_map(np.eye(n))
    response.driving_E_to_toroidal_potential_operator = None
    response.induced_poloidal_potential_to_E_coeffs_operator = as_linear_map(
        np.ones((2, n, n)), input_shape=(n,), output_shape=(2, n)
    )
    response.config = SimpleNamespace(enable_interhemispheric_coupling=False)
    response._interhemispheric_electric_field_constraint = None

    try:
        set_backend("numpy")
        operators = response.source_to_W_operators(
            include_boundary_Br=False, include_Q_eff=False, include_E_neutral_wind=False
        )
        matrices = {
            name: operator.to_matrix(backend="jax") for name, operator in operators.items()
        }
    finally:
        set_backend(previous_backend)

    assert all("jax" in type(matrix).__module__ for matrix in matrices.values())
