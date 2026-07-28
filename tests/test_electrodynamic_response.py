"""Tests for electrodynamic response construction and application."""

from types import SimpleNamespace

import numpy as np
import pytest

from pynamit.math import JAX_AVAILABLE, einsum_linear_map, get_array_module, set_backend, use_jax
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    resistance_from_log_conductance_coordinates,
)
from pynamit.simulation.response import ElectrodynamicResponse


def _dummy_constraint_map():
    return as_linear_map(np.eye(1), input_shape=(1,), output_shape=(1,))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_apply_operator_keeps_linear_map_on_jax():
    """Response application should use LinearMap directly."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    coeffs = jnp.asarray([7.0, 11.0])
    operator = einsum_linear_map(
        component_tensors=[matrix],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    try:
        set_backend("jax")
        result = ElectrodynamicResponse._apply_operator(operator, coeffs, (2,))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ np.asarray(coeffs))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_apply_operator_absence_uses_linear_map_backend_context():
    """Absent-input outputs follow the LinearMap backend context."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    operator = einsum_linear_map(
        component_tensors=[jnp.asarray(np.eye(2))],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    try:
        set_backend("numpy")
        result = ElectrodynamicResponse._apply_operator(operator, None, (2,))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), np.zeros(2))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_u_coeffs_to_E_coeffs_is_linear_map_on_jax():
    """Wind-to-E is exposed as a shaped LinearMap."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    n = 3
    helmholtz_analysis = np.arange(2 * n * 2 * 4, dtype=float).reshape(2, n, 2, 4) / 10.0
    bu = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4) / 20.0
    helmholtz_synthesis = np.arange(2 * 4 * 2 * n, dtype=float).reshape(2, 4, 2, n) / 30.0
    coeffs = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0

    u_to_uxB_grid = np.einsum("pqg,qgrs->pgrs", bu, helmholtz_synthesis, optimize=True)
    expected = np.tensordot(helmholtz_analysis, u_to_uxB_grid, axes=([2, 3], [0, 1]))
    expected = np.tensordot(expected, coeffs, axes=([2, 3], [0, 1]))

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n),
        helmholtz_analysis_operator=as_linear_map(
            jnp.asarray(helmholtz_analysis), input_shape=(2, 4), output_shape=(2, n)
        ),
        wind_motional_E_tensor=jnp.asarray(bu),
        horizontal_transform=SimpleNamespace(
            helmholtz_coeffs_to_gridded_vector_operator=as_linear_map(
                jnp.asarray(helmholtz_synthesis), input_shape=(2, n), output_shape=(2, 4)
            )
        ),
    )
    response._u_coeffs_to_E_coeffs_cache = None

    try:
        set_backend("jax")
        operator = response.u_coeffs_to_E_coeffs
        result = ElectrodynamicResponse._apply_operator(operator, jnp.asarray(coeffs), (2, n))
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

    q_representation = SimpleNamespace(
        get_helmholtz_synthesis_operator=lambda grid: as_linear_map(
            synthesis, input_shape=(2, n), output_shape=(2, n_grid)
        )
    )
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n),
        model_grid=SimpleNamespace(size=n_grid),
        helmholtz_analysis_operator=as_linear_map(
            helmholtz_analysis, input_shape=(2, n_grid), output_shape=(2, n)
        ),
    )
    response.Q_eff = SimpleNamespace(field_space=SimpleNamespace(representation=q_representation))
    response._Q_eff_synthesis_operator_cache = None
    response._Q_eff_to_E_coeffs_cache = None
    response._resistance_tensor_on_grid = M_total

    operator = response.Q_eff_to_E_coeffs
    result = ElectrodynamicResponse._apply_operator(operator, coeffs, (2, n))

    assert isinstance(operator, LinearMap)
    assert operator.output_shape == (2, n)
    assert operator.input_shape == (2, n)
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_induction_matrix_assembly_stays_on_jax():
    """Dense induction assembly should not bounce through NumPy."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    n = 3
    divergence_free_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 10.0
    driving_E_matrix = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 20.0
    driving_E_to_m_imp = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 30.0
    E_imp_matrix = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 40.0

    expected = np.tensordot(divergence_free_potential, driving_E_matrix, axes=([1, 2], [0, 1]))
    m_imp_matrix = np.tensordot(driving_E_to_m_imp, driving_E_matrix, axes=([1, 2], [0, 1]))
    E_imp_to_df = np.tensordot(divergence_free_potential, E_imp_matrix, axes=([1, 2], [0, 1]))
    expected = expected + E_imp_to_df @ m_imp_matrix

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n),
        surface_to_poloidal_operator=as_linear_map(jnp.eye(n)),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            jnp.asarray(divergence_free_potential), input_shape=(2, n), output_shape=(n,)
        ),
    )
    response._m_ind_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[jnp.asarray(driving_E_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response._m_imp_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[jnp.asarray(E_imp_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response._runtime_m_imp_to_E_coeffs_cache = None
    response._driving_E_to_m_imp_matrix = jnp.asarray(driving_E_to_m_imp)
    response._jr_to_m_imp_operator = None
    response._driving_E_to_m_imp_operator = None
    response._driving_E_to_total_E_operator = None
    response._driving_E_to_E_df_operator = None
    response._m_ind_to_E_df_operator_cache = as_linear_map(jnp.asarray(expected))
    response._m_ind_feedback_operator = None
    response._m_ind_feedback_matrix = None
    response.config = SimpleNamespace(enable_interhemispheric_coupling=True)
    response._interhemispheric_electric_field_constraint_cache = _dummy_constraint_map()

    try:
        set_backend("jax")
        response._build_m_ind_feedback_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(response._m_ind_feedback_matrix).__module__
    np.testing.assert_allclose(np.asarray(response._m_ind_feedback_matrix), expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_steady_state_operator_preserves_jax_matrix():
    """Steady-state map should use LinearMap without forcing NumPy."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    coeffs = np.array([7.0, 11.0])

    response = object.__new__(ElectrodynamicResponse)
    response._noninductive_E_df_to_steady_m_ind_matrix = jnp.asarray(matrix)
    response._noninductive_E_df_to_steady_m_ind_operator = None

    try:
        set_backend("jax")
        result = response.noninductive_E_df_to_steady_m_ind_operator.matvec(jnp.asarray(coeffs))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ coeffs)


def test_steady_state_operator_keeps_cross_space_bridge_structured():
    """Ordinary steady-state application avoids a dense surface map."""
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
        _matvec=apply_surface,
        _rmatvec=apply_surface_adjoint,
        _matmat=apply_surface,
        _rmatmat=apply_surface_adjoint,
        input_shape=(5,),
        output_shape=(2,),
    )
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        poloidal_basis=SimpleNamespace(index_length=2),
        surface_to_poloidal_operator=surface_operator,
    )
    response._m_ind_feedback_matrix = feedback_matrix
    response._noninductive_E_df_to_steady_m_ind_matrix = None
    response._noninductive_E_df_to_steady_m_ind_operator = None

    probe = np.linspace(-1.0, 1.0, 5)
    operator = response.noninductive_E_df_to_steady_m_ind_operator
    actual = operator.matvec(probe)
    expected = -np.linalg.pinv(feedback_matrix, rtol=1e-15) @ surface_matrix @ probe

    assert response._noninductive_E_df_to_steady_m_ind_matrix is None
    assert surface_operator._cached_dense(np) is None
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    explicit = response.noninductive_E_df_to_steady_m_ind_matrix
    expected_matrix = -np.linalg.pinv(feedback_matrix, rtol=1e-15) @ surface_matrix
    np.testing.assert_allclose(explicit, expected_matrix, rtol=1e-13, atol=1e-13)


def test_m_imp_runtime_solve_uses_one_physical_rhs():
    """Runtime m_imp application solves the active RHS directly."""
    n = 3
    radial_current_constraint = np.arange(n * n, dtype=float).reshape(n, n) / 10.0
    electric_field_difference = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 20.0
    jr_coeffs = np.arange(n, dtype=float) / 30.0
    driving_E = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0
    weight = 0.25

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n),
        radial_current_constraint_operator=as_linear_map(radial_current_constraint),
        interhemispheric_electric_field_difference_operator=as_linear_map(
            electric_field_difference, input_shape=(2, n), output_shape=(n,)
        ),
    )
    response._m_imp_problem_cache = SimpleNamespace(num_data_terms=2)
    response._interhemispheric_electric_field_constraint_cache = _dummy_constraint_map()
    response.project_surface_scalar_mean_free = lambda coeffs: coeffs
    response.config = SimpleNamespace(
        enable_interhemispheric_coupling=True, interhemispheric_electric_field_weight=weight
    )

    captured_rhs = None

    def solve_response(rhs_entries):
        nonlocal captured_rhs
        captured_rhs = rhs_entries
        return rhs_entries[0] + rhs_entries[1]

    response._solve_m_imp_response = solve_response

    expected_jr_rhs = radial_current_constraint @ jr_coeffs
    expected_E_rhs = (
        -weight * electric_field_difference.reshape(n, 2 * n) @ driving_E.reshape(2 * n)
    )

    np.testing.assert_allclose(
        response._solve_for_m_imp(jr_coeffs, driving_E), expected_jr_rhs + expected_E_rhs
    )
    np.testing.assert_allclose(captured_rhs[0], expected_jr_rhs)
    np.testing.assert_allclose(captured_rhs[1], expected_E_rhs)


def test_m_ind_E_response_solves_only_poloidal_source_columns():
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
    m_imp_to_E = (
        np.arange(2 * n_surface * n_surface, dtype=float).reshape(2 * n_surface, n_surface) / 30.0
    )
    divergence_free = (
        np.arange(n_surface * 2 * n_surface, dtype=float).reshape(n_surface, 2 * n_surface) / 40.0
    )
    solved_m_imp = (
        np.arange(n_surface * n_poloidal, dtype=float).reshape(n_surface, n_poloidal) / 50.0
    )
    weight = 0.25

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n_surface),
        interhemispheric_electric_field_difference_operator=as_linear_map(
            difference, input_shape=(2, n_surface), output_shape=(n_constraint,)
        ),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            divergence_free, input_shape=(2, n_surface), output_shape=(n_surface,)
        ),
    )
    response._interhemispheric_electric_field_constraint_cache = _dummy_constraint_map()
    response._m_imp_problem_cache = SimpleNamespace(
        num_data_terms=2, A=[SimpleNamespace(), SimpleNamespace(output_shape=(n_constraint,))]
    )
    response._m_imp_to_E_coeffs_cache = as_linear_map(
        m_imp_to_E, input_shape=(n_surface,), output_shape=(2, n_surface)
    )
    response.config = SimpleNamespace(
        enable_interhemispheric_coupling=True, interhemispheric_electric_field_weight=weight
    )

    captured_rhs = None

    def solve_response(rhs_entries):
        nonlocal captured_rhs
        captured_rhs = rhs_entries
        return solved_m_imp

    response._solve_m_imp_response = solve_response
    operator = response._create_driving_source_to_E_df_operator(
        as_linear_map(source, input_shape=(n_poloidal,), output_shape=(2, n_surface))
    )

    expected_rhs = -weight * difference @ source
    expected = divergence_free @ (source + m_imp_to_E @ solved_m_imp)
    np.testing.assert_allclose(captured_rhs[1], expected_rhs)
    assert captured_rhs[1].shape == (n_constraint, n_poloidal)
    np.testing.assert_allclose(operator.to_matrix(backend="numpy"), expected)


def test_m_imp_problem_uses_radial_current_constraint_operator_directly():
    """The radial-current constraint should retain its LinearMap."""
    n = 3
    radial_current_constraint = np.arange(n * n, dtype=float).reshape(n, n) / 10.0
    m_imp_to_jr = np.diag(np.array([2.0, 3.0, 5.0]))

    class GeometryStub:
        horizontal_basis = SimpleNamespace(index_length=n)
        surface_gauge_operator = None
        radial_current_constraint_operator = as_linear_map(
            radial_current_constraint, input_shape=(n,), output_shape=(n,)
        )
        m_imp_to_jr_operator = as_linear_map(m_imp_to_jr)

        @property
        def radial_current_constraint_matrix(self):
            raise AssertionError("m_imp problem should use the LinearMap operator")

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = GeometryStub()
    response._m_imp_problem_cache = None
    response._interhemispheric_electric_field_constraint_cache = None
    response.config = SimpleNamespace(
        enable_interhemispheric_coupling=False, m_imp_regularization_lambda=0.0
    )

    problem = response._m_imp_problem

    np.testing.assert_allclose(
        problem.get_system_linear_map().to_matrix(backend="numpy"),
        radial_current_constraint @ m_imp_to_jr,
    )


def test_interhemispheric_constraint_uses_geometry_operator_without_dense_property():
    """The IH E constraint should compose LinearMaps directly."""
    n = 2
    n_ll = 3
    E_outer = np.arange(2 * n_ll * 2 * n, dtype=float).reshape(2, n_ll, 2, n) / 10.0
    m_imp_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 20.0

    class GeometryStub:
        horizontal_basis = SimpleNamespace(index_length=n)
        interhemispheric_electric_field_difference_operator = as_linear_map(
            E_outer, input_shape=(2, n), output_shape=(2, n_ll)
        )

        @property
        def interhemispheric_electric_field_difference_matrix(self):
            raise AssertionError("constraint should use the LinearMap operator")

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = GeometryStub()
    response._m_imp_to_E_coeffs_cache = as_linear_map(
        m_imp_to_E, input_shape=(n,), output_shape=(2, n)
    )
    response._interhemispheric_electric_field_constraint_cache = None

    constraint = response._interhemispheric_electric_field_constraint

    expected = E_outer.reshape(2 * n_ll, 2 * n) @ m_imp_to_E.reshape(2 * n, n)
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

        def get_scalar_evaluation_operator(self, grid):
            assert grid is model_grid
            return as_linear_map(synthesis, input_shape=(n_coeffs,), output_shape=(n_grid,))

        def get_scalar_evaluation_matrix(self, _grid):
            raise AssertionError("conductance synthesis should use the operator API")

    conductance_basis = ConductanceBasis()
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        model_grid=model_grid, pedersen_geometry_tensor=bP, hall_geometry_tensor=bH
    )
    field_space = SimpleNamespace(representation=conductance_basis)
    response.log_conductance_magnitude = SimpleNamespace(
        array=log_magnitude, field_space=field_space
    )
    response.log_hall_to_pedersen_ratio = SimpleNamespace(array=log_ratio, field_space=field_space)
    response._resistance_tensor_on_grid = None

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


def test_resistance_tensor_rejects_incompatible_conductance_storage_bases():
    """Require one space for both conductance coordinates."""
    n_grid = 4
    n_coeffs = 3
    synthesis = np.ones((n_grid, n_coeffs))

    class ConductanceBasis:
        def __init__(self, name):
            self.name = name

        def coefficients_are_compatible_with(self, other):
            return self.name == getattr(other, "name", None)

        def get_scalar_evaluation_operator(self, _grid):
            return as_linear_map(synthesis, input_shape=(n_coeffs,), output_shape=(n_grid,))

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        model_grid=object(),
        pedersen_geometry_tensor=np.ones((2, 2, n_grid)),
        hall_geometry_tensor=0.0,
    )
    response.log_conductance_magnitude = SimpleNamespace(
        array=np.ones(n_coeffs),
        field_space=SimpleNamespace(representation=ConductanceBasis("magnitude")),
    )
    response.log_hall_to_pedersen_ratio = SimpleNamespace(
        array=np.ones(n_coeffs),
        field_space=SimpleNamespace(representation=ConductanceBasis("ratio")),
    )
    response._resistance_tensor_on_grid = None

    with pytest.raises(ValueError, match="coefficient-compatible"):
        _ = response.resistance_tensor_on_grid


def test_model_operator_accessors_match_runtime_operator_chain():
    """Dense accessors should expose the same E_df/rate operators."""
    n = 3
    divergence_free_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 10.0
    u_to_E = np.arange(2 * n * 2 * n, dtype=float).reshape(2, n, 2, n) / 20.0
    m_imp_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 30.0
    driving_E_to_m_imp = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 40.0
    jr_to_m_imp = np.arange(n * n, dtype=float).reshape(n, n) / 50.0
    m_ind_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 60.0
    scale = 2.5

    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            divergence_free_potential, input_shape=(2, n), output_shape=(n,)
        ),
        surface_to_poloidal_operator=as_linear_map(np.eye(n)),
        faraday_rate_scale=scale,
        Br_to_gridded_JS_operator=lambda: None,
    )
    response._u_coeffs_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[u_to_E],
        einsum_string_dense="cmrs->cmrs",
        einsum_string_matvec="cmrs,rs->cm",
        einsum_string_rmatvec="cm,cmrs->rs",
        output_shape=(2, n),
        input_shape=(2, n),
    )
    response._m_imp_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[m_imp_to_E],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response._runtime_m_imp_to_E_coeffs_cache = None
    response._Br_to_E_coeffs_cache = None
    response._jr_to_m_imp_matrix = jr_to_m_imp
    response._driving_E_to_m_imp_matrix = driving_E_to_m_imp
    response._jr_to_m_imp_operator = None
    response._driving_E_to_m_imp_operator = None
    response._m_ind_to_E_coeffs_cache = as_linear_map(
        m_ind_to_E, input_shape=(n,), output_shape=(2, n)
    )
    response._driving_E_to_total_E_operator = None
    response._driving_E_to_E_df_operator = None
    response.Q_eff = None
    response.E_neutral_wind = None
    response.config = SimpleNamespace(enable_interhemispheric_coupling=True)
    response._interhemispheric_electric_field_constraint_cache = _dummy_constraint_map()

    D = divergence_free_potential.reshape(n, 2 * n)
    U = u_to_E.reshape(2 * n, 2 * n)
    M_imp_to_E = m_imp_to_E.reshape(2 * n, n)
    driving_E_feedback = driving_E_to_m_imp.reshape(n, 2 * n)
    driving_E_to_total_E = np.eye(2 * n) + M_imp_to_E @ driving_E_feedback
    M_ind_to_E = m_ind_to_E.reshape(2 * n, n)

    expected_edf = {
        "E_df_from_u": D @ driving_E_to_total_E @ U,
        "E_df_from_jr": D @ M_imp_to_E @ jr_to_m_imp,
        "E_df_from_m_ind": D @ driving_E_to_total_E @ M_ind_to_E,
    }
    expected_rates = {
        key.replace("E_df_from_", "d_m_ind_dt_from_"): scale * value
        for key, value in expected_edf.items()
    }
    response._m_ind_to_E_df_operator_cache = as_linear_map(expected_edf["E_df_from_m_ind"])

    response.geometry.poloidal_basis = response.geometry.horizontal_basis
    runtime_m_imp_to_E = response._runtime_m_imp_to_E_coeffs
    assert isinstance(runtime_m_imp_to_E, LinearMap)
    assert runtime_m_imp_to_E is response._runtime_m_imp_to_E_coeffs
    np.testing.assert_allclose(
        runtime_m_imp_to_E.matvec(np.arange(n, dtype=float)),
        M_imp_to_E @ np.arange(n, dtype=float),
    )

    edf_matrices = response.E_df_matrices()
    rate_matrices = response.m_ind_rate_matrices()

    assert isinstance(response._driving_E_to_total_E_operator, LinearMap)
    assert set(edf_matrices) == set(expected_edf)
    assert set(rate_matrices) == set(expected_rates)
    for key, expected in expected_edf.items():
        np.testing.assert_allclose(edf_matrices[key], expected)
    for key, expected in expected_rates.items():
        np.testing.assert_allclose(rate_matrices[key], expected)

    sample = np.arange(2 * n, dtype=float)
    operators = response.E_df_operators()
    np.testing.assert_allclose(
        operators["E_df_from_u"].matvec(sample), expected_edf["E_df_from_u"] @ sample
    )

    scipy_operator = operators["E_df_from_u"].as_linear_operator()
    np.testing.assert_allclose(scipy_operator.matvec(sample), expected_edf["E_df_from_u"] @ sample)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_model_matrix_accessors_accept_explicit_jax_backend():
    """Dense model accessors should accept backend='jax'."""
    previous_backend = use_jax()
    n = 2
    response = object.__new__(ElectrodynamicResponse)
    response.geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(index_length=n),
        helmholtz_divergence_free_potential_operator=as_linear_map(
            np.arange(n * 2 * n, dtype=float).reshape(n, 2, n),
            input_shape=(2, n),
            output_shape=(n,),
        ),
        surface_to_poloidal_operator=as_linear_map(np.eye(n)),
        faraday_rate_scale=1.0,
        Br_to_gridded_JS_operator=lambda: None,
    )
    response._u_coeffs_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[np.ones((2, n, 2, n))],
        einsum_string_dense="cmrs->cmrs",
        einsum_string_matvec="cmrs,rs->cm",
        einsum_string_rmatvec="cm,cmrs->rs",
        output_shape=(2, n),
        input_shape=(2, n),
    )
    response._m_imp_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[np.ones((2, n, n))],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    response._Br_to_E_coeffs_cache = None
    response._jr_to_m_imp_matrix = np.eye(n)
    response._driving_E_to_m_imp_matrix = None
    response._jr_to_m_imp_operator = None
    response._driving_E_to_m_imp_operator = None
    response._m_ind_to_E_coeffs_cache = as_linear_map(
        np.ones((2, n, n)), input_shape=(n,), output_shape=(2, n)
    )
    response._driving_E_to_total_E_operator = None
    response._driving_E_to_E_df_operator = None
    response._m_ind_to_E_df_operator_cache = None
    response.Q_eff = None
    response.E_neutral_wind = None
    response.config = SimpleNamespace(enable_interhemispheric_coupling=False)
    response._interhemispheric_electric_field_constraint_cache = None

    try:
        set_backend("numpy")
        matrices = response.E_df_matrices(backend="jax")
    finally:
        set_backend(previous_backend)

    assert all("jax" in type(matrix).__module__ for matrix in matrices.values())
