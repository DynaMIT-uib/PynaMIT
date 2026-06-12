"""Tests for State operator application helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from pynamit.math import JAX_AVAILABLE, einsum_linear_map, set_backend, use_jax
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.simulation.state import State


def _dummy_constraint_map():
    return as_linear_map(np.eye(1), input_shape=(1,), output_shape=(1,))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_apply_operator_keeps_linear_map_on_jax():
    """State operator application should use LinearMap directly."""
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
        result = State._apply_operator(None, operator, coeffs, (2,))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ np.asarray(coeffs))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_apply_operator_zero_uses_linear_map_backend_context():
    """Zero outputs follow the LinearMap backend context."""
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
        result = State._apply_operator(None, operator, 0, (2,))
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

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = SimpleNamespace(
        helmholtz_analysis_matrix=jnp.asarray(helmholtz_analysis),
        bu=jnp.asarray(bu),
        spherical_transform=SimpleNamespace(
            helmholtz_coeffs_to_gridded_vector=jnp.asarray(helmholtz_synthesis)
        ),
    )
    state._u_coeffs_to_E_coeffs_cache = None

    try:
        set_backend("jax")
        operator = state.u_coeffs_to_E_coeffs
        result = State._apply_operator(None, operator, jnp.asarray(coeffs), (2, n))
    finally:
        set_backend(previous_backend)

    assert isinstance(operator, LinearMap)
    assert operator.output_shape == (2, n)
    assert operator.input_shape == (2, n)
    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), expected)


def test_Q_eff_coeffs_to_E_coeffs_uses_conductance_tensor_operator():
    """Q_eff maps through the conductance tensor before E analysis."""
    n = 3
    n_grid = 4
    helmholtz_analysis = np.arange(2 * n * 2 * n_grid, dtype=float).reshape(
        2, n, 2, n_grid
    ) / 10.0
    M_total = np.arange(2 * 2 * n_grid, dtype=float).reshape(2, 2, n_grid) / 20.0
    M_total += np.array([[[2.0], [0.0]], [[0.0], [3.0]]])
    synthesis = np.arange(2 * n_grid * 2 * n, dtype=float).reshape(
        2, n_grid, 2, n
    ) / 30.0
    coeffs = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0

    q_on_grid = np.einsum("qgrs,rs->qg", synthesis, coeffs, optimize=True)
    E_on_grid = np.einsum("pqg,qg->pg", M_total, q_on_grid, optimize=True)
    expected = np.einsum("cmpg,pg->cm", helmholtz_analysis, E_on_grid, optimize=True)

    q_representation = SimpleNamespace(
        get_helmholtz_synthesis_operator=lambda grid: as_linear_map(
            synthesis,
            input_shape=(2, n),
            output_shape=(2, n_grid),
        )
    )
    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = SimpleNamespace(
        grid=SimpleNamespace(size=n_grid),
        helmholtz_analysis_matrix=helmholtz_analysis,
    )
    state.Q_eff = SimpleNamespace(representation=q_representation)
    state._Q_eff_synthesis_operator_cache = {}
    state._Q_eff_to_E_coeffs_cache = None
    state._M_total_on_grid = M_total

    operator = state.Q_eff_to_E_coeffs
    result = State._apply_operator(None, operator, coeffs, (2, n))

    assert isinstance(operator, LinearMap)
    assert operator.output_shape == (2, n)
    assert operator.input_shape == (2, n)
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_induction_matrix_assembly_stays_on_jax(monkeypatch):
    """Dense induction assembly should not bounce through NumPy."""
    import jax.numpy as jnp
    import pynamit.simulation.state as state_module

    previous_backend = use_jax()
    n = 3
    divergence_free_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 10.0
    E_direct_matrix = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 20.0
    E_direct_to_m_imp = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 30.0
    E_imp_matrix = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 40.0

    expected = np.tensordot(
        divergence_free_potential,
        E_direct_matrix,
        axes=([1, 2], [0, 1]),
    )
    m_imp_matrix = np.tensordot(
        E_direct_to_m_imp,
        E_direct_matrix,
        axes=([1, 2], [0, 1]),
    )
    E_imp_to_df = np.tensordot(
        divergence_free_potential,
        E_imp_matrix,
        axes=([1, 2], [0, 1]),
    )
    expected = expected + E_imp_to_df @ m_imp_matrix

    def fail_to_numpy(_):
        raise AssertionError("Induction matrix assembly should stay on the active backend")

    monkeypatch.setattr(state_module, "to_numpy", fail_to_numpy)

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = SimpleNamespace(
        helmholtz_divergence_free_potential_operator=as_linear_map(
            jnp.asarray(divergence_free_potential),
            input_shape=(2, n),
            output_shape=(n,),
        )
    )
    state._m_ind_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[jnp.asarray(E_direct_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    state._m_imp_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[jnp.asarray(E_imp_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    state._E_direct_to_m_imp_matrix = jnp.asarray(E_direct_to_m_imp)
    state._m_ind_to_E_df_operator = None
    state._m_ind_to_E_df_matrix = None
    state.connect_hemispheres = True
    state._E_map_constraint_cache = _dummy_constraint_map()
    state._ensure_m_imp_response_matrices = lambda: None

    try:
        set_backend("jax")
        state._build_m_ind_to_E_df_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(state._m_ind_to_E_df_matrix).__module__
    np.testing.assert_allclose(np.asarray(state._m_ind_to_E_df_matrix), expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_steady_state_operator_preserves_jax_matrix():
    """Steady-state map should use LinearMap without forcing NumPy."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    coeffs = np.array([7.0, 11.0])

    state = object.__new__(State)
    state._E_noind_to_m_ind_steady_matrix = jnp.asarray(matrix)
    state._E_noind_to_m_ind_steady_operator = None

    try:
        set_backend("jax")
        result = state.E_noind_to_m_ind_steady_operator.matvec(jnp.asarray(coeffs))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ coeffs)


def test_m_imp_solve_uses_shaped_response_operators():
    """m_imp response application should go through StateOperators."""
    n = 3
    jr_to_m_imp = np.arange(n * n, dtype=float).reshape(n, n) / 10.0
    E_direct_to_m_imp = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 20.0
    jr_coeffs = np.arange(n, dtype=float) / 30.0
    E_direct_coeffs = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state._jr_to_m_imp_matrix = jr_to_m_imp
    state._E_direct_to_m_imp_matrix = E_direct_to_m_imp
    state._ensure_m_imp_response_matrices = lambda: None
    state.project_scalar_mean_free = lambda coeffs: coeffs
    state.connect_hemispheres = True

    expected = jr_to_m_imp @ jr_coeffs
    expected += E_direct_to_m_imp.reshape(n, 2 * n) @ E_direct_coeffs.reshape(2 * n)

    np.testing.assert_allclose(
        state._solve_for_m_imp(jr_coeffs, E_direct_coeffs),
        expected,
    )


def test_m_imp_problem_uses_jr_apex_operator_without_dense_property():
    """The jr constraint should use Geometry's LinearMap directly."""
    n = 3
    jr_to_apex = np.arange(n * n, dtype=float).reshape(n, n) / 10.0
    m_imp_to_jr = np.diag(np.array([2.0, 3.0, 5.0]))

    class GeometryStub:
        jr_coeffs_to_j_apex_operator = as_linear_map(
            jr_to_apex,
            input_shape=(n,),
            output_shape=(n,),
        )
        m_imp_to_jr_operator = as_linear_map(m_imp_to_jr)

        @property
        def jr_coeffs_to_j_apex(self):
            raise AssertionError("m_imp_problem should use the LinearMap operator")

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = GeometryStub()
    state._m_imp_problem = None
    state._E_map_constraint_cache = None
    state.connect_hemispheres = False
    state.m_imp_regularization_lambda = 0.0

    problem = state.m_imp_problem

    np.testing.assert_allclose(
        problem.get_system_linear_map().to_matrix(backend="numpy"),
        jr_to_apex @ m_imp_to_jr,
    )


def test_E_map_constraint_uses_geometry_operator_without_dense_property():
    """The IH E constraint should compose LinearMaps directly."""
    n = 2
    n_ll = 3
    E_outer = np.arange(2 * n_ll * 2 * n, dtype=float).reshape(
        2, n_ll, 2, n
    ) / 10.0
    m_imp_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 20.0

    class GeometryStub:
        E_coeffs_to_E_apex_ll_diff_operator = as_linear_map(
            E_outer,
            input_shape=(2, n),
            output_shape=(2, n_ll),
        )

        @property
        def E_coeffs_to_E_apex_ll_diff(self):
            raise AssertionError("_E_map_constraint should use the LinearMap operator")

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = GeometryStub()
    state._m_imp_to_E_coeffs_cache = as_linear_map(
        m_imp_to_E,
        input_shape=(n,),
        output_shape=(2, n),
    )
    state._E_map_constraint_cache = None

    constraint = state._E_map_constraint

    expected = E_outer.reshape(2 * n_ll, 2 * n) @ m_imp_to_E.reshape(2 * n, n)
    np.testing.assert_allclose(
        constraint.to_matrix(backend="numpy"),
        expected,
    )


def test_M_total_on_grid_uses_conductance_synthesis_operator_without_matrix():
    """Conductance synthesis should avoid grid-evaluation matrices."""
    n_grid = 4
    n_coeffs = 3
    synthesis = np.arange(n_grid * n_coeffs, dtype=float).reshape(n_grid, n_coeffs) / 10.0
    etaP = np.array([1.0, 2.0, 3.0])
    etaH = np.array([4.0, 5.0, 6.0])
    bP = np.arange(2 * 2 * n_grid, dtype=float).reshape(2, 2, n_grid) / 20.0
    bH = np.arange(2 * 2 * n_grid, dtype=float).reshape(2, 2, n_grid) / 30.0
    model_grid = object()

    class ConductanceBasis:
        def coefficients_are_compatible_with(self, _basis):
            return False

        def get_scalar_evaluation_operator(self, grid):
            assert grid is model_grid
            return as_linear_map(
                synthesis,
                input_shape=(n_coeffs,),
                output_shape=(n_grid,),
            )

        def get_scalar_evaluation_matrix(self, _grid):
            raise AssertionError("M_total_on_grid should use the operator API")

    conductance_basis = ConductanceBasis()
    state = object.__new__(State)
    state.basis = object()
    state.geometry = SimpleNamespace(
        grid=model_grid,
        spherical_transform_zero_added=SimpleNamespace(source=object()),
        bP=bP,
        bH=bH,
    )
    state.etaP = SimpleNamespace(coeffs=etaP, representation=conductance_basis)
    state.etaH = SimpleNamespace(coeffs=etaH, representation=conductance_basis)
    state._M_total_on_grid = None

    conductance_on_grid = synthesis @ np.stack([etaP, etaH], axis=1)
    expected = np.einsum(
        "sijk,sk->ijk",
        np.stack([bP, bH], axis=0),
        conductance_on_grid.T,
        optimize=True,
    )

    np.testing.assert_allclose(state.M_total_on_grid, expected)


def test_M_total_on_grid_rejects_incompatible_conductance_bases():
    """Pedersen and Hall must share one coefficient space."""
    n_grid = 4
    n_coeffs = 3
    synthesis = np.ones((n_grid, n_coeffs))

    class ConductanceBasis:
        def __init__(self, name):
            self.name = name

        def coefficients_are_compatible_with(self, other):
            return self.name == getattr(other, "name", None)

        def get_scalar_evaluation_operator(self, _grid):
            return as_linear_map(
                synthesis,
                input_shape=(n_coeffs,),
                output_shape=(n_grid,),
            )

    state = object.__new__(State)
    state.basis = object()
    state.geometry = SimpleNamespace(grid=object(), bP=np.ones((2, 2, n_grid)), bH=0.0)
    state.etaP = SimpleNamespace(
        coeffs=np.ones(n_coeffs),
        representation=ConductanceBasis("pedersen"),
    )
    state.etaH = SimpleNamespace(
        coeffs=np.ones(n_coeffs),
        representation=ConductanceBasis("hall"),
    )
    state._M_total_on_grid = None

    with pytest.raises(ValueError, match="coefficient-compatible"):
        _ = state.M_total_on_grid


def test_model_operator_accessors_match_runtime_operator_chain():
    """Dense accessors should expose the same E_df/rate operators."""
    n = 3
    divergence_free_potential = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 10.0
    u_to_E = np.arange(2 * n * 2 * n, dtype=float).reshape(2, n, 2, n) / 20.0
    m_imp_to_E = np.arange(2 * n * n, dtype=float).reshape(2, n, n) / 30.0
    E_direct_to_m_imp = np.arange(n * 2 * n, dtype=float).reshape(n, 2, n) / 40.0
    jr_to_m_imp = np.arange(n * n, dtype=float).reshape(n, n) / 50.0
    m_ind_to_E_df = np.arange(n * n, dtype=float).reshape(n, n) / 60.0
    scale = 2.5

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = SimpleNamespace(
        helmholtz_divergence_free_potential_operator=as_linear_map(
            divergence_free_potential,
            input_shape=(2, n),
            output_shape=(n,),
        ),
        E_df_to_d_m_ind_dt=scale,
        Br_to_gridded_JS=None,
    )
    state._u_coeffs_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[u_to_E],
        einsum_string_dense="cmrs->cmrs",
        einsum_string_matvec="cmrs,rs->cm",
        einsum_string_rmatvec="cm,cmrs->rs",
        output_shape=(2, n),
        input_shape=(2, n),
    )
    state._m_imp_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[m_imp_to_E],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    state._Br_to_E_coeffs_cache = None
    state._jr_to_m_imp_matrix = jr_to_m_imp
    state._E_direct_to_m_imp_matrix = E_direct_to_m_imp
    state._m_ind_to_E_df_operator = as_linear_map(m_ind_to_E_df)
    state._direct_E_coeffs_to_total_E_coeffs_operator = None
    state._direct_E_coeffs_to_E_df_operator = None
    state.connect_hemispheres = True
    state._E_map_constraint_cache = _dummy_constraint_map()
    state._ensure_m_imp_response_matrices = lambda: None

    D = divergence_free_potential.reshape(n, 2 * n)
    U = u_to_E.reshape(2 * n, 2 * n)
    M_imp_to_E = m_imp_to_E.reshape(2 * n, n)
    E_direct_feedback = E_direct_to_m_imp.reshape(n, 2 * n)
    direct_E_to_total_E = np.eye(2 * n) + M_imp_to_E @ E_direct_feedback

    expected_edf = {
        "edf_from_u": D @ direct_E_to_total_E @ U,
        "edf_from_jr": D @ M_imp_to_E @ jr_to_m_imp,
        "edf_from_m_ind": m_ind_to_E_df,
    }
    expected_rates = {
        key.replace("edf_from_", "dt_m_ind_from_"): scale * value
        for key, value in expected_edf.items()
    }

    runtime_m_imp_to_E = state._m_imp_to_E_coeffs_runtime
    assert isinstance(runtime_m_imp_to_E, LinearMap)
    assert runtime_m_imp_to_E is state._m_imp_to_E_coeffs_runtime
    np.testing.assert_allclose(
        runtime_m_imp_to_E.matvec(np.arange(n, dtype=float)),
        M_imp_to_E @ np.arange(n, dtype=float),
    )

    edf_matrices = state.operators.E_df_matrices()
    rate_matrices = state.operators.rates_matrices()

    assert state._direct_E_coeffs_to_total_E_coeffs_operator is None
    assert set(edf_matrices) == set(expected_edf)
    assert set(rate_matrices) == set(expected_rates)
    for key, expected in expected_edf.items():
        np.testing.assert_allclose(edf_matrices[key], expected)
    for key, expected in expected_rates.items():
        np.testing.assert_allclose(rate_matrices[key], expected)

    sample = np.arange(2 * n, dtype=float)
    operators = state.operators.E_df()
    np.testing.assert_allclose(
        operators["edf_from_u"].matvec(sample),
        expected_edf["edf_from_u"] @ sample,
    )

    scipy_operator = operators["edf_from_u"].as_linear_operator()
    np.testing.assert_allclose(
        scipy_operator.matvec(sample),
        expected_edf["edf_from_u"] @ sample,
    )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_model_matrix_accessors_accept_explicit_jax_backend():
    """Dense model accessors should accept backend='jax'."""
    previous_backend = use_jax()
    n = 2
    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = SimpleNamespace(
        helmholtz_divergence_free_potential_operator=as_linear_map(
            np.arange(n * 2 * n, dtype=float).reshape(n, 2, n),
            input_shape=(2, n),
            output_shape=(n,),
        ),
        E_df_to_d_m_ind_dt=1.0,
        Br_to_gridded_JS=None,
    )
    state._u_coeffs_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[np.ones((2, n, 2, n))],
        einsum_string_dense="cmrs->cmrs",
        einsum_string_matvec="cmrs,rs->cm",
        einsum_string_rmatvec="cm,cmrs->rs",
        output_shape=(2, n),
        input_shape=(2, n),
    )
    state._m_imp_to_E_coeffs_cache = einsum_linear_map(
        component_tensors=[np.ones((2, n, n))],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    state._Br_to_E_coeffs_cache = None
    state._jr_to_m_imp_matrix = np.eye(n)
    state._E_direct_to_m_imp_matrix = None
    state._m_ind_to_E_df_operator = as_linear_map(np.eye(n))
    state._direct_E_coeffs_to_total_E_coeffs_operator = None
    state._direct_E_coeffs_to_E_df_operator = None
    state.connect_hemispheres = False
    state._E_map_constraint_cache = None
    state._ensure_m_imp_response_matrices = lambda: None

    try:
        set_backend("numpy")
        matrices = state.operators.E_df_matrices(backend="jax")
    finally:
        set_backend(previous_backend)

    assert all("jax" in type(matrix).__module__ for matrix in matrices.values())
