"""Tests for State operator application helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from pynamit.math import JAX_AVAILABLE, set_backend, use_jax
from pynamit.math.linear_map import as_linear_map
from pynamit.math.tensor_chain import TensorChain
from pynamit.simulation.state import State


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_apply_operator_keeps_tensor_chain_on_jax():
    """Tensor chains should not be forced through NumPy."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    coeffs = jnp.asarray([7.0, 11.0])
    chain = TensorChain(
        component_tensors=[matrix],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    try:
        set_backend("jax")
        result = State._apply_operator(None, chain, coeffs, (2,))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ np.asarray(coeffs))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_apply_operator_handles_scipy_linear_operator_on_jax_backend():
    """SciPy operators still use a synchronized NumPy application."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    matrix = np.array([[2.0, -1.0], [0.5, 4.0]])
    coeffs = jnp.asarray([3.0, -2.0])
    operator = LinearOperator(
        matrix.shape,
        matvec=lambda vector: matrix @ np.asarray(vector),
        dtype=matrix.dtype,
    )

    try:
        set_backend("jax")
        result = State._apply_operator(None, operator, coeffs, (2,))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix @ np.asarray(coeffs))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_u_coeffs_to_E_coeffs_is_tensor_chain_on_jax():
    """Wind-to-E application should use the TensorChain path."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    n = 3
    G_helmholtz_pinv = np.arange(2 * n * 2 * 4, dtype=float).reshape(2, n, 2, 4) / 10.0
    bu = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4) / 20.0
    G_helmholtz = np.arange(2 * 4 * 2 * n, dtype=float).reshape(2, 4, 2, n) / 30.0
    coeffs = np.arange(2 * n, dtype=float).reshape(2, n) / 40.0

    G_u_to_uxB_grid = np.einsum("pqg,qgrs->pgrs", bu, G_helmholtz, optimize=True)
    expected = np.tensordot(G_helmholtz_pinv, G_u_to_uxB_grid, axes=([2, 3], [0, 1]))
    expected = np.tensordot(expected, coeffs, axes=([2, 3], [0, 1]))

    state = object.__new__(State)
    state.basis = SimpleNamespace(index_length=n)
    state.geometry = SimpleNamespace(
        G_helmholtz_pinv=jnp.asarray(G_helmholtz_pinv),
        bu=jnp.asarray(bu),
        field_transform=SimpleNamespace(G_helmholtz=jnp.asarray(G_helmholtz)),
    )

    try:
        set_backend("jax")
        operator = state._create_u_to_E_operator()
        result = State._apply_operator(None, operator, jnp.asarray(coeffs), (2, n))
    finally:
        set_backend(previous_backend)

    assert isinstance(operator, TensorChain)
    assert "jax" in type(result).__module__
    np.testing.assert_allclose(np.asarray(result), expected)


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
    state._m_ind_to_E_coeffs = TensorChain(
        component_tensors=[jnp.asarray(E_direct_matrix)],
        einsum_string_dense="cml->cml",
        einsum_string_matvec="cml,l->cm",
        einsum_string_rmatvec="cm,cml->l",
        output_shape=(2, n),
        input_shape=(n,),
    )
    state._m_imp_to_E_coeffs = TensorChain(
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
    state._E_map_constraint_operator = object()
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
