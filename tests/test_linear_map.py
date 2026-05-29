"""Tests for the LinearMap abstraction."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import LinearMap, as_linear_map, diagonal_linear_map
from pynamit.math.tensor_chain import TensorChain
from pynamit.math import JAX_AVAILABLE, set_backend, use_jax


def test_dense_linear_map_matches_matrix_operations():
    """Dense maps match matrix operations."""
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    other = np.array([[2.0, -1.0], [0.5, 4.0]])
    x = np.array([0.25, -2.0])
    y = np.array([1.0, -3.0, 2.0])
    block = np.column_stack([x, x + 1.0])

    linear_map = as_linear_map(matrix)

    np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    np.testing.assert_allclose(linear_map.rmatvec(y), matrix.T @ y)
    np.testing.assert_allclose(linear_map.matmat(block), matrix @ block)
    np.testing.assert_allclose((linear_map @ as_linear_map(other)).to_dense(), matrix @ other)


def test_diagonal_linear_map_matches_dense_diagonal():
    """Diagonal helper matches dense diagonal application."""
    diag = diagonal_linear_map(np.array([2.0, 3.0]))
    expected = np.diag([2.0, 3.0])
    x = np.arange(2.0)

    np.testing.assert_allclose(diag.matvec(x), expected @ x)
    np.testing.assert_allclose(diag.to_dense(), expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_dense_linear_map_accepts_numpy_inputs_with_jax_backend():
    """Dense maps stay NumPy-facing until called with JAX inputs."""
    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    x = np.array([0.25, -2.0])

    try:
        set_backend("jax")
        linear_map = as_linear_map(matrix)

        np.testing.assert_allclose(linear_map.to_dense(), matrix)
        np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    finally:
        set_backend(previous_backend)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_dense_linear_map_preserves_jax_dense_source(monkeypatch):
    """JAX dense inputs should not materialize during creation."""
    import jax.numpy as jnp
    import pynamit.math.linear_map as linear_map_module

    matrix = jnp.asarray([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    x = jnp.asarray([0.25, -2.0])

    def fail_asarray(_):
        raise AssertionError("as_linear_map should preserve JAX dense inputs")

    with monkeypatch.context() as context:
        context.setattr(linear_map_module.np, "asarray", fail_asarray)
        linear_map = as_linear_map(matrix)

    result = linear_map.matvec(x)
    with monkeypatch.context() as context:
        context.setattr(linear_map_module, "to_numpy", fail_asarray)
        dense = linear_map.materialize_dense(jnp)

    assert "jax" in type(result).__module__
    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(result), np.asarray(matrix) @ np.asarray(x))
    np.testing.assert_allclose(np.asarray(dense), np.asarray(matrix))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_diagonal_linear_map_preserves_jax_dense_source(monkeypatch):
    """JAX diagonal inputs should not materialize during creation."""
    import jax.numpy as jnp
    import pynamit.math.linear_map as linear_map_module

    diagonal = jnp.asarray([2.0, 3.0])
    x = jnp.asarray([0.25, -2.0])

    def fail_asarray(_):
        raise AssertionError("as_linear_map should preserve JAX diagonal inputs")

    with monkeypatch.context() as context:
        context.setattr(linear_map_module.np, "asarray", fail_asarray)
        linear_map = as_linear_map(diagonal)

    result = linear_map.matvec(x)
    with monkeypatch.context() as context:
        context.setattr(linear_map_module, "to_numpy", fail_asarray)
        dense = linear_map.materialize_dense(jnp)

    assert "jax" in type(result).__module__
    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(result), np.asarray(diagonal) * np.asarray(x))
    np.testing.assert_allclose(np.asarray(dense), np.diag(np.asarray(diagonal)))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_linear_map_materialize_dense_uses_active_backend():
    """Dense materialization can stay on the active backend."""
    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    base = as_linear_map(matrix)
    matrix_free = LinearMap(
        shape=base.shape,
        dtype=base.dtype,
        _matvec=base.matvec,
        _rmatvec=base.rmatvec,
        _matmat=base.matmat,
        _rmatmat=base.rmatmat,
    )

    try:
        set_backend("jax")
        dense = matrix_free.materialize_dense()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(dense, matrix)


def test_sparse_linear_map_uses_sparse_normal_diagonal():
    """Sparse maps avoid generic densifying for normal diagonals."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    linear_map = as_linear_map(csr_matrix(matrix))

    np.testing.assert_allclose(linear_map.normal_matrix_diag(), np.sum(matrix**2, axis=0))
    np.testing.assert_allclose(linear_map.to_dense(), matrix)


def test_composed_linear_map_normal_diagonal_uses_matmat_path():
    """Composed maps do not densify for normal diagonals."""
    matrix = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    weights = np.array([2.0, -1.0, 0.25])
    base = as_linear_map(matrix)

    def fail_to_dense():
        raise AssertionError("normal_matrix_diag should not call to_dense")

    matrix_free = LinearMap(
        shape=base.shape,
        dtype=base.dtype,
        _matvec=base.matvec,
        _rmatvec=base.rmatvec,
        _matmat=base.matmat,
        _rmatmat=base.rmatmat,
        _to_dense=fail_to_dense,
    )
    composed = diagonal_linear_map(weights) @ matrix_free
    expected = np.sum((weights[:, None] * matrix) ** 2, axis=0)

    np.testing.assert_allclose(composed.normal_matrix_diag(), expected)


def test_tensor_chain_converts_to_linear_map():
    """TensorChain can be used through the LinearMap interface."""
    matrix = np.array([[1.0, 2.0, -1.0], [0.0, 3.0, 4.0]])
    chain = TensorChain(
        component_tensors=[matrix],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(3,),
    )
    linear_map = as_linear_map(chain)
    x = np.array([2.0, -1.0, 0.5])

    np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    np.testing.assert_allclose(linear_map.to_dense(), matrix)


def test_tensor_chain_batched_application_matches_dense():
    """TensorChain batched application matches dense matrix products."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 6))
    b = rng.normal(size=(6, 4))
    chain = TensorChain(
        component_tensors=[a, b],
        einsum_string_dense="ij,jk->ik",
        einsum_string_matvec="ij,jk,k->i",
        einsum_string_rmatvec="i,ij,jk->k",
        output_shape=(5,),
        input_shape=(4,),
    )
    dense = chain.to_dense()
    x_block = rng.normal(size=(4, 7))
    y_block = rng.normal(size=(5, 7))

    np.testing.assert_allclose(chain.matmat(x_block), dense @ x_block)
    np.testing.assert_allclose(chain.rmatmat(y_block), dense.T @ y_block)
    np.testing.assert_allclose(chain.normal_matrix_diag(), np.sum(dense**2, axis=0))
    np.testing.assert_allclose(as_linear_map(chain).normal_matrix_diag(), np.sum(dense**2, axis=0))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_tensor_chain_materialize_dense_uses_active_backend():
    """TensorChain dense materialization can stay on JAX."""
    previous_backend = use_jax()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
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
        dense = chain.materialize_dense()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(dense), matrix)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_tensor_chain_dtype_does_not_materialize_jax_components(monkeypatch):
    """TensorChain dtype should only inspect dtype metadata."""
    import jax.numpy as jnp
    import pynamit.math.tensor_chain as tensor_chain_module

    chain = TensorChain(
        component_tensors=[jnp.asarray([[1.0, 2.0], [3.0, 5.0]])],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    def fail_to_numpy(_):
        raise AssertionError("dtype should not materialize component arrays")

    monkeypatch.setattr(tensor_chain_module, "to_numpy", fail_to_numpy)

    assert np.dtype(chain.dtype) == np.dtype(float)
    assert chain.to_linear_map().dtype == np.dtype(float)


def test_tensor_chain_complex_adjoint_matches_dense():
    """TensorChain adjoints match dense conjugate transpose products."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    b = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    chain = TensorChain(
        component_tensors=[a, b],
        einsum_string_dense="ij,jk->ik",
        einsum_string_matvec="ij,jk,k->i",
        einsum_string_rmatvec="i,ij,jk->k",
        output_shape=(3,),
        input_shape=(2,),
        scaling_factor=2.0 - 0.5j,
    )
    dense = chain.to_dense()
    y = rng.normal(size=3) + 1j * rng.normal(size=3)
    y_block = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))

    np.testing.assert_allclose(chain.rmatvec(y), dense.conj().T @ y)
    np.testing.assert_allclose(chain.rmatmat(y_block), dense.conj().T @ y_block)
    np.testing.assert_allclose(chain.normal_matrix_diag(), np.sum(np.abs(dense) ** 2, axis=0))


def test_least_squares_accepts_linear_map_and_sparse_inputs():
    """LeastSquaresProblem normalizes LinearMap and sparse operators."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]

    for operator in [as_linear_map(A), csr_matrix(A)]:
        problem = LeastSquaresProblem(A=operator, solution_shape=2, data_shapes=3)
        solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-12)
        solution = solver.solve(problem, rhs, maxiter=200)
        np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)
