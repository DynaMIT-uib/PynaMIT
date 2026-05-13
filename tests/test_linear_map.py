"""Tests for the LinearMap abstraction."""

import numpy as np
from scipy.sparse import csr_matrix

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import as_linear_map, block_linear_map, diagonal_linear_map
from pynamit.math.tensor_chain import TensorChain


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


def test_diagonal_and_block_linear_maps_match_dense_blocks():
    """Structured helpers match dense blocks."""
    diag = diagonal_linear_map(np.array([2.0, 3.0]))
    dense = as_linear_map(np.array([[1.0, -1.0], [4.0, 2.0]]))
    block_map = block_linear_map([[diag, dense], [dense, diag]])
    expected = np.block([[diag.to_dense(), dense.to_dense()], [dense.to_dense(), diag.to_dense()]])
    x = np.arange(4.0)

    np.testing.assert_allclose(block_map.matvec(x), expected @ x)
    np.testing.assert_allclose(block_map.to_dense(), expected)


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
