"""Tests for least-squares solver helpers."""

import numpy as np
import pytest

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math import JAX_AVAILABLE, set_backend, use_jax


def test_normal_pinv_solves_block_rhs():
    """Normal-equation pseudo-inverse supports reusable RHS maps."""
    A = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_uses_normal_equation_cutoff():
    """Normal pseudo-inverse applies cutoff after forming A* A."""
    A = np.diag([1.0, 1e-8])
    rhs = np.array([1.0, 1e-8])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 0.0]))


def test_normal_pinv_keeps_modes_above_normal_equation_cutoff():
    """Normal pseudo-inverse keeps modes above the A* A cutoff."""
    A = np.diag([1.0, 1e-6])
    rhs = np.array([1.0, 1e-6])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 1.0]))


def test_normal_pinv_does_not_use_direct_solve(monkeypatch):
    """Normal pseudo-inverse also used for full-rank systems."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    def fail_solve(*args, **kwargs):
        raise AssertionError("normal_pinv should apply a pseudo-inverse, not solve")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv"])
def test_dense_solvers_preserve_jax_output_when_backend_enabled(solver_name):
    """Dense solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.assemble_dense_system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_svd_solver_preserves_jax_output_when_backend_enabled():
    """SVD solver keeps JAX-facing assembly and output."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.assemble_dense_system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(solver="svd", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_normal_pinv_matches_numpy_hermitian_reference_when_jax_enabled():
    """JAX normal-pinv matches the hermitian reference."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solver_solves_block_rhs_with_base_preconditioner(solver_name):
    """Iterative block RHS solves reuse the base preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0, -1.0], [3.0, 1.0, 0.5], [0.5, -2.0, 4.0], [1.5, 0.0, 2.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-12, preconditioner="jacobi")
    preconditioner = solver.build_preconditioner(problem)

    assert preconditioner.shape == (2, 2)
    solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
@pytest.mark.parametrize("solver_name", ["cgls", "lsmr"])
@pytest.mark.parametrize("preconditioner_type", [None, "jacobi", "pinv"])
def test_iterative_solvers_preserve_jax_output_when_backend_enabled(
    solver_name, preconditioner_type
):
    """Iterative solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(
            solver=solver_name, tolerance=1e-12, preconditioner=preconditioner_type
        )
        preconditioner = solver.build_preconditioner(problem)
        solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_jax_lsmr_solves_underdetermined_block_rhs():
    """Internal JAX LSMR handles rectangular underdetermined systems."""
    A = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=4, data_shapes=3)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-12)
        solution = solver.solve(problem, rhs, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["normal", "cg"])
def test_solver_aliases_are_not_accepted(solver_name):
    """Solver modes use explicit public names."""
    with pytest.raises(ValueError):
        LeastSquaresSolver(solver=solver_name)
