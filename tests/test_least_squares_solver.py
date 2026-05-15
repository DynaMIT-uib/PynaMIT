"""Tests for least-squares solver helpers."""

import numpy as np
import pytest

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.utils import JAX_AVAILABLE, set_backend, use_jax


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
    """Normal pseudo-inverse stays a pseudo-inverse even for full-rank systems."""
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
@pytest.mark.parametrize(
    ("solver_name", "numpy_linalg_name"),
    [("normal_solve", "solve"), ("normal_pinv", "pinv"), ("svd", "svd")],
)
def test_dense_solvers_use_jax_linalg_when_backend_enabled(
    solver_name, numpy_linalg_name, monkeypatch
):
    """Dense solvers stay on JAX linalg when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    def fail_numpy_linalg(*args, **kwargs):
        raise AssertionError("dense solve should use the active JAX backend")

    try:
        set_backend("jax")
        monkeypatch.setattr(np.linalg, numpy_linalg_name, fail_numpy_linalg)
        solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_normal_pinv_matches_numpy_hermitian_reference_when_jax_enabled(monkeypatch):
    """JAX normal pseudo-inverse uses backend pinv and matches the Hermitian reference."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()
    real_pinv = np.linalg.pinv

    def fail_numpy_pinv(*args, **kwargs):
        raise AssertionError("JAX normal_pinv should use backend pinv")

    def fail_solve(*args, **kwargs):
        raise AssertionError("normal_pinv should apply a pseudo-inverse, not solve")

    try:
        set_backend("jax")
        monkeypatch.setattr(np.linalg, "pinv", fail_numpy_pinv)
        monkeypatch.setattr(np.linalg, "solve", fail_solve)
        solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    A_H = A.T.conj()
    expected = real_pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
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


@pytest.mark.parametrize("solver_name", ["normal", "cg"])
def test_solver_aliases_are_not_accepted(solver_name):
    """Solver modes use explicit public names."""
    with pytest.raises(ValueError):
        LeastSquaresSolver(solver=solver_name)
