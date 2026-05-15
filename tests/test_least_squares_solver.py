"""Tests for least-squares solver helpers."""

import numpy as np
import pytest

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver


def test_normal_pinv_solves_block_rhs():
    """Normal-equation pseudo-inverse supports reusable RHS maps."""
    A = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rcond=solver.tolerance, hermitian=True) @ (A_H @ rhs)
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
