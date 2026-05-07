"""Tests for least-squares solver helpers."""

import numpy as np
import pytest

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver


def test_normal_pinv_solves_multiscenario_rhs():
    """Normal-equation pseudo-inverse supports reusable RHS maps."""
    A = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [0.0, 0.0],
        ]
    )
    rhs = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rcond=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_alias_is_not_accepted():
    """Normal-equation modes use explicit solve/pinv names."""
    with pytest.raises(ValueError):
        LeastSquaresSolver(solver="normal")
