"""Tests for exact equality-constrained least-squares solves."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver


def _kkt_reference(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    rcond: float = 1e-13,
) -> np.ndarray:
    """Dense reference using KKT pseudoinverse solve."""
    n = A.shape[1]
    m_c = C.shape[0]
    K = np.block([[A.T @ A, C.T], [C, np.zeros((m_c, m_c), dtype=A.dtype)]])
    rhs = np.vstack([A.T @ b, d])
    sol = np.linalg.pinv(K, rcond=rcond) @ rhs
    return sol[:n]


@pytest.mark.parametrize(
    "solver_name, tol_x",
    [
        ("svd", 1e-8),
        ("normal_eq", 1e-8),
        ("lsmr", 1e-6),
        ("cgls", 1e-6),
    ],
)
def test_equality_constrained_solve_matches_kkt_reference(solver_name: str, tol_x: float) -> None:
    rng = np.random.default_rng(42)
    m, n, m_c, n_s = 40, 20, 2, 3

    A = rng.standard_normal((m, n))
    b = rng.standard_normal((m, n_s))
    C = rng.standard_normal((m_c, n))
    d = rng.standard_normal((m_c, n_s))

    problem = LeastSquaresProblem(
        A=[A],
        solution_shape=(n,),
        data_shapes=[(m,)],
        matrix_free=False,
    )
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-13)
    x = np.asarray(
        solver.solve(
            problem,
            [b],
            equality_operator=C,
            equality_rhs=d,
            elimination_rcond=1e-13,
        )
    )

    # Hard constraints should be met exactly up to numerical precision.
    c_res = np.linalg.norm(C @ x - d) / max(np.linalg.norm(d), 1e-30)
    assert c_res < 1e-10

    # Compare to dense KKT reference.
    x_ref = _kkt_reference(A, b, C, d, rcond=1e-13)
    x_err = np.linalg.norm(x - x_ref) / max(np.linalg.norm(x_ref), 1e-30)
    assert x_err < tol_x
