"""Provides a configurable solver for LeastSquaresProblem objects."""

from __future__ import annotations
import warnings
from typing import Callable, Dict, Final, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from .least_squares_problem import LeastSquaresProblem


ITERATION_SAFETY_FACTOR: Final = 10


class LeastSquaresSolver:
    """A collection of algorithms for solving least-squares problems."""

    VALID_SOLVERS: Final[List[str]] = ["normal_solve", "normal_pinv", "lsmr", "cg", "svd"]
    VALID_PRECONDITIONERS: Final[List[str]] = ["jacobi", "pinv"]

    def __init__(
        self, solver: str = "lsmr", tolerance: float = 1e-13, preconditioner: Optional[str] = None
    ):
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        self.solver = solver
        self.tolerance = tolerance

        if preconditioner is not None and preconditioner not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        self.preconditioner_type = preconditioner

        self._solve_methods: Dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal_solve": self._solve_normal_solve,
            "normal_pinv": self._solve_normal_pinv,
            "lsmr": self._solve_lsmr,
            "cg": self._solve_cg,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: Union[np.ndarray, List[np.ndarray]],
        preconditioner: Optional[LinearOperator] = None,
        solver: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solve least-squares problem for given right-hand side(s)."""
        rhs_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(rhs)
        if rhs_block is None:
            dtype = problem.A[0].dtype if problem.A else np.float64
            return np.zeros(problem.solution_shape + scenario_shape, dtype=dtype)

        self._validate_preconditioner_shape(problem, preconditioner, num_scenarios)
        solver_name = self.solver if solver is None else solver
        if solver_name not in self._solve_methods:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        solver_func = self._solve_methods[solver_name]
        solution_block = solver_func(problem, rhs_block, num_scenarios, preconditioner, **kwargs)
        return solution_block.reshape(problem.solution_shape + scenario_shape)

    def build_preconditioner(
        self,
        problem: LeastSquaresProblem,
        preconditioner_type: Optional[str] = None,
        num_scenarios: int = 1,
    ) -> Optional[LinearOperator]:
        """Build preconditioner for the specified solver and problem."""
        p_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if p_type is None:
            return None
        if p_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        if self.solver in ["cg", "normal_solve"]:
            return self._build_normal_eq_preconditioner(problem, p_type, num_scenarios)
        if self.solver == "lsmr":
            return self._build_lsmr_preconditioner(problem, p_type, num_scenarios)
        return None

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        u, s, vt = problem.svd
        s_inv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        s_inv[s > cutoff] = 1.0 / s[s > cutoff]
        return vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_block))

    def _solve_normal_solve(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Solve the normal equations with a direct dense solve."""
        G = problem.dense_system_matrix
        G_H = G.T.conj()
        return np.linalg.solve(G_H @ G, G_H @ rhs_block)

    def _solve_normal_pinv(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Solve through the pseudo-inverse of the normal equations."""
        G = problem.dense_system_matrix
        G_H = G.T.conj()
        normal_pinv = np.linalg.pinv(G_H @ G, rcond=self.tolerance, hermitian=True)
        return normal_pinv @ (G_H @ rhs_block)

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        M: Optional[LinearOperator],
        **kwargs,
    ) -> np.ndarray:
        G = problem.get_system_operator(num_scenarios)
        op_to_solve, sol_transform = G, lambda sol: sol
        if M is not None:
            op_to_solve = LinearOperator(
                G.shape,
                matvec=lambda y: G.matvec(M.matvec(y)),
                rmatvec=lambda d: M.rmatvec(G.rmatvec(d)),
                dtype=G.dtype,
            )

            def sol_transform(y_block):
                return M.matvec(y_block.flatten()).reshape(y_block.shape)

        m, n = G.shape[0] // num_scenarios, problem.solution_size
        max_iter = kwargs.pop(
            "maxiter", ITERATION_SAFETY_FACTOR * min(m, n) if m > 0 and n > 0 else n
        )
        lsmr_kwargs = {
            "atol": self.tolerance,
            "btol": self.tolerance,
            "maxiter": max_iter,
            **kwargs,
        }
        sol_y_flat, istop, *_ = lsmr(op_to_solve, rhs_block.flatten(), **lsmr_kwargs)
        if istop not in [0, 1, 2]:
            warnings.warn(f"LSMR may not have converged (istop={istop}).", RuntimeWarning)
        return sol_transform(sol_y_flat.reshape(problem.solution_size, num_scenarios))

    def _solve_cg(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        M: Optional[LinearOperator],
        **kwargs,
    ) -> np.ndarray:
        G = problem.get_system_operator(num_scenarios)
        normal_op = LinearOperator(
            (G.shape[1], G.shape[1]), matvec=lambda x: G.rmatvec(G.matvec(x)), dtype=G.dtype
        )
        # Use the operator's own rmatvec on the flattened rhs
        cg_rhs = G.rmatvec(rhs_block.flatten())

        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
        sol_flat, exit_code = cg(normal_op, cg_rhs, **cg_kwargs)
        if exit_code != 0:
            warnings.warn(f"CG solver did not converge (exit_code={exit_code}).", RuntimeWarning)
        return sol_flat.reshape(problem.solution_size, num_scenarios)

    def _validate_preconditioner_shape(
        self, problem: LeastSquaresProblem, M: Optional[LinearOperator], num_scenarios: int
    ):
        if M is None:
            return
        expected_size = problem.solution_size * num_scenarios
        expected_shape = (expected_size, expected_size)
        if M.shape != expected_shape:
            raise ValueError(f"Preconditioner shape {M.shape} != expected {expected_shape}")

    def _build_normal_eq_preconditioner(
        self, problem: LeastSquaresProblem, p_type: str, num_scenarios: int
    ) -> LinearOperator:
        size = problem.solution_size * num_scenarios
        shape = (size, size)
        if p_type == "jacobi":
            G = problem.get_system_operator(num_scenarios=1)
            diag = LeastSquaresProblem._compute_normal_matrix_diag(G)
            full_inv_diag = np.tile(1.0 / diag, num_scenarios)
            full_inv_diag[np.isinf(full_inv_diag)] = 1.0
            return LinearOperator(
                shape,
                matvec=lambda x: x * full_inv_diag,
                rmatvec=lambda x: x * full_inv_diag,
                dtype=diag.dtype,
            )
        if p_type == "pinv":
            vt, s_pinv, s_inv_sq = self._get_pinv_components(problem, self.tolerance)

            def matvec(x_flat):
                x_block = x_flat.reshape(problem.solution_size, num_scenarios)
                y_block = vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))
                return y_block.flatten()

            return LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=vt.dtype)
        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for CG solver.")

    def _build_lsmr_preconditioner(
        self, problem: LeastSquaresProblem, p_type: str, num_scenarios: int
    ) -> LinearOperator:
        size = problem.solution_size * num_scenarios
        shape = (size, size)
        if p_type == "jacobi":
            G = problem.get_system_operator(num_scenarios=1)
            diag = LeastSquaresProblem._compute_normal_matrix_diag(G)
            sqrt_inv = np.sqrt(1.0 / diag, where=diag != 0, out=np.ones_like(diag))
            full_sqrt_inv = np.tile(sqrt_inv, num_scenarios)
            return LinearOperator(
                shape,
                matvec=lambda v: v * full_sqrt_inv,
                rmatvec=lambda v: v * full_sqrt_inv,
                dtype=diag.dtype,
            )
        if p_type == "pinv":
            vt, s_pinv, _ = self._get_pinv_components(problem, self.tolerance)

            def matvec(y_flat):
                y_block = y_flat.reshape(problem.solution_size, num_scenarios)
                x_block = vt.T.conj() @ (s_pinv[:, None] * (vt @ y_block))
                return x_block.flatten()

            return LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=vt.dtype)
        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for LSMR solver.")

    def _get_pinv_components(
        self, problem: LeastSquaresProblem, tol: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, s, vt = problem.svd
        s_pinv = np.zeros_like(s)
        cutoff = tol * (s[0] if s.size > 0 else 0)
        s_pinv[s > cutoff] = 1.0 / s[s > cutoff]
        return vt, s_pinv, s_pinv**2
