"""
Provides a stateless, configurable solver for LeastSquaresProblem objects.
"""

from __future__ import annotations
import math
import warnings
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from .least_squares_problem import LeastSquaresProblem


ITERATION_SAFETY_FACTOR: Final = 10


class LeastSquaresSolver:
    """A collection of algorithms for solving least-squares problems.

    This solver is stateless. It is configured upon initialization and then
    can be used to solve multiple problems.
    """

    VALID_SOLVERS: Final[List[str]] = ["normal", "lsmr", "cg", "svd"]
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

        # REFACTORED: Use dispatch dictionaries for both forward and adjoint solves
        self._solve_methods: Dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal": self._solve_normal,
            "lsmr": self._solve_lsmr,
            "cg": self._solve_cg,
        }
        self._solve_adjoint_methods: Dict[str, Callable] = {
            "svd": self._solve_adjoint_svd,
            "normal": self._solve_adjoint_normal,
            "lsmr": self._solve_adjoint_lsmr,
            "cg": self._solve_adjoint_cg,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: Union[np.ndarray, List[np.ndarray]],
        preconditioner: Optional[LinearOperator] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solves Gx = d for x."""
        rhs_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(rhs)

        if rhs_block is None:  # Handle empty/None RHS gracefully
            dtype = problem.A[0].dtype if problem.A else np.float64
            return np.zeros(problem.solution_shape + scenario_shape, dtype=dtype)

        self._validate_preconditioner_shape(problem, preconditioner, num_scenarios)

        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(problem, rhs_block, num_scenarios, preconditioner, **kwargs)

        return solution_block.reshape(problem.solution_shape + scenario_shape)

    def solve_adjoint(
        self,
        problem: LeastSquaresProblem,
        grad_x: np.ndarray,
        preconditioner: Optional[LinearOperator] = None,
    ) -> List[np.ndarray]:
        """Solves for grad_d, where G.H @ grad_d = grad_x."""
        if grad_x.shape != problem.solution_shape:
            raise ValueError(
                f"Shape of grad_x {grad_x.shape} != solution_shape {problem.solution_shape}"
            )

        num_scenarios = 1
        grad_x_block = grad_x.reshape(problem.solution_size, num_scenarios)
        self._validate_preconditioner_shape(problem, preconditioner, num_scenarios)

        # REFACTORED: Use a dispatch dictionary instead of a large if/elif block
        adjoint_solver_func = self._solve_adjoint_methods[self.solver]
        grad_d_block = adjoint_solver_func(problem, grad_x_block, num_scenarios, preconditioner)

        # Un-stack the result into a list of gradients corresponding to each b term
        grad_b_list = []
        row = 0
        for A_item in problem.A:
            grad_b = grad_d_block[row : row + A_item.num_rows, :]
            # Reshape and remove the scenario dimension if it's 1
            reshaped_grad = grad_b.reshape(A_item.output_shape + (num_scenarios,))
            grad_b_list.append(reshaped_grad.squeeze(axis=-1))
            row += A_item.num_rows

        return grad_b_list

    def build_preconditioner(
        self,
        problem: LeastSquaresProblem,
        preconditioner_type: Optional[str] = None,
        num_scenarios: int = 1,
    ) -> Optional[LinearOperator]:
        """Builds a preconditioner for the given problem."""
        p_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if p_type is None:
            return None
        if p_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")

        # REFACTORED: Delegate to solver-specific builder methods
        if self.solver in ["cg", "normal"]:
            return self._build_normal_eq_preconditioner(problem, p_type, num_scenarios)
        if self.solver == "lsmr":
            return self._build_lsmr_preconditioner(problem, p_type, num_scenarios)

        return None  # SVD and other direct solvers don't use preconditioners

    # ------------------- Forward Solver Implementations -------------------

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        # API CHANGE: problem.get_svd_decomposition() -> problem.svd
        u, s, vt = problem.svd
        s_inv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        s_inv[s > cutoff] = 1.0 / s[s > cutoff]
        return vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_block))

    def _solve_normal(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        # API CHANGE: problem.get_dense_system_matrix() -> problem.dense_system_matrix
        G = problem.dense_system_matrix
        G_H = G.T.conj()
        return np.linalg.solve(G_H @ G, G_H @ rhs_block)

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        M: Optional[LinearOperator],
        **kwargs,
    ) -> np.ndarray:
        # API CHANGE: Unpacking the tuple is now safe
        G, _, _ = problem.get_system_operator(num_scenarios)
        op_to_solve, sol_transform = G, lambda sol: sol

        if M is not None:  # M is a right preconditioner P for LSMR
            op_to_solve = LinearOperator(
                G.shape,
                matvec=lambda y: G.matvec(M.matvec(y)),
                rmatvec=lambda d: M.rmatvec(G.rmatvec(d)),
                dtype=G.dtype,
            )
            sol_transform = lambda y_block: M.matvec(y_block.flatten()).reshape(y_block.shape)

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
        # API CHANGE: Unpacking the tuple is now safe
        G, rmatvec_block, _ = problem.get_system_operator(num_scenarios)

        # CG solves (G.H G) x = G.H d
        normal_op = LinearOperator(
            (G.shape[1], G.shape[1]), matvec=lambda x: G.rmatvec(G.matvec(x)), dtype=G.dtype
        )
        cg_rhs = rmatvec_block(rhs_block).flatten()

        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}

        sol_flat, exit_code = cg(normal_op, cg_rhs, **cg_kwargs)
        if exit_code != 0:
            warnings.warn(f"CG solver did not converge (exit_code={exit_code}).", RuntimeWarning)

        return sol_flat.reshape(problem.solution_size, num_scenarios)

    # ------------------- Adjoint Solver Implementations -------------------

    def _solve_adjoint_svd(
        self, problem: LeastSquaresProblem, grad_x_block: np.ndarray, *args
    ) -> np.ndarray:
        u, s, vt = problem.svd
        s_inv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        s_inv[s > cutoff] = 1.0 / s[s > cutoff]
        # grad_d = G @ pinv(G.H @ G) @ grad_x = U @ diag(1/s) @ V.H @ grad_x
        return u @ (s_inv[:, None] * (vt @ grad_x_block))

    def _solve_adjoint_normal(
        self, problem: LeastSquaresProblem, grad_x_block: np.ndarray, *args
    ) -> np.ndarray:
        G = problem.dense_system_matrix
        # y = inv(G.H G) @ grad_x, then grad_d = G @ y
        y = np.linalg.solve(G.T.conj() @ G, grad_x_block)
        return G @ y

    def _solve_adjoint_lsmr(
        self, problem: LeastSquaresProblem, grad_x_block: np.ndarray, num_scenarios: int, *args
    ) -> np.ndarray:
        G, _, _ = problem.get_system_operator(num_scenarios)
        # We need to solve G.H @ grad_d = grad_x. This is a least-squares problem for grad_d.
        adjoint_op = G.adjoint
        lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance}
        grad_d_flat, istop, *_ = lsmr(adjoint_op, grad_x_block.flatten(), **lsmr_kwargs)
        if istop not in [0, 1, 2]:
            warnings.warn(f"Adjoint LSMR may not have converged (istop={istop}).", RuntimeWarning)
        return grad_d_flat.reshape(adjoint_op.shape[1], num_scenarios)

    def _solve_adjoint_cg(
        self,
        problem: LeastSquaresProblem,
        grad_x_block: np.ndarray,
        num_scenarios: int,
        M: Optional[LinearOperator],
    ) -> np.ndarray:
        G, _, matvec_block = problem.get_system_operator(num_scenarios)
        # We solve (G.H G) y = grad_x, then grad_d = G @ y
        normal_op = LinearOperator(
            (G.shape[1], G.shape[1]), matvec=lambda x: G.rmatvec(G.matvec(x)), dtype=G.dtype
        )

        cg_kwargs = {"rtol": self.tolerance, "M": M}
        y_flat, exit_code = cg(normal_op, grad_x_block.flatten(), **cg_kwargs)
        if exit_code != 0:
            warnings.warn(
                f"Adjoint CG solve did not converge (exit_code={exit_code}).", RuntimeWarning
            )

        return matvec_block(y_flat.reshape(problem.solution_size, num_scenarios))

    # ------------------- Preconditioner and Helpers -------------------

    def _validate_preconditioner_shape(
        self, problem: LeastSquaresProblem, M: Optional[LinearOperator], num_scenarios: int
    ):
        """DRY helper to check preconditioner shape."""
        if M is None:
            return
        expected_size = problem.solution_size * num_scenarios
        expected_shape = (expected_size, expected_size)
        if M.shape != expected_shape:
            raise ValueError(f"Preconditioner shape {M.shape} != expected {expected_shape}")

    def _build_normal_eq_preconditioner(
        self, problem: LeastSquaresProblem, p_type: str, num_scenarios: int
    ) -> LinearOperator:
        """Builds preconditioner M that approximates inv(G.H @ G) for CG."""
        size = problem.solution_size * num_scenarios
        shape = (size, size)

        if p_type == "jacobi":
            # API CHANGE: Directly use the static method from the class
            G, _, _ = problem.get_system_operator(num_scenarios=1)
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
        """Builds right preconditioner P for LSMR, where x = Py."""
        size = problem.solution_size * num_scenarios
        shape = (size, size)

        if p_type == "jacobi":
            G, _, _ = problem.get_system_operator(num_scenarios=1)
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
        """DRY helper for SVD-based preconditioner components."""
        _, s, vt = problem.svd
        s_pinv = np.zeros_like(s)
        cutoff = tol * (s[0] if s.size > 0 else 0)
        s_pinv[s > cutoff] = 1.0 / s[s > cutoff]
        return vt, s_pinv, s_pinv**2
