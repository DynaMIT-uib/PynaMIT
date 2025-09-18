"""
Provides a stateless solver for LeastSquaresProblem objects.
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

    This solver is stateless. All necessary information, including the problem
    definition and any pre-computed preconditioners, must be passed as arguments
    to the solution methods.

    For iterative solvers like 'lsmr' and 'cg', performance can be improved by
    using a preconditioner. The `build_preconditioner` method can be used to
    compute a preconditioner, which can then be passed to `solve` and
    `solve_adjoint` for multiple solves.
    """

    VALID_SOLVERS: Final[List[str]] = ["normal", "lsmr", "cg", "svd"]
    VALID_PRECONDITIONERS: Final[List[str]] = ["jacobi", "pinv"]

    def __init__(
        self,
        solver: str = "lsmr",
        tolerance: float = 1e-13,
        preconditioner: Optional[str] = None,
    ):
        """
        Initializes the solver with a configuration.

        Args:
            solver: The solution algorithm to use. Must be one of `VALID_SOLVERS`.
            tolerance: The convergence tolerance for iterative solvers.
            preconditioner: The default type of preconditioner to build, e.g.,
                "jacobi". Must be one of `VALID_PRECONDITIONERS`.
        """
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        self.solver = solver
        self.tolerance = tolerance

        if preconditioner is not None and preconditioner not in self.VALID_PRECONDITIONERS:
            raise ValueError(
                f"Preconditioner string must be one of {self.VALID_PRECONDITIONERS}"
            )
        self.preconditioner_type = preconditioner

        self._solve_methods: Dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal": self._solve_normal,
            "lsmr": self._solve_lsmr,
            "cg": self._solve_cg,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: Union[np.ndarray, List[np.ndarray]],
        preconditioner: Optional[LinearOperator] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solves a least-squares problem for a given right-hand side.

        Args:
            problem: The `LeastSquaresProblem` to solve.
            rhs: The right-hand side of the equation, as a single array or a
                list of arrays corresponding to the problem's operators.
            preconditioner: An optional pre-computed `LinearOperator` for
                accelerating iterative solvers.
            **kwargs: Additional keyword arguments to pass to the underlying
                scipy solver (e.g., `maxiter`).

        Returns:
            The solution vector `x` that minimizes ||Ax - b||.
        """
        rhs_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(rhs)

        # FIX: Check for the zero-RHS case first, before any other logic.
        # This handles inputs like [None] and prevents the preconditioner
        # shape check from failing with num_scenarios=0.
        if rhs_block is None:
            dtype = problem.A[0].op.dtype if problem.A else np.float64
            return np.zeros(problem.solution_shape + scenario_shape, dtype=dtype)

        if preconditioner is not None:
            expected_shape = (
                problem.solution_size * num_scenarios,
                problem.solution_size * num_scenarios,
            )
            if preconditioner.shape != expected_shape:
                raise ValueError(
                    f"Preconditioner shape {preconditioner.shape} does not match "
                    f"expected shape {expected_shape} for {num_scenarios} scenarios."
                )

        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(problem, rhs_block, num_scenarios, preconditioner, **kwargs)

        return solution_block.reshape(problem.solution_shape + scenario_shape)

    def solve_adjoint(
        self,
        problem: LeastSquaresProblem,
        grad_x: np.ndarray,
        preconditioner: Optional[LinearOperator] = None,
    ) -> List[np.ndarray]:
        """Solves the adjoint problem.

        Given the gradient with respect to the solution `x`, this computes the
        gradient with respect to the right-hand side `rhs` terms.

        Args:
            problem: The `LeastSquaresProblem` defining the forward operation.
            grad_x: Gradient with respect to the solution, with a shape matching
                    the problem's `solution_shape`.
            preconditioner: An optional pre-computed `LinearOperator`. Currently
                only used by the 'cg' solver.

        Returns:
            A list of gradients, where each element corresponds to an `rhs`
            term from the forward problem.
        """
        if grad_x.shape != problem.solution_shape:
            raise ValueError(
                f"Shape of grad_x {grad_x.shape} does not match solution_shape {problem.solution_shape}"
            )

        num_scenarios = 1
        grad_x_block = grad_x.reshape(problem.solution_size, num_scenarios)

        if preconditioner is not None:
            expected_shape = (
                problem.solution_size * num_scenarios,
                problem.solution_size * num_scenarios,
            )
            if preconditioner.shape != expected_shape:
                raise ValueError(
                    f"Preconditioner shape {preconditioner.shape} does not match "
                    f"expected shape {expected_shape} for {num_scenarios} scenarios."
                )

        grad_d_block = self._solve_adjoint(
            problem, grad_x_block, num_scenarios, preconditioner
        )

        grad_b_list = []
        row = 0
        for A_item in problem.A:
            num_rows = A_item.op.shape[0]
            grad_b = grad_d_block[row : row + num_rows, :]
            grad_b_list.append(grad_b.reshape(A_item.output_shape + (num_scenarios,)))
            row += num_rows

        return [gb.squeeze(axis=-1) if gb.shape[-1] == 1 else gb for gb in grad_b_list]

    def build_preconditioner(
        self,
        problem: LeastSquaresProblem,
        preconditioner_type: Optional[str] = None,
        num_scenarios: int = 1,
    ) -> Optional[LinearOperator]:
        """Builds a preconditioner for the given problem.

        The returned preconditioner is a LinearOperator that can be passed to
        the `solve` and `solve_adjoint` methods. The structure of the operator
        depends on the solver ('lsmr' or 'cg') configured in the constructor.

        Args:
            problem: The least-squares problem to build the preconditioner for.
            preconditioner_type: The type to build, e.g., "jacobi" or "pinv".
                If None, uses the type specified in the solver's constructor.
            num_scenarios: The number of concurrent solves (scenarios) the
                preconditioner should be built for.

        Returns:
            A LinearOperator representing the preconditioner, or None if no
            preconditioner is applicable.
        """
        p_type = preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        if p_type is None:
            return None
        if p_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(
                f"Preconditioner string must be one of {self.VALID_PRECONDITIONERS}"
            )

        solution_size = problem.solution_size
        total_size = solution_size * num_scenarios
        shape = (total_size, total_size)

        if self.solver == "cg":
            # For CG, the preconditioner M approximates inv(A^H A)
            if p_type == "jacobi":
                diag = self._compute_jacobi_diag(problem)
                full_inv = np.tile(1.0 / diag, num_scenarios)
                full_inv[np.isinf(full_inv)] = 1.0
                # For diagonal M, matvec and rmatvec are the same.
                return LinearOperator(shape, matvec=lambda x: x * full_inv, rmatvec=lambda x: x * full_inv, dtype=diag.dtype)

            if p_type == "pinv":
                # M = V @ diag(1/s^2) @ V.H is Hermitian.
                vt, _, s_inv_sq = self._compute_pinv_components(problem, self.tolerance)

                def matvec(x_flat):
                    x_block = x_flat.reshape(solution_size, num_scenarios)
                    y_block = vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))
                    return y_block.flatten()

                return LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=vt.dtype)

        elif self.solver == "lsmr":
            # For LSMR, P is a right preconditioner. We solve A*P*y = b, and x = P*y.
            if p_type == "jacobi":
                diag = self._compute_jacobi_diag(problem)
                sqrt_inv = np.sqrt(1.0 / diag, where=diag != 0, out=np.ones_like(diag))
                full_sqrt_inv = np.tile(sqrt_inv, num_scenarios)

                # For diagonal P, matvec and rmatvec are the same.
                def apply(v):
                    return v * full_sqrt_inv

                return LinearOperator(
                    shape, matvec=apply, rmatvec=apply, dtype=diag.dtype
                )

            if p_type == "pinv":
                # P = V.H @ diag(1/s) @ V is Hermitian.
                vt, s_pinv, _ = self._compute_pinv_components(problem, self.tolerance)

                def matvec(y_flat):
                    y_block = y_flat.reshape(solution_size, num_scenarios)
                    x_block = vt.T.conj() @ (s_pinv[:, None] * (vt @ y_block))
                    return x_block.flatten()

                # rmatvec is the same as matvec because P is Hermitian.
                return LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=vt.dtype)

        return None

    # ------------------- Forward and Adjoint Solver Implementations -------------------

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        u, s, vt = problem.get_svd_decomposition()
        s_inv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        stable_s = s > cutoff
        s_inv[stable_s] = 1.0 / s[stable_s]
        return vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_block))

    def _solve_normal(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        G_dense = problem.get_dense_system_matrix()
        G_H = G_dense.T.conj()
        G_H_G = G_H @ G_dense
        return np.linalg.solve(G_H_G, G_H @ rhs_block)

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        preconditioner: Optional[LinearOperator],
        **kwargs,
    ) -> np.ndarray:
        base_op, _, _ = problem.get_system_operator(num_scenarios)
        op_to_solve, solution_transform = base_op, lambda sol_block: sol_block

        if preconditioner is not None:
            # op_to_solve = A @ P
            # op_to_solve.H = P.H @ A.H
            op_to_solve = LinearOperator(
                base_op.shape,
                matvec=lambda y: base_op.matvec(preconditioner.matvec(y)),
                rmatvec=lambda d: preconditioner.rmatvec(base_op.rmatvec(d)),
                dtype=base_op.dtype,
            )
            solution_transform = lambda y_block: preconditioner.matvec(
                y_block.flatten()
            ).reshape(y_block.shape)

        m, n = base_op.shape[0] // num_scenarios, problem.solution_size
        default_max_iter = ITERATION_SAFETY_FACTOR * min(m, n) if min(m, n) > 0 else n
        max_iter = kwargs.pop("maxiter", default_max_iter)
        lsmr_kwargs = {
            "atol": self.tolerance,
            "btol": self.tolerance,
            "maxiter": max_iter,
            **kwargs,
        }
        # LSMR requires op_to_solve.rmatvec to exist.
        sol_y_flat, istop, *_ = lsmr(op_to_solve, rhs_block.flatten(), **lsmr_kwargs)
        if istop not in [0, 1, 2]:
            warnings.warn(f"LSMR may not have fully converged (istop={istop}).", RuntimeWarning)
        solution_y = sol_y_flat.reshape(problem.solution_size, num_scenarios)
        return solution_transform(solution_y)

    def _solve_cg(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        preconditioner: Optional[LinearOperator],
        **kwargs,
    ) -> np.ndarray:
        base_op, rmatvec_block, _ = problem.get_system_operator(num_scenarios)
        normal_matvec = lambda x: base_op.rmatvec(base_op.matvec(x))
        cg_op = LinearOperator(
            (base_op.shape[1], base_op.shape[1]), matvec=normal_matvec, dtype=base_op.dtype
        )

        rhs_flat = rmatvec_block(rhs_block).flatten()
        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        cg_kwargs = {"rtol": self.tolerance, "M": preconditioner, "maxiter": max_iter, **kwargs}
        sol_flat, exit_code = cg(cg_op, rhs_flat, **cg_kwargs)
        if exit_code != 0:
            warnings.warn(f"CG solver did not converge (exit_code={exit_code}).", RuntimeWarning)
        return sol_flat.reshape(problem.solution_size, num_scenarios)

    def _solve_adjoint(
        self,
        problem: LeastSquaresProblem,
        grad_x_block: np.ndarray,
        num_scenarios: int,
        preconditioner: Optional[LinearOperator],
    ) -> np.ndarray:
        """Dispatches to the correct adjoint solver based on self.solver."""
        if self.solver == "svd":
            u, s, vt = problem.get_svd_decomposition()
            s_inv = np.zeros_like(s)
            cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
            stable_s = s > cutoff
            s_inv[stable_s] = 1.0 / s[stable_s]
            return u @ (s_inv[:, None] * (vt @ grad_x_block))

        if self.solver == "normal":
            G_dense = problem.get_dense_system_matrix()
            G_H_G = G_dense.T.conj() @ G_dense
            y = np.linalg.solve(G_H_G, grad_x_block)
            return G_dense @ y

        if self.solver == "lsmr":
            # Adjoint solve: find grad_d such that A.H @ grad_d = grad_x
            base_op, _, _ = problem.get_system_operator(num_scenarios)
            adjoint_op = base_op.adjoint
            lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance}
            grad_d_flat, istop, *_ = lsmr(adjoint_op, grad_x_block.flatten(), **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                warnings.warn(
                    f"Adjoint LSMR may not have fully converged (istop={istop}).", RuntimeWarning
                )
            return grad_d_flat.reshape(adjoint_op.shape[1], num_scenarios)

        if self.solver == "cg":
            # Adjoint solve for CG is the same as the forward solve on the normal equations.
            # We want y where (A.H A) y = grad_x. Then grad_b = A @ y.
            base_op, _, matvec_block = problem.get_system_operator(num_scenarios)
            normal_matvec = lambda x: base_op.rmatvec(base_op.matvec(x))
            cg_op = LinearOperator(
                (base_op.shape[1], base_op.shape[1]), matvec=normal_matvec, dtype=base_op.dtype
            )

            cg_kwargs = {"rtol": self.tolerance, "M": preconditioner}
            y_flat, exit_code = cg(cg_op, grad_x_block.flatten(), **cg_kwargs)
            if exit_code != 0:
                warnings.warn(
                    f"Adjoint CG solve did not converge (exit_code={exit_code}).", RuntimeWarning
                )
            y_block = y_flat.reshape(problem.solution_size, num_scenarios)
            return matvec_block(y_block)

        raise RuntimeError(f"Adjoint solver for '{self.solver}' not implemented.")

    # ------------------- Preconditioner Component Helpers -------------------

    def _compute_pinv_components(
        self, problem: LeastSquaresProblem, tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes SVD-based components for the 'pinv' preconditioner."""
        _, s, vt = problem.get_svd_decomposition()
        s_pinv = np.zeros_like(s)
        cutoff = tolerance * (s[0] if s.size > 0 else 0)
        stable = s > cutoff
        s_pinv[stable] = 1.0 / s[stable]
        s_inv_sq = s_pinv**2
        return vt, s_pinv, s_inv_sq

    def _compute_jacobi_diag(self, problem: LeastSquaresProblem) -> np.ndarray:
        """Computes the diagonal of the normal matrix for the 'jacobi' preconditioner."""
        base_op, _, _ = problem.get_system_operator()
        return problem._compute_normal_matrix_diag(base_op)