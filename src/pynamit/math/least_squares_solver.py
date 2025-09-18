"""
Provides a solver that computes and caches components necessary
for solving a LeastSquaresProblem.
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
    """A stateful solver that is bound to a specific LeastSquaresProblem.

    This solver computes and caches components for a single problem at a time.
    To solve a different problem, or to solve the same problem after it has
    been modified, you **must** first bind it to the solver using the
    `update_problem` method, which clears any previous cache.
    """

    VALID_SOLVERS: Final[List[str]] = ["normal", "lsmr", "cg", "svd"]
    VALID_PRECONDITIONERS: Final[List[str]] = ["jacobi", "pinv"]

    def __init__(
        self,
        solver: str = "lsmr",
        tolerance: float = 1e-13,
        preconditioner: Optional[Union[str, LinearOperator]] = None,
        use_scaled_lambdas: bool = True,
        problem: Optional[LeastSquaresProblem] = None,
    ):
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        self.solver = solver

        if isinstance(preconditioner, str):
            if preconditioner not in self.VALID_PRECONDITIONERS:
                raise ValueError(
                    f"Preconditioner string must be one of {self.VALID_PRECONDITIONERS}"
                )
        elif preconditioner is not None and not isinstance(preconditioner, LinearOperator):
            raise TypeError("Preconditioner must be a string, a LinearOperator, or None.")
        self.preconditioner = preconditioner

        self.tolerance = tolerance
        self.use_scaled_lambdas = use_scaled_lambdas

        self.problem: Optional[LeastSquaresProblem] = None
        self._cache: Dict[Any, Any] = {}

        self._solve_methods: Dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal": self._solve_normal,
            "lsmr": self._solve_lsmr,
            "cg": self._solve_cg,
        }

        if problem is not None:
            self.update_problem(problem)

    def update_problem(self, problem: LeastSquaresProblem) -> None:
        """Binds the solver to a new problem and clears the cache."""
        self.problem = problem
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clears all cached components for the current problem."""
        self._cache.clear()

    def solve(self, rhs: Union[np.ndarray, List[np.ndarray]], **kwargs) -> np.ndarray:
        """Solves the currently bound least-squares problem."""
        if self.problem is None:
            raise RuntimeError("Solver must be bound to a problem first. Call update_problem().")

        lambdas = self._get_lambdas_for_problem()
        rhs_block, scenario_shape, num_scenarios = self.problem.assemble_rhs_block(
            rhs, lambdas=lambdas
        )

        if rhs_block is None:
            dtype = self.problem.A[0].op.dtype if self.problem.A else np.float64
            return np.zeros(self.problem.solution_shape, dtype=dtype)

        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(rhs_block, num_scenarios, **kwargs)

        return solution_block.reshape(self.problem.solution_shape + scenario_shape)

    def solve_adjoint(self, grad_x: np.ndarray) -> List[np.ndarray]:
        """Solves the adjoint problem.

        Given the gradient with respect to the solution `x`, this computes the
        gradient with respect to the right-hand side `rhs` terms.

        Args:
            grad_x: Gradient with respect to the solution, with a shape matching
                    the problem's `solution_shape`.

        Returns:
            A list of gradients, where each element corresponds to an `rhs`
            term from the forward solve.
        """
        if self.problem is None:
            raise RuntimeError("Solver must be bound to a problem first. Call update_problem().")

        if grad_x.shape != self.problem.solution_shape:
            raise ValueError(
                f"Shape of grad_x {grad_x.shape} does not match solution_shape {self.problem.solution_shape}"
            )

        num_scenarios = 1
        grad_x_block = grad_x.reshape(self.problem.solution_size, num_scenarios)

        grad_d_block = self._solve_adjoint(grad_x_block, num_scenarios)

        grad_b_list = []
        row = 0
        for A_item in self.problem.A:
            num_rows = A_item.op.shape[0]
            grad_b = grad_d_block[row : row + num_rows, :]
            grad_b_list.append(grad_b.reshape(A_item.output_shape + (num_scenarios,)))
            row += num_rows

        return [gb.squeeze(axis=-1) if gb.shape[-1] == 1 else gb for gb in grad_b_list]

    def _get_or_compute(self, key: Tuple, compute_func: Callable[[], Any]) -> Any:
        """Manages cache access for the currently bound problem."""
        if key in self._cache:
            return self._cache[key]

        result = compute_func()
        self._cache[key] = result
        return result

    def _get_preconditioner_id(self) -> Union[str, int, None]:
        """Returns a hashable identifier for the current preconditioner."""
        if isinstance(self.preconditioner, LinearOperator):
            return id(self.preconditioner)
        return self.preconditioner

    # ------------------- Forward and Adjoint Solver Implementations -------------------

    def _solve_svd(self, rhs_block: np.ndarray, num_scenarios: int, **kwargs) -> np.ndarray:
        u, s_inv, vt = self._get_svd_components()
        return vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_block))

    def _solve_normal(self, rhs_block: np.ndarray, num_scenarios: int, **kwargs) -> np.ndarray:
        G_H_G, G_H = self._get_normal_components()
        return np.linalg.solve(G_H_G, G_H @ rhs_block)

    def _solve_lsmr(self, rhs_block: np.ndarray, num_scenarios: int, **kwargs) -> np.ndarray:
        op, transform = self._get_lsmr_components(num_scenarios)
        m = op.shape[0] // num_scenarios
        n = self.problem.solution_size
        default_max_iter = ITERATION_SAFETY_FACTOR * min(m, n) if min(m, n) > 0 else n
        max_iter = kwargs.pop("maxiter", default_max_iter)
        lsmr_kwargs = {
            "atol": self.tolerance,
            "btol": self.tolerance,
            "maxiter": max_iter,
            **kwargs,
        }
        sol_y_flat, istop, *_ = lsmr(op, rhs_block.flatten(), **lsmr_kwargs)
        if istop not in [0, 1, 2]:
            warnings.warn(f"LSMR may not have fully converged (istop={istop}).", RuntimeWarning)
        solution_y = sol_y_flat.reshape(self.problem.solution_size, num_scenarios)
        return transform(solution_y)

    def _solve_cg(self, rhs_block: np.ndarray, num_scenarios: int, **kwargs) -> np.ndarray:
        cg_op, M, rmatvec_block = self._get_cg_components(num_scenarios)
        rhs_flat = rmatvec_block(rhs_block).flatten()
        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * self.problem.solution_size)
        cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
        sol_flat, exit_code = cg(cg_op, rhs_flat, **cg_kwargs)
        if exit_code != 0:
            warnings.warn(f"CG solver did not converge (exit_code={exit_code}).", RuntimeWarning)
        return sol_flat.reshape(self.problem.solution_size, num_scenarios)

    def _solve_adjoint(self, grad_x_block: np.ndarray, num_scenarios: int) -> np.ndarray:
        """Dispatches to the correct adjoint solver based on self.solver."""
        if self.solver == "svd":
            u, s_inv, vt = self._get_svd_components()
            return u @ (s_inv[:, None] * (vt @ grad_x_block))

        if self.solver == "normal":
            G_H_G, _ = self._get_normal_components()
            G_dense = self._get_dense_system_matrix()
            y = np.linalg.solve(G_H_G, grad_x_block)
            return G_dense @ y

        if self.solver == "lsmr":
            lambdas = self._get_lambdas_for_problem()
            base_op, _, _ = self.problem.get_system_operator(num_scenarios, lambdas=lambdas)
            adjoint_op = base_op.adjoint
            lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance}
            grad_d_flat, istop, *_ = lsmr(adjoint_op, grad_x_block.flatten(), **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                warnings.warn(
                    f"Adjoint LSMR may not have fully converged (istop={istop}).", RuntimeWarning
                )
            return grad_d_flat.reshape(adjoint_op.shape[1], num_scenarios)

        if self.solver == "cg":
            cg_op, M, _ = self._get_cg_components(num_scenarios)
            cg_kwargs = {"rtol": self.tolerance, "M": M}
            y_flat, exit_code = cg(cg_op, grad_x_block.flatten(), **cg_kwargs)
            if exit_code != 0:
                warnings.warn(
                    f"Adjoint CG solve did not converge (exit_code={exit_code}).", RuntimeWarning
                )
            y_block = y_flat.reshape(self.problem.solution_size, num_scenarios)
            lambdas = self._get_lambdas_for_problem()
            _, _, matvec_block = self.problem.get_system_operator(num_scenarios, lambdas=lambdas)
            return matvec_block(y_block)

        raise RuntimeError(f"Adjoint solver for '{self.solver}' not implemented.")

    # ------------------- Component Getters with Caching -------------------

    def _get_lambdas_for_problem(self) -> List[float]:
        if not self.use_scaled_lambdas:
            return self.problem.regularization_weights
        key = ("scaled_lambdas",)

        def compute():
            data_op = self.problem.get_data_operator()
            diag_A_T_A = self._compute_normal_matrix_diag(data_op)
            data_scale = np.median(diag_A_T_A[diag_A_T_A > 0]) if np.any(diag_A_T_A > 0) else 1.0
            scaled_lambdas = []
            for i, L_item in enumerate(self.problem.regularization_matrices):
                raw_weight = self.problem.regularization_weights[i]
                if raw_weight == 0 or L_item is None:
                    scaled_lambdas.append(0.0)
                    continue
                diag_L_T_L = self._compute_normal_matrix_diag(L_item.op)
                reg_scale = (
                    np.median(diag_L_T_L[diag_L_T_L > 0]) if np.any(diag_L_T_L > 0) else 1.0
                )
                scale_factor = math.sqrt(data_scale / reg_scale) if reg_scale > 1e-14 else 0.0
                scaled_lambdas.append(math.sqrt(raw_weight) * scale_factor)
            return scaled_lambdas

        return self._get_or_compute(key, compute)

    def _get_dense_system_matrix(self) -> np.ndarray:
        key = ("G_dense",)

        def compute():
            lambdas = self._get_lambdas_for_problem()
            return self.problem.get_dense_system_matrix(lambdas)

        return self._get_or_compute(key, compute)

    def _get_svd_decomposition(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("svd_decomposition",)

        def compute():
            G_dense = self._get_dense_system_matrix()
            return np.linalg.svd(G_dense, full_matrices=False)

        return self._get_or_compute(key, compute)

    def _get_svd_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("svd_components", self.tolerance)

        def compute():
            u, s, vt = self._get_svd_decomposition()
            s_inv = np.zeros_like(s)
            cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
            stable_s = s > cutoff
            s_inv[stable_s] = 1.0 / s[stable_s]
            return u, s_inv, vt

        return self._get_or_compute(key, compute)

    def _get_pinv_preconditioner_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("pinv_preconditioner", self.tolerance)

        def compute():
            _, s, vt = self._get_svd_decomposition()
            s_pinv = np.zeros_like(s)
            cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
            stable = s > cutoff
            s_pinv[stable] = 1.0 / s[stable]
            s_inv_sq = s_pinv**2
            return vt, s_pinv, s_inv_sq

        return self._get_or_compute(key, compute)

    def _get_normal_components(self) -> Tuple[np.ndarray, np.ndarray]:
        key = ("normal_components",)

        def compute():
            G_dense = self._get_dense_system_matrix()
            G_H = G_dense.T.conj()
            return G_H @ G_dense, G_H

        return self._get_or_compute(key, compute)

    def _get_jacobi_preconditioner_diag(self) -> np.ndarray:
        key = ("jacobi_diag",)

        def compute():
            lambdas = self._get_lambdas_for_problem()
            base_op, _, _ = self.problem.get_system_operator(lambdas=lambdas)
            return self._compute_normal_matrix_diag(base_op)

        return self._get_or_compute(key, compute)

    def _get_lsmr_components(self, num_scenarios: int) -> Tuple[LinearOperator, Callable]:
        key = ("lsmr_components", num_scenarios, self._get_preconditioner_id())

        def compute():
            lambdas = self._get_lambdas_for_problem()
            base_op, _, _ = self.problem.get_system_operator(num_scenarios, lambdas=lambdas)
            op_to_solve, solution_transform = base_op, lambda sol_block: sol_block
            if isinstance(self.preconditioner, LinearOperator):
                precond_op = self.preconditioner
                op_to_solve = LinearOperator(
                    base_op.shape,
                    matvec=lambda y: base_op.matvec(precond_op.matvec(y)),
                    rmatvec=lambda d: precond_op.rmatvec(base_op.rmatvec(d)),
                    dtype=base_op.dtype,
                )
                solution_transform = lambda y_block: precond_op.matvec(y_block.flatten()).reshape(
                    y_block.shape
                )
            elif self.preconditioner == "jacobi":
                diag = self._get_jacobi_preconditioner_diag()
                sqrt_inv = np.sqrt(1.0 / diag, where=diag != 0, out=np.ones_like(diag))

                def matvec(y_flat):
                    y_pre = (
                        y_flat.reshape(self.problem.solution_size, num_scenarios)
                        * sqrt_inv[:, None]
                    )
                    return base_op.matvec(y_pre.flatten())

                def rmatvec(d_flat):
                    res = base_op.rmatvec(d_flat).reshape(
                        self.problem.solution_size, num_scenarios
                    )
                    return (res * sqrt_inv[:, None]).flatten()

                op_to_solve = LinearOperator(
                    base_op.shape, matvec=matvec, rmatvec=rmatvec, dtype=base_op.dtype
                )
                solution_transform = lambda y_block: y_block * sqrt_inv[:, None]
            elif self.preconditioner == "pinv":
                vt, s_pinv, _ = self._get_pinv_preconditioner_components()
                p_matvec = lambda y: vt.T.conj() @ (s_pinv[:, None] * (vt @ y))
                op_to_solve = LinearOperator(
                    base_op.shape,
                    matvec=lambda y: base_op.matvec(
                        p_matvec(y.reshape(self.problem.solution_size, num_scenarios)).flatten()
                    ),
                    rmatvec=lambda d: p_matvec(
                        base_op.rmatvec(d).reshape(self.problem.solution_size, num_scenarios)
                    ).flatten(),
                    dtype=base_op.dtype,
                )
                solution_transform = p_matvec
            return op_to_solve, solution_transform

        return self._get_or_compute(key, compute)

    def _get_cg_components(
        self, num_scenarios: int
    ) -> Tuple[LinearOperator, Optional[LinearOperator], Callable]:
        key = ("cg_components", num_scenarios, self._get_preconditioner_id())

        def compute():
            lambdas = self._get_lambdas_for_problem()
            base_op, rmatvec_block, _ = self.problem.get_system_operator(
                num_scenarios, lambdas=lambdas
            )
            normal_matvec = lambda x: base_op.rmatvec(base_op.matvec(x))
            cg_op = LinearOperator(
                (base_op.shape[1], base_op.shape[1]), matvec=normal_matvec, dtype=base_op.dtype
            )
            M = None
            if isinstance(self.preconditioner, LinearOperator):
                M = self.preconditioner
            elif self.preconditioner == "jacobi":
                diag = self._get_jacobi_preconditioner_diag()
                full_inv = np.tile(1.0 / diag, num_scenarios)
                full_inv[np.isinf(full_inv)] = 1.0
                M = LinearOperator(cg_op.shape, matvec=lambda x: x * full_inv, dtype=diag.dtype)
            elif self.preconditioner == "pinv":
                vt, _, s_inv_sq = self._get_pinv_preconditioner_components()

                def precon_matvec(x_flat):
                    x_block = x_flat.reshape(self.problem.solution_size, num_scenarios)
                    y_block = vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))
                    return y_block.flatten()

                M = LinearOperator(cg_op.shape, matvec=precon_matvec, dtype=vt.dtype)
            return cg_op, M, rmatvec_block

        return self._get_or_compute(key, compute)

    @staticmethod
    def _compute_normal_matrix_diag(op: Union[LinearOperator, np.ndarray]) -> np.ndarray:
        if isinstance(op, np.ndarray):
            return np.sum(np.abs(op) ** 2, axis=0)
        n_cols = op.shape[1]
        diag = np.zeros(n_cols, dtype=op.dtype)
        e = np.zeros(n_cols, dtype=op.dtype)
        for i in range(n_cols):
            e[i] = 1.0
            col = op.matvec(e)
            diag[i] = np.dot(col.conj(), col).real
            e[i] = 0.0
        return diag
