"""
Provides a solver that computes and caches components necessary
for solving a LeastSquaresProblem.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Union, Any, List
from scipy.sparse.linalg import LinearOperator, cg, lsmr
from pynamit.math.least_squares_problem import LeastSquaresProblem

ITERATION_SAFETY_FACTOR = 10


class LeastSquaresSolver:
    """
    A solver that computes and caches factorizations and preconditioners.

    This class holds the configuration for a solution method and is responsible
    for generating and caching any expensive, solver-specific components
    (e.g., SVD factorizations, preconditioners) for any problem it is asked
    to solve. It manages an internal cache to avoid re-computation when
    solving the same problem multiple times.
    """

    def __init__(
        self,
        solver: str = "lsmr",
        tolerance: float = 1e-13,
        preconditioner: Optional[Union[str, LinearOperator]] = None,
        use_scaled_lambdas: bool = True,
    ):
        """
        Initializes the solver configuration.

        Args:
            solver: The numerical algorithm to use. One of ['normal', 'lsmr', 'cg', 'svd'].
            tolerance: The convergence tolerance for iterative solvers.
            preconditioner: The preconditioner to use for 'cg' or 'lsmr'.
                Can be a string ['jacobi', 'pinv'], None, or an externally
                provided LinearOperator (for 'cg' solver only).
            use_scaled_lambdas: If True, automatically scale regularization
                weights to balance data and regularization terms.
        """
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers:
            raise ValueError(f"Solver must be one of {solvers}")

        if not (
            preconditioner is None
            or isinstance(preconditioner, str)
            or isinstance(preconditioner, LinearOperator)
        ):
            raise TypeError("Preconditioner must be a string, a LinearOperator, or None.")

        if isinstance(preconditioner, str) and preconditioner not in ["jacobi", "pinv"]:
            raise ValueError(f"Preconditioner string must be one of ['jacobi', 'pinv']")

        self.solver = solver
        self.tolerance = tolerance
        self.preconditioner = preconditioner
        self.use_scaled_lambdas = use_scaled_lambdas
        self._cache = {}  # Cache for solver-specific components, keyed by problem id

    def solve(
        self, problem: LeastSquaresProblem, b: Union[Any, List[Any]], **kwargs
    ) -> np.ndarray:
        """Solves the given least-squares problem for the right-hand-side b."""
        lambdas = self._get_lambdas_for_problem(problem)
        d_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(b, lambdas=lambdas)

        if d_block is None:  # No valid b was provided
            dtype = problem.A[0].op.dtype if problem.A else np.float64
            return np.zeros(problem.solution_shape, dtype=dtype)

        sol_block = None
        if self.solver == "svd":
            u, s_inv, vt = self._get_svd_components(problem)
            sol_block = vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ d_block))

        elif self.solver == "normal":
            G_T_G, G_T = self._get_normal_components(problem)
            sol_block = np.linalg.solve(G_T_G, G_T @ d_block)

        elif self.solver == "lsmr":
            if isinstance(self.preconditioner, LinearOperator):
                raise ValueError(
                    "External LinearOperator preconditioners are only supported for the 'cg' solver."
                )
            op, transform = self._get_lsmr_components(problem, num_scenarios)
            m, n = op.shape[0] // num_scenarios, op.shape[1] // num_scenarios
            max_iter = kwargs.pop(
                "maxiter",
                ITERATION_SAFETY_FACTOR * min(m, n) if min(m, n) > 0 else problem.solution_size,
            )
            lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance, "maxiter": max_iter, **kwargs}
            sol_y_flat, istop, *_ = lsmr(op, d_block.flatten(), **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                print(f"Warning: LSMR may not have fully converged (istop={istop}).")
            sol_block = transform(sol_y_flat.reshape(problem.solution_size, num_scenarios))

        elif self.solver == "cg":
            cg_op, M, rmatvec_block = self._get_cg_components(problem, num_scenarios)
            rhs_flat = rmatvec_block(d_block).flatten()
            max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(cg_op, rhs_flat, **cg_kwargs)
            if exit_code != 0:
                print(f"Warning: CG solver did not converge (exit_code={exit_code}).")
            sol_block = sol_flat.reshape(problem.solution_size, num_scenarios)

        return sol_block.reshape(problem.solution_shape + scenario_shape)

    # --- Cache Management ---

    def _get_problem_cache(self, problem: LeastSquaresProblem) -> dict:
        problem_id = id(problem)
        if problem_id not in self._cache or self._cache[problem_id].get("version") != problem._version:
            self._cache[problem_id] = {"version": problem._version}
        return self._cache[problem_id]

    def clear_cache(self) -> None:
        self._cache.clear()

    def clear_cache_for_problem(self, problem: LeastSquaresProblem) -> None:
        self._cache.pop(id(problem), None)

    # --- Solver Component Generation and Caching ---

    def _get_lambdas_for_problem(self, problem: LeastSquaresProblem) -> List[float]:
        if self.use_scaled_lambdas:
            return self._get_scaled_lambdas(problem)
        return problem.regularization_weights

    def _get_scaled_lambdas(self, problem: LeastSquaresProblem) -> List[float]:
        cache = self._get_problem_cache(problem)
        if "scaled_lambdas" in cache:
            return cache["scaled_lambdas"]

        data_rows = sum(a.op.shape[0] for a in problem.A)
        dtype = problem.A[0].op.dtype if problem.A else np.float64

        def data_matvec(x_flat: np.ndarray) -> np.ndarray:
            x_block = x_flat.reshape(problem.solution_size, 1)
            output_blocks = []
            for i, a_item in enumerate(problem.A):
                res_block = problem.apply_op_to_block(a_item.op, x_block)
                w_item = problem.sqrt_weights[i]
                if w_item is not None:
                    res_block = (
                        problem.densify_op(w_item) * res_block
                        if w_item.input_shape == (1,)
                        else problem.apply_op_to_block(w_item.op, res_block)
                    )
                output_blocks.append(res_block)
            return np.vstack(output_blocks).ravel() if output_blocks else np.array([], dtype=dtype)

        data_op = LinearOperator(shape=(data_rows, problem.solution_size), matvec=data_matvec, dtype=dtype)
        diag_A_T_A = np.zeros(problem.solution_size, dtype=data_op.dtype)
        for i in range(problem.solution_size):
            e = np.zeros(problem.solution_size); e[i] = 1.0
            col = data_op.matvec(e)
            diag_A_T_A[i] = np.dot(col.conj(), col).real
        data_scale = np.median(diag_A_T_A[diag_A_T_A > 0]) if np.any(diag_A_T_A > 0) else 1.0

        scaled_lambdas: List[float] = []
        for i, L_item in enumerate(problem.regularization_matrices):
            raw_weight = problem.regularization_weights[i]
            if raw_weight == 0 or L_item is None:
                scaled_lambdas.append(0.0); continue
            diag_L_T_L = np.zeros(problem.solution_size, dtype=L_item.op.dtype)
            for j in range(problem.solution_size):
                e_j = np.zeros(problem.solution_size); e_j[j] = 1.0
                col_j = problem.apply_op_to_block(L_item.op, e_j[:, np.newaxis])[:, 0]
                diag_L_T_L[j] = np.dot(col_j.conj(), col_j).real
            reg_scale = np.median(diag_L_T_L[diag_L_T_L > 0]) if np.any(diag_L_T_L > 0) else 1.0
            scale_factor = np.sqrt(data_scale / reg_scale) if reg_scale > 1e-14 else 0.0
            scaled_lambdas.append(math.sqrt(raw_weight) * scale_factor)

        cache["scaled_lambdas"] = scaled_lambdas
        return scaled_lambdas

    def _get_dense_system_matrix(self, problem: LeastSquaresProblem) -> np.ndarray:
        cache = self._get_problem_cache(problem)
        cache_key = f"G_dense_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]

        lambdas = self._get_lambdas_for_problem(problem)
        all_A_weighted, all_L_weighted = [], []

        for i, a_item in enumerate(problem.A):
            op = problem.densify_op(a_item)
            w_item = problem.sqrt_weights[i]
            if w_item is not None:
                w_op = problem.densify_op(w_item)
                op = w_op * op if w_item.input_shape == (1,) else w_op @ op
            all_A_weighted.append(op)

        for i, L_item in enumerate(problem.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_L_weighted.append(lambdas[i] * problem.densify_op(L_item))

        G_dense = (
            np.vstack(all_A_weighted + all_L_weighted)
            if (all_A_weighted or all_L_weighted)
            else np.zeros((0, problem.solution_size), dtype=problem.A[0].op.dtype if problem.A else np.float64)
        )
        cache[cache_key] = G_dense
        return G_dense

    def _get_svd_components(self, problem: LeastSquaresProblem) -> tuple:
        cache = self._get_problem_cache(problem)
        cache_key = f"svd_components_{self.tolerance}_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]
        G_dense = self._get_dense_system_matrix(problem)
        u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
        s_inv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        stable_s = s > cutoff
        s_inv[stable_s] = 1.0 / s[stable_s]
        result = (u, s_inv, vt)
        cache[cache_key] = result
        return result

    def _get_normal_components(self, problem: LeastSquaresProblem) -> tuple:
        cache = self._get_problem_cache(problem)
        cache_key = f"normal_components_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]
        G_dense = self._get_dense_system_matrix(problem)
        result = (G_dense.T.conj() @ G_dense, G_dense.T.conj())
        cache[cache_key] = result
        return result

    def _get_jacobi_preconditioner_diag(self, problem: LeastSquaresProblem) -> np.ndarray:
        cache = self._get_problem_cache(problem)
        cache_key = f"jacobi_diag_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]
        lambdas = self._get_lambdas_for_problem(problem)
        base_op, _, _ = problem.get_system_operator(lambdas=lambdas)
        diag = np.zeros(problem.solution_size, dtype=base_op.dtype)
        for i in range(problem.solution_size):
            e = np.zeros(problem.solution_size); e[i] = 1.0
            col = base_op.matvec(e)
            diag[i] = np.dot(col.conj(), col).real
        cache[cache_key] = diag
        return diag

    def _get_pinv_preconditioner_components(self, problem: LeastSquaresProblem) -> tuple:
        cache = self._get_problem_cache(problem)
        cache_key = f"pinv_components_{self.tolerance}_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]
        G_dense = self._get_dense_system_matrix(problem)
        u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
        s_pinv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        stable = s > cutoff
        s_pinv[stable] = 1.0 / s[stable]
        s_inv_sq = s_pinv**2
        result = (u, s, vt, s_pinv, s_inv_sq)
        cache[cache_key] = result
        return result

    def _get_lsmr_components(self, problem: LeastSquaresProblem, num_scenarios: int) -> tuple:
        cache = self._get_problem_cache(problem)
        cache_key = f"lsmr_components_{num_scenarios}_{self.preconditioner}_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]
        lambdas = self._get_lambdas_for_problem(problem)
        base_op, _, matvec_block = problem.get_system_operator(num_scenarios, lambdas=lambdas)
        op_to_solve, solution_transform = base_op, lambda sol_block: sol_block
        if self.preconditioner == "jacobi":
            diag = self._get_jacobi_preconditioner_diag(problem)
            sqrt_inv = np.sqrt(1.0 / diag); sqrt_inv[np.isinf(sqrt_inv)] = 1.0
            def precond_matvec(y_flat):
                return base_op.matvec((y_flat.reshape(problem.solution_size, num_scenarios) * sqrt_inv[:, None]).flatten())
            def precond_rmatvec(d_flat):
                return (base_op.rmatvec(d_flat).reshape(problem.solution_size, num_scenarios) * sqrt_inv[:, None]).flatten()
            op_to_solve = LinearOperator(base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype)
            solution_transform = lambda sol_y_block: sol_y_block * sqrt_inv[:, None]
        elif self.preconditioner == "pinv":
            _, _, vt, s_pinv, _ = self._get_pinv_preconditioner_components(problem)
            def p_matvec(y_block): return vt.T.conj() @ (s_pinv[:, None] * (vt @ y_block))
            def precond_matvec(y_flat): return matvec_block(p_matvec(y_flat.reshape(problem.solution_size, num_scenarios))).flatten()
            def precond_rmatvec(d_flat): return p_matvec(base_op.rmatvec(d_flat).reshape(problem.solution_size, num_scenarios)).flatten()
            op_to_solve = LinearOperator(base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype)
            solution_transform = p_matvec
        result = (op_to_solve, solution_transform)
        cache[cache_key] = result
        return result

    def _get_cg_components(self, problem: LeastSquaresProblem, num_scenarios: int) -> tuple:
        preconditioner_id = id(self.preconditioner) if isinstance(self.preconditioner, LinearOperator) else self.preconditioner
        cache = self._get_problem_cache(problem)
        cache_key = f"cg_components_{num_scenarios}_{preconditioner_id}_{self.use_scaled_lambdas}"
        if cache_key in cache:
            return cache[cache_key]
        lambdas = self._get_lambdas_for_problem(problem)
        base_op, rmatvec_block, _ = problem.get_system_operator(num_scenarios, lambdas=lambdas)
        def normal_matvec(x_flat): return base_op.rmatvec(base_op.matvec(x_flat))
        cg_op = LinearOperator((base_op.shape[1], base_op.shape[1]), matvec=normal_matvec, rmatvec=normal_matvec, dtype=base_op.dtype)
        M = None
        if isinstance(self.preconditioner, LinearOperator): M = self.preconditioner
        elif self.preconditioner == "jacobi":
            diag = self._get_jacobi_preconditioner_diag(problem)
            diag_inv = 1.0 / diag; diag_inv[np.isinf(diag_inv)] = 1.0
            full_inv = np.tile(diag_inv, num_scenarios)
            def precon_matvec(x_flat): return x_flat * full_inv
            M = LinearOperator(cg_op.shape, matvec=precon_matvec, rmatvec=precon_matvec, dtype=diag.dtype)
        elif self.preconditioner == "pinv":
            _, _, vt, _, s_inv_sq = self._get_pinv_preconditioner_components(problem)
            def precon_block(x_block): return vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))
            def precon_matvec(x_flat): return precon_block(x_flat.reshape(problem.solution_size, num_scenarios)).flatten()
            M = LinearOperator(cg_op.shape, matvec=precon_matvec, rmatvec=precon_matvec, dtype=vt.dtype)
        result = (cg_op, M, rmatvec_block)
        cache[cache_key] = result
        return result