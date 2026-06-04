"""Configurable solver for ``LeastSquaresProblem`` objects."""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from pynamit.math.backend import (
    block_after_jax_linalg,
    block_until_ready,
    get_array_module,
    to_numpy,
)
from .least_squares_problem import LeastSquaresProblem
from .linear_map import LinearMap, as_linear_map, diagonal_linear_map

ITERATION_SAFETY_FACTOR: Final = 10
LEAST_SQUARES_SOLVER_ENV: Final = "PYNAMIT_LEAST_SQUARES_SOLVER"
PreconditionerInput = Optional[Union[LinearOperator, LinearMap]]


class LeastSquaresSolver:
    """A collection of algorithms for solving least-squares problems."""

    VALID_SOLVERS: Final[Tuple[str, ...]] = ("normal_solve", "normal_pinv", "lsmr", "cgls", "svd")
    VALID_PRECONDITIONERS: Final[Tuple[str, ...]] = ("jacobi", "pinv")

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
            "cgls": self._solve_cgls,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: Union[np.ndarray, List[np.ndarray]],
        preconditioner: PreconditionerInput = None,
        **kwargs,
    ) -> Any:
        """Solve least-squares problem for given right-hand side(s)."""
        rhs_block, rhs_shape, num_rhs = problem.assemble_rhs_block(rhs)
        if rhs_block is None:
            dtype = problem.A[0].dtype if problem.A else np.float64
            return get_array_module().zeros(problem.solution_shape + rhs_shape, dtype=dtype)

        preconditioner_map = as_linear_map(preconditioner) if preconditioner is not None else None
        self._validate_preconditioner_shape(problem, preconditioner_map)
        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(problem, rhs_block, num_rhs, preconditioner_map, **kwargs)
        return solution_block.reshape(problem.solution_shape + rhs_shape)

    def build_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner_type: Optional[str] = None
    ) -> Optional[LinearMap]:
        """Build preconditioner for the specified solver and problem."""
        p_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if p_type is None:
            return None
        if p_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        if self.solver == "cgls":
            return self._build_normal_eq_preconditioner(problem, p_type)
        if self.solver == "lsmr":
            return self._build_lsmr_preconditioner(problem, p_type)
        return None

    def build_response_solver(
        self,
        problem: LeastSquaresProblem,
        preconditioner: PreconditionerInput = None,
    ) -> Callable[[Union[np.ndarray, List[np.ndarray]]], Any]:
        """Return a reusable solver for matching RHS response blocks."""
        preconditioner_map = as_linear_map(preconditioner) if preconditioner is not None else None
        self._validate_preconditioner_shape(problem, preconditioner_map)
        if self.solver == "normal_pinv":
            return self._build_normal_pinv_response_solver(problem)

        def solve_response(rhs: Union[np.ndarray, List[np.ndarray]]) -> Any:
            return self.solve(problem, rhs, preconditioner=preconditioner_map)

        return solve_response

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        u, s, vt = problem.svd
        rhs_np = to_numpy(rhs_block)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        safe_s = np.where(s > cutoff, s, 1.0)
        s_inv = np.where(s > cutoff, 1.0 / safe_s, np.zeros_like(s))
        solution = vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_np))
        return xp.asarray(solution)

    def _solve_normal_solve(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Solve the normal equations with a direct dense solve."""
        xp, normal_matrix, normal_rhs = self._dense_normal_equations(problem, rhs_block)
        return block_after_jax_linalg(xp.linalg.solve(normal_matrix, normal_rhs))

    def _solve_normal_pinv(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Solve through the pseudo-inverse of the normal equations."""
        xp, normal_matrix, normal_rhs = self._dense_normal_equations(problem, rhs_block)
        del normal_matrix
        normal_pinv = problem.dense_normal_pinv(self.tolerance)
        # Finish this dependent backend matmul before callers assemble
        # NumPy/SciPy blocks.
        return block_until_ready(normal_pinv @ normal_rhs)

    def _build_normal_pinv_response_solver(
        self, problem: LeastSquaresProblem
    ) -> Callable[[Union[np.ndarray, List[np.ndarray]]], Any]:
        """Build a normal-pinv response solver with cached factors."""
        xp, _, system_matrix_adjoint, _ = problem.dense_normal_equations()
        normal_pinv = problem.dense_normal_pinv(self.tolerance)

        def solve_response(rhs: Union[np.ndarray, List[np.ndarray]]) -> Any:
            rhs_block, rhs_shape, _ = problem.assemble_rhs_block(rhs)
            if rhs_block is None:
                dtype = problem.A[0].dtype if problem.A else np.float64
                return xp.zeros(problem.solution_shape + rhs_shape, dtype=dtype)
            solution_block = normal_pinv @ (system_matrix_adjoint @ rhs_block)
            return block_until_ready(
                solution_block.reshape(problem.solution_shape + rhs_shape)
            )

        return solve_response

    def _dense_normal_equations(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray
    ) -> Tuple[Any, Any, Any]:
        """Return dense normal-equation matrix and RHS."""
        xp, _, system_matrix_adjoint, normal_matrix = problem.dense_normal_equations()
        rhs = block_until_ready(xp.asarray(rhs_block))
        return xp, normal_matrix, system_matrix_adjoint @ rhs

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_rhs: int,
        M: Optional[LinearMap],
        **kwargs,
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        if xp is not np:
            return self._solve_lsmr_jax(problem, rhs_block, num_rhs, M, **kwargs)

        system_map = problem.get_system_linear_map()
        op_to_solve, sol_transform = self._preconditioned_system(system_map, M)
        lsmr_kwargs = self._lsmr_kwargs(system_map, kwargs)
        op = op_to_solve.as_linear_operator()
        rhs_np = to_numpy(rhs_block)
        columns = []
        for col in range(num_rhs):
            sol_y, istop, *_ = lsmr(op, rhs_np[:, col], **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                warnings.warn(
                    f"LSMR may not have converged for RHS column {col} (istop={istop}).",
                    RuntimeWarning,
                )
            columns.append(sol_transform(sol_y))
        return np.column_stack(columns)

    def _solve_lsmr_jax(
        self,
        problem: LeastSquaresProblem,
        rhs_block: Any,
        num_rhs: int,
        M: Optional[LinearMap],
        **kwargs,
    ) -> Any:
        """Solve rectangular least squares with internal JAX LSMR."""
        from pynamit.math.jax_lsmr import lsmr as jax_lsmr

        xp = get_array_module(rhs_block)
        system_map = problem.get_system_linear_map()
        op_to_solve, sol_transform = self._preconditioned_system(system_map, M)
        lsmr_kwargs = self._lsmr_kwargs(system_map, kwargs)
        columns = []
        for col in range(num_rhs):
            sol_y, istop, *_ = jax_lsmr(op_to_solve, rhs_block[:, col], **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                warnings.warn(
                    f"LSMR may not have converged for RHS column {col} (istop={istop}).",
                    RuntimeWarning,
                )
            columns.append(sol_transform(sol_y))
        return xp.stack(columns, axis=1)

    def _preconditioned_system(
        self, system_map: LinearMap, M: Optional[LinearMap]
    ) -> tuple[LinearMap, Callable[[Any], Any]]:
        """Return the solve operator and solution transform."""
        if M is None:
            return system_map, lambda y_vec: y_vec
        return system_map @ M, M.matvec

    def _lsmr_kwargs(self, system_map: LinearMap, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return LSMR options with the default iteration cap."""
        m, n = system_map.shape
        max_iter = kwargs.pop(
            "maxiter", ITERATION_SAFETY_FACTOR * min(m, n) if m > 0 and n > 0 else n
        )
        return {
            "atol": self.tolerance,
            "btol": self.tolerance,
            "maxiter": max_iter,
            **kwargs,
        }

    def _solve_cgls(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_rhs: int,
        M: Optional[LinearMap],
        **kwargs,
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        if xp is not np:
            return self._solve_cgls_jax(problem, rhs_block, num_rhs, M, **kwargs)

        system_map = problem.get_system_linear_map()
        normal_op = LinearOperator(
            (system_map.shape[1], system_map.shape[1]),
            matvec=lambda x: np.asarray(system_map.rmatvec(system_map.matvec(x))),
            dtype=system_map.dtype,
        )
        rhs_np = to_numpy(rhs_block)
        cg_rhs = np.asarray(system_map.rmatmat(rhs_np)).reshape(
            problem.solution_size,
            num_rhs,
        )

        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        cg_kwargs = {
            "rtol": self.tolerance,
            "M": M.as_linear_operator() if M is not None else None,
            "maxiter": max_iter,
            **kwargs,
        }
        columns = []
        for col in range(num_rhs):
            sol, exit_code = cg(normal_op, cg_rhs[:, col], **cg_kwargs)
            if exit_code != 0:
                warnings.warn(
                    f"CGLS solver did not converge for RHS column {col} (exit_code={exit_code}).",
                    RuntimeWarning,
                )
            columns.append(sol)
        return np.column_stack(columns)

    def _solve_cgls_jax(
        self,
        problem: LeastSquaresProblem,
        rhs_block: Any,
        num_rhs: int,
        M: Optional[LinearMap],
        **kwargs,
    ) -> Any:
        """Solve normal equations with JAX CG."""
        from jax.scipy.sparse.linalg import cg as jax_cg

        system_map = problem.get_system_linear_map()
        cg_rhs = system_map.rmatmat(rhs_block).reshape(problem.solution_size, num_rhs)
        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        tolerance = kwargs.pop("tol", kwargs.pop("rtol", self.tolerance))
        cg_kwargs = {"tol": tolerance, "atol": kwargs.pop("atol", 0.0), "maxiter": max_iter}
        cg_kwargs.update(kwargs)

        def normal_matvec(x_vec):
            return system_map.rmatvec(system_map.matvec(x_vec))

        preconditioner = None if M is None else M.matvec
        columns = []
        for col in range(num_rhs):
            sol, _ = jax_cg(normal_matvec, cg_rhs[:, col], M=preconditioner, **cg_kwargs)
            columns.append(sol)
        return get_array_module(rhs_block).stack(columns, axis=1)

    def _validate_preconditioner_shape(self, problem: LeastSquaresProblem, M: Optional[LinearMap]):
        if M is None:
            return
        expected_shape = (problem.solution_size, problem.solution_size)
        if M.shape != expected_shape:
            raise ValueError(f"Preconditioner shape {M.shape} != expected {expected_shape}")

    def _build_normal_eq_preconditioner(
        self, problem: LeastSquaresProblem, p_type: str
    ) -> LinearMap:
        if p_type == "jacobi":
            return self._build_jacobi_preconditioner(problem, square_root=False)
        if p_type == "pinv":
            return self._build_pinv_preconditioner(problem, squared=True)
        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for CGLS solver.")

    def _build_lsmr_preconditioner(self, problem: LeastSquaresProblem, p_type: str) -> LinearMap:
        if p_type == "jacobi":
            return self._build_jacobi_preconditioner(problem, square_root=True)
        if p_type == "pinv":
            return self._build_pinv_preconditioner(problem, squared=False)
        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for LSMR solver.")

    def _build_jacobi_preconditioner(
        self, problem: LeastSquaresProblem, *, square_root: bool
    ) -> LinearMap:
        """Build a diagonal preconditioner from ``diag(A* A)``."""
        diag = problem.get_system_linear_map().normal_matrix_diag()
        inv_diag = np.divide(1.0, diag, out=np.ones_like(diag), where=diag != 0)
        values = np.sqrt(inv_diag) if square_root else inv_diag
        return diagonal_linear_map(
            values,
            input_shape=problem.solution_shape,
            output_shape=problem.solution_shape,
        )

    def _build_pinv_preconditioner(
        self, problem: LeastSquaresProblem, *, squared: bool
    ) -> LinearMap:
        """Build a spectral pseudo-inverse preconditioner."""
        vt, s_pinv, s_pinv_sq = self._get_pinv_components(problem, self.tolerance)
        weights = s_pinv_sq if squared else s_pinv
        return self._build_spectral_preconditioner(
            problem.solution_size,
            vt,
            weights,
            problem.solution_shape,
        )

    def _build_spectral_preconditioner(
        self, size: int, vt: Any, weights: Any, solution_shape: Tuple[int, ...]
    ) -> LinearMap:
        """Build a backend-aware spectral preconditioner."""
        xp = get_array_module(vt, weights)
        vt_arr = xp.asarray(vt)
        weights_arr = xp.asarray(weights)

        def matvec(x_flat):
            x = xp.asarray(x_flat).reshape(size)
            return vt_arr.T.conj() @ (weights_arr * (vt_arr @ x))

        def rmatvec(x_flat):
            x = xp.asarray(x_flat).reshape(size)
            return vt_arr.T.conj() @ (xp.conjugate(weights_arr) * (vt_arr @ x))

        def matmat(block):
            x = xp.asarray(block).reshape(size, -1)
            return vt_arr.T.conj() @ (weights_arr.reshape(-1, 1) * (vt_arr @ x))

        def rmatmat(block):
            x = xp.asarray(block).reshape(size, -1)
            return vt_arr.T.conj() @ (
                xp.conjugate(weights_arr).reshape(-1, 1) * (vt_arr @ x)
            )

        dtype = np.result_type(vt_arr.dtype, weights_arr.dtype)
        return LinearMap(
            shape=(size, size),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _backend_context=(vt_arr, weights_arr),
            output_shape=solution_shape,
            input_shape=solution_shape,
        )

    def _get_pinv_components(
        self, problem: LeastSquaresProblem, tol: float
    ) -> Tuple[Any, Any, Any]:
        """Return SVD factors for preconditioners."""
        xp = get_array_module()
        if xp is np:
            _, s, vt = problem.svd
            s_pinv = np.zeros_like(s)
            cutoff = tol * (s[0] if s.size > 0 else 0)
            s_pinv[s > cutoff] = 1.0 / s[s > cutoff]
        else:
            system_matrix = block_until_ready(problem.assemble_dense_system_matrix())
            _, s, vt = block_after_jax_linalg(
                xp.linalg.svd(system_matrix, full_matrices=False)
            )
            cutoff = tol * (s[0] if s.size > 0 else 0)
            safe_s = xp.where(s > cutoff, s, 1.0)
            s_pinv = xp.where(s > cutoff, 1.0 / safe_s, xp.zeros_like(s))
            s_pinv = block_until_ready(s_pinv)

        return vt, s_pinv, s_pinv**2


def get_default_least_squares_solver(default: str = "normal_pinv") -> str:
    """Return the configured default least-squares solver."""
    solver = os.environ.get(LEAST_SQUARES_SOLVER_ENV, default)
    if solver not in LeastSquaresSolver.VALID_SOLVERS:
        raise ValueError(f"Solver must be one of {LeastSquaresSolver.VALID_SOLVERS}")
    return solver
