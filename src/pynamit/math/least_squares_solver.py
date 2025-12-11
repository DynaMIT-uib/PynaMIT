"""Provides a configurable solver for LeastSquaresProblem objects."""

from __future__ import annotations
import warnings
from typing import Callable, Dict, Final, List, Optional, Tuple, Union, TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr


from .least_squares_problem import LeastSquaresProblem
from .linear_map import LinearMap
from pynamit.utils import xp, asarray, use_jax

ITERATION_SAFETY_FACTOR: Final = 10
RHSInput: TypeAlias = Union[np.ndarray, List[np.ndarray]]


class LeastSquaresSolver:
    """A collection of algorithms for solving least-squares problems."""

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

        self._solve_methods: Dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal": self._solve_normal,
            "lsmr": self._solve_lsmr,
            "cg": self._solve_cg,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: RHSInput,
        preconditioner: Optional[Union[LinearOperator, LinearMap]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solve least-squares problem for given right-hand side(s)."""
        rhs_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(rhs)
        if rhs_block is None:
            dtype = problem.A[0].dtype if problem.A else xp.float64
            return xp.zeros(problem.solution_shape + scenario_shape, dtype=dtype)

        rhs_block = asarray(rhs_block)

        self._validate_preconditioner_shape(problem, preconditioner, num_scenarios)
        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(problem, rhs_block, num_scenarios, preconditioner, **kwargs)

        solution_block = asarray(solution_block)
        return solution_block.reshape(problem.solution_shape + scenario_shape)

    def build_preconditioner(
        self,
        problem: LeastSquaresProblem,
        preconditioner_type: Optional[str] = None,
        num_scenarios: int = 1,
    ) -> Optional[LinearMap]:
        """Build preconditioner for the specified solver and problem."""
        p_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if p_type is None:
            return None
        if p_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        if self.solver in ["cg", "normal"]:
            return self._build_normal_eq_preconditioner(problem, p_type, num_scenarios)
        if self.solver == "lsmr":
            return self._build_lsmr_preconditioner(problem, p_type, num_scenarios)
        return None

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        G = asarray(problem.dense_system_matrix)
        b = asarray(rhs_block)
        u, s, vt = xp.linalg.svd(G, full_matrices=False)

        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        s_inv = xp.where(s > cutoff, 1.0 / s, 0.0)
        # s_inv[:, None] broadcasting works for both xp
        return vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ b))

    def _solve_normal(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        G = asarray(problem.dense_system_matrix)
        b = asarray(rhs_block)
        G_H = G.T.conj()
        return xp.linalg.solve(G_H @ G, G_H @ b)

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        M: Optional[Union[LinearOperator, LinearMap]],
        **kwargs,
    ) -> np.ndarray:
        # LSMR is not available in JAX, so we force NumPy backend for this solver
        rhs_np = np.asarray(rhs_block)
        return self._solve_lsmr_numpy_backend(problem, rhs_np, num_scenarios, M, **kwargs)

    def _solve_lsmr_numpy_backend(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        M: Optional[Union[LinearOperator, LinearMap]],
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
        # Unified LinearMap for the system operator G
        linear_map = problem.get_system_linear_map(
            num_scenarios=num_scenarios, include_regularization=True
        )

        # Prepare RHS for normal equations: G^T * b
        # We flatten rhs_block to match the linear map's expectations
        rhs_flat = rhs_block.flatten() if rhs_block.ndim > 1 else rhs_block
        normal_rhs = linear_map.rmatvec(rhs_flat)
        x0_flat = None
        if "x0" in kwargs:
            x0 = kwargs.pop("x0")
            x0_flat = x0.flatten() if x0 is not None else None

        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        tol = self.tolerance

        if use_jax():
            try:
                from jax.scipy.sparse.linalg import cg as jax_cg
            except ImportError as exc:
                raise RuntimeError("JAX backend is required for the JAX CG solver.") from exc

            # JAX functional interface
            def normal_mv(x):
                return linear_map.rmatvec(linear_map.matvec(x))

            M_func = None
            if M is not None:

                def M_func(x):
                    return M.matvec(x)

            sol_flat, info = jax_cg(
                normal_mv, normal_rhs, x0=x0_flat, tol=tol, maxiter=max_iter, M=M_func, **kwargs
            )
            if info is not None and info != 0:
                warnings.warn(
                    f"JAX CG solver may not have converged (info={int(info)}).", RuntimeWarning
                )

        else:
            # SciPy LinearOperator interface
            def matvec_normal(x):
                return linear_map.rmatvec(linear_map.matvec(x))

            normal_op = LinearOperator(
                (linear_map.shape[1], linear_map.shape[1]),
                matvec=matvec_normal,
                dtype=linear_map.dtype,
            )

            cg_kwargs = {"rtol": tol, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(normal_op, normal_rhs, x0=x0_flat, **cg_kwargs)
            if exit_code != 0:
                warnings.warn(f"CG solver did not converge (exit_code={exit_code}).", RuntimeWarning)

        return sol_flat.reshape(problem.solution_size, num_scenarios)

    def _validate_preconditioner_shape(
        self, problem: LeastSquaresProblem, M: Optional[Union[LinearOperator, LinearMap]], num_scenarios: int
    ):
        if M is None:
            return
        expected_size = problem.solution_size * num_scenarios
        expected_shape = (expected_size, expected_size)
        if M.shape != expected_shape:
            raise ValueError(f"Preconditioner shape {M.shape} != expected {expected_shape}")

    def _build_normal_eq_preconditioner(
        self, problem: LeastSquaresProblem, p_type: str, num_scenarios: int
    ) -> LinearMap:
        size = problem.solution_size * num_scenarios
        shape = (size, size)
        dtype = problem.A[0].dtype if problem.A else xp.float64

        if p_type == "jacobi":
            G = problem.get_system_operator(num_scenarios=1)
            # Use data_operator (LinearMap) to compute diagonal for robustness
            diag = LeastSquaresProblem._compute_normal_matrix_diag(problem.data_operator)
            
            full_inv_diag = xp.tile(1.0 / diag, num_scenarios)
            full_inv_diag = xp.where(xp.isinf(full_inv_diag), 1.0, full_inv_diag)
            
            def matvec(x):
                return x * full_inv_diag
                
            return LinearMap(
                shape=shape,
                dtype=dtype,
                _matvec=matvec,
                _rmatvec=matvec,
            )

        if p_type == "pinv":
            vt, s_pinv, s_inv_sq = self._get_pinv_components(problem, self.tolerance)

            def matvec(x_flat):
                x_block = x_flat.reshape(problem.solution_size, num_scenarios)
                y_block = vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))
                return y_block.flatten()

            return LinearMap(
                shape=shape,
                dtype=vt.dtype,
                _matvec=matvec,
                _rmatvec=matvec,
            )
        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for CG solver.")

    def _build_lsmr_preconditioner(
        self, problem: LeastSquaresProblem, p_type: str, num_scenarios: int
    ) -> LinearMap:
        size = problem.solution_size * num_scenarios
        shape = (size, size)
        dtype = problem.A[0].dtype if problem.A else xp.float64
        
        if p_type == "jacobi":
            diag = LeastSquaresProblem._compute_normal_matrix_diag(problem.data_operator)
            sqrt_inv = xp.sqrt(1.0 / diag, where=diag != 0, out=xp.ones_like(diag))
            full_sqrt_inv = xp.tile(sqrt_inv, num_scenarios)
            
            def matvec(v):
                return v * full_sqrt_inv

            return LinearMap(
                shape=shape,
                dtype=dtype,
                _matvec=matvec,
                _rmatvec=matvec,
            )

        if p_type == "pinv":
            vt, s_pinv, _ = self._get_pinv_components(problem, self.tolerance)

            def matvec(y_flat):
                y_block = y_flat.reshape(problem.solution_size, num_scenarios)
                x_block = vt.T.conj() @ (s_pinv[:, None] * (vt @ y_block))
                return x_block.flatten()

            return LinearMap(
                shape=shape,
                dtype=vt.dtype,
                _matvec=matvec,
                _rmatvec=matvec,
            )

        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for LSMR solver.")

    def _get_pinv_components(
        self, problem: LeastSquaresProblem, tol: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, s, vt = problem.svd
        s_pinv = xp.zeros_like(s)
        cutoff = tol * (s[0] if s.size > 0 else 0)
        s_pinv = xp.where(s > cutoff, 1.0 / s, s_pinv)  # Safe generic assignment
        return vt, s_pinv, s_pinv**2
