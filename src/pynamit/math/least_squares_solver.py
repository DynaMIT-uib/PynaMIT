"""Provides a configurable solver for LeastSquaresProblem objects."""

from __future__ import annotations
import warnings
from typing import Callable, Final, Optional, TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from .least_squares_problem import LeastSquaresProblem, ProcessedOperator
from .linear_map import LinearMap, as_linear_map
from pynamit.utils import xp, asarray, use_jax

ITERATION_SAFETY_FACTOR: Final = 10
RHSInput: TypeAlias = np.ndarray | list[np.ndarray]


class LeastSquaresSolver:
    """A collection of algorithms for solving least-squares problems."""

    VALID_SOLVERS: Final[list[str]] = ["normal_eq", "lsmr", "cgls", "svd"]
    VALID_PRECONDITIONERS: Final[list[str]] = ["jacobi", "pinv"]

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

        self._solve_methods: dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal_eq": self._solve_normal,
            "lsmr": self._solve_lsmr,
            "cgls": self._solve_cg,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: RHSInput,
        preconditioner: Optional[LinearOperator | LinearMap] = None,
        equality_operator: Optional[np.ndarray | LinearOperator | LinearMap] = None,
        equality_rhs: Optional[np.ndarray] = None,
        elimination_rcond: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solve least-squares problem for given right-hand side(s)."""
        warning_label = kwargs.pop("warning_label", None)
        if equality_operator is not None:
            return self._solve_with_equality_constraints(
                problem=problem,
                rhs=rhs,
                preconditioner=preconditioner,
                equality_operator=equality_operator,
                equality_rhs=equality_rhs,
                elimination_rcond=elimination_rcond,
                warning_label=warning_label,
                **kwargs,
            )

        rhs_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(rhs)
        if rhs_block is None:
            dtype = problem.A[0].dtype if problem.A else xp.float64
            # return xp.zeros(problem.solution_shape + scenario_shape, dtype=dtype)
            # Fix: shape concatenation
            return xp.zeros(problem.solution_shape + scenario_shape, dtype=dtype)

        rhs_block = asarray(rhs_block)

        self._validate_preconditioner_shape(problem, preconditioner, num_scenarios)
        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(
            problem,
            rhs_block,
            num_scenarios,
            preconditioner,
            warning_label=warning_label,
            **kwargs,
        )

        solution_block = asarray(solution_block)
        return solution_block.reshape(problem.solution_shape + scenario_shape)

    def _solve_with_equality_constraints(
        self,
        problem: LeastSquaresProblem,
        rhs: RHSInput,
        preconditioner: Optional[LinearOperator | LinearMap],
        equality_operator: np.ndarray | LinearOperator | LinearMap,
        equality_rhs: Optional[np.ndarray] = None,
        elimination_rcond: Optional[float] = None,
        warning_label: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solve constrained LS with exact equalities ``C x = d``.

        Uses null-space elimination:
            x = x0 + Z y,  where C x0 = d and C Z = 0.
        Then solves the reduced unconstrained LS problem for ``y`` using the
        configured backend solver (svd/normal_eq/lsmr/cgls).
        """
        rhs_block, scenario_shape, num_scenarios = problem.assemble_rhs_block(rhs)
        if rhs_block is None:
            dtype = problem.A[0].dtype if problem.A else xp.float64
            return xp.zeros(problem.solution_shape + scenario_shape, dtype=dtype)

        if preconditioner is not None:
            warnings.warn(
                self._format_warning_message(
                    warning_label,
                    "Preconditioner is ignored for equality-constrained solve; "
                    "preconditioning is not yet mapped through the null-space reduction.",
                ),
                RuntimeWarning,
            )

        G = np.asarray(problem.dense_system_matrix)
        b = np.asarray(rhs_block).reshape(G.shape[0], num_scenarios)
        n = int(problem.solution_size)

        C = np.asarray(as_linear_map(equality_operator).to_dense())
        if C.ndim == 1:
            C = C.reshape(1, -1)
        if C.ndim != 2 or C.shape[1] != n:
            raise ValueError(
                f"equality_operator must be 2D with {n} columns, got {C.shape}"
            )

        if C.shape[0] == 0:
            # No constraints: fall back to standard solve path.
            return self.solve(
                problem,
                rhs,
                preconditioner=preconditioner,
                warning_label=warning_label,
                **kwargs,
            )

        rcond = self.tolerance if elimination_rcond is None else max(float(elimination_rcond), 0.0)
        d_block = self._prepare_constraint_rhs(
            equality_rhs=equality_rhs,
            n_constraints=C.shape[0],
            num_scenarios=num_scenarios,
            scenario_shape=scenario_shape,
        )

        # Particular solution satisfying C x0 = d (minimum norm).
        C_pinv = np.linalg.pinv(C, rcond=rcond)
        x0 = C_pinv @ d_block

        # Null-space basis Z of C.
        _, s_c, vh_c = np.linalg.svd(C, full_matrices=True)
        if s_c.size == 0:
            rank_c = 0
        else:
            cutoff_c = rcond * float(s_c[0])
            rank_c = int(np.sum(s_c > cutoff_c))
        Z = vh_c[rank_c:].T  # (n, k)

        if Z.shape[1] == 0:
            x = x0
        else:
            A_red = G @ Z
            b_red = b - (G @ x0)
            y = self._solve_reduced_system(A_red, b_red, warning_label=warning_label, **kwargs)
            x = x0 + Z @ y

        x = asarray(x)
        return x.reshape(problem.solution_shape + scenario_shape)

    @staticmethod
    def _prepare_constraint_rhs(
        equality_rhs: Optional[np.ndarray],
        n_constraints: int,
        num_scenarios: int,
        scenario_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Normalize equality RHS into shape ``(n_constraints, num_scenarios)``."""
        if equality_rhs is None:
            return np.zeros((n_constraints, num_scenarios), dtype=float)

        d = np.asarray(equality_rhs)
        if d.ndim == 1:
            if d.shape[0] != n_constraints:
                raise ValueError(
                    f"equality_rhs length {d.shape[0]} != n_constraints {n_constraints}"
                )
            return np.repeat(d[:, None], num_scenarios, axis=1)

        if d.shape[0] != n_constraints:
            raise ValueError(
                f"equality_rhs first dim {d.shape[0]} != n_constraints {n_constraints}"
            )

        d_block = d.reshape(n_constraints, -1)
        if d_block.shape[1] == 1 and num_scenarios > 1:
            d_block = np.repeat(d_block, num_scenarios, axis=1)
        if d_block.shape[1] != num_scenarios:
            raise ValueError(
                "equality_rhs scenario shape mismatch: "
                f"expected trailing shape {scenario_shape} (num={num_scenarios}), "
                f"got {d.shape[1:]}"
            )
        return d_block

    def _solve_reduced_system(
        self,
        A_red: np.ndarray,
        b_red: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Solve reduced unconstrained LS for all scenarios."""
        warning_label = kwargs.pop("warning_label", None)
        equilibrate_columns = bool(kwargs.pop("equilibrate_columns", True))
        if self.solver == "svd":
            U, s, Vt = np.linalg.svd(A_red, full_matrices=False)
            cutoff = self.tolerance * (s[0] if s.size > 0 else 0.0)
            s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
            return Vt.T @ (s_inv[:, None] * (U.T @ b_red))

        if self.solver == "normal_eq":
            AtA = A_red.T @ A_red
            Atb = A_red.T @ b_red
            try:
                return np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(AtA, Atb, rcond=self.tolerance)[0]

        if self.solver == "lsmr":
            m, k = A_red.shape
            A_eff = A_red
            inv_col_scale = None
            if equilibrate_columns:
                A_eff, inv_col_scale = self._equilibrate_columns_dense(A_red)
            # Reduced constrained solves can be noticeably harder than the
            # unreduced row-form problem after null-space elimination. Give
            # reduced LSMR a much larger default budget to avoid istop=7 on
            # well-behaved but slow-converging constrained problems.
            max_iter = kwargs.pop(
                "maxiter", 10 * ITERATION_SAFETY_FACTOR * min(m, k) if m > 0 and k > 0 else k
            )
            lsmr_kwargs = {
                "atol": self.tolerance,
                "btol": self.tolerance,
                "maxiter": max_iter,
                "damp": 0.0,
                **kwargs,
            }
            y_scaled = np.zeros((A_eff.shape[1], b_red.shape[1]), dtype=A_eff.dtype)
            for j in range(b_red.shape[1]):
                lsmr_result = lsmr(A_eff, b_red[:, j], **lsmr_kwargs)
                y_j = lsmr_result[0]
                istop = int(lsmr_result[1])
                if not self._lsmr_stop_is_acceptable(
                    lsmr_result,
                    normb=float(np.linalg.norm(b_red[:, j])),
                    atol=self.tolerance,
                    btol=self.tolerance,
                    practical_tol_floor=1e-10,
                ):
                    warnings.warn(
                        self._format_warning_message(
                            warning_label,
                            f"Reduced LSMR may not have converged (istop={istop}).",
                        ),
                        RuntimeWarning,
                    )
                y_scaled[:, j] = y_j
            if inv_col_scale is None:
                return y_scaled
            return inv_col_scale[:, None] * y_scaled

        if self.solver == "cgls":
            A_eff = A_red
            inv_col_scale = None
            if equilibrate_columns:
                A_eff, inv_col_scale = self._equilibrate_columns_dense(A_red)

            AtA = A_eff.T @ A_eff
            Atb = A_eff.T @ b_red
            max_iter = kwargs.pop("maxiter", 10 * ITERATION_SAFETY_FACTOR * A_red.shape[1])
            y_scaled = np.zeros((A_eff.shape[1], b_red.shape[1]), dtype=A_eff.dtype)
            for j in range(b_red.shape[1]):
                y_j, exit_code = cg(
                    AtA,
                    Atb[:, j],
                    rtol=self.tolerance,
                    maxiter=max_iter,
                    **kwargs,
                )
                if exit_code != 0:
                    warnings.warn(
                        self._format_warning_message(
                            warning_label,
                            f"Reduced CG did not converge (exit_code={exit_code}).",
                        ),
                        RuntimeWarning,
                    )
                y_scaled[:, j] = y_j
            if inv_col_scale is None:
                return y_scaled
            return inv_col_scale[:, None] * y_scaled

        raise ValueError(f"Unsupported solver type '{self.solver}' for reduced solve.")

    @staticmethod
    def _equilibrate_columns_dense(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return column-equilibrated dense matrix and inverse scaling vector.

        This is a semantics-preserving right scaling (change of variables):
            A x = b,  x = D^{-1} z  ->  (A D^{-1}) z = b
        The returned vector is ``diag(D^{-1})`` for reconstructing ``x``.
        """
        a_np = np.asarray(a)
        if a_np.ndim != 2 or a_np.shape[1] == 0:
            return a_np, np.ones((a_np.shape[1] if a_np.ndim == 2 else 0,), dtype=float)

        col_norm = np.linalg.norm(a_np, axis=0)
        max_col = float(np.max(col_norm)) if col_norm.size > 0 else 0.0
        if not np.isfinite(max_col) or max_col <= 0.0:
            inv_col_scale = np.ones(a_np.shape[1], dtype=a_np.dtype)
            return a_np, inv_col_scale

        floor = np.sqrt(np.finfo(float).eps) * max_col
        scale = np.where(col_norm > floor, col_norm, 1.0)
        inv_col_scale = (1.0 / scale).astype(a_np.dtype, copy=False)
        return a_np * inv_col_scale[None, :], inv_col_scale

    @staticmethod
    def _lsmr_stop_is_acceptable(
        lsmr_result: tuple,
        *,
        normb: float,
        atol: float,
        btol: float,
        practical_tol_floor: float = 0.0,
    ) -> bool:
        """Accept only SciPy LSMR success terminations.

        Extra keyword arguments are kept for call-site compatibility with the
        previous practical-convergence policy.
        """
        istop = int(lsmr_result[1])
        return istop in (0, 1, 2, 4, 5)

    def build_preconditioner(
        self,
        problem: LeastSquaresProblem,
        preconditioner_type: Optional[str] = None,
        num_scenarios: int = 1,
        pinv_rcond: Optional[float] = None,
        pinv_mode: str = "symmetric",
    ) -> Optional[LinearMap]:
        """Build preconditioner for the specified solver and problem."""
        p_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if p_type is None:
            return None
        if p_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        if self.solver in ["cgls", "normal_eq"]:
            return self._build_normal_eq_preconditioner(
                problem, p_type, num_scenarios, pinv_rcond=pinv_rcond
            )
        if self.solver == "lsmr":
            return self._build_lsmr_preconditioner(
                problem,
                p_type,
                num_scenarios,
                pinv_rcond=pinv_rcond,
                pinv_mode=pinv_mode,
            )
        return None

    def build_equality_constrained_components_from_normal(
        self,
        H: np.ndarray,
        C: np.ndarray,
        *,
        pinv_rcond: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Build linear maps for the hard-constrained quadratic system.

        Solves the KKT system in pseudoinverse form:
            [H C^T] [x] = [g]
            [C  0 ] [λ]   [d]

        and returns operators such that:
            x = X_g g + X_d d
        """
        H_np = np.asarray(H)
        C_np = np.asarray(C)
        if H_np.ndim != 2 or H_np.shape[0] != H_np.shape[1]:
            raise ValueError(f"H must be square 2D array, got shape {H_np.shape}")
        if C_np.ndim != 2 or C_np.shape[1] != H_np.shape[0]:
            raise ValueError(
                f"C must be 2D with {H_np.shape[0]} columns, got shape {C_np.shape}"
            )

        rcond = self.tolerance if pinv_rcond is None else max(float(pinv_rcond), 0.0)
        n_x = H_np.shape[0]

        if C_np.size == 0 or C_np.shape[0] == 0:
            X_g = self._pinv_symmetric(H_np, rcond)
            X_d = np.zeros((n_x, 0), dtype=H_np.dtype)
            return {"X_g": X_g, "X_d": X_d, "P_space": X_g @ H_np}

        n_c = C_np.shape[0]
        zeros_cc = np.zeros((n_c, n_c), dtype=H_np.dtype)
        K = np.block([[H_np, C_np.T], [C_np, zeros_cc]])
        K_pinv = self._pinv_symmetric(K, rcond)

        X_g = K_pinv[:n_x, :n_x]
        X_d = K_pinv[:n_x, n_x:]
        return {"X_g": X_g, "X_d": X_d, "P_space": X_g @ H_np}

    @staticmethod
    def _pinv_symmetric(a: np.ndarray, rcond: float) -> np.ndarray:
        """Robust pseudoinverse for (near-)symmetric matrices.

        Uses an eigen-decomposition of the symmetrized matrix to avoid
        occasional SVD non-convergence in large KKT blocks.
        """
        a_np = np.asarray(a)
        if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
            return np.linalg.pinv(a_np, rcond=max(float(rcond), 0.0))

        a_sym = 0.5 * (a_np + a_np.T.conj())
        rcond = max(float(rcond), 0.0)
        try:
            eigvals, eigvecs = np.linalg.eigh(a_sym)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(a_sym, rcond=rcond)

        max_abs = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            return np.zeros_like(a_sym)

        cutoff = rcond * max_abs
        inv_eigvals = np.where(np.abs(eigvals) > cutoff, 1.0 / eigvals, 0.0)
        return (eigvecs * inv_eigvals) @ eigvecs.T.conj()

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
        M: Optional[LinearOperator | LinearMap],
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
        M: Optional[LinearOperator | LinearMap],
        **kwargs,
    ) -> np.ndarray:
        warning_label = kwargs.pop("warning_label", None)
        equilibrate_columns = bool(kwargs.pop("equilibrate_columns", True))
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

        inv_col_scale = None
        if equilibrate_columns and M is None:
            inv_col_scale = self._get_problem_column_equilibration_inverse(
                problem, include_regularization=True
            )
            if inv_col_scale is not None:
                base_op = op_to_solve
                op_to_solve = LinearOperator(
                    base_op.shape,
                    matvec=lambda z: base_op.matvec(inv_col_scale * z),
                    rmatvec=lambda r: inv_col_scale * base_op.rmatvec(r),
                    dtype=base_op.dtype,
                )
                base_transform = sol_transform

                def sol_transform(y_block):
                    scaled = (inv_col_scale[:, None] * y_block).reshape(y_block.shape)
                    return base_transform(scaled)
        elif equilibrate_columns and M is not None:
            inv_col_scale = None

        m, n = G.shape[0] // num_scenarios, problem.solution_size
        max_iter = kwargs.pop(
            "maxiter", ITERATION_SAFETY_FACTOR * min(m, n) if m > 0 and n > 0 else n
        )
        lsmr_kwargs = {
            "atol": self.tolerance,
            "btol": self.tolerance,
            "maxiter": max_iter,
            "damp": 0.0,
            **kwargs,
        }
        lsmr_result = lsmr(op_to_solve, rhs_block.flatten(), **lsmr_kwargs)
        sol_y_flat = lsmr_result[0]
        istop = int(lsmr_result[1])
        if not self._lsmr_stop_is_acceptable(
            lsmr_result,
            normb=float(np.linalg.norm(rhs_block)),
            atol=self.tolerance,
            btol=self.tolerance,
        ):
            warnings.warn(
                self._format_warning_message(
                    warning_label,
                    f"LSMR may not have converged (istop={istop}).",
                ),
                RuntimeWarning,
            )
        return sol_transform(sol_y_flat.reshape(problem.solution_size, num_scenarios))

    def _solve_cg(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_scenarios: int,
        M: Optional[LinearOperator],
        **kwargs,
    ) -> np.ndarray:
        warning_label = kwargs.pop("warning_label", None)
        equilibrate_columns = bool(kwargs.pop("equilibrate_columns", True))
        # Unified LinearMap for the system operator G
        linear_map = problem.get_system_linear_map(
            num_scenarios=num_scenarios, include_regularization=True
        )
        inv_col_scale = None
        if equilibrate_columns and M is None:
            inv_col_scale = self._get_problem_column_equilibration_inverse(
                problem, include_regularization=True
            )
        elif equilibrate_columns and M is not None:
            inv_col_scale = None

        # Prepare RHS for normal equations: G^T * b
        # We flatten rhs_block to match the linear map's expectations
        rhs_flat = rhs_block.flatten() if rhs_block.ndim > 1 else rhs_block
        normal_rhs = linear_map.rmatvec(rhs_flat)
        if inv_col_scale is not None:
            normal_rhs = inv_col_scale * normal_rhs
        x0_flat = None
        if "x0" in kwargs:
            x0 = kwargs.pop("x0")
            x0_flat = x0.flatten() if x0 is not None else None
            if x0_flat is not None and inv_col_scale is not None:
                # x = D^{-1} z  -> z = D x
                safe = np.where(inv_col_scale != 0, 1.0 / inv_col_scale, 1.0)
                x0_flat = safe * x0_flat

        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        tol = self.tolerance

        if use_jax():
            try:
                from jax.scipy.sparse.linalg import cg as jax_cg
            except ImportError as exc:
                raise RuntimeError("JAX backend is required for the JAX CG solver.") from exc

            # JAX functional interface
            def normal_mv(x):
                if inv_col_scale is None:
                    return linear_map.rmatvec(linear_map.matvec(x))
                return inv_col_scale * linear_map.rmatvec(linear_map.matvec(inv_col_scale * x))

            M_func = None
            if M is not None:

                def M_func(x):
                    return M.matvec(x)

            sol_flat, info = jax_cg(
                normal_mv, normal_rhs, x0=x0_flat, tol=tol, maxiter=max_iter, M=M_func, **kwargs
            )
            if info is not None and info != 0:
                warnings.warn(
                    self._format_warning_message(
                        warning_label,
                        f"JAX CG solver may not have converged (info={int(info)}).",
                    ),
                    RuntimeWarning,
                )

        else:
            # SciPy LinearOperator interface
            def matvec_normal(x):
                if inv_col_scale is None:
                    return linear_map.rmatvec(linear_map.matvec(x))
                return inv_col_scale * linear_map.rmatvec(linear_map.matvec(inv_col_scale * x))

            normal_op = LinearOperator(
                (linear_map.shape[1], linear_map.shape[1]),
                matvec=matvec_normal,
                dtype=linear_map.dtype,
            )

            cg_kwargs = {"rtol": tol, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(normal_op, normal_rhs, x0=x0_flat, **cg_kwargs)
            if exit_code != 0:
                warnings.warn(
                    self._format_warning_message(
                        warning_label,
                        f"CG solver did not converge (exit_code={exit_code}).",
                    ),
                    RuntimeWarning,
                )

        if inv_col_scale is not None:
            sol_flat = inv_col_scale * sol_flat
        return sol_flat.reshape(problem.solution_size, num_scenarios)

    def _get_problem_column_equilibration_inverse(
        self,
        problem: LeastSquaresProblem,
        *,
        include_regularization: bool,
    ) -> Optional[np.ndarray]:
        """Return inverse column scaling for semantics-preserving right scaling."""
        col_scale = problem.get_column_scale(include_regularization=include_regularization)
        if col_scale is None:
            return None
        col = np.asarray(col_scale, dtype=float).reshape(-1)
        if col.size != problem.solution_size:
            return None
        finite = np.isfinite(col)
        if not np.any(finite):
            return None
        max_col = float(np.max(np.abs(col[finite])))
        if max_col <= 0.0:
            return None
        floor = np.sqrt(np.finfo(float).eps) * max_col
        denom = np.where(np.abs(col) > floor, col, 1.0)
        inv = 1.0 / denom
        inv[~finite] = 1.0
        return inv.astype(float, copy=False)

    @staticmethod
    def _format_warning_message(label: Optional[str], message: str) -> str:
        """Append a lightweight solve label to warnings for attribution."""
        if label is None:
            return message
        label_str = str(label).strip()
        if not label_str:
            return message
        return f"{message} [solve={label_str}]"

    def _validate_preconditioner_shape(
        self,
        problem: LeastSquaresProblem,
        M: Optional[LinearOperator | LinearMap],
        num_scenarios: int,
    ):
        if M is None:
            return
        expected_size = problem.solution_size * num_scenarios
        expected_shape = (expected_size, expected_size)
        if M.shape != expected_shape:
            raise ValueError(f"Preconditioner shape {M.shape} != expected {expected_shape}")

    def _build_normal_eq_preconditioner(
        self,
        problem: LeastSquaresProblem,
        p_type: str,
        num_scenarios: int,
        pinv_rcond: Optional[float] = None,
    ) -> LinearMap:
        size = problem.solution_size * num_scenarios
        shape = (size, size)
        dtype = problem.A[0].dtype if problem.A else xp.float64

        if p_type == "jacobi":
            # Build Jacobi from the full system map (data + explicit regularization rows)
            # so the preconditioner matches the solved normal equations.
            full_map = problem.get_system_linear_map(
                num_scenarios=1, include_regularization=True
            )
            full_item = ProcessedOperator(
                linear_map=full_map,
                output_shape=(full_map.shape[0],),
                input_shape=(full_map.shape[1],),
            )
            diag = LeastSquaresProblem._compute_normal_matrix_diag(full_item)

            full_inv_diag = xp.tile(1.0 / diag, num_scenarios)
            full_inv_diag = xp.where(xp.isinf(full_inv_diag), 1.0, full_inv_diag)

            def matvec(x):
                return x * full_inv_diag

            return LinearMap(shape=shape, dtype=dtype, _matvec=matvec, _rmatvec=matvec)

        if p_type == "pinv":
            _, vt, s_pinv, s_inv_sq = self._get_pinv_components(
                problem, self.tolerance, pinv_rcond=pinv_rcond
            )

            def matvec(x_flat):
                x_block = x_flat.reshape(problem.solution_size, num_scenarios)
                y_block = vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))
                return y_block.flatten()

            return LinearMap(shape=shape, dtype=vt.dtype, _matvec=matvec, _rmatvec=matvec)
        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for CG solver.")

    def _build_lsmr_preconditioner(
        self,
        problem: LeastSquaresProblem,
        p_type: str,
        num_scenarios: int,
        pinv_rcond: Optional[float] = None,
        pinv_mode: str = "symmetric",
    ) -> LinearMap:
        size = problem.solution_size * num_scenarios
        shape = (size, size)
        dtype = problem.A[0].dtype if problem.A else xp.float64

        if p_type == "jacobi":
            # Build Jacobi from the full system map (data + explicit regularization rows)
            # so LSMR preconditioning reflects any regularization present in the problem.
            full_map = problem.get_system_linear_map(
                num_scenarios=1, include_regularization=True
            )
            full_item = ProcessedOperator(
                linear_map=full_map,
                output_shape=(full_map.shape[0],),
                input_shape=(full_map.shape[1],),
            )
            diag = LeastSquaresProblem._compute_normal_matrix_diag(full_item)
            sqrt_inv = xp.sqrt(xp.where(diag != 0, 1.0 / diag, 1.0))
            if xp is not np:
                 # JAX compatible replacement for 'out' behavior (set zeros to 1s before sqrt, then mask?)
                 # The logic above 'xp.ones_like(diag)' implies where diag==0 result is 1.
                 # My replacement `1.0/diag` on diag==0 is inf.
                 # Better:
                 safe_diag = xp.where(diag != 0, diag, 1.0)
                 sqrt_inv = xp.sqrt(1.0 / safe_diag)
                 # Restore 1s where diag was 0 (matches out=ones_like)
                 # Actually out=ones_like sets default value.
                 # So if diag==0, result is 1.
                 # My code does 1.0/1.0 = 1.0. Correct.
            else:
                 sqrt_inv = xp.sqrt(1.0 / diag, where=diag != 0, out=xp.ones_like(diag))
            full_sqrt_inv = xp.tile(sqrt_inv, num_scenarios)

            def matvec(v):
                return v * full_sqrt_inv

            return LinearMap(shape=shape, dtype=dtype, _matvec=matvec, _rmatvec=matvec)

        if p_type == "pinv":
            u, vt, s_pinv, _ = self._get_pinv_components(
                problem, self.tolerance, pinv_rcond=pinv_rcond, return_u=True
            )
            if pinv_mode not in ("symmetric", "true"):
                raise ValueError("pinv_mode must be 'symmetric' or 'true'.")

            if pinv_mode == "true":
                # LSMR uses right preconditioning (x = M y), so M must be square
                # in solution space. The true Moore-Penrose form V S^+ U^T maps
                # data-space to solution-space and is therefore only valid here
                # if the stacked system is square (m == n).
                if u.shape[0] != problem.solution_size * num_scenarios:
                    raise ValueError(
                        "pinv_mode='true' is not valid for rectangular LSMR systems "
                        "(right preconditioner must be square in solution space). "
                        "Use pinv_mode='symmetric'."
                    )

                def matvec(y_flat):
                    y_block = y_flat.reshape(problem.solution_size, num_scenarios)
                    x_block = vt.T.conj() @ (s_pinv[:, None] * (u.T.conj() @ y_block))
                    return x_block.flatten()

                def rmatvec(y_flat):
                    y_block = y_flat.reshape(problem.solution_size, num_scenarios)
                    x_block = u @ (s_pinv[:, None] * (vt @ y_block))
                    return x_block.flatten()
            else:
                def matvec(y_flat):
                    y_block = y_flat.reshape(problem.solution_size, num_scenarios)
                    x_block = vt.T.conj() @ (s_pinv[:, None] * (vt @ y_block))
                    return x_block.flatten()

                rmatvec = matvec

            return LinearMap(shape=shape, dtype=vt.dtype, _matvec=matvec, _rmatvec=rmatvec)

        raise NotImplementedError(f"Preconditioner '{p_type}' not implemented for LSMR solver.")

    def _get_pinv_components(
        self,
        problem: LeastSquaresProblem,
        tol: float,
        *,
        pinv_rcond: Optional[float] = None,
        return_u: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u, s, vt = problem.svd
        s_pinv = xp.zeros_like(s)
        cutoff = (pinv_rcond if pinv_rcond is not None else tol) * (s[0] if s.size > 0 else 0)
        s_pinv = xp.where(s > cutoff, 1.0 / s, s_pinv)  # Safe generic assignment
        if not return_u:
            # Maintain backward compatibility for callers that ignore U
            u = xp.zeros((0, 0), dtype=vt.dtype)
        return u, vt, s_pinv, s_pinv**2
