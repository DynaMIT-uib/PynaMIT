"""Helpers for coupled steady-state induction solves.

This module isolates gauge-elimination and reduced-system assembly used by
`State.steady_state_coupled`, so `State` can focus on physics orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings
from typing import Any, Callable, Dict, Optional, Literal

import numpy as np
from scipy.sparse.linalg import lsmr as scipy_lsmr, LinearOperator as ScipyLinearOperator

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, SimulationMode
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.utils import asarray, xp

TimedSolveFn = Callable[..., np.ndarray]
PsiGaugeRowBuilder = Callable[[int], np.ndarray]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReducedCoupledSystem:
    """Gauge-reduced coupled system shared by steady-state and time stepping."""

    full_operator: LinearMap
    selector: np.ndarray
    reduced_operator: LinearMap

    @property
    def n_total(self) -> int:
        return int(self.selector.shape[0])

    @property
    def n_reduced(self) -> int:
        return int(self.selector.shape[1])

    def reduce_vector(self, vector: Any) -> np.ndarray:
        arr = np.asarray(vector, dtype=float).reshape(self.n_total)
        return np.asarray(self.selector.T @ arr, dtype=float).reshape(self.n_reduced)

    def expand_vector(self, reduced_vector: Any) -> np.ndarray:
        arr = np.asarray(reduced_vector, dtype=float).reshape(self.n_reduced)
        return np.asarray(self.selector @ arr, dtype=float).reshape(self.n_total)

    def project_vector(self, vector: Any) -> np.ndarray:
        return self.expand_vector(self.reduce_vector(vector))


def _as_2d_coupled_linear_map(coupled_operator: Any, *, n_total: int) -> LinearMap:
    """Return operator as `LinearMap` with explicit `(2N, 2N)` dense fallback."""
    operator_obj = coupled_operator
    if not hasattr(operator_obj, "matvec"):
        operator_arr = asarray(operator_obj)
        if operator_arr.ndim == 4:
            operator_obj = operator_arr.reshape(n_total, n_total)
        else:
            operator_obj = operator_arr

    operator_map = as_linear_map(operator_obj)
    if operator_map._to_dense is None:
        return operator_map

    dense_base_op = operator_map

    def _to_dense_2d() -> np.ndarray:
        dense = dense_base_op.to_dense()
        dense_arr = asarray(dense)
        if dense_arr.ndim != 2:
            dense_arr = dense_arr.reshape(n_total, n_total)
        return dense_arr

    return LinearMap(
        shape=(n_total, n_total),
        dtype=dense_base_op.dtype,
        _matvec=dense_base_op.matvec,
        _rmatvec=dense_base_op.rmatvec,
        _matmat=dense_base_op.matmat,
        _rmatmat=dense_base_op.rmatmat,
        _to_dense=_to_dense_2d,
        source=dense_base_op,
    )


def _build_coupled_selector(
    *,
    n_scalar: int,
    apply_m_ind_gauge: bool,
    psi_gauge_row_builder: PsiGaugeRowBuilder,
    m_ind_gauge_row_builder: PsiGaugeRowBuilder,
    apply_psi_gauge: bool,
) -> np.ndarray:
    """Build orthonormal selector for the coupled gauge-constrained subspace."""
    n = n_scalar
    n_total = 2 * n

    psi_row = None
    m_ind_row = None
    use_index_selector = False
    fixed_indices: list[int] = []

    if apply_psi_gauge and n > 0:
        psi_row = np.asarray(psi_gauge_row_builder(n), dtype=float)
        if psi_row.ndim == 1:
            psi_row = psi_row.reshape(1, -1)
        if psi_row.shape == (1, n):
            pin_row = np.zeros((1, n), dtype=float)
            pin_row[0, 0] = 1.0
            if float(np.linalg.norm(psi_row - pin_row)) <= 1e-12:
                use_index_selector = True
                fixed_indices.append(0)

    if apply_m_ind_gauge and n_total > n:
        m_ind_row = np.asarray(m_ind_gauge_row_builder(n), dtype=float)
        if m_ind_row.ndim == 1:
            m_ind_row = m_ind_row.reshape(1, -1)
        if m_ind_row.shape == (1, n):
            pin_row = np.zeros((1, n), dtype=float)
            pin_row[0, 0] = 1.0
            is_pin_row = float(np.linalg.norm(m_ind_row - pin_row)) <= 1e-12
            if is_pin_row and (use_index_selector or not apply_psi_gauge):
                fixed_indices.append(n)

    if use_index_selector or (not apply_psi_gauge and apply_m_ind_gauge and n_total > n):
        fixed_idx = (
            np.unique(np.asarray(fixed_indices, dtype=int))
            if fixed_indices
            else np.zeros(0, dtype=int)
        )
        free_mask = np.ones(n_total, dtype=bool)
        if fixed_idx.size > 0:
            free_mask[fixed_idx] = False
        free_idx = np.flatnonzero(free_mask)
        selector = np.zeros((n_total, free_idx.size), dtype=np.float64)
        if free_idx.size > 0:
            selector[free_idx, np.arange(free_idx.size)] = 1.0
        return selector

    constraint_rows: list[np.ndarray] = []
    if apply_psi_gauge and psi_row is not None and psi_row.shape[1] == n and psi_row.shape[0] > 0:
        c_psi = np.zeros((psi_row.shape[0], n_total), dtype=float)
        c_psi[:, :n] = psi_row
        constraint_rows.append(c_psi)
    if (
        apply_m_ind_gauge
        and n_total > n
        and m_ind_row is not None
        and m_ind_row.ndim == 2
        and m_ind_row.shape[1] == n
        and m_ind_row.shape[0] > 0
    ):
        c_mind = np.zeros((m_ind_row.shape[0], n_total), dtype=float)
        c_mind[:, n:] = m_ind_row
        constraint_rows.append(c_mind)

    if not constraint_rows:
        return np.eye(n_total, dtype=float)

    c = np.vstack(constraint_rows)
    _, s_c, vh_c = np.linalg.svd(c, full_matrices=True)
    if s_c.size == 0:
        rank_c = 0
    else:
        rtol_c = float(np.finfo(float).eps * max(c.shape))
        cutoff_c = rtol_c * float(s_c[0])
        rank_c = int(np.sum(s_c > cutoff_c))
    return vh_c[rank_c:].T


def _build_projected_square_linear_map(coupled_map: LinearMap, selector: np.ndarray) -> LinearMap:
    """Return reduced square operator `S^T L S` as `LinearMap`."""
    selector_arr = np.asarray(selector, dtype=float)
    n_total, n_reduced = selector_arr.shape
    if n_reduced == n_total:
        return coupled_map

    def matvec(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced)
        return np.asarray(
            selector_arr.T @ np.asarray(coupled_map.matvec(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced)

    def rmatvec(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced)
        return np.asarray(
            selector_arr.T @ np.asarray(coupled_map.rmatvec(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced)

    def matmat(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced, -1)
        return np.asarray(
            selector_arr.T @ np.asarray(coupled_map.matmat(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced, -1)

    def rmatmat(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced, -1)
        return np.asarray(
            selector_arr.T @ np.asarray(coupled_map.rmatmat(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced, -1)

    reduced_to_dense = None
    if coupled_map._to_dense is not None:

        def reduced_to_dense() -> np.ndarray:
            dense = np.asarray(coupled_map.to_dense(), dtype=float)
            if dense.ndim != 2:
                dense = dense.reshape(n_total, n_total)
            return np.asarray(selector_arr.T @ dense @ selector_arr, dtype=float)

    return LinearMap(
        shape=(n_reduced, n_reduced),
        dtype=coupled_map.dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=reduced_to_dense,
        source=(coupled_map, selector_arr),
    )


def _build_reduced_coupled_system(
    *,
    coupled_operator: Any,
    n_scalar: int,
    apply_m_ind_gauge: bool,
    psi_gauge_row_builder: PsiGaugeRowBuilder,
    m_ind_gauge_row_builder: PsiGaugeRowBuilder,
    apply_psi_gauge: bool,
) -> ReducedCoupledSystem:
    """Build the shared gauge-reduced coupled system representation."""
    n_total = 2 * n_scalar
    coupled_map = _as_2d_coupled_linear_map(coupled_operator, n_total=n_total)
    selector = _build_coupled_selector(
        n_scalar=n_scalar,
        apply_m_ind_gauge=apply_m_ind_gauge,
        psi_gauge_row_builder=psi_gauge_row_builder,
        m_ind_gauge_row_builder=m_ind_gauge_row_builder,
        apply_psi_gauge=apply_psi_gauge,
    )
    reduced_operator = _build_projected_square_linear_map(coupled_map, selector)
    return ReducedCoupledSystem(
        full_operator=coupled_map,
        selector=np.asarray(selector, dtype=float),
        reduced_operator=reduced_operator,
    )


@dataclass
class CoupledSteadyStateSolver:
    """Solve steady-state coupled systems with optional gauge elimination."""

    n_scalar: int
    apply_m_ind_gauge: bool
    preconditioner_type: Optional[str]
    psi_gauge_row_builder: PsiGaugeRowBuilder
    m_ind_gauge_row_builder: PsiGaugeRowBuilder
    timed_solve: TimedSolveFn
    column_scale_cache: Optional[Dict[tuple[Any, ...], np.ndarray]] = None
    exact_column_scale_dim_limit: int = 2000
    solver_tolerance: float = 1e-13
    steady_state_regularization_lambda: float = 1e-10

    def solve(
        self,
        *,
        coupled_operator: Any,
        forcing_flat: np.ndarray,
        solver: str,
        preconditioner: Optional[LinearMap],
        apply_psi_gauge: bool,
        column_scale_cache_key: Optional[tuple[Any, ...]] = None,
    ) -> np.ndarray:
        """Solve `L y = -forcing` for flattened coupled state `y`."""
        reduced_system = _build_reduced_coupled_system(
            coupled_operator=coupled_operator,
            n_scalar=self.n_scalar,
            apply_m_ind_gauge=self.apply_m_ind_gauge,
            psi_gauge_row_builder=self.psi_gauge_row_builder,
            m_ind_gauge_row_builder=self.m_ind_gauge_row_builder,
            apply_psi_gauge=apply_psi_gauge,
        )
        n_total = reduced_system.n_total
        forcing = asarray(forcing_flat).reshape(n_total)

        coupled_map = reduced_system.full_operator
        selector = reduced_system.selector
        n_reduced = reduced_system.n_reduced
        if n_reduced == 0:
            return np.zeros((n_total,), dtype=np.asarray(forcing).dtype)

        operator_map = reduced_system.reduced_operator

        # Medium-term iterative path: solve the gauge-projected Tikhonov normal
        # equations. This provides an explicit branch selector for near-singular
        # coupled steady-state operators and avoids LSMR-specific conlim stops.
        if (
            solver in ("lsmr", "cgls")
            and preconditioner is None
            and self.preconditioner_type is None
        ):
            y_reduced = self._solve_projected_tikhonov_iterative(
                coupled_map=coupled_map,
                selector=selector,
                forcing=forcing,
                cache_key=(
                    (*column_scale_cache_key, "projected_tikhonov")
                    if column_scale_cache_key is not None
                    else None
                ),
            )
            if n_reduced == n_total:
                return asarray(y_reduced)
            return asarray(reduced_system.expand_vector(y_reduced))

        if solver in ("svd", "normal_eq"):
            y_reduced = self._solve_projected_tikhonov_dense(
                coupled_map=coupled_map,
                selector=selector,
                forcing=forcing,
                cache_key=(
                    (*column_scale_cache_key, "projected_tikhonov")
                    if column_scale_cache_key is not None
                    else None
                ),
            )
            if n_reduced == n_total:
                return asarray(y_reduced)
            return asarray(reduced_system.expand_vector(y_reduced))

        column_scale = self._maybe_get_exact_column_scale(
            operator_map=operator_map,
            solver=solver,
            preconditioner=preconditioner,
            cache_key=column_scale_cache_key,
        )

        problem = LeastSquaresProblem(
            A=[operator_map],
            solution_shape=(n_reduced,),
            data_shapes=[(n_total,)],
            column_scale=column_scale,
        )
        ls_solver = LeastSquaresSolver(solver=solver, preconditioner=self.preconditioner_type)

        preconditioner_to_use = preconditioner
        if preconditioner is None:
            preconditioner_to_use = ls_solver.build_preconditioner(
                problem=problem, preconditioner_type=self.preconditioner_type, num_scenarios=1
            )
        elif n_reduced != n_total:
            preconditioner_to_use = self._reduce_preconditioner(
                preconditioner=preconditioner,
                selector=selector,
                n_total=n_total,
                n_reduced=n_reduced,
            )

        y_reduced = asarray(
            self.timed_solve(
                "state.coupled_ss",
                ls_solver,
                problem,
                [-forcing],
                preconditioner=preconditioner_to_use,
            )
        ).reshape(-1)

        if n_reduced == n_total:
            return asarray(y_reduced)
        return asarray(reduced_system.expand_vector(y_reduced))

    def _solve_projected_tikhonov_iterative(
        self,
        *,
        coupled_map: LinearMap,
        selector: np.ndarray,
        forcing: np.ndarray,
        cache_key: Optional[tuple[Any, ...]],
    ) -> np.ndarray:
        """Solve projected coupled steady-state with explicit Tikhonov branch selection.

        Solves in reduced coordinates ``z``:
            min ||A z - b||^2 + lambda ||z||^2,
            A = S^T L S,  b = -S^T f
        using LSMR on the augmented least-squares system
            [A          ] z ~= [b]
            [sqrt(l) * I]      [0]
        This avoids the conditioning loss from CG on normal equations.
        """
        s, rhs, reg_lambda = self._prepare_projected_tikhonov_system(
            coupled_map=coupled_map, selector=selector, forcing=forcing
        )
        n_reduced = int(s.shape[1])
        op_proj = self._build_projected_square_operator(coupled_map=coupled_map, selector=s)

        inv_col_scale = self._get_projected_column_equilibration_inverse(
            coupled_map=coupled_map, selector=s, cache_key=cache_key, reg_lambda=reg_lambda
        )
        sqrt_reg = float(np.sqrt(max(reg_lambda, 0.0)))
        aug_rows = n_reduced + n_reduced

        def _aug_matvec(w: np.ndarray) -> np.ndarray:
            w_arr = np.asarray(w, dtype=float).reshape(n_reduced)
            z = w_arr if inv_col_scale is None else (inv_col_scale * w_arr)
            top = np.asarray(op_proj.matvec(z), dtype=float).reshape(n_reduced)
            if sqrt_reg > 0.0:
                bot = sqrt_reg * z
            else:
                bot = np.zeros_like(z)
            return np.concatenate([top, bot]).astype(float, copy=False)

        def _aug_rmatvec(v: np.ndarray) -> np.ndarray:
            v_arr = np.asarray(v, dtype=float).reshape(aug_rows)
            top = v_arr[:n_reduced]
            bot = v_arr[n_reduced:]
            out = np.asarray(op_proj.rmatvec(top), dtype=float).reshape(n_reduced)
            if sqrt_reg > 0.0:
                out = out + sqrt_reg * bot
            if inv_col_scale is not None:
                out = inv_col_scale * out
            return out.astype(float, copy=False)

        aug_op = ScipyLinearOperator(
            shape=(aug_rows, n_reduced), matvec=_aug_matvec, rmatvec=_aug_rmatvec, dtype=np.float64
        )
        aug_rhs = np.concatenate([rhs, np.zeros(n_reduced, dtype=float)])
        maxiter = max(1, 20 * n_reduced)
        # Use LSMR on the augmented Tikhonov system; this preserves the explicit
        # branch selector while avoiding normal-equation conditioning blow-up.
        lsmr_out = scipy_lsmr(
            aug_op, aug_rhs, atol=0.0, btol=float(self.solver_tolerance), maxiter=maxiter
        )
        w = np.asarray(lsmr_out[0], dtype=float).reshape(n_reduced)
        istop = int(lsmr_out[1])
        if istop not in (0, 1, 2, 4, 5):
            warnings.warn(
                f"Projected Tikhonov LSMR may not have converged (istop={istop}).", RuntimeWarning
            )
        if inv_col_scale is None:
            return w
        return inv_col_scale * w

    def _solve_projected_tikhonov_dense(
        self,
        *,
        coupled_map: LinearMap,
        selector: np.ndarray,
        forcing: np.ndarray,
        cache_key: Optional[tuple[Any, ...]],
    ) -> np.ndarray:
        """Dense projected Tikhonov solve matching the iterative branch selector."""
        s, rhs, reg_lambda = self._prepare_projected_tikhonov_system(
            coupled_map=coupled_map, selector=selector, forcing=forcing
        )
        n_reduced = int(s.shape[1])

        dense = np.asarray(coupled_map.to_dense(), dtype=float)
        if dense.ndim != 2:
            dense = dense.reshape(int(coupled_map.shape[0]), int(coupled_map.shape[1]))
        a = np.asarray(s.T @ dense @ s, dtype=float)
        inv_col_scale = self._get_projected_column_equilibration_inverse(
            coupled_map=coupled_map, selector=s, cache_key=cache_key, reg_lambda=reg_lambda
        )
        sqrt_reg = float(np.sqrt(max(reg_lambda, 0.0)))
        eye = np.eye(n_reduced, dtype=float)

        if inv_col_scale is None:
            a_aug = np.vstack([a, sqrt_reg * eye]) if sqrt_reg > 0.0 else a
            b_aug = (
                np.concatenate([rhs, np.zeros(n_reduced, dtype=float)]) if sqrt_reg > 0.0 else rhs
            )
            x, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=max(float(self.solver_tolerance), 1e-15))
            return np.asarray(x, dtype=float).reshape(n_reduced)

        d_inv = np.asarray(inv_col_scale, dtype=float).reshape(-1)
        a_aug_scaled = (
            np.vstack([a * d_inv[None, :], (sqrt_reg * d_inv)[:, None] * eye])
            if sqrt_reg > 0.0
            else (a * d_inv[None, :])
        )
        b_aug = np.concatenate([rhs, np.zeros(n_reduced, dtype=float)]) if sqrt_reg > 0.0 else rhs
        w, *_ = np.linalg.lstsq(
            a_aug_scaled, b_aug, rcond=max(float(self.solver_tolerance), 1e-15)
        )
        return d_inv * w

    def _prepare_projected_tikhonov_system(
        self, *, coupled_map: LinearMap, selector: np.ndarray, forcing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Prepare projected Tikhonov system components shared by dense/iterative solves."""
        del coupled_map  # Included for signature symmetry; no operator eval required here.
        s = np.asarray(selector, dtype=float)
        n_total = int(s.shape[0])
        rhs = np.asarray(-(s.T @ np.asarray(forcing).reshape(n_total)), dtype=float).reshape(-1)
        reg_lambda = float(self.steady_state_regularization_lambda)
        return s, rhs, reg_lambda

    def _build_projected_square_operator(
        self, *, coupled_map: LinearMap, selector: np.ndarray
    ) -> ScipyLinearOperator:
        """Build the projected square operator ``S^T L S`` as a SciPy LinearOperator."""
        s = np.asarray(selector, dtype=float)
        n_reduced = int(s.shape[1])

        def _proj_matvec(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float).reshape(n_reduced)
            return (
                np.asarray(s.T @ coupled_map.matvec(s @ x_arr), dtype=float)
                .reshape(n_reduced)
                .copy()
            )

        def _proj_rmatvec(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float).reshape(n_reduced)
            return (
                np.asarray(s.T @ coupled_map.rmatvec(s @ x_arr), dtype=float)
                .reshape(n_reduced)
                .copy()
            )

        return ScipyLinearOperator(
            shape=(n_reduced, n_reduced),
            matvec=_proj_matvec,
            rmatvec=_proj_rmatvec,
            dtype=np.float64,
        )

    def _get_projected_column_equilibration_inverse(
        self,
        *,
        coupled_map: LinearMap,
        selector: np.ndarray,
        cache_key: Optional[tuple[Any, ...]],
        reg_lambda: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Exact column equilibration for the projected square Tikhonov system."""
        n_reduced = int(selector.shape[1])
        if n_reduced <= 0 or n_reduced > int(self.exact_column_scale_dim_limit):
            return None

        if cache_key is not None and self.column_scale_cache is not None:
            cached = self.column_scale_cache.get(cache_key)
            if cached is not None and int(np.asarray(cached).size) == n_reduced:
                col = np.asarray(cached, dtype=float).reshape(-1)
                return self._column_scale_to_inverse(col)

        try:
            dense = np.asarray(coupled_map.to_dense(), dtype=float)
        except Exception:
            return None
        if dense.ndim != 2:
            dense = dense.reshape(int(coupled_map.shape[0]), int(coupled_map.shape[1]))
        s = np.asarray(selector, dtype=float)
        dense_proj = s.T @ dense @ s
        col_scale_sq = np.sum(dense_proj * dense_proj, axis=0, dtype=float)
        if reg_lambda > 0.0:
            col_scale_sq = col_scale_sq + float(reg_lambda)
        col_scale = np.sqrt(col_scale_sq).astype(float, copy=False)

        if cache_key is not None and self.column_scale_cache is not None:
            self.column_scale_cache[cache_key] = col_scale
        return self._column_scale_to_inverse(col_scale)

    @staticmethod
    def _column_scale_to_inverse(col_scale: np.ndarray) -> Optional[np.ndarray]:
        col = np.asarray(col_scale, dtype=float).reshape(-1)
        if col.size == 0:
            return None
        finite = np.isfinite(col)
        if not np.any(finite):
            return None
        max_col = float(np.max(np.abs(col[finite])))
        if not np.isfinite(max_col) or max_col <= 0.0:
            return None
        floor = np.sqrt(np.finfo(float).eps) * max_col
        denom = np.where(np.abs(col) > floor, col, 1.0)
        inv = 1.0 / denom
        inv[~finite] = 1.0
        return inv.astype(float, copy=False)

    def _maybe_get_exact_column_scale(
        self,
        *,
        operator_map: LinearMap,
        solver: str,
        preconditioner: Optional[LinearMap],
        cache_key: Optional[tuple[Any, ...]],
    ) -> Optional[np.ndarray]:
        """Return cached exact column norms for iterative coupled steady-state solves.

        This is a semantics-preserving right scaling (change of variables) used by
        iterative least-squares solvers. We intentionally skip it when a
        preconditioner is active because `LeastSquaresSolver` currently disables
        simultaneous right-scaling + preconditioning.
        """
        if solver not in ("lsmr", "cgls"):
            return None
        if preconditioner is not None or self.preconditioner_type is not None:
            return None

        n_cols = int(operator_map.shape[1])
        if n_cols <= 0 or n_cols > int(self.exact_column_scale_dim_limit):
            return None

        if cache_key is not None and self.column_scale_cache is not None:
            cached = self.column_scale_cache.get(cache_key)
            if cached is not None and int(np.asarray(cached).size) == n_cols:
                return np.asarray(cached, dtype=float).reshape(-1)

        try:
            dense = operator_map.to_dense()
        except Exception:
            return None

        dense_arr = np.asarray(dense)
        if dense_arr.ndim != 2:
            dense_arr = dense_arr.reshape(operator_map.shape[0], operator_map.shape[1])
        if dense_arr.shape[1] != n_cols:
            return None

        col_scale = np.linalg.norm(dense_arr, axis=0).astype(float, copy=False)
        if cache_key is not None and self.column_scale_cache is not None:
            self.column_scale_cache[cache_key] = col_scale
        return col_scale

    @staticmethod
    def _reduce_preconditioner(
        *, preconditioner: LinearMap, selector: np.ndarray, n_total: int, n_reduced: int
    ) -> LinearMap:
        """Map full-space preconditioner into reduced coordinates."""
        pre_map = as_linear_map(preconditioner)
        selector_arr = asarray(selector)

        if pre_map.shape == (n_reduced, n_reduced):
            return pre_map
        if pre_map.shape != (n_total, n_total):
            raise ValueError(
                f"Invalid coupled preconditioner shape {pre_map.shape}; "
                f"expected {(n_total, n_total)} or {(n_reduced, n_reduced)}."
            )

        def reduced_p_matvec(v: np.ndarray) -> np.ndarray:
            v_arr = asarray(v).reshape(-1)
            return selector_arr.T @ pre_map.matvec(selector_arr @ v_arr)

        def reduced_p_rmatvec(v: np.ndarray) -> np.ndarray:
            v_arr = asarray(v).reshape(-1)
            return selector_arr.T @ pre_map.rmatvec(selector_arr @ v_arr)

        return LinearMap(
            shape=(n_reduced, n_reduced),
            dtype=pre_map.dtype,
            _matvec=reduced_p_matvec,
            _rmatvec=reduced_p_rmatvec,
        )


class CoupledOperators:
    """Assemble and expose coupled full-induction operators for a ``State``."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._dt_psi_from_toroidal_rhs_dense_cache: Dict[bool, np.ndarray] = {}
        self._dt_psi_from_E_dense_cache: Dict[bool, np.ndarray] = {}
        self._dt_m_ind_from_E_dense_cache: Optional[np.ndarray] = None

    def build_coupled_preconditioner(self) -> Optional[LinearMap]:
        """Build the preconditioner for the coupled ``(2N, 2N)`` induction system."""
        st = self.state
        if st.preconditioner is None:
            return None

        n = st.solution_space.index_length
        l_tensor = st.coupled_induction_tensor
        l_map = as_linear_map(asarray(l_tensor).reshape(2 * n, 2 * n))
        problem = LeastSquaresProblem(A=[l_map], solution_shape=(2 * n,), data_shapes=[(2 * n,)])
        solver = LeastSquaresSolver(solver=st.solver_type, preconditioner=st.preconditioner)
        return solver.build_preconditioner(
            problem=problem, preconditioner_type=st.preconditioner, num_scenarios=1
        )

    def _dense_E_coeff_operator_matrix(self, op: Any) -> np.ndarray:
        """Return dense ``(2N, N)`` matrix for a coefficient->E operator."""
        st = self.state
        n = st.solution_space.index_length
        arr = np.asarray(to_dense(op))
        if arr.ndim == 3:
            arr = arr.reshape(2 * n, n)
        elif arr.ndim != 2:
            arr = arr.reshape(2 * n, n)
        return np.asarray(arr, dtype=float)

    def _get_dt_psi_from_E_dense(self, *, apply_psi_gauge: bool) -> np.ndarray:
        """Dense conductance-independent map ``E_coeffs -> dpsi/dt``."""
        cached = self._dt_psi_from_E_dense_cache.get(bool(apply_psi_gauge))
        if cached is not None:
            return cached
        st = self.state
        dt_psi_from_rhs = self._get_dt_psi_from_toroidal_rhs_dense(
            apply_psi_gauge=bool(apply_psi_gauge)
        )
        e_to_toroidal_rhs = np.asarray(
            to_dense(st.toroidal_matrices.toroidal_rhs_from_E_operator), dtype=float
        )
        dt_psi_from_E = np.asarray(dt_psi_from_rhs @ e_to_toroidal_rhs, dtype=float)
        self._dt_psi_from_E_dense_cache[bool(apply_psi_gauge)] = dt_psi_from_E
        return dt_psi_from_E

    def _get_dt_psi_from_toroidal_rhs_dense(self, *, apply_psi_gauge: bool) -> np.ndarray:
        """Dense conductance-independent map ``toroidal_rhs -> dpsi/dt``."""
        cached = self._dt_psi_from_toroidal_rhs_dense_cache.get(bool(apply_psi_gauge))
        if cached is not None:
            return cached

        st = self.state
        constraint_system = st.dt_alpha_constraint_system
        feedback_reg_lambda = self._toroidal_feedback_regularization_lambda()
        dt_psi_from_rhs = np.asarray(
            st.toroidal_matrices.build_dt_psi_from_toroidal_rhs_matrix(
                m_imp_to_jr_operator=st.poloidal_matrices.m_imp_to_jr,
                constraint_operator=constraint_system.hard_operator,
                weighting=st.toroidal_weighting,
                regularization_lambda=feedback_reg_lambda,
                penalty_operator=constraint_system.soft_operator,
                penalty_scaling=float(constraint_system.soft_scaling),
                hinv_rtol=0.0,
                apply_psi_gauge=apply_psi_gauge,
            ),
            dtype=float,
        )
        self._dt_psi_from_toroidal_rhs_dense_cache[bool(apply_psi_gauge)] = dt_psi_from_rhs
        return dt_psi_from_rhs

    def _get_dt_m_ind_from_E_dense(self) -> np.ndarray:
        """Dense conductance-independent map ``E_coeffs -> dm_ind/dt``."""
        if self._dt_m_ind_from_E_dense_cache is None:
            st = self.state
            scale = st.poloidal_matrices.E_df_to_d_m_ind_dt
            self._dt_m_ind_from_E_dense_cache = np.asarray(
                scale * np.asarray(st.E_coeffs_to_E_df_matrix), dtype=float
            )
        return self._dt_m_ind_from_E_dense_cache

    def _toroidal_feedback_regularization_lambda(self) -> float:
        """Regularization used when assembling the coupled linear feedback operator.

        The coupled operator should represent physical linear dynamics with hard
        gauge/constraint handling only. Keep Tikhonov regularization in the
        runtime RHS solve path to avoid suppressing true feedback blocks in
        the assembled operator.
        """
        st = self.state
        runtime_lambda = float(getattr(st, "toroidal_regularization_lambda", 0.0))
        if runtime_lambda <= 0.0:
            return 0.0
        # CS-dominant runs remain sensitive near Nyquist and require the runtime
        # stabilization level for robust integration.
        mode = getattr(st, "mode", None)
        if mode == SimulationMode.CS_DOMINANT:
            return runtime_lambda
        # Runtime forcing solves can require stronger damping for robust RHS
        # inversion. For assembled feedback operators, cap lambda to avoid
        # overwhelming physical toroidal feedback while retaining numerical
        # stabilization against near-null directions.
        return min(runtime_lambda, 1e-12)

    def _stabilize_poloidal_self_block(self, l11: np.ndarray) -> np.ndarray:
        """Enforce dissipative self-block semantics for CS-dominant full induction.

        The CS-dominant poloidal self block can contain a tiny anti-diffusive
        numerical mode at high spectral-to-grid ratios. We keep the skew part
        unchanged and clip only positive eigenvalues of the symmetric part so
        `Re(lambda) <= 0` by construction.
        """
        st = self.state
        mode = getattr(st, "mode", None)
        if mode != SimulationMode.CS_DOMINANT:
            return np.asarray(l11)
        if getattr(st, "dynamics_mode", "") != DynamicsMode.FULL_INDUCTION:
            return np.asarray(l11)

        a = np.asarray(l11, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            return a

        sym = 0.5 * (a + a.T)
        skew = a - sym
        evals, evecs = np.linalg.eigh(sym)
        evals_clipped = np.minimum(evals, 0.0)
        sym_stable = (evecs * evals_clipped) @ evecs.T
        return np.asarray(skew + sym_stable, dtype=float)

    def get_coupled_induction_tensor(self) -> np.ndarray:
        """Build the coupled tensor ``L_coupled`` with shape ``(2, N, 2, N)``."""
        st = self.state
        n = st.solution_space.index_length
        apply_psi_gauge = bool(st.apply_psi_gauge)

        dt_psi_from_E = self._get_dt_psi_from_E_dense(apply_psi_gauge=apply_psi_gauge)
        toroidal_to_E = self._dense_E_coeff_operator_matrix(st.toroidal_to_E_coeffs)
        mind_to_E = self._dense_E_coeff_operator_matrix(st.m_ind_to_E_coeffs)

        dt_psi_from_psi = asarray(dt_psi_from_E @ toroidal_to_E)
        dt_psi_from_m_ind = np.asarray(dt_psi_from_E @ mind_to_E, dtype=float)

        dt_m_ind_from_E = self._get_dt_m_ind_from_E_dense()
        dt_m_ind_from_psi = asarray(dt_m_ind_from_E @ toroidal_to_E)
        dt_m_ind_from_m_ind = asarray(dt_m_ind_from_E @ mind_to_E)
        dt_m_ind_from_m_ind = self._stabilize_poloidal_self_block(dt_m_ind_from_m_ind)
        dt_psi_from_m_ind = asarray(dt_psi_from_m_ind)

        top_row = xp.stack([dt_psi_from_psi, dt_psi_from_m_ind], axis=1)
        bottom_row = xp.stack([dt_m_ind_from_psi, dt_m_ind_from_m_ind], axis=1)
        l_coupled = xp.stack([top_row, bottom_row], axis=0)
        try:
            l_flat = np.asarray(l_coupled, dtype=float).reshape(2 * n, 2 * n)
            st.diagnostics.analyze_coupled_stability(
                l_flat, label=f"tensor:psi_gauge={int(apply_psi_gauge)}"
            )
        except Exception as exc:
            logger.debug("Coupled stability diagnostic skipped: %s", exc)
        return l_coupled

    def get_coupled_induction_operator(
        self,
        dt_psi_from_psi: Any = None,
        dt_psi_from_m_ind: Any = None,
        dt_m_ind_from_psi: Any = None,
        dt_m_ind_from_m_ind: Any = None,
        matrix_free: bool = False,
        solver: str = "lsmr",
    ) -> LinearMap:
        """Build matrix-free/dense coupled operator for ``y=[psi, m_ind]`` dynamics."""
        from pynamit.simulation.induction.operators import BlockCoupledOperator

        st = self.state
        n = st.solution_space.index_length
        apply_psi_gauge = bool(st.apply_psi_gauge)
        if st.dense_full_operators and matrix_free:
            matrix_free = False

        if dt_psi_from_psi is None or dt_psi_from_m_ind is None:
            psi_to_E_coeffs = st.toroidal_to_E_coeffs
            mind_to_E_coeffs = st.m_ind_to_E_coeffs
            if not matrix_free:
                psi_to_E_coeffs = self._dense_E_coeff_operator_matrix(psi_to_E_coeffs)
                mind_to_E_coeffs = self._dense_E_coeff_operator_matrix(mind_to_E_coeffs)

            dt_psi_from_E_dense = self._get_dt_psi_from_E_dense(apply_psi_gauge=apply_psi_gauge)
            dt_psi_from_E_map = as_linear_map(dt_psi_from_E_dense)

            if dt_psi_from_psi is None:
                if matrix_free:
                    dt_psi_from_psi = dt_psi_from_E_map @ as_linear_map(psi_to_E_coeffs)
                else:
                    dt_psi_from_psi = dt_psi_from_E_dense @ np.asarray(
                        psi_to_E_coeffs, dtype=float
                    )

            if dt_psi_from_m_ind is None:
                if matrix_free:
                    dt_psi_from_m_ind = dt_psi_from_E_map @ as_linear_map(mind_to_E_coeffs)
                else:
                    dt_psi_from_m_ind = dt_psi_from_E_dense @ np.asarray(
                        mind_to_E_coeffs, dtype=float
                    )

        if dt_m_ind_from_psi is None:
            dt_m_ind_from_E = self._get_dt_m_ind_from_E_dense()
            toroidal_to_E = self._dense_E_coeff_operator_matrix(st.toroidal_to_E_coeffs)
            dt_m_ind_from_psi = np.asarray(dt_m_ind_from_E @ toroidal_to_E, dtype=float)

        if dt_m_ind_from_m_ind is None:
            dt_m_ind_from_E = self._get_dt_m_ind_from_E_dense()
            mind_to_E = self._dense_E_coeff_operator_matrix(st.m_ind_to_E_coeffs)
            dt_m_ind_from_m_ind = np.asarray(dt_m_ind_from_E @ mind_to_E, dtype=float)
            dt_m_ind_from_m_ind = self._stabilize_poloidal_self_block(dt_m_ind_from_m_ind)

        block_op = BlockCoupledOperator(
            L_00=dt_psi_from_psi,
            L_01=dt_psi_from_m_ind,
            L_10=dt_m_ind_from_psi,
            L_11=dt_m_ind_from_m_ind,
            n=n,
        )
        return block_op.to_linear_map()

    @staticmethod
    def _densify_linear_operator(operator: Any, n_total: int) -> np.ndarray:
        """Convert linear operator to dense ``(2N, 2N)`` with safe fallback."""
        lm = as_linear_map(operator)
        try:
            dense = asarray(lm.to_dense())
            return dense.reshape(n_total, n_total)
        except Exception:
            eye = np.eye(n_total, dtype=float)
            cols = [asarray(lm.matvec(eye[:, i])) for i in range(n_total)]
            return np.column_stack(cols)

    def get_coupled_induction_matrix(
        self, source: Literal["dense", "sparse", "auto"] = "auto", flatten: bool = True
    ) -> np.ndarray:
        """Expose coupled operator matrix in dense form."""
        st = self.state
        n = st.solution_space.index_length
        n_total = 2 * n

        chosen = source
        if chosen == "auto":
            chosen = "dense" if st.dense_full_operators else "sparse"

        if chosen == "dense":
            l4 = asarray(st.coupled_induction_tensor)
            return l4.reshape(n_total, n_total) if flatten else l4

        if chosen == "sparse":
            op = st.coupled_induction_operator_sparse
            l2 = self._densify_linear_operator(op, n_total=n_total)
            return l2 if flatten else l2.reshape(2, n, 2, n)

        raise ValueError(f"Unknown source={source!r}; expected 'dense', 'sparse', or 'auto'.")

    def get_coupled_induction_blocks(
        self, source: Literal["dense", "sparse", "auto"] = "auto"
    ) -> Dict[str, np.ndarray]:
        """Expose coupled block matrices keyed by physical role."""
        l4 = self.get_coupled_induction_matrix(source=source, flatten=False)
        return {
            "dt_psi_from_psi": asarray(l4[0, :, 0, :]),
            "dt_psi_from_m_ind": asarray(l4[0, :, 1, :]),
            "dt_m_ind_from_psi": asarray(l4[1, :, 0, :]),
            "dt_m_ind_from_m_ind": asarray(l4[1, :, 1, :]),
        }

    def get_coupled_operator_for_steady_state(self, *, solver: Optional[str] = None) -> Any:
        """Return coupled operator used by steady-state coupled solve."""
        st = self.state
        if solver is None:
            solver = st.solver_type

        n_total = 2 * st.solution_space.index_length
        use_dense = (
            st.dense_full_operators
            or (st.integrator == IntegratorKind.EXPONENTIAL)
            or (n_total <= 600)
        )
        if use_dense:
            return st.coupled_induction_tensor

        matrix_free_solver = solver if solver in ("lsmr", "cgls") else "lsmr"
        return self.get_coupled_induction_operator(matrix_free=True, solver=matrix_free_solver)

    def get_coupled_operator_for_time_integration(
        self, *, use_dense: Optional[bool] = None
    ) -> Any:
        """Return coupled operator used by non-exponential full-induction stepping."""
        st = self.state
        if use_dense is None:
            use_dense = bool(st.dense_full_operators)

        if use_dense:
            return st.coupled_induction_tensor

        return st.coupled_induction_operator_sparse

    def get_coupled_reduced_time_integration_system(
        self, *, use_dense: Optional[bool] = None
    ) -> ReducedCoupledSystem:
        """Return the gauge-reduced coupled system used by runtime time stepping."""
        st = self.state
        coupled_operator = self.get_coupled_operator_for_time_integration(use_dense=use_dense)
        return _build_reduced_coupled_system(
            coupled_operator=coupled_operator,
            n_scalar=st.solution_space.index_length,
            apply_m_ind_gauge=st.apply_m_ind_gauge,
            psi_gauge_row_builder=st.constraints.get_psi_gauge_row,
            m_ind_gauge_row_builder=st.constraints.get_m_ind_gauge_row,
            apply_psi_gauge=bool(st.apply_psi_gauge),
        )

    def get_hl_projection_matrix(self, n_coeffs: int) -> np.ndarray:
        """Return the full-space HL projector induced by the reduced ``m_imp`` system."""
        hl_projection = self.state.constraints.get_hl_projection_matrix(n_coeffs)
        feedback_system = self.state.m_imp_feedback_system
        if feedback_system.full_size == int(n_coeffs):
            return feedback_system.get_hl_projection_full(hl_projection)
        return hl_projection

    def get_m_imp_from_jr_matrix(self, input_basis: Optional[Any] = None) -> np.ndarray:
        """Expose dense map from input ``jr`` coefficients to imposed ``m_imp``."""
        st = self.state
        feedback_system = st.m_imp_feedback_system
        if input_basis is None and st.jr is not None:
            input_basis = st.jr.spec

        op_rhs = as_linear_map(st.geometry.get_constraint_scalar_operator(input_basis))
        rhs0 = np.asarray(to_dense(op_rhs))
        n_scenarios = int(rhs0.shape[1]) if rhs0.ndim == 2 else 1
        rhs_terms = [rhs0]
        for term_index in range(1, feedback_system.problem.num_data_terms):
            n_rows = int(feedback_system.problem.A[term_index].num_rows)
            rhs_terms.append(np.zeros((n_rows, n_scenarios), dtype=rhs0.dtype))

        h_mat, _ = feedback_system.problem.get_normal_equation_components(data_term_index=0)
        h_shape = tuple(np.asarray(h_mat).shape)
        dim_max = max(int(h_shape[0]), int(h_shape[1])) if len(h_shape) >= 2 else 1
        rcond = float(np.finfo(float).eps * max(dim_max, 1))
        solver = LeastSquaresSolver(solver="svd", tolerance=max(float(rcond), 1e-15))
        m_imp_from_jr_reduced = np.asarray(
            solver.solve(feedback_system.problem, rhs_terms)
        ).reshape(feedback_system.problem.solution_size, n_scenarios)
        if st.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            hl_projection = st.constraints.get_hl_projection_matrix(feedback_system.full_size)
            m_imp_from_jr_reduced = np.asarray(
                feedback_system.project_hl(m_imp_from_jr_reduced, hl_projection), dtype=float
            )
        m_imp_from_jr = np.asarray(feedback_system.expand_solution(m_imp_from_jr_reduced))
        return asarray(m_imp_from_jr)

    def get_external_forcing_matrices(
        self, input_basis_jr: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """Expose dense rate maps from ``u`` and ``jr`` into the coupled system."""
        st = self.state
        n = st.solution_space.index_length
        n2 = 2 * n
        feedback_reg_lambda = self._toroidal_feedback_regularization_lambda()

        dt_psi_from_E = np.asarray(
            self._get_dt_psi_from_E_dense(apply_psi_gauge=bool(st.apply_psi_gauge)), dtype=float
        )
        dt_m_ind_from_E = np.asarray(self._get_dt_m_ind_from_E_dense(), dtype=float)

        e_from_u = np.asarray(to_dense(as_linear_map(st.u_coeffs_to_E_coeffs)))
        m_imp_from_jr = np.asarray(self.get_m_imp_from_jr_matrix(input_basis=input_basis_jr))
        e_from_jr = np.asarray(to_dense(as_linear_map(st.m_imp_to_E_coeffs))) @ m_imp_from_jr

        return {
            "dt_psi_from_u": asarray(dt_psi_from_E @ e_from_u),
            "dt_psi_from_jr": asarray(dt_psi_from_E @ e_from_jr),
            "dt_m_ind_from_u": asarray(dt_m_ind_from_E @ e_from_u),
            "dt_m_ind_from_jr": asarray(dt_m_ind_from_E @ e_from_jr),
            "dt_psi_from_E": asarray(dt_psi_from_E),
            "dt_m_ind_from_E": asarray(dt_m_ind_from_E),
            "E_from_u": asarray(e_from_u),
            "E_from_jr": asarray(e_from_jr),
            "m_imp_from_jr": asarray(m_imp_from_jr),
        }
