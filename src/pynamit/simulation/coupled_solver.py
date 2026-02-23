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
from scipy.sparse.linalg import (
    lsmr as scipy_lsmr,
    LinearOperator as ScipyLinearOperator,
)

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.simulation.geometry_utils import to_dense
from pynamit.utils import asarray, xp

TimedSolveFn = Callable[..., np.ndarray]
PsiGaugeRowBuilder = Callable[[int], np.ndarray]
logger = logging.getLogger(__name__)


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
        use_pinning: bool,
        column_scale_cache_key: Optional[tuple[Any, ...]] = None,
    ) -> np.ndarray:
        """Solve `L y = -forcing` for flattened coupled state `y`."""
        n_total = 2 * self.n_scalar
        forcing = asarray(forcing_flat).reshape(n_total)

        coupled_map = self._as_2d_linear_map(coupled_operator, n_total=n_total)
        selector = self._build_selector(use_pinning=use_pinning)

        n_reduced = int(selector.shape[1])
        if n_reduced == 0:
            return np.zeros((n_total,), dtype=np.asarray(forcing).dtype)

        if n_reduced != n_total:
            operator_map = self._build_reduced_operator(coupled_map, selector)
        else:
            operator_map = coupled_map

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
            return asarray(selector @ y_reduced)

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
            return asarray(selector @ y_reduced)

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
        ls_solver = LeastSquaresSolver(
            solver=solver,
            preconditioner=self.preconditioner_type,
        )

        preconditioner_to_use = preconditioner
        if preconditioner is None:
            preconditioner_to_use = ls_solver.build_preconditioner(
                problem=problem,
                preconditioner_type=self.preconditioner_type,
                num_scenarios=1,
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
        return asarray(selector @ y_reduced)

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
        s = np.asarray(selector, dtype=float)
        n_total, n_reduced = int(s.shape[0]), int(s.shape[1])
        rhs = np.asarray(-(s.T @ np.asarray(forcing).reshape(n_total)), dtype=float).reshape(-1)
        reg_lambda = float(self.steady_state_regularization_lambda)

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

        op_proj: ScipyLinearOperator = ScipyLinearOperator(
            shape=(n_reduced, n_reduced),
            matvec=_proj_matvec,
            rmatvec=_proj_rmatvec,
            dtype=np.float64,
        )

        inv_col_scale = self._get_projected_column_equilibration_inverse(
            coupled_map=coupled_map,
            selector=s,
            cache_key=cache_key,
            reg_lambda=reg_lambda,
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
            shape=(aug_rows, n_reduced),
            matvec=_aug_matvec,
            rmatvec=_aug_rmatvec,
            dtype=np.float64,
        )
        aug_rhs = np.concatenate([rhs, np.zeros(n_reduced, dtype=float)])
        maxiter = max(1, 20 * n_reduced)
        # Use LSMR on the augmented Tikhonov system; this preserves the explicit
        # branch selector while avoiding normal-equation conditioning blow-up.
        lsmr_out = scipy_lsmr(
            aug_op,
            aug_rhs,
            atol=0.0,
            btol=float(self.solver_tolerance),
            maxiter=maxiter,
        )
        w = np.asarray(lsmr_out[0], dtype=float).reshape(n_reduced)
        istop = int(lsmr_out[1])
        if istop not in (0, 1, 2, 4, 5):
            warnings.warn(
                f"Projected Tikhonov LSMR may not have converged (istop={istop}).",
                RuntimeWarning,
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
        s = np.asarray(selector, dtype=float)
        n_total, n_reduced = int(s.shape[0]), int(s.shape[1])
        rhs = np.asarray(-(s.T @ np.asarray(forcing).reshape(n_total)), dtype=float).reshape(-1)
        reg_lambda = float(self.steady_state_regularization_lambda)

        dense = np.asarray(coupled_map.to_dense(), dtype=float)
        if dense.ndim != 2:
            dense = dense.reshape(int(coupled_map.shape[0]), int(coupled_map.shape[1]))
        a = np.asarray(s.T @ dense @ s, dtype=float)
        inv_col_scale = self._get_projected_column_equilibration_inverse(
            coupled_map=coupled_map,
            selector=s,
            cache_key=cache_key,
            reg_lambda=reg_lambda,
        )
        sqrt_reg = float(np.sqrt(max(reg_lambda, 0.0)))
        eye = np.eye(n_reduced, dtype=float)

        if inv_col_scale is None:
            a_aug = np.vstack([a, sqrt_reg * eye]) if sqrt_reg > 0.0 else a
            b_aug = (
                np.concatenate([rhs, np.zeros(n_reduced, dtype=float)])
                if sqrt_reg > 0.0
                else rhs
            )
            x, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=max(float(self.solver_tolerance), 1e-15))
            return np.asarray(x, dtype=float).reshape(n_reduced)

        d_inv = np.asarray(inv_col_scale, dtype=float).reshape(-1)
        a_aug_scaled = np.vstack(
            [
                a * d_inv[None, :],
                (sqrt_reg * d_inv)[:, None] * eye,
            ]
        ) if sqrt_reg > 0.0 else (a * d_inv[None, :])
        b_aug = (
            np.concatenate([rhs, np.zeros(n_reduced, dtype=float)])
            if sqrt_reg > 0.0
            else rhs
        )
        w, *_ = np.linalg.lstsq(
            a_aug_scaled,
            b_aug,
            rcond=max(float(self.solver_tolerance), 1e-15),
        )
        return d_inv * w

    def _get_projected_column_equilibration_inverse(
        self,
        *,
        coupled_map: LinearMap,
        selector: np.ndarray,
        cache_key: Optional[tuple[Any, ...]],
        reg_lambda: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Exact column equilibration for projected square GMRES system."""
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

    def _as_2d_linear_map(self, coupled_operator: Any, *, n_total: int) -> LinearMap:
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

    def _build_selector(self, *, use_pinning: bool) -> np.ndarray:
        """Build variable-elimination selector for gauge-constrained solve."""
        n = self.n_scalar
        n_total = 2 * n

        psi_row = None
        m_ind_row = None
        use_index_selector = False
        fixed_indices: list[int] = []

        if use_pinning and n > 0:
            psi_row = np.asarray(self.psi_gauge_row_builder(n), dtype=float)
            if psi_row.ndim == 1:
                psi_row = psi_row.reshape(1, -1)
            if psi_row.shape == (1, n):
                pin_row = np.zeros((1, n), dtype=float)
                pin_row[0, 0] = 1.0
                if float(np.linalg.norm(psi_row - pin_row)) <= 1e-12:
                    use_index_selector = True
                    fixed_indices.append(0)

        if self.apply_m_ind_gauge and n_total > n:
            m_ind_row = np.asarray(self.m_ind_gauge_row_builder(n), dtype=float)
            if m_ind_row.ndim == 1:
                m_ind_row = m_ind_row.reshape(1, -1)
            if m_ind_row.shape == (1, n):
                pin_row = np.zeros((1, n), dtype=float)
                pin_row[0, 0] = 1.0
                is_pin_row = float(np.linalg.norm(m_ind_row - pin_row)) <= 1e-12
                if is_pin_row and (use_index_selector or not use_pinning):
                    fixed_indices.append(n)

        if use_index_selector or (not use_pinning and self.apply_m_ind_gauge and n_total > n):
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
        if use_pinning and psi_row is not None and psi_row.shape[1] == n and psi_row.shape[0] > 0:
            c_psi = np.zeros((psi_row.shape[0], n_total), dtype=float)
            c_psi[:, :n] = psi_row
            constraint_rows.append(c_psi)
        if (
            self.apply_m_ind_gauge
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

    @staticmethod
    def _build_reduced_operator(operator_map: LinearMap, selector: np.ndarray) -> LinearMap:
        """Build reduced operator `A_red = A @ selector` as `LinearMap`."""
        selector_arr = asarray(selector)
        n_total = int(operator_map.shape[0])
        n_reduced = int(selector.shape[1])

        def reduced_matvec(x: np.ndarray) -> np.ndarray:
            x_arr = asarray(x).reshape(-1)
            return operator_map.matvec(selector_arr @ x_arr)

        def reduced_rmatvec(y: np.ndarray) -> np.ndarray:
            y_arr = asarray(y).reshape(-1)
            return selector_arr.T @ operator_map.rmatvec(y_arr)

        reduced_to_dense = None
        if operator_map._to_dense is not None:

            def reduced_to_dense() -> np.ndarray:
                dense = asarray(operator_map.to_dense())
                if dense.ndim != 2:
                    dense = dense.reshape(n_total, n_total)
                return dense @ selector

        return LinearMap(
            shape=(n_total, n_reduced),
            dtype=operator_map.dtype,
            _matvec=reduced_matvec,
            _rmatvec=reduced_rmatvec,
            _to_dense=reduced_to_dense,
            source=operator_map,
        )

    @staticmethod
    def _reduce_preconditioner(
        *,
        preconditioner: LinearMap,
        selector: np.ndarray,
        n_total: int,
        n_reduced: int,
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


class CoupledOperatorAPI:
    """Assemble and expose coupled full-induction operators for a ``State``."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._dtpsi_from_E_dense_cache: Dict[bool, np.ndarray] = {}
        self._dmind_from_E_dense_cache: Optional[np.ndarray] = None

    def _dense_E_coeff_operator_matrix(self, op: Any) -> np.ndarray:
        """Return dense ``(2N, N)`` matrix for a coefficient->E operator."""
        st = self.state
        n = st.solution_basis.index_length
        arr = np.asarray(to_dense(op))
        if arr.ndim == 3:
            arr = arr.reshape(2 * n, n)
        elif arr.ndim != 2:
            arr = arr.reshape(2 * n, n)
        return np.asarray(arr, dtype=float)

    def _get_dtpsi_from_E_dense(self, *, use_pinning: bool) -> np.ndarray:
        """Dense conductance-independent map ``E_coeffs -> dpsi/dt``."""
        cached = self._dtpsi_from_E_dense_cache.get(bool(use_pinning))
        if cached is not None:
            return cached

        st = self.state
        n = st.solution_basis.index_length
        constraint_op = st.constraints.induction_constraint_operator_hard
        feedback_reg_lambda = self._toroidal_feedback_regularization_lambda()
        dtpsi_from_E = np.asarray(
            st.toroidal_matrices.build_psi_dynamics_matrix(
                psi_to_E_operator=np.eye(2 * n, dtype=float),
                m_imp_to_jr_operator=st.poloidal_matrices.m_imp_to_jr,
                jr_map_operator=constraint_op,
                weighting=st.toroidal_weighting,
                regularization_lambda=feedback_reg_lambda,
                penalty_operator=None,
                penalty_scaling=0.0,
                hinv_rtol=0.0,
                use_pinning=use_pinning,
            ),
            dtype=float,
        )
        self._dtpsi_from_E_dense_cache[bool(use_pinning)] = dtpsi_from_E
        return dtpsi_from_E

    def _get_dmind_from_E_dense(self) -> np.ndarray:
        """Dense conductance-independent map ``E_coeffs -> dm_ind/dt``."""
        if self._dmind_from_E_dense_cache is None:
            st = self.state
            scale = st.poloidal_matrices.E_df_to_d_m_ind_dt
            self._dmind_from_E_dense_cache = np.asarray(
                scale * np.asarray(st.E_coeffs_to_E_df_matrix),
                dtype=float,
            )
        return self._dmind_from_E_dense_cache

    def _toroidal_feedback_regularization_lambda(self) -> float:
        """Regularization used when assembling the coupled linear feedback operator.

        The coupled operator should represent physical linear dynamics with hard
        gauge/constraint handling only. Keep Tikhonov regularization in the
        runtime forcing solve path to avoid suppressing true feedback blocks in
        the assembled operator.
        """
        st = self.state
        runtime_lambda = float(getattr(st, "toroidal_regularization_lambda", 0.0))
        if runtime_lambda <= 0.0:
            return 0.0
        # CS-dominant runs remain sensitive near Nyquist and require the runtime
        # stabilization level for robust integration.
        mode = getattr(st, "mode", None)
        if str(mode).lower().endswith("cs_dominant"):
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
        if not str(mode).lower().endswith("cs_dominant"):
            return np.asarray(l11)
        if getattr(st, "dynamics_mode", "") != "full_induction":
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

    def get_coupled_induction_tensor(self, use_pinning: Optional[bool] = None) -> np.ndarray:
        """Build the coupled tensor ``L_coupled`` with shape ``(2, N, 2, N)``."""
        st = self.state
        n = st.solution_basis.index_length
        if use_pinning is None:
            use_pinning = st.apply_psi_gauge

        dtpsi_from_E = self._get_dtpsi_from_E_dense(use_pinning=bool(use_pinning))
        toroidal_to_E = self._dense_E_coeff_operator_matrix(st.toroidal_to_E_coeffs)
        mind_to_E = self._dense_E_coeff_operator_matrix(st.m_ind_to_E_coeffs)

        dtpsi_from_psi = asarray(dtpsi_from_E @ toroidal_to_E)
        dtpsi_from_mind = asarray(dtpsi_from_E @ mind_to_E)

        dmind_from_E = self._get_dmind_from_E_dense()
        dmind_from_psi = asarray(dmind_from_E @ toroidal_to_E)
        dmind_from_mind = asarray(dmind_from_E @ mind_to_E)
        dmind_from_mind = self._stabilize_poloidal_self_block(dmind_from_mind)

        top_row = xp.stack([dtpsi_from_psi, dtpsi_from_mind], axis=1)
        bottom_row = xp.stack([dmind_from_psi, dmind_from_mind], axis=1)
        l_coupled = xp.stack([top_row, bottom_row], axis=0)
        try:
            l_flat = np.asarray(l_coupled, dtype=float).reshape(2 * n, 2 * n)
            st._analyze_coupled_stability(
                l_flat,
                label=f"tensor:pinning={int(bool(use_pinning))}",
            )
        except Exception as exc:
            logger.debug("Coupled stability diagnostic skipped: %s", exc)
        return l_coupled

    def get_coupled_induction_operator(
        self,
        dtpsi_from_psi: Any = None,
        dtpsi_from_mind: Any = None,
        dmind_from_psi: Any = None,
        dmind_from_mind: Any = None,
        matrix_free: bool = False,
        solver: str = "lsmr",
        use_pinning: Optional[bool] = None,
    ) -> LinearMap:
        """Build matrix-free/dense coupled operator for ``y=[psi, m_ind]`` dynamics."""
        from pynamit.simulation.operators import BlockCoupledOperator

        st = self.state
        n = st.solution_basis.index_length
        if use_pinning is None:
            use_pinning = st.apply_psi_gauge
        if st.dense_full_operators and matrix_free:
            matrix_free = False

        if dtpsi_from_psi is None or dtpsi_from_mind is None:
            psi_to_E_coeffs = st.toroidal_to_E_coeffs
            mind_to_E_coeffs = st.m_ind_to_E_coeffs
            if not matrix_free:
                psi_to_E_coeffs = self._dense_E_coeff_operator_matrix(psi_to_E_coeffs)
                mind_to_E_coeffs = self._dense_E_coeff_operator_matrix(mind_to_E_coeffs)

            dtpsi_from_E_dense = self._get_dtpsi_from_E_dense(use_pinning=bool(use_pinning))
            dtpsi_from_E_map = as_linear_map(dtpsi_from_E_dense)

            if dtpsi_from_psi is None:
                if matrix_free:
                    dtpsi_from_psi = dtpsi_from_E_map @ as_linear_map(psi_to_E_coeffs)
                else:
                    dtpsi_from_psi = dtpsi_from_E_dense @ np.asarray(psi_to_E_coeffs, dtype=float)

            if dtpsi_from_mind is None:
                if matrix_free:
                    dtpsi_from_mind = dtpsi_from_E_map @ as_linear_map(mind_to_E_coeffs)
                else:
                    dtpsi_from_mind = dtpsi_from_E_dense @ np.asarray(mind_to_E_coeffs, dtype=float)

        if dmind_from_psi is None:
            dmind_from_E = self._get_dmind_from_E_dense()
            toroidal_to_E = self._dense_E_coeff_operator_matrix(st.toroidal_to_E_coeffs)
            dmind_from_psi = np.asarray(dmind_from_E @ toroidal_to_E, dtype=float)

        if dmind_from_mind is None:
            dmind_from_E = self._get_dmind_from_E_dense()
            mind_to_E = self._dense_E_coeff_operator_matrix(st.m_ind_to_E_coeffs)
            dmind_from_mind = np.asarray(dmind_from_E @ mind_to_E, dtype=float)
            dmind_from_mind = self._stabilize_poloidal_self_block(dmind_from_mind)

        block_op = BlockCoupledOperator(
            L_00=dtpsi_from_psi,
            L_01=dtpsi_from_mind,
            L_10=dmind_from_psi,
            L_11=dmind_from_mind,
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
        self,
        source: Literal["dense", "sparse", "auto"] = "auto",
        flatten: bool = True,
        use_pinning: Optional[bool] = None,
    ) -> np.ndarray:
        """Expose coupled operator matrix in dense form."""
        st = self.state
        if use_pinning is None:
            use_pinning = st.apply_psi_gauge
        n = st.solution_basis.index_length
        n_total = 2 * n

        chosen = source
        if chosen == "auto":
            chosen = "dense" if st.dense_full_operators else "sparse"

        if chosen == "dense":
            if use_pinning == st.apply_psi_gauge:
                l4 = asarray(st.coupled_induction_tensor)
            else:
                l4 = asarray(self.get_coupled_induction_tensor(use_pinning=use_pinning))
            return l4.reshape(n_total, n_total) if flatten else l4

        if chosen == "sparse":
            if use_pinning == st.apply_psi_gauge:
                op = st.coupled_induction_operator_sparse
            else:
                solver = st.solver_type if st.solver_type in ("lsmr", "cgls") else "lsmr"
                op = self.get_coupled_induction_operator(
                    matrix_free=True,
                    solver=solver,
                    use_pinning=use_pinning,
                )
            l2 = self._densify_linear_operator(op, n_total=n_total)
            return l2 if flatten else l2.reshape(2, n, 2, n)

        raise ValueError(f"Unknown source={source!r}; expected 'dense', 'sparse', or 'auto'.")

    def get_coupled_induction_blocks(
        self,
        source: Literal["dense", "sparse", "auto"] = "auto",
        use_pinning: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """Expose coupled block matrices keyed by physical role."""
        l4 = self.get_coupled_induction_matrix(
            source=source,
            flatten=False,
            use_pinning=use_pinning,
        )
        return {
            "dtpsi_from_psi": asarray(l4[0, :, 0, :]),
            "dtpsi_from_mind": asarray(l4[0, :, 1, :]),
            "dmind_from_psi": asarray(l4[1, :, 0, :]),
            "dmind_from_mind": asarray(l4[1, :, 1, :]),
        }

    def get_coupled_operator_for_steady_state(
        self,
        *,
        solver: Optional[str] = None,
        use_pinning: Optional[bool] = None,
    ) -> Any:
        """Return coupled operator used by steady-state coupled solve."""
        st = self.state
        if solver is None:
            solver = st.solver_type
        if use_pinning is None:
            use_pinning = st.apply_psi_gauge

        n_total = 2 * st.solution_basis.index_length
        use_dense = st.dense_full_operators or (st.integrator == "exponential") or (n_total <= 600)
        if use_dense:
            if use_pinning == st.apply_psi_gauge:
                return st.coupled_induction_tensor
            return self.get_coupled_induction_tensor(use_pinning=use_pinning)

        matrix_free_solver = solver if solver in ("lsmr", "cgls") else "lsmr"
        return self.get_coupled_induction_operator(
            matrix_free=True,
            solver=matrix_free_solver,
            use_pinning=use_pinning,
        )

    def get_coupled_operator_for_time_integration(
        self,
        *,
        use_dense: Optional[bool] = None,
        use_pinning: Optional[bool] = None,
    ) -> Any:
        """Return coupled operator used by non-exponential full-induction stepping."""
        st = self.state
        if use_dense is None:
            use_dense = bool(st.dense_full_operators)
        if use_pinning is None:
            use_pinning = st.apply_psi_gauge

        if use_dense:
            if use_pinning == st.apply_psi_gauge:
                return st.coupled_induction_tensor
            return self.get_coupled_induction_tensor(use_pinning=use_pinning)

        if use_pinning == st.apply_psi_gauge:
            return st.coupled_induction_operator_sparse

        solver = st.solver_type if st.solver_type in ("lsmr", "cgls") else "lsmr"
        return self.get_coupled_induction_operator(
            matrix_free=True,
            solver=solver,
            use_pinning=use_pinning,
        )

    def get_hl_projection_matrix(self, n_coeffs: int) -> np.ndarray:
        """Return dense projector used by high-lat mode projection."""
        bundle = self.state.constraints.induction_constraint_bundle_hard
        if bundle is None:
            return np.eye(n_coeffs, dtype=float)
        q_hl = np.asarray(bundle.get("Q_hl", np.zeros((n_coeffs, 0), dtype=float)))
        m_metric = np.asarray(bundle.get("Q_metric", np.eye(n_coeffs, dtype=float)))
        if (
            q_hl.ndim == 2
            and q_hl.shape[0] == n_coeffs
            and q_hl.shape[1] > 0
            and m_metric.ndim == 2
            and m_metric.shape == (n_coeffs, n_coeffs)
        ):
            return np.asarray(q_hl @ (q_hl.T @ m_metric), dtype=float)
        return np.eye(n_coeffs, dtype=float)

    def get_m_imp_from_jr_matrix(self, input_basis: Optional[Any] = None) -> np.ndarray:
        """Expose dense map from input ``jr`` coefficients to imposed ``m_imp``."""
        st = self.state
        if input_basis is None and st.jr is not None:
            input_basis = st.jr.basis

        op_rhs = as_linear_map(st.geometry.get_jr_operator(input_basis))
        rhs0 = np.asarray(to_dense(op_rhs))
        n_scenarios = int(rhs0.shape[1]) if rhs0.ndim == 2 else 1
        rhs_terms = [rhs0]
        for term_index in range(1, st.m_imp_problem.num_data_terms):
            n_rows = int(st.m_imp_problem.A[term_index].num_rows)
            rhs_terms.append(np.zeros((n_rows, n_scenarios), dtype=rhs0.dtype))

        h_mat, _ = st.m_imp_problem.get_normal_equation_components(data_term_index=0)
        h_shape = tuple(np.asarray(h_mat).shape)
        dim_max = max(int(h_shape[0]), int(h_shape[1])) if len(h_shape) >= 2 else 1
        rcond = float(np.finfo(float).eps * max(dim_max, 1))
        solver = LeastSquaresSolver(solver="svd", tolerance=max(float(rcond), 1e-15))
        m_imp_from_jr = np.asarray(solver.solve(st.m_imp_problem, rhs_terms)).reshape(
            st.solution_basis.index_length, n_scenarios
        )
        if st.dynamics_mode == "full_induction":
            m_imp_from_jr = self.get_hl_projection_matrix(m_imp_from_jr.shape[0]) @ m_imp_from_jr
        p = np.asarray(st.constraints.m_imp_gauge_projector)
        if p.shape == (m_imp_from_jr.shape[0], m_imp_from_jr.shape[0]):
            m_imp_from_jr = p @ m_imp_from_jr
        return asarray(m_imp_from_jr)

    def get_external_forcing_matrices(
        self, input_basis_jr: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """Expose dense forcing maps for coupled rates from ``u`` and ``jr``."""
        st = self.state
        n = st.solution_basis.index_length
        n2 = 2 * n
        feedback_reg_lambda = self._toroidal_feedback_regularization_lambda()

        dtpsi_from_E = np.asarray(
            self._get_dtpsi_from_E_dense(use_pinning=bool(st.apply_psi_gauge)),
            dtype=float,
        )
        dmind_from_E = np.asarray(self._get_dmind_from_E_dense(), dtype=float)

        e_from_u = np.asarray(to_dense(as_linear_map(st.u_coeffs_to_E_coeffs)))
        m_imp_from_jr = np.asarray(self.get_m_imp_from_jr_matrix(input_basis=input_basis_jr))
        e_from_jr = np.asarray(to_dense(as_linear_map(st.m_imp_to_E_coeffs))) @ m_imp_from_jr

        return {
            "dtpsi_from_u": asarray(dtpsi_from_E @ e_from_u),
            "dtpsi_from_jr": asarray(dtpsi_from_E @ e_from_jr),
            "dmind_from_u": asarray(dmind_from_E @ e_from_u),
            "dmind_from_jr": asarray(dmind_from_E @ e_from_jr),
            "dtpsi_from_E": asarray(dtpsi_from_E),
            "dmind_from_E": asarray(dmind_from_E),
            "E_from_u": asarray(e_from_u),
            "E_from_jr": asarray(e_from_jr),
            "m_imp_from_jr": asarray(m_imp_from_jr),
        }
