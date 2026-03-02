"""Toroidal solve/orchestration helpers.

This module keeps `ToroidalSystemMatrices` focused on toroidal operator/RHS
assembly while collecting gauge handling, `dt_alpha -> dt_psi` conversion, and
exported solve/dynamics maps in one helper object.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.simulation.induction.operator_api_utils import (
    build_linear_map,
    coerce_dense_operator_matrix,
)
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.utils import asarray, to_numpy

class ToroidalOperatorAPI:
    """Expose solve/orchestration routines built on top of toroidal operators."""

    def __init__(self, matrices: Any) -> None:
        self._mats = matrices

    def solve_dt_psi_superposed(
        self,
        rhs_physics: np.ndarray,
        rhs_constraint: np.ndarray,
        constraint_operator: Any,
        m_imp_to_jr_operator: Any,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Solve for `dpsi/dt` via one-shot constrained `dt_alpha` solve."""
        rhs_p = np.asarray(to_numpy(rhs_physics)).reshape(-1)
        dtalpha_to_dt_psi = self._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )

        if constraint_operator is None:
            M_alpha = self._mats._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha = M_alpha @ rhs_p
            return asarray((dtalpha_to_dt_psi @ dtalpha).reshape(-1))

        maps = self._mats._get_constrained_dtalpha_maps(
            alpha_map_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        rhs_c = np.asarray(to_numpy(rhs_constraint)).reshape(-1)
        dtalpha = maps["M_phys_dtalpha"] @ rhs_p
        if maps["M_corr_dtalpha"].shape[1] > 0:
            dtalpha = dtalpha + maps["M_corr_dtalpha"] @ rhs_c
        dt_psi = dtalpha_to_dt_psi @ dtalpha
        return asarray(dt_psi.reshape(-1))

    def _get_psi_gauge_projector_dense(
        self,
        m_imp_to_jr_operator: Any,
        use_pinning: Optional[bool] = None,
    ) -> np.ndarray:
        """Return explicit gauge projector applied after MP inversion."""
        mats = self._mats
        if use_pinning is None:
            use_pinning = mats.is_cs

        gauge_mode = "mean_zero"
        cache_key = (id(m_imp_to_jr_operator), bool(use_pinning), gauge_mode)
        cached = mats._psi_gauge_projector_cache.get(cache_key)
        if cached is not None:
            return cached

        op_m_to_jr = as_linear_map(m_imp_to_jr_operator)
        m_to_jr_dense = to_dense(op_m_to_jr)
        n = int(m_to_jr_dense.shape[1])
        identity = np.eye(n, dtype=m_to_jr_dense.dtype)

        if not use_pinning:
            mats._psi_gauge_projector_cache[cache_key] = identity
            return identity

        if mats.is_cs and hasattr(mats.basis, "get_scalar_gauge_projector_for_operator"):
            gauge_projector = np.asarray(
                mats.basis.get_scalar_gauge_projector_for_operator(
                    m_to_jr_dense,
                    mode=gauge_mode,
                    rcond=mats._default_pinv_rcond(m_to_jr_dense.shape),
                )
            )
            mats._psi_gauge_projector_cache[cache_key] = gauge_projector
            return gauge_projector

        gauge_row = None
        if hasattr(mats.basis, "get_scalar_gauge_constraint_matrix"):
            gauge_row = np.asarray(
                mats.basis.get_scalar_gauge_constraint_matrix(
                    n_coeff=n,
                    mode=gauge_mode,
                )
            )
        if gauge_row is None:
            if not hasattr(mats.basis, "get_evaluation_matrix"):
                raise RuntimeError(
                    "Scalar gauge projector requested, but basis does not provide "
                    "get_scalar_gauge_constraint_matrix() or get_evaluation_matrix(grid)."
                )
            g_mat = np.asarray(to_dense(as_linear_map(mats.basis.get_evaluation_matrix(mats.grid))))
            if g_mat.ndim != 2 or g_mat.shape[1] != n:
                raise RuntimeError(
                    "Failed to build generic scalar gauge row from basis.get_evaluation_matrix(grid)."
                )
            if hasattr(mats.grid, "weights") and mats.grid.weights is not None:
                w = np.asarray(to_numpy(mats.grid.weights)).reshape(-1)
                if w.size != g_mat.shape[0]:
                    raise RuntimeError(
                        "Grid weights size mismatch while building generic scalar gauge row."
                    )
                w = np.maximum(w, 0.0)
                w_sum = float(np.sum(w))
                if not np.isfinite(w_sum) or w_sum <= 0.0:
                    raise RuntimeError(
                        "Non-positive grid weights sum while building generic scalar gauge row."
                    )
                w = w / w_sum
            else:
                w = np.full(g_mat.shape[0], 1.0 / max(g_mat.shape[0], 1), dtype=float)
            gauge_row = (w @ g_mat).reshape(1, -1)
        if gauge_row.ndim == 1:
            gauge_row = gauge_row.reshape(1, -1)
        gauge_row = gauge_row.astype(m_to_jr_dense.dtype, copy=False)

        z_const = np.ones((n, 1), dtype=m_to_jr_dense.dtype)
        rel_const_null = np.linalg.norm(m_to_jr_dense @ z_const) / max(
            np.linalg.norm(m_to_jr_dense) * np.linalg.norm(z_const), 1e-30
        )
        if rel_const_null < 1e-6:
            null_basis = z_const
        else:
            _, s_vals, vh = np.linalg.svd(m_to_jr_dense, full_matrices=False)
            if s_vals.size == 0:
                null_basis = np.zeros((n, 0), dtype=m_to_jr_dense.dtype)
            else:
                svd_rtol = np.finfo(float).eps * max(m_to_jr_dense.shape)
                null_mask = s_vals <= svd_rtol * s_vals[0]
                null_basis = (
                    vh[null_mask].T
                    if np.any(null_mask)
                    else np.zeros((n, 0), dtype=m_to_jr_dense.dtype)
                )

        gauge_projector = identity
        if null_basis.shape[1] > 0:
            gauge_on_null = gauge_row @ null_basis
            if np.linalg.norm(gauge_on_null) > 0:
                gauge_on_null_pinv = np.linalg.pinv(gauge_on_null)
                gauge_projector = identity - (null_basis @ gauge_on_null_pinv @ gauge_row)

        mats._psi_gauge_projector_cache[cache_key] = gauge_projector
        return gauge_projector

    def _get_dtalpha_to_dt_psi_map_cached(
        self,
        *,
        m_imp_to_jr_operator: Any,
        use_pinning: bool,
    ) -> np.ndarray:
        """Return cached dense map `dt_alpha -> dpsi/dt`."""
        mats = self._mats
        key = (id(m_imp_to_jr_operator), bool(use_pinning))
        cached = mats._dtalpha_to_dt_psi_map_cache.get(key)
        if cached is not None:
            return cached

        alpha_to_psi = np.asarray(to_numpy(mats.alpha_to_psi_coeff_operator))
        gauge_projector = np.asarray(
            self._get_psi_gauge_projector_dense(
                m_imp_to_jr_operator,
                use_pinning=use_pinning,
            )
        )
        map_dtalpha_to_dt_psi = gauge_projector @ alpha_to_psi
        mats._dtalpha_to_dt_psi_map_cache[key] = map_dtalpha_to_dt_psi
        return map_dtalpha_to_dt_psi

    def build_dt_psi_from_toroidal_rhs_matrix(
        self,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Build dense map `toroidal_rhs -> dpsi/dt`."""
        mats = self._mats
        if constraint_operator is not None:
            maps = mats._get_constrained_dtalpha_maps(
                alpha_map_operator=constraint_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha_from_K = maps["M_phys_dtalpha"]
        else:
            dtalpha_from_K = mats._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )

        dtalpha_to_dt_psi = self._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )
        return asarray(dtalpha_to_dt_psi @ dtalpha_from_K)

    def build_psi_dynamics_matrix(
        self,
        psi_to_E_operator: np.ndarray,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Build the linear operator `psi -> dpsi/dt`."""
        mats = self._mats
        N = mats.basis.index_length
        E_to_rhs = to_numpy(mats.toroidal_rhs_from_E_operator)
        psi_to_E = coerce_dense_operator_matrix(
            psi_to_E_operator,
            n_component_rows=2,
            n_cols=N,
        )

        dt_psi_from_K = self.build_dt_psi_from_toroidal_rhs_matrix(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            constraint_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
            use_pinning=use_pinning,
        )
        return asarray((dt_psi_from_K @ E_to_rhs) @ psi_to_E)

    def get_psi_dynamics_operator(
        self,
        psi_to_E_operator: Any,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        dense: bool = False,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> "LinearMap":
        """Get linear operator `psi -> dpsi/dt`."""
        mats = self._mats
        N = mats.basis.index_length

        if dense or constraint_operator is not None:
            L_dense = self.build_psi_dynamics_matrix(
                psi_to_E_operator,
                m_imp_to_jr_operator,
                constraint_operator=constraint_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                use_pinning=use_pinning,
            )
            return as_linear_map(L_dense)

        if isinstance(psi_to_E_operator, LinearMap):
            psi_to_E_op = psi_to_E_operator
        else:
            psi_to_E_op = as_linear_map(
                coerce_dense_operator_matrix(
                    psi_to_E_operator,
                    n_component_rows=2,
                    n_cols=N,
                )
            )

        E_to_rhs_op = as_linear_map(np.asarray(to_numpy(mats.toroidal_rhs_from_E_operator)))
        dtalpha_from_K_op = as_linear_map(
            mats._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        )
        dtalpha_to_dt_psi_op = as_linear_map(
            self._get_dtalpha_to_dt_psi_map_cached(
                m_imp_to_jr_operator=m_imp_to_jr_operator,
                use_pinning=use_pinning,
            )
        )

        def matvec(x: np.ndarray) -> np.ndarray:
            y = psi_to_E_op.matvec(asarray(x).reshape(-1))
            y = E_to_rhs_op.matvec(y)
            y = dtalpha_from_K_op.matvec(y)
            y = dtalpha_to_dt_psi_op.matvec(y)
            return asarray(y)

        def rmatvec(x: np.ndarray) -> np.ndarray:
            y = dtalpha_to_dt_psi_op.rmatvec(asarray(x).reshape(-1))
            y = dtalpha_from_K_op.rmatvec(y)
            y = E_to_rhs_op.rmatvec(y)
            y = psi_to_E_op.rmatvec(y)
            return asarray(y)

        def to_dense_func() -> np.ndarray:
            return self.build_psi_dynamics_matrix(
                psi_to_E_operator,
                m_imp_to_jr_operator,
                constraint_operator=constraint_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                use_pinning=use_pinning,
            )

        return build_linear_map(
            shape=(N, N),
            matvec=matvec,
            rmatvec=rmatvec,
            dense_builder=to_dense_func,
            dtype=np.float64,
        )
