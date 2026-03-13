"""Constraint and Gauge mapping module for the simulation state.

This module isolates the extraction of LL mismatch constraints,
row compression, metric construction, and gauge projection rules.
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, Optional, Dict
from functools import cached_property

import numpy as np

from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.primitives.basis import is_cs_like_basis
from pynamit.simulation.settings import DynamicsMode, LLConstraintMode
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.utils import asarray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReducedScalarSystem:
    """Gauge-reduced scalar system shared by legacy steady-state and time stepping."""

    selector: np.ndarray
    full_operator: Optional[LinearMap] = None
    reduced_operator: Optional[LinearMap] = None

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


@dataclass(frozen=True)
class DtAlphaConstraintSystem:
    """Bundle LL compatibility handling for the full-induction ``dt_alpha`` solve."""

    ll_mode: LLConstraintMode
    c_ll: np.ndarray
    hard_operator: Optional[np.ndarray]
    soft_operator: Optional[np.ndarray]
    soft_scaling: float = 0.0

    @staticmethod
    def _as_2d_matrix(operator: Any) -> np.ndarray:
        """Normalize one constraint operator to a dense 2-D matrix."""
        arr = np.asarray(operator, dtype=float)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    @property
    def n_coeff(self) -> int:
        """Return the flattened ``dt_alpha`` coefficient dimension."""
        for operator in (self.hard_operator, self.soft_operator, self.c_ll):
            if operator is None:
                continue
            arr = self._as_2d_matrix(operator)
            if arr.shape[1] > 0:
                return int(arr.shape[1])
        return 0

    def _resolve_driver(self, driver_coeffs: Optional[np.ndarray], n_coeff: int) -> np.ndarray:
        """Coerce driver coefficients to the expected ``dt_alpha`` size."""
        if n_coeff <= 0:
            return np.zeros(0, dtype=float)
        if driver_coeffs is None:
            return np.zeros(n_coeff, dtype=float)

        driver = np.asarray(driver_coeffs, dtype=float).reshape(-1)
        if driver.size == n_coeff:
            return driver
        if float(np.linalg.norm(driver)) == 0.0:
            return np.zeros(n_coeff, dtype=float)
        raise RuntimeError(
            "Constraint RHS assembly mismatch: dt_alpha driver dimension does not match "
            f"constraint operator columns ({driver.size} != {n_coeff})."
        )

    def build_hard_rhs(self, driver_coeffs: Optional[np.ndarray]) -> np.ndarray:
        """Build the hard-constraint RHS in active solve coordinates."""
        if self.hard_operator is None:
            return np.zeros(0, dtype=float)

        hard = self._as_2d_matrix(self.hard_operator)
        if hard.shape[0] == 0:
            return np.zeros(0, dtype=float)

        if self.ll_mode != LLConstraintMode.HARD or self.c_ll.shape[0] == 0:
            return np.zeros(hard.shape[0], dtype=float)

        driver = self._resolve_driver(driver_coeffs, hard.shape[1])
        rhs = -np.asarray(self.c_ll, dtype=float) @ driver
        if rhs.shape[0] != hard.shape[0]:
            raise RuntimeError(
                "Constraint RHS assembly mismatch: hard RHS row count does not match "
                "active dt_alpha hard-constraint rows."
            )
        return np.asarray(rhs, dtype=float)

    def build_soft_rhs(self, driver_coeffs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Build the soft LL penalty RHS in active solve coordinates."""
        if self.soft_operator is None:
            return None

        soft = self._as_2d_matrix(self.soft_operator)
        if soft.shape[0] == 0:
            return None

        driver = self._resolve_driver(driver_coeffs, soft.shape[1])
        return np.asarray(-(soft @ driver), dtype=float)


def _as_square_linear_map(operator: Any, *, n_total: int) -> LinearMap:
    """Return a square operator as ``LinearMap`` with a 2-D dense fallback."""
    operator_obj = operator
    if not hasattr(operator_obj, "matvec"):
        operator_arr = asarray(operator_obj)
        if operator_arr.ndim != 2:
            operator_obj = operator_arr.reshape(n_total, n_total)
        else:
            operator_obj = operator_arr

    operator_map = as_linear_map(operator_obj)
    if operator_map._to_dense is None:
        return operator_map

    dense_base_op = operator_map

    def _to_dense_2d() -> np.ndarray:
        dense = np.asarray(dense_base_op.to_dense(), dtype=float)
        if dense.ndim != 2:
            dense = dense.reshape(n_total, n_total)
        return dense

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


def _build_projected_square_linear_map(operator_map: LinearMap, selector: np.ndarray) -> LinearMap:
    """Return reduced square operator ``S^T L S`` as ``LinearMap``."""
    selector_arr = np.asarray(selector, dtype=float)
    n_total, n_reduced = selector_arr.shape
    if n_reduced == n_total:
        return operator_map

    def matvec(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced)
        return np.asarray(
            selector_arr.T @ np.asarray(operator_map.matvec(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced)

    def rmatvec(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced)
        return np.asarray(
            selector_arr.T @ np.asarray(operator_map.rmatvec(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced)

    def matmat(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced, -1)
        return np.asarray(
            selector_arr.T @ np.asarray(operator_map.matmat(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced, -1)

    def rmatmat(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(n_reduced, -1)
        return np.asarray(
            selector_arr.T @ np.asarray(operator_map.rmatmat(selector_arr @ x_arr), dtype=float),
            dtype=float,
        ).reshape(n_reduced, -1)

    reduced_to_dense = None
    if operator_map._to_dense is not None:

        def reduced_to_dense() -> np.ndarray:
            dense = np.asarray(operator_map.to_dense(), dtype=float)
            if dense.ndim != 2:
                dense = dense.reshape(n_total, n_total)
            return np.asarray(selector_arr.T @ dense @ selector_arr, dtype=float)

    return LinearMap(
        shape=(n_reduced, n_reduced),
        dtype=operator_map.dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=reduced_to_dense,
        source=(operator_map, selector_arr),
    )


class StateConstraints:
    """Manages constraint subspaces, gauge projections, and metrics for the simulation state."""

    def __init__(
        self,
        geometry: Any,
        solution_space: Any,
        dynamics_mode: DynamicsMode | str,
        connect_hemispheres: bool,
        apply_psi_gauge: bool,
        apply_m_ind_gauge: bool,
        apply_m_imp_gauge: bool,
    ):
        self.geometry = geometry
        self.solution_space = solution_space
        self.dynamics_mode = DynamicsMode(str(dynamics_mode))
        self.connect_hemispheres = connect_hemispheres
        self.apply_psi_gauge = apply_psi_gauge
        self.apply_m_ind_gauge = apply_m_ind_gauge
        self.apply_m_imp_gauge = apply_m_imp_gauge

    def _extract_ll_constraint_rows(self) -> Any:
        """Extract low-latitude (LL) rows from the constraint-scalar operator."""
        op = self.geometry.get_constraint_scalar_operator(self.solution_space)
        if not (self.dynamics_mode == DynamicsMode.FULL_INDUCTION and self.connect_hemispheres):
            return op

        ll_mask = getattr(self.geometry, "ll_mask", None)
        if ll_mask is None:
            raise RuntimeError(
                "LL mask is required for full_induction with connect_hemispheres=True."
            )
        row_mask = np.asarray(ll_mask, dtype=bool).reshape(-1)

        op_lm = as_linear_map(op)
        if row_mask.size != op_lm.shape[0]:
            raise RuntimeError(
                f"LL mask size mismatch: mask={int(row_mask.size)} rows={int(op_lm.shape[0])}."
            )
        if not np.any(row_mask):
            raise RuntimeError("LL mask contains no active rows for full_induction constraints.")

        if hasattr(op, "tocsr"):
            return op.tocsr()[row_mask]
        return np.ascontiguousarray(to_dense(op_lm)[row_mask, :])

    def _orthonormalize_columns(self, A: np.ndarray, rtol: float) -> np.ndarray:
        """Return an orthonormal basis for the column space of A."""
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.size == 0:
            n_rows = A.shape[0] if A.ndim == 2 else 0
            return np.zeros((n_rows, 0), dtype=float)
        u, s, _ = np.linalg.svd(A, full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            return np.zeros((A.shape[0], 0), dtype=float)
        thresh = max(float(rtol), 0.0) * float(s[0])
        keep = s > thresh
        if not np.any(keep):
            return np.zeros((A.shape[0], 0), dtype=float)
        return np.ascontiguousarray(u[:, keep])

    def _m_orthonormalize_columns(
        self, A: np.ndarray, metric: np.ndarray, rtol: float
    ) -> np.ndarray:
        """Return an ``M``-orthonormal basis spanning ``col(A)``."""
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.size == 0:
            n_rows = A.shape[0] if A.ndim == 2 else 0
            return np.zeros((n_rows, 0), dtype=float)

        M = np.asarray(metric, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != A.shape[0]:
            return self._orthonormalize_columns(A, rtol=rtol)

        G = 0.5 * ((A.T @ M @ A) + (A.T @ M @ A).T)
        try:
            evals, evecs = np.linalg.eigh(G)
        except np.linalg.LinAlgError:
            return self._orthonormalize_columns(A, rtol=rtol)

        if evals.size == 0:
            return np.zeros((A.shape[0], 0), dtype=float)

        order = np.argsort(evals)[::-1]
        evals = np.asarray(evals[order], dtype=float)
        evecs = np.asarray(evecs[:, order], dtype=float)
        max_eval = float(np.max(evals))
        if not np.isfinite(max_eval) or max_eval <= 0:
            return np.zeros((A.shape[0], 0), dtype=float)

        thresh = max(float(rtol), 0.0) * max_eval
        keep = evals > thresh
        if not np.any(keep):
            return np.zeros((A.shape[0], 0), dtype=float)

        scale = np.sqrt(np.maximum(evals[keep], 0.0))
        Q = A @ (evecs[:, keep] / scale.reshape(1, -1))
        return np.ascontiguousarray(Q)

    @staticmethod
    def _normalize_constraint_rows(C: np.ndarray) -> np.ndarray:
        """Row-normalize constraint matrix and drop zero rows."""
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] == 0:
            return np.zeros((0, C.shape[1] if C.ndim == 2 else 0), dtype=float)
        row_norm = np.linalg.norm(C, axis=1)
        keep = row_norm > 0
        if not np.any(keep):
            return np.zeros((0, C.shape[1]), dtype=float)
        C_use = C[keep] / row_norm[keep].reshape(-1, 1)
        return np.ascontiguousarray(C_use)

    def _basis_has_mean_free_scalar_space(self) -> bool:
        """Return whether solution-basis scalar coefficients are mean-free by construction."""
        if not hasattr(self.solution_space, "scalar_fields_are_mean_free_by_construction"):
            return False
        try:
            return bool(self.solution_space.scalar_fields_are_mean_free_by_construction())
        except Exception:
            return False

    def get_psi_gauge_row(self, n_coeff: int) -> np.ndarray:
        """Return scalar gauge row for psi coefficients."""
        if self._basis_has_mean_free_scalar_space():
            return np.zeros((0, n_coeff), dtype=float)
        if hasattr(self.solution_space, "get_scalar_gauge_constraint_matrix"):
            try:
                row = np.asarray(
                    self.solution_space.get_scalar_gauge_constraint_matrix(
                        n_coeff=n_coeff, mode="mean_zero"
                    )
                )
                if row.ndim == 1:
                    row = row.reshape(1, -1)
                if row.ndim == 2 and row.shape[1] == n_coeff and row.shape[0] > 0:
                    return row.astype(float, copy=False)
            except Exception:
                pass
        row = np.zeros((1, n_coeff), dtype=float)
        row[0, 0] = 1.0
        return row

    def get_m_ind_gauge_row(self, n_coeff: int) -> np.ndarray:
        """Return scalar gauge row for m_ind coefficients."""
        return self.get_psi_gauge_row(n_coeff)

    def get_m_imp_gauge_row(self, n_coeff: int) -> np.ndarray:
        """Return scalar gauge row for m_imp coefficients."""
        return self.get_psi_gauge_row(n_coeff)

    def _build_scalar_selector(
        self, *, n_coeff: int, apply_gauge: bool, gauge_row: np.ndarray
    ) -> np.ndarray:
        """Return an orthonormal selector spanning the scalar gauge subspace."""
        n = int(n_coeff)
        if not apply_gauge or n <= 0:
            return np.eye(n, dtype=float)

        row = np.asarray(gauge_row, dtype=float)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if row.ndim != 2 or row.shape[1] != n or row.shape[0] == 0:
            return np.eye(n, dtype=float)

        pin_row = np.zeros((1, n), dtype=float)
        pin_row[0, 0] = 1.0
        if row.shape == (1, n) and float(np.linalg.norm(row - pin_row)) <= 1e-12:
            selector = np.zeros((n, max(n - 1, 0)), dtype=float)
            if n > 1:
                selector[1:, np.arange(n - 1)] = 1.0
            return selector

        _, s_row, vh_row = np.linalg.svd(row, full_matrices=True)
        if s_row.size == 0:
            return np.eye(n, dtype=float)
        rtol = float(np.finfo(float).eps * max(row.shape))
        cutoff = rtol * float(s_row[0])
        rank = int(np.sum(s_row > cutoff))
        return np.asarray(vh_row[rank:].T, dtype=float)

    def _get_reduced_scalar_system(
        self, *, apply_gauge: bool, gauge_row: np.ndarray, linear_operator: Any | None = None
    ) -> ReducedScalarSystem:
        """Return a gauge-reduced scalar system for the configured solution space."""
        n = int(self.solution_space.index_length)
        selector = self._build_scalar_selector(
            n_coeff=n, apply_gauge=bool(apply_gauge), gauge_row=gauge_row
        )
        if linear_operator is None:
            return ReducedScalarSystem(selector=np.asarray(selector, dtype=float))

        operator_map = _as_square_linear_map(linear_operator, n_total=n)
        reduced_operator = _build_projected_square_linear_map(operator_map, selector)
        return ReducedScalarSystem(
            selector=np.asarray(selector, dtype=float),
            full_operator=operator_map,
            reduced_operator=reduced_operator,
        )

    def get_m_ind_reduced_system(
        self, *, linear_operator: Any | None = None
    ) -> ReducedScalarSystem:
        """Return the gauge-reduced scalar system used by legacy ``m_ind`` evolution."""
        return self._get_reduced_scalar_system(
            apply_gauge=bool(self.apply_m_ind_gauge),
            gauge_row=self.get_m_ind_gauge_row(int(self.solution_space.index_length)),
            linear_operator=linear_operator,
        )

    def get_m_imp_reduced_system(
        self, *, linear_operator: Any | None = None
    ) -> ReducedScalarSystem:
        """Return the gauge-reduced scalar system used by imposed ``m_imp`` solves."""
        return self._get_reduced_scalar_system(
            apply_gauge=bool(self.apply_m_imp_gauge),
            gauge_row=self.get_m_imp_gauge_row(int(self.solution_space.index_length)),
            linear_operator=linear_operator,
        )

    @cached_property
    def m_ind_gauge_projector(self) -> np.ndarray:
        """Dense scalar gauge projector for legacy m_ind evolution."""
        n = self.solution_space.index_length
        if not self.apply_m_ind_gauge or not is_cs_like_basis(self.solution_space):
            return np.eye(n, dtype=float)

        row = np.asarray(self.get_m_ind_gauge_row(n), dtype=float)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if row.ndim != 2 or row.shape[1] != n or row.shape[0] == 0:
            return np.eye(n, dtype=float)

        C = row
        CCt = C @ C.T
        if CCt.size == 0:
            return np.eye(n, dtype=float)
        rcond = max(float(np.finfo(float).eps * max(CCt.shape)), 1e-15)
        CCt_pinv = np.linalg.pinv(CCt, rcond=rcond)
        P = np.eye(n, dtype=float) - C.T @ CCt_pinv @ C
        return np.asarray(0.5 * (P + P.T), dtype=float)

    @cached_property
    def m_imp_gauge_projector(self) -> np.ndarray:
        """Dense scalar gauge projector for imposed toroidal scalar m_imp."""
        n = self.solution_space.index_length
        if not self.apply_m_imp_gauge or not is_cs_like_basis(self.solution_space):
            return np.eye(n, dtype=float)

        row = np.asarray(self.get_m_imp_gauge_row(n), dtype=float)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if row.ndim != 2 or row.shape[1] != n or row.shape[0] == 0:
            return np.eye(n, dtype=float)

        C = row
        CCt = C @ C.T
        if CCt.size == 0:
            return np.eye(n, dtype=float)
        rcond = max(float(np.finfo(float).eps * max(CCt.shape)), 1e-15)
        CCt_pinv = np.linalg.pinv(CCt, rcond=rcond)
        P = np.eye(n, dtype=float) - C.T @ CCt_pinv @ C
        return np.asarray(0.5 * (P + P.T), dtype=float)

    def apply_m_ind_gauge_projection(self, coeffs: np.ndarray) -> np.ndarray:
        """Project scalar coefficients onto configured m_ind gauge subspace."""
        arr = np.asarray(asarray(coeffs)).reshape(-1)
        P = np.asarray(self.m_ind_gauge_projector)
        if P.shape != (arr.size, arr.size):
            return asarray(arr)
        return asarray(P @ arr)

    def apply_m_imp_gauge_projection(self, coeffs: np.ndarray) -> np.ndarray:
        """Project scalar coefficients onto configured m_imp gauge subspace."""
        arr = np.asarray(asarray(coeffs)).reshape(-1)
        P = np.asarray(self.m_imp_gauge_projector)
        if P.shape != (arr.size, arr.size):
            return asarray(arr)
        return asarray(P @ arr)

    def _compress_constraint_rows(self, C: np.ndarray, rtol: float) -> np.ndarray:
        """Compress row-space of C into a compact full-row-rank basis."""
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.size == 0:
            n_cols = C.shape[1] if C.ndim == 2 else 0
            return np.zeros((0, n_cols), dtype=float)
        row_norm = np.linalg.norm(C, axis=1)
        keep_rows = row_norm > 0
        if not np.any(keep_rows):
            return np.zeros((0, C.shape[1]), dtype=float)
        C_use = C[keep_rows]
        _, s, vh = np.linalg.svd(C_use, full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            return np.zeros((0, C.shape[1]), dtype=float)
        if rtol <= 0:
            rtol = self._resolve_constraint_rank_rtol(spectrum=s, label="LL row compression")
        thresh = max(float(rtol), 0.0) * float(s[0])
        keep = s > thresh
        if not np.any(keep):
            return np.zeros((0, C.shape[1]), dtype=float)
        return np.ascontiguousarray(vh[keep, :])

    @staticmethod
    def _auto_relative_cutoff_from_spectrum(
        spectrum: np.ndarray,
        *,
        default_rtol: float = 1e-8,
        min_rel: float = 1e-12,
        max_rel: float = 1e-2,
        gap_min_decades: float = 1.0,
    ) -> float:
        """Pick a relative rank cutoff from the largest spectral gap."""
        vals = np.asarray(spectrum, dtype=float).reshape(-1)
        if vals.size == 0:
            return float(np.clip(default_rtol, min_rel, max_rel))

        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float(np.clip(default_rtol, min_rel, max_rel))
        vals = np.sort(vals)[::-1]

        s_max = float(vals[0])
        if s_max <= 0:
            return float(np.clip(default_rtol, min_rel, max_rel))

        rel = vals / s_max
        rel = rel[np.isfinite(rel) & (rel > min_rel)]
        if rel.size < 2:
            return float(np.clip(default_rtol, min_rel, max_rel))

        log_rel = np.log10(rel)
        gaps = log_rel[:-1] - log_rel[1:]
        if gaps.size == 0:
            return float(np.clip(default_rtol, min_rel, max_rel))

        idx = int(np.argmax(gaps))
        if gaps[idx] < gap_min_decades:
            return float(np.clip(default_rtol, min_rel, max_rel))

        rtol = float(rel[idx + 1])
        return float(np.clip(rtol, min_rel, max_rel))

    def _resolve_constraint_rank_rtol(self, spectrum: np.ndarray, label: str) -> float:
        """Resolve a relative rank cutoff for compressed constraint rows."""
        auto = self._auto_relative_cutoff_from_spectrum(spectrum)
        logger.info("Auto %s rank rtol from spectral gap: %.3e", label, auto)
        return auto

    def _get_ll_constraint_row_weights(self, n_rows: int, dtype: Any) -> np.ndarray:
        """Area weights aligned with LL constraint rows."""
        ll_mask = getattr(self.geometry, "ll_mask", None)
        weights_all = np.asarray(self.geometry.grid.weights).reshape(-1)

        if ll_mask is None:
            raise RuntimeError("LL mask is required for LL hard-constraint row weights.")
        ll_idx = np.asarray(ll_mask, dtype=bool).reshape(-1)
        if weights_all.size != ll_idx.size:
            raise RuntimeError(
                "LL mask/grid weight size mismatch while building LL hard-constraint row weights."
            )
        w = weights_all[ll_idx].astype(dtype, copy=False)
        if w.size != n_rows:
            raise RuntimeError(
                "LL row-weight size mismatch while building LL hard-constraint rows: "
                f"expected {int(n_rows)}, got {int(w.size)}."
            )

        w = np.maximum(w, 0.0)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise RuntimeError(
                "Non-positive LL row-weight sum while building LL hard-constraint rows."
            )
        return w / w_sum

    @cached_property
    def induction_constraint_bundle_hard(self) -> Optional[Dict[str, np.ndarray]]:
        """Build compact hard-constraint blocks for LL compatibility."""
        if not (self.dynamics_mode == DynamicsMode.FULL_INDUCTION and self.connect_hemispheres):
            return None

        ll_op = self._extract_ll_constraint_rows()
        C_ll_raw = np.asarray(to_dense(as_linear_map(ll_op)), dtype=float)
        if C_ll_raw.ndim != 2:
            C_ll_raw = C_ll_raw.reshape(C_ll_raw.shape[0], -1)
        if C_ll_raw.shape[0] == 0:
            return None

        ll_mask = getattr(self.geometry, "ll_mask", None)
        if ll_mask is None:
            raise RuntimeError(
                "LL mask is required to build hard induction constraints "
                "for full_induction with connect_hemispheres=True."
            )

        ll_mask_np = np.asarray(ll_mask, dtype=bool).reshape(-1)
        if ll_mask_np.size != self.geometry.grid.size:
            raise RuntimeError(
                "LL mask/grid size mismatch for hard split: "
                f"mask={int(ll_mask_np.size)} grid={int(self.geometry.grid.size)}."
            )
        w_ll = self._get_ll_constraint_row_weights(C_ll_raw.shape[0], float)
        C_ll_weighted = np.sqrt(w_ll).reshape(-1, 1) * C_ll_raw
        C_ll = self._normalize_constraint_rows(C_ll_weighted)
        C_ll = self._compress_constraint_rows(C_ll, rtol=0.0)
        if C_ll.shape[0] == 0:
            raise RuntimeError("LL hard-constraint row set is empty after row compression.")

        C_total = np.ascontiguousarray(C_ll)
        return {"C_total": C_total, "C_ll": C_ll}

    @cached_property
    def induction_constraint_operator_hard(self) -> Any:
        """Hard constraint operator for LL compatibility."""
        if self.dynamics_mode == DynamicsMode.FULL_INDUCTION and not self.connect_hemispheres:
            return None
        if self.dynamics_mode != DynamicsMode.FULL_INDUCTION:
            return self.geometry.get_constraint_scalar_operator(self.solution_space)

        bundle = self.induction_constraint_bundle_hard
        if bundle is None or bundle["C_total"].shape[0] == 0:
            raise RuntimeError("Hard induction constraint bundle is missing or empty.")
        return bundle["C_total"]

    def build_dt_alpha_constraint_system(
        self, *, ll_mode: LLConstraintMode, soft_scaling: float
    ) -> DtAlphaConstraintSystem:
        """Build the active LL constraint system for full-induction ``dt_alpha``."""
        n_coeff = int(self.solution_space.index_length)
        if self.dynamics_mode != DynamicsMode.FULL_INDUCTION:
            return DtAlphaConstraintSystem(
                ll_mode=ll_mode,
                c_ll=np.zeros((0, n_coeff), dtype=float),
                hard_operator=None,
                soft_operator=None,
                soft_scaling=0.0,
            )

        bundle = self.induction_constraint_bundle_hard
        if bundle is None:
            return DtAlphaConstraintSystem(
                ll_mode=ll_mode,
                c_ll=np.zeros((0, n_coeff), dtype=float),
                hard_operator=None,
                soft_operator=None,
                soft_scaling=0.0,
            )

        c_ll = np.asarray(bundle.get("C_ll", np.zeros((0, 0), dtype=float)), dtype=float)
        if c_ll.ndim != 2:
            c_ll = c_ll.reshape(c_ll.shape[0], -1)

        if ll_mode == LLConstraintMode.HARD:
            hard_operator = c_ll if c_ll.shape[0] > 0 else None
            soft_operator = None
        else:
            hard_operator = None
            soft_operator = (
                c_ll if ll_mode == LLConstraintMode.SOFT and c_ll.shape[0] > 0 else None
            )

        return DtAlphaConstraintSystem(
            ll_mode=ll_mode,
            c_ll=np.asarray(c_ll, dtype=float),
            hard_operator=hard_operator,
            soft_operator=soft_operator,
            soft_scaling=float(soft_scaling) if soft_operator is not None else 0.0,
        )
