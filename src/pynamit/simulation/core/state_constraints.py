"""Constraint and Gauge mapping module for the simulation state.

This module isolates the extraction of LL mismatch constraints, 
row compression, metric construction, and gauge projection rules.
"""

from __future__ import annotations
import logging
from typing import Any, Tuple, Optional, Dict
from functools import cached_property

import numpy as np

from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.utils import asarray

logger = logging.getLogger(__name__)

class StateConstraints:
    """Manages constraint subspaces, gauge projections, and metrics for the simulation state."""

    def __init__(
        self,
        geometry: Any,
        solution_space: Any,
        dynamics_mode: str,
        connect_hemispheres: bool,
        magnetospheric_toroidal_lock: bool,
        apply_psi_gauge: bool,
        apply_m_ind_gauge: bool,
    ):
        self.geometry = geometry
        self.solution_space = solution_space
        self.dynamics_mode = dynamics_mode
        self.connect_hemispheres = connect_hemispheres
        self.magnetospheric_toroidal_lock = magnetospheric_toroidal_lock
        self.apply_psi_gauge = apply_psi_gauge
        self.apply_m_ind_gauge = apply_m_ind_gauge

    def _extract_ll_constraint_rows(self) -> Any:
        """Extract low-latitude (LL) rows from the constraint-scalar operator."""
        op = self.geometry.get_constraint_scalar_operator(self.solution_space)
        if not (self.dynamics_mode == "full_induction" and self.connect_hemispheres):
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
                "LL mask size mismatch: "
                f"mask={int(row_mask.size)} rows={int(op_lm.shape[0])}."
            )
        if not np.any(row_mask):
            raise RuntimeError(
                "LL mask contains no active rows for full_induction constraints."
            )

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
        self,
        A: np.ndarray,
        metric: np.ndarray,
        rtol: float,
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
                        n_coeff=n_coeff,
                        mode="mean_zero",
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

    @cached_property
    def m_ind_gauge_projector(self) -> np.ndarray:
        """Dense scalar gauge projector for legacy m_ind evolution."""
        n = self.solution_space.index_length
        kind = getattr(self.solution_space, "kind", "")
        if not self.apply_m_ind_gauge or kind not in ("CS", "GRID"):
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
        kind = getattr(self.solution_space, "kind", "")
        if kind not in ("CS", "GRID"):
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
            rtol = self._resolve_constraint_rank_rtol(
                spectrum=s,
                label="LL row compression",
            )
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
        """Resolve rank cutoff for HL/LL constraint decomposition."""
        auto = self._auto_relative_cutoff_from_spectrum(spectrum)
        logger.info("Auto %s rank rtol from spectral gap: %.3e", label, auto)
        return auto

    def _build_energy_metric_pair(
        self,
        ll_mask: np.ndarray,
        weights: np.ndarray,
        n_coeffs: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build global and HL magnetic-energy metrics in coefficient space."""
        try:
            curl = np.asarray(self.solution_space.get_curl_matrix(self.geometry.grid), dtype=float)
        except Exception:
            logger.warning("Energy-metric split: curl operator unavailable.", exc_info=True)
            return None, None

        if curl.ndim != 3 or curl.shape[0] != 2:
            logger.warning(
                "Energy-metric split skipped: unexpected curl shape %s (expected (2, N_grid, N_coeffs)).",
                tuple(curl.shape),
            )
            return None, None
        G_th = curl[0]
        G_ph = curl[1]

        if (
            G_th.ndim != 2
            or G_ph.ndim != 2
            or G_th.shape != G_ph.shape
            or G_th.shape[0] != ll_mask.size
            or G_th.shape[1] != n_coeffs
        ):
            logger.warning(
                "Energy-metric split skipped: derivative shapes incompatible (G_th=%s, G_ph=%s, expected=(%d,%d)).",
                tuple(G_th.shape),
                tuple(G_ph.shape),
                int(ll_mask.size),
                int(n_coeffs),
            )
            return None, None

        hl_mask = ~ll_mask
        sqrt_w = np.sqrt(weights)
        sqrt_w_hl = np.sqrt(weights * hl_mask.astype(float))

        Gth_w = sqrt_w.reshape(-1, 1) * G_th
        Gph_w = sqrt_w.reshape(-1, 1) * G_ph
        M = (Gth_w.T @ Gth_w) + (Gph_w.T @ Gph_w)

        Gth_w_hl = sqrt_w_hl.reshape(-1, 1) * G_th
        Gph_w_hl = sqrt_w_hl.reshape(-1, 1) * G_ph
        M_hl = (Gth_w_hl.T @ Gth_w_hl) + (Gph_w_hl.T @ Gph_w_hl)

        M = 0.5 * (M + M.T)
        M_hl = 0.5 * (M_hl + M_hl.T)
        return M, M_hl

    def _build_apex_metric_pair(
        self,
        ll_mask: np.ndarray,
        weights: np.ndarray,
        n_coeffs: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build global and HL Gram metrics in apex sample space."""
        apex_op = self.geometry.get_constraint_scalar_reference_operator(self.solution_space)

        A_apex = np.asarray(to_dense(as_linear_map(apex_op)), dtype=float)
        if A_apex.ndim != 2:
            logger.warning(
                "Apex-metric split skipped: unexpected apex map shape %s.",
                tuple(A_apex.shape),
            )
            return None, None
        if A_apex.shape[0] != ll_mask.size or A_apex.shape[1] != n_coeffs:
            logger.warning(
                "Apex-metric split skipped: apex map shape %s incompatible with "
                "mask size %d and n_coeffs %d.",
                tuple(A_apex.shape),
                int(ll_mask.size),
                int(n_coeffs),
            )
            return None, None

        hl_mask = ~ll_mask
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        sqrt_w_hl = np.sqrt(weights * hl_mask.astype(float)).reshape(-1, 1)

        A_w = sqrt_w * A_apex
        A_w_hl = sqrt_w_hl * A_apex
        B = A_w.T @ A_w
        B_hl = A_w_hl.T @ A_w_hl
        B = 0.5 * (B + B.T)
        B_hl = 0.5 * (B_hl + B_hl.T)
        return B, B_hl

    def _build_hl_ll_subspaces(self, n_coeffs: int, ll_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build compact HL/LL dominant subspaces in coefficient space."""
        ll_mask = np.asarray(ll_mask, dtype=bool).reshape(-1)
        hl_mask = ~ll_mask
        if not np.any(ll_mask) or not np.any(hl_mask):
            return np.zeros((n_coeffs, 0), dtype=float), np.zeros((n_coeffs, 0), dtype=float)

        weights = np.asarray(self.geometry.grid.weights).reshape(-1)
        if weights.size != ll_mask.size:
            weights = np.ones(ll_mask.size, dtype=float)
        weights = np.maximum(weights, 0.0)
        wsum = float(np.sum(weights))
        if not np.isfinite(wsum) or wsum <= 0:
            weights = np.ones_like(weights) / max(weights.size, 1)
        else:
            weights = weights / wsum

        M, M_hl = self._build_apex_metric_pair(ll_mask=ll_mask, weights=weights, n_coeffs=n_coeffs)
        if M is None or M_hl is None:
            return np.zeros((n_coeffs, 0), dtype=float), np.zeros((n_coeffs, 0), dtype=float)

        U, svals, _ = np.linalg.svd(M, full_matrices=False)
        if svals.size == 0 or svals[0] <= 0:
            return np.zeros((n_coeffs, 0), dtype=float), np.zeros((n_coeffs, 0), dtype=float)
        rtol = self._resolve_constraint_rank_rtol(
            spectrum=svals,
            label="HL/LL split",
        )
        keep_s = svals > (rtol * float(svals[0]))
        if not np.any(keep_s):
            return np.zeros((n_coeffs, 0), dtype=float), np.zeros((n_coeffs, 0), dtype=float)
        U_r = U[:, keep_s]
        s_r = svals[keep_s]
        S_inv_half = U_r / np.sqrt(s_r).reshape(1, -1)

        B = S_inv_half.T @ M_hl @ S_inv_half
        B = 0.5 * (B + B.T)
        evals, evecs = np.linalg.eigh(B)
        order = np.argsort(evals)[::-1]
        lambda_desc = np.clip(np.asarray(evals[order], dtype=float), 0.0, 1.0)
        V_desc = S_inv_half @ evecs[:, order]

        n_modes = int(lambda_desc.size)
        if n_modes < 3:
            hl_idx_raw = np.array([0], dtype=int) if n_modes > 0 else np.zeros(0, dtype=int)
            ll_idx_raw = (
                np.array([n_modes - 1], dtype=int)
                if n_modes > 1
                else np.zeros(0, dtype=int)
            )
            shannon_hl = int(np.rint(float(np.sum(lambda_desc)))) if n_modes > 0 else 0
            K = int(np.clip(shannon_hl, 0, max(n_modes - 1, 0)))
            hl_cut = float(lambda_desc[0]) if n_modes > 0 else 0.0
            ll_cut = float(lambda_desc[-1]) if n_modes > 0 else 0.0
        else:
            shannon_hl = int(np.rint(float(np.sum(lambda_desc))))
            K = int(np.clip(shannon_hl, 1, n_modes - 2))
            gaps = lambda_desc[:-1] - lambda_desc[1:]

            idx_h = int(np.argmax(gaps[:K]))
            idx_l = int(K + np.argmax(gaps[K:]))

            if idx_h >= idx_l:
                idx_h = max(0, K - 1)
                idx_l = min(n_modes - 2, K)

            hl_idx_raw = np.arange(0, idx_h + 1, dtype=int)
            ll_idx_raw = np.arange(idx_l + 1, n_modes, dtype=int)

            hl_cut = float(0.5 * (lambda_desc[idx_h] + lambda_desc[idx_h + 1]))
            ll_cut = float(0.5 * (lambda_desc[idx_l] + lambda_desc[idx_l + 1]))

        hl_sel = np.zeros_like(lambda_desc, dtype=bool)
        ll_sel = np.zeros_like(lambda_desc, dtype=bool)
        hl_sel[hl_idx_raw] = True
        ll_sel[ll_idx_raw] = True

        Q_hl = (
            self._m_orthonormalize_columns(V_desc[:, hl_sel], metric=M, rtol=rtol)
            if np.any(hl_sel)
            else np.zeros((n_coeffs, 0), dtype=float)
        )
        Q_ll = (
            self._m_orthonormalize_columns(V_desc[:, ll_sel], metric=M, rtol=rtol)
            if np.any(ll_sel)
            else np.zeros((n_coeffs, 0), dtype=float)
        )

        n_modes = int(V_desc.shape[1])
        n_hl = int(Q_hl.shape[1])
        n_ll = int(Q_ll.shape[1])
        hl_conc = float(np.mean(lambda_desc[hl_sel])) if np.any(hl_sel) else 0.0
        ll_conc = float(np.mean(1.0 - lambda_desc[ll_sel])) if np.any(ll_sel) else 0.0
        logger.info(
            "Energy split (shannon-two-gap): n=%d, K=%d, hl=%d, ll=%d, mid=%d, hl_raw=%d, ll_raw=%d, hl_cut=%.3f, ll_cut=%.3f, mean_hl=%.3f, mean_ll=%.3f",
            n_modes,
            int(K),
            n_hl,
            n_ll,
            int(max(n_modes - n_hl - n_ll, 0)),
            int(hl_idx_raw.size),
            int(ll_idx_raw.size),
            float(hl_cut),
            float(ll_cut),
            hl_conc,
            ll_conc,
        )
        return Q_hl, Q_ll

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
        """Build compact hard-constraint blocks for LL symmetry and optional HL lock."""
        if not (self.dynamics_mode == "full_induction" and self.connect_hemispheres):
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

        Q_hl, Q_ll = self._build_hl_ll_subspaces(C_ll_raw.shape[1], ll_mask_np)
        weights = np.asarray(self.geometry.grid.weights).reshape(-1)
        if weights.size != ll_mask_np.size:
            weights = np.ones(ll_mask_np.size, dtype=float)
        weights = np.maximum(weights, 0.0)
        wsum = float(np.sum(weights))
        if not np.isfinite(wsum) or wsum <= 0:
            weights = np.ones_like(weights) / max(weights.size, 1)
        else:
            weights = weights / wsum
        M_total, _ = self._build_apex_metric_pair(
            ll_mask=ll_mask_np,
            weights=weights,
            n_coeffs=C_ll_raw.shape[1],
        )
        if M_total is None or M_total.shape != (C_ll_raw.shape[1], C_ll_raw.shape[1]):
            raise RuntimeError(
                "Failed to build apex metric for hard induction constraints."
            )

        ll_anchor_rtol = max(float(np.finfo(float).eps * max(M_total.shape)), 0.0)
        Q_ll_anchor = self._m_orthonormalize_columns(
            C_ll_raw.T,
            metric=M_total,
            rtol=ll_anchor_rtol,
        )
        if Q_ll_anchor.shape[1] == 0:
            raise RuntimeError(
                "LL hard-constraint subspace is empty: unable to build "
                "metric-orthonormal LL mismatch row-space basis."
            )

        gram_ll = 0.5 * ((Q_ll_anchor.T @ M_total @ Q_ll_anchor) + (Q_ll_anchor.T @ M_total @ Q_ll_anchor).T)
        rcond_ll = float(np.finfo(float).eps * max(gram_ll.shape))
        gram_ll_pinv = np.linalg.pinv(gram_ll, rcond=rcond_ll)
        P_ll = Q_ll_anchor @ (gram_ll_pinv @ (Q_ll_anchor.T @ M_total))

        C_ll_mode = C_ll_raw @ P_ll
        w_ll = self._get_ll_constraint_row_weights(C_ll_mode.shape[0], float)
        C_ll_mode = np.sqrt(w_ll).reshape(-1, 1) * C_ll_mode
        C_ll = self._normalize_constraint_rows(C_ll_mode)
        C_ll = self._compress_constraint_rows(C_ll, rtol=0.0)
        if C_ll.shape[0] == 0:
            raise RuntimeError(
                "LL hard-constraint row set is empty after LL row-space projection."
            )

        q_hl_raw = np.asarray(Q_hl, dtype=float)
        if q_hl_raw.ndim == 2 and q_hl_raw.shape[1] > 0:
            Q_hl = self._m_orthonormalize_columns(
                q_hl_raw - (P_ll @ q_hl_raw),
                metric=M_total,
                rtol=ll_anchor_rtol,
            )
        else:
            Q_hl = np.zeros((C_ll_raw.shape[1], 0), dtype=float)

        Q_ll = Q_ll_anchor

        if self.magnetospheric_toroidal_lock and Q_hl.shape[1] > 0:
            C_hl = self._normalize_constraint_rows(Q_hl.T @ M_total)
        else:
            C_hl = np.zeros((0, C_ll_raw.shape[1]), dtype=float)
        C_total = np.ascontiguousarray(np.vstack([C_ll, C_hl]))
        return {
            "C_total": C_total,
            "C_ll": C_ll,
            "C_hl": C_hl,
            "Q_ll": Q_ll,
            "Q_hl": Q_hl,
            "Q_metric": np.asarray(M_total, dtype=float),
        }

    @cached_property
    def induction_constraint_operator_hard(self) -> Any:
        """Hard constraint operator for LL symmetry and optional HL lock."""
        if self.dynamics_mode == "full_induction" and not self.connect_hemispheres:
            return None
        if self.dynamics_mode != "full_induction":
            return self.geometry.get_constraint_scalar_operator(self.solution_space)

        bundle = self.induction_constraint_bundle_hard
        if bundle is None or bundle["C_total"].shape[0] == 0:
            raise RuntimeError(
                "Hard induction constraint bundle is missing or empty."
            )
        return bundle["C_total"]
