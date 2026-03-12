"""Runtime diagnostics helpers for the simulation state."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from pynamit.math.linear_map import as_linear_map
from pynamit.simulation.settings import DynamicsMode
from pynamit.utils import to_numpy

logger = logging.getLogger(__name__)


class StateDiagnostics:
    """Own runtime diagnostics that are derived from live state operators."""

    def __init__(self, state: Any) -> None:
        self._state = state
        self._coupled_stability_warned_keys: set[Tuple[Any, ...]] = set()

    def reset_stability_warnings(self) -> None:
        """Forget previously emitted coupled-stability warnings."""
        self._coupled_stability_warned_keys.clear()

    def analyze_coupled_stability(
        self, l_flat: np.ndarray, *, label: str, unstable_tol: float = 1e-10
    ) -> Dict[str, float]:
        """Analyze coupled-operator spectrum and warn on unstable modes."""
        arr = np.asarray(l_flat, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"Coupled stability analysis requires a square matrix, got {arr.shape}."
                )

        eigvals = np.linalg.eigvals(arr)
        real = np.real(eigvals)
        max_real = float(np.max(real)) if real.size > 0 else 0.0
        min_real = float(np.min(real)) if real.size > 0 else 0.0
        n_pos = int(np.sum(real > float(unstable_tol)))
        n_total = int(real.size)

        report = {
            "max_real": max_real,
            "min_real": min_real,
            "positive_real_count": float(n_pos),
            "n_eigs": float(n_total),
        }

        if max_real > float(unstable_tol):
            key = (label, arr.shape[0], round(max_real, 9), round(min_real, 9), n_pos)
            if key not in self._coupled_stability_warned_keys:
                msg = (
                    "Coupled full-induction operator has unstable eigenmodes "
                    f"(label={label}, max Re(lambda)={max_real:.3e}, "
                    f"positive modes={n_pos}/{n_total}). "
                    "Explicit Euler integration is expected to be unstable for this operator."
                )
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self._coupled_stability_warned_keys.add(key)
        return report

    def get_coupled_stability_report(
        self, *, source: Literal["dense", "sparse", "auto"] = "dense"
    ) -> Dict[str, float]:
        """Return spectral stability report for the coupled full-induction operator."""
        st = self._state
        l_flat = np.asarray(st.get_coupled_induction_matrix(source=source, flatten=True))
        return self.analyze_coupled_stability(
            l_flat, label=f"{source}:psi_gauge={int(bool(st.apply_psi_gauge))}"
        )

    def _summarize_dtalpha_component(
        self,
        name: str,
        dt_alpha: np.ndarray,
        *,
        c_ll: np.ndarray,
        c_hl: np.ndarray,
        c_total: np.ndarray,
        rhs_physics: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return norms and constraint conflicts for one ``dt_alpha`` component."""
        dt_alpha_vec = np.asarray(dt_alpha, dtype=float).reshape(-1)
        ll_conflict = c_ll @ dt_alpha_vec if c_ll.size > 0 else np.zeros(0, dtype=float)
        hl_conflict = c_hl @ dt_alpha_vec if c_hl.size > 0 else np.zeros(0, dtype=float)
        total_conflict = c_total @ dt_alpha_vec if c_total.size > 0 else np.zeros(0, dtype=float)
        dt_alpha_norm = float(np.linalg.norm(dt_alpha_vec))
        ll_conflict_norm = float(np.linalg.norm(ll_conflict))
        hl_conflict_norm = float(np.linalg.norm(hl_conflict))
        total_conflict_norm = float(np.linalg.norm(total_conflict))

        summary: Dict[str, Any] = {
            "name": name,
            "dt_alpha": dt_alpha_vec,
            "dt_alpha_norm": dt_alpha_norm,
            "ll_conflict": ll_conflict,
            "ll_conflict_norm": ll_conflict_norm,
            "ll_conflict_ratio": ll_conflict_norm / max(dt_alpha_norm, 1e-30),
            "hl_conflict": hl_conflict,
            "hl_conflict_norm": hl_conflict_norm,
            "hl_conflict_ratio": hl_conflict_norm / max(dt_alpha_norm, 1e-30),
            "constraint_conflict": total_conflict,
            "constraint_conflict_norm": total_conflict_norm,
            "constraint_conflict_ratio": total_conflict_norm / max(dt_alpha_norm, 1e-30),
        }
        if rhs_physics is not None:
            rhs_vec = np.asarray(rhs_physics, dtype=float).reshape(-1)
            summary["rhs_physics"] = rhs_vec
            summary["rhs_physics_norm"] = float(np.linalg.norm(rhs_vec))
        return summary

    def _summarize_e_component(
        self,
        name: str,
        e_coeffs: np.ndarray,
        *,
        dtalpha_from_rhs: np.ndarray,
        c_ll: np.ndarray,
        c_hl: np.ndarray,
        c_total: np.ndarray,
    ) -> Dict[str, Any]:
        """Return toroidal forcing diagnostics for one electric-field component."""
        st = self._state
        e_vec = np.asarray(e_coeffs, dtype=float)
        rhs_physics = np.asarray(
            st.toroidal_matrices.compute_toroidal_rhs_from_E(e_vec), dtype=float
        ).reshape(-1)
        dt_alpha = np.asarray(dtalpha_from_rhs @ rhs_physics, dtype=float).reshape(-1)
        summary = self._summarize_dtalpha_component(
            name, dt_alpha, c_ll=c_ll, c_hl=c_hl, c_total=c_total, rhs_physics=rhs_physics
        )
        summary["E_coeff_norm"] = float(np.linalg.norm(e_vec.reshape(-1)))
        return summary

    def get_toroidal_driver_balance_report(self) -> Dict[str, Any]:
        """Report LL/HL conflict diagnostics for the live toroidal forcing channels."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Toroidal driver balance diagnostics require full_induction mode.")

        n = int(st.solution_space.index_length)
        e_shape = (2, n)
        zero_e = np.zeros(e_shape, dtype=float)

        bundle = st.constraints.induction_constraint_bundle_hard
        if bundle is None:
            c_ll = np.zeros((0, n), dtype=float)
            c_hl = np.zeros((0, n), dtype=float)
            c_total = np.zeros((0, n), dtype=float)
        else:
            c_ll = np.asarray(bundle.get("C_ll", np.zeros((0, n), dtype=float)), dtype=float)
            c_hl = np.asarray(bundle.get("C_hl", np.zeros((0, n), dtype=float)), dtype=float)
            c_total = np.asarray(bundle.get("C_total", np.zeros((0, n), dtype=float)), dtype=float)

        dtalpha_from_rhs = np.asarray(
            st.toroidal_matrices.build_dtalpha_from_toroidal_rhs_matrix(
                constraint_operator=None,
                weighting=st.toroidal_weighting,
                regularization_lambda=st.toroidal_regularization_lambda,
                penalty_operator=None,
                penalty_scaling=0.0,
                hinv_rtol=0.0,
            ),
            dtype=float,
        )

        e_wind = zero_e.copy()
        if st.u is not None:
            e_wind = np.asarray(
                st._apply_operator(st.u_coeffs_to_E_coeffs, st.u.coeffs, e_shape), dtype=float
            )

        e_br = zero_e.copy()
        if st.Br is not None:
            e_br = np.asarray(
                st._apply_operator(st.Br_to_E_coeffs, st.Br.coeffs, e_shape), dtype=float
            )

        e_direct = e_wind + e_br
        m_imp_imposed = np.zeros(n, dtype=float)
        if st.jr is not None:
            if st.m_imp_imposed is not None and not st._imposed_toroidal_dirty:
                m_imp_imposed = np.asarray(st.m_imp_imposed, dtype=float).reshape(-1)
            else:
                m_imp_imposed = np.asarray(
                    st._build_imposed_toroidal_baseline(np.asarray(st.jr.coeffs), e_direct),
                    dtype=float,
                ).reshape(-1)
        e_magnetic = np.asarray(
            st._apply_operator(st.m_imp_to_E_coeffs, m_imp_imposed, e_shape), dtype=float
        )
        e_total = e_direct + e_magnetic

        components: Dict[str, Dict[str, Any]] = {
            "wind": self._summarize_e_component(
                "wind",
                e_wind,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                c_hl=c_hl,
                c_total=c_total,
            ),
            "Br": self._summarize_e_component(
                "Br",
                e_br,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                c_hl=c_hl,
                c_total=c_total,
            ),
            "magnetic_imposed": self._summarize_e_component(
                "magnetic_imposed",
                e_magnetic,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                c_hl=c_hl,
                c_total=c_total,
            ),
            "total_external": self._summarize_e_component(
                "total_external",
                e_total,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                c_hl=c_hl,
                c_total=c_total,
            ),
        }

        dt_alpha_driver = st._get_dt_alpha_driver_coeffs()
        dt_alpha_driver_raw = None
        if st.dt_m_imp_driver is not None:
            m_imp_to_jr = as_linear_map(st.poloidal_matrices.m_imp_to_jr)
            jr_to_alpha = as_linear_map(st.toroidal_matrices.jr_to_alpha_coeff_operator)
            dt_m_imp_raw = np.asarray(st.dt_m_imp_driver.coeffs, dtype=float).reshape(-1)
            dt_alpha_driver_raw = np.asarray(
                jr_to_alpha.matvec(m_imp_to_jr.matvec(dt_m_imp_raw)), dtype=float
            ).reshape(-1)
            components["magnetic_driver_raw"] = self._summarize_dtalpha_component(
                "magnetic_driver_raw", dt_alpha_driver_raw, c_ll=c_ll, c_hl=c_hl, c_total=c_total
            )

        l_alpha = np.asarray(to_numpy(st.toroidal_matrices.dtalpha_operator), dtype=float).reshape(
            n, n
        )
        if dt_alpha_driver is not None:
            dt_alpha_driver_vec = np.asarray(dt_alpha_driver, dtype=float).reshape(-1)
            components["magnetic_driver"] = self._summarize_dtalpha_component(
                "magnetic_driver", dt_alpha_driver_vec, c_ll=c_ll, c_hl=c_hl, c_total=c_total
            )
            rhs_driver_feedback = -(l_alpha @ dt_alpha_driver_vec)
            dtalpha_driver_feedback = np.asarray(
                dtalpha_from_rhs @ rhs_driver_feedback, dtype=float
            ).reshape(-1)
            components["driver_feedback_rhs"] = self._summarize_dtalpha_component(
                "driver_feedback_rhs",
                dtalpha_driver_feedback,
                c_ll=c_ll,
                c_hl=c_hl,
                c_total=c_total,
                rhs_physics=rhs_driver_feedback,
            )
            rhs_residual = (
                np.asarray(components["total_external"]["rhs_physics"]) + rhs_driver_feedback
            )
            dtalpha_residual = np.asarray(dtalpha_from_rhs @ rhs_residual, dtype=float).reshape(-1)
            components["residual_after_driver_subtraction"] = self._summarize_dtalpha_component(
                "residual_after_driver_subtraction",
                dtalpha_residual,
                c_ll=c_ll,
                c_hl=c_hl,
                c_total=c_total,
                rhs_physics=rhs_residual,
            )

        return {
            "dynamics_mode": str(st.dynamics_mode),
            "magnetospheric_toroidal_lock": bool(st.magnetospheric_toroidal_lock),
            "apply_psi_gauge": bool(st.apply_psi_gauge),
            "n_solution_coeffs": n,
            "n_dt_alpha": int(dtalpha_from_rhs.shape[0]),
            "constraint_rows": {
                "ll": int(c_ll.shape[0]),
                "hl": int(c_hl.shape[0]),
                "total": int(c_total.shape[0]),
            },
            "components": components,
        }
