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
        alpha_to_normal_current_rm_grid: Optional[np.ndarray] = None,
        alpha_to_closure_potential_rm_coeff: Optional[np.ndarray] = None,
        alpha_to_divergent_closure_current_rm_grid: Optional[np.ndarray] = None,
        rhs_physics: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return norms and constraint conflicts for one ``dt_alpha`` component."""
        dt_alpha_vec = np.asarray(dt_alpha, dtype=float).reshape(-1)
        ll_conflict = c_ll @ dt_alpha_vec if c_ll.size > 0 else np.zeros(0, dtype=float)
        dt_alpha_norm = float(np.linalg.norm(dt_alpha_vec))
        ll_conflict_norm = float(np.linalg.norm(ll_conflict))

        summary: Dict[str, Any] = {
            "name": name,
            "dt_alpha": dt_alpha_vec,
            "dt_alpha_norm": dt_alpha_norm,
            "ll_conflict": ll_conflict,
            "ll_conflict_norm": ll_conflict_norm,
            "ll_conflict_ratio": ll_conflict_norm / max(dt_alpha_norm, 1e-30),
            "constraint_conflict": ll_conflict,
            "constraint_conflict_norm": ll_conflict_norm,
            "constraint_conflict_ratio": ll_conflict_norm / max(dt_alpha_norm, 1e-30),
        }
        if alpha_to_normal_current_rm_grid is not None:
            jn_rm = np.asarray(alpha_to_normal_current_rm_grid, dtype=float) @ dt_alpha_vec
            summary["rm_normal_current"] = jn_rm
            summary["rm_normal_current_norm"] = float(np.linalg.norm(jn_rm))
        if alpha_to_closure_potential_rm_coeff is not None:
            chi_rm = np.asarray(alpha_to_closure_potential_rm_coeff, dtype=float) @ dt_alpha_vec
            summary["rm_closure_potential"] = chi_rm
            summary["rm_closure_potential_norm"] = float(np.linalg.norm(chi_rm))
        if alpha_to_divergent_closure_current_rm_grid is not None:
            closure_current_rm = np.tensordot(
                np.asarray(alpha_to_divergent_closure_current_rm_grid, dtype=float),
                dt_alpha_vec,
                axes=([2], [0]),
            )
            summary["rm_divergent_closure_current"] = closure_current_rm
            summary["rm_divergent_closure_current_norm"] = float(
                np.linalg.norm(closure_current_rm.reshape(-1))
            )
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
        alpha_to_normal_current_rm_grid: Optional[np.ndarray] = None,
        alpha_to_closure_potential_rm_coeff: Optional[np.ndarray] = None,
        alpha_to_divergent_closure_current_rm_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return toroidal forcing diagnostics for one electric-field component."""
        st = self._state
        e_vec = np.asarray(e_coeffs, dtype=float)
        rhs_physics = np.asarray(
            st.toroidal_matrices.compute_toroidal_rhs_from_E(e_vec), dtype=float
        ).reshape(-1)
        dt_alpha = np.asarray(dtalpha_from_rhs @ rhs_physics, dtype=float).reshape(-1)
        summary = self._summarize_dtalpha_component(
            name,
            dt_alpha,
            c_ll=c_ll,
            alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
            alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
            alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
            rhs_physics=rhs_physics,
        )
        summary["E_coeff_norm"] = float(np.linalg.norm(e_vec.reshape(-1)))
        return summary

    def get_toroidal_driver_balance_report(self) -> Dict[str, Any]:
        """Report LL-compatibility diagnostics for the live toroidal forcing channels."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Toroidal driver balance diagnostics require full_induction mode.")

        n = int(st.solution_space.index_length)
        e_shape = (2, n)
        zero_e = np.zeros(e_shape, dtype=float)

        constraint_system = st.dt_alpha_constraint_system
        c_ll = np.asarray(constraint_system.c_ll, dtype=float)
        rm_closure_ops = st.poloidal_matrices.toroidal_rm_closure_operators
        alpha_to_normal_current_rm_grid = np.asarray(
            rm_closure_ops.alpha_to_normal_current_rm_grid, dtype=float
        )
        alpha_to_closure_potential_rm_coeff = np.asarray(
            rm_closure_ops.alpha_to_closure_potential_rm_coeff, dtype=float
        )
        alpha_to_divergent_closure_current_rm_grid = np.asarray(
            rm_closure_ops.alpha_to_divergent_closure_current_rm_grid, dtype=float
        )

        dtalpha_from_rhs = np.asarray(
            st.toroidal_matrices.build_dtalpha_from_toroidal_rhs_matrix(
                constraint_operator=None,
                weighting=st.toroidal_weighting,
                regularization_lambda=st.toroidal_regularization_lambda,
                penalty_operator=constraint_system.soft_operator,
                penalty_scaling=float(constraint_system.soft_scaling),
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
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
            ),
            "Br": self._summarize_e_component(
                "Br",
                e_br,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
            ),
            "magnetic_imposed": self._summarize_e_component(
                "magnetic_imposed",
                e_magnetic,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
            ),
            "total_external": self._summarize_e_component(
                "total_external",
                e_total,
                dtalpha_from_rhs=dtalpha_from_rhs,
                c_ll=c_ll,
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
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
                "magnetic_driver_raw",
                dt_alpha_driver_raw,
                c_ll=c_ll,
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
            )

        l_alpha = np.asarray(to_numpy(st.toroidal_matrices.dtalpha_operator), dtype=float).reshape(
            n, n
        )
        if dt_alpha_driver is not None:
            dt_alpha_driver_vec = np.asarray(dt_alpha_driver, dtype=float).reshape(-1)
            components["magnetic_driver"] = self._summarize_dtalpha_component(
                "magnetic_driver",
                dt_alpha_driver_vec,
                c_ll=c_ll,
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
            )
            rhs_driver_feedback = -(l_alpha @ dt_alpha_driver_vec)
            dtalpha_driver_feedback = np.asarray(
                dtalpha_from_rhs @ rhs_driver_feedback, dtype=float
            ).reshape(-1)
            components["driver_feedback_rhs"] = self._summarize_dtalpha_component(
                "driver_feedback_rhs",
                dtalpha_driver_feedback,
                c_ll=c_ll,
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
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
                alpha_to_normal_current_rm_grid=alpha_to_normal_current_rm_grid,
                alpha_to_closure_potential_rm_coeff=alpha_to_closure_potential_rm_coeff,
                alpha_to_divergent_closure_current_rm_grid=alpha_to_divergent_closure_current_rm_grid,
                rhs_physics=rhs_residual,
            )

        return {
            "dynamics_mode": str(st.dynamics_mode),
            "magnetospheric_toroidal_lock": bool(st.magnetospheric_toroidal_lock),
            "apply_psi_gauge": bool(st.apply_psi_gauge),
            "n_solution_coeffs": n,
            "n_dt_alpha": int(dtalpha_from_rhs.shape[0]),
            "constraint_rows": {"ll": int(c_ll.shape[0]), "total": int(c_ll.shape[0])},
            "components": components,
        }

    def get_toroidal_rm_reaction_report(self) -> Dict[str, Any]:
        """Summarize runtime-open, shell-surrogate, and RM boundary-source operators."""
        st = self._state
        proto = st.toroidal_rm_reaction_prototype

        def _norm(arr: Any) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        return {
            "magnetospheric_toroidal_lock": bool(st.magnetospheric_toroidal_lock),
            "RM": None if st.RM is None else float(st.RM),
            "shell_boundary_closure": {
                "rm_to_ri_norm": _norm(proto.rm_to_ri),
                "ri_to_rm_norm": _norm(proto.ri_to_rm),
                "roundtrip_gain_norm": _norm(proto.roundtrip_gain),
                "closure_denominator_norm": _norm(proto.closure_denominator),
                "closure_inv_norm": _norm(proto.closure_inv),
                "reaction_operator_norm": _norm(proto.shell_reaction_operator),
                "sheet_boundary_psi_rm_norm": _norm(proto.alpha_to_sheet_boundary_psi_rm),
                "fixed_point_residual_norm": _norm(
                    proto.alpha_to_psi_shell_closed
                    - proto.alpha_to_psi_open
                    - (proto.roundtrip_gain @ proto.alpha_to_psi_shell_closed)
                ),
                "denominator_residual_norm": _norm(
                    (proto.closure_denominator @ proto.alpha_to_psi_shell_closed)
                    - proto.alpha_to_psi_open
                ),
                "reaction_operator_residual_norm": _norm(
                    (proto.alpha_to_psi_shell_closed - proto.alpha_to_psi_open)
                    - (proto.shell_reaction_operator @ proto.alpha_to_psi_open)
                ),
                "runtime_vs_shell_closed_mismatch_norm": _norm(
                    proto.alpha_to_psi_closed - proto.alpha_to_psi_shell_closed
                ),
                "runtime_vs_shell_radial_mismatch_norm": _norm(
                    proto.radial_closure_dt_psi_closed - proto.radial_closure_dt_psi_shell_closed
                ),
                "sheet_rm_value_mismatch_norm": _norm(
                    proto.alpha_to_sheet_boundary_psi_rm
                    - (proto.ri_to_rm @ proto.alpha_to_psi_open)
                ),
            },
            "alpha_to_psi": {
                "open_norm": _norm(proto.alpha_to_psi_open),
                "closed_norm": _norm(proto.alpha_to_psi_closed),
                "reaction_norm": _norm(proto.alpha_to_psi_reaction),
                "closure_residual_norm": _norm(
                    proto.alpha_to_psi_closed
                    - proto.alpha_to_psi_open
                    - proto.alpha_to_psi_reaction
                ),
            },
            "radial_closure_dt_psi": {
                "open_norm": _norm(proto.radial_closure_dt_psi_open),
                "closed_norm": _norm(proto.radial_closure_dt_psi_closed),
                "reaction_norm": _norm(proto.radial_closure_dt_psi_reaction),
                "closure_residual_norm": _norm(
                    proto.radial_closure_dt_psi_closed
                    - proto.radial_closure_dt_psi_open
                    - proto.radial_closure_dt_psi_reaction
                ),
            },
            "toroidal_feedback_dtalpha": {
                "open_norm": _norm(proto.toroidal_feedback_dtalpha_open),
                "closed_norm": _norm(proto.toroidal_feedback_dtalpha_closed),
                "reaction_norm": _norm(proto.toroidal_feedback_dtalpha_reaction),
                "closure_residual_norm": _norm(
                    proto.toroidal_feedback_dtalpha_closed
                    - proto.toroidal_feedback_dtalpha_open
                    - proto.toroidal_feedback_dtalpha_reaction
                ),
            },
            "dynamic_pfac": {
                "open_norm": _norm(proto.dynamic_pfac_open),
                "closed_norm": _norm(proto.dynamic_pfac_closed),
                "reaction_norm": _norm(proto.dynamic_pfac_reaction),
                "alpha_reaction_norm": _norm(proto.alpha_to_dynamic_pfac_reaction),
                "closure_residual_norm": _norm(
                    proto.dynamic_pfac_closed
                    - proto.dynamic_pfac_open
                    - proto.dynamic_pfac_reaction
                ),
            },
            "rm_boundary_closure": {
                "normal_current_operator_norm": _norm(proto.alpha_to_normal_current_rm_grid),
                "closure_potential_operator_norm": _norm(
                    proto.alpha_to_closure_potential_rm_coeff
                ),
                "divergent_closure_current_operator_norm": _norm(
                    proto.alpha_to_divergent_closure_current_rm_grid
                ),
            },
        }
