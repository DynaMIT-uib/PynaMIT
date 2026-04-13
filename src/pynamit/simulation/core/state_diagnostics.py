"""Runtime diagnostics helpers for the simulation state."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from pynamit.math.linear_map import as_linear_map
from pynamit.math.constants import mu0
from pynamit.primitives.basis import get_repo_cf_helmholtz_sign
from pynamit.simulation.settings import DynamicsMode
from pynamit.simulation.spatial import to_dense
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

        l_alpha = np.asarray(
            to_numpy(st.toroidal_matrices.physics_residual_coeff_operator), dtype=float
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
            "toroidal_closure_mode": str(getattr(st, "toroidal_closure_mode", "")),
            "apply_psi_gauge": bool(st.apply_psi_gauge),
            "n_solution_coeffs": n,
            "n_dt_alpha": int(dtalpha_from_rhs.shape[0]),
            "constraint_rows": {"ll": int(c_ll.shape[0]), "total": int(c_ll.shape[0])},
            "components": components,
        }

    def get_toroidal_projected_closure_report(self) -> Dict[str, Any]:
        """Report first-principles projected/perpendicular tangential closure diagnostics."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Toroidal projected-closure diagnostics require full_induction mode.")

        tor = st.toroidal_matrices
        current = np.asarray(to_numpy(tor.dtalpha_operator), dtype=float)
        current_physics = np.asarray(to_numpy(tor.physics_residual_coeff_operator), dtype=float)
        projected = np.asarray(to_numpy(tor.first_principles_projected_dtalpha_operator), dtype=float)
        perpendicular = np.asarray(
            to_numpy(tor.first_principles_perpendicular_dtalpha_operator), dtype=float
        )

        def _norm(arr: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        operator_report: Dict[str, Any] = {
            "current_dtalpha_operator_norm": _norm(current),
            "current_physics_operator_norm": _norm(current_physics),
            "first_principles_projected_norm": _norm(projected),
            "first_principles_perpendicular_norm": _norm(perpendicular),
            "current_minus_projected_norm": _norm(current - projected),
            "current_minus_projected_ratio": _norm(current - projected)
            / max(_norm(projected), 1e-30),
            "current_physics_rows": int(current_physics.shape[0]),
            "perpendicular_to_projected_ratio": _norm(perpendicular)
            / max(_norm(projected), 1e-30),
        }

        driver_report = self.get_toroidal_driver_balance_report()
        component_report: Dict[str, Any] = {}
        for name, component in driver_report["components"].items():
            dt_alpha = component.get("dt_alpha")
            if dt_alpha is None:
                continue
            dt_alpha_vec = np.asarray(dt_alpha, dtype=float).reshape(-1)
            current_action = current @ dt_alpha_vec
            current_physics_action = current_physics @ dt_alpha_vec
            projected_action = projected @ dt_alpha_vec
            perpendicular_action = perpendicular @ dt_alpha_vec
            mismatch_action = current_action - projected_action
            component_report[name] = {
                "dt_alpha_norm": float(np.linalg.norm(dt_alpha_vec)),
                "current_action_norm": float(np.linalg.norm(current_action)),
                "current_physics_action_norm": float(np.linalg.norm(current_physics_action)),
                "projected_action_norm": float(np.linalg.norm(projected_action)),
                "perpendicular_action_norm": float(np.linalg.norm(perpendicular_action)),
                "mismatch_action_norm": float(np.linalg.norm(mismatch_action)),
                "perpendicular_to_projected_ratio": float(np.linalg.norm(perpendicular_action))
                / max(float(np.linalg.norm(projected_action)), 1e-30),
                "mismatch_to_projected_ratio": float(np.linalg.norm(mismatch_action))
                / max(float(np.linalg.norm(projected_action)), 1e-30),
            }

        return {
            "dynamics_mode": str(st.dynamics_mode),
            "toroidal_closure_mode": str(getattr(st, "toroidal_closure_mode", "")),
            "operator_report": operator_report,
            "component_report": component_report,
        }

    def get_dynamic_toroidal_pi_report(self) -> Dict[str, Any]:
        """Return the explicit runtime realization of the dynamic ``Pi`` operator.

        The document-style ``Pi`` object is the raw toroidal-to-poloidal magnetic
        return for the dynamic ``psi`` branch. In the current runtime it is
        realized as the selected PFAC block, then composed through

            ``Pi -> J_S -> E -> rhs -> dpsi/dt``.

        The returned report also splits the live ``psi`` dynamics into the sum
        of:

        - the direct shell-current contribution already present in the top
          poloidal ``J_S`` block;
        - the nonlocal PFAC ``Pi`` contribution carried by the toroidal
          ``J_S`` block.
        """
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError(
                "Dynamic toroidal Pi diagnostics require full_induction mode."
            )
        if st.toroidal_to_JS_coeffs is None or st.toroidal_to_E_coeffs is None:
            raise ValueError("Dynamic toroidal Pi diagnostics require explicit psi operators.")
        from pynamit.primitives.basis import get_repo_cf_helmholtz_sign, get_repo_df_helmholtz_sign

        tor = st.toroidal_matrices
        n = int(st.solution_space.index_length)

        pi_operator = np.asarray(st.poloidal_matrices.dynamic_toroidal_pi_operator, dtype=float)
        pi_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_effective_operator, dtype=float
        )
        pi_open_operator = np.asarray(st.poloidal_matrices.dynamic_toroidal_pi_open_operator, dtype=float)
        pi_closed_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_closed_operator, dtype=float
        )
        pi_reaction_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_reaction_operator, dtype=float
        )
        pi_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_shielding_operator, dtype=float
        )
        pi_rm_boundary_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_open_operator, dtype=float
        )
        pi_rm_boundary_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_effective_operator, dtype=float
        )
        pi_rm_boundary_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_shielding_operator,
            dtype=float,
        )
        pi_to_br_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_br_open_operator, dtype=float
        )
        pi_to_br_closed_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_br_closed_operator, dtype=float
        )
        pi_to_br_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_br_effective_operator, dtype=float
        )
        pi_to_br_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_br_shielding_operator, dtype=float
        )
        if getattr(st, "RM", None) in (None, 0):
            pi_to_br_rm_open_operator = np.zeros_like(pi_to_br_open_operator)
            pi_to_br_rm_effective_operator = np.zeros_like(pi_to_br_open_operator)
            pi_to_br_rm_shielding_operator = np.zeros_like(pi_to_br_open_operator)
            pi_to_dbr_rm_open_operator = np.zeros_like(pi_to_br_open_operator)
            pi_to_dbr_rm_effective_operator = np.zeros_like(pi_to_br_open_operator)
            pi_to_dbr_rm_shielding_operator = np.zeros_like(pi_to_br_open_operator)
        else:
            pi_to_br_rm_open_operator = np.asarray(
                st.poloidal_matrices.dynamic_toroidal_pi_to_br_rm_open_operator, dtype=float
            )
            pi_to_br_rm_effective_operator = np.asarray(
                st.poloidal_matrices.dynamic_toroidal_pi_to_br_rm_effective_operator, dtype=float
            )
            pi_to_br_rm_shielding_operator = np.asarray(
                st.poloidal_matrices.dynamic_toroidal_pi_to_br_rm_shielding_operator, dtype=float
            )
            pi_to_dbr_rm_open_operator = np.asarray(
                st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_open_operator, dtype=float
            )
            pi_to_dbr_rm_effective_operator = np.asarray(
                st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_effective_operator, dtype=float
            )
            pi_to_dbr_rm_shielding_operator = np.asarray(
                st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_shielding_operator, dtype=float
            )
        pi_to_dbr_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_open_operator, dtype=float
        )
        pi_to_dbr_closed_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_closed_operator, dtype=float
        )
        pi_to_dbr_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_effective_operator, dtype=float
        )
        pi_to_dbr_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_pi_to_dbr_shielding_operator, dtype=float
        )
        shell_pi_harmonic_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_open_operator, dtype=float
        )
        shell_pi_harmonic_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_effective_operator, dtype=float
        )
        shell_pi_harmonic_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_shielding_operator, dtype=float
        )
        shell_pi_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_open_operator, dtype=float
        )
        shell_pi_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float
        )
        shell_pi_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_shielding_operator, dtype=float
        )
        shell_pi_to_br_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_br_open_operator, dtype=float
        )
        shell_pi_to_br_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_br_effective_operator, dtype=float
        )
        shell_pi_to_br_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_br_shielding_operator, dtype=float
        )
        shell_pi_harmonic_to_dbr_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_to_dbr_open_operator, dtype=float
        )
        shell_pi_harmonic_to_dbr_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_to_dbr_effective_operator,
            dtype=float,
        )
        shell_pi_harmonic_to_dbr_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_to_dbr_shielding_operator,
            dtype=float,
        )
        shell_pi_to_dbr_open_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_open_operator, dtype=float
        )
        shell_pi_to_dbr_effective_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_effective_operator, dtype=float
        )
        shell_pi_to_dbr_shielding_operator = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_shielding_operator, dtype=float
        )
        psi_to_js = np.asarray(st.toroidal_to_JS_coeffs.to_dense(), dtype=float)
        js_to_e = np.asarray(st.JS_to_E_coeffs.to_dense(), dtype=float)
        psi_to_e = np.asarray(st.toroidal_to_E_coeffs.to_dense(), dtype=float)
        e_to_rhs = np.asarray(to_numpy(tor.toroidal_rhs_from_E_operator), dtype=float)
        rhs_to_dtpsi = np.asarray(
            st.coupled_operators._get_dt_psi_from_toroidal_rhs_dense(
                apply_psi_gauge=bool(st.apply_psi_gauge)
            ),
            dtype=float,
        )
        e_to_dtpsi = np.asarray(
            st.coupled_operators._get_dt_psi_from_E_dense(
                apply_psi_gauge=bool(st.apply_psi_gauge)
            ),
            dtype=float,
        )

        psi_to_js_poloidal_block = np.asarray(psi_to_js[:n, :], dtype=float)
        psi_to_js_toroidal_block = np.asarray(psi_to_js[n:, :], dtype=float)
        repo_cf_sign = float(get_repo_cf_helmholtz_sign())
        repo_df_sign = float(get_repo_df_helmholtz_sign())
        expected_psi_to_js_poloidal_block = (
            -(1.0 / (repo_cf_sign * float(mu0)))
        ) * np.eye(n)
        expected_psi_to_js_toroidal_block = (
            -(1.0 / repo_df_sign)
        ) * np.asarray(pi_operator, dtype=float)

        direct_poloidal_psi_to_js_operator = np.vstack(
            [
                expected_psi_to_js_poloidal_block,
                np.zeros((n, n), dtype=float),
            ]
        )
        pfac_pi_to_js_operator = np.vstack(
            [
                np.zeros((n, n), dtype=float),
                expected_psi_to_js_toroidal_block,
            ]
        )

        psi_to_e_factorized = np.asarray(js_to_e @ psi_to_js, dtype=float)
        direct_poloidal_psi_to_e_operator = np.asarray(
            js_to_e @ direct_poloidal_psi_to_js_operator, dtype=float
        )
        pfac_pi_to_e_operator = np.asarray(js_to_e @ pfac_pi_to_js_operator, dtype=float)
        rhs_from_psi = np.asarray(e_to_rhs @ psi_to_e, dtype=float)
        rhs_from_direct_poloidal = np.asarray(
            e_to_rhs @ direct_poloidal_psi_to_e_operator, dtype=float
        )
        rhs_from_pfac_pi = np.asarray(e_to_rhs @ pfac_pi_to_e_operator, dtype=float)
        psi_to_dtpsi_from_rhs_chain = np.asarray(rhs_to_dtpsi @ rhs_from_psi, dtype=float)
        direct_poloidal_psi_to_dtpsi_operator = np.asarray(
            rhs_to_dtpsi @ rhs_from_direct_poloidal, dtype=float
        )
        pfac_pi_to_dtpsi_operator = np.asarray(rhs_to_dtpsi @ rhs_from_pfac_pi, dtype=float)
        psi_to_dtpsi = np.asarray(e_to_dtpsi @ psi_to_e, dtype=float)
        coupled_top_left = np.asarray(
            st.get_coupled_induction_blocks(source="dense")["dt_psi_from_psi"], dtype=float
        )

        def _norm(arr: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        return {
            "magnetospheric_shielding": bool(st.magnetospheric_shielding),
            "apply_psi_gauge": bool(st.apply_psi_gauge),
            "n_solution_coeffs": n,
            "pi_operator": pi_operator,
            "pi_like_dynamic_pfac_operator": pi_operator,
            "pi_effective_operator": pi_effective_operator,
            "pi_open_operator": pi_open_operator,
            "pi_closed_operator": pi_closed_operator,
            "pi_reaction_operator": pi_reaction_operator,
            "pi_shielding_operator": pi_shielding_operator,
            "pi_rm_boundary_open_operator": pi_rm_boundary_open_operator,
            "pi_rm_boundary_effective_operator": pi_rm_boundary_effective_operator,
            "pi_rm_boundary_shielding_operator": pi_rm_boundary_shielding_operator,
            "pi_to_br_open_operator": pi_to_br_open_operator,
            "pi_to_br_closed_operator": pi_to_br_closed_operator,
            "pi_to_br_effective_operator": pi_to_br_effective_operator,
            "pi_to_br_shielding_operator": pi_to_br_shielding_operator,
            "pi_to_br_rm_open_operator": pi_to_br_rm_open_operator,
            "pi_to_br_rm_effective_operator": pi_to_br_rm_effective_operator,
            "pi_to_br_rm_shielding_operator": pi_to_br_rm_shielding_operator,
            "pi_to_dbr_open_operator": pi_to_dbr_open_operator,
            "pi_to_dbr_closed_operator": pi_to_dbr_closed_operator,
            "pi_to_dbr_effective_operator": pi_to_dbr_effective_operator,
            "pi_to_dbr_shielding_operator": pi_to_dbr_shielding_operator,
            "pi_to_dbr_rm_open_operator": pi_to_dbr_rm_open_operator,
            "pi_to_dbr_rm_effective_operator": pi_to_dbr_rm_effective_operator,
            "pi_to_dbr_rm_shielding_operator": pi_to_dbr_rm_shielding_operator,
            "shell_pi_harmonic_open_operator": shell_pi_harmonic_open_operator,
            "shell_pi_harmonic_effective_operator": shell_pi_harmonic_effective_operator,
            "shell_pi_harmonic_shielding_operator": shell_pi_harmonic_shielding_operator,
            "shell_pi_open_operator": shell_pi_open_operator,
            "shell_pi_effective_operator": shell_pi_effective_operator,
            "shell_pi_shielding_operator": shell_pi_shielding_operator,
            "shell_pi_to_br_open_operator": shell_pi_to_br_open_operator,
            "shell_pi_to_br_effective_operator": shell_pi_to_br_effective_operator,
            "shell_pi_to_br_shielding_operator": shell_pi_to_br_shielding_operator,
            "shell_pi_harmonic_to_dbr_open_operator": shell_pi_harmonic_to_dbr_open_operator,
            "shell_pi_harmonic_to_dbr_effective_operator": shell_pi_harmonic_to_dbr_effective_operator,
            "shell_pi_harmonic_to_dbr_shielding_operator": shell_pi_harmonic_to_dbr_shielding_operator,
            "shell_pi_to_dbr_open_operator": shell_pi_to_dbr_open_operator,
            "shell_pi_to_dbr_effective_operator": shell_pi_to_dbr_effective_operator,
            "shell_pi_to_dbr_shielding_operator": shell_pi_to_dbr_shielding_operator,
            "psi_to_js": psi_to_js,
            "psi_to_js_poloidal_block": psi_to_js_poloidal_block,
            "psi_to_js_toroidal_block": psi_to_js_toroidal_block,
            "expected_psi_to_js_poloidal_block": expected_psi_to_js_poloidal_block,
            "expected_psi_to_js_toroidal_block": expected_psi_to_js_toroidal_block,
            "direct_poloidal_psi_to_js_operator": direct_poloidal_psi_to_js_operator,
            "pfac_pi_to_js_operator": pfac_pi_to_js_operator,
            "js_to_e": js_to_e,
            "psi_to_e": psi_to_e,
            "psi_to_e_factorized": psi_to_e_factorized,
            "direct_poloidal_psi_to_e_operator": direct_poloidal_psi_to_e_operator,
            "pfac_pi_to_e_operator": pfac_pi_to_e_operator,
            "e_to_rhs": e_to_rhs,
            "rhs_to_dtpsi": rhs_to_dtpsi,
            "e_to_dtpsi": e_to_dtpsi,
            "rhs_from_psi": rhs_from_psi,
            "rhs_from_direct_poloidal": rhs_from_direct_poloidal,
            "rhs_from_pfac_pi": rhs_from_pfac_pi,
            "psi_to_dtpsi_from_rhs_chain": psi_to_dtpsi_from_rhs_chain,
            "psi_to_dtpsi": psi_to_dtpsi,
            "direct_poloidal_psi_to_dtpsi_operator": direct_poloidal_psi_to_dtpsi_operator,
            "pfac_pi_to_dtpsi_operator": pfac_pi_to_dtpsi_operator,
            "coupled_top_left": coupled_top_left,
            "psi_to_js_poloidal_difference_norm": _norm(
                psi_to_js_poloidal_block - expected_psi_to_js_poloidal_block
            ),
            "psi_to_js_toroidal_difference_norm": _norm(
                psi_to_js_toroidal_block - expected_psi_to_js_toroidal_block
            ),
            "pi_effective_split_difference_norm": _norm(
                pi_effective_operator - (pi_open_operator + pi_shielding_operator)
            ),
            "pi_reaction_vs_shielding_difference_norm": _norm(
                pi_reaction_operator - pi_shielding_operator
            ),
            "pi_rm_boundary_effective_split_difference_norm": _norm(
                pi_rm_boundary_effective_operator
                - (pi_rm_boundary_open_operator + pi_rm_boundary_shielding_operator)
            ),
            "pi_to_br_effective_split_difference_norm": _norm(
                pi_to_br_effective_operator - (pi_to_br_open_operator + pi_to_br_shielding_operator)
            ),
            "pi_to_br_rm_effective_split_difference_norm": _norm(
                pi_to_br_rm_effective_operator
                - (pi_to_br_rm_open_operator + pi_to_br_rm_shielding_operator)
            ),
            "pi_to_dbr_effective_split_difference_norm": _norm(
                pi_to_dbr_effective_operator
                - (pi_to_dbr_open_operator + pi_to_dbr_shielding_operator)
            ),
            "pi_to_dbr_rm_effective_split_difference_norm": _norm(
                pi_to_dbr_rm_effective_operator
                - (pi_to_dbr_rm_open_operator + pi_to_dbr_rm_shielding_operator)
            ),
            "shell_pi_effective_split_difference_norm": _norm(
                shell_pi_effective_operator - (shell_pi_open_operator + shell_pi_shielding_operator)
            ),
            "shell_pi_harmonic_effective_split_difference_norm": _norm(
                shell_pi_harmonic_effective_operator
                - (shell_pi_harmonic_open_operator + shell_pi_harmonic_shielding_operator)
            ),
            "shell_pi_to_br_effective_split_difference_norm": _norm(
                shell_pi_to_br_effective_operator
                - (shell_pi_to_br_open_operator + shell_pi_to_br_shielding_operator)
            ),
            "shell_pi_to_dbr_effective_split_difference_norm": _norm(
                shell_pi_to_dbr_effective_operator
                - (shell_pi_to_dbr_open_operator + shell_pi_to_dbr_shielding_operator)
            ),
            "shell_pi_open_zero_br_norm": _norm(shell_pi_to_br_open_operator),
            "shell_pi_effective_zero_br_norm": _norm(shell_pi_to_br_effective_operator),
            "shell_pi_shielding_zero_br_norm": _norm(shell_pi_to_br_shielding_operator),
            "shell_pi_open_reconstruction_difference_norm": _norm(
                pi_open_operator - (shell_pi_harmonic_open_operator + shell_pi_open_operator)
            ),
            "shell_pi_effective_reconstruction_difference_norm": _norm(
                pi_effective_operator
                - (shell_pi_harmonic_effective_operator + shell_pi_effective_operator)
            ),
            "shell_pi_shielding_reconstruction_difference_norm": _norm(
                pi_shielding_operator
                - (shell_pi_harmonic_shielding_operator + shell_pi_shielding_operator)
            ),
            "psi_to_e_factorization_difference_norm": _norm(psi_to_e - psi_to_e_factorized),
            "psi_to_e_pi_split_difference_norm": _norm(
                psi_to_e - (direct_poloidal_psi_to_e_operator + pfac_pi_to_e_operator)
            ),
            "rhs_pi_split_difference_norm": _norm(
                rhs_from_psi - (rhs_from_direct_poloidal + rhs_from_pfac_pi)
            ),
            "psi_to_dtpsi_rhs_chain_difference_norm": _norm(
                psi_to_dtpsi - psi_to_dtpsi_from_rhs_chain
            ),
            "psi_to_dtpsi_pi_split_difference_norm": _norm(
                psi_to_dtpsi
                - (direct_poloidal_psi_to_dtpsi_operator + pfac_pi_to_dtpsi_operator)
            ),
            "psi_to_dtpsi_coupled_difference_norm": _norm(psi_to_dtpsi - coupled_top_left),
        }

    def get_dynamic_toroidal_pi_radius_report(self, radius: float) -> Dict[str, Any]:
        """Return the arbitrary-radius PFAC ``Pi`` split and magnetic images."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Dynamic toroidal Pi diagnostics require full_induction mode.")
        report = st.poloidal_matrices.get_dynamic_toroidal_pi_radius_report(float(radius))

        def _norm(arr: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        report = {key: np.asarray(value, dtype=float) for key, value in report.items()}
        report.update(
            {
                "radius": float(radius),
                "pi_open_split_difference_norm": _norm(
                    report["pi_open_total"]
                    - (report["pi_open_internal"] + report["pi_open_external"])
                ),
                "pi_effective_split_difference_norm": _norm(
                    report["pi_effective_total"]
                    - (report["pi_effective_internal"] + report["pi_effective_external"])
                ),
                "pi_shielding_split_difference_norm": _norm(
                    report["pi_shielding_total"]
                    - (report["pi_shielding_internal"] + report["pi_shielding_external"])
                ),
                "pi_to_br_open_split_difference_norm": _norm(
                    report["pi_to_br_open_total"]
                    - (report["pi_to_br_open_internal"] + report["pi_to_br_open_external"])
                ),
                "pi_to_br_effective_split_difference_norm": _norm(
                    report["pi_to_br_effective_total"]
                    - (
                        report["pi_to_br_effective_internal"]
                        + report["pi_to_br_effective_external"]
                    )
                ),
                "pi_to_dbr_open_split_difference_norm": _norm(
                    report["pi_to_dbr_open_total"]
                    - (report["pi_to_dbr_open_internal"] + report["pi_to_dbr_open_external"])
                ),
                "pi_to_dbr_effective_split_difference_norm": _norm(
                    report["pi_to_dbr_effective_total"]
                    - (
                        report["pi_to_dbr_effective_internal"]
                        + report["pi_to_dbr_effective_external"]
                    )
                ),
            }
        )
        return report

    def get_dynamic_toroidal_operator_chain_report(self) -> Dict[str, Any]:
        """Backward-compatible alias for the full dynamic ``psi`` chain report."""
        return self.get_dynamic_toroidal_pi_report()

    def get_reduced_shell_closure_report(self) -> Dict[str, Any]:
        """Return reduced-shell tangential and radial-return operators on ``q``.

        This report exposes the document-style one-boundary shell operators in
        the active coefficient space, without yet promoting either one to the
        live runtime solve:

        - tangential reduced-shell candidate
          ``K_I q = B0S·grad_S q + (rhat x B0S)·grad_S(Pi_shell[q])``
        - radial-shell magnetic-return ingredient
          ``q -> d_r B_r[q]``

        The runtime shell-corrected ``Pi_shell`` block is used throughout, so
        the report is tied to the same open/closed PFAC semantics as the active
        state.
        """
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Reduced-shell diagnostics require full_induction mode.")

        tor = st.toroidal_matrices
        n = int(tor.basis.index_length)
        ri = float(tor.RI)

        shell_pi_chi = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float
        )
        shell_pi_to_dbr_chi = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_effective_operator, dtype=float
        )

        q_to_chi = (1.0 / ri) * np.eye(n, dtype=float)
        chi_to_q = ri * np.eye(n, dtype=float)
        q_to_shell_pi = np.asarray(shell_pi_chi @ q_to_chi, dtype=float)
        q_to_shell_dbr = np.asarray(shell_pi_to_dbr_chi @ q_to_chi, dtype=float)

        b0th = np.asarray(to_numpy(tor.b_field.vec.theta), dtype=float).reshape(-1)
        b0ph = np.asarray(to_numpy(tor.b_field.vec.phi), dtype=float).reshape(-1)
        p = np.asarray(to_dense(tor.projection_matrix), dtype=float)

        if tor.is_cs:
            g = np.asarray(to_dense(tor.basis.get_evaluation_matrix(tor.grid)), dtype=float)
            d_th, d_ph, _ = tor.cs_rhs_derivative_operators
            parallel_raw = np.asarray(
                p @ (((b0th[:, None] * np.asarray(d_th, dtype=float)) + (b0ph[:, None] * np.asarray(d_ph, dtype=float))) @ g),
                dtype=float,
            )
            perpendicular_raw = np.asarray(
                p @ (((-b0ph[:, None] * np.asarray(d_th, dtype=float)) + (b0th[:, None] * np.asarray(d_ph, dtype=float))) @ g),
                dtype=float,
            )
        else:
            g_th = np.asarray(
                to_dense(tor.basis.get_evaluation_matrix(tor.grid, derivative="theta")), dtype=float
            )
            g_ph = np.asarray(
                to_dense(tor.basis.get_evaluation_matrix(tor.grid, derivative="phi")), dtype=float
            )
            parallel_raw = np.asarray(
                p @ ((b0th[:, None] * g_th) + (b0ph[:, None] * g_ph)), dtype=float
            )
            perpendicular_raw = np.asarray(
                p @ ((-b0ph[:, None] * g_th) + (b0th[:, None] * g_ph)), dtype=float
            )

        tangential_local_q = np.asarray((1.0 / ri) * parallel_raw, dtype=float)
        tangential_return_q = np.asarray(((1.0 / ri) * perpendicular_raw) @ q_to_shell_pi, dtype=float)
        tangential_total_q = np.asarray(tangential_local_q + tangential_return_q, dtype=float)

        radial_lambda_chi = np.asarray(
            st.radial_shell_response_model.build_lambda_gap_operator(tor), dtype=float
        )
        radial_source_from_q = np.asarray((1.0 / ri) * radial_lambda_chi, dtype=float)

        def _norm(arr: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        def _svd_summary(op: np.ndarray, *, rtol: float = 1e-10, n_keep: int = 6) -> Dict[str, Any]:
            s = np.linalg.svd(np.asarray(op, dtype=float), compute_uv=False)
            s = np.asarray(s, dtype=float).reshape(-1)
            if s.size == 0:
                return {
                    "largest": 0.0,
                    "smallest": np.zeros(0, dtype=float),
                    "estimated_nullity": 0,
                }
            cutoff = float(rtol) * max(float(np.max(s)), 1e-30)
            return {
                "largest": float(np.max(s)),
                "smallest": np.asarray(np.sort(s)[: min(int(n_keep), s.size)], dtype=float),
                "estimated_nullity": int(np.sum(s <= cutoff)),
            }

        return {
            "n_solution_coeffs": n,
            "repo_cf_sign": float(get_repo_cf_helmholtz_sign()),
            "q_to_chi_operator": q_to_chi,
            "chi_to_q_operator": chi_to_q,
            "q_to_shell_pi_operator": q_to_shell_pi,
            "q_to_shell_dbr_operator": q_to_shell_dbr,
            "tangential_parallel_raw_operator": parallel_raw,
            "tangential_perpendicular_raw_operator": perpendicular_raw,
            "tangential_local_q_operator": tangential_local_q,
            "tangential_return_q_operator": tangential_return_q,
            "tangential_total_q_operator": tangential_total_q,
            "radial_source_from_q_operator": radial_source_from_q,
            "tangential_total_split_difference_norm": _norm(
                tangential_total_q - (tangential_local_q + tangential_return_q)
            ),
            "tangential_local_matches_fieldline_advection_norm": _norm(
                tangential_local_q - ((1.0 / ri) * np.asarray(tor.fieldline_advection_operator_raw, dtype=float))
            ),
            "tangential_total_svd": _svd_summary(tangential_total_q),
            "radial_source_from_q_svd": _svd_summary(radial_source_from_q),
            "q_to_shell_dbr_norm": _norm(q_to_shell_dbr),
            "q_to_shell_pi_norm": _norm(q_to_shell_pi),
        }

    def _build_current_full_induction_noninductive_forcing(self) -> Dict[str, Any]:
        """Reconstruct the live non-inductive toroidal forcing without mutating state."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION:
            raise ValueError("Non-inductive forcing reconstruction requires full_induction mode.")

        n = int(st.solution_space.index_length)
        e_shape = (2, n)
        if st.u is None:
            e_direct = np.zeros(e_shape, dtype=float)
        else:
            e_direct = np.asarray(
                st._apply_operator(st.u_coeffs_to_E_coeffs, st.u.coeffs, e_shape), dtype=float
            )
        if st.Br is not None:
            e_direct = np.asarray(
                e_direct
                + np.asarray(
                    st._apply_operator(st.Br_to_E_coeffs, st.Br.coeffs, e_shape), dtype=float
                ),
                dtype=float,
            )

        jr_coeffs = None if st.jr is None else np.asarray(st.jr.coeffs, dtype=float).reshape(-1)
        if jr_coeffs is None:
            m_imp = np.zeros(n, dtype=float)
        elif st.m_imp_imposed is not None and not st._imposed_toroidal_dirty:
            m_imp = np.asarray(st.m_imp_imposed, dtype=float).reshape(-1)
        else:
            m_imp = np.asarray(
                st.induction.build_imposed_toroidal_baseline(jr_coeffs, e_direct), dtype=float
            ).reshape(-1)

        e_imposed = np.asarray(st._apply_operator(st.m_imp_to_E_coeffs, m_imp, e_shape), dtype=float)
        e_known = np.asarray(e_direct + e_imposed, dtype=float)
        return {
            "E_direct": e_direct,
            "E_known": e_known,
            "m_imp": m_imp,
        }

    def get_radial_shell_feedback_comparison_report(self) -> Dict[str, Any]:
        """Compare shell-electric and connector-scalar radial-shell feedback operators."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Radial-shell feedback diagnostics require full_induction mode.")

        from pynamit.simulation.induction import (
            EquivalentNonlocalRadialShellResponseModel,
            ExteriorToroidalScalarRadialResponseModel,
            IncrementalIdealityCorrectedExteriorToroidalUpdateModel,
            NonlocalShellElectricRadialResponseModel,
            RMToroidalBoundaryUpdateModel,
        )

        tor = st.toroidal_matrices
        rm_boundary_mode = "closed"
        if not (
            getattr(st, "RM", None) not in (None, 0)
            and bool(getattr(st, "magnetospheric_shielding", True))
        ):
            rm_boundary_mode = "open"

        equivalent_shell_model = EquivalentNonlocalRadialShellResponseModel()
        shell_feedback_model = NonlocalShellElectricRadialResponseModel(
            shell_response_model=equivalent_shell_model
        )
        shell_feedback_model.bind_state(st)

        raw_update_model = RMToroidalBoundaryUpdateModel(rm_boundary_mode=rm_boundary_mode)
        connector_feedback_model = ExteriorToroidalScalarRadialResponseModel(
            exterior_update_model=raw_update_model
        )
        connector_feedback_model.bind_state(st)
        corrected_update_model = IncrementalIdealityCorrectedExteriorToroidalUpdateModel(
            shell_feedback_response_model=equivalent_shell_model,
            base_exterior_update_model=RMToroidalBoundaryUpdateModel(
                rm_boundary_mode=rm_boundary_mode
            ),
        )
        corrected_connector_feedback_model = ExteriorToroidalScalarRadialResponseModel(
            exterior_update_model=corrected_update_model
        )
        corrected_connector_feedback_model.bind_state(st)

        shell_feedback = np.asarray(
            shell_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
        )
        connector_feedback = np.asarray(
            connector_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
        )
        corrected_connector_feedback = np.asarray(
            corrected_connector_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
        )
        diff = shell_feedback - connector_feedback
        corrected_diff = shell_feedback - corrected_connector_feedback

        def _norm(arr: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        shell_norm = _norm(shell_feedback)
        connector_norm = _norm(connector_feedback)
        diff_norm = _norm(diff)
        cosine = float(
            np.sum(shell_feedback * connector_feedback)
            / max(shell_norm * connector_norm, 1e-30)
        )

        # Direct connector matching check in dtpsi-space:
        #   dtpsi_target(from forcing-side scalar response) ?= dtpsi_connector
        n = int(tor.basis.index_length)
        psi_to_e = np.asarray(st.toroidal_to_E_coeffs.to_dense(), dtype=float)
        alpha_to_psi = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)
        rhs_from_e = np.asarray(shell_feedback_model.build_rhs_operator(tor), dtype=float)
        feedback_rhs_from_dtalpha = rhs_from_e @ psi_to_e @ alpha_to_psi
        jr_to_psi = np.asarray(to_numpy(tor.jr_to_psi_coeff_operator), dtype=float)
        trace_connector_lhs = jr_to_psi @ ((1.0 / float(mu0)) * feedback_rhs_from_dtalpha)
        raw_update_model.bind_state(st)
        connector_dtpsi_from_alpha = np.asarray(
            raw_update_model.build_dtpsi_from_dtalpha_operator(tor), dtype=float
        )
        trace_connector_rhs = connector_dtpsi_from_alpha
        trace_connector_diff = trace_connector_lhs - trace_connector_rhs
        trace_lhs_norm = _norm(trace_connector_lhs)
        trace_rhs_norm = _norm(trace_connector_rhs)
        trace_diff_norm = _norm(trace_connector_diff)
        trace_cosine = float(
            np.sum(trace_connector_lhs * trace_connector_rhs)
            / max(trace_lhs_norm * trace_rhs_norm, 1e-30)
        )
        corrected_trace_rhs = np.asarray(
            corrected_update_model.build_dtpsi_from_dtalpha_operator(tor), dtype=float
        )
        corrected_trace_diff = trace_connector_lhs - corrected_trace_rhs
        corrected_trace_rhs_norm = _norm(corrected_trace_rhs)
        corrected_trace_diff_norm = _norm(corrected_trace_diff)
        corrected_trace_cosine = float(
            np.sum(trace_connector_lhs * corrected_trace_rhs)
            / max(trace_lhs_norm * corrected_trace_rhs_norm, 1e-30)
        )
        correction_dtpsi = np.asarray(
            corrected_update_model.build_correction_dtpsi_from_dtalpha_operator(tor), dtype=float
        )

        return {
            "dynamics_mode": str(st.dynamics_mode),
            "toroidal_closure_mode": str(getattr(st, "toroidal_closure_mode", "")),
            "assumed_shell_trace_outer_boundary_mode": "not_used",
            "assumed_rm_boundary_mode": rm_boundary_mode,
            "shell_feedback_norm": shell_norm,
            "connector_feedback_norm": connector_norm,
            "difference_norm": diff_norm,
            "difference_to_shell_ratio": diff_norm / max(shell_norm, 1e-30),
            "difference_to_connector_ratio": diff_norm / max(connector_norm, 1e-30),
            "max_abs_difference": float(np.max(np.abs(diff))),
            "cosine_similarity": cosine,
            "corrected_connector_feedback_norm": _norm(corrected_connector_feedback),
            "corrected_difference_norm": _norm(corrected_diff),
            "corrected_difference_to_shell_ratio": _norm(corrected_diff)
            / max(shell_norm, 1e-30),
            "corrected_difference_to_connector_ratio": _norm(corrected_diff)
            / max(_norm(corrected_connector_feedback), 1e-30),
            "corrected_max_abs_difference": float(np.max(np.abs(corrected_diff))),
            "corrected_cosine_similarity": float(
                np.sum(shell_feedback * corrected_connector_feedback)
                / max(shell_norm * _norm(corrected_connector_feedback), 1e-30)
            ),
            "connector_correction_dtpsi_norm": _norm(correction_dtpsi),
            "incremental_connector_trace_lhs_norm": trace_lhs_norm,
            "incremental_connector_trace_rhs_norm": trace_rhs_norm,
            "incremental_connector_trace_difference_norm": trace_diff_norm,
            "incremental_connector_trace_difference_to_lhs_ratio": trace_diff_norm
            / max(trace_lhs_norm, 1e-30),
            "incremental_connector_trace_difference_to_rhs_ratio": trace_diff_norm
            / max(trace_rhs_norm, 1e-30),
            "incremental_connector_trace_max_abs_difference": float(
                np.max(np.abs(trace_connector_diff))
            ),
            "incremental_connector_trace_cosine_similarity": trace_cosine,
            "incremental_corrected_trace_rhs_norm": corrected_trace_rhs_norm,
            "incremental_corrected_trace_difference_norm": corrected_trace_diff_norm,
            "incremental_corrected_trace_difference_to_lhs_ratio": corrected_trace_diff_norm
            / max(trace_lhs_norm, 1e-30),
            "incremental_corrected_trace_difference_to_rhs_ratio": corrected_trace_diff_norm
            / max(corrected_trace_rhs_norm, 1e-30),
            "incremental_corrected_trace_max_abs_difference": float(
                np.max(np.abs(corrected_trace_diff))
            ),
            "incremental_corrected_trace_cosine_similarity": corrected_trace_cosine,
        }

    def get_radial_shell_forcing_comparison_report(self) -> Dict[str, Any]:
        """Compare reduced inductive and frozen-conductance incremental forcing operators."""
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Radial-shell forcing diagnostics require full_induction mode.")

        from pynamit.simulation.induction import (
            EquivalentNonlocalRadialShellResponseModel,
            FrozenConductanceIncrementalKnownElectricRadialResponseModel,
        )

        tor = st.toroidal_matrices
        condensed_model = EquivalentNonlocalRadialShellResponseModel()
        condensed_model.bind_state(st)
        explicit_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
        explicit_model.bind_state(st)

        condensed_rhs = np.asarray(condensed_model.build_rhs_operator(tor), dtype=float)
        explicit_rhs = np.asarray(explicit_model.build_rhs_operator(tor), dtype=float)
        diff = condensed_rhs - explicit_rhs

        def _norm(arr: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        condensed_norm = _norm(condensed_rhs)
        explicit_norm = _norm(explicit_rhs)
        diff_norm = _norm(diff)
        cosine = float(
            np.sum(condensed_rhs * explicit_rhs) / max(condensed_norm * explicit_norm, 1e-30)
        )

        n = int(st.solution_space.index_length)
        e_shape = (2, n)
        zero_e = np.zeros(e_shape, dtype=float)

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

        component_inputs = {
            "wind": e_wind,
            "Br": e_br,
            "magnetic_imposed": e_magnetic,
            "total_external": e_total,
        }
        component_report: Dict[str, Any] = {}
        for name, e_coeffs in component_inputs.items():
            e_vec = np.asarray(e_coeffs, dtype=float).reshape(-1)
            condensed_action = condensed_rhs @ e_vec
            explicit_action = explicit_rhs @ e_vec
            action_diff = condensed_action - explicit_action
            condensed_action_norm = _norm(condensed_action)
            explicit_action_norm = _norm(explicit_action)
            e_coeffs_arr = np.asarray(e_coeffs, dtype=float).reshape(2, n)
            e_cf = np.vstack([e_coeffs_arr[0], np.zeros(n, dtype=float)]).reshape(-1)
            e_df = np.vstack([np.zeros(n, dtype=float), e_coeffs_arr[1]]).reshape(-1)
            condensed_cf = condensed_rhs @ e_cf
            condensed_df = condensed_rhs @ e_df
            explicit_cf = explicit_rhs @ e_cf
            explicit_df = explicit_rhs @ e_df
            component_report[name] = {
                "E_coeff_norm": _norm(e_vec),
                "E_cf_norm": _norm(e_cf),
                "E_df_norm": _norm(e_df),
                "condensed_action_norm": condensed_action_norm,
                "explicit_action_norm": explicit_action_norm,
                "difference_norm": _norm(action_diff),
                "difference_to_condensed_ratio": _norm(action_diff)
                / max(condensed_action_norm, 1e-30),
                "difference_to_explicit_ratio": _norm(action_diff)
                / max(explicit_action_norm, 1e-30),
                "cosine_similarity": float(
                    np.sum(condensed_action * explicit_action)
                    / max(condensed_action_norm * explicit_action_norm, 1e-30)
                ),
                "max_abs_difference": float(np.max(np.abs(action_diff))),
                "condensed_cf_action_norm": _norm(condensed_cf),
                "condensed_df_action_norm": _norm(condensed_df),
                "explicit_cf_action_norm": _norm(explicit_cf),
                "explicit_df_action_norm": _norm(explicit_df),
            }

        return {
            "dynamics_mode": str(st.dynamics_mode),
            "toroidal_closure_mode": str(getattr(st, "toroidal_closure_mode", "")),
            "radial_shell_forcing_mode": str(getattr(st, "radial_shell_forcing_mode", "")),
            "active_response_model": getattr(getattr(st, "radial_shell_response_model", None), "description", None),
            "condensed_operator_norm": condensed_norm,
            "explicit_operator_norm": explicit_norm,
            "difference_norm": diff_norm,
            "difference_to_condensed_ratio": diff_norm / max(condensed_norm, 1e-30),
            "difference_to_explicit_ratio": diff_norm / max(explicit_norm, 1e-30),
            "cosine_similarity": cosine,
            "max_abs_difference": float(np.max(np.abs(diff))),
            "component_report": component_report,
        }

    def get_pragmatic_homogeneous_rm_connector_report(
        self, *, outer_boundary_mode: str = "shielded"
    ) -> Dict[str, Any]:
        """Report the pragmatic shell-state-to-``chi`` path.

        This keeps the existing runtime's upstream forcing assembly

            ``(u, Br, jr, Sigma) -> E_{S,I}^{known}``

        and applies the homogeneous-``R_M`` connector operator only
        downstream:

            ``E_{S,I}^{known} -> chi_I``

        with the connector trace ``q_I = R_I chi_I`` retained as the exact
        forcing-side preimage.

        The current implementation uses the explicit column-assembled
        homogeneous-outer-data operator, which is presently backed by the
        finite-``R_M`` harmonic side-trace baseline for the chosen outer mode.
        """
        st = self._state
        if st.dynamics_mode != DynamicsMode.FULL_INDUCTION or st.toroidal_matrices is None:
            raise ValueError("Pragmatic connector diagnostics require full_induction mode.")

        from pynamit.simulation.induction import (
            HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel,
            build_known_source_operator_from_q_trace,
        )

        tor = st.toroidal_matrices
        model = HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel(
            outer_boundary_mode=outer_boundary_mode
        )
        model.bind_state(st)

        d_q = np.asarray(model.build_q_trace_operator(tor), dtype=float)
        d_chi = (1.0 / float(st.RI)) * np.asarray(d_q, dtype=float)
        gamma_known = np.asarray(build_known_source_operator_from_q_trace(tor, model), dtype=float)

        def _norm(arr: Any) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        n = int(st.solution_space.index_length)
        e_shape = (2, n)
        zero_e = np.zeros(e_shape, dtype=float)

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

        component_inputs = {
            "wind": e_wind,
            "Br": e_br,
            "magnetic_imposed": e_magnetic,
            "total_external": e_total,
        }
        component_report: Dict[str, Any] = {}
        for name, e_coeffs in component_inputs.items():
            e_vec = np.asarray(e_coeffs, dtype=float).reshape(-1)
            q = np.asarray(d_q @ e_vec, dtype=float).reshape(-1)
            chi = np.asarray(d_chi @ e_vec, dtype=float).reshape(-1)
            dtjr = np.asarray(gamma_known @ e_vec, dtype=float).reshape(-1)
            component_report[name] = {
                "E_coeff_norm": _norm(e_vec),
                "chi": chi,
                "chi_norm": _norm(chi),
                "q": q,
                "q_norm": _norm(q),
                "dtjr_known": dtjr,
                "dtjr_known_norm": _norm(dtjr),
            }

        dt_alpha_driver = st._get_dt_alpha_driver_coeffs()
        driver_report: Optional[Dict[str, Any]] = None
        q_known_total = np.asarray(component_report["total_external"]["q"], dtype=float).reshape(-1)
        chi_known_total = np.asarray(
            component_report["total_external"]["chi"], dtype=float
        ).reshape(-1)
        if dt_alpha_driver is not None:
            dt_alpha_driver_vec = np.asarray(dt_alpha_driver, dtype=float).reshape(-1)
            alpha_to_psi = np.asarray(
                to_numpy(st.toroidal_matrices.alpha_to_psi_coeff_operator), dtype=float
            )
            alpha_to_jr = np.asarray(
                to_numpy(st.toroidal_matrices.alpha_to_jr_coeff_operator), dtype=float
            )
            chi_driver = np.asarray(alpha_to_psi @ dt_alpha_driver_vec, dtype=float).reshape(-1)
            q_driver = float(st.RI) * chi_driver
            driver_report = {
                "dt_alpha_driver": dt_alpha_driver_vec,
                "dt_alpha_driver_norm": _norm(dt_alpha_driver_vec),
                "chi_driver": chi_driver,
                "chi_driver_norm": _norm(chi_driver),
                "q_driver": q_driver,
                "q_driver_norm": _norm(q_driver),
                "jr_driver": np.asarray(alpha_to_jr @ dt_alpha_driver_vec, dtype=float).reshape(-1),
                "jr_driver_norm": _norm(alpha_to_jr @ dt_alpha_driver_vec),
            }
            q_known_total = q_known_total + q_driver
            chi_known_total = chi_known_total + chi_driver

        live_dtpsi = None if st.d_psi_dt is None else np.asarray(st.d_psi_dt, dtype=float).reshape(-1)
        total_chi = np.asarray(component_report["total_external"]["chi"], dtype=float).reshape(-1)
        live_difference = None if live_dtpsi is None else (chi_known_total - live_dtpsi)
        live_q = None if live_dtpsi is None else (float(st.RI) * live_dtpsi)
        known_q_difference = None if live_q is None else (q_known_total - live_q)

        return {
            "outer_boundary_mode": str(outer_boundary_mode),
            "description": model.description,
            "chi_operator_norm": _norm(d_chi),
            "q_operator_norm": _norm(d_q),
            "gamma_known_norm": _norm(gamma_known),
            "component_report": component_report,
            "driver_report": driver_report,
            "known_total_chi": chi_known_total,
            "known_total_chi_norm": _norm(chi_known_total),
            "known_total_q": q_known_total,
            "known_total_q_norm": _norm(q_known_total),
            "live_chi_norm": None if live_dtpsi is None else _norm(live_dtpsi),
            "live_dtpsi_norm": None if live_dtpsi is None else _norm(live_dtpsi),
            "pragmatic_total_chi_norm": _norm(total_chi),
            "known_chi_difference_norm": None if live_difference is None else _norm(live_difference),
            "live_difference_norm": None if live_difference is None else _norm(live_difference),
            "live_q_norm": None if live_q is None else _norm(live_q),
            "known_q_difference_norm": (
                None if known_q_difference is None else _norm(known_q_difference)
            ),
        }

    def get_pragmatic_homogeneous_rm_chi_report(
        self, *, outer_boundary_mode: str = "shielded"
    ) -> Dict[str, Any]:
        """Compatibility wrapper exposing the pragmatic report in ``chi``-first language."""
        return self.get_pragmatic_homogeneous_rm_connector_report(
            outer_boundary_mode=outer_boundary_mode
        )

    def get_magnetospheric_boundary_report(self) -> Dict[str, Any]:
        """Summarize induced boundary operators at ``R_M``."""
        st = self._state
        ops = st.poloidal_rm_boundary_operators
        tor_ops = st.toroidal_rm_boundary_operators
        magnetic_ops = st.magnetic_rm_boundary_operators

        def _norm(arr: Any) -> float:
            return float(np.linalg.norm(np.asarray(arr, dtype=float).reshape(-1)))

        toroidal_boundary_open = np.asarray(tor_ops.alpha_to_boundary_psi_rm, dtype=float)
        toroidal_boundary_effective = np.asarray(toroidal_boundary_open, dtype=float)

        return {
            "magnetospheric_shielding": bool(st.magnetospheric_shielding),
            "RM": None if st.RM is None else float(st.RM),
            "m_ind_to_br_rm": {
                "open_norm": _norm(ops.m_ind_to_br_rm_open),
                "effective_norm": _norm(ops.m_ind_to_br_rm_effective),
                "shielding_norm": _norm(ops.m_ind_to_br_rm_shielding),
            },
            "m_ind_to_magnetic_potential_rm": {
                "open_norm": _norm(magnetic_ops.m_ind_to_magnetic_potential_rm_open),
                "effective_norm": _norm(magnetic_ops.m_ind_to_magnetic_potential_rm_effective),
                "shielding_norm": _norm(magnetic_ops.m_ind_to_magnetic_potential_rm_shielding),
            },
            "dynamic_psi_to_ve_rm": {
                "open_norm": _norm(ops.dynamic_psi_to_ve_rm_open),
                "effective_norm": _norm(ops.dynamic_psi_to_ve_rm_effective),
                "shielding_norm": _norm(ops.dynamic_psi_to_ve_rm_shielding),
            },
            "dynamic_alpha_to_psi_rm": {
                "open_norm": _norm(toroidal_boundary_open),
                "effective_norm": _norm(toroidal_boundary_effective),
                "shielding_norm": _norm(toroidal_boundary_effective - toroidal_boundary_open),
            },
        }
