"""State-level induction and coupled-evolution orchestration helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pynamit.math.integration import ExponentialIntegrator
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.simulation.core.coupled_solver import CoupledSteadyStateSolver
from pynamit.utils import asarray, to_numpy, xp

TimedSolveFn = Callable[..., np.ndarray]
AvailableMemoryFn = Callable[[], Optional[int]]

logger = logging.getLogger(__name__)


class StateInduction:
    """Expose induction/coupled orchestration on top of a `State` instance."""

    def __init__(
        self,
        state: Any,
        *,
        timed_solve: TimedSolveFn,
        available_memory_bytes: AvailableMemoryFn,
    ) -> None:
        self._state = state
        self._timed_solve = timed_solve
        self._available_memory_bytes = available_memory_bytes

    def _legacy_connect_hemispheres(self) -> bool:
        """Return whether legacy poloidal feedback closes hemispheres."""
        st = self._state
        return bool(st.connect_hemispheres and st.dynamics_mode != "full_induction")

    def _calculate_total_E_field(
        self,
        E_direct_coeffs: np.ndarray,
        jr_coeffs: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply imposed toroidal baseline to a direct electric field."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        m_imp = st._build_imposed_toroidal_baseline(jr_coeffs, E_direct_coeffs)
        E_imp = st._apply_operator(st.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction feedback."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        if st.u is None:
            E_direct = xp.zeros(E_shape)
        else:
            E_direct = st._apply_operator(
                st.u_coeffs_to_E_coeffs,
                asarray(st.u.coeffs),
                E_shape,
            )
        if st.Br is not None:
            E_direct += st._apply_operator(
                st.Br_to_E_coeffs,
                asarray(st.Br.coeffs),
                E_shape,
            )

        jr_coeffs = None if st.jr is None else asarray(st.jr.coeffs)
        if st.dynamics_mode == "full_induction":
            return self._calculate_dynamic_state(E_direct, jr_coeffs)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def _calculate_dynamic_state(
        self,
        E_direct_coeffs: np.ndarray,
        jr_coeffs: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble non-inductive forcing and update toroidal residual rate."""
        st = self._state
        n = st.solution_space.index_length

        if st.psi is None:
            st.psi = xp.zeros(n)
            st.d_psi_dt = xp.zeros(n)

        E_external = asarray(E_direct_coeffs)
        if jr_coeffs is None:
            m_imp_curr = xp.zeros(n)
            st.m_imp_imposed = asarray(m_imp_curr)
            st._imposed_toroidal_dirty = False
        elif st.m_imp_imposed is None or st._imposed_toroidal_dirty:
            m_imp_curr = st._build_imposed_toroidal_baseline(jr_coeffs, E_external)
            st.m_imp_imposed = asarray(m_imp_curr)
            st._imposed_toroidal_dirty = False
        else:
            m_imp_curr = asarray(st.m_imp_imposed)
            st.m_imp_imposed = asarray(m_imp_curr)

        E_imposed_toroidal = st._apply_operator(st.m_imp_to_E_coeffs, m_imp_curr, (2, n))
        E_noninductive = E_external + E_imposed_toroidal
        st.d_psi_dt = self.solve_dt_psi(E_noninductive)
        return E_noninductive, asarray(m_imp_curr)

    def solve_dt_psi(self, E_known: np.ndarray) -> np.ndarray:
        """Solve constrained system for `dpsi/dt`."""
        st = self._state
        dt_alpha_driver_coeffs = st._get_dt_alpha_driver_coeffs()
        twist_rate_known_grid = None
        dr_twist_rate_known_grid = None
        if st.use_toroidal_twist_rate_known_from_poloidal:
            try:
                (
                    twist_rate_known_grid,
                    dr_twist_rate_known_grid,
                ) = self._build_toroidal_twist_rate_known_terms_from_poloidal(E_known)
            except Exception as exc:
                logger.warning(
                    "Failed to build toroidal u-known RHS terms from poloidal branch: %s",
                    exc,
                )

        rhs_physics = st.toroidal_matrices.compute_toroidal_rhs_from_E(
            asarray(E_known),
            twist_rate_known_grid=twist_rate_known_grid,
            dr_twist_rate_known_grid=dr_twist_rate_known_grid,
            allow_missing_dr_twist_rate_known=False,
        )
        if dt_alpha_driver_coeffs is not None:
            L_alpha = np.asarray(to_numpy(st.toroidal_matrices.dtalpha_operator), dtype=float)
            rhs_physics = np.asarray(rhs_physics).reshape(-1) - L_alpha @ np.asarray(
                dt_alpha_driver_coeffs
            ).reshape(-1)

        solution = st.toroidal_matrices.solve_dt_psi_superposed(
            rhs_physics=rhs_physics,
            rhs_constraint=st._build_dt_alpha_constraint_rhs(dt_alpha_driver_coeffs),
            constraint_operator=st.dt_alpha_constraint_operator_hard,
            m_imp_to_jr_operator=st.poloidal_matrices.m_imp_to_jr,
            weighting=st.toroidal_weighting,
            regularization_lambda=st.toroidal_regularization_lambda,
            penalty_operator=None,
            penalty_scaling=0.0,
            hinv_rtol=0.0,
            use_pinning=st.apply_psi_gauge,
        )
        if solution is None:
            raise RuntimeError("Toroidal superposed dpsi/dt solve returned no solution.")
        return asarray(solution)

    def _build_toroidal_twist_rate_known_terms_from_poloidal(
        self,
        E_known: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build toroidal `(twist_rate_known_grid, dr_twist_rate_known_grid)` from poloidal forcing."""
        st = self._state
        E_df_known = asarray(st.solution_space.get_toroidal_potential_coeffs(E_known))
        dt_m_ind_known = asarray(st.poloidal_matrices.E_df_to_d_m_ind_dt * E_df_known)
        analysis_basis = getattr(st.toroidal_matrices, "rhs_derivative_basis", st.solution_space)
        return st.poloidal_matrices.build_toroidal_twist_rate_known_terms_from_dt_m_ind(
            dt_m_ind_known,
            analysis_basis=analysis_basis,
            radial_model=st.toroidal_twist_rate_known_radial_model,
        )

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total induced E-field coefficients."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        E_direct_ind = st._apply_operator(st.m_ind_to_E_coeffs, asarray(m_ind), E_shape)
        if st.dynamics_mode == "full_induction":
            return E_direct_ind, xp.zeros(st.solution_space.index_length)
        return self._calculate_total_E_field(E_direct_ind, None)

    def calculate_psi_E_coeffs(self, psi: np.ndarray) -> np.ndarray:
        """Map inductive toroidal residual `psi` to E-field coefficients."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        return st._apply_operator(st.toroidal_to_E_coeffs, asarray(psi), E_shape)

    def build_m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix mapping `m_ind` to div-free E-field."""
        st = self._state
        return st.poloidal_matrices.build_induction_matrix(
            problem=st.m_imp_problem,
            solver=st.m_imp_solver,
            E_map_constraint_operator=st.geometry.E_coeffs_to_E_apex_ll_diff,
            ih_constraint_scaling=st.ih_constraint_scaling,
            connect_hemispheres=self._legacy_connect_hemispheres(),
            m_ind_to_E_operator=st.m_ind_to_E_coeffs,
            m_imp_to_E_operator=st.m_imp_to_E_coeffs,
        )

    def build_E_coeffs_to_E_df_matrix(self) -> np.ndarray:
        """Operator extracting toroidal potential (`E_df`) from vector coefficients."""
        st = self._state
        N = st.solution_space.index_length
        kind = getattr(st.solution_space, "kind", "")

        if kind == "SH":
            zeros = np.zeros((N, N))
            eye = np.eye(N)
            return asarray(np.hstack([zeros, eye]))

        if kind in ("CS", "GRID"):
            P = st.solution_space.construct_projection_matrix(st.geometry.grid)
            if P.ndim != 4 or P.shape[0] != 2 or P.shape[2] != 2:
                raise ValueError(
                    "Projection matrix must have canonical shape (2, n_coeffs, 2, n_grid), "
                    f"got {getattr(P, 'shape', None)}."
                )
            return asarray(P[1].reshape(N, 2 * P.shape[3]))

        M = np.zeros((N, 2 * N))
        for i in range(2 * N):
            e_i = np.zeros(2 * N)
            e_i[i] = 1.0
            coeffs = e_i.reshape(2, N)
            M[:, i] = asarray(st.solution_space.get_toroidal_potential_coeffs(coeffs))
        return asarray(M)

    def get_induction_operator(self) -> LinearMap:
        """Get matrix-free induction operator (`m_ind -> E_df`)."""
        st = self._state
        return st.poloidal_matrices.get_induction_operator(
            problem=st.m_imp_problem,
            solver=st.m_imp_solver,
            preconditioner=st.m_imp_preconditioner,
            E_map_constraint_operator=st.geometry.E_coeffs_to_E_apex_ll_diff,
            ih_constraint_scaling=st.ih_constraint_scaling,
            connect_hemispheres=self._legacy_connect_hemispheres(),
            m_ind_to_E_operator=st.m_ind_to_E_coeffs,
            m_imp_to_E_operator=st.m_imp_to_E_coeffs,
        )

    def build_coupled_forcing(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Build coupled forcing tensor `K` for `[psi, m_ind]` dynamics."""
        st = self._state
        scale = st.poloidal_matrices.E_df_to_d_m_ind_dt
        E_noind_field = st.poloidal_matrices.solution_space.get_toroidal_potential_coeffs(
            E_coeffs_noind
        )
        k1 = asarray(scale * E_noind_field)
        if st.d_psi_dt is not None:
            k0 = asarray(st.d_psi_dt)
        else:
            k0 = xp.zeros_like(k1)
        return xp.stack([k0, k1])

    def solve_steady_state_model_variables(
        self,
        E_coeffs_noind: np.ndarray,
        *,
        update_state: bool = True,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Compute steady-state initialization for the current dynamics mode."""
        st = self._state
        N = st.solution_space.index_length
        if st.dynamics_mode == "full_induction":
            y_ss = st._solve_linear_steady_state(
                linear_operator=None,
                forcing=self.build_coupled_forcing(E_coeffs_noind),
                solution_shape=(2, N),
                solver=st.solver_type,
                use_pinning=st.apply_psi_gauge,
            )
            psi = asarray(y_ss[0])
            m_ind = asarray(y_ss[1])
            if update_state:
                st.psi = psi
            return psi, m_ind

        k_legacy = asarray(
            st.poloidal_matrices.solution_space.get_toroidal_potential_coeffs(E_coeffs_noind)
        )
        m_ss = st._solve_linear_steady_state(
            linear_operator=st.m_ind_to_E_df_matrix,
            forcing=k_legacy,
            solution_shape=(N,),
            solver=st.solver_type,
        )
        return None, asarray(m_ss)

    def evolve_model_variables(
        self,
        m_ind: np.ndarray,
        dt: float,
        E_coeffs_noind: np.ndarray,
        *,
        steady_state_m_ind: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Advance model variables by one time step."""
        st = self._state
        if st.dynamics_mode == "full_induction":
            if psi is None:
                if st.psi is None:
                    st.psi = xp.zeros((st.solution_space.index_length,))
                psi = st.psi

            y = xp.stack([asarray(psi), asarray(m_ind)])
            K = self.build_coupled_forcing(E_coeffs_noind)
            if isinstance(st.poloidal_integrator, ExponentialIntegrator):
                use_pinning = st.apply_psi_gauge
                N = st.solution_space.index_length
                m = 2 * N
                exp_kwargs: Dict[str, Any] = {
                    "max_step_scale": 10.0,
                    "max_substeps": 32768,
                }
                if st.exponential_solver == "expm":
                    exp_kwargs["affine_expm_mode"] = "dense"
                elif st.exponential_solver == "expm_multiply":
                    exp_kwargs["affine_expm_mode"] = "action"
                else:
                    raise ValueError(
                        f"Unknown exponential_solver setting: {st.exponential_solver!r}"
                    )

                use_dense_coupled_operator = bool(st.dense_full_operators)
                avail_bytes = self._available_memory_bytes()
                if use_dense_coupled_operator and avail_bytes is not None:
                    matrix_bytes = int(m) * int(m) * np.dtype(float).itemsize
                    peak_factor = 3 if st.exponential_solver == "expm_multiply" else 8
                    estimated_peak_bytes = int(peak_factor * matrix_bytes)
                    if estimated_peak_bytes > int(0.80 * avail_bytes):
                        need_gib = estimated_peak_bytes / float(1024 ** 3)
                        avail_gib = avail_bytes / float(1024 ** 3)
                        raise MemoryError(
                            "Coupled exponential step would likely exceed available memory: "
                            f"need ~{need_gib:.2f} GiB, available ~{avail_gib:.2f} GiB. "
                            "Reduce resolution or use a non-exponential integrator for this run."
                        )

                coupled_dynamics_operator = st.get_coupled_operator_for_time_integration(
                    use_dense=use_dense_coupled_operator,
                    use_pinning=use_pinning,
                )
                if st.induction_null_diagnostics:
                    if st._coupled_null_basis is None or st._coupled_null_basis.shape[0] != m:
                        diag_dense = None
                        if m <= 2000:
                            diag_dense = np.asarray(
                                st._densify_linear_operator(coupled_dynamics_operator, m)
                            )
                        if diag_dense is not None:
                            st._update_coupled_null_basis(np.asarray(diag_dense))
                    st._check_forcing_null_projection(np.asarray(K).reshape(m))

                y_new = st._evolve_linear_state(
                    y=np.asarray(y).reshape(m),
                    dt=float(dt),
                    linear_operator=coupled_dynamics_operator,
                    forcing=np.asarray(K).reshape(m),
                    exponential_kwargs=exp_kwargs,
                ).reshape(2, N)
            else:
                coupled_operator = st.get_coupled_operator_for_time_integration(
                    use_dense=st.dense_full_operators,
                    use_pinning=st.apply_psi_gauge,
                )
                y_new = st._evolve_linear_state(
                    y=y,
                    dt=dt,
                    linear_operator=coupled_operator,
                    forcing=K,
                )
            psi_new = asarray(y_new[0])
            m_ind_new = asarray(y_new[1])
            st.psi = psi_new
            return psi_new, m_ind_new

        use_dense_rate_operator = bool(
            st.dense_full_operators or isinstance(st.poloidal_integrator, ExponentialIntegrator)
        )
        forcing = None
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]]
        if use_dense_rate_operator:
            scale = st.poloidal_matrices.E_df_to_d_m_ind_dt
            linear_operator = scale * asarray(st.m_ind_to_E_df_matrix)
            E_noind_field = st.poloidal_matrices.solution_space.get_toroidal_potential_coeffs(
                E_coeffs_noind
            )
            forcing = asarray(scale * E_noind_field)
            rates_func = None
        else:
            linear_operator = None

            def rates_func(y: np.ndarray, t: float) -> np.ndarray:
                return st.poloidal_matrices.compute_rates(
                    m_ind=y,
                    t=t,
                    E_coeffs_noind=E_coeffs_noind,
                    induction_matrix=None,
                    m_ind_to_E_operator=st.m_ind_to_E_coeffs,
                    problem=st.m_imp_problem,
                    solver=st.m_imp_solver,
                    preconditioner=st.m_imp_preconditioner,
                    E_map_constraint_operator=st.geometry.E_coeffs_to_E_apex_ll_diff,
                    ih_constraint_scaling=st.ih_constraint_scaling,
                    connect_hemispheres=self._legacy_connect_hemispheres(),
                    m_imp_to_E_operator=st.m_imp_to_E_coeffs,
                )

        if isinstance(st.poloidal_integrator, ExponentialIntegrator) and linear_operator is None:
            scale = st.poloidal_matrices.E_df_to_d_m_ind_dt
            linear_operator = scale * st.m_ind_to_E_df_matrix

        m_ind_new = st._evolve_linear_state(
            y=asarray(m_ind),
            dt=dt,
            linear_operator=linear_operator,
            forcing=forcing,
            rates_func=rates_func,
            steady_state=asarray(steady_state_m_ind) if steady_state_m_ind is not None else None,
        )
        m_ind_new = st.constraints.apply_m_ind_gauge_projection(m_ind_new)
        return None, asarray(m_ind_new)
