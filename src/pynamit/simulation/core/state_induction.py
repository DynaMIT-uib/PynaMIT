"""State-level induction and coupled-evolution orchestration helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pynamit.math.integration import ExponentialIntegrator, ScipySolveIVPIntegrator
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.primitives.basis import is_cs_like_basis, is_sh_basis
from pynamit.simulation.induction.poloidal_solver import MImpFeedbackSystem
from pynamit.simulation.core.coupled_solver import CoupledSteadyStateSolver
from pynamit.simulation.settings import DynamicsMode, ExponentialSolverKind, LLConstraintMode
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
        timed_structured_solve: TimedSolveFn,
        available_memory_bytes: AvailableMemoryFn,
    ) -> None:
        self._state = state
        self._timed_solve = timed_solve
        self._timed_structured_solve = timed_structured_solve
        self._available_memory_bytes = available_memory_bytes

    def _legacy_connect_hemispheres(self) -> bool:
        """Return whether legacy poloidal feedback closes hemispheres."""
        st = self._state
        return bool(
            st.connect_hemispheres
            and st.dynamics_mode != DynamicsMode.FULL_INDUCTION
            and st.get_effective_ll_constraint_mode() != LLConstraintMode.OFF
        )

    def build_m_imp_feedback_system(self) -> MImpFeedbackSystem:
        """Build the bundled reduced ``m_imp`` feedback solve definition."""
        st = self._state
        logger.info("Defining new least-squares problem for m_imp.")
        m_imp_reduced_system = st.get_m_imp_reduced_system()

        e_constraint_op = None
        if (
            st.connect_hemispheres
            and st.dynamics_mode != DynamicsMode.FULL_INDUCTION
            and st.E_map_constraint_operator is not None
        ):
            e_constraint_op = st.E_map_constraint_operator
        ll_mode = st.get_effective_ll_constraint_mode()

        constraint_scalar_operator = st.geometry.get_constraint_scalar_operator(st.solution_space)

        subproblem = st.geometry.poloidal_matrices.build_least_squares_subproblem(
            constraint_scalar_operator=constraint_scalar_operator,
            E_constraint_operator=e_constraint_op,
            connect_hemispheres=(e_constraint_op is not None and ll_mode == LLConstraintMode.SOFT),
            ih_constraint_scaling=st.ih_constraint_scaling,
            regularization_lambda=st.m_imp_regularization_lambda,
            m_imp_selector=np.asarray(m_imp_reduced_system.selector, dtype=float),
            weighting=st.poloidal_weighting,
        )
        preconditioner = st.m_imp_solver.build_preconditioner(
            problem=subproblem.problem, num_scenarios=1
        )
        solve_system = subproblem.with_equality()
        if e_constraint_op is not None and ll_mode == LLConstraintMode.HARD:
            selector = np.asarray(m_imp_reduced_system.selector, dtype=float)
            equality_operator = as_linear_map(e_constraint_op) @ as_linear_map(selector)
            solve_system = solve_system.with_equality(equality_operator=equality_operator)
        return MImpFeedbackSystem(
            solve_system=solve_system,
            selector=np.asarray(m_imp_reduced_system.selector, dtype=float),
            preconditioner=preconditioner,
        )

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply imposed toroidal baseline to a direct electric field."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        m_imp = self.build_imposed_toroidal_baseline(jr_coeffs, E_direct_coeffs)
        E_imp = st._apply_operator(st.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction feedback."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        if st.u is None:
            E_direct = xp.zeros(E_shape)
        else:
            E_direct = st._apply_operator(st.u_coeffs_to_E_coeffs, asarray(st.u.coeffs), E_shape)
        if st.Br is not None:
            E_direct += st._apply_operator(st.Br_to_E_coeffs, asarray(st.Br.coeffs), E_shape)

        jr_coeffs = None if st.jr is None else asarray(st.jr.coeffs)
        if st.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            return self._calculate_dynamic_state(E_direct, jr_coeffs)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def _calculate_dynamic_state(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
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
            m_imp_curr = self.build_imposed_toroidal_baseline(jr_coeffs, E_external)
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
        constraint_system = st.dt_alpha_constraint_system
        dt_alpha_driver_coeffs = st._get_dt_alpha_driver_coeffs()
        rhs_physics = st.toroidal_matrices.compute_toroidal_rhs_from_E(asarray(E_known))
        if dt_alpha_driver_coeffs is not None:
            L_alpha = np.asarray(to_numpy(st.toroidal_matrices.dtalpha_operator), dtype=float)
            rhs_physics = np.asarray(rhs_physics).reshape(-1) - L_alpha @ np.asarray(
                dt_alpha_driver_coeffs
            ).reshape(-1)

        solution = st.toroidal_matrices.solve_dt_psi_superposed(
            rhs_physics=rhs_physics,
            rhs_constraint=constraint_system.build_hard_rhs(dt_alpha_driver_coeffs),
            constraint_operator=constraint_system.hard_operator,
            m_imp_to_jr_operator=st.poloidal_matrices.m_imp_to_jr,
            weighting=st.toroidal_weighting,
            regularization_lambda=st.toroidal_regularization_lambda,
            penalty_operator=constraint_system.soft_operator,
            penalty_scaling=float(constraint_system.soft_scaling),
            penalty_rhs=constraint_system.build_soft_rhs(dt_alpha_driver_coeffs),
            hinv_rtol=0.0,
            apply_psi_gauge=st.apply_psi_gauge,
        )
        if solution is None:
            raise RuntimeError("Toroidal superposed dpsi/dt solve returned no solution.")
        return asarray(solution)

    def build_imposed_toroidal_baseline(
        self, jr_coeffs: Optional[np.ndarray], E_direct_coeffs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Build imposed toroidal baseline ``m_imp`` from external driver inputs."""
        st = self._state
        n = st.solution_space.index_length
        if st.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            if jr_coeffs is None:
                return xp.zeros(n)
            input_basis = st.jr.spec if st.jr is not None else None
            m_imp_from_jr = np.asarray(st.get_m_imp_from_jr_matrix(input_basis=input_basis))
            jr_vec = np.asarray(jr_coeffs).reshape(-1)
            return asarray(m_imp_from_jr @ jr_vec)

        ll_mode = st.get_effective_ll_constraint_mode()
        use_e_constraint = (
            st.connect_hemispheres
            and st.E_map_constraint_operator is not None
            and ll_mode != LLConstraintMode.OFF
        )
        if jr_coeffs is None and not use_e_constraint:
            return xp.zeros(n)
        feedback_system = st.m_imp_feedback_system

        rhs_entries: list[Optional[Any]] = [None] * feedback_system.problem.num_data_terms
        equality_rhs_input: Optional[Any] = None
        if jr_coeffs is not None:
            op_rhs = st.geometry.get_constraint_scalar_operator(st.jr.spec if st.jr else None)
            rhs_entries[0] = as_linear_map(op_rhs).matvec(asarray(jr_coeffs).reshape(-1))

        if use_e_constraint:
            if E_direct_coeffs is None:
                raise ValueError(
                    "E_direct_coeffs is required for imposed baseline solve with IH E-constraint."
                )
            e_map_op = st.geometry.E_coeffs_to_E_apex_ll_diff
            e_direct_input = asarray(E_direct_coeffs)
            if hasattr(e_map_op, "apply"):
                b_E = e_map_op.apply(e_direct_input)
            else:
                raise TypeError(
                    "E_coeffs_to_E_apex_ll_diff must provide an 'apply' method "
                    "(ConstraintOperator)."
                )
            b_E_flat = xp.reshape(b_E, (-1,))
            if ll_mode == LLConstraintMode.SOFT:
                rhs_entries[1] = st.ih_constraint_scaling * b_E_flat
            elif ll_mode == LLConstraintMode.HARD:
                equality_rhs_input = b_E_flat
        if equality_rhs_input is not None and all(entry is None for entry in rhs_entries):
            rhs_entries[0] = xp.zeros(int(feedback_system.problem.A[0].num_rows), dtype=float)

        solution = self._timed_structured_solve(
            "state.m_imp",
            feedback_system.solve_system,
            st.m_imp_solver,
            rhs_entries,
            preconditioner=feedback_system.preconditioner,
            equality_rhs_input=equality_rhs_input,
        )
        if solution is None:
            solution = xp.zeros(feedback_system.problem.solution_size)
        return asarray(feedback_system.expand_solution(solution))

    def map_dt_jr_driver_to_dt_m_imp(self, dt_jr_coeffs: np.ndarray) -> np.ndarray:
        """Map driver derivative ``dt_jr`` to toroidal driver derivative ``dt_m_imp``."""
        st = self._state
        dt_jr_vec = np.asarray(dt_jr_coeffs).reshape(-1)
        m_imp_from_jr = np.asarray(st.get_m_imp_from_jr_matrix(input_basis=st.solution_space))
        if m_imp_from_jr.ndim != 2 or m_imp_from_jr.shape[1] != dt_jr_vec.size:
            raise RuntimeError(
                "dt_jr -> dt_m_imp mapping dimension mismatch: "
                f"map={m_imp_from_jr.shape}, driver={dt_jr_vec.shape}."
            )
        return asarray(m_imp_from_jr @ dt_jr_vec)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total induced E-field coefficients."""
        st = self._state
        E_shape = (2, st.solution_space.index_length)
        E_direct_ind = st._apply_operator(st.m_ind_to_E_coeffs, asarray(m_ind), E_shape)
        if st.dynamics_mode == DynamicsMode.FULL_INDUCTION:
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
            feedback_system=st.m_imp_feedback_system,
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
        if is_sh_basis(st.solution_space):
            zeros = np.zeros((N, N))
            eye = np.eye(N)
            return asarray(np.hstack([zeros, eye]))

        if is_cs_like_basis(st.solution_space):
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
            feedback_system=st.m_imp_feedback_system,
            solver=st.m_imp_solver,
            E_map_constraint_operator=st.geometry.E_coeffs_to_E_apex_ll_diff,
            ih_constraint_scaling=st.ih_constraint_scaling,
            connect_hemispheres=self._legacy_connect_hemispheres(),
            m_ind_to_E_operator=st.m_ind_to_E_coeffs,
            m_imp_to_E_operator=st.m_imp_to_E_coeffs,
        )

    def apply_state_linear_operator(
        self, operator: Any, state: np.ndarray, output_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Apply a state-space linear operator to a flattened/stacked state."""
        state_arr = asarray(state)
        state_shape = tuple(state_arr.shape)
        state_flat = state_arr.reshape(-1)

        if hasattr(operator, "matvec"):
            out_flat = asarray(operator.matvec(state_flat)).reshape(-1)
        else:
            op_arr = asarray(operator)
            if op_arr.ndim == 4:
                if state_arr.ndim != 2:
                    raise ValueError(
                        "4D coupled operator requires 2D state shaped (n_state, n_coeffs)."
                    )
                out_arr = asarray(xp.einsum("ijkl,kl->ij", op_arr, state_arr, optimize=True))
                return out_arr.reshape(output_shape or state_shape)
            out_flat = asarray(op_arr).reshape(state_flat.size, state_flat.size) @ state_flat

        return asarray(out_flat).reshape(output_shape or state_shape)

    def evolve_linear_state(
        self,
        y: np.ndarray,
        dt: float,
        *,
        linear_operator: Optional[Any] = None,
        forcing: Optional[np.ndarray] = None,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        steady_state: Optional[np.ndarray] = None,
        exponential_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Shared linear-state evolution for legacy and full-induction paths."""
        st = self._state
        y_arr = asarray(y)
        y_shape = tuple(y_arr.shape)
        forcing_arr = (
            xp.zeros_like(y_arr) if forcing is None else asarray(forcing).reshape(y_shape)
        )

        if isinstance(st.poloidal_integrator, ExponentialIntegrator):
            if linear_operator is None:
                raise ValueError("Exponential integration requires linear_operator.")

            step_kwargs: Dict[str, Any] = dict(exponential_kwargs or {})
            if "affine_expm_mode" not in step_kwargs:
                if st.exponential_solver == ExponentialSolverKind.EXPM:
                    step_kwargs["affine_expm_mode"] = "dense"
                elif st.exponential_solver == ExponentialSolverKind.EXPM_MULTIPLY:
                    step_kwargs["affine_expm_mode"] = "action"
                else:
                    raise ValueError(
                        f"Unknown exponential_solver setting: {st.exponential_solver!r}"
                    )
            affine_mode = str(step_kwargs.get("affine_expm_mode", "auto")).lower()

            n_total = int(y_arr.size)
            if forcing is not None and affine_mode == "action":
                if hasattr(linear_operator, "matvec"):
                    linear_operator_for_step = linear_operator
                else:
                    linear_operator_for_step = np.asarray(
                        asarray(linear_operator), dtype=float
                    ).reshape(n_total, n_total)
            else:
                if hasattr(linear_operator, "matvec"):
                    l_dense = st._densify_linear_operator(linear_operator, n_total)
                else:
                    l_dense = asarray(linear_operator).reshape(n_total, n_total)
                linear_operator_for_step = np.asarray(l_dense, dtype=float)

            forcing_flat = (
                None
                if forcing is None
                else np.asarray(asarray(forcing_arr), dtype=float).reshape(n_total)
            )
            steady_state_flat = (
                None
                if steady_state is None
                else np.asarray(steady_state, dtype=float).reshape(n_total)
            )
            if forcing_flat is None and steady_state_flat is None:
                raise ValueError(
                    "Exponential integration requires either forcing or steady_state."
                )
            y_next_flat = st.poloidal_integrator.step(
                y=np.asarray(y_arr, dtype=float).reshape(n_total),
                dt=float(dt),
                linear_operator=linear_operator_for_step,
                forcing=forcing_flat,
                steady_state=steady_state_flat,
                **step_kwargs,
            )
            return asarray(y_next_flat).reshape(y_shape)

        if rates_func is None:
            if linear_operator is None:
                raise ValueError("Either rates_func or linear_operator must be provided.")

            def default_rates_func(y_curr: np.ndarray, _t: float) -> np.ndarray:
                y_curr_arr = asarray(y_curr).reshape(y_shape)
                rates = self.apply_state_linear_operator(
                    linear_operator, y_curr_arr, output_shape=y_shape
                )
                return asarray(rates + forcing_arr)

            rates = default_rates_func
        else:
            rates = rates_func

        if isinstance(st.poloidal_integrator, ScipySolveIVPIntegrator):
            y0_flat = asarray(y_arr).reshape(-1)

            def scipy_rates(y_curr: np.ndarray, t_curr: float) -> np.ndarray:
                y_curr_shaped = asarray(y_curr).reshape(y_shape)
                rates_curr = asarray(rates(y_curr_shaped, t_curr)).reshape(y_shape)
                return asarray(rates_curr).reshape(-1)

            y_next_flat = st.poloidal_integrator.step(y=y0_flat, dt=dt, rates_func=scipy_rates)
            return asarray(y_next_flat).reshape(y_shape)

        return asarray(st.poloidal_integrator.step(y=y_arr, dt=dt, rates_func=rates)).reshape(
            y_shape
        )

    def solve_linear_steady_state(
        self,
        *,
        linear_operator: Any,
        forcing: np.ndarray,
        solution_shape: Tuple[int, ...],
        solver: Optional[str] = None,
        preconditioner: Optional[LinearMap] = None,
    ) -> np.ndarray:
        """Solve a linear steady-state system ``A x = -forcing``."""
        st = self._state
        rhs = asarray(forcing).reshape(-1)
        n_total = int(np.prod(solution_shape))

        if (
            len(solution_shape) == 2
            and int(solution_shape[0]) == 2
            and int(solution_shape[1]) == st.solution_space.index_length
        ):
            apply_psi_gauge = bool(st.apply_psi_gauge)
            if solver is None:
                solver = st.solver_type

            coupled_operator = linear_operator
            using_default_coupled_operator = False
            if coupled_operator is None:
                using_default_coupled_operator = True
                coupled_operator = st.get_coupled_operator_for_steady_state(solver=solver)
            steady_solver = CoupledSteadyStateSolver(
                n_scalar=st.solution_space.index_length,
                apply_m_ind_gauge=st.apply_m_ind_gauge,
                preconditioner_type=st.preconditioner,
                psi_gauge_row_builder=st.constraints.get_psi_gauge_row,
                m_ind_gauge_row_builder=st.constraints.get_m_ind_gauge_row,
                timed_solve=self._timed_solve,
                column_scale_cache=st._coupled_steady_state_column_scale_cache,
                solver_tolerance=float(getattr(st.m_imp_solver, "tolerance", 1e-13)),
                steady_state_regularization_lambda=1e-10,
            )
            column_scale_cache_key = None
            if using_default_coupled_operator:
                column_scale_cache_key = (apply_psi_gauge, int(st.solution_space.index_length))
            y_ss_flat = steady_solver.solve(
                coupled_operator=coupled_operator,
                forcing_flat=rhs,
                solver=solver,
                preconditioner=preconditioner,
                apply_psi_gauge=apply_psi_gauge,
                column_scale_cache_key=column_scale_cache_key,
            )
            return asarray(y_ss_flat).reshape(solution_shape)

        if linear_operator is None:
            raise ValueError("Single-state steady-state solve requires linear_operator.")

        vec_b = -rhs
        reduced_system = st.get_m_ind_reduced_system(linear_operator=linear_operator)
        if reduced_system.n_reduced == 0:
            return asarray(np.zeros((n_total,), dtype=float)).reshape(solution_shape)

        induction_op = reduced_system.reduced_operator
        if induction_op is None:
            raise RuntimeError("Reduced m_ind steady-state solve requires a reduced operator.")

        ls_problem = LeastSquaresProblem(
            A=[induction_op],
            solution_shape=(reduced_system.n_reduced,),
            data_shapes=[(reduced_system.n_reduced,)],
        )
        ls_solver = LeastSquaresSolver(solver=(solver or "lsmr"), tolerance=1e-10)
        sol_reduced = self._timed_solve(
            "state.steady_state_single",
            ls_solver,
            ls_problem,
            [reduced_system.reduce_vector(vec_b)],
            preconditioner=preconditioner if reduced_system.n_reduced == n_total else None,
        )
        return asarray(reduced_system.expand_vector(sol_reduced)).reshape(solution_shape)

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
        self, E_coeffs_noind: np.ndarray, *, update_state: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Compute steady-state initialization for the current dynamics mode."""
        st = self._state
        N = st.solution_space.index_length
        if st.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            y_ss = self.solve_linear_steady_state(
                linear_operator=None,
                forcing=self.build_coupled_forcing(E_coeffs_noind),
                solution_shape=(2, N),
                solver=st.solver_type,
            )
            psi = asarray(y_ss[0])
            m_ind = asarray(y_ss[1])
            if update_state:
                st.psi = psi
            return psi, m_ind

        k_legacy = asarray(
            st.poloidal_matrices.solution_space.get_toroidal_potential_coeffs(E_coeffs_noind)
        )
        m_ss = self.solve_linear_steady_state(
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
        steady_state_psi: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Advance model variables by one time step."""
        st = self._state
        if st.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            if psi is None:
                if st.psi is None:
                    st.psi = xp.zeros((st.solution_space.index_length,))
                psi = st.psi

            y = xp.stack([asarray(psi), asarray(m_ind)])
            K = self.build_coupled_forcing(E_coeffs_noind)
            if isinstance(st.poloidal_integrator, ExponentialIntegrator):
                N = st.solution_space.index_length
                exp_kwargs: Dict[str, Any] = {"max_step_scale": 10.0, "max_substeps": 32768}
                if st.exponential_solver == ExponentialSolverKind.EXPM:
                    exp_kwargs["affine_expm_mode"] = "dense"
                elif st.exponential_solver == ExponentialSolverKind.EXPM_MULTIPLY:
                    exp_kwargs["affine_expm_mode"] = "action"
                else:
                    raise ValueError(
                        f"Unknown exponential_solver setting: {st.exponential_solver!r}"
                    )

                use_dense_coupled_operator = bool(st.dense_full_operators)
                reduced_system = st.get_coupled_reduced_time_integration_system(
                    use_dense=use_dense_coupled_operator
                )
                m_reduced = reduced_system.n_reduced
                avail_bytes = self._available_memory_bytes()
                if use_dense_coupled_operator and avail_bytes is not None:
                    matrix_bytes = int(m_reduced) * int(m_reduced) * np.dtype(float).itemsize
                    peak_factor = (
                        3 if st.exponential_solver == ExponentialSolverKind.EXPM_MULTIPLY else 8
                    )
                    estimated_peak_bytes = int(peak_factor * matrix_bytes)
                    if estimated_peak_bytes > int(0.80 * avail_bytes):
                        need_gib = estimated_peak_bytes / float(1024**3)
                        avail_gib = avail_bytes / float(1024**3)
                        raise MemoryError(
                            "Coupled exponential step would likely exceed available memory: "
                            f"need ~{need_gib:.2f} GiB, available ~{avail_gib:.2f} GiB. "
                            "Reduce resolution or use a non-exponential integrator for this run."
                        )

                y_reduced = reduced_system.reduce_vector(y)
                K_reduced = reduced_system.reduce_vector(K)
                st.run_coupled_null_diagnostics(reduced_system.reduced_operator, K_reduced)
                y_new_reduced = self.evolve_linear_state(
                    y=y_reduced,
                    dt=float(dt),
                    linear_operator=reduced_system.reduced_operator,
                    forcing=K_reduced,
                    exponential_kwargs=exp_kwargs,
                )
                y_new = reduced_system.expand_vector(y_new_reduced).reshape(2, N)
            else:
                reduced_system = st.get_coupled_reduced_time_integration_system(
                    use_dense=st.dense_full_operators
                )
                y_reduced = reduced_system.reduce_vector(y)
                K_reduced = reduced_system.reduce_vector(K)
                st.run_coupled_null_diagnostics(reduced_system.reduced_operator, K_reduced)
                y_new_reduced = self.evolve_linear_state(
                    y=y_reduced,
                    dt=dt,
                    linear_operator=reduced_system.reduced_operator,
                    forcing=K_reduced,
                )
                N = st.solution_space.index_length
                y_new = reduced_system.expand_vector(y_new_reduced).reshape(2, N)
            psi_new = asarray(y_new[0])
            m_ind_new = asarray(y_new[1])
            st.psi = psi_new
            return psi_new, m_ind_new

        use_exponential_integrator = isinstance(st.poloidal_integrator, ExponentialIntegrator)
        use_frozen_steady_state = use_exponential_integrator and steady_state_m_ind is not None
        use_dense_rate_operator = bool(st.dense_full_operators or use_exponential_integrator)

        if use_dense_rate_operator:
            scale = st.poloidal_matrices.E_df_to_d_m_ind_dt
            full_linear_operator = scale * asarray(st.m_ind_to_E_df_matrix)
            reduced_system = st.get_m_ind_reduced_system(linear_operator=full_linear_operator)
            if reduced_system.n_reduced == 0:
                return None, asarray(reduced_system.expand_vector(np.zeros((0,), dtype=float)))

            forcing_reduced = None
            if not use_frozen_steady_state:
                E_noind_field = st.poloidal_matrices.solution_space.get_toroidal_potential_coeffs(
                    E_coeffs_noind
                )
                full_forcing = asarray(scale * E_noind_field)
                forcing_reduced = reduced_system.reduce_vector(full_forcing)

            steady_state_reduced = (
                reduced_system.reduce_vector(steady_state_m_ind)
                if steady_state_m_ind is not None
                else None
            )
            m_ind_new_reduced = self.evolve_linear_state(
                y=reduced_system.reduce_vector(asarray(m_ind)),
                dt=dt,
                linear_operator=reduced_system.reduced_operator,
                forcing=forcing_reduced,
                rates_func=None,
                steady_state=steady_state_reduced,
            )
            return None, asarray(reduced_system.expand_vector(m_ind_new_reduced))

        reduced_system = st.get_m_ind_reduced_system()
        if reduced_system.n_reduced == 0:
            return None, asarray(reduced_system.expand_vector(np.zeros((0,), dtype=float)))

        def full_rates_func(y: np.ndarray, t: float) -> np.ndarray:
            return st.poloidal_matrices.compute_rates(
                m_ind=y,
                t=t,
                E_coeffs_noind=E_coeffs_noind,
                induction_matrix=None,
                m_ind_to_E_operator=st.m_ind_to_E_coeffs,
                feedback_system=st.m_imp_feedback_system,
                solver=st.m_imp_solver,
                E_map_constraint_operator=st.geometry.E_coeffs_to_E_apex_ll_diff,
                ih_constraint_scaling=st.ih_constraint_scaling,
                connect_hemispheres=self._legacy_connect_hemispheres(),
                m_imp_to_E_operator=st.m_imp_to_E_coeffs,
            )

        def reduced_rates_func(y_reduced: np.ndarray, t: float) -> np.ndarray:
            y_full = reduced_system.expand_vector(y_reduced)
            rates_full = full_rates_func(y_full, t)
            return reduced_system.reduce_vector(rates_full)

        m_ind_new_reduced = self.evolve_linear_state(
            y=reduced_system.reduce_vector(asarray(m_ind)),
            dt=dt,
            linear_operator=None,
            forcing=None,
            rates_func=reduced_rates_func,
            steady_state=None,
        )
        return None, asarray(reduced_system.expand_vector(m_ind_new_reduced))
