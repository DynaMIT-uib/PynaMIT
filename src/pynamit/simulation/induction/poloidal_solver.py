"""Poloidal solve/orchestration helpers.

This module keeps `PoloidalSystemMatrices` focused on operator assembly while
collecting feedback solves, induction-operator exposure, and steady-state/rate
orchestration in one small helper object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from pynamit.math.linear_map import as_linear_map
from pynamit.math.linear_map import LinearMap
from pynamit.math.structured_least_squares import (
    ConstrainedStructuredLeastSquaresSubproblem,
    StructuredLeastSquaresSubproblem,
)
from pynamit.simulation.induction.operator_utils import (
    build_linear_map,
    coerce_dense_operator_matrix,
)
from pynamit.utils import asarray, xp

if TYPE_CHECKING:
    from pynamit.math.least_squares_problem import LeastSquaresProblem
    from pynamit.simulation.induction.poloidal import PoloidalSystemMatrices

TimedSolveFn = Callable[..., np.ndarray]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MImpFeedbackSystem:
    """Bundle the reduced ``m_imp`` solve definition with selector-based mappings."""

    solve_system: ConstrainedStructuredLeastSquaresSubproblem
    selector: Optional[np.ndarray] = None
    preconditioner: Optional[LinearMap] = None

    @property
    def subproblem(self) -> StructuredLeastSquaresSubproblem:
        """Compatibility view of the unconstrained structured subproblem."""
        return self.solve_system.subproblem

    @property
    def problem(self) -> "LeastSquaresProblem":
        """Compatibility view of the underlying least-squares problem."""
        return self.solve_system.problem

    @property
    def full_size(self) -> int:
        """Return the full ``m_imp`` coefficient dimension."""
        if self.selector is None:
            return int(self.solve_system.solution_size)
        return int(np.asarray(self.selector, dtype=float).shape[0])

    @property
    def reduced_size(self) -> int:
        """Return the reduced ``m_imp`` coefficient dimension."""
        if self.selector is None:
            return int(self.solve_system.solution_size)
        return int(np.asarray(self.selector, dtype=float).shape[1])

    def reduce_solution(self, m_imp_solution: np.ndarray) -> np.ndarray:
        """Reduce full ``m_imp`` coefficients to the gauge-constrained coordinates."""
        m_imp_arr = asarray(m_imp_solution)
        if self.selector is None:
            return asarray(m_imp_arr)

        selector = np.asarray(self.selector, dtype=float)
        if m_imp_arr.ndim == 1:
            if m_imp_arr.size != selector.shape[0]:
                raise ValueError(
                    "Full m_imp vector has incompatible size: "
                    f"{m_imp_arr.size} != {selector.shape[0]}."
                )
            return asarray(
                selector.T @ np.asarray(m_imp_arr, dtype=float).reshape(selector.shape[0])
            )

        block = np.asarray(m_imp_arr, dtype=float).reshape(selector.shape[0], -1)
        return asarray(selector.T @ block).reshape(selector.shape[1], -1)

    def expand_solution(self, m_imp_solution: np.ndarray) -> np.ndarray:
        """Expand reduced ``m_imp`` coefficients to the full solution coordinates."""
        m_imp_arr = asarray(m_imp_solution)
        if self.selector is None:
            return asarray(m_imp_arr)

        expander = as_linear_map(np.asarray(self.selector, dtype=float))
        if m_imp_arr.ndim == 1:
            if m_imp_arr.size != expander.shape[1]:
                raise ValueError(
                    "Reduced m_imp vector has incompatible size: "
                    f"{m_imp_arr.size} != {expander.shape[1]}."
                )
            return asarray(expander.matvec(m_imp_arr.reshape(expander.shape[1])))

        block = m_imp_arr.reshape(expander.shape[1], -1)
        return asarray(expander.matmat(block)).reshape(expander.shape[0], -1)

    def reduce_full_operator(self, operator: Any) -> np.ndarray:
        """Project a full-space operator to reduced ``m_imp`` coordinates."""
        operator_arr = np.asarray(operator, dtype=float).reshape(self.full_size, self.full_size)
        if self.selector is None:
            return operator_arr
        selector = np.asarray(self.selector, dtype=float)
        return np.asarray(selector.T @ operator_arr @ selector, dtype=float)

    def expand_reduced_operator(self, operator: Any) -> np.ndarray:
        """Lift a reduced-space operator back to full ``m_imp`` coordinates."""
        operator_arr = np.asarray(operator, dtype=float).reshape(
            self.reduced_size, self.reduced_size
        )
        if self.selector is None:
            return operator_arr
        selector = np.asarray(self.selector, dtype=float)
        return np.asarray(selector @ operator_arr @ selector.T, dtype=float)


class PoloidalSolver:
    """Expose solve/orchestration routines built on top of poloidal operators."""

    def __init__(
        self,
        matrices: "PoloidalSystemMatrices",
        timed_solve: TimedSolveFn,
        timed_structured_solve: TimedSolveFn,
    ) -> None:
        self._mats = matrices
        self._timed_solve = timed_solve
        self._timed_structured_solve = timed_structured_solve

    def build_induction_matrix(
        self,
        feedback_system: MImpFeedbackSystem,
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_ind_to_E_operator: Any = None,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Construct the dense induction matrix `(m_ind -> E_df)`."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self._mats.solution_space.index_length

        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        E_direct_dense = coerce_dense_operator_matrix(
            m_ind_to_E_operator, n_component_rows=2, n_cols=n
        ).reshape(2, n, n)

        if feedback_system.solve_system.equality_operator is not None:
            m_imp_cols = []
            for scenario_index in range(n):
                m_imp_col, _ = self.solve_for_m_imp(
                    E_direct_coeffs=E_direct_dense[:, :, scenario_index],
                    feedback_system=feedback_system,
                    solver=solver,
                    E_map_constraint_operator=E_map_constraint_operator,
                    ih_constraint_scaling=ih_constraint_scaling,
                    connect_hemispheres=connect_hemispheres,
                    m_imp_to_E_operator=m_imp_to_E_operator,
                )
                m_imp_cols.append(np.asarray(m_imp_col).reshape(-1))
            m_imp_block = asarray(np.column_stack(m_imp_cols))
        else:
            rhs_entries = self._mats._build_m_imp_rhs_entries(
                feedback_system.problem,
                E_direct_coeffs=E_direct_dense,
                E_constraint_operator=E_map_constraint_operator,
                ih_constraint_scaling=ih_constraint_scaling,
                connect_hemispheres=connect_hemispheres,
            )
            m_imp_block = self._mats._solve_m_imp_feedback_block(
                problem=feedback_system.problem,
                solver=solver,
                rhs_entries=rhs_entries,
                num_expected_scenarios=n,
            )
            m_imp_block = feedback_system.expand_solution(m_imp_block)

        m_imp_to_E = as_linear_map(m_imp_to_E_operator)
        E_imp_flat = m_imp_to_E.matmat(asarray(m_imp_block))
        E_imp_block = asarray(E_imp_flat).reshape(2, n, n)
        total_E = E_direct_dense + E_imp_block

        curled_scenarios = self._mats._extract_toroidal_potential_coeffs(total_E)
        logger.info("Dense induction operator built.")
        return asarray(curled_scenarios)

    def get_induction_operator(
        self,
        feedback_system: MImpFeedbackSystem,
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_ind_to_E_operator: Any = None,
        m_imp_to_E_operator: Any = None,
    ) -> "LinearMap":
        """Get matrix-free induction operator `(m_ind -> E_df)`."""
        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        if m_imp_to_E_operator is None:
            raise ValueError("m_imp_to_E_operator is required")
        n = self._mats.solution_space.index_length
        m_ind_to_E = as_linear_map(m_ind_to_E_operator)

        def _build_dense() -> np.ndarray:
            return self.build_induction_matrix(
                feedback_system=feedback_system,
                solver=solver,
                E_map_constraint_operator=E_map_constraint_operator,
                ih_constraint_scaling=ih_constraint_scaling,
                connect_hemispheres=connect_hemispheres,
                m_ind_to_E_operator=m_ind_to_E_operator,
                m_imp_to_E_operator=m_imp_to_E_operator,
            )

        def matvec(m_ind_vec: np.ndarray) -> np.ndarray:
            m_ind_vec = asarray(m_ind_vec).flatten()
            E_ind_coeffs = m_ind_to_E.matvec(m_ind_vec).reshape(2, -1)
            if connect_hemispheres:
                _, E_imp = self.solve_for_m_imp(
                    E_direct_coeffs=E_ind_coeffs,
                    feedback_system=feedback_system,
                    solver=solver,
                    E_map_constraint_operator=E_map_constraint_operator,
                    ih_constraint_scaling=ih_constraint_scaling,
                    connect_hemispheres=connect_hemispheres,
                    m_imp_to_E_operator=m_imp_to_E_operator,
                )
                E_ind_coeffs = E_ind_coeffs + E_imp
            E_df = self._mats._extract_toroidal_potential_coeffs(E_ind_coeffs)
            return asarray(E_df).flatten()

        return build_linear_map(
            shape=(n, n), matvec=matvec, dense_builder=_build_dense, dtype=np.float64
        )

    def solve_for_m_imp(
        self,
        E_direct_coeffs: np.ndarray,
        feedback_system: MImpFeedbackSystem,
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for `m_imp` given direct E coefficients and return `(m_imp, E_imp)`."""
        ll_equality_rhs = None
        rhs_entries = self._mats._build_m_imp_rhs_entries(
            feedback_system.problem,
            E_direct_coeffs=E_direct_coeffs,
            E_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
        )
        if (
            connect_hemispheres
            and E_map_constraint_operator is not None
            and feedback_system.solve_system.equality_operator is not None
        ):
            b_E = self._mats._apply_E_constraint_operator(
                E_map_constraint_operator, E_direct_coeffs
            )
            ll_equality_rhs = self._mats._reshape_constraint_rhs_block(b_E)
        if ll_equality_rhs is not None and all(entry is None for entry in rhs_entries):
            rhs_entries[0] = xp.zeros(int(feedback_system.problem.A[0].num_rows), dtype=float)
        solution = self._timed_structured_solve(
            "poloidal.m_imp",
            feedback_system.solve_system,
            solver,
            rhs_entries,
            preconditioner=feedback_system.preconditioner,
            equality_rhs_input=ll_equality_rhs,
        )
        if solution is None:
            m_imp_solution = xp.zeros(feedback_system.problem.solution_size)
        else:
            m_imp_solution = asarray(solution)
        m_imp = feedback_system.expand_solution(m_imp_solution)

        if m_imp_to_E_operator is None:
            raise ValueError("m_imp_to_E_operator is required")
        m_imp_to_E = as_linear_map(m_imp_to_E_operator)
        E_imp = m_imp_to_E.matvec(m_imp).reshape(2, -1)
        return asarray(m_imp), asarray(E_imp)

    def compute_rates(
        self,
        m_ind: np.ndarray,
        t: float,
        E_coeffs_noind: np.ndarray,
        induction_matrix: Optional[np.ndarray] = None,
        m_ind_to_E_operator: Any = None,
        feedback_system: Optional[MImpFeedbackSystem] = None,
        solver: Optional[Any] = None,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Calculate `d(m_ind)/dt` rates."""
        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        if m_imp_to_E_operator is None:
            raise ValueError("m_imp_to_E_operator is required")

        backend_m_ind = asarray(m_ind)
        E_df_noind = self._mats._extract_toroidal_potential_coeffs(E_coeffs_noind)

        if induction_matrix is not None:
            E_df_ind = asarray(induction_matrix) @ backend_m_ind
        else:
            E_ind_coeffs = m_ind_to_E_operator.matvec(backend_m_ind).reshape(2, -1)
            if connect_hemispheres and feedback_system is not None:
                _, E_imp = self.solve_for_m_imp(
                    E_direct_coeffs=E_ind_coeffs,
                    feedback_system=feedback_system,
                    solver=solver,
                    E_map_constraint_operator=E_map_constraint_operator,
                    ih_constraint_scaling=ih_constraint_scaling,
                    connect_hemispheres=connect_hemispheres,
                    m_imp_to_E_operator=m_imp_to_E_operator,
                )
                E_ind_coeffs = E_ind_coeffs + E_imp
            E_df_ind = E_ind_coeffs[1]

        E_df_total = E_df_ind + E_df_noind
        return self._mats.E_df_to_d_m_ind_dt * E_df_total

    def steady_state_m_ind(
        self, E_coeffs_noind: np.ndarray, induction_matrix: Any, solver: str = "lsmr"
    ) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        vec_b = -self._mats._extract_toroidal_potential_coeffs(E_coeffs_noind)

        if hasattr(induction_matrix, "matvec"):
            from pynamit.math.least_squares_problem import LeastSquaresProblem
            from pynamit.math.least_squares_solver import LeastSquaresSolver

            n = self._mats.solution_space.index_length
            induction_op = as_linear_map(induction_matrix)
            problem = LeastSquaresProblem(
                A=[induction_op], solution_shape=(n,), data_shapes=[(n,)]
            )
            ls_solver = LeastSquaresSolver(solver=solver, tolerance=1e-10)
            return asarray(
                self._timed_solve(
                    "poloidal.steady_state_m_ind", ls_solver, problem, [vec_b], maxiter=5000
                )
            )

        L = asarray(induction_matrix)
        result = xp.linalg.lstsq(L, vec_b, rcond=1e-13)
        return asarray(result[0])
