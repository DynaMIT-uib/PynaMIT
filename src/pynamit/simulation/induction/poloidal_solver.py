"""Poloidal solve/orchestration helpers.

This module keeps `PoloidalSystemMatrices` focused on operator assembly while
collecting feedback solves, induction-operator exposure, and steady-state/rate
orchestration in one small helper object.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from pynamit.math.linear_map import as_linear_map
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


class PoloidalSolver:
    """Expose solve/orchestration routines built on top of poloidal operators."""

    def __init__(self, matrices: "PoloidalSystemMatrices", timed_solve: TimedSolveFn) -> None:
        self._mats = matrices
        self._timed_solve = timed_solve

    def build_induction_matrix(
        self,
        problem: "LeastSquaresProblem",
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_ind_to_E_operator: Any = None,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Construct the dense induction matrix `(m_ind -> E_df)`."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self._mats.solution_basis.index_length

        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        E_direct_dense = coerce_dense_operator_matrix(
            m_ind_to_E_operator,
            n_component_rows=2,
            n_cols=n,
        ).reshape(2, n, n)

        rhs_entries = self._mats._build_m_imp_rhs_entries(
            problem,
            E_direct_coeffs=E_direct_dense,
            E_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
        )
        m_imp_block = self._mats._solve_m_imp_feedback_block(
            problem=problem,
            solver=solver,
            rhs_entries=rhs_entries,
            num_expected_scenarios=n,
        )

        m_imp_to_E = as_linear_map(m_imp_to_E_operator)
        E_imp_flat = m_imp_to_E.matmat(asarray(m_imp_block))
        E_imp_block = asarray(E_imp_flat).reshape(2, n, n)
        total_E = E_direct_dense + E_imp_block

        curled_scenarios = self._mats._extract_toroidal_potential_coeffs(total_E)
        logger.info("Dense induction operator built.")
        return asarray(curled_scenarios)

    def get_induction_operator(
        self,
        problem: "LeastSquaresProblem",
        solver: Any,
        preconditioner: Optional[Any] = None,
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
        n = self._mats.solution_basis.index_length
        m_ind_to_E = as_linear_map(m_ind_to_E_operator)

        def _build_dense() -> np.ndarray:
            return self.build_induction_matrix(
                problem=problem,
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
            if connect_hemispheres and problem is not None:
                _, E_imp = self.solve_for_m_imp(
                    E_direct_coeffs=E_ind_coeffs,
                    problem=problem,
                    solver=solver,
                    preconditioner=preconditioner,
                    E_map_constraint_operator=E_map_constraint_operator,
                    ih_constraint_scaling=ih_constraint_scaling,
                    connect_hemispheres=connect_hemispheres,
                    m_imp_to_E_operator=m_imp_to_E_operator,
                )
                E_ind_coeffs = E_ind_coeffs + E_imp
            E_df = self._mats._extract_toroidal_potential_coeffs(E_ind_coeffs)
            return asarray(E_df).flatten()

        return build_linear_map(
            shape=(n, n),
            matvec=matvec,
            dense_builder=_build_dense,
            dtype=np.float64,
        )

    def solve_for_m_imp(
        self,
        E_direct_coeffs: np.ndarray,
        problem: "LeastSquaresProblem",
        solver: Any,
        preconditioner: Optional[Any] = None,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for `m_imp` given direct E coefficients and return `(m_imp, E_imp)`."""
        rhs_entries = self._mats._build_m_imp_rhs_entries(
            problem,
            E_direct_coeffs=E_direct_coeffs,
            E_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
        )
        solution = self._timed_solve(
            "poloidal.m_imp",
            solver,
            problem=problem,
            rhs=rhs_entries,
            preconditioner=preconditioner,
        )
        if solution is None:
            m_imp = xp.zeros(self._mats.solution_basis.index_length)
        else:
            m_imp = asarray(solution)

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
        problem: Optional[Any] = None,
        solver: Optional[Any] = None,
        preconditioner: Optional[Any] = None,
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
            if connect_hemispheres and problem is not None:
                _, E_imp = self.solve_for_m_imp(
                    E_direct_coeffs=E_ind_coeffs,
                    problem=problem,
                    solver=solver,
                    preconditioner=preconditioner,
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
        self,
        E_coeffs_noind: np.ndarray,
        induction_matrix: Any,
        solver: str = "lsmr",
    ) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        vec_b = -self._mats._extract_toroidal_potential_coeffs(E_coeffs_noind)

        if hasattr(induction_matrix, "matvec"):
            from pynamit.math.least_squares_problem import LeastSquaresProblem
            from pynamit.math.least_squares_solver import LeastSquaresSolver

            n = self._mats.solution_basis.index_length
            induction_op = as_linear_map(induction_matrix)
            problem = LeastSquaresProblem(
                A=[induction_op],
                solution_shape=(n,),
                data_shapes=[(n,)],
            )
            ls_solver = LeastSquaresSolver(solver=solver, tolerance=1e-10)
            return asarray(
                self._timed_solve(
                    "poloidal.steady_state_m_ind",
                    ls_solver,
                    problem,
                    [vec_b],
                    maxiter=5000,
                )
            )

        L = asarray(induction_matrix)
        result = xp.linalg.lstsq(L, vec_b, rcond=1e-13)
        return asarray(result[0])
