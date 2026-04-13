"""Toroidal solve/orchestration helpers.

This module keeps `ToroidalSystemMatrices` focused on toroidal operator/RHS
assembly while collecting gauge handling, `dt_alpha -> dt_psi` conversion, and
exported solve/dynamics maps in one helper object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.math.structured_least_squares import (
    ConstrainedStructuredLeastSquaresSubproblem,
    StructuredLeastSquaresDataTerm,
    StructuredLeastSquaresSubproblem,
)
from pynamit.simulation.induction.operator_utils import (
    build_linear_map,
    coerce_dense_operator_matrix,
)
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.utils import asarray, to_numpy

if TYPE_CHECKING:
    from pynamit.math.least_squares_problem import LeastSquaresProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DtAlphaSolveSystem:
    """Bundle the structured ``dt_alpha`` least-squares solve definition."""

    solve_system: ConstrainedStructuredLeastSquaresSubproblem
    physics_rhs_lift: np.ndarray
    penalty_term_index: Optional[int] = None

    @property
    def subproblem(self) -> StructuredLeastSquaresSubproblem:
        """Compatibility view of the unconstrained structured subproblem."""
        return self.solve_system.subproblem

    @property
    def problem(self) -> "LeastSquaresProblem":
        """Compatibility view of the underlying least-squares problem."""
        return self.solve_system.problem

    @property
    def n_coeff(self) -> int:
        """Return the flattened ``dt_alpha`` coefficient dimension."""
        return int(self.solve_system.solution_size)

    @property
    def grid_rows(self) -> int:
        """Return the number of grid residual rows in the physics term."""
        return int(np.asarray(self.physics_rhs_lift, dtype=float).shape[0])

    def physics_rhs_to_grid_rhs(self, rhs_coeffs: np.ndarray) -> np.ndarray:
        """Map coefficient-space RHS columns to grid-space RHS columns."""
        rhs_lift = np.asarray(self.physics_rhs_lift, dtype=float)
        rhs_arr = np.asarray(to_numpy(rhs_coeffs))
        if rhs_arr.ndim == 1:
            return rhs_lift @ rhs_arr.reshape(-1, 1)
        rhs_2d = rhs_arr.reshape(rhs_arr.shape[0], -1)
        return rhs_lift @ rhs_2d

    def build_rhs_terms(
        self,
        rhs_physics_coeffs: np.ndarray,
        *,
        penalty_rhs: Optional[np.ndarray] = None,
        penalty_scaling: float = 0.0,
    ) -> list[np.ndarray]:
        """Assemble structured RHS terms for the ``dt_alpha`` least-squares solve."""
        rhs_grid = np.asarray(self.physics_rhs_to_grid_rhs(rhs_physics_coeffs))
        if rhs_grid.ndim == 1:
            rhs_grid = rhs_grid.reshape(-1, 1)
        n_scenarios = int(rhs_grid.shape[1])
        rhs_terms = [rhs_grid]
        for term_index in range(1, int(self.problem.num_data_terms)):
            n_rows = int(self.problem.A[term_index].num_rows)
            rhs_terms.append(np.zeros((n_rows, n_scenarios), dtype=rhs_grid.dtype))
        if self.penalty_term_index is not None and penalty_rhs is not None:
            penalty_rhs_arr = np.asarray(to_numpy(penalty_rhs))
            n_rows = int(self.problem.A[self.penalty_term_index].num_rows)
            if penalty_rhs_arr.ndim == 1:
                if penalty_rhs_arr.shape[0] != n_rows:
                    raise ValueError(
                        f"penalty_rhs length {penalty_rhs_arr.shape[0]} != penalty rows {n_rows}"
                    )
                penalty_rhs_arr = np.repeat(penalty_rhs_arr[:, None], n_scenarios, axis=1)
            else:
                penalty_rhs_arr = penalty_rhs_arr.reshape(n_rows, -1)
                if penalty_rhs_arr.shape[1] == 1 and n_scenarios > 1:
                    penalty_rhs_arr = np.repeat(penalty_rhs_arr, n_scenarios, axis=1)
                if penalty_rhs_arr.shape[1] != n_scenarios:
                    raise ValueError(
                        "penalty_rhs scenario shape mismatch: "
                        f"expected {n_scenarios}, got {penalty_rhs_arr.shape[1]}"
                    )
            rhs_terms[self.penalty_term_index] = np.asarray(penalty_scaling) * penalty_rhs_arr
        return rhs_terms


class ToroidalSolver:
    """Expose solve/orchestration routines built on top of toroidal operators."""

    def __init__(self, matrices: Any) -> None:
        self._mats = matrices

    @staticmethod
    def _coerce_equality_rhs(rhs: Any) -> Any:
        """Normalize cached equality RHS inputs without changing scenario layout."""
        if rhs is None:
            return None
        return np.asarray(to_numpy(rhs))

    def _get_dtalpha_solve_system(
        self,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
    ) -> DtAlphaSolveSystem:
        """Build or fetch the weighted least-squares solve system for ``dt_alpha``."""
        mats = self._mats
        cache_key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
        )
        cached = mats._toroidal_problem_cache.get(cache_key)
        if cached is not None:
            return cached

        from pynamit.math.linear_map import as_linear_map, diagonal_linear_map

        residual_coeff_operator = np.asarray(to_numpy(mats.physics_residual_coeff_operator))
        if residual_coeff_operator.ndim != 2:
            residual_coeff_operator = residual_coeff_operator.reshape(
                residual_coeff_operator.shape[0], -1
            )
        n_coeff = int(residual_coeff_operator.shape[1])
        a_grid = np.asarray(to_numpy(mats.physics_residual_row_operator), dtype=float)
        physics_rhs_lift = np.asarray(to_numpy(mats.physics_rhs_lift_operator), dtype=float)
        op_a_grid = as_linear_map(a_grid)
        physics_weight = mats._build_physics_sqrt_weight(op_a_grid.shape[0], weighting)

        data_terms = [
            StructuredLeastSquaresDataTerm(
                operator=op_a_grid, data_shape=(op_a_grid.shape[0],), sqrt_weight=physics_weight
            )
        ]
        penalty_term_index: Optional[int] = None

        if penalty_operator is not None and penalty_scaling > 0:
            op_penalty = as_linear_map(penalty_operator) * penalty_scaling
            penalty_term_index = len(data_terms)
            data_terms.append(
                StructuredLeastSquaresDataTerm(
                    operator=op_penalty, data_shape=(op_penalty.shape[0],), sqrt_weight=None
                )
            )

        if regularization_lambda > 0:
            op_reg = diagonal_linear_map(np.ones(n_coeff)) * float(
                np.sqrt(max(regularization_lambda, 0.0))
            )
            data_terms.append(
                StructuredLeastSquaresDataTerm(
                    operator=op_reg, data_shape=(op_reg.shape[0],), sqrt_weight=None
                )
            )

        subproblem = StructuredLeastSquaresSubproblem(
            solution_shape=n_coeff, data_terms=tuple(data_terms)
        )
        solve_system = subproblem.with_equality()
        solve_bundle = DtAlphaSolveSystem(
            solve_system=solve_system,
            physics_rhs_lift=physics_rhs_lift,
            penalty_term_index=penalty_term_index,
        )
        mats._toroidal_problem_cache[cache_key] = solve_bundle
        return solve_bundle

    def _resolve_rcond(self, *, n_coeff: int, hinv_rtol: float) -> float:
        """Resolve pseudoinverse cutoff used by constrained elimination."""
        mats = self._mats
        if hinv_rtol > 0:
            return max(float(hinv_rtol), 0.0)
        rcond = mats._default_pinv_rcond((n_coeff, n_coeff))
        logger.info("Auto hard-solve rtol (default pseudoinverse cutoff): %.3e", float(rcond))
        return float(max(rcond, 0.0))

    def _solve_dtalpha_problem(
        self,
        rhs_physics_coeffs: np.ndarray,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        penalty_rhs: Optional[np.ndarray] = None,
        hinv_rtol: float = 0.0,
        equality_operator: Any = None,
        equality_rhs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve the weighted toroidal least-squares problem in ``dt_alpha``."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        mats = self._mats
        solve_bundle = self._get_dtalpha_solve_system(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        solve_system = solve_bundle.solve_system
        if equality_operator is not None:
            solve_system = solve_system.with_equality(
                equality_operator=equality_operator, equality_rhs_builder=self._coerce_equality_rhs
            )
        problem = solve_system.problem
        n_coeff = int(solve_bundle.n_coeff)
        rcond = self._resolve_rcond(n_coeff=n_coeff, hinv_rtol=hinv_rtol)

        rhs_terms = solve_bundle.build_rhs_terms(
            rhs_physics_coeffs, penalty_rhs=penalty_rhs, penalty_scaling=penalty_scaling
        )
        n_scenarios = int(np.asarray(rhs_terms[0]).shape[1])

        solver = LeastSquaresSolver(
            solver=mats.toroidal_solver,
            tolerance=max(rcond, mats.toroidal_tolerance),
            preconditioner=mats.toroidal_preconditioner,
        )
        preconditioner = None
        if equality_operator is None and mats.toroidal_preconditioner is not None:
            preconditioner = solver.build_preconditioner(
                problem=problem,
                preconditioner_type=mats.toroidal_preconditioner,
                num_scenarios=n_scenarios,
                pinv_rcond=rcond,
            )
        sol = solve_system.solve(
            solver,
            rhs_terms,
            preconditioner=preconditioner,
            equality_rhs_input=equality_rhs,
            elimination_rcond=rcond,
        )
        return np.asarray(sol)

    def _get_unconstrained_dtalpha_map_cached(
        self,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Return cached dense map ``rhs_physics -> dt_alpha``."""
        mats = self._mats
        solve_bundle = self._get_dtalpha_solve_system(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        n_coeff = int(solve_bundle.n_coeff)
        rcond = self._resolve_rcond(n_coeff=n_coeff, hinv_rtol=hinv_rtol)
        key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            mats._toroidal_solver_signature(),
        )
        cached = mats._dtalpha_unconstrained_map_cache.get(key)
        if cached is not None:
            return cached
        rhs_dim = int(np.asarray(self._mats.toroidal_rhs_from_E_operator, dtype=float).shape[0])
        rhs_physics_basis = np.eye(rhs_dim, dtype=float)
        alpha_map = np.asarray(
            self._solve_dtalpha_problem(
                rhs_physics_coeffs=rhs_physics_basis,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        ).reshape(n_coeff, rhs_dim)
        mats._dtalpha_unconstrained_map_cache[key] = alpha_map
        return alpha_map

    def _get_constrained_dtalpha_maps(
        self,
        *,
        alpha_map_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Return cached constrained maps for hard ``dt_alpha`` constraints."""
        mats = self._mats
        c_dtalpha = np.asarray(to_dense(as_linear_map(alpha_map_operator)))
        if c_dtalpha.ndim != 2:
            c_dtalpha = c_dtalpha.reshape(c_dtalpha.shape[0], -1)
        n_coeff = int(c_dtalpha.shape[1])
        m_constraints = int(c_dtalpha.shape[0])
        rhs_dim = int(np.asarray(mats.toroidal_rhs_from_E_operator, dtype=float).shape[0])
        rcond = self._resolve_rcond(n_coeff=n_coeff, hinv_rtol=hinv_rtol)
        key = (
            id(alpha_map_operator),
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            mats._toroidal_solver_signature(),
        )
        cached = mats._dtalpha_constrained_maps_cache.get(key)
        if cached is not None:
            return cached

        m_phys_dtalpha = np.asarray(
            self._solve_dtalpha_problem(
                rhs_physics_coeffs=np.eye(rhs_dim, dtype=float),
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                equality_operator=c_dtalpha,
                equality_rhs=np.zeros(m_constraints, dtype=float),
            )
        ).reshape(n_coeff, rhs_dim)

        if m_constraints > 0:
            m_corr_dtalpha = np.asarray(
                self._solve_dtalpha_problem(
                    rhs_physics_coeffs=np.zeros((rhs_dim, m_constraints), dtype=float),
                    weighting=weighting,
                    regularization_lambda=regularization_lambda,
                    penalty_operator=penalty_operator,
                    penalty_scaling=penalty_scaling,
                    hinv_rtol=hinv_rtol,
                    equality_operator=c_dtalpha,
                    equality_rhs=np.eye(m_constraints, dtype=float),
                )
            ).reshape(n_coeff, m_constraints)
        else:
            m_corr_dtalpha = np.zeros((n_coeff, 0), dtype=float)

        maps = {
            "C_dtalpha": c_dtalpha,
            "M_phys_dtalpha": np.asarray(m_phys_dtalpha),
            "M_corr_dtalpha": np.asarray(m_corr_dtalpha),
        }
        mats._dtalpha_constrained_maps_cache[key] = maps
        return maps

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
        penalty_rhs: Optional[np.ndarray] = None,
        hinv_rtol: float = 0.0,
        apply_psi_gauge: bool = False,
    ) -> np.ndarray:
        """Solve for `dpsi/dt` via one-shot constrained `dt_alpha` solve."""
        rhs_p = np.asarray(to_numpy(rhs_physics)).reshape(-1)
        dtalpha_to_dt_psi = self._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator, apply_psi_gauge=apply_psi_gauge
        )

        if constraint_operator is None:
            M_alpha = self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha = M_alpha @ rhs_p
            if penalty_operator is not None and penalty_scaling > 0 and penalty_rhs is not None:
                penalty_rhs_vec = np.asarray(to_numpy(penalty_rhs)).reshape(-1)
                dtalpha = dtalpha + np.asarray(
                    self._solve_dtalpha_problem(
                        rhs_physics_coeffs=np.zeros_like(rhs_p),
                        weighting=weighting,
                        regularization_lambda=regularization_lambda,
                        penalty_operator=penalty_operator,
                        penalty_scaling=penalty_scaling,
                        penalty_rhs=penalty_rhs_vec,
                        hinv_rtol=hinv_rtol,
                    )
                ).reshape(-1)
            return asarray((dtalpha_to_dt_psi @ dtalpha).reshape(-1))

        maps = self._get_constrained_dtalpha_maps(
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
        if penalty_operator is not None and penalty_scaling > 0 and penalty_rhs is not None:
            penalty_rhs_vec = np.asarray(to_numpy(penalty_rhs)).reshape(-1)
            dtalpha = dtalpha + np.asarray(
                self._solve_dtalpha_problem(
                    rhs_physics_coeffs=np.zeros_like(rhs_p),
                    weighting=weighting,
                    regularization_lambda=regularization_lambda,
                    penalty_operator=penalty_operator,
                    penalty_scaling=penalty_scaling,
                    penalty_rhs=penalty_rhs_vec,
                    hinv_rtol=hinv_rtol,
                    equality_operator=constraint_operator,
                    equality_rhs=np.zeros_like(rhs_c),
                )
            ).reshape(-1)
        dt_psi = dtalpha_to_dt_psi @ dtalpha
        return asarray(dt_psi.reshape(-1))

    def _get_psi_gauge_projector_dense(
        self, m_imp_to_jr_operator: Any, apply_psi_gauge: Optional[bool] = None
    ) -> np.ndarray:
        """Return explicit gauge projector applied after MP inversion."""
        mats = self._mats
        if apply_psi_gauge is None:
            apply_psi_gauge = mats.is_cs

        gauge_mode = "mean_zero"
        cache_key = (id(m_imp_to_jr_operator), bool(apply_psi_gauge), gauge_mode)
        cached = mats._psi_gauge_projector_cache.get(cache_key)
        if cached is not None:
            return cached

        op_m_to_jr = as_linear_map(m_imp_to_jr_operator)
        m_to_jr_dense = to_dense(op_m_to_jr)
        n = int(m_to_jr_dense.shape[1])
        identity = np.eye(n, dtype=m_to_jr_dense.dtype)

        if not apply_psi_gauge:
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
                mats.basis.get_scalar_gauge_constraint_matrix(n_coeff=n, mode=gauge_mode)
            )
        if gauge_row is None:
            if not hasattr(mats.basis, "get_evaluation_matrix"):
                raise RuntimeError(
                    "Scalar gauge projector requested, but basis does not provide "
                    "get_scalar_gauge_constraint_matrix() or get_evaluation_matrix(grid)."
                )
            g_mat = np.asarray(
                to_dense(as_linear_map(mats.basis.get_evaluation_matrix(mats.grid)))
            )
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
        self, *, m_imp_to_jr_operator: Any, apply_psi_gauge: bool
    ) -> np.ndarray:
        """Return cached dense map `dt_alpha -> dpsi/dt`."""
        mats = self._mats
        key = (id(m_imp_to_jr_operator), bool(apply_psi_gauge))
        cached = mats._dtalpha_to_dt_psi_map_cache.get(key)
        if cached is not None:
            return cached

        alpha_to_psi = np.asarray(to_numpy(mats.alpha_to_psi_coeff_operator))
        gauge_projector = np.asarray(
            self._get_psi_gauge_projector_dense(
                m_imp_to_jr_operator, apply_psi_gauge=apply_psi_gauge
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
        apply_psi_gauge: bool = False,
    ) -> np.ndarray:
        """Build dense map `toroidal_rhs -> dpsi/dt`."""
        mats = self._mats
        if constraint_operator is not None:
            maps = self._get_constrained_dtalpha_maps(
                alpha_map_operator=constraint_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha_from_K = maps["M_phys_dtalpha"]
        else:
            dtalpha_from_K = self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )

        dtalpha_to_dt_psi = self._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator, apply_psi_gauge=apply_psi_gauge
        )
        return asarray(dtalpha_to_dt_psi @ dtalpha_from_K)

    def build_dtalpha_from_toroidal_rhs_matrix(
        self,
        *,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Build dense map ``toroidal_rhs -> dt_alpha``."""
        if constraint_operator is not None:
            maps = self._get_constrained_dtalpha_maps(
                alpha_map_operator=constraint_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            return asarray(maps["M_phys_dtalpha"])

        return asarray(
            self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        )

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
        apply_psi_gauge: bool = False,
    ) -> np.ndarray:
        """Build the linear operator `psi -> dpsi/dt`."""
        mats = self._mats
        N = mats.basis.index_length
        E_to_rhs = to_numpy(mats.toroidal_rhs_from_E_operator)
        psi_to_E = coerce_dense_operator_matrix(psi_to_E_operator, n_component_rows=2, n_cols=N)

        dt_psi_from_K = self.build_dt_psi_from_toroidal_rhs_matrix(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            constraint_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
            apply_psi_gauge=apply_psi_gauge,
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
        apply_psi_gauge: bool = False,
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
                apply_psi_gauge=apply_psi_gauge,
            )
            return as_linear_map(L_dense)

        if isinstance(psi_to_E_operator, LinearMap):
            psi_to_E_op = psi_to_E_operator
        else:
            psi_to_E_op = as_linear_map(
                coerce_dense_operator_matrix(psi_to_E_operator, n_component_rows=2, n_cols=N)
            )

        E_to_rhs_op = as_linear_map(np.asarray(to_numpy(mats.toroidal_rhs_from_E_operator)))
        dtalpha_from_K_op = as_linear_map(
            self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        )
        dtalpha_to_dt_psi_op = as_linear_map(
            self._get_dtalpha_to_dt_psi_map_cached(
                m_imp_to_jr_operator=m_imp_to_jr_operator, apply_psi_gauge=apply_psi_gauge
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
                apply_psi_gauge=apply_psi_gauge,
            )

        return build_linear_map(
            shape=(N, N),
            matvec=matvec,
            rmatvec=rmatvec,
            dense_builder=to_dense_func,
            dtype=np.float64,
            domain_space="psi_coeffs",
            codomain_space="dt_psi_coeffs",
        )
