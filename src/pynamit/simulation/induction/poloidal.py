"""Poloidal System Matrices Module.

This module implements the `PoloidalSystemMatrices` class, which is responsible
for assembling the least-squares system matrices required for the poloidal
induction solver.

Unlike ToroidalSystemMatrices which solves a direct linear system L * x = K,
poloidal induction uses a constrained least-squares formulation.
"""

from __future__ import annotations
import logging
import os
import time
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from functools import cached_property

from pynamit.utils import to_numpy, asarray, xp, tensor_pinv
from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.math.structured_least_squares import (
    StructuredLeastSquaresDataTerm,
    StructuredLeastSquaresRegularizationTerm,
    StructuredLeastSquaresSubproblem,
)
from pynamit.math.constants import mu0
from pynamit.simulation.induction.poloidal_closure import (
    PoloidalRMBoundaryOperators,
    PoloidalClosureProjector,
    RMCouplingOperators,
)
from pynamit.simulation.spatial.geometry_utils import to_dense

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.field import Field
    from pynamit.simulation.spatial.pfac import PFACIntegrator
    from pynamit.math.least_squares_problem import LeastSquaresProblem
    from pynamit.simulation.induction.poloidal_solver import MImpFeedbackSystem

logger = logging.getLogger(__name__)


def _timed_solve(label: str, solver: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    """Optionally time least-squares solves when PYNAMIT_TIMING_SOLVES is set."""
    kwargs.setdefault("warning_label", label)
    if os.getenv("PYNAMIT_TIMING_SOLVES", "").strip() in ("", "0"):
        return solver.solve(*args, **kwargs)
    t0 = time.perf_counter()
    out = solver.solve(*args, **kwargs)
    dt = time.perf_counter() - t0
    solver_name = getattr(solver, "solver", "unknown")
    print(f"TIMING solve[{label}] ({solver_name}): {dt:.3f}s", flush=True)
    return out


def _timed_structured_solve(
    label: str, solve_system: Any, solver: Any, *args: Any, **kwargs: Any
) -> np.ndarray:
    """Optionally time structured subproblem solves when PYNAMIT_TIMING_SOLVES is set."""
    kwargs.setdefault("warning_label", label)
    if os.getenv("PYNAMIT_TIMING_SOLVES", "").strip() in ("", "0"):
        return solve_system.solve(solver, *args, **kwargs)
    t0 = time.perf_counter()
    out = solve_system.solve(solver, *args, **kwargs)
    dt = time.perf_counter() - t0
    solver_name = getattr(solver, "solver", "unknown")
    print(f"TIMING solve[{label}] ({solver_name}): {dt:.3f}s", flush=True)
    return out


class PoloidalSystemMatrices:
    """Assembles system matrices for Poloidal Induction.

    Unlike ToroidalSystemMatrices which solves L*x=K directly,
    poloidal induction uses a constrained least-squares formulation.

    Note on terminology:
    - The variable `m_imp` is the toroidal magnetic scalar (T), since
      jr = (RI/mu0) * Laplacian(m_imp).
    - This module name ("poloidal") refers to the electrostatic/current-system
      solve, not the magnetic-scalar naming convention.

    The imposed toroidal magnetic potential m_imp relates to:
    - Radial current: jr = (RI/mu0) * Laplacian(m_imp)
    - Pre-resistivity vector (JS-like):
        J = -grad(m_imp)/mu0 + Tor(Ve) via PFAC coupling
      (Ve is a magnetic poloidal potential; resistivity maps J -> E)

    Parameters
    ----------
    basis : Basis
        The spectral basis (SHBasis) for physics computations.
    solution_space : Basis
        The basis used for solution variables (may differ from basis).
    grid : Grid
        The spatial grid for evaluation.
    b_field : Field
        The background magnetic field evaluated on the grid.
    RI : float
        Radius of the ionosphere.
    pfac_integrator : PFACIntegrator
        Integrator for PFAC (field-aligned current) coupling.
    """

    def __init__(
        self,
        basis: "Basis",
        solution_space: "Basis",
        grid: "Grid",
        b_field: "Field",
        RI: float,
        pfac_integrator: "PFACIntegrator",
    ):
        self.basis = basis
        self.solution_space = solution_space
        self.grid = grid
        self.b_field = b_field
        self.RI = RI
        self._pfac = pfac_integrator

    @cached_property
    def _poloidal_closure_projector(self) -> PoloidalClosureProjector:
        """Closure-basis projector for poloidal RM/PFAC operators."""
        return PoloidalClosureProjector(
            solution_space=self.solution_space,
            closure_basis=self._pfac.basis,
            grid=self.grid,
            pfac_integrator=self._pfac,
        )

    @cached_property
    def _rm_coupling_solution_operators(self) -> Optional[RMCouplingOperators]:
        """RM coupling operators represented in solution coefficient space."""
        return self._poloidal_closure_projector.rm_coupling_solution_operators

    def lift_closure_scalar_output_operator_to_solution(self, operator: np.ndarray) -> np.ndarray:
        """Lift a closure-basis scalar-output operator into solution coefficients."""
        return self._poloidal_closure_projector.lift_scalar_output_operator_to_solution(operator)

    @cached_property
    def solver(self) -> "PoloidalSolver":
        """Helper exposing solve/orchestration routines built on poloidal operators."""
        from pynamit.simulation.induction.poloidal_solver import PoloidalSolver

        return PoloidalSolver(
            self, timed_solve=_timed_solve, timed_structured_solve=_timed_structured_solve
        )

    # -------------------------------------------------------------------------
    # Core Operators (Laplacian-based)
    # -------------------------------------------------------------------------

    @cached_property
    def m_imp_to_jr(self) -> np.ndarray:
        """Operator mapping m_imp to radial current jr.

        Physics: jr = (RI/mu0) * Laplacian(m_imp).
        (m_imp is the toroidal magnetic scalar T.)

        Returns
        -------
        np.ndarray
            Operator (matrix or diagonal) mapping potential to current.
        """
        return (self.RI / mu0) * self.solution_space.get_laplacian_operator(self.RI)

    @cached_property
    def m_ind_to_Br(self) -> np.ndarray:
        """Operator mapping induced potential m_ind to radial field Br.

        Physics: Br = -(RI^2) * Laplacian(m_ind).
        (m_ind is the poloidal magnetic scalar P.)

        Returns
        -------
        np.ndarray
            Operator mapping induced potential to radial field.
        """
        return -(self.RI**2) * self.solution_space.get_laplacian_operator(self.RI)

    @cached_property
    def m_ind_to_Br_pinv(self) -> np.ndarray:
        """Cached pseudoinverse of ``m_ind_to_Br``.

        This operator depends only on the solution basis and radius, not on
        conductance. Caching it here avoids recomputing the same dense
        pseudoinverse whenever conductance updates invalidate the downstream
        ``Br -> E`` conductivity operator.
        """
        m_ind_to_Br = np.asarray(to_dense(self.m_ind_to_Br))
        rcond = max(float(np.finfo(float).eps * max(m_ind_to_Br.shape)), 1e-15)
        return np.asarray(np.linalg.pinv(m_ind_to_Br, rcond=rcond))

    @property
    def E_df_to_d_m_ind_dt(self) -> float:
        """Scaling factor for induction equation.

        Physics: d(m_ind)/dt = (1/RI) * E_df

        Returns
        -------
        float
            Scaling factor for time derivative.
        """
        return 1.0 / self.RI

    def _project_scalar_coeffs_to_basis(self, coeffs: np.ndarray, target_basis: Any) -> np.ndarray:
        """Project scalar coefficients to ``target_basis`` through grid values."""
        coeffs = np.asarray(to_numpy(coeffs)).reshape(-1)
        if target_basis is self.solution_space and coeffs.size == int(
            self.solution_space.index_length
        ):
            return coeffs

        G_src = np.asarray(to_dense(self.solution_space.get_evaluation_matrix(self.grid)))
        P_tgt = np.asarray(to_dense(target_basis.construct_scalar_projection_matrix(self.grid)))
        if coeffs.size != G_src.shape[1]:
            raise ValueError(
                "Scalar coefficient size mismatch for projection: "
                f"coeffs={coeffs.shape}, G_src={G_src.shape}."
            )
        return np.asarray(P_tgt @ (G_src @ coeffs), dtype=float).reshape(-1)

    # -------------------------------------------------------------------------
    # JS-like Operators (Pre-Resistivity)
    # -------------------------------------------------------------------------

    @cached_property
    def m_imp_to_JS_coeffs_poloidal(self) -> np.ndarray:
        """Operator mapping m_imp to poloidal JS-like coefficients.

        Physics: J_p = -grad(m_imp)/mu0

        The poloidal JS-like component from imposed potential is simply
        a scaled identity in coefficient space.

        Returns
        -------
        np.ndarray
            Shape (L, L) operator where L = solution_space.index_length
        """
        L = self.solution_space.index_length
        return (1.0 / mu0) * np.eye(L)

    @cached_property
    def m_imp_to_JS_coeffs_toroidal(self) -> np.ndarray:
        """Operator mapping m_imp to toroidal JS-like coefficients via PFAC.

        Physics: Ve is a magnetic poloidal potential derived from PFAC,
        contributing to the toroidal component of the JS-like vector.

        Returns
        -------
        np.ndarray
            Shape (L, L) operator mapping m_imp to toroidal JS-like component.
        """
        return self._apply_imposed_toroidal_shielding(self.T_to_Ve)

    @cached_property
    def m_imp_to_JS_coeffs(self) -> np.ndarray:
        """Combined operator mapping m_imp to full JS-like VSH coefficients.

        Returns stacked [poloidal; toroidal] operator. This is the pre-resistivity
        vector used by the conductivity operator to obtain E.

        Returns
        -------
        np.ndarray
            Shape (2*L, L) operator mapping m_imp to [J_p; J_t] coefficients.
        """
        p_op = self.m_imp_to_JS_coeffs_poloidal
        t_op = self.m_imp_to_JS_coeffs_toroidal
        return np.vstack([p_op, t_op])

    # -------------------------------------------------------------------------
    # Sheet Current Operators
    # -------------------------------------------------------------------------

    def _apply_rm_poloidal_closure(self, t_to_ve: np.ndarray) -> np.ndarray:
        """Apply RM shielding closure operator in solution coefficient space."""
        if self._pfac.RM is None:
            return t_to_ve

        try:
            return self._poloidal_closure_projector.apply_rm_closure(t_to_ve)
        except ValueError:
            logger.warning(
                "Skipping RM poloidal closure: operator mismatch (%s vs %s).",
                np.asarray(self._rm_coupling_solution_operators.roundtrip_inv).shape
                if self._rm_coupling_solution_operators is not None
                else None,
                t_to_ve.shape,
            )
            return t_to_ve

    def _apply_imposed_toroidal_shielding(self, t_to_ve: np.ndarray) -> np.ndarray:
        """Apply RM shielding for imposed toroidal source (m_imp) unconditionally."""
        return self._apply_rm_poloidal_closure(t_to_ve)

    def _dynamic_psi_shielding_enabled(self) -> bool:
        """Return whether dynamic ``psi -> Ve`` PFAC should use RM shielding.

        The dynamic ``psi`` channel is a toroidal source, but the ``psi -> Ve``
        pathway is an induced poloidal response. It should therefore follow the
        magnetospheric shielding policy.
        """
        if self._pfac.RM is None:
            return False
        return bool(self._pfac.magnetospheric_shielding)

    def _get_dynamic_toroidal_pfac_operator(self) -> np.ndarray:
        """Return runtime dynamic-psi PFAC operator.

        This pathway carries the induced poloidal response of the dynamic
        toroidal source, so it follows the magnetospheric shielding policy:
        open when ``magnetospheric_shielding`` is false and RM-closed
        when it is true.
        """
        if self._dynamic_psi_shielding_enabled():
            return np.asarray(self.dynamic_toroidal_pfac_closed_operator)
        return np.asarray(self.dynamic_toroidal_pfac_open_operator)

    @cached_property
    def dynamic_toroidal_pfac_open_operator(self) -> np.ndarray:
        """Open-boundary PFAC operator for the dynamic ``psi`` channel."""
        return np.asarray(self.T_to_Ve_open)

    @cached_property
    def dynamic_toroidal_pfac_reaction_operator(self) -> np.ndarray:
        """Incremental runtime RM reaction on the dynamic ``psi`` PFAC operator."""
        if not self._dynamic_psi_shielding_enabled():
            shape = np.asarray(self.dynamic_toroidal_pfac_open_operator).shape
            return np.zeros(shape, dtype=float)
        return np.asarray(
            self.dynamic_toroidal_pfac_closed_operator - self.dynamic_toroidal_pfac_open_operator,
            dtype=float,
        )

    @cached_property
    def dynamic_toroidal_pfac_closed_operator(self) -> np.ndarray:
        """RM-closed PFAC operator for the dynamic ``psi`` channel."""
        return np.asarray(self.T_to_Ve)

    @cached_property
    def poloidal_rm_boundary_operators(self) -> PoloidalRMBoundaryOperators:
        """Induced poloidal boundary operators just above ``R_M``.

        The open operators represent the induced above-``R_M`` continuation.
        When ``magnetospheric_shielding`` is enabled, the effective induced
        boundary operators are set to zero by construction and the shielding
        operators are the canceled open branch.
        """
        n = int(self.solution_space.index_length)
        zeros = np.zeros((n, n), dtype=float)
        ops = self._rm_coupling_solution_operators
        if ops is None:
            return PoloidalRMBoundaryOperators(
                m_ind_to_br_rm_open=zeros,
                m_ind_to_br_rm_effective=zeros,
                m_ind_to_br_rm_shielding=zeros,
                dynamic_psi_to_ve_rm_open=zeros,
                dynamic_psi_to_ve_rm_effective=zeros,
                dynamic_psi_to_ve_rm_shielding=zeros,
            )

        ri_to_rm = np.asarray(ops.ri_to_rm, dtype=float)
        m_ind_to_br_rm_open = np.asarray(
            ri_to_rm @ np.asarray(to_dense(self.m_ind_to_Br)), dtype=float
        )
        dynamic_psi_to_ve_rm_open = np.asarray(
            ri_to_rm @ np.asarray(self.dynamic_toroidal_pfac_open_operator), dtype=float
        )

        if self._pfac.magnetospheric_shielding:
            m_ind_to_br_rm_effective = zeros
            dynamic_psi_to_ve_rm_effective = zeros
        else:
            m_ind_to_br_rm_effective = m_ind_to_br_rm_open
            dynamic_psi_to_ve_rm_effective = dynamic_psi_to_ve_rm_open

        return PoloidalRMBoundaryOperators(
            m_ind_to_br_rm_open=m_ind_to_br_rm_open,
            m_ind_to_br_rm_effective=m_ind_to_br_rm_effective,
            m_ind_to_br_rm_shielding=np.asarray(
                m_ind_to_br_rm_effective - m_ind_to_br_rm_open, dtype=float
            ),
            dynamic_psi_to_ve_rm_open=dynamic_psi_to_ve_rm_open,
            dynamic_psi_to_ve_rm_effective=dynamic_psi_to_ve_rm_effective,
            dynamic_psi_to_ve_rm_shielding=np.asarray(
                dynamic_psi_to_ve_rm_effective - dynamic_psi_to_ve_rm_open, dtype=float
            ),
        )

    @cached_property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid.

        This combines:
        1. Gradient of m_imp (scaled by -RI/mu0)
        2. PFAC coupling contribution via T_to_Ve

        Returns
        -------
        np.ndarray
            Shape (2, N_grid, L) operator mapping m_imp to JS components.
        """
        grad_op = as_linear_map(self.solution_space.get_gradient_matrix(self.grid))
        G_grad = (1.0 / self.RI) * (grad_op * (-self.RI / mu0))
        G_total = to_dense(G_grad).reshape(2, -1, self.solution_space.index_length)

        # Add PFAC coupling: JS += G_Ve_to_JS @ T_to_Ve @ m_imp
        T_to_Ve_eff = self._apply_imposed_toroidal_shielding(self.T_to_Ve)
        JS_coupling = np.tensordot(self.G_Ve_to_JS, T_to_Ve_eff, axes=([2], [0]))
        G_total = G_total + JS_coupling

        return G_total

    @cached_property
    def G_Ve_to_JS(self) -> np.ndarray:
        """Operator mapping external potential Ve to sheet current.

        This is the VSH induction operator: -1/mu0 * Curl @ Scaling.

        Returns
        -------
        np.ndarray
            Shape (2, N_grid, L) operator.
        """
        scaling_op = self.solution_space.get_potential_scaling_operator()
        curl_op = as_linear_map(self.solution_space.get_curl_matrix(self.grid))

        G_lin = (-1.0 / mu0) * (curl_op @ scaling_op)
        return to_dense(G_lin).reshape(2, -1, self.solution_space.index_length)

    # -------------------------------------------------------------------------
    # PFAC Coupling
    # -------------------------------------------------------------------------

    @cached_property
    def T_to_Ve(self) -> np.ndarray:
        """Mapping from toroidal potential T to poloidal potential Ve.

        Delegates to PFACIntegrator for the actual computation.

        Returns
        -------
        np.ndarray
            Shape (L, L) operator mapping T to Ve with RM-closed boundary.
        """
        # The PFAC integrator returns an xr.DataArray; extract values
        T_to_Ve_da = self._pfac.compute_T_to_Ve(
            self.G_Ve_to_JS_closure, self.grid, rm_boundary_mode="closed"
        )
        return T_to_Ve_da.values

    @cached_property
    def T_to_Ve_open(self) -> np.ndarray:
        """Open-boundary mapping from toroidal potential T to poloidal potential Ve."""
        T_to_Ve_da = self._pfac.compute_T_to_Ve(
            self.G_Ve_to_JS_closure, self.grid, rm_boundary_mode="open"
        )
        return T_to_Ve_da.values

    @cached_property
    def toroidal_rm_closure_operators(self):
        """Operators describing the ``R_M`` normal-current closure of dynamic ``alpha``."""
        return self._pfac.compute_toroidal_rm_closure_operators(self.grid)

    @cached_property
    def G_Ve_to_JS_closure(self) -> np.ndarray:
        """Closure-basis version of Ve-to-JS operator for PFAC integration.

        Returns
        -------
        np.ndarray
            Shape (2, N_grid, L_closure) operator using PFAC closure basis.
        """
        closure_basis = self._poloidal_closure_projector.closure_basis
        scaling_op = closure_basis.get_potential_scaling_operator()
        curl_op = as_linear_map(closure_basis.get_curl_matrix(self.grid))

        G_lin = (-1.0 / mu0) * (curl_op @ scaling_op)
        return to_dense(G_lin).reshape(2, -1, closure_basis.index_length)

    # -------------------------------------------------------------------------
    # Projection Operators
    # -------------------------------------------------------------------------

    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Projection matrix from grid values to basis coefficients.

        For GL grids with quadrature weights, uses exact analysis.
        For other grids (e.g., Cubed-Sphere), uses pseudo-inverse.

        Returns
        -------
        np.ndarray
            Projection matrix mapping grid values to coefficients.
        """
        if hasattr(self.grid, "weights"):
            # Exact quadrature projection
            weights = self.grid.weights
            G = to_numpy(self.solution_space.get_evaluation_matrix(self.grid))

            # Weighted least-squares: P = (G^T W G)^{-1} G^T W
            GtW = G.T * weights
            M = GtW @ G
            P = np.linalg.solve(M, GtW)
            return asarray(P)

        # Fallback to pseudo-inverse
        G = to_dense(self.solution_space.get_evaluation_matrix(self.grid))
        return tensor_pinv(G, n_leading_flattened=1)

    # -------------------------------------------------------------------------
    # Least-Squares Problem Construction
    # -------------------------------------------------------------------------

    def build_least_squares_subproblem(
        self,
        constraint_scalar_operator: np.ndarray,
        E_constraint_operator: Optional[LinearMap] = None,
        connect_hemispheres: bool = True,
        ih_constraint_scaling: float = 1.0,
        regularization_lambda: float = 0.0,
        m_imp_selector: np.ndarray = None,
        weighting: str = "none",
    ) -> StructuredLeastSquaresSubproblem:
        """Build the structured least-squares subproblem for ``m_imp``.

        The problem structure is:
            minimize || A @ m_imp - b ||^2

        Where A consists of:
        1. jr constraint: jr_map @ (m_imp_to_jr @ m_imp) = jr_data
        2. E-field mapping constraint (if connect_hemispheres)
        3. Tikhonov regularization (if lambda > 0)

        Parameters
        ----------
        constraint_scalar_operator : np.ndarray
            Operator mapping coefficients to the configured constraint scalar.
        E_constraint_operator : LinearMap, optional
            Operator enforcing E-field mapping at low latitudes.
        connect_hemispheres : bool
            Whether to include interhemispheric E-field constraint.
        ih_constraint_scaling : float
            Scaling factor for the IH constraint term.
        regularization_lambda : float
            Tikhonov regularization weight.
        m_imp_selector : np.ndarray
            Full-to-reduced orthonormal selector used to solve for ``m_imp`` in
            reduced coordinates. Gauge enforcement is handled by this selector.

        Returns
        -------
        StructuredLeastSquaresSubproblem
            The assembled structured subproblem.
        """
        from pynamit.math.linear_map import diagonal_linear_map

        data_terms: list[StructuredLeastSquaresDataTerm] = []
        n_full = int(self.solution_space.index_length)
        selector = np.asarray(m_imp_selector, dtype=float)
        if selector.ndim != 2 or selector.shape[0] != n_full:
            raise ValueError(
                "m_imp_selector must have shape "
                f"({n_full}, n_reduced), got {getattr(selector, 'shape', None)}."
            )
        selector_map = as_linear_map(selector)
        solution_size = int(selector.shape[1])

        # 1. Radial current constraint: jr_map @ m_imp_to_jr @ m_imp = jr_data
        op_apex = as_linear_map(constraint_scalar_operator)
        op_m_to_jr = as_linear_map(self.m_imp_to_jr)
        op_jr = (op_apex @ op_m_to_jr) @ selector_map

        # Calculate weights for the jr constraint (Br-based weighting to handle equatorial singularity)
        jr_weight = None
        if weighting != "none":
            br = to_numpy(self.b_field.vec.r).flatten()
            if weighting == "quadratic":
                jr_weight = np.abs(br)  # sqrt(Br^2) = |Br|
            elif weighting == "linear":
                jr_weight = np.sqrt(np.abs(br))  # sqrt(|Br|)
            # Normalize to avoid scaling rows too far from other constraint magnitudes
            if jr_weight is not None and np.max(jr_weight) > 0:
                jr_weight = jr_weight / np.max(jr_weight)
        data_terms.append(
            StructuredLeastSquaresDataTerm(
                operator=op_jr, data_shape=(op_jr.shape[0],), sqrt_weight=jr_weight
            )
        )

        # 2. E-field mapping constraint (interhemispheric)
        if connect_hemispheres and E_constraint_operator is not None:
            op_E = as_linear_map(E_constraint_operator) * ih_constraint_scaling
            op_E = op_E @ selector_map
            data_terms.append(
                StructuredLeastSquaresDataTerm(
                    operator=op_E, data_shape=(op_E.shape[0],), sqrt_weight=None
                )
            )

        # 3. Tikhonov regularization
        regularization_terms: list[StructuredLeastSquaresRegularizationTerm] = []
        if regularization_lambda > 0:
            identity_op = diagonal_linear_map(xp.ones(solution_size))
            regularization_terms.append(
                StructuredLeastSquaresRegularizationTerm(
                    operator=identity_op, weight=float(regularization_lambda)
                )
            )

        return StructuredLeastSquaresSubproblem(
            solution_shape=solution_size,
            data_terms=tuple(data_terms),
            regularization_terms=tuple(regularization_terms),
        )

    def build_least_squares_problem(
        self,
        constraint_scalar_operator: np.ndarray,
        E_constraint_operator: Optional[LinearMap] = None,
        connect_hemispheres: bool = True,
        ih_constraint_scaling: float = 1.0,
        regularization_lambda: float = 0.0,
        m_imp_selector: np.ndarray = None,
        weighting: str = "none",
    ) -> "LeastSquaresProblem":
        """Compatibility wrapper returning the materialized LS problem."""
        return self.build_least_squares_subproblem(
            constraint_scalar_operator=constraint_scalar_operator,
            E_constraint_operator=E_constraint_operator,
            connect_hemispheres=connect_hemispheres,
            ih_constraint_scaling=ih_constraint_scaling,
            regularization_lambda=regularization_lambda,
            m_imp_selector=m_imp_selector,
            weighting=weighting,
        ).problem

    def compute_rhs_from_jr(
        self, jr_coeffs: np.ndarray, constraint_scalar_operator: np.ndarray
    ) -> np.ndarray:
        """Compute RHS vector for the least-squares problem from jr data.

        This is analogous to ToroidalSystemMatrices.compute_K_from_E().

        Parameters
        ----------
        jr_coeffs : np.ndarray
            Radial current coefficients (input data).
        constraint_scalar_operator : np.ndarray
            Operator mapping coefficients to the configured constraint scalar.

        Returns
        -------
        np.ndarray
            RHS vector for the jr constraint term.
        """
        op_rhs = as_linear_map(constraint_scalar_operator)
        return op_rhs.matvec(asarray(jr_coeffs).reshape(-1))

    def _extract_toroidal_potential_coeffs(self, E_coeffs: Any) -> np.ndarray:
        """Extract the toroidal electric-potential coefficients from E coefficients."""
        return asarray(self.solution_space.get_toroidal_potential_coeffs(E_coeffs))

    def _apply_E_constraint_operator(
        self, E_constraint_operator: Any, E_coeffs: Any
    ) -> np.ndarray:
        """Apply the interhemispheric E-constraint operator to one or many scenarios."""
        if not hasattr(E_constraint_operator, "apply"):
            raise TypeError(
                "E_map_constraint_operator must provide an 'apply' method (ConstraintOperator)."
            )
        return asarray(E_constraint_operator.apply(asarray(E_coeffs)))

    @staticmethod
    def _reshape_constraint_rhs_block(term: Any) -> np.ndarray:
        """Flatten a constraint tensor to the canonical RHS block layout."""
        term_arr = asarray(term)
        if term_arr.ndim <= 2:
            return xp.reshape(term_arr, (-1,))
        return xp.reshape(term_arr, (-1, term_arr.shape[-1]))

    def _build_m_imp_rhs_entries(
        self,
        problem: "LeastSquaresProblem",
        *,
        E_direct_coeffs: Optional[np.ndarray] = None,
        E_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
    ) -> list[Optional[Any]]:
        """Build RHS entries for `m_imp` feedback solves."""
        rhs_entries = [None] * problem.num_data_terms
        if connect_hemispheres and E_constraint_operator is not None:
            if E_direct_coeffs is None:
                raise ValueError(
                    "E_direct_coeffs is required when E-constraint feedback is active."
                )
            b_E = self._apply_E_constraint_operator(E_constraint_operator, E_direct_coeffs)
            if len(rhs_entries) > 1:
                rhs_entries[1] = ih_constraint_scaling * self._reshape_constraint_rhs_block(b_E)
        return rhs_entries

    def _solve_m_imp_feedback_block(
        self,
        *,
        problem: "LeastSquaresProblem",
        solver: Any,
        rhs_entries: list[Optional[Any]],
        num_expected_scenarios: int,
    ) -> np.ndarray:
        """Solve batched `m_imp` feedback responses using the cached problem SVD."""
        rhs_block, _, num_scenarios = problem.assemble_rhs_block(rhs_entries)
        if rhs_block is None:
            op_rows = problem.get_system_operator().shape[0]
            rhs_block = xp.zeros((op_rows, num_expected_scenarios), dtype=float)
            num_scenarios = num_expected_scenarios
        rhs_block = asarray(rhs_block)

        u, s, vt = problem.svd
        if s.size == 0:
            m_imp_block = xp.zeros((problem.solution_size, num_scenarios), dtype=rhs_block.dtype)
        else:
            tol = getattr(solver, "tolerance", 0.0)
            cutoff = tol * s[0] if tol > 0 else 0.0
            s_inv = xp.where(s > cutoff, 1.0 / s, 0.0)
            tmp = u.T.conj() @ rhs_block
            tmp = s_inv[:, None] * tmp
            m_imp_block = vt.T.conj() @ tmp

        if int(num_scenarios) != int(num_expected_scenarios):
            raise RuntimeError(
                f"Expected {num_expected_scenarios} scenarios when building induction operator, "
                f"got {num_scenarios}."
            )
        return asarray(m_imp_block)

    # -------------------------------------------------------------------------
    # Additional Operators for m_ind (induced potential)
    # -------------------------------------------------------------------------

    @cached_property
    def G_m_ind_to_JS(self) -> Optional[np.ndarray]:
        """Operator mapping m_ind to sheet current on grid.

        This operator combines two physical effects:
        1. Local "Vacuum" Induction: m_ind -> E -> J.
        2. Gap Region / Magnetospheric Boundary Coupling: m_ind -> Coupling -> J.

        Returns
        -------
        np.ndarray or None
            Shape (2, N_grid, L) operator, or None if not applicable.
        """
        if self.G_Ve_to_JS is None:
            return None

        G = self.G_Ve_to_JS.copy()

        # Add magnetospheric coupling if RM is defined and lock is enabled
        if self._pfac.RM is not None and self._pfac.magnetospheric_shielding:
            ops = self._rm_coupling_solution_operators
            if ops is not None:
                rm_feedback_op = np.asarray(ops.feedback)
                G = G + np.tensordot(self.G_Ve_to_JS, rm_feedback_op, axes=([2], [0]))

        return G

    # -------------------------------------------------------------------------
    # Time Evolution (Induction Logic)
    # -------------------------------------------------------------------------

    def build_induction_matrix(
        self,
        feedback_system: "MImpFeedbackSystem",
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_ind_to_E_operator: Any = None,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Construct the dense matrix for the induction operator (m_ind -> E_df)."""
        return self.solver.build_induction_matrix(
            feedback_system=feedback_system,
            solver=solver,
            E_map_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
            m_ind_to_E_operator=m_ind_to_E_operator,
            m_imp_to_E_operator=m_imp_to_E_operator,
        )

    def get_induction_operator(
        self,
        feedback_system: "MImpFeedbackSystem",
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_ind_to_E_operator: Any = None,
        m_imp_to_E_operator: Any = None,
    ) -> "LinearMap":
        """Get matrix-free induction operator (m_ind -> E_df)."""
        return self.solver.get_induction_operator(
            feedback_system=feedback_system,
            solver=solver,
            E_map_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
            m_ind_to_E_operator=m_ind_to_E_operator,
            m_imp_to_E_operator=m_imp_to_E_operator,
        )

    def solve_for_m_imp(
        self,
        E_direct_coeffs: np.ndarray,
        feedback_system: "MImpFeedbackSystem",
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for m_imp given E_direct and return `(m_imp, E_imp)`."""
        return self.solver.solve_for_m_imp(
            E_direct_coeffs=E_direct_coeffs,
            feedback_system=feedback_system,
            solver=solver,
            E_map_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
            m_imp_to_E_operator=m_imp_to_E_operator,
        )

    def compute_rates(
        self,
        m_ind: np.ndarray,
        t: float,
        E_coeffs_noind: np.ndarray,
        induction_matrix: Optional[np.ndarray] = None,
        m_ind_to_E_operator: Any = None,
        # Solvers for implicit Feedback
        feedback_system: Optional["MImpFeedbackSystem"] = None,
        solver: Optional[Any] = None,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Calculate d(m_ind)/dt rates."""
        return self.solver.compute_rates(
            m_ind=m_ind,
            t=t,
            E_coeffs_noind=E_coeffs_noind,
            induction_matrix=induction_matrix,
            m_ind_to_E_operator=m_ind_to_E_operator,
            feedback_system=feedback_system,
            solver=solver,
            E_map_constraint_operator=E_map_constraint_operator,
            ih_constraint_scaling=ih_constraint_scaling,
            connect_hemispheres=connect_hemispheres,
            m_imp_to_E_operator=m_imp_to_E_operator,
        )

    def get_potential_to_JS_operator(self, potential_type: str) -> "LinearMap":
        """Get spectral (VSH) pre-resistivity operator for given potential type.

        This operator maps magnetic scalars to the JS-like vector coefficients.
        The resistivity operator (eta) is applied afterward to obtain E.
        """
        L = self.solution_space.index_length

        if potential_type in ("m_imp", "psi"):
            # Poloidal part from toroidal magnetic scalar source.
            p_op = (1.0 / mu0) * np.eye(L)
            # PFAC coupling contributes to toroidal component of JS-like vector.
            if potential_type == "m_imp":
                t_op = self._apply_imposed_toroidal_shielding(self.T_to_Ve)
            else:
                t_op = self._get_dynamic_toroidal_pfac_operator()
            return as_linear_map(np.vstack([p_op, t_op]))

        elif potential_type == "m_ind":
            # E_t = -1/mu0 * Scaling(m_ind) * Y^T
            scaling = self.solution_space.get_potential_scaling_operator()
            t_mat = (-1.0 / mu0) * to_dense(scaling)

            if self._pfac.RM is not None and self._pfac.magnetospheric_shielding:
                ops = self._rm_coupling_solution_operators
                if ops is not None:
                    rm_feedback_op = np.asarray(ops.feedback)
                    t_mat = t_mat @ (np.eye(L) + rm_feedback_op)

            return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))

        elif potential_type == "Br":
            # Br path is purely toroidal, represented in solution coefficient space.
            ops = self._rm_coupling_solution_operators
            if ops is None:
                raise ValueError(
                    "Br pathway requires RM coupling operators when RM is configured."
                )

            rm_to_ri = np.asarray(ops.rm_to_ri)
            roundtrip_inv = np.asarray(ops.roundtrip_inv)
            # Explicit imposed Br forcing is always treated as an imposed RM
            # boundary channel and therefore uses the closed roundtrip branch.
            br_factor_op = -(rm_to_ri @ roundtrip_inv)

            scaling = np.asarray(to_dense(self.solution_space.get_potential_scaling_operator()))
            t_mat = (-1.0 / mu0) * (scaling @ br_factor_op @ self.m_ind_to_Br_pinv)
            return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))

        raise ValueError(f"Unknown potential_type: {potential_type}")

    def steady_state_m_ind(
        self, E_coeffs_noind: np.ndarray, induction_matrix: Any, solver: str = "lsmr"
    ) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        return self.solver.steady_state_m_ind(
            E_coeffs_noind=E_coeffs_noind, induction_matrix=induction_matrix, solver=solver
        )
