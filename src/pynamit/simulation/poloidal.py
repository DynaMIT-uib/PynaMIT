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
from typing import Any, Literal, Optional, TYPE_CHECKING

import numpy as np
from functools import cached_property

from pynamit.utils import to_numpy, asarray, xp, tensor_pinv
from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.math.constants import mu0
from pynamit.simulation.geometry_utils import to_dense
from pynamit.simulation.poloidal_closure import PoloidalClosureProjector, RMCouplingOperators
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from pynamit.utils import use_jax
from pynamit.primitives.field import Field


if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.field import Field
    from pynamit.simulation.pfac import PFACIntegrator
    from pynamit.math.least_squares_problem import LeastSquaresProblem

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
    solution_basis : Basis
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
        solution_basis: "Basis",
        grid: "Grid",
        b_field: "Field",
        RI: float,
        pfac_integrator: "PFACIntegrator",
    ):
        self.basis = basis
        self.solution_basis = solution_basis
        self.grid = grid
        self.b_field = b_field
        self.RI = RI
        self._pfac = pfac_integrator

    @cached_property
    def _poloidal_closure_projector(self) -> PoloidalClosureProjector:
        """Closure-basis projector for poloidal RM/PFAC operators."""
        return PoloidalClosureProjector(
            solution_basis=self.solution_basis,
            closure_basis=self._pfac.basis,
            grid=self.grid,
            pfac_integrator=self._pfac,
        )

    @cached_property
    def _rm_coupling_solution_operators(self) -> Optional[RMCouplingOperators]:
        """RM coupling operators represented in solution coefficient space."""
        return self._poloidal_closure_projector.rm_coupling_solution_operators

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
        return (self.RI / mu0) * self.solution_basis.get_laplacian_operator(self.RI)

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
        return -(self.RI**2) * self.solution_basis.get_laplacian_operator(self.RI)

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
            Shape (L, L) operator where L = solution_basis.index_length
        """
        L = self.solution_basis.index_length
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
        return self._apply_imposed_toroidal_poloidal_lock(self.T_to_Ve)

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

    def _apply_imposed_toroidal_poloidal_lock(self, t_to_ve: np.ndarray) -> np.ndarray:
        """Apply RM closure for imposed toroidal source (m_imp) unconditionally."""
        return self._apply_rm_poloidal_closure(t_to_ve)

    def _apply_dynamic_toroidal_poloidal_lock(self, t_to_ve: np.ndarray) -> np.ndarray:
        """Apply RM closure for dynamic toroidal source (psi) when enabled."""
        if not self._pfac.magnetospheric_poloidal_lock:
            return t_to_ve
        if not self._pfac.lock_toroidal_source_channels:
            return t_to_ve
        return self._apply_rm_poloidal_closure(t_to_ve)

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
        grad_op = as_linear_map(self.solution_basis.get_gradient_matrix(self.grid))
        G_grad = (1.0 / self.RI) * (grad_op * ((-self.RI / mu0)))
        G_total = to_dense(G_grad).reshape(2, -1, self.solution_basis.index_length)

        # Add PFAC coupling: JS += G_Ve_to_JS @ T_to_Ve @ m_imp
        T_to_Ve_eff = self._apply_imposed_toroidal_poloidal_lock(self.T_to_Ve)
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
        scaling_op = self.solution_basis.get_potential_scaling_operator()
        curl_op = as_linear_map(self.solution_basis.get_curl_matrix(self.grid))

        G_lin = (-1.0 / mu0) * (curl_op @ scaling_op)
        return to_dense(G_lin).reshape(2, -1, self.solution_basis.index_length)

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
            Shape (L, L) operator mapping T to Ve.
        """
        # The PFAC integrator returns an xr.DataArray; extract values
        T_to_Ve_da = self._pfac.compute_T_to_Ve(self.G_Ve_to_JS_closure, self.grid)
        return T_to_Ve_da.values

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
            G = to_numpy(self.solution_basis.get_G(self.grid))

            # Weighted least-squares: P = (G^T W G)^{-1} G^T W
            GtW = G.T * weights
            M = GtW @ G
            P = np.linalg.solve(M, GtW)
            return asarray(P)

        # Fallback to pseudo-inverse
        G = to_dense(self.solution_basis.get_evaluation_matrix(self.grid))
        return tensor_pinv(G, n_leading_flattened=1)

    # -------------------------------------------------------------------------
    # Least-Squares Problem Construction
    # -------------------------------------------------------------------------

    def build_least_squares_problem(
        self,
        jr_map_operator: np.ndarray,
        E_constraint_operator: Optional[LinearMap] = None,
        connect_hemispheres: bool = True,
        ih_constraint_scaling: float = 1.0,
        regularization_lambda: float = 0.0,
        use_pinning: bool = False,
        weighting: str = "none",
    ) -> "LeastSquaresProblem":
        """Build the least-squares problem for m_imp.

        The problem structure is:
            minimize || A @ m_imp - b ||^2

        Where A consists of:
        1. jr constraint: jr_map @ (m_imp_to_jr @ m_imp) = jr_data
        2. E-field mapping constraint (if connect_hemispheres)
        3. Pinning constraint (if use_pinning): m_imp[0] = 0
        4. Tikhonov regularization (if lambda > 0)

        Parameters
        ----------
        jr_map_operator : np.ndarray
            Operator mapping jr coefficients to apex current (jr_map_sim).
        E_constraint_operator : LinearMap, optional
            Operator enforcing E-field mapping at low latitudes.
        connect_hemispheres : bool
            Whether to include interhemispheric E-field constraint.
        ih_constraint_scaling : float
            Scaling factor for the IH constraint term.
        regularization_lambda : float
            Tikhonov regularization weight.
        use_pinning : bool
            Whether to specificall pin the first potential coefficient to zero.
            Necessary for Cubed-Sphere basis to resolve the constant null mode.

        Returns
        -------
        LeastSquaresProblem
            The assembled least-squares problem.
        """
        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import diagonal_linear_map

        operators = []
        data_shapes = []
        sqrt_weights = []

        # 1. Radial current constraint: jr_map @ m_imp_to_jr @ m_imp = jr_data
        op_apex = as_linear_map(jr_map_operator)
        op_m_to_jr = as_linear_map(self.m_imp_to_jr)
        op_jr = op_apex @ op_m_to_jr

        operators.append(op_jr)
        data_shapes.append((op_jr.shape[0],))

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
        sqrt_weights.append(jr_weight)

        # 2. E-field mapping constraint (interhemispheric)
        if connect_hemispheres and E_constraint_operator is not None:
            op_E = E_constraint_operator.with_scaling(ih_constraint_scaling)
            operators.append(op_E)
            data_shapes.append((op_E.shape[0],))
            sqrt_weights.append(None)  # No weighting for E-field constraint

        # 3. Pinning Constraint
        if use_pinning:
            n = self.solution_basis.index_length
            if hasattr(self.solution_basis, "get_scalar_gauge_constraint_matrix"):
                row = np.asarray(
                    self.solution_basis.get_scalar_gauge_constraint_matrix(
                        n_coeff=n,
                        mode="mean_zero",
                    )
                )
            else:
                row = np.zeros((1, n))
                row[0, 0] = 1.0
            op_pin = as_linear_map(row)
            operators.append(op_pin)
            data_shapes.append((op_pin.shape[0],))
            sqrt_weights.append(None)  # No weighting for pinning constraint

        # 4. Tikhonov regularization
        reg_ops = []
        reg_weights = []
        if regularization_lambda > 0:
            n = self.solution_basis.index_length
            identity_op = diagonal_linear_map(xp.ones(n))
            reg_ops.append(identity_op)
            reg_weights.append(regularization_lambda)

        return LeastSquaresProblem(
            A=operators,
            solution_shape=self.solution_basis.index_length,
            data_shapes=data_shapes,
            sqrt_weights=sqrt_weights,
            regularization_matrices=reg_ops,
            regularization_weights=reg_weights,
        )

    def compute_rhs_from_jr(
        self,
        jr_coeffs: np.ndarray,
        jr_map_operator: np.ndarray,
    ) -> np.ndarray:
        """Compute RHS vector for the least-squares problem from jr data.

        This is analogous to ToroidalSystemMatrices.compute_K_from_E().

        Parameters
        ----------
        jr_coeffs : np.ndarray
            Radial current coefficients (input data).
        jr_map_operator : np.ndarray
            Operator mapping jr to apex current (jr_map_spectral or jr_map_sim).

        Returns
        -------
        np.ndarray
            RHS vector for the jr constraint term.
        """
        op_rhs = as_linear_map(jr_map_operator)
        return op_rhs.matvec(asarray(jr_coeffs).reshape(-1))

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
        if self._pfac.RM is not None and self._pfac.magnetospheric_poloidal_lock:
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
        problem: "LeastSquaresProblem",
        solver: Any,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_ind_to_E_operator: Any = None,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Construct the dense matrix for the induction operator (m_ind -> E_df).

        This moves logic previously in State._build_m_ind_to_E_df_matrix.
        """
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.solution_basis.index_length
        
        # 1. Get Direct Contribution (m_ind -> E_ind)
        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        m_ind_to_E = m_ind_to_E_operator
        
        E_direct_dense = asarray(m_ind_to_E.to_dense()).reshape(2, n, n)
        
        # 2. Compute Imposed Feedback (Constraint Response)
        # We need to construct the RHS for each "scenario" (each basis function being excited).
        # The problem.num_data_terms includes: [jr_constraint, E_mapping, pinning_constraint(optional)]
        rhs_entries = [None] * problem.num_data_terms if problem.num_data_terms > 0 else []

        if connect_hemispheres and E_map_constraint_operator is not None:
            # Canonical path: ConstraintOperator with rank-4 tensor semantics.
            E_map_op = E_map_constraint_operator
            
            if hasattr(E_map_op, "apply"):
                 term = E_map_op.apply(E_direct_dense)
            else:
                 raise TypeError(
                     "E_map_constraint_operator must provide an 'apply' method "
                     "(ConstraintOperator)."
                 )
            
            # Reshape b_E_block from (2, Mask, Batch) to (2*Mask, Batch)
            # This matches the flattening logic used for the LHS operator.
            b_E_block = xp.reshape(term, (-1, n))

            if len(rhs_entries) > 1:
                rhs_entries[1] = ih_constraint_scaling * b_E_block

        # IMPORTANT: If pinning is active (num_data_terms > 2), we leave that entry as None (Zero).
        # This is correct because the induction terms (scenarios) do not drive the pinning value.
        # However, assemble_rhs_block expects rhs_entries to match num_data_terms.
        
        rhs_block, _, num_scenarios = problem.assemble_rhs_block(rhs_entries)
        
        if rhs_block is None:
            op_rows = problem.get_system_operator().shape[0]
            rhs_block = xp.zeros((op_rows, n), dtype=E_direct_dense.dtype)
            num_scenarios = n
        rhs_block = asarray(rhs_block)

        # Solve in batch using cached SVD
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

        if num_scenarios != n:
            raise RuntimeError(
                f"Expected {n} scenarios when building induction operator, got {num_scenarios}."
            )

        # Map imposed potential response back to E-field coefficients
        m_imp_to_E = as_linear_map(m_imp_to_E_operator)
        
        m_imp_flat = asarray(m_imp_block)
        E_imp_flat = m_imp_to_E.matmat(m_imp_flat)
        E_imp_block = asarray(E_imp_flat).reshape(2, n, n)

        total_E = E_direct_dense + E_imp_block

        # Basis-agnostic extraction of the induction-driving E-field part (Toroidal Potential)
        # total_E has shape (2, n, n) = (component, coeffs, scenarios)
        curled_scenarios = self.solution_basis.get_toroidal_potential_coeffs(total_E)
        
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
        """Get matrix-free induction operator (m_ind -> E_df).

        Returns a LinearMap that computes the divergence-free E-field
        contribution from m_ind without materializing the full dense matrix.
        Each matvec application solves for the m_imp feedback.

        Note: The adjoint (rmatvec) requires the dense matrix, which is
        cached on first use. For truly matrix-free operation, use CG solver
        which only needs matvec (via normal equations).

        Parameters
        ----------
        problem : LeastSquaresProblem
            The m_imp least-squares problem.
        solver : LeastSquaresSolver
            Solver for the m_imp problem.
        preconditioner : optional
            Preconditioner for the solver.
        E_map_constraint_operator : optional
            Operator for interhemispheric constraint.
        ih_constraint_scaling : float
            Scaling for interhemispheric constraint.
        connect_hemispheres : bool
            Whether to include hemisphere coupling.
        m_ind_to_E_operator : Any
            Operator mapping m_ind to E coefficients (required).
        m_imp_to_E_operator : Any
            Operator mapping m_imp to E coefficients (required).

        Returns
        -------
        LinearMap
            Matrix-free operator for m_ind -> E_df.
        """
        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        if m_imp_to_E_operator is None:
            raise ValueError("m_imp_to_E_operator is required")
        n = self.solution_basis.index_length

        # Get the m_ind -> E operator
        m_ind_to_E = as_linear_map(m_ind_to_E_operator)

        # Cache for dense matrix (built lazily for rmatvec)
        _cache = {"dense_matrix": None}

        def _get_dense():
            """Get or build dense matrix (cached)."""
            if _cache["dense_matrix"] is None:
                _cache["dense_matrix"] = self.build_induction_matrix(
                    problem=problem,
                    solver=solver,
                    E_map_constraint_operator=E_map_constraint_operator,
                    ih_constraint_scaling=ih_constraint_scaling,
                    connect_hemispheres=connect_hemispheres,
                    m_ind_to_E_operator=m_ind_to_E_operator,
                    m_imp_to_E_operator=m_imp_to_E_operator,
                )
            return _cache["dense_matrix"]

        def matvec(m_ind_vec):
            """Compute E_df = induction_operator @ m_ind."""
            m_ind_vec = asarray(m_ind_vec).flatten()

            # 1. Direct E from m_ind
            E_ind_coeffs = m_ind_to_E.matvec(m_ind_vec).reshape(2, -1)

            # 2. Add feedback (m_imp) if coupled
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

            # 3. Extract toroidal potential (E_df)
            E_df = self.solution_basis.get_toroidal_potential_coeffs(E_ind_coeffs)
            return asarray(E_df).flatten()

        def rmatvec(y):
            """Compute adjoint: induction_operator.T @ y.

            For LSMR we need rmatvec. Uses cached dense matrix.
            """
            dense_matrix = _get_dense()
            return asarray(dense_matrix.T @ asarray(y).flatten())

        return LinearMap(
            shape=(n, n),
            dtype=np.float64,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _to_dense=_get_dense,
            source=None,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
         """Solve for m_imp given E_direct (e.g. from m_ind + E_noind).
         
         Returns (m_imp, E_imp).
         """
         rhs_entries = [None] * problem.num_data_terms
         
         if connect_hemispheres and E_map_constraint_operator is not None:
            E_map_op = E_map_constraint_operator
            E_direct_input = asarray(E_direct_coeffs)
            
            if hasattr(E_map_op, "apply"):
                 b_E = E_map_op.apply(E_direct_input)
            else:
                 raise TypeError(
                     "E_map_constraint_operator must provide an 'apply' method "
                     "(ConstraintOperator)."
                 )
            
            if len(rhs_entries) > 1:
                rhs_entries[1] = ih_constraint_scaling * xp.reshape(b_E, (-1,))
                
         solution = _timed_solve(
             "poloidal.m_imp",
             solver,
             problem=problem,
             rhs=rhs_entries,
             preconditioner=preconditioner,
         )
         if solution is None:
             m_imp = xp.zeros(self.solution_basis.index_length)
         else:
             m_imp = asarray(solution)
             
         if m_imp_to_E_operator is None:
             raise ValueError("m_imp_to_E_operator is required")

         # Compute E_imp
         m_imp_to_E = as_linear_map(m_imp_to_E_operator)
             
         E_imp = m_imp_to_E.matvec(m_imp).reshape(2, -1)
         return m_imp, E_imp

    def compute_rates(
        self,
        m_ind: np.ndarray,
        t: float,
        E_coeffs_noind: np.ndarray,
        induction_matrix: Optional[np.ndarray] = None,
        m_ind_to_E_operator: Any = None,
        # Solvers for implicit Feedback
        problem: Optional[Any] = None,
        solver: Optional[Any] = None,
        preconditioner: Optional[Any] = None,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Any = None,
    ) -> np.ndarray:
        """Calculate d(m_ind)/dt rates."""
        if m_ind_to_E_operator is None:
            raise ValueError("m_ind_to_E_operator is required")
        if m_imp_to_E_operator is None:
            raise ValueError("m_imp_to_E_operator is required")
        # E_df_total = E_df_ind + E_df_noind
        # If we have the dense induction matrix (m_ind -> E_df), use it directly.
        # Otherwise, compute E_ind via operator.
        
        backend_m_ind = asarray(m_ind)
        
        # Extract Toroidal Potential part of no-induction E-field
        E_noind_field = Field.from_coefficients(self.solution_basis, E_coeffs_noind, field_type="tangential")
        E_df_noind = asarray(E_noind_field.toroidal_potential().coeffs)
        
        if induction_matrix is not None:
             # Matrix ALREADY includes the feedback (m_imp) response!
             E_df_ind = asarray(induction_matrix) @ backend_m_ind
        else:
             # Explicit construction: Must manually solve for m_imp feedback
             # 1. Direct E from m_ind
             m_ind_to_E = m_ind_to_E_operator
             E_ind_coeffs = m_ind_to_E.matvec(backend_m_ind).reshape(2, -1)
             
             # 2. Add Feedback (m_imp) if coupled
             if connect_hemispheres and problem is not None:
                  # Use PoloidalSystemMatrices.solve_for_m_imp which handles dependencies
                  _, E_imp = self.solve_for_m_imp(
                       E_direct_coeffs=E_ind_coeffs,
                       problem=problem,
                       solver=solver,
                       preconditioner=preconditioner,
                       E_map_constraint_operator=E_map_constraint_operator,
                       ih_constraint_scaling=ih_constraint_scaling,
                       connect_hemispheres=connect_hemispheres,
                       m_imp_to_E_operator=m_imp_to_E_operator
                  )
                  E_ind_coeffs = E_ind_coeffs + E_imp

             E_df_ind = E_ind_coeffs[1]
             
        E_df_total = E_df_ind + E_df_noind
        
        d_m_ind_dt = self.E_df_to_d_m_ind_dt * E_df_total
        return d_m_ind_dt



    def get_potential_to_JS_operator(self, potential_type: str) -> "LinearMap":
        """Get spectral (VSH) pre-resistivity operator for given potential type.
        
        This operator maps magnetic scalars to the JS-like vector coefficients.
        The resistivity operator (eta) is applied afterward to obtain E.
        """
        L = self.solution_basis.index_length

        if potential_type in ("m_imp", "psi"):
             # Poloidal part from toroidal magnetic scalar source.
            p_op = (1.0 / mu0) * np.eye(L)
            # PFAC coupling contributes to toroidal component of JS-like vector.
            if potential_type == "m_imp":
                t_op = self._apply_imposed_toroidal_poloidal_lock(self.T_to_Ve)
            else:
                t_op = self._apply_dynamic_toroidal_poloidal_lock(self.T_to_Ve)
            return as_linear_map(np.vstack([p_op, t_op]))

        elif potential_type == "m_ind":
            # E_t = -1/mu0 * Scaling(m_ind) * Y^T
            scaling = self.solution_basis.get_potential_scaling_operator()
            t_mat = (-1.0 / mu0) * to_dense(scaling)

            if self._pfac.RM is not None and self._pfac.magnetospheric_poloidal_lock:
                ops = self._rm_coupling_solution_operators
                if ops is not None:
                    rm_feedback_op = np.asarray(ops.feedback)
                    t_mat = t_mat @ (np.eye(L) + rm_feedback_op)

            return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))
            
        elif potential_type == "Br":
            # Br path is purely toroidal, represented in solution coefficient space.
            ops = self._rm_coupling_solution_operators
            if ops is None:
                raise ValueError("Br pathway requires RM coupling operators when RM is configured.")

            rm_to_ri = np.asarray(ops.rm_to_ri)
            roundtrip_inv = np.asarray(ops.roundtrip_inv)
            if self._pfac.magnetospheric_poloidal_lock:
                br_factor_op = -(rm_to_ri @ roundtrip_inv)
            else:
                br_factor_op = -rm_to_ri

            m_ind_to_Br = np.asarray(to_dense(self.m_ind_to_Br))
            rcond = max(float(np.finfo(float).eps * max(m_ind_to_Br.shape)), 1e-15)
            m_ind_to_Br_inv = np.linalg.pinv(m_ind_to_Br, rcond=rcond)

            scaling = np.asarray(to_dense(self.solution_basis.get_potential_scaling_operator()))
            t_mat = (-1.0 / mu0) * (scaling @ br_factor_op @ m_ind_to_Br_inv)
            return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))

        raise ValueError(f"Unknown potential_type: {potential_type}")

    def steady_state_m_ind(
        self,
        E_coeffs_noind: np.ndarray,
        induction_matrix: Any,
        solver: str = "lsmr",
    ) -> np.ndarray:
        """Calculate the steady-state induced potential.

        Uses least-squares (lstsq or iterative solver) to handle
        ill-conditioned induction matrices. This ensures the minimum-norm
        solution is returned, avoiding numerical instability from
        near-null-space components that can differ between backends.

        Parameters
        ----------
        E_coeffs_noind : np.ndarray
            Non-inductive E-field coefficients with shape (2, n_coeffs) or (2*n_coeffs,).
        induction_matrix : np.ndarray or LinearMap
            Induction matrix mapping m_ind -> E_df. Can be a dense array
            or a LinearMap for matrix-free operation.
        solver : str, optional
            Solver to use for matrix-free case: "lsmr" or "cgls". Default "lsmr".
            For dense matrices, lstsq is always used regardless of this parameter.

        Returns
        -------
        np.ndarray
            Steady-state induced potential coefficients.
        """
        # op_A * m_ss + const = 0 -> op_A * m_ss = -const
        # Extract Toroidal Potential consistently using basis method
        vec_b = -asarray(self.solution_basis.get_toroidal_potential_coeffs(E_coeffs_noind))

        # Check if induction_matrix supports matrix-free operation
        if hasattr(induction_matrix, "matvec"):
            # Matrix-free path: Use iterative LeastSquaresSolver
            from pynamit.math.least_squares_problem import LeastSquaresProblem
            from pynamit.math.least_squares_solver import LeastSquaresSolver

            n = self.solution_basis.index_length
            induction_op = as_linear_map(induction_matrix)

            problem = LeastSquaresProblem(
                A=[induction_op],
                solution_shape=(n,),
                data_shapes=[(n,)],
            )

            # Use LSMR or CG for the steady-state inversion
            # Increase maxiter for ill-conditioned systems
            ls_solver = LeastSquaresSolver(solver=solver, tolerance=1e-10)
            return asarray(_timed_solve(
                "poloidal.steady_state_m_ind",
                ls_solver,
                problem,
                [vec_b],
                maxiter=5000,
            ))

        # Dense path: Use lstsq for numerical stability
        L = asarray(induction_matrix)
        # rcond=1e-13 filters out singular values below this relative threshold
        result = xp.linalg.lstsq(L, vec_b, rcond=1e-13)
        return result[0]
