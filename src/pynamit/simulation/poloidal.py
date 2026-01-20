"""Poloidal System Matrices Module.

This module implements the `PoloidalSystemMatrices` class, which is responsible
for assembling the least-squares system matrices required for the poloidal
induction solver.

Unlike ToroidalSystemMatrices which solves a direct linear system L * x = K,
poloidal induction uses a constrained least-squares formulation.
"""

from __future__ import annotations
import logging
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from functools import cached_property

from pynamit.utils import to_numpy, asarray, xp, tensor_pinv
from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.math.constants import mu0
from pynamit.simulation.geometry_utils import to_dense
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from pynamit.utils import use_jax


if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.field import Field
    from pynamit.simulation.pfac import PFACIntegrator
    from pynamit.math.least_squares_problem import LeastSquaresProblem

logger = logging.getLogger(__name__)


class PoloidalSystemMatrices:
    """Assembles system matrices for Poloidal Induction.

    Unlike ToroidalSystemMatrices which solves L*x=K directly,
    poloidal induction uses a constrained least-squares formulation.

    The poloidal potential m_imp relates to:
    - Radial current: jr = (RI/mu0) * Laplacian(m_imp)
    - E-field: E = -grad(m_imp)/mu0 + Tor(Ve) via PFAC coupling

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

    # -------------------------------------------------------------------------
    # Core Operators (Laplacian-based)
    # -------------------------------------------------------------------------

    @cached_property
    def m_imp_to_jr(self) -> np.ndarray:
        """Operator mapping m_imp to radial current jr.

        Physics: jr = (RI/mu0) * Laplacian(m_imp)

        Returns
        -------
        np.ndarray
            Operator (matrix or diagonal) mapping potential to current.
        """
        return (self.RI / mu0) * self.solution_basis.get_laplacian_operator(self.RI)

    @cached_property
    def m_ind_to_Br(self) -> np.ndarray:
        """Operator mapping induced potential m_ind to radial field Br.

        Physics: Br = -(RI^2) * Laplacian(m_ind)

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
    # E-field Operators
    # -------------------------------------------------------------------------

    @cached_property
    def m_imp_to_E_coeffs_poloidal(self) -> np.ndarray:
        """Operator mapping m_imp to poloidal E-field coefficients.

        Physics: E_p = -grad(m_imp)/mu0

        The poloidal part of E from imposed potential is simply
        a scaled identity in coefficient space.

        Returns
        -------
        np.ndarray
            Shape (L, L) operator where L = solution_basis.index_length
        """
        L = self.solution_basis.index_length
        return (1.0 / mu0) * np.eye(L)

    @cached_property
    def m_imp_to_E_coeffs_toroidal(self) -> np.ndarray:
        """Operator mapping m_imp to toroidal E-field coefficients via PFAC.

        Physics: E_t = Tor(Ve) where Ve is computed from T_to_Ve mapping.

        Returns
        -------
        np.ndarray
            Shape (L, L) operator mapping m_imp to toroidal E component.
        """
        return self.T_to_Ve

    @cached_property
    def m_imp_to_E_coeffs(self) -> np.ndarray:
        """Combined operator mapping m_imp to full E-field VSH coefficients.

        Returns stacked [poloidal; toroidal] operator.

        Returns
        -------
        np.ndarray
            Shape (2*L, L) operator mapping m_imp to [E_p; E_t] coefficients.
        """
        p_op = self.m_imp_to_E_coeffs_poloidal
        t_op = self.m_imp_to_E_coeffs_toroidal
        return np.vstack([p_op, t_op])

    # -------------------------------------------------------------------------
    # Sheet Current Operators
    # -------------------------------------------------------------------------

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
        JS_coupling = np.tensordot(self.G_Ve_to_JS, self.T_to_Ve, axes=([2], [0]))
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
        T_to_Ve_da = self._pfac.compute_T_to_Ve(self.G_Ve_to_JS_sh, self.grid)
        return T_to_Ve_da.values

    @cached_property
    def G_Ve_to_JS_sh(self) -> np.ndarray:
        """Spectral (SH) version of Ve-to-JS operator for PFAC integration.

        Returns
        -------
        np.ndarray
            Shape (2, N_grid, L_sh) operator using spectral basis.
        """
        scaling_op = self.basis.get_potential_scaling_operator()
        curl_op = as_linear_map(self.basis.get_curl_matrix(self.grid))

        G_lin = (-1.0 / mu0) * (curl_op @ scaling_op)
        return to_dense(G_lin).reshape(2, -1, self.basis.index_length)

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

        # 1. Radial current constraint: jr_map @ m_imp_to_jr @ m_imp = jr_data
        op_apex = as_linear_map(jr_map_operator)
        op_m_to_jr = as_linear_map(self.m_imp_to_jr)
        op_jr = op_apex @ op_m_to_jr

        operators.append(op_jr)
        data_shapes.append((op_jr.shape[0],))

        # 2. E-field mapping constraint (interhemispheric)
        if connect_hemispheres and E_constraint_operator is not None:
            op_E = E_constraint_operator.with_scaling(ih_constraint_scaling)
            operators.append(op_E)
            data_shapes.append((op_E.shape[0],))

        # 3. Pinning Constraint
        if use_pinning:
            # m_imp[0] = 0.
            # Create a 1-row operator: [1, 0, 0, ...]
            # We can use a sparse matrix for efficiency or a DenseLinearMap
            # For a single row, dense is fine, or sparse.
            n = self.solution_basis.index_length
            row = np.zeros((1, n))
            row[0, 0] = 1.0 # Pin first element
            # Scale constraint to be strictly enforced (high weight)
            # Though standard formulation effectively weights it by 1 unless scaled.
            # Pinning should be "exact", so let's weight it high? 
            # Or just rely on it being the only definition for this mode.
            # Given the null space, any non-zero weight works.
            # Let's use a modest scaling to avoid bad conditioning in the other direction?
            # Actually, standard weight is usually fine if row is normalized.
            op_pin = as_linear_map(row)
            operators.append(op_pin)
            data_shapes.append((1,))

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

        # Add magnetospheric coupling if RM is defined
        if self._pfac.RM is not None:
            br_shift_sh, vi_shift_sh, den = self._pfac.get_coupling_factors()
            G_coupling_sh = self.G_Ve_to_JS_sh * (br_shift_sh * vi_shift_sh / den)

            # Handle basis transformation if needed
            if self.solution_basis is not self.basis:
                # Build adapter for hybrid basis
                E_sh = to_dense(self.basis.get_evaluation_matrix(self.grid))
                E_sol = to_dense(self.solution_basis.get_evaluation_matrix(self.grid))
                input_adapter = tensor_pinv(E_sol, rtol=1e-12) @ E_sh
                G = G + np.tensordot(G_coupling_sh, input_adapter, axes=([2], [0]))
            else:
                G = G + G_coupling_sh

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
        m_ind_to_E_operator: Optional[Any] = None,
        m_imp_to_E_operator: Optional[Any] = None,
    ) -> np.ndarray:
        """Construct the dense matrix for the induction operator (m_ind -> E_df).

        This moves logic previously in State._build_m_ind_to_E_df_matrix.
        """
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.solution_basis.index_length
        
        # 1. Get Direct Contribution (m_ind -> E_ind)
        # Use injected operator (legacy/grid) or fallback to internal exact spectral logic
        if m_ind_to_E_operator is not None:
             m_ind_to_E = m_ind_to_E_operator
        else:
             m_ind_to_E = self.get_potential_to_E_operator("m_ind")
        
        if m_ind_to_E is None:
             # Should not happen if physics is active
             return xp.zeros((n, n))

        E_direct_dense = asarray(m_ind_to_E.to_dense()).reshape(2, n, n)
        
        # 2. Compute Imposed Feedback (Constraint Response)
        # We need to construct the RHS for each "scenario" (each basis function being excited).
        # The problem.num_data_terms includes: [jr_constraint, E_mapping, pinning_constraint(optional)]
        rhs_entries = [None] * problem.num_data_terms if problem.num_data_terms > 0 else []

        if connect_hemispheres and E_map_constraint_operator is not None:
            # Op is ConstraintOperator (Rank agnostic)
            E_map_op = E_map_constraint_operator
            
            if hasattr(E_map_op, "apply"):
                 term = E_map_op.apply(E_direct_dense)
            else:
                 E_map_op = asarray(E_map_op)
                 if E_map_op.ndim == 4:
                      term = -xp.tensordot(E_map_op, E_direct_dense, axes=([2, 3], [0, 1]))
                 else:
                      term = -xp.tensordot(E_map_op, E_direct_dense, axes=([2], [0]))
            
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
        if m_imp_to_E_operator is not None:
             m_imp_to_E = as_linear_map(m_imp_to_E_operator)
        else:
             m_imp_to_E = as_linear_map(self.m_imp_to_E_coeffs)
        
        m_imp_flat = asarray(m_imp_block)
        E_imp_flat = m_imp_to_E.matmat(m_imp_flat)
        E_imp_block = asarray(E_imp_flat).reshape(2, n, n)

        total_E = E_direct_dense + E_imp_block
        logger.info("Dense induction operator built.")
        
        # Return E_df part (Toroidal component, index 1)
        return asarray(total_E[1])

    def solve_for_m_imp(
         self,
         E_direct_coeffs: np.ndarray,
         problem: "LeastSquaresProblem",
         solver: Any,
         preconditioner: Optional[Any] = None,
         E_map_constraint_operator: Optional[Any] = None,
         ih_constraint_scaling: float = 1.0,
         connect_hemispheres: bool = True,
         m_imp_to_E_operator: Optional[Any] = None,
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
                 E_map_op = asarray(E_map_op)
                 if E_map_op.ndim == 4:
                      b_E = -xp.tensordot(E_map_op, E_direct_input, axes=([2, 3], [0, 1]))
                 else:
                      b_E = -xp.tensordot(E_map_op, E_direct_input, axes=([2], [0]))
            
            if len(rhs_entries) > 1:
                rhs_entries[1] = ih_constraint_scaling * xp.reshape(b_E, (-1,))
                
         solution = solver.solve(problem=problem, rhs=rhs_entries, preconditioner=preconditioner)
         if solution is None:
             m_imp = xp.zeros(self.solution_basis.index_length)
         else:
             m_imp = asarray(solution)
             
         # Compute E_imp
         if m_imp_to_E_operator is not None:
             m_imp_to_E = as_linear_map(m_imp_to_E_operator)
         else:
             m_imp_to_E = as_linear_map(self.m_imp_to_E_coeffs)
             
         E_imp = m_imp_to_E.matvec(m_imp).reshape(2, -1)
         return m_imp, E_imp

    def compute_rates(
        self,
        m_ind: np.ndarray,
        t: float,
        E_coeffs_noind: np.ndarray,
        induction_matrix: Optional[np.ndarray] = None,
        m_ind_to_E_operator: Optional[Any] = None,
        # Solvers for implicit Feedback
        problem: Optional[Any] = None,
        solver: Optional[Any] = None,
        preconditioner: Optional[Any] = None,
        E_map_constraint_operator: Optional[Any] = None,
        ih_constraint_scaling: float = 1.0,
        connect_hemispheres: bool = True,
        m_imp_to_E_operator: Optional[Any] = None,
    ) -> np.ndarray:
        """Calculate d(m_ind)/dt rates."""
        # E_df_total = E_df_ind + E_df_noind
        # If we have the dense induction matrix (m_ind -> E_df), use it directly.
        # Otherwise, compute E_ind via operator.
        
        backend_m_ind = asarray(m_ind)
        E_df_noind = asarray(E_coeffs_noind[1]) # Index 1 is Toroidal/DivFree
        
        if induction_matrix is not None:
             # Matrix ALREADY includes the feedback (m_imp) response!
             E_df_ind = asarray(induction_matrix) @ backend_m_ind
        else:
             # Explicit construction: Must manually solve for m_imp feedback
             # 1. Direct E from m_ind
             if m_ind_to_E_operator is not None:
                  m_ind_to_E = m_ind_to_E_operator
             else:
                  m_ind_to_E = self.get_potential_to_E_operator("m_ind")
             
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



    def get_potential_to_E_operator(self, potential_type: str) -> "LinearMap":
        """Get spectral (VSH) E-field operator for given potential type.
        
        Copied/Moved from Geometry._get_E_operator_spectral to centralize logic.
        """
        L = self.solution_basis.index_length

        if potential_type == "m_imp":
             # Poloidal part: E_p = (1/mu0) * m_imp
            p_op = (1.0 / mu0) * np.eye(L)
            t_op = self.T_to_Ve
            return as_linear_map(np.vstack([p_op, t_op]))

        elif potential_type == "m_ind":
            # E_t = -1/mu0 * Scaling(m_ind) * Y^T
            scaling = self.solution_basis.get_potential_scaling_operator()
            t_mat = (-1.0 / mu0) * to_dense(scaling)

            if self._pfac.RM is not None:
                br, vi, den = self._pfac.get_coupling_factors()
                coupling = br * vi / den
                t_mat = t_mat * (1.0 + coupling)

            return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))
            
        elif potential_type == "Br":
             # Br path is purely toroidal
            br_shift, vi_shift, den = self._pfac.get_coupling_factors()
            L_op = np.diag(to_dense(self.basis.get_laplacian_operator(self.RI)))
            m_ind_to_Br = -(self.RI**2) * L_op

            scaling = self.basis.get_potential_scaling_operator()
            t_mat = (-1.0 / mu0) * to_dense(scaling) * (-br_shift / den / m_ind_to_Br)[:, None]
            return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))

        raise ValueError(f"Unknown potential_type: {potential_type}")

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray, induction_matrix: np.ndarray) -> np.ndarray:
        """Calculate the steady-state induced potential.

        Uses least-squares (lstsq) instead of direct solve to handle
        ill-conditioned induction matrices. This ensures the minimum-norm
        solution is returned, avoiding numerical instability from
        near-null-space components that can differ between backends.
        """
        # op_A * m_ss + const = 0 -> op_A * m_ss = -const
        vec_b = -asarray(E_coeffs_noind[1])
        L = asarray(induction_matrix)
        # Use lstsq for numerical stability with ill-conditioned matrices
        # rcond=1e-13 filters out singular values below this relative threshold
        result = xp.linalg.lstsq(L, vec_b, rcond=1e-13)
        return result[0]

