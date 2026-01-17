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
    ) -> "LeastSquaresProblem":
        """Build the least-squares problem for m_imp.

        The problem structure is:
            minimize || A @ m_imp - b ||^2

        Where A consists of:
        1. jr constraint: jr_map @ (m_imp_to_jr @ m_imp) = jr_data
        2. E-field mapping constraint (if connect_hemispheres)
        3. Tikhonov regularization (if lambda > 0)

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

        # 3. Tikhonov regularization
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
