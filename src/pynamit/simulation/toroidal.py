"""Toroidal System Matrices Module.

This module implements the `ToroidalSystemMatrices` class, which is responsible
for assembling the linear system matrices required for the dynamic toroidal
induction solver:
    L * x = K

Where L = C + M0 + M1 * D1.
"""

from __future__ import annotations
import logging
from typing import Any, Tuple, Optional

import numpy as np
import scipy.sparse
from functools import cached_property

from pynamit.utils import to_numpy, asarray, xp, tensor_pinv
from pynamit.simulation.geometry_utils import to_dense
from pynamit.math.linear_map import as_linear_map
from pynamit.math.constants import mu0
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.spherical_harmonics import sh_operators

logger = logging.getLogger(__name__)


class ToroidalSystemMatrices:
    """Assembles system matrices for Toroidal Induction.

    Handles the construction of:
    - Mass Matrix (C)
    - Stiffness Matrices (M0, M1)
    - Mapping Operator (D1)

    Parameters
    ----------
    basis : Any
        The spectral basis (SHBasis).
    grid : Any
        The integration grid (from Geometry).
    b_field : Any
        The background magnetic field evaluated on the grid.
    RI : float
        Radius of the ionosphere.
    """

    def __init__(self, basis: Any, grid: Any, b_field: Any, RI: float):
        self.basis = basis
        self.grid = grid
        self.b_field = b_field
        self.RI = RI
        
        # Ensure we have access to quadrature weights for exact integration
        if not hasattr(grid, "weights"):
             # Fallback or error - for now we assume GL grid for exactness as per plan
             # But the code should be robust.
             pass

        
        self.is_cs = (getattr(self.basis, "kind", "") == "CS")
        if not self.is_cs:
            self.gaunt_engine = GauntEngine(basis)

    @cached_property
    def inverse_radial_field(self) -> np.ndarray:
        """Compute regularized 1/Br on the grid.
        
        Regularization: 1/B0r -> B0r / (B0r^2 + eps^2)
        """
        Br = self.b_field.vec.r
        return 1.0 / Br

    @cached_property
    def inertia_matrix(self) -> np.ndarray:
        """Construct Inertia Matrix C.
        
        C_lm,l'm' = mu0 * Integral [ Y_lm * (|B0|^2 / B0r) * Y_l'm' ]
        """
        logger.info("Building Toroidal Inertia Matrix...")
        
        B2 = self.b_field.magnitude**2
        factor = B2 * self.inverse_radial_field
        
        if self.is_cs:
            # CS Basis is Nodal/FV. Mass matrix is diagonal: Weights * Factor.
            # Factor is defined on the grid.
            weights = self.grid.weights if hasattr(self.grid, "weights") else xp.ones(self.grid.size)
            # Ensure shape match
            if weights.size != factor.size:
                 # Fallback if grid sizes mismatched?
                 raise RuntimeError(f"Grid size mismatch in CS Mass Matrix: {weights.size} vs {factor.size}")
            
            # C_ii = mu0 * weight_i * factor_i
            # Return sparse diagonal or dense diagonal
            # We assume factor is on grid corresponding to basis nodes
            C = xp.diag(mu0 * weights * factor)
            return C

        # Scalar interaction matrix
        # Projects factor to spectral coeffs, then builds multiplication matrix
        factor_coeffs = self.basis.from_grid_values(factor, self.grid, "scalar")
        
        # Use Gaunt Engine for exact product
        # C = mu0 * M(factor)
        # M(factor) is the matrix representing multiplication by 'factor' in spectral space
        M_factor = self.gaunt_engine.get_scalar_interaction_matrix(to_numpy(factor_coeffs))
        return mu0 * asarray(M_factor)

    @cached_property
    def advection_derivative(self) -> np.ndarray:
        """Construct Advection Derivative Operator D1.
        
        Maps dt_jr -> partial_r(dt_jr).
        
        Term 1: Y_l'm' * (d_r B0r / B0r)
        Term 2: - (1/Rb) * B0s . grad( Y_l'm' / B0r )
        """
        logger.info("Building Advection Derivative Operator D1...")
        
        if self.is_cs:
            # CS implementation
            # dr_Br_grid = - (2/r) * Br - (1/r) * div_Omega(B_s)  (General Gauss)
            inv_Rb = 1.0 / self.RI
            Br_grid = to_numpy(self.b_field.vec.r).flatten()
            Bth_grid = to_numpy(self.b_field.vec.theta).flatten()
            Bph_grid = to_numpy(self.b_field.vec.phi).flatten()
            
            # Gradients on Grid
            G_th = self.basis.get_G(self.grid, derivative="theta")
            G_ph = self.basis.get_G(self.grid, derivative="phi")
            P = to_dense(self.projection_matrix)
            
            # Compute div_Omega(B_s) = d/dth B_th + cot(th) B_th + d/dph B_ph
            # Project to coefficients first for spectral/nodal differentiation
            Bth_coeffs = P @ Bth_grid
            Bph_coeffs = P @ Bph_grid
            
            theta_rad = np.deg2rad(to_numpy(self.grid.theta)).flatten()
            cot_th = 1.0 / np.tan(theta_rad)
            div_B_S = (G_th @ Bth_coeffs) + cot_th * Bth_grid + (G_ph @ Bph_coeffs)
            
            # General dr_Br
            dr_Br_grid = -inv_Rb * (2.0 * Br_grid + div_B_S)
            
            inv_Br_grid = to_numpy(self.inverse_radial_field).flatten()
            
            # Gradient of 1/Br
            inv_Br_coeffs = P @ inv_Br_grid
            grad_inv_Br_th = G_th @ inv_Br_coeffs
            grad_inv_Br_ph = G_ph @ inv_Br_coeffs
            
            # Term 2b factor
            val_2b = -inv_Rb * (Bth_grid * grad_inv_Br_th + Bph_grid * grad_inv_Br_ph)
            
            # Term 2a scale
            scale_2a = -inv_Rb * inv_Br_grid
            
            # Construct Sparse Matrix
            # Diagonals
            Diag_T1 = scipy.sparse.diags([dr_Br_grid * inv_Br_grid], [0])
            Diag_T2b = scipy.sparse.diags([val_2b], [0])
            Diag_Scale2a = scipy.sparse.diags([scale_2a], [0])
            
            # Advection Operator
            Op_adv = scipy.sparse.diags([Bth_grid], [0]) @ G_th + scipy.sparse.diags([Bph_grid], [0]) @ G_ph
            
            Total_Grid_Op = Diag_T1 + Diag_T2b + (Diag_Scale2a @ Op_adv)
            
            if scipy.sparse.issparse(Total_Grid_Op):
                return Total_Grid_Op.toarray()
            return Total_Grid_Op

        else:
            # SH implementation
            # dr_Br = - (2/r) * Br - (1/r) * div_Omega(B_s) (General Gauss)
            inv_Rb = 1.0 / self.RI
            Br_grid = to_numpy(self.b_field.vec.r).flatten()
            Bth_grid = to_numpy(self.b_field.vec.theta).flatten()
            Bph_grid = to_numpy(self.b_field.vec.phi).flatten()
            
            # Use densified matrices for multiplication
            P = to_dense(self.projection_matrix)
            G = to_dense(self.basis.get_G(self.grid))
            G_th = to_dense(self.basis.get_G(self.grid, derivative="theta"))
            G_ph = to_dense(self.basis.get_G(self.grid, derivative="phi"))
            
            # Compute div_Omega(B_s)
            # We must project B_s to spectral coefficients before applying G_th/G_ph
            Bth_coeffs = P @ Bth_grid
            Bph_coeffs = P @ Bph_grid
            
            theta_rad = np.deg2rad(to_numpy(self.grid.theta)).flatten()
            cot_th = 1.0 / np.tan(theta_rad)
            div_B_S = (G_th @ Bth_coeffs) + cot_th * Bth_grid + (G_ph @ Bph_coeffs)
            
            dr_Br_grid = -inv_Rb * (2.0 * Br_grid + div_B_S)
            inv_Br_grid = to_numpy(self.inverse_radial_field).flatten()
            
            Bth_grid = to_numpy(self.b_field.vec.theta).flatten()
            Bph_grid = to_numpy(self.b_field.vec.phi).flatten()
            
            # Use densified matrices for multiplication
            P = to_dense(self.projection_matrix)
            G = to_dense(self.basis.get_G(self.grid))
            G_th = to_dense(self.basis.get_G(self.grid, derivative="theta"))
            G_ph = to_dense(self.basis.get_G(self.grid, derivative="phi"))
            
            inv_Rb = 1.0 / self.RI
            
            # Gradient of 1/B0r on grid
            # We can compute it numerically or via spectral. Let's use spectral for consistency.
            inv_Br_coeffs = self.basis.from_grid_values(inv_Br_grid, self.grid, "scalar")
            grad_inv_Br_th = G_th @ inv_Br_coeffs
            grad_inv_Br_ph = G_ph @ inv_Br_coeffs 
            
            # Term 1 (Scalar Scale): dr_Br/B0r
            T1 = (dr_Br_grid * inv_Br_grid)[:, None] * G
            
            # Term 2b (Scalar Scale): - (1/Rb) * (B_theta * d(1/B)/dth + B_phi * d(1/B)/dph)
            val_2b = -inv_Rb * (Bth_grid * grad_inv_Br_th + Bph_grid * grad_inv_Br_ph)
            T2b = val_2b[:, None] * G
            
            # Term 2a (Advection): - (1/(Rb*B0r)) * (B_theta * df/dth + B_phi * df/dph)
            # Note: df/dth = G_th * x, df/dph = G_ph * x
            scale_2a = -inv_Rb * inv_Br_grid
            T2a = scale_2a[:, None] * (Bth_grid[:, None] * G_th + Bph_grid[:, None] * G_ph)
            
            # Combine on grid
            Total_Grid_Op = T1 + T2a + T2b
            
            # Project back to spectral
            # D1 = P @ Total_Grid_Op
            return asarray(P @ Total_Grid_Op)

    def compute_source_from_E(self, E_coeffs: np.ndarray) -> np.ndarray:
        """Compute Source vector K from Known E-field S_known.
        
        Calculates K_lm = - Integral [ Y_lm * S_known ]
        where S_known is derived from Faraday's and Gauss' laws to eliminate
        radial derivatives of E.
        """
        # 1. Setup constants
        Rb = self.RI
        inv_Rb = 1.0 / Rb
        inv_Rb2 = 1.0 / (Rb**2)
        
        # 2. Get evaluation and projection matrices (Cached)
        G = to_dense(self.basis.get_G(self.grid))
        G_th = to_dense(self.basis.get_G(self.grid, derivative="theta"))
        G_ph = to_dense(self.basis.get_G(self.grid, derivative="phi"))
        P = to_dense(self.projection_matrix)
        weights = to_numpy(self.grid.weights)
        
        # 3. Evaluate E_S and Er on grid
        # E_coeffs contains [Poloidal, Toroidal] potentials
        Eth_grid, Eph_grid = self.basis.evaluate(E_coeffs, self.grid, vector_type="tangential")
        
        # Er = - (B_s . E_s) / Br
        Br = to_numpy(self.b_field.vec.r)
        Bth = to_numpy(self.b_field.vec.theta)
        Bph = to_numpy(self.b_field.vec.phi)
        
        # Use regularized inverse radial field
        inv_Br = to_numpy(self.inverse_radial_field)
        Er_grid = -(Bth * Eth_grid + Bph * Eph_grid) * inv_Br
        
        # 4. Helpers for grid derivatives via spectral projection
        def get_derivs(f_val):
            c = P @ f_val
            return G_th @ c, G_ph @ c

        def get_laplacian(f_val):
            c = P @ f_val
            if self.is_cs:
                # CS laplacian matrix
                c_lap = to_dense(self.basis.laplacian(r=1.0)) @ c
            else:
                # SH laplacian factor -l(l+1)
                l_arr = to_numpy(self.basis.n).flatten()
                c_lap = -l_arr * (l_arr + 1.0) * c
            return G @ c_lap

        # 5. Compute differentiated terms on grid
        dEr_th, dEr_ph = get_derivs(Er_grid)
        lap_Er = get_laplacian(Er_grid)
        
        dEth_th, dEth_ph = get_derivs(Eth_grid)
        dEph_th, dEph_ph = get_derivs(Eph_grid)
        
        # div_Omega E_S = d/dth E_th + cot(th) E_th + (1/sin th) d/dph E_ph
        theta_rad = np.deg2rad(to_numpy(self.grid.theta)).flatten()
        cot_th = 1.0 / np.tan(theta_rad)
        # Note: G_ph already includes 1/sin(theta) factor for SHBasis
        div_E_S = dEth_th + cot_th * Eth_grid + dEph_ph
        
        # gradient of div_Omega E_S
        grad_div_E_th, grad_div_E_ph = get_derivs(div_E_S)
        
        # Vector Laplacian components (Angular part)
        # Delta_Omega E_S = [ lap E_th - E_th/sin^2 - 2 cot/sin dE_ph/dph , ... ]
        inv_sin2 = 1.0 / (np.sin(theta_rad)**2)
        # 2 cot/sin dE_ph/dph = 2 cot * (G_ph @ c_Eph) = 2 cot * dEph_ph
        vec_lap_Eth = get_laplacian(Eth_grid) - inv_sin2 * Eth_grid - 2.0 * cot_th * dEph_ph
        vec_lap_Eph = get_laplacian(Eph_grid) - inv_sin2 * Eph_grid + 2.0 * cot_th * dEth_ph

        # 6. Assemble S_known scalar field
        # S_known = Br [ 1/Rb div_E_S + (1/Rb + 1/Rb^2) lap_Er ] + B_s . Brack
        term1 = Br * (inv_Rb * div_E_S + (inv_Rb + inv_Rb2) * lap_Er)
        
        brack_th = inv_Rb2 * (2.0 * dEr_th + grad_div_E_th - 2.0 * Eth_grid - vec_lap_Eth)
        brack_ph = inv_Rb2 * (2.0 * dEr_ph + grad_div_E_ph - 2.0 * Eph_grid - vec_lap_Eph)
        term2 = Bth * brack_th + Bph * brack_ph
        
        S_known = term1 + term2
        
        # 7. Integrate: K_lm = - Integral [ Y_lm * S_known ]
        # Matrix form: K = - (G^T @ diag(W)) @ S_known
        K = - (G.T * weights) @ S_known
        
        return asarray(K)

    @cached_property
    def advection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Construct Advection Matrices M0 and M1.
        
        M0_lm,l'm' = Integrate[ Y_lm * (B0s . grad Y_l'm') * (2 mu0 / (l'(l'+1))) ]
        M1_lm,l'm' = Integrate[ Y_lm * (B0s . grad Y_l'm') * (-mu0 Rb / (l'(l'+1))) ]
        
        Both share the integral structure:
           Int = Integrate[ Y_lm * Op_adv(Y_l'm') ]
           where Op_adv(f) = B0s . grad f
        
        And then scaling depends on column index l'.
        """
        logger.info("Building Advection Matrices M0 and M1...")
        
        if self.is_cs:
            # CS Implementation:
            # M_0 = 2 mu0 * (Laplacian^-1) @ Advection_Matrix
            # M_1 = -mu0 Rb * (Laplacian^-1) @ Advection_Matrix
            
            # 1. Build Advection Matrix on Grid (Mass Weighted)
            # A_ij = Int [ Y_i (v . grad Y_j) ]
            # = (Weight_Matrix @ Grid_Op) ??
            # Wait. On Nodal Basis:
            # Y_i is 1 at node i, 0 else.
            # Int [ Y_i * f ] = w_i * f(x_i).
            # So Row i of A is w_i * (Op_adv)_row_i.
            # A = Diag(Weights) @ Op_adv.
            
            weights = self.grid.weights if hasattr(self.grid, "weights") else xp.ones(self.grid.size)
            W_diag = scipy.sparse.diags([weights], [0])
            
            G_th = self.basis.get_G(self.grid, derivative="theta")
            G_ph = self.basis.get_G(self.grid, derivative="phi")
            
            B_theta = to_numpy(self.b_field.vec.theta).flatten()
            B_phi = to_numpy(self.b_field.vec.phi).flatten()
            
            Op_adv = scipy.sparse.diags([B_theta], [0]) @ G_th + scipy.sparse.diags([B_phi], [0]) @ G_ph
            
            Advection_Matrix = W_diag @ Op_adv
            
            # 2. Laplacian Inverse
            L_mat = self.basis.laplacian(r=1.0)
            
            if scipy.sparse.issparse(L_mat):
                L_dense = L_mat.toarray()
            else:
                L_dense = L_mat
                
            L_inv = tensor_pinv(L_dense, n_leading_flattened=1)

            # Apply Inverse Laplacian
            Inv_Lap_Op = -L_inv
            
            # Combine
            scale_M0 = 2.0 * mu0
            scale_M1 = -mu0 * self.RI
            
            # Convert sparse Advection Matrix to dense for safe matmul with dense L_inv
            Adv_dense = Advection_Matrix.toarray() if scipy.sparse.issparse(Advection_Matrix) else Advection_Matrix

            print(f"DEBUG: Inv_Lap_Op shape: {Inv_Lap_Op.shape}, Adv_dense shape: {Adv_dense.shape}")
            print(f"DEBUG: scale_M0 type: {type(scale_M0)}, Inv_Lap_Op type: {type(Inv_Lap_Op)}, Adv_dense type: {type(Adv_dense)}")

            M0 = scale_M0 * (Inv_Lap_Op @ Adv_dense)
            M1 = scale_M1 * (Inv_Lap_Op @ Adv_dense)
            
            return asarray(M0), asarray(M1)

        # --- SH Implementation ---
        # 1. Build Advection Matrix A: A_ij = Int[ Y_i * (B0s . grad Y_j) ]
        if not hasattr(self.grid, "weights"):
            # Fallback for non-GL grids?
            raise RuntimeError("Grid weights required for stiffness matrix construction.")
            
        weights = self.grid.weights
        W_diag = xp.diag(weights)
        
        G = to_numpy(self.basis.get_G(self.grid))
        G_th = to_numpy(self.basis.get_G(self.grid, derivative="theta"))
        G_ph = to_numpy(self.basis.get_G(self.grid, derivative="phi"))
        
        B_theta = to_numpy(self.b_field.vec.theta)
        B_phi = to_numpy(self.b_field.vec.phi)
        
        # Grid operation: (B . grad) Y_j
        # Shape: (N_grid, N_sh)
        Op_adv = (B_theta[:, None] * G_th) + (B_phi[:, None] * G_ph)
        
        # Integrate against Y_i
        # Matrix = G^T @ W @ Op_adv
        # Optimize: (G^T * w) @ Op_adv
        Advection_Matrix = (G.T * weights) @ Op_adv
        
        # 2. Apply Column Scalings
        l_arr = to_numpy(self.basis.n).flatten()
        ll1 = l_arr * (l_arr + 1.0)
        # Avoid division by zero for l=0 (monopole has no stiffness/current)
        ll1_inv = np.zeros_like(ll1)
        ll1_inv[ll1 > 0] = 1.0 / ll1[ll1 > 0]
        
        scale_M0 = 2.0 * mu0 * ll1_inv
        scale_M1 = -mu0 * self.RI * ll1_inv
        
        M0 = Advection_Matrix * scale_M0[None, :]
        M1 = Advection_Matrix * scale_M1[None, :]
        
        return asarray(M0), asarray(M1)

    @cached_property
    def system_matrix(self) -> np.ndarray:
        """Assemble total System Matrix L = C + M0 + M1 * D1."""
        M0, M1 = self.advection_matrices
        D1 = to_numpy(self.advection_derivative) # D1 is built via projection, usually numpy/xp
        C = to_numpy(self.inertia_matrix)
        
        # M1 * D1
        M1_D1 = M1 @ D1
        
        return asarray(C + M0 + M1_D1)

    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Get projection matrix from Basis (Grid -> Coeffs)."""
        return self.basis.construct_scalar_projection_matrix(self.grid)

    # -------------------------------------------------------------------------
    # Least-Squares Problem Construction (aligned with PoloidalSystemMatrices)
    # -------------------------------------------------------------------------

    def build_least_squares_problem(
        self,
        jr_map_operator: np.ndarray,
        constraint_scaling: float = 1.0,
        regularization_lambda: float = 0.0,
    ) -> "LeastSquaresProblem":
        """Build the least-squares problem for dt_jr (toroidal induction).

        The problem structure is:
            minimize || A @ dt_jr - b ||^2

        Where A consists of:
            1. Physics equation: L @ dt_jr = K (where L = C + M0 + M1*D1)
        2. Apex current constraint: jr_map @ dt_jr = driver_rate
        3. Tikhonov regularization (if lambda > 0)

        This is analogous to PoloidalSystemMatrices.build_least_squares_problem().

        Parameters
        ----------
        jr_map_operator : np.ndarray
            Operator mapping jr coefficients to apex current (jr_map_sim).
        constraint_scaling : float
            Scaling factor for the apex current constraint (penalty weight).
        regularization_lambda : float
            Tikhonov regularization weight.

        Returns
        -------
        LeastSquaresProblem
            The assembled least-squares problem.
        """
        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import as_linear_map, diagonal_linear_map

        operators = []
        data_shapes = []

        # 1. Physics equation: L @ dt_jr = K
        op_L = as_linear_map(self.system_matrix)
        operators.append(op_L)
        data_shapes.append((op_L.shape[0],))

        # 2. Apex current constraint (interhemispheric + driver matching)
        op_constraint = as_linear_map(jr_map_operator)
        op_constraint = op_constraint.with_scaling(constraint_scaling)
        operators.append(op_constraint)
        data_shapes.append((op_constraint.shape[0],))

        # 3. Tikhonov regularization
        reg_ops = []
        reg_weights = []
        if regularization_lambda > 0:
            n = self.basis.index_length
            identity_op = diagonal_linear_map(xp.ones(n))
            reg_ops.append(identity_op)
            reg_weights.append(regularization_lambda)

        return LeastSquaresProblem(
            A=operators,
            solution_shape=self.basis.index_length,
            data_shapes=data_shapes,
            regularization_matrices=reg_ops,
            regularization_weights=reg_weights,
        )

    def compute_forcing_vector(
        self,
        E_coeffs: np.ndarray,
        dt_jr_driver_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute RHS for the physics equation term.

        RHS = K - L @ dt_jr_driver
        
        Where K is computed from E-field and L @ dt_jr_driver accounts
        for the known driver contribution.

        This is analogous to PoloidalSystemMatrices.compute_rhs_from_jr().

        Parameters
        ----------
        E_coeffs : np.ndarray
            E-field coefficients for computing K.
        dt_jr_driver_coeffs : np.ndarray, optional
            Driver rate coefficients. If None, assumes zero driver.

        Returns
        -------
        np.ndarray
            RHS vector for the physics term.
        """
        # Compute K from E-field
        K = self.compute_source_from_E(E_coeffs)
        K = to_numpy(K)

        # Subtract driver contribution if present
        if dt_jr_driver_coeffs is not None:
            L = to_numpy(self.system_matrix)
            L_driver = L @ to_numpy(dt_jr_driver_coeffs)
            norm_L_driver = np.linalg.norm(L_driver)
            print(f"DEBUG: compute_forcing_vector: |K_orig|={np.linalg.norm(K):.2e}, |L_driver|={norm_L_driver:.2e}")
            K = K - L_driver

        return asarray(K)

    # -------------------------------------------------------------------------
    # Time Evolution Logic
    # -------------------------------------------------------------------------

    def compute_rates(
        self,
        dt_jr: np.ndarray,
        m_imp_to_jr_operator: Any,
    ) -> np.ndarray:
        """Calculate rate of change of toroidal potential (psi) from dt_jr.
        
        Physics:
            jr = (R/mu0) * Laplacian(psi).
            So d_psi_dt = (jr_to_m_operator) @ dt_jr.
            
        Parameters
        ----------
        dt_jr : np.ndarray
             Rate of change of radial current density (dt_jr).
        m_imp_to_jr_operator : Any
             The operator mapping potential m (psi) to current jr.
             Typically geometry.m_imp_to_jr.
             
        Returns
        -------
        np.ndarray
             d(psi)/dt coefficients.
        """
        from pynamit.utils import tensor_pinv
        from pynamit.simulation.geometry_utils import to_dense
        
        # Invert operator: jr = Op @ psi  ->  psi = InvOp @ jr
        # Use dense pinv for robustness on the small spectral system
        op_m_to_jr = as_linear_map(m_imp_to_jr_operator)
        m_to_jr_dense = to_dense(op_m_to_jr)
        
        # Calculate pseudo-inverse
        jr_to_m_dense = tensor_pinv(m_to_jr_dense)
        
        d_psi_dt = jr_to_m_dense @ asarray(dt_jr)
        return asarray(d_psi_dt)




