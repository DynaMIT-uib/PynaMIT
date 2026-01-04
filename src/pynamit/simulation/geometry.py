"""Geometry module.

This module contains the Geometry class, which encapsulates the spatial
grid, basis evaluators, magnetic field properties, and interhemispheric
mappings.
"""

from __future__ import annotations
import logging
from typing import Optional, Any

import numpy as np
import xarray as xr
from functools import cached_property
import scipy.sparse

from pynamit.math.constants import mu0
from pynamit.primitives.grid import Grid
from pynamit.primitives.field import Field
from pynamit.utils import tensor_pinv
from pynamit.primitives.basis import Basis
from pynamit.math.linear_map import as_linear_map
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynamit.primitives.grid_basis import GridBasis
    from pynamit.math.gaunt import GauntEngine
    from pynamit.simulation.dynamics import SimulationMode


logger = logging.getLogger(__name__)


class Geometry:
    """Encapsulates the geometric setup for the ionospheric simulation.

    This class manages grids, basis and field evaluators, geometric
    factors derived from the main magnetic field, and interhemispheric
    mappings. It provides a clean interface for the main State class to
    access pre-computed geometric quantities.
    """

    def __init__(
        self,
        basis: Basis,
        grid_basis: "GridBasis",
        mainfield: Any,
        settings: Any,
        PFAC_matrix: Optional[xr.DataArray] = None,
        solution_basis: Optional[Any] = None,
    ) -> None:
        """Initialize the geometric context.
        
        Parameters
        ----------
        basis : SHBasis
            The spectral basis used for spherical harmonic operations.
        grid_basis : GridBasis
            The basis defining the spatial grid (e.g., CSBasis).
        mainfield : Mainfield
            The main magnetic field model.
        settings : Any
            Simulation settings.
        PFAC_matrix : xr.DataArray, optional
            Pre-computed PFAC matrix.
        solution_basis : Any, optional
            The basis used for the solution state variables.
        """
        self.basis = basis
        self.solution_basis = solution_basis if solution_basis is not None else basis
        self.mainfield = mainfield

        # Allow pre-computed PFAC matrix (must override cached_property if provided)
        if PFAC_matrix is not None:
            self.T_to_Ve = PFAC_matrix

        # Store relevant settings
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.FAC_integration_steps = settings.FAC_integration_steps

        # Initialize core geometric objects
        self._init_evaluators(grid_basis)
        
        self.input_adapter = None
        if self.solution_basis is not self.basis:
             try:
                 # Generic check for "different basis types implies adapter needed"
                 if getattr(self.solution_basis, "kind", "") != getattr(self.basis, "kind", ""):
                      logger.info("Basis mismatch detected: initializing hybrid adapter.")
                      G_dense = self.basis.get_evaluation_matrix(self.grid)

                      if scipy.sparse.issparse(G_dense):
                           G_dense = G_dense.toarray()
                      self.input_adapter = tensor_pinv(G_dense, n_leading_flattened=1)
             except Exception:
                  logger.warning("Failed to initialize basis adapter. Proceeding without one.", exc_info=True)

        self._init_constraint_mappings()

        # 1. Solution/Simulation basis operators (for the unknowns on the solver grid)
        self.m_imp_to_jr = (self.RI / mu0) * self.solution_basis.get_laplacian_operator(self.RI)
        self.m_ind_to_Br = -(self.RI**2) * self.solution_basis.get_laplacian_operator(self.RI)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI

        # 2. Spectral Induction operators (Required for T_to_Ve and coupling)
        scaling_op_sh = self.basis.get_potential_scaling_operator()
        curl_sh = as_linear_map(self.basis.get_curl_matrix(self.grid))
        
        # Scaling factor: -1/mu0 (Standard Pynamit Induction Factor)
        # We include the scaling op (2n + 1) which relates m to potential.
        G_lin_sh = (-1.0 / mu0) * (curl_sh @ scaling_op_sh)
        self.G_Ve_to_JS_sh = G_lin_sh.to_dense().reshape(2, -1, self.basis.index_length)

        # 3. Grid-native Induction operators (for local induction loop)
        self.G_Ve_to_JS = self.G_Ve_to_JS_sh
        if self.solution_basis.kind == "CS":
             try:
                  scaling_op_cs = self.solution_basis.get_potential_scaling_operator()
                  curl_cs = as_linear_map(self.solution_basis.get_curl_matrix(self.grid))
                  
                  # G = (1/RI) * Curl * (-RI/mu0 * S) 
                  G_lin_cs = (1.0 / self.RI) * (curl_cs @ ((-self.RI / mu0) * scaling_op_cs))
                  self.G_Ve_to_JS = G_lin_cs.to_dense().reshape(2, -1, self.solution_basis.index_length)
             except (NotImplementedError, AttributeError):
                  # Fallback to spectral
                  logger.warning("CS Basis does not support operator construction. Falling back to spectral.")
                  pass

    @cached_property
    def gaunt_engine(self) -> "GauntEngine":
        """Lazy-loaded GauntEngine for spectral interaction matrices."""
        from pynamit.math.gaunt import GauntEngine
        return GauntEngine(self.solution_basis)

    def _synthesize_to_gaunt(self, grid_data: np.ndarray) -> np.ndarray:
        """Synthesize grid data from simulation grid to GauntEngine grid."""
        from pynamit.utils import xp, asarray
        # grid_data: (..., N_grid)
        # 1. Project to extended basis
        G_scalar = self.basis_zero_added.get_evaluation_matrix(self.grid)
        if hasattr(G_scalar, "toarray"): G_scalar = G_scalar.toarray()
        
        if hasattr(self.grid_basis, "weights"):
            weights = self.grid_basis.weights
            GtW = G_scalar.T * weights
            GtWG = GtW @ G_scalar
            P_scalar = xp.linalg.solve(GtWG, GtW)
        else:
            from pynamit.utils import tensor_pinv
            P_scalar = tensor_pinv(asarray(G_scalar), n_leading_flattened=1)
            
        coeffs = xp.tensordot(P_scalar, grid_data, axes=([1], [-1]))
        
        # 2. Synthesize to Gaunt grid
        engine = self.gaunt_engine
        G_quad = self.basis_zero_added.get_evaluation_matrix(engine.quad_grid)
        if hasattr(G_quad, "toarray"): G_quad = G_quad.toarray()
        
        # (Q, L) @ (L, ...) -> (Q, ...)
        res = xp.tensordot(G_quad, coeffs, axes=([1], [0]))
        return xp.moveaxis(res, 0, -1)

    def get_E_vsh_operator(self, potential_type: str) -> "LinearMap":
        """Get operator mapping potential coefficients to VSH E-field coefficients."""
        from pynamit.math.linear_map import as_linear_map
        L = self.solution_basis.index_length
        
        if potential_type == "m_imp":
             # Poloidal part: E_p = -grad(m_imp)/mu0 -> p_coeffs = (1/mu0) * m_imp
             p_op = (1.0 / mu0) * np.eye(L)
             # Toroidal part: E_t = Tor(Ve_coeffs) -> t_coeffs = T_to_Ve @ m_imp
             t_op = self.T_to_Ve.values
             return as_linear_map(np.vstack([p_op, t_op]))
             
        elif potential_type == "m_ind":
             # E_t = -1/mu0 * Scaling(m_ind) * Y^T
             scaling = self.solution_basis.get_potential_scaling_operator()
             t_mat = (-1.0 / mu0) * scaling.to_dense()
             
             if self.RM is not None:
                 br = np.diag(self.basis.get_radial_shift_operator(self.RM, self.RI, kind="external").to_dense())
                 vi = np.diag(self.basis.get_radial_shift_operator(self.RI, self.RM, kind="internal").to_dense())
                 coupling = (br * vi / (1.0 - br * vi))
                 t_mat = t_mat * (1.0 + coupling)
                 
             return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))

        elif potential_type == "Br":
             # Br path is purely toroidal
             br_shift = np.diag(self.basis.get_radial_shift_operator(self.RM, self.RI, kind="external").to_dense())
             vi_shift = np.diag(self.basis.get_radial_shift_operator(self.RI, self.RM, kind="internal").to_dense())
             den = 1.0 - br_shift * vi_shift
             L_op = np.diag(self.basis.get_laplacian_operator(self.RI).to_dense())
             m_ind_to_Br = -(self.RI**2) * L_op
             
             scaling = self.basis.get_potential_scaling_operator()
             t_mat = (-1.0 / mu0) * scaling.to_dense() * (-br_shift / den / m_ind_to_Br)[:, None]
             return as_linear_map(np.vstack([np.zeros((L, L)), t_mat]))

        raise ValueError(f"Unknown potential_type: {potential_type}")

    def get_conductivity_operator(
        self, 
        mode: Any, 
        potential_type: str, 
        sigma_grid: np.ndarray
    ) -> "LinearMap":
        """Construct a unified conductivity operator (Potential -> JS_coeffs)."""
        from pynamit.simulation.dynamics import SimulationMode
        from pynamit.utils import to_numpy
        from pynamit.math.linear_map import as_linear_map
        
        if mode == SimulationMode.PURE_SPECTRAL:
            # Galerkin Path: Exact spectral interaction
            op_E_vsh = self.get_E_vsh_operator(potential_type)
            # Build interaction matrix (2L, 2L)
            sigma_quad = self._synthesize_to_gaunt(sigma_grid)
            M_vsh = self.gaunt_engine.get_vector_interaction_matrix(to_numpy(sigma_quad))
            return as_linear_map(M_vsh) @ op_E_vsh
        else:
            # Transform Path: P @ sigma @ G
            G_grid = getattr(self, f"G_{potential_type}_to_JS", None)
            if G_grid is None: return None
            
            from pynamit.simulation.state import _ResistanceOperator
            from pynamit.math.linear_map import LinearMap
            res_op = _ResistanceOperator(sigma_grid)
            op_M = LinearMap(
                shape=res_op.shape,
                dtype=res_op.dtype,
                _matvec=res_op.matvec,
                _rmatvec=res_op.rmatvec,
                _matmat=res_op.matmat,
                _rmatmat=res_op.rmatmat,
                _to_dense=res_op.to_dense,
                source=res_op
            )
            
            op_G = as_linear_map(G_grid.reshape(-1, G_grid.shape[-1]))
            op_P = as_linear_map(self.projection_matrix)
            return op_P @ op_M @ op_G


    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Projection matrix (Grid Vector -> Basis Coefficients).

        For Helmholtz decomposition of vector fields E = -grad(Φ) + r×grad(Ψ).

        For Gauss-Legendre grids with quadrature weights, uses exact analysis:
            A = G^T @ diag(weights)
        which gives machine-precision transforms for orthonormal SH.

        For other grids (e.g., Cubed-Sphere), uses pseudo-inverse.
        """
        # Check if grid_basis has quadrature weights for exact analysis
        if hasattr(self.grid_basis, "weights"):
            logger.info("Using exact GL quadrature for Helmholtz decomposition.")
            return self._build_exact_helmholtz_analysis()

        # Default: use pseudo-inverse based projection
        return self.solution_basis.construct_projection_matrix(self.grid)

    def _build_exact_helmholtz_analysis(self) -> np.ndarray:
        """Build exact Helmholtz analysis matrix using GL quadrature.

        Uses weighted least-squares with GL quadrature weights:
            A = (G^T W G)^{-1} G^T W

        where W = diag(weights) accounts for the quadrature measure.
        This is equivalent to pseudo-inverse but uses exact quadrature
        inner products, giving machine-precision for GL grids.

        Returns
        -------
        A : ndarray
            Analysis matrix of shape (2*N_sh, 2*N_grid).
        """
        import scipy.sparse

        # Get gradient operators from the spectral basis
        G_th = self.basis.get_G(self.grid, derivative="theta")
        G_ph = self.basis.get_G(self.grid, derivative="phi")

        # Convert to dense if sparse
        if scipy.sparse.issparse(G_th):
            G_th = G_th.toarray()
        if scipy.sparse.issparse(G_ph):
            G_ph = G_ph.toarray()

        n_grid = G_th.shape[0]
        n_sh = G_th.shape[1]

        # Build Helmholtz basis vectors:
        # -grad(Y) for poloidal potential Φ
        # r×grad(Y) for toroidal potential Ψ
        G_grad = np.array([G_th, G_ph])        # (2, N_grid, N_sh)
        G_rxgrad = np.array([-G_ph, G_th])     # (2, N_grid, N_sh)

        # G_helmholtz: (vec_comp, grid_pt, pot_type, sh_idx)
        G_helmholtz = np.stack([-G_grad, G_rxgrad], axis=2)

        # Flatten to 2D: (2*N_grid, 2*N_sh)
        G_flat = G_helmholtz.transpose(0, 1, 2, 3).reshape(2 * n_grid, 2 * n_sh)

        # Build weight matrix: W = diag([w, w]) for both vector components
        weights = self.grid_basis.weights  # (N_grid,)
        W_diag = np.tile(weights, 2)  # (2*N_grid,)

        # Weighted least-squares: A = (G^T W G)^{-1} G^T W
        # Using efficient formulation: solve (G^T W G) A^T = G^T W
        GtW = G_flat.T * W_diag  # (2*N_sh, 2*N_grid), broadcast multiply
        GtWG = GtW @ G_flat     # (2*N_sh, 2*N_sh)

        # Solve the normal equations
        # A = (GtWG)^{-1} @ GtW
        A = np.linalg.solve(GtWG, GtW)

        return A

        
    def _init_evaluators(self, grid_basis: "GridBasis") -> None:
        """Set up grid, basis evaluators, and field evaluators."""
        self.grid_basis = grid_basis
        self.grid = grid_basis.grid

        # Use polymorphic method to get zero-added basis (for Monopole support in SH)
        self.basis_zero_added = self.basis.get_extended_basis()

        self.b_field = self.mainfield.discretize(self.grid, self.RI)

        # Optional evaluators for the conjugate hemisphere
        self.cp_grid = self.cp_b_field = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_b_field = self.mainfield.discretize(self.cp_grid, self.RI)

    # --- Geometric Helper Methods ---
    # Methods to replicate prior logic using basis/field API directly.

    def _create_apex_operators(self, field: Field, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """Create current and field mapping operators for a given field/grid pair."""
        radial_to_apex, horizontal_to_apex = self._get_transformation_matrices(field)

        # jr_coeffs_to_j_apex
        jr_op = self.basis.get_scaled_matrix(grid, radial_to_apex)

        # E_coeffs_to_E_apex
        E_op = np.einsum(
            "ijk,jklm->iklm",
            horizontal_to_apex,
            self.basis.get_vector_basis_matrix(grid),
            optimize=True,
        )
        return jr_op, E_op

    def _init_constraint_mappings(self) -> None:
        """Initialize geometric operators related to constraints."""
        kind = self.mainfield.kind
        if kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - 6371e3) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)

        # Main Hemisphere Operators
        self.jr_map_spectral, E_coeffs_to_E_apex = self._create_apex_operators(
            self.b_field, self.grid
        )
        
        self.E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres:
            # Conjugate Hemisphere Operators
            jr_coeffs_to_j_apex_cp, E_coeffs_to_E_apex_cp = self._create_apex_operators(
                self.cp_b_field, self.cp_grid
            )

            # Apply correction in SPECTRAL domain
            self.jr_map_spectral[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]
            
            # Create E-field mapping difference operator for constraint
            self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                 (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask]
            )

            if self.input_adapter is not None:
                 # Pre-compose with analysis matrix (SH->Grid)
                 logger.info("Pre-composing hybrid constraint operator (SH->Grid)...")
                 self.E_coeffs_to_E_apex_ll_diff = np.tensordot(
                     self.E_coeffs_to_E_apex_ll_diff, 
                     self.input_adapter, 
                     axes=([-1], [0])
                 )
                 # Result: (2, Mask, 2, N_grid)


        if self.input_adapter is not None:
             # Chain analysis to get Simulation operator (Grid inputs) for LHS
             self.jr_map_sim = self.jr_map_spectral @ self.input_adapter
        else:
             self.jr_map_sim = self.jr_map_spectral
             

    def get_jr_operator(self, input_basis: Any = None) -> np.ndarray:
        """Get the operator mapping jr to J_apex suitable for the input basis.
        
        If input basis kind matches the Physics basis kind (e.g. SH), use the Physics/Spectral operator.
        Otherwise (e.g. input is Grid but simulation is hybrid), use the 
        Simulation operator (which includes adapter).
        """
        physics_kind = getattr(self.basis, "kind", None)
        input_kind = getattr(input_basis, "kind", None)
        
        # Default to physics operator if no input specified (theoretical)
        # Or if kinds match.
        if input_basis is None or input_kind == physics_kind:
             return self.jr_map_spectral
        
        return self.jr_map_sim

    @cached_property
    def bP(self) -> np.ndarray:
        """Pedersen geometric factor for conductance tensor."""
        mag = self.b_field.magnitude
        b_th, b_ph, b_r = (
            self.b_field.vec.theta / mag,
            self.b_field.vec.phi / mag,
            self.b_field.vec.r / mag,
        )
        return np.array([[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]])

    @cached_property
    def bH(self) -> np.ndarray:
        """Hall geometric factor for conductance tensor."""
        br = self.b_field.vec.r / self.b_field.magnitude
        return np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])

    @cached_property
    def bu(self) -> np.ndarray:
        """Geometric factor for u x B electric field."""
        Br = self.b_field.vec.r
        return -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])

    @cached_property
    def T_to_Ve(self) -> xr.DataArray:
        """Mapping external toroidal (T) to poloidal (Ve) potential."""
        return self._build_T_to_Ve()

    def _build_T_to_Ve(self) -> xr.DataArray:
        """Construct the T_to_Ve operator by integrating radially.
        
        Physics:
          1. Internal potential m_imp produces radial current jr via Laplacian.
          2. jr maps to sheet current JS(rk) in the magnetospheric layers.
          3. Integrated JS produces poloidal potential Ve at ionospheric boundary.
        
        Implementation:
          We use a 'Spectral Accelerated Kernel'. For SH basis, this stays in 
          coefficient space. For grid bases (CS), it maps to SH for propagation.
        """
        n_sol = self.solution_basis.index_length
        n_sh = self.basis.index_length
        T_to_Ve = xr.DataArray(np.zeros((n_sol, n_sol)), dims=("current_pot", "field_pot"))
        
        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return T_to_Ve

        # Radial steps
        rk_steps = np.asarray(self.FAC_integration_steps)
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        # 1. Integration Backbone (Spectral space)
        # We perform the bulk of the math on SH coefficients regardless of simulation basis
        G_inv_sh = tensor_pinv(self.G_Ve_to_JS_sh, n_leading_flattened=2, rtol=0).reshape(n_sh, -1)
        
        # Source operator factor: RI/mu0 * Laplacian
        m_imp_to_jr_sh_op = (self.RI / mu0) * self.basis.get_laplacian_operator(self.RI).to_dense()

        # Is this a pure spectral simulation?
        is_pure_sh = (self.solution_basis is self.basis or self.solution_basis.kind == "SH")
        
        # Accumulator (Spectral result: Ve_sh_coeffs / m_imp_coeffs)
        T_accum = np.zeros((n_sh, n_sol))

        for i, rk in enumerate(rks):
            # Coordinate mapping
            theta_m, phi_m = self.mainfield.map_coords(self.RI, rk, self.grid.theta, self.grid.phi)
            m_grid = Grid(theta=theta_m, phi=phi_m)
            rk_b = self.mainfield.discretize(self.grid, rk)
            m_b = self.mainfield.discretize(m_grid, self.RI)

            # Footprint evaluation of the DRIVING potential (m_imp)
            M_source = self.solution_basis.get_evaluation_matrix(m_grid)
            if hasattr(M_source, "toarray"): M_source = M_source.toarray()
            
            # Laplacian on solution basis (if CS, this is grid-local Laplacian)
            L_sol = self.solution_basis.get_laplacian_operator(self.RI).to_dense()
            m_imp_to_jr_sol_op = (self.RI / mu0) * L_sol

            # Source term mapping: m_imp_sol -> jr_grid -> JS_grid(rk)
            m_imp_to_JS_rk = np.einsum(
                "ij,jk->ijk",
                [rk_b.vec.theta / m_b.vec.r, rk_b.vec.phi / m_b.vec.r],
                M_source @ m_imp_to_jr_sol_op, optimize=True
            ).reshape(-1, n_sol)

            # Radial propagation factors (exact spectral decay)
            prop_vec = np.diag(self.basis.get_radial_shift_operator(rk, self.RI, kind="external").to_dense())
            if self.RM is not None:
                # Reflection and boundaries
                S_ext_RM = np.diag(self.basis.get_radial_shift_operator(self.RM, self.RI, kind="external").to_dense())
                S_int_rk = np.diag(self.basis.get_radial_shift_operator(rk, self.RM, kind="internal").to_dense())
                S_int_RI = np.diag(self.basis.get_radial_shift_operator(self.RI, self.RM, kind="internal").to_dense())
                prop_vec = prop_vec - S_ext_RM * S_int_rk
                factor_vec = -1.0 / (1.0 - S_ext_RM * S_int_RI)
            else:
                factor_vec = -1.0 * np.ones(n_sh)

            # Ve_coeffs += Delta_k * Prop(row) * G_inv * JS_grid * Factor(col)
            # Legacy consistency: prop scales the inverse (rows), factor scales the result (cols)
            step_mat = G_inv_sh @ m_imp_to_JS_rk
            step_mat = prop_vec[:, None] * step_mat
            if self.RM is not None:
                step_mat = step_mat * factor_vec
            else:
                step_mat = step_mat * -1.0
                
            T_accum += Delta_k[i] * step_mat

        # Map back to result basis
        if is_pure_sh:
            T_to_Ve.values = T_accum
        else:
            # Map integrated SH coefficients back to Solver coefficients (Hybrid)
            E_sh = self.basis.get_evaluation_matrix(self.grid)
            if hasattr(E_sh, "toarray"): E_sh = E_sh.toarray()
            E_sol = self.solution_basis.get_evaluation_matrix(self.grid)
            if hasattr(E_sol, "toarray"): E_sol = E_sol.toarray()
            P_sol = tensor_pinv(E_sol, rtol=0)
            T_to_Ve.values = (P_sol @ E_sh) @ T_accum
            
        return T_to_Ve

    @cached_property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        grad_op = as_linear_map(self.solution_basis.get_gradient_matrix(self.grid))
        G_grad = (1.0 / self.RI) * (grad_op * ((-self.RI / mu0)))
        G_total = G_grad.to_dense().reshape(2, -1, self.solution_basis.index_length)
        JS_coupling = np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0]))
        G_total += JS_coupling
        return G_total

    @cached_property
    def G_m_ind_to_JS(self) -> np.ndarray:
        """Operator mapping m_ind to sheet current on grid."""
        if self.G_Ve_to_JS is None:
             return None
        G = self.G_Ve_to_JS.copy()
        
        if self.RM is not None:
             # Add Magnetospheric Coupling (Standard legacy logic)
             br_shift_sh = np.diag(self.basis.get_radial_shift_operator(self.RM, self.RI, kind="external").to_dense())
             vi_shift_sh = np.diag(self.basis.get_radial_shift_operator(self.RI, self.RM, kind="internal").to_dense())
             den = 1.0 - br_shift_sh * vi_shift_sh
             
             G_coupling_sh = self.G_Ve_to_JS_sh * (br_shift_sh * vi_shift_sh / den)
             
             if self.input_adapter is not None:
                  G += np.tensordot(G_coupling_sh, self.input_adapter, axes=([2], [0]))
             else:
                  G += G_coupling_sh
                   
        return G

    @cached_property
    def G_Br_to_JS(self) -> Optional[np.ndarray]:
        """Operator mapping boundary Br to sheet current on grid."""
        if self.RM is None:
             return None
             
        br_shift_sh = np.diag(self.basis.get_radial_shift_operator(self.RM, self.RI, kind="external").to_dense())
        vi_shift_sh = np.diag(self.basis.get_radial_shift_operator(self.RI, self.RM, kind="internal").to_dense())
        den = 1.0 - br_shift_sh * vi_shift_sh
        
        L_op_sh = np.diag(self.basis.get_laplacian_operator(self.RI).to_dense())
        m_ind_to_Br_sh = -(self.RI**2) * L_op_sh
        
        G_Br_sh = self.G_Ve_to_JS_sh * (-br_shift_sh / den / m_ind_to_Br_sh)
        return G_Br_sh
    def _get_transformation_matrices(self, dfield: Field):
        """Compute transformation matrices for Apex coordinates."""
        # Get basis vectors
        r_eval = dfield.r_loc if dfield.r_loc is not None else self.RI
        bv = dfield.basis_vectors(r_eval, dfield.grid.theta, dfield.grid.phi)
        d3 = bv[2] # (3, N) e.g. [d3r, d3th, d3ph]
        e1 = bv[3]
        e2 = bv[4]

        # Get unit vector components
        mag = dfield.magnitude
        br, btheta, bphi = (
            dfield.vec.r / mag,
            dfield.vec.theta / mag,
            dfield.vec.phi / mag,
        )
        
        # Ensure flattened simple vectors
        br, btheta, bphi = br.flatten(), btheta.flatten(), bphi.flatten()
        ones = np.ones_like(br)
        zeros = np.zeros_like(br)

        # 1. Radial to Apex: Map radial vector (0, 0, 1)_sph -> Apex
        # Regularize division by br to handle magnetic equator (where br→0).
        # At the equator, the magnetic field is purely horizontal, so radial
        # vectors are perpendicular to B and should have minimal field-aligned
        # contribution. This regularization prevents division by zero on GL grids
        # which may place quadrature points exactly at the magnetic equator.
        EPS = 1e-10
        br_safe = np.where(np.abs(br) < EPS, np.sign(br + EPS) * EPS, br)
        ratio_theta = btheta / br_safe
        ratio_phi = bphi / br_safe
        
        # Transformation: proj_radial = d3 . r_hat_sph
        rad_to_fp = np.stack([ones, ratio_theta, ratio_phi], axis=0)
        fp_to_apex = np.array(d3)
        radial_to_apex = np.sum(fp_to_apex * rad_to_fp, axis=0) # (N,)
        
        # 2. Horizontal to Apex: Map horizontal vectors (th, ph) -> Apex
        # Transformation: [e1, e2] @ [th_comp; ph_comp]
        
        # Horizontal components in field-orthogonal basis
        c1 = np.stack([-ratio_theta, ones, zeros], axis=0)
        c2 = np.stack([-ratio_phi, zeros, ones], axis=0)
        horiz_to_fo = np.stack([c1, c2], axis=1) # (3, 2, N)
        
        # Field-orthogonal basis to Apex
        fo_to_apex = np.stack([e1, e2], axis=0) # (2, 3, N)
        
        # Compose mapping: (2, 3, N) @ (3, 2, N) -> (2, 2, N)
        horizontal_to_apex = np.einsum("ikn,kjn->ijn", fo_to_apex, horiz_to_fo, optimize=True)
        
        return radial_to_apex, horizontal_to_apex
