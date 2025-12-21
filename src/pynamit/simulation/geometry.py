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
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field import Field
from pynamit.utils import tensor_pinv
from pynamit.spherical_harmonics.sh_basis import SHBasis


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
        basis: SHBasis,
        grid_basis: Any,
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
        grid_basis : Any
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
                 # Check if adapter is needed specifically for Grid->SH
                 # Generic check for "different basis types implies adapter needed"
                 if getattr(self.solution_basis, "kind", "") != getattr(self.basis, "kind", ""):
                      logger.info("Basis mismatch detected: initializing hybrid adapter.")
                      G_dense = self.basis_evaluator.G

                      if scipy.sparse.issparse(G_dense):
                           G_dense = G_dense.toarray()
                      self.input_adapter = tensor_pinv(G_dense, n_leading_flattened=1)
             except Exception:
                  logger.warning("Failed to initialize basis adapter. Proceeding without one.")

        self._init_constraint_mappings()

        # Use solution basis for the Laplacian operator (m_imp -> jr)
        self.m_imp_to_jr = self.RI / mu0 * self.solution_basis.laplacian(self.RI)
        
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI
        # Always use spectral laplacian for induction scaling factors
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)

        # Induction operators: specialized logic removed.
        # Use simple attribute check.
        basis_for_induction = self.solution_basis if hasattr(self.solution_basis, "coeffs_to_delta_V") else self.basis
        
        if hasattr(basis_for_induction, "coeffs_to_delta_V"):
             Ve_to_J_df_coeffs = -self.RI / mu0 * basis_for_induction.coeffs_to_delta_V
             self.G_Ve_to_JS = (1.0 / self.RI) * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs
        else:
             self.G_Ve_to_JS = None





    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Projection matrix (Grid Vector -> Basis Coefficients)."""
        return self.solution_basis.construct_projection_matrix(self.basis_evaluator)

        
    def _init_evaluators(self, grid_basis: Any) -> None:
        """Set up grid, basis evaluators, and field evaluators."""
        self.grid = Grid(theta=grid_basis.arr_theta, phi=grid_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        
        # Use polymorphic method to get zero-added basis (for Monopole support in SH)
        if hasattr(self.basis, "get_extended_basis"):
            zero_added_basis = self.basis.get_extended_basis()
            self.basis_evaluator_zero_added = BasisEvaluator(zero_added_basis, self.grid)
        else:
            self.basis_evaluator_zero_added = self.basis_evaluator
             
        self.b_field = self.mainfield.discretize(self.grid, self.RI)

        # Optional evaluators for the conjugate hemisphere
        self.cp_grid = self.cp_basis_evaluator = self.cp_b_field = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_field = self.mainfield.discretize(self.cp_grid, self.RI)

    def _create_apex_operators(self, field: Field, evaluator: BasisEvaluator) -> tuple[np.ndarray, np.ndarray]:
        """Create current and field mapping operators for a given field/basis pair."""
        radial_to_apex, horizontal_to_apex = self._get_transformation_matrices(field)

        # jr_coeffs_to_j_apex
        jr_op = evaluator.scaled_G(radial_to_apex)

        # E_coeffs_to_E_apex
        E_op = np.einsum(
            "ijk,jklm->iklm",
            horizontal_to_apex,
            evaluator.G_helmholtz,
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
            self.b_field, self.basis_evaluator
        )
        
        self.E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres:
            # Conjugate Hemisphere Operators
            jr_coeffs_to_j_apex_cp, E_coeffs_to_E_apex_cp = self._create_apex_operators(
                self.cp_b_field, self.cp_basis_evaluator
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
        """Construct the T_to_Ve operator by integrating radially."""
        n = self.basis.index_length
        T_to_Ve = xr.DataArray(np.zeros((n, n)), dims=("i", "j"))
        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return T_to_Ve

        rk_steps = np.asarray(self.FAC_integration_steps)
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        if np.any(rks < self.RI):
            raise ValueError(
                "All FAC integration steps must be outside the ionospheric boundary (RI)."
            )
        if self.RM is not None and np.any(rks > self.RM):
            raise ValueError(
                "All FAC integration steps must be inside the magnetospheric boundary (RM)."
            )

        JS_rk_to_Ve_rk = tensor_pinv(self.G_Ve_to_JS, n_leading_flattened=2, rtol=0)
        m_imp_to_jr_coeffs = self.RI / mu0 * self.basis.laplacian(self.RI)

        for i, rk in enumerate(rks):
            logger.debug("PFAC integration step %d/%d (rk=%s)", i + 1, rks.size, rk)
            theta_mapped, phi_mapped = self.mainfield.map_coords(
                self.RI, rk, self.grid.theta, self.grid.phi
            )
            mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
            rk_b_field = self.mainfield.discretize(self.grid, rk)
            mapped_b_field = self.mainfield.discretize(mapped_grid, self.RI)
            mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)

            m_imp_to_jr_grid = mapped_basis_evaluator.scaled_G(m_imp_to_jr_coeffs)
            jr_to_JS_rk = np.array(
                [
                    rk_b_field.vec.theta / mapped_b_field.vec.r,
                    rk_b_field.vec.phi / mapped_b_field.vec.r,
                ]
            )
            m_imp_to_JS_rk = np.einsum("ij,jk->ijk", jr_to_JS_rk, m_imp_to_jr_grid, optimize=True)

            Ve_rk_to_Ve = self.basis.radial_shift_Ve(rk, self.RI).reshape((-1, 1, 1))
            if self.RM is not None:
                Ve_rk_to_Ve -= (
                    self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(rk, self.RM)
                ).reshape((-1, 1, 1))
                factor = -1.0 / (
                    1.0
                    - self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(self.RI, self.RM)
                )
            else:
                factor = -1.0

            JS_rk_to_Ve = JS_rk_to_Ve_rk * Ve_rk_to_Ve
            T_to_Ve += Delta_k[i] * factor * np.tensordot(JS_rk_to_Ve, m_imp_to_JS_rk, axes=2)
        return T_to_Ve

    # ----- G operators mapping to sheet current (JS) -----

    @cached_property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        G_T_to_JS = -1.0 / self.RI * self.basis_evaluator.G_grad * (self.RI / mu0)
        
        if self.G_Ve_to_JS is not None:
             term = np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0]))
             G_T_to_JS += term
             
        if self.input_adapter is not None:
             G_T_to_JS = np.tensordot(G_T_to_JS, self.input_adapter, axes=([2], [0]))
        
        return G_T_to_JS

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
        ratio_theta = btheta / br
        ratio_phi = bphi / br
        
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

    @cached_property
    def G_m_ind_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        G = self.G_Ve_to_JS.copy()
        # Use spectral physics (basis) for scaling
        m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)

        # Scale spectral operator first
        if self.RM is not None:
            br_shift = self.basis.radial_shift_Ve(self.RM, self.RI)
            vi_shift = self.basis.radial_shift_Vi(self.RI, self.RM)
            den = 1.0 - br_shift * vi_shift
            
            # Note: G_Ve_to_JS is already spectral if initialized with basis
            G_Br_to_JS_spectral = self.G_Ve_to_JS * (-br_shift / den / m_ind_to_Br)
            G *= 1.0 + (br_shift * vi_shift / den)
            
            # G_Br_to_JS should match Br input basis (SH)
            self.G_Br_to_JS = G_Br_to_JS_spectral

        # Map complete operator to grid if hybrid to match m_ind input (Grid)
        if self.input_adapter is not None:
             G = np.tensordot(G, self.input_adapter, axes=([2], [0]))
        
        return G
