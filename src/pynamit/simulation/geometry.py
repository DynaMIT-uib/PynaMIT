"""Geometry module.

This module contains the Geometry class, which encapsulates the spatial
grid, basis evaluators, magnetic field properties, and interhemispheric
mappings.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, TYPE_CHECKING

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
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.simulation.geometry_utils import to_dense, get_radial_shift_diagonal
from pynamit.simulation.pfac import PFACIntegrator
from pynamit.simulation.constraints import ApexMapper

if TYPE_CHECKING:
    from pynamit.primitives.grid import GridBasis
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

        # Initialize hybrid adapter for basis transformation
        self.input_adapter = self._init_input_adapter()

        # Initialize PFAC integrator
        self._pfac = PFACIntegrator(
            basis=self.basis,
            solution_basis=self.solution_basis,
            mainfield=self.mainfield,
            RI=self.RI,
            RM=self.RM,
            FAC_integration_steps=self.FAC_integration_steps,
            ignore_PFAC=self.ignore_PFAC,
        )

        # Initialize apex mapper and constraint mappings
        self._apex_mapper = ApexMapper(
            mainfield=self.mainfield,
            basis=self.basis,
            latitude_boundary=self.latitude_boundary,
            connect_hemispheres=self.connect_hemispheres,
        )
        self._init_constraint_mappings()

        # Solution/Simulation basis operators (for the unknowns on the solver grid)
        self.m_imp_to_jr = (self.RI / mu0) * self.solution_basis.get_laplacian_operator(self.RI)
        self.m_ind_to_Br = -(self.RI**2) * self.solution_basis.get_laplacian_operator(self.RI)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI

    def _init_input_adapter(self) -> Optional[np.ndarray]:
        """Initialize hybrid basis adapter if needed.

        Returns
        -------
        np.ndarray or None
            Input adapter matrix, or None if not needed.
        """
        if self.solution_basis is self.basis:
            return None

        try:
            # Check for different basis types
            if getattr(self.solution_basis, "kind", "") != getattr(self.basis, "kind", ""):
                logger.info("Basis mismatch detected: initializing hybrid adapter.")
                G_dense = to_dense(self.basis.get_evaluation_matrix(self.grid))
                return tensor_pinv(G_dense, n_leading_flattened=1)
        except Exception:
            logger.warning(
                "Failed to initialize basis adapter. Proceeding without one.",
                exc_info=True
            )
        return None

    def _synthesize_to_gaunt(self, grid_data: np.ndarray) -> np.ndarray:
        """Synthesize grid data from simulation grid to GauntEngine grid."""
        from pynamit.utils import xp, asarray

        # 1. Project to extended basis
        G_scalar = to_dense(self.basis_zero_added.get_evaluation_matrix(self.grid))

        if hasattr(self.grid_basis, "weights"):
            weights = self.grid_basis.weights
            GtW = G_scalar.T * weights
            GtWG = GtW @ G_scalar
            P_scalar = xp.linalg.solve(GtWG, GtW)
        else:
            P_scalar = tensor_pinv(asarray(G_scalar), n_leading_flattened=1)

        coeffs = xp.tensordot(P_scalar, grid_data, axes=([1], [-1]))

        # 2. Synthesize to Gaunt grid
        quad_grid = GauntEngine(self.solution_basis).quad_grid
        G_quad = to_dense(self.basis_zero_added.get_evaluation_matrix(quad_grid))

        res = xp.tensordot(G_quad, coeffs, axes=([1], [0]))
        return xp.moveaxis(res, 0, -1)

    def get_potential_to_E_operator(
        self,
        potential_type: str,
        mode: Optional[Any] = None
    ) -> "LinearMap":
        """Get operator mapping potential coefficients to E-field representation.

        This method provides a unified interface for potential-to-E-field operators
        across different simulation modes.

        Parameters
        ----------
        potential_type : str
            Type of potential: "m_imp", "m_ind", or "Br".
        mode : SimulationMode, optional
            Simulation mode. If None or PURE_SPECTRAL, returns VSH representation.
            For other modes, returns grid-based operator.

        Returns
        -------
        LinearMap
            For spectral mode: Operator mapping potential coeffs to VSH E-field coeffs.
            For grid mode: Operator mapping potential coeffs to E-field on grid.
        """
        from pynamit.simulation.dynamics import SimulationMode

        # Determine if we should use spectral (VSH) or grid representation
        use_spectral = (mode is None or mode == SimulationMode.PURE_SPECTRAL)

        if use_spectral:
            return self._get_E_operator_spectral(potential_type)
        else:
            return self._get_E_operator_grid(potential_type)

    def _get_E_operator_spectral(self, potential_type: str) -> "LinearMap":
        """Get spectral (VSH) E-field operator for given potential type."""
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
            t_mat = (-1.0 / mu0) * to_dense(scaling)

            if self.RM is not None:
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

    def _get_E_operator_grid(self, potential_type: str) -> Optional["LinearMap"]:
        """Get grid-based E-field operator for given potential type."""
        G_grid = getattr(self, f"G_{potential_type}_to_JS", None)
        if G_grid is None:
            return None
        return as_linear_map(G_grid.reshape(-1, G_grid.shape[-1]))

    def get_conductivity_operator(
        self,
        mode: Any,
        potential_type: str,
        eta_grid: np.ndarray,
        etaP: Optional[Any] = None,
        etaH: Optional[Any] = None
    ) -> "LinearMap":
        """Construct a unified conductivity operator (Potential -> JS_coeffs).

        This operator maps potential coefficients to sheet current coefficients,
        incorporating the resistivity tensor (η) which relates E-field to current.

        Parameters
        ----------
        mode : SimulationMode
            The simulation mode (PURE_SPECTRAL or grid-based).
        potential_type : str
            Type of potential: "m_imp", "m_ind", or "Br".
        eta_grid : np.ndarray
            Resistivity tensor on the grid, shape (2, 2, N).
        etaP : Field, optional
            Pedersen resistivity as spectral field (for analytic path).
        etaH : Field, optional
            Hall resistivity as spectral field (for analytic path).

        Returns
        -------
        LinearMap
            Operator mapping potential coefficients to JS coefficients.
        """
        from pynamit.simulation.dynamics import SimulationMode
        from pynamit.utils import to_numpy

        if mode == SimulationMode.PURE_SPECTRAL:
            return self._get_conductivity_operator_spectral(
                potential_type, eta_grid, etaP, etaH
            )
        else:
            return self._get_conductivity_operator_grid(potential_type, eta_grid)

    def _get_conductivity_operator_spectral(
        self,
        potential_type: str,
        eta_grid: np.ndarray,
        etaP: Optional[Any] = None,
        etaH: Optional[Any] = None
    ) -> "LinearMap":
        """Build spectral (Galerkin) conductivity operator."""
        from pynamit.utils import to_numpy

        # Get potential-to-E operator in VSH representation
        op_E = self.get_potential_to_E_operator(potential_type, mode=None)

        # Build resistivity interaction matrix in VSH space
        M_vsh = self._build_resistivity_interaction_matrix(eta_grid, etaP, etaH)

        return as_linear_map(M_vsh) @ op_E

    def _get_conductivity_operator_grid(
        self,
        potential_type: str,
        eta_grid: np.ndarray
    ) -> Optional["LinearMap"]:
        """Build grid-based (transform) conductivity operator."""
        from pynamit.simulation.operators import ResistivityTensorOperator

        # Get potential-to-E operator (grid representation)
        op_E = self._get_E_operator_grid(potential_type)
        if op_E is None:
            return None

        # Apply resistivity tensor and project back to coefficients
        op_eta = ResistivityTensorOperator(eta_grid).to_linear_map()
        op_P = as_linear_map(self.projection_matrix)
        
        # DEBUG: Isolate backend divergence


        return op_P @ op_eta @ op_E

    def _build_resistivity_interaction_matrix(
        self,
        eta_grid: np.ndarray,
        etaP: Optional[Any] = None,
        etaH: Optional[Any] = None
    ) -> np.ndarray:
        """Build resistivity interaction matrix in VSH space.

        Tries multiple approaches in order of preference:
        1. Isotropic analytic (for radial mainfield with spectral resistivity)
        2. General analytic tensor (anisotropic via Gaunt integrals)
        3. Quadrature fallback (robust numerical integration)
        """
        from pynamit.utils import to_numpy

        # Try isotropic analytic path
        use_isotropic = (
            etaP is not None and etaH is not None and
            getattr(self.mainfield, "kind", "dipole") == "radial"
        )

        if use_isotropic:
            try:
                logger.info("Building Analytic Interaction Matrix (Isotropic/Radial)...")
                engine = GauntEngine(self.solution_basis)
                return engine.get_isotropic_interaction_matrix(etaP.coeffs, etaH.coeffs)
            except Exception as e:
                logger.warning(f"Isotropic Analytic construction failed ({e}).")

        # Try general analytic tensor path
        try:
            logger.info("Building General Analytic Interaction Matrix (Anisotropic Tensor)...")
            eta_quad = self._synthesize_to_gaunt(eta_grid)

            engine = GauntEngine(self.solution_basis)
            return engine.get_interaction_matrix_from_real_grid(
                eta_quad[0, 0], eta_quad[1, 1], eta_quad[0, 1], eta_quad[1, 0]
            )
        except Exception as e:
            logger.warning(
                f"General Analytic construction failed ({e}), falling back to Quadrature."
            )

        # Quadrature fallback (robust)
        eta_quad = self._synthesize_to_gaunt(eta_grid)
        engine = GauntEngine(self.solution_basis)
        return engine.get_vector_interaction_matrix(to_numpy(eta_quad))

    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Projection matrix (Grid Vector -> Basis Coefficients).

        For Helmholtz decomposition of vector fields E = -grad(Φ) + r×grad(Ψ).

        For Gauss-Legendre grids with quadrature weights, uses exact analysis:
            A = G^T @ diag(weights)
        which gives machine-precision transforms for orthonormal SH.

        For other grids (e.g., Cubed-Sphere), uses pseudo-inverse.
        """
        if hasattr(self.grid_basis, "weights"):
            logger.info("Using exact GL quadrature for Helmholtz decomposition.")
            return self._build_exact_helmholtz_analysis()

        return self.solution_basis.construct_projection_matrix(self.grid)

    def _build_exact_helmholtz_analysis(self) -> np.ndarray:
        """Build exact Helmholtz analysis matrix using GL quadrature.

        Uses weighted least-squares with GL quadrature weights:
            A = (G^T W G)^{-1} G^T W

        where W = diag(weights) accounts for the quadrature measure.
        """
        # Get gradient operators from the spectral basis
        G_th = self.basis.get_G(self.grid, derivative="theta")
        G_ph = self.basis.get_G(self.grid, derivative="phi")

        # Convert to dense if sparse
        G_th = to_dense(G_th)
        G_ph = to_dense(G_ph)

        n_grid = G_th.shape[0]
        n_sh = G_th.shape[1]

        # Build Helmholtz basis vectors
        G_grad = np.array([G_th, G_ph])
        G_rxgrad = np.array([G_ph, -G_th])

        # G_helmholtz: (vec_comp, grid_pt, pot_type, sh_idx)
        G_helmholtz = np.stack([-G_grad, G_rxgrad], axis=2)

        # Flatten to 2D: (2*N_grid, 2*N_sh)
        G_flat = G_helmholtz.transpose(0, 1, 2, 3).reshape(2 * n_grid, 2 * n_sh)

        # Build weight matrix
        weights = self.grid_basis.weights
        W_diag = np.tile(weights, 2)

        # Weighted least-squares: A = (G^T W G)^{-1} G^T W
        GtW = G_flat.T * W_diag
        GtWG = GtW @ G_flat

        A = np.linalg.solve(GtWG, GtW)
        return A

    def _init_evaluators(self, grid_basis: "GridBasis") -> None:
        """Set up grid, basis evaluators, and field evaluators."""
        self.grid_basis = grid_basis
        self.grid = grid_basis.grid

        # Use polymorphic method to get zero-added basis (for monopole support in SH)
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

    def _init_constraint_mappings(self) -> None:
        """Initialize geometric operators related to constraints."""
        mappings = self._apex_mapper.build_constraint_mappings(
            grid=self.grid,
            b_field=self.b_field,
            RI=self.RI,
            cp_grid=self.cp_grid,
            cp_b_field=self.cp_b_field,
            input_adapter=self.input_adapter,
        )

        self.ll_mask = mappings.ll_mask
        self.jr_map_spectral = mappings.jr_map_spectral
        self.jr_map_sim = mappings.jr_map_sim
        self.E_coeffs_to_E_apex_ll_diff = mappings.E_coeffs_to_E_apex_ll_diff

    def get_jr_operator(self, input_basis: Any = None) -> np.ndarray:
        """Get the operator mapping jr to J_apex suitable for the input basis.

        If input basis kind matches the Physics basis kind (e.g. SH), use the
        Physics/Spectral operator. Otherwise, use the Simulation operator
        (which includes adapter).
        """
        physics_kind = getattr(self.basis, "kind", None)
        input_kind = getattr(input_basis, "kind", None)

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
        return self._pfac.compute_T_to_Ve(self.G_Ve_to_JS_sh, self.grid)

    @cached_property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        grad_op = as_linear_map(self.solution_basis.get_gradient_matrix(self.grid))
        G_grad = (1.0 / self.RI) * (grad_op * ((-self.RI / mu0)))
        G_total = to_dense(G_grad).reshape(2, -1, self.solution_basis.index_length)
        JS_coupling = np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0]))
        G_total += JS_coupling
        return G_total

    @cached_property
    def G_m_ind_to_JS(self) -> np.ndarray:
        """Operator mapping m_ind to sheet current on grid.

        This operator combines two physical effects:
        1. Local "Vacuum" Induction: m_ind -> E -> J.
        2. Gap Region / Magnetospheric Boundary Coupling: m_ind -> Coupling -> J.
        """
        if self.G_Ve_to_JS is None:
            return None
        G = self.G_Ve_to_JS.copy()

        if self.RM is not None:
            br_shift_sh, vi_shift_sh, den = self._pfac.get_coupling_factors()
            G_coupling_sh = self.G_Ve_to_JS_sh * (br_shift_sh * vi_shift_sh / den)

            if self.input_adapter is not None:
                G += np.tensordot(G_coupling_sh, self.input_adapter, axes=([2], [0]))
            else:
                G += G_coupling_sh

        return G

    def _compute_vsh_operator(self, basis: Basis) -> np.ndarray:
        """Compute generic VSH induction operator (-1/mu0 * Curl @ Scaling)."""
        scaling_op = basis.get_potential_scaling_operator()
        curl_op = as_linear_map(basis.get_curl_matrix(self.grid))

        G_lin = (-1.0 / mu0) * (curl_op @ scaling_op)
        return to_dense(G_lin).reshape(2, -1, basis.index_length)

    @cached_property
    def G_Ve_to_JS_sh(self) -> np.ndarray:
        """Spectral Induction operator (SH Basis)."""
        return self._compute_vsh_operator(self.basis)

    @cached_property
    def G_Ve_to_JS(self) -> np.ndarray:
        """Grid-native Induction operator (Solution Basis)."""
        if self.solution_basis.kind == "CS":
            try:
                return self._compute_vsh_operator(self.solution_basis)
            except (NotImplementedError, AttributeError):
                logger.warning(
                    "CS Basis does not support operator construction. Falling back to spectral."
                )

        # If SH basis (or fallback), reuse the SH operator
        if self.solution_basis is self.basis:
            return self.G_Ve_to_JS_sh

        return self._compute_vsh_operator(self.solution_basis)
