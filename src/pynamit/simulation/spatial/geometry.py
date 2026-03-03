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
from pynamit.primitives.field_spec import FieldSpec
from pynamit.primitives.grid import Grid
from pynamit.primitives.field import Field
from pynamit.utils import tensor_pinv
from pynamit.primitives.basis import Basis
from pynamit.math.linear_map import as_linear_map
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.simulation.induction.operators import ResistivityTensorOperator
from pynamit.simulation.induction.poloidal import PoloidalSystemMatrices
from pynamit.simulation.settings import SimulationMode
from pynamit.simulation.spatial.geometry_utils import (
    to_dense,
    get_radial_shift_diagonal,
    canonicalize_vector_basis_matrix,
)
from pynamit.simulation.spatial.pfac import PFACIntegrator
from pynamit.simulation.spatial.constraints import ApexMapper

if TYPE_CHECKING:
    from pynamit.primitives.grid import GridBasis


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
        solution_space: Optional[Any] = None,
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
        solution_space : Any, optional
            The basis used for the solution state variables.
        """
        self.basis = basis
        self.solution_space = solution_space if solution_space is not None else basis
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
        self.northern_hemisphere_apex_constraints = bool(
            getattr(settings, "northern_hemisphere_apex_constraints", False)
        )
        self.FAC_integration_steps = settings.FAC_integration_steps

        # Initialize core geometric objects
        self._init_evaluators(grid_basis)

        # Initialize hybrid adapter for basis transformation
        self.input_adapter = self._init_input_adapter()

        # Select closure basis for PFAC/radial coupling semantics.
        self.pfac_closure_basis = self._select_pfac_closure_basis(settings)

        # Initialize PFAC integrator
        self._pfac = PFACIntegrator(
            basis=self.pfac_closure_basis,
            solution_space=self.solution_space,
            mainfield=self.mainfield,
            RI=self.RI,
            RM=self.RM,
            FAC_integration_steps=self.FAC_integration_steps,
            ignore_PFAC=self.ignore_PFAC,
            magnetospheric_poloidal_lock=bool(
                getattr(settings, "magnetospheric_poloidal_lock", True)
            ),
            lock_toroidal_source_channels=(
                getattr(settings, "dynamics_mode", "legacy") == "full_induction"
            ),
        )

        # Initialize apex mapper and constraint mappings
        self._apex_mapper = ApexMapper(
            mainfield=self.mainfield,
            basis=self.basis,
            latitude_boundary=self.latitude_boundary,
            connect_hemispheres=self.connect_hemispheres,
            northern_hemisphere_apex_constraints=self.northern_hemisphere_apex_constraints,
            dynamics_mode=getattr(settings, "dynamics_mode", "legacy"),
        )
        self._init_constraint_mappings()

        # Initialize Poloidal System Matrices
        # Note: We defer initialization until after grid is set up
        self._poloidal_matrices: Optional[PoloidalSystemMatrices] = None
        self._poloidal_results_operators_cache: dict[tuple[Any, ...], Any] = {}

    def _select_pfac_closure_basis(self, settings: Any) -> Basis:
        """Select basis used for PFAC/radial closure operations.

        For ``cs_dominant`` we keep CS as the solution/state basis but use
        an auxiliary SH basis for PFAC/radial closure semantics.
        """
        mode = getattr(settings, "simulation_mode", None)
        mode_value = getattr(mode, "value", mode)
        sol_kind = getattr(self.solution_space, "kind", "")
        if sol_kind in ("CS", "GRID") and mode_value == "cs_dominant":
            from pynamit.spherical_harmonics.sh_basis import SHBasis

            logger.info(
                "Using SH auxiliary closure basis for PFAC/radial coupling in cs_dominant."
            )
            return FieldSpec(
                basis=SHBasis(int(settings.Nmax), int(settings.Mmax), mean_free=False),
                field_type="scalar",
                mean_free=True,
            )
        return self.basis

    @property
    def poloidal_matrices(self) -> PoloidalSystemMatrices:
        """Lazy-initialized poloidal system matrices.

        Returns
        -------
        PoloidalSystemMatrices
            The assembled poloidal system matrices.
        """
        if self._poloidal_matrices is None:
            self._poloidal_matrices = PoloidalSystemMatrices(
                basis=self.basis,
                solution_space=self.solution_space,
                grid=self.grid,
                b_field=self.b_field,
                RI=self.RI,
                pfac_integrator=self._pfac,
            )
        return self._poloidal_matrices

    def get_poloidal_results_operators(
        self,
        grid: Optional[Grid] = None,
        *,
        basis: Optional[Any] = None,
    ) -> Any:
        """Return explicit postprocessing operators for a target grid.

        This delegates to the shared results-operator bundle builder and uses
        the live simulation geometry/PFAC configuration.
        """
        from pynamit.postprocess.results_operators import build_poloidal_results_operators

        target_grid = self.grid if grid is None else grid
        target_basis = self.solution_space if basis is None else basis
        cache_key = (
            getattr(target_grid, "hash", id(target_grid)),
            getattr(target_basis, "signature", id(target_basis)),
        )
        cached = self._poloidal_results_operators_cache.get(cache_key)
        if cached is not None:
            return cached

        t_to_ve = np.asarray(self.poloidal_matrices.T_to_Ve, dtype=float)
        if hasattr(self.poloidal_matrices, "_apply_imposed_toroidal_poloidal_lock"):
            t_to_ve = np.asarray(
                self.poloidal_matrices._apply_imposed_toroidal_poloidal_lock(t_to_ve),
                dtype=float,
            )

        bundle = build_poloidal_results_operators(
            basis=target_basis,
            grid=target_grid,
            RI=float(self.RI),
            T_to_Ve=t_to_ve,
            RM=getattr(self, "RM", None),
        )
        self._poloidal_results_operators_cache[cache_key] = bundle
        return bundle

    def _init_input_adapter(self) -> Optional[np.ndarray]:
        """Initialize hybrid basis adapter if needed.

        Returns
        -------
        np.ndarray or None
            Input adapter matrix, or None if not needed.
        """
        if self.solution_space is self.basis:
            return None

        try:
            # Check for different basis types
            if getattr(self.solution_space, "kind", "") != getattr(self.basis, "kind", ""):
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
        quad_grid = GauntEngine(self.solution_space).quad_grid
        G_quad = to_dense(self.basis_zero_added.get_evaluation_matrix(quad_grid))

        res = xp.tensordot(G_quad, coeffs, axes=([1], [0]))
        return xp.moveaxis(res, 0, -1)

    def get_potential_to_JS_operator(
        self,
        potential_type: str,
        mode: Optional[Any] = None
    ) -> "LinearMap":
        """Get operator mapping potential coefficients to JS-like representation.

        This is the pre-resistivity mapping. The resistivity tensor (eta) is applied
        afterward to obtain the physical E-field.

        Parameters
        ----------
        potential_type : str
            Type of potential: "m_imp", "psi", "m_ind", or "Br".
        mode : SimulationMode, optional
            Simulation mode. If None or PURE_SPECTRAL, returns VSH representation.
            For other modes, returns grid-based operator.

        Returns
        -------
        LinearMap
            For spectral mode: Operator mapping potential coeffs to JS-like VSH coeffs.
            For grid mode: Operator mapping potential coeffs to JS-like grid values.
        """
        # Determine if we should use spectral (VSH) or grid representation
        use_spectral = (mode is None or mode == SimulationMode.PURE_SPECTRAL)

        if use_spectral:
            return self._get_JS_operator_spectral(potential_type)
        else:
            return self._get_JS_operator_grid(potential_type)

    def _get_JS_operator_spectral(self, potential_type: str) -> "LinearMap":
        """Get spectral (VSH) JS-like operator for given potential type.
        
        Delegates to PoloidalSystemMatrices.
        """
        return self.poloidal_matrices.get_potential_to_JS_operator(potential_type)

    def _get_JS_operator_grid(self, potential_type: str) -> Optional["LinearMap"]:
        """Get grid-based JS-like operator for given potential type.

        Preferred path uses precomputed grid operators ``G_<type>_to_JS``.
        If unavailable (e.g., Br in CS_DOMINANT), fall back to evaluating the
        spectral JS operator on the grid so downstream grid resistivity can be
        applied consistently.
        """
        G_grid = None
        # Route canonical poloidal operators explicitly through PoloidalSystemMatrices
        # instead of relying on Geometry facade attributes.
        if potential_type == "m_imp":
            G_grid = self.poloidal_matrices.G_m_imp_to_JS
        elif potential_type == "m_ind":
            G_grid = self.poloidal_matrices.G_m_ind_to_JS
        else:
            G_grid = getattr(self, f"G_{potential_type}_to_JS", None)
        if G_grid is None:
            # Fallback: evaluate spectral JS coefficients on grid.
            try:
                op_js_coeff = self._get_JS_operator_spectral(potential_type)
                G_vec = canonicalize_vector_basis_matrix(
                    self.solution_space.get_vector_basis_matrix(self.grid),
                    basis_index_length=self.solution_space.index_length,
                )
                # (2, N_grid, 2, N_coeffs) -> (2*N_grid, 2*N_coeffs)
                op_eval = as_linear_map(
                    G_vec.reshape(
                        G_vec.shape[0] * G_vec.shape[1],
                        G_vec.shape[2] * G_vec.shape[3],
                    )
                )
                return op_eval @ op_js_coeff
            except Exception as exc:
                logger.warning(
                    "Grid JS fallback failed for %s (%s).",
                    potential_type,
                    exc,
                )
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
        """Construct a unified conductivity operator (Potential -> E_coeffs).

        This operator maps potential coefficients to E-field coefficients,
        incorporating the resistivity tensor (η) which relates E = η·J.

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
            Operator mapping potential coefficients to E coefficients.
        """
        from pynamit.utils import to_numpy

        if mode == SimulationMode.PURE_SPECTRAL:
            return self._get_conductivity_operator_spectral(
                potential_type, eta_grid, etaP, etaH
            )
        else:
            return self._get_conductivity_operator_grid(potential_type, eta_grid)

    def get_potential_to_E_coeffs_operator(
        self,
        mode: Any,
        potential_type: str,
        eta_grid: np.ndarray,
        etaP: Optional[Any] = None,
        etaH: Optional[Any] = None,
    ) -> "LinearMap":
        """Explicit name for Potential -> E operator (post-resistivity)."""
        return self.get_conductivity_operator(
            mode=mode,
            potential_type=potential_type,
            eta_grid=eta_grid,
            etaP=etaP,
            etaH=etaH,
        )

    def _get_conductivity_operator_spectral(
        self,
        potential_type: str,
        eta_grid: np.ndarray,
        etaP: Optional[Any] = None,
        etaH: Optional[Any] = None
    ) -> "LinearMap":
        """Build spectral (Galerkin) conductivity operator."""
        from pynamit.utils import to_numpy

        # Get potential-to-JS operator in VSH representation
        op_JS = self.get_potential_to_JS_operator(potential_type, mode=None)

        # Build resistivity interaction matrix in VSH space
        M_vsh = self._build_resistivity_interaction_matrix(eta_grid, etaP, etaH)

        return as_linear_map(M_vsh) @ op_JS

    def _get_conductivity_operator_grid(
        self,
        potential_type: str,
        eta_grid: np.ndarray
    ) -> Optional["LinearMap"]:
        """Build grid-based (transform) conductivity operator."""
        # Get potential-to-JS operator (grid representation)
        op_JS = self._get_JS_operator_grid(potential_type)
        if op_JS is None:
            return None

        # Apply resistivity tensor and project back to coefficients
        op_eta = ResistivityTensorOperator(eta_grid).to_linear_map()
        op_P = as_linear_map(self.projection_matrix)
        
        # DEBUG: Isolate backend divergence


        return op_P @ op_eta @ op_JS

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
                engine = GauntEngine(self.solution_space)
                return engine.get_isotropic_interaction_matrix(etaP.coeffs, etaH.coeffs)
            except Exception as e:
                logger.warning(f"Isotropic Analytic construction failed ({e}).")

        # Try general analytic tensor path
        try:
            logger.info("Building General Analytic Interaction Matrix (Anisotropic Tensor)...")
            eta_quad = self._synthesize_to_gaunt(eta_grid)

            engine = GauntEngine(self.solution_space)
            return engine.get_interaction_matrix_from_real_grid(
                eta_quad[0, 0], eta_quad[1, 1], eta_quad[0, 1], eta_quad[1, 0]
            )
        except Exception as e:
            logger.warning(
                f"General Analytic construction failed ({e}), falling back to Quadrature."
            )

        # Quadrature fallback (robust)
        eta_quad = self._synthesize_to_gaunt(eta_grid)
        engine = GauntEngine(self.solution_space)
        return engine.get_vector_interaction_matrix(to_numpy(eta_quad))

    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Projection matrix (Grid Vector -> Basis Coefficients).

        For Helmholtz decomposition of vector fields E = -grad(Φ) + r×grad(Ψ).

        For Gauss-Legendre grids with quadrature weights, uses exact analysis:
            A = G^T @ diag(weights)
        which gives machine-precision transforms for orthonormal SH.

        For other grids (e.g., Cubed-Sphere), uses pseudo-inverse.

        Returns a 2D matrix of shape (2*N_coeffs, 2*N_grid).
        """
        if hasattr(self.grid_basis, "weights"):
            logger.info("Using exact GL quadrature for Helmholtz decomposition.")
            return self._build_exact_helmholtz_analysis()

        P = self.solution_space.construct_projection_matrix(self.grid)
        # CSBasis returns 4D tensor (2, N_coeffs, 2, N_grid), flatten to 2D
        if P.ndim == 4:
            P = P.reshape(P.shape[0] * P.shape[1], P.shape[2] * P.shape[3])
        return P

    def _build_exact_helmholtz_analysis(self) -> np.ndarray:
        """Build exact Helmholtz analysis matrix using GL quadrature.

        Uses weighted least-squares with GL quadrature weights:
            A = (G^T W G)^{-1} G^T W

        where W = diag(weights) accounts for the quadrature measure.
        """
        # Get gradient operators from the spectral basis
        G_th = self.basis.get_evaluation_matrix(self.grid, derivative="theta")
        G_ph = self.basis.get_evaluation_matrix(self.grid, derivative="phi")

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
        self.constraint_scalar_map_spectral = mappings.constraint_scalar_map_spectral
        self.constraint_scalar_map_sim = mappings.constraint_scalar_map_sim
        self.constraint_scalar_map_reference_spectral = (
            mappings.constraint_scalar_map_reference_spectral
        )
        self.constraint_scalar_map_reference_sim = mappings.constraint_scalar_map_reference_sim
        self.E_coeffs_to_E_apex_ll_diff = mappings.E_coeffs_to_E_apex_ll_diff

    def get_constraint_scalar_operator(self, input_basis: Any = None) -> np.ndarray:
        """Get the operator mapping coefficients to the configured constraint scalar.

        If input basis kind matches the Physics basis kind (e.g. SH), use the
        Physics/Spectral operator. Otherwise, use the Simulation operator
        (which includes adapter).
        """
        physics_kind = getattr(self.basis, "kind", None)
        input_kind = getattr(input_basis, "kind", None)

        if input_basis is None or input_kind == physics_kind:
            return self.constraint_scalar_map_spectral

        return self.constraint_scalar_map_sim

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
        return self._pfac.compute_T_to_Ve(self.G_Ve_to_JS_closure, self.grid)

    def _compute_vsh_operator(self, basis: Basis) -> np.ndarray:
        """Compute generic VSH induction operator (-1/mu0 * Curl @ Scaling)."""
        scaling_op = basis.get_potential_scaling_operator()
        curl_op = as_linear_map(basis.get_curl_matrix(self.grid))

        G_lin = (-1.0 / mu0) * (curl_op @ scaling_op)
        return to_dense(G_lin).reshape(2, -1, basis.index_length)

    @cached_property
    def G_Ve_to_JS_closure(self) -> np.ndarray:
        """Closure-basis induction operator for PFAC/radial coupling."""
        return self._compute_vsh_operator(self._pfac.basis)

    @cached_property
    def G_Ve_to_JS(self) -> np.ndarray:
        """Grid-native Induction operator (Solution Basis)."""
        if self.solution_space.kind == "CS":
            try:
                return self._compute_vsh_operator(self.solution_space)
            except (NotImplementedError, AttributeError):
                logger.warning(
                    "CS Basis does not support operator construction. Falling back to spectral."
                )

        # If SH basis (or fallback), reuse the SH operator
        if self.solution_space is self.basis:
            return self.G_Ve_to_JS_closure

        return self._compute_vsh_operator(self.solution_space)
