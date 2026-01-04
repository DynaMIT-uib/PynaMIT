"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required
for simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any, List, Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from functools import cached_property

from pynamit.primitives.field import Field
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver

from pynamit.math.linear_map import as_linear_map, LinearMap, diagonal_linear_map
from pynamit.simulation.geometry import Geometry
from pynamit.primitives.basis import Basis
from pynamit.math.constants import mu0
from pynamit.utils import asarray, use_jax, xp, to_numpy, tensor_pinv

logger = logging.getLogger(__name__)


class _ResistanceOperator:
    """Helper for block-diagonal resistance operator application.
    
    Wraps M_total_on_grid (2, 2, N) and behaves as a linear operator
    on flattened vectors (2*N).
    """
    def __init__(self, M: np.ndarray):
        self.M = asarray(M)
        self.n = M.shape[2]
        self.shape = (2 * self.n, 2 * self.n)
        self.dtype = M.dtype

    def matvec(self, x: Any) -> Any:
        # x is flat (2*N). Reshape to (2, N)
        x_reshaped = asarray(x).reshape(2, self.n)
        # M is (2, 2, N). x is (2, N).
        # y_0 = M_00 * x_0 + M_01 * x_1
        # y_1 = M_10 * x_0 + M_11 * x_1
        # Einsum: ijk, jk -> ik
        y = xp.einsum("ijk,jk->ik", self.M, x_reshaped)
        return y.reshape(-1)

    def rmatvec(self, y: Any) -> Any:
        y_reshaped = asarray(y).reshape(2, self.n)
        # Transpose M in first two dims for adjoint
        # MT_ijk = M_jik
        # Einsum: jik, jk -> ik
        res = xp.einsum("jik,jk->ik", self.M, y_reshaped)
        return res.reshape(-1)
        
    def matmat(self, X: Any) -> Any:
        # X is (2*N, Cols)
        cols = X.shape[1]
        X_reshaped = asarray(X).reshape(2, self.n, cols)
        # Einsum: ijk, jkl -> ikl
        res = xp.einsum("ijk,jkl->ikl", self.M, X_reshaped)
        return res.reshape(2 * self.n, cols)

    def rmatmat(self, Y: Any) -> Any:
        cols = Y.shape[1]
        Y_reshaped = asarray(Y).reshape(2, self.n, cols)
        res = xp.einsum("jik,jkl->ikl", self.M, Y_reshaped)
        return res.reshape(2 * self.n, cols)
        
    def to_dense(self) -> np.ndarray:
        # M is (2, 2, N). We construct the (2N, 2N) block matrix.
        # Structure:
        # [ diag(M_00)  diag(M_01) ]
        # [ diag(M_10)  diag(M_11) ]
        M_np = to_numpy(self.M)
        d00 = np.diag(M_np[0, 0])
        d01 = np.diag(M_np[0, 1])
        d10 = np.diag(M_np[1, 0])
        d11 = np.diag(M_np[1, 1])
        return np.block([[d00, d01], [d10, d11]])


class State:
    """Manages the ionospheric electrodynamic state.

    This class encapsulates the physical state (e.g., potentials,
    currents), handles the construction of all necessary numerical
    operators based on the provided geometry and settings, and
    orchestrates the time evolution of the system. It uses a Geometry
    object to manage the underlying grid and mappings.
    """

    def __init__(
        self,
        basis: Basis,
        mainfield: Any,
        grid_basis: Any,
        settings: Any,
        PFAC_matrix: Optional[np.ndarray] = None,
        solution_basis: Optional[Any] = None,
    ) -> None:
        """Initialize the State object.
        
        Parameters
        ----------
        basis : Basis
            The spectral basis.
        mainfield : Mainfield
            The main magnetic field.
        grid_basis : Any
            The basis defining the spatial grid (e.g., CSBasis).
        settings : Any
            Simulation settings.
        PFAC_matrix : np.ndarray, optional
            Pre-computed PFAC.
        solution_basis : Any, optional
            The basis for solution variables.
        """
        self.basis = basis
        self.solution_basis = solution_basis if solution_basis is not None else basis
        self._init_settings(settings)

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(
            basis, grid_basis, mainfield, settings, PFAC_matrix, solution_basis=self.solution_basis
        )

        # Operator for mapping velocity field `u` to E-field
        # (independent of conductance)
        self.u_coeffs_to_E_coeffs = self._create_u_to_E_operator()

        # The solver is configured here but remains stateless.
        self.m_imp_solver = LeastSquaresSolver(
            solver=self.solver_type, preconditioner=self.preconditioner
        )

        # Initialize state variables
        self.u: Optional[Field] = None
        self.Br: Optional[Field] = None
        self.jr: Optional[Field] = None
        self.etaP: Optional[Field] = None
        self.etaH: Optional[Field] = None

        # State tracking
        self.previous_input_data = {}

        # Invalidate all caches
        self._invalidate_caches()

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        self.solver_type = getattr(settings, "least_squares_solver", "cg")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.static_preconditioner = getattr(settings, "static_preconditioner", False)
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        
        # Mode Handling
        from pynamit.simulation.dynamics import SimulationMode
        # If settings has no simulation_mode, fallback to backward compat check
        if hasattr(settings, "simulation_mode"):
            self.mode = settings.simulation_mode
        else:
             # Legacy Fallback
             pure = getattr(settings, "pure_spectral", False)
             self.mode = SimulationMode.PURE_SPECTRAL if pure else SimulationMode.SPECTRAL_TRANSFORM
             
        # Map mode to legacy flags for internal checks (if any remain)
        self.pure_spectral = (self.mode == SimulationMode.PURE_SPECTRAL)

    def _create_u_to_E_operator(self) -> np.ndarray:
        """Operator mapping wind coefficients to E coefficients."""
        bu = asarray(self.geometry.bu)
        G_helmholtz = asarray(self.geometry.basis.get_vector_basis_matrix(self.geometry.grid))
        G_u_to_uxB_grid = xp.einsum("ijk,jklm->iklm", bu, G_helmholtz, optimize=True)
        
        # Flatten operator to (Output Grid Dims, Input Coeff Dims)
        # G shape is (2, N_grid, L...)
        # We want to combine (2, N_grid) into rows.
        grid_dim_prod = G_u_to_uxB_grid.shape[0] * G_u_to_uxB_grid.shape[1]
        G_u_to_uxB_flat = G_u_to_uxB_grid.reshape(grid_dim_prod, -1)

        # Projection Operator P
        P_matrix = self.geometry.projection_matrix
        
        # Apply projection: P @ G
        # Handles both Spectral (Pinv) and Grid (Identity) cases via polymorphism.
        if hasattr(P_matrix, "dot"):
            res_flat = P_matrix.dot(G_u_to_uxB_flat)
        else:
            res_flat = asarray(P_matrix) @ G_u_to_uxB_flat
            
        return res_flat.reshape(2, -1, G_u_to_uxB_grid.shape[-1])

    def _invalidate_caches(self) -> None:
        """Invalidate all conductance-dependent cached properties."""
        for attr in [
            "M_total_on_grid",
            "m_ind_to_E_coeffs",
            "m_imp_to_E_coeffs",
            "Br_to_E_coeffs",
            "E_map_constraint_operator",
            "m_ind_to_E_df_matrix",
            "m_imp_problem",
            "m_imp_preconditioner",
        ]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self._operator_linear_map_cache: Dict[
            Tuple[int, Tuple[int, ...], Tuple[int, ...]], Any
        ] = {}




    def _get_linear_map(
        self, op: Any, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
    ) -> LinearMap:
        if isinstance(op, LinearMap):
            return op
        cache_key = (id(op), input_shape, output_shape)
        cached = self._operator_linear_map_cache.get(cache_key)
        if cached is None:
            cached = as_linear_map(op, input_shape=input_shape, output_shape=output_shape)
            self._operator_linear_map_cache[cache_key] = cached
        return cached

    # ----- Cached Physical Properties (dependent on conductance) -----

    @cached_property
    def M_total_on_grid(self) -> np.ndarray:
        """Resistance tensor on the spatial grid."""
        if self.etaP is None or self.etaH is None:
            raise RuntimeError(
                "Conductance must be set before accessing conductance-dependent properties."
            )
        
        # Evaluate conductance fields on the simulation grid
        # This works regardless of the storage basis (SH, CS, etc.)
        theta = self.geometry.grid.theta
        phi = self.geometry.grid.phi
        r_dummy = self.geometry.RI
        
        etaP_val, _, _ = self.etaP.evaluate(r_dummy, theta, phi)
        etaH_val, _, _ = self.etaH.evaluate(r_dummy, theta, phi)
        
        eta_stacked = xp.stack([asarray(etaP_val), asarray(etaH_val)], axis=0)
        b_stacked = xp.stack([asarray(self.geometry.bP), asarray(self.geometry.bH)], axis=0)
        
        # Contract species (s) and grid points (k)
        # b_stacked: (s, i, j, k)
        # eta_stacked: (s, k)
        # output: (i, j, k)
        return xp.einsum("sijk,sk->ijk", b_stacked, eta_stacked, optimize=True)

    def _build_gaunt_vector_operator(
        self, m_total_coeffs: np.ndarray
    ) -> LinearMap:
        """
        Build the vector interaction operator for pure spectral mode.

        Takes conductance coefficients (in extended basis with n=0) and builds
        the VSH interaction matrix using the solution basis's GauntEngine.

        Parameters
        ----------
        m_total_coeffs : np.ndarray
            Conductance tensor coefficients in extended basis, shape (L_ext, 2, 2)

        Returns
        -------
        LinearMap
            Operator mapping VSH E coefficients to VSH J coefficients
        """
        from pynamit.math.gaunt import GauntEngine

        # Create GauntEngine with solution basis (determines quad grid)
        engine = GauntEngine(self.solution_basis)
        Q = engine.quad_grid.size

        # Synthesize conductance coefficients to the GauntEngine's quadrature grid
        # Use extended basis for synthesis (to preserve n=0 monopole component)
        extended_basis = self.geometry.basis_zero_added
        G_scalar_quad = extended_basis.get_evaluation_matrix(engine.quad_grid)
        if hasattr(G_scalar_quad, "toarray"):
            G_scalar_quad = G_scalar_quad.toarray()

        # Synthesize: (Q, L_ext) @ (L_ext, 2, 2) -> (Q, 2, 2)
        sigma_quad = xp.tensordot(G_scalar_quad, m_total_coeffs, axes=([1], [0]))
        # Transpose to (2, 2, Q) for GauntEngine
        sigma_quad = xp.transpose(sigma_quad, (1, 2, 0))

        # Build vector interaction matrix using the solution basis's GauntEngine
        M = engine.get_vector_interaction_matrix(to_numpy(sigma_quad))
        return as_linear_map(M)

    def _create_E_coeffs_operator(
        self, G_X_to_JS: Optional[np.ndarray], mapping_type: str = "poloidal"
    ) -> Optional[LinearMap]:
        from pynamit.simulation.dynamics import SimulationMode

        # PURE_SPECTRAL Path
        if self.mode == SimulationMode.PURE_SPECTRAL:
            # JS_coeffs = Σ_coeffs * E_coeffs
            m_total = self.M_total_on_grid # (2, 2, Q) - Conductance tensor on grid

            # Use extended basis (with monopole n=0) for conductance projection
            # This is critical because constant conductance needs n=0 to be represented
            extended_basis = self.geometry.basis_zero_added
            G_scalar = extended_basis.get_evaluation_matrix(self.geometry.grid)
            if hasattr(G_scalar, "toarray"):
                G_scalar = G_scalar.toarray()

            # Use exact GL quadrature if grid_basis has weights, else pseudo-inverse
            if hasattr(self.geometry.grid_basis, "weights"):
                # Weighted least-squares: A = (G^T W G)^{-1} G^T W
                # This accounts for non-orthonormal Schmidt quasi-normalized SH
                weights = self.geometry.grid_basis.weights
                GtW = G_scalar.T * weights  # (N_sh_ext, N_grid)
                GtWG = GtW @ G_scalar       # (N_sh_ext, N_sh_ext) - mass matrix
                P_scalar = xp.linalg.solve(GtWG, GtW)
            else:
                P_scalar = tensor_pinv(asarray(G_scalar), n_leading_flattened=1)

            # Project conductance to extended basis coefficients: (L_ext, 2, 2)
            m_total_coeffs = xp.tensordot(P_scalar, m_total, axes=([1], [2]))

            # Build Vector Interaction Operator using solution basis
            # The helper synthesizes to GauntEngine's quad grid internally
            op_M_spec = self._build_gaunt_vector_operator(m_total_coeffs)
            
            # Analytical Mapping (Gradient or Curl)
            # The VSH basis vectors are: Poloidal = -grad(Y), Toroidal = r×grad(Y)
            # When we synthesize E = G_vsh @ coeffs, the signs in the basis vectors apply.
            #
            # For GL path: E = (-1/mu0) * grad_matrix @ c = (-1/mu0) * grad(Y) * c
            # For pure spectral poloidal: E = (-grad Y) @ (factor * c)
            #   To match GL: (-grad Y) @ (factor * c) = (-1/mu0) * grad(Y) * c
            #   So: -factor = -1/mu0, meaning factor = +1/mu0
            #
            # For GL path toroidal: E = (-1/mu0) * curl_matrix @ scaling @ c
            # For pure spectral toroidal: E = (r×grad Y) @ (factor * scaling * c)
            #   The r×grad basis has same sign as curl_matrix, so factor = -1/mu0

            if mapping_type == "poloidal":
                # Poloidal (gradient) path: VSH basis is -grad(Y)
                # Need +1/mu0 to cancel the minus in the basis and match GL's -1/mu0 * grad
                phys_factor = 1.0 / mu0
                op_G_spec = self.solution_basis.get_gradient_operator()
                return op_M_spec @ (phys_factor * op_G_spec)
            else:
                # Toroidal (curl) path: VSH basis is r×grad(Y) (same sign as curl_matrix)
                # Need -1/mu0 to match GL's (-1/mu0) * curl * scaling
                phys_factor = -1.0 / mu0
                op_G_spec = self.solution_basis.get_curl_operator()
                scaling_op = self.solution_basis.get_potential_scaling_operator()
                return op_M_spec @ (phys_factor * op_G_spec @ scaling_op)

        # SPECTRAL_TRANSFORM (Legacy/Pseudo-Spectral) or CS_DOMINANT
        # For now, both rely on the grid-based construction provided by G_X_to_JS
        if G_X_to_JS is None:
            return None
        
        # Geometry (G): Coeffs -> Grid Vector
        G_backend = asarray(G_X_to_JS)
        op_G = as_linear_map(G_backend.reshape(-1, G_backend.shape[-1]))
        
        # Resistance (M): Grid Vector -> Grid Vector (Block Diagonal)
        res_op = _ResistanceOperator(self.M_total_on_grid)
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
        
        # Projection (P): Grid Vector -> Solution Basis
        P_matrix = self.geometry.projection_matrix
        op_P = as_linear_map(asarray(P_matrix) if not hasattr(P_matrix, "toarray") else P_matrix)
        
        return op_P @ op_M @ op_G
        

    @cached_property
    def m_ind_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping m_ind coefficients to E coefficients."""
        return self._create_E_coeffs_operator(self.geometry.G_m_ind_to_JS, mapping_type="toroidal")

    @cached_property
    def m_imp_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping m_imp coefficients to E coefficients."""
        return self._create_E_coeffs_operator(self.geometry.G_m_imp_to_JS, mapping_type="poloidal")

    @cached_property
    def Br_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping Br coefficients to E coefficients."""
        return self._create_E_coeffs_operator(
            getattr(self.geometry, "G_Br_to_JS", None), mapping_type="toroidal"
        )

    @cached_property
    def E_map_constraint_operator(self) -> Optional[LinearMap]:
        """Operator enforcing E-field mapping at low latitudes."""
        # This tensor maps E-field coefficients (or grid values) to
        # the difference in E_apex at conjugate points.
        # Shape: (2, Mask, 2, L)
        outer_tensor = self.geometry.E_coeffs_to_E_apex_ll_diff
        
        if outer_tensor is None:
            return None

        # Outer: Violation Check (E_apex difference). Flatten to (2*Mask, 2*L).
        outer_t = asarray(outer_tensor)
        n_mask, n_in = outer_t.shape[1], outer_t.shape[3]
        op_outer = as_linear_map(outer_t.reshape(2 * n_mask, 2 * n_in))

        # Inner: m_imp -> E-field
        op_inner = self.m_imp_to_E_coeffs
        if op_inner is None:
            return None

        # Composition: Constraint @ (m_imp -> E)
        return op_outer @ op_inner

    # ----- Solver Setup and Execution -----
    @cached_property
    def m_imp_problem(self) -> LeastSquaresProblem:
        """The least-squares problem definition for `m_imp`."""
        logger.info("Defining new least-squares problem for m_imp.")
        operators, data_shapes = [], []

        # Radial current (jr) must match imposed field.
        # Generalize m_imp_to_jr application:
        # If it's a matrix (Grid basis), use matrix multiplication.
        # If it's a vector (SH basis diagonal), use broadcasting.
        m_imp_to_jr = self.geometry.m_imp_to_jr
        jr_coeffs_to_j_apex = self.geometry.jr_map_sim
        
        op_apex = as_linear_map(jr_coeffs_to_j_apex)
        
        # Handle m_imp_to_jr (Matrix or Diagonal Scaling)
        # 1D or 2D handled automatically by as_linear_map
        op_m_to_jr = as_linear_map(m_imp_to_jr)
             
        op_jr = op_apex @ op_m_to_jr
        operators.append(op_jr)
        data_shapes.append((op_jr.shape[0],))

        # E-field must map at low latitudes.
        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            op_E = self.E_map_constraint_operator.with_scaling(self.ih_constraint_scaling)
            operators.append(op_E)
            data_shapes.append((op_E.shape[0],))

        # Add Tikhonov regularization if lambda is set.
        reg_ops, reg_weights = [], []
        if self.m_imp_regularization_lambda > 0:
            n = self.solution_basis.index_length
            # Use diagonal map for backend-agnostic identity
            identity_op = diagonal_linear_map(xp.ones(n))
            reg_ops.append(identity_op)
            reg_weights.append(self.m_imp_regularization_lambda)

        return LeastSquaresProblem(
            A=operators,
            solution_shape=self.solution_basis.index_length,
            data_shapes=data_shapes,
            regularization_matrices=reg_ops,
            regularization_weights=reg_weights,
        )

    @cached_property
    def m_imp_preconditioner(self) -> Optional[LinearMap]:
        """Preconditioner for the m_imp least-squares problem."""
        logger.info("Building new preconditioner for m_imp solver.")
        return self.m_imp_solver.build_preconditioner(problem=self.m_imp_problem, num_scenarios=1)

    def _solve_for_m_imp(
        self, jr_coeffs: Optional[np.ndarray], E_direct_coeffs: np.ndarray
    ) -> np.ndarray:
        """Solves for the imposed potential coefficients `m_imp`."""
        problem = self.m_imp_problem
        preconditioner = self.m_imp_preconditioner

        rhs_entries: List[Optional[Any]] = [None] * problem.num_data_terms
        if jr_coeffs is not None:

            # Select operator based on input basis
            op_rhs = self.geometry.get_jr_operator(self.jr.basis if self.jr else None)

            # Compute RHS: op @ jr_coeffs
            rhs_entries[0] = as_linear_map(op_rhs).matvec(asarray(jr_coeffs).reshape(-1))

        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            # Op is now basis-consistent (SH or Grid) thanks to Geometry.
            E_map_op = asarray(self.geometry.E_coeffs_to_E_apex_ll_diff)
            E_direct_input = asarray(E_direct_coeffs)

            # E_map_op: (2, Mask, 2, L)
            # E_direct_input: (2, L)
            # Contract Component(2) and Basis(L) dimensions.
            # Axes: Op dim 2,3 against Input dim 0,1.
            b_E = -xp.tensordot(E_map_op, E_direct_input, axes=([2, 3], [0, 1]))
            
            rhs_entries[1] = self.ih_constraint_scaling * xp.reshape(b_E, (-1,))

        solver = self.m_imp_solver
        solution = solver.solve(problem=problem, rhs=rhs_entries, preconditioner=preconditioner)
        if solution is None:
            solution = xp.zeros(self.solution_basis.index_length)
        return asarray(solution)

    # ----- State Update -----

    def _has_input_changed(self, key: str, current_data: dict, vars_for_key: list) -> bool:
        """Check if the input data has changed since the last update."""
        FLOAT_ERROR_MARGIN = 1e-6

        if key not in self.previous_input_data:
            return True

        prev_data = self.previous_input_data[key]
        for var in vars_for_key:
            if var not in prev_data or not np.allclose(
                current_data[var], prev_data[var], rtol=FLOAT_ERROR_MARGIN, atol=0.0
            ):
                return True
        return False

    def update(self, input_manager: Any, time: float, interpolation: bool = False) -> None:
        """Update the state variables based on the current input."""
        conductance_updated = False
        for key in input_manager.input_keys:
            current_data = input_manager.get_entry(key, time, interpolation)
            if current_data is None:
                continue

            # Check if the data has changed since the last time.
            if not self._has_input_changed(key, current_data, input_manager.variables[key]):
                continue

            # Update cache and proceed
            self.previous_input_data[key] = current_data
            updated_input = current_data

            storage_base = input_manager.get_storage_basis(key)
            if key == "conductance":
                conductance_updated = True
                self.etaP = Field.from_coefficients(storage_base, coeffs=updated_input["etaP"])
                self.etaH = Field.from_coefficients(storage_base, coeffs=updated_input["etaH"])
            elif key == "jr":
                self.jr = Field.from_coefficients(storage_base, coeffs=updated_input["jr"])
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None.")
                self.Br = Field.from_coefficients(storage_base, coeffs=updated_input["Br"])
            elif key == "u":
                self.u = Field.from_coefficients(
                    storage_base,
                    coeffs=updated_input["u"].reshape((2, -1)),
                    field_type="tangential",
                )

        if conductance_updated:
            logger.info("Conductance updated: invalidating caches and problem definition.")
            # Cache the preconditioner if it is static and not to be invalidated
            preconditioner_to_keep = (
                self.m_imp_preconditioner
                if self.static_preconditioner and hasattr(self, "m_imp_preconditioner")
                else None
            )

            self._invalidate_caches()

            # If we kept a static preconditioner, manually inject it back into the cached_proprety cache
            # The way to do this with cached_property is to set the attribute on the instance
            if preconditioner_to_keep is not None:
                logger.info("...retaining static preconditioner due to setting.")
                self.m_imp_preconditioner = preconditioner_to_keep

    # ----- State Calculation -----

    def _apply_operator(self, op: Any, coeffs: Any, output_shape: Tuple[int, ...]) -> np.ndarray:
        if op is None or coeffs is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return xp.zeros(output_shape)

        coeffs_shape = tuple(int(dim) for dim in asarray(coeffs).shape)
        linear_map = self._get_linear_map(op, coeffs_shape, output_shape)
        flat_in = linear_map.shape[1]
        backend_coeffs = asarray(coeffs).reshape(flat_in)
        res_flat = linear_map.matvec(backend_coeffs)
        res_backend = asarray(res_flat).reshape(output_shape)
        return res_backend

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.solution_basis.index_length)
        m_imp = self._solve_for_m_imp(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction effects."""

        E_shape = (2, self.solution_basis.index_length)
        u_coeffs = 0 if self.u is None else asarray(self.u.coeffs)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, u_coeffs, E_shape)
        if self.Br is not None:
            E_direct += self._apply_operator(self.Br_to_E_coeffs, asarray(self.Br.coeffs), E_shape)

        jr_coeffs = None if self.jr is None else asarray(self.jr.coeffs)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total E-field coefficients."""
        E_shape = (2, self.solution_basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, asarray(m_ind), E_shape)
        return self._calculate_total_E_field(E_direct_ind, None)

    # ----- Time Evolution -----

    @cached_property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix mapping m_ind to div-free E-field."""
        return self._build_m_ind_to_E_df_matrix()

    def _build_m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Construct the dense matrix for the induction operator."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.solution_basis.index_length
        if self.m_ind_to_E_coeffs is None:
            logger.info("Dense induction operator built (degenerate: no mapping available).")
            return xp.zeros((n, n))

        # Direct contribution from induced potential (without imposed solver feedback)
        E_direct_dense = asarray(self.m_ind_to_E_coeffs.to_dense()).reshape(2, n, n)

        problem = self.m_imp_problem
        rhs_entries = [None] * problem.num_data_terms if problem.num_data_terms > 0 else []

        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            E_map_op = asarray(self.geometry.E_coeffs_to_E_apex_ll_diff)
            
            # Map div-free E-field coefficients to constraint violations at magnetic equator
            # E_map_op: (2, Mask, 2, L) (or L_grid if hybrid)
            # E_direct_dense: (2, L, N) (L is basis length)
            # Contract: Component(0) vs Component(2), Basis(1) vs Basis(3)
            term = xp.tensordot(E_map_op, E_direct_dense, axes=([2, 3], [0, 1])) 
            
            b_E_block = -term
            if len(rhs_entries) > 1:
                rhs_entries[1] = self.ih_constraint_scaling * b_E_block

        rhs_block, _, num_scenarios = problem.assemble_rhs_block(rhs_entries)
        if rhs_block is None:
            op_rows = problem.get_system_operator().shape[0]
            rhs_block = xp.zeros((op_rows, n), dtype=E_direct_dense.dtype)
            num_scenarios = n
        rhs_block = asarray(rhs_block)

        # Solve in batch using cached SVD decomposition
        u, s, vt = problem.svd
        if s.size == 0:
            m_imp_block = xp.zeros((problem.solution_size, num_scenarios), dtype=rhs_block.dtype)
        else:
            tol = getattr(self.m_imp_solver, "tolerance", 0.0)
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
        if self.m_imp_to_E_coeffs is not None:
            m_imp_flat = asarray(m_imp_block)
            E_imp_flat = self.m_imp_to_E_coeffs.matmat(m_imp_flat)
            E_imp_block = asarray(E_imp_flat).reshape(2, n, n)
        else:
            E_imp_block = xp.zeros_like(E_direct_dense)

        total_E = E_direct_dense + E_imp_block
        logger.info("Dense induction operator built.")
        return asarray(total_E[1])

    def _calculate_d_m_ind_dt(self, m_ind: np.ndarray, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of the induced potential.

        This is the right-hand side of the ODE: d(m_ind)/dt = f(m_ind).
        The non-induced E-field is treated as a constant parameter for
        the ODE.
        """
        # Calculate the E-field contribution from the current induced
        # potential.
        E_ind_coeffs, _ = self.calculate_ind_coeffs(m_ind)
        E_df_ind = E_ind_coeffs[1]

        # Total divergence-free E-field is the sum of induced and
        # non-induced parts.
        E_df_total = E_df_ind + E_coeffs_noind[1]

        # Calculate the time derivative using the geometry operator.
        d_m_ind_dt = self.geometry.E_df_to_d_m_ind_dt * E_df_total
        return d_m_ind_dt

    def evolve_m_ind(
        self,
        m_ind: np.ndarray,
        dt: float,
        E_coeffs_noind: np.ndarray,
        steady_state_m_ind: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evolves the induced potential `m_ind` forward in time.

        Uses the integration scheme specified by `self.integrator`.
        Supports 'euler', 'exponential', and any method supported by
        `scipy.solve_ivp`.
        """
        backend_m_ind = asarray(m_ind)
        backend_E_noind = asarray(E_coeffs_noind)

        if self.integrator == "euler":
            d_m_ind_dt = self._calculate_d_m_ind_dt(backend_m_ind, backend_E_noind)
            return backend_m_ind + dt * d_m_ind_dt

        elif self.integrator == "exponential":
            # The exponential integrator requires the dense operator
            # matrix.
            op_A = asarray(self.geometry.E_df_to_d_m_ind_dt * self.m_ind_to_E_df_matrix)

            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(backend_E_noind)
            diff = backend_m_ind - asarray(steady_state_m_ind)

            if use_jax():
                from jax.scipy.linalg import expm as jax_expm

                evolved = jax_expm(dt * op_A) @ diff + asarray(steady_state_m_ind)
                return evolved

            # Use scipy.linalg.expm for NumPy
            op_A_np = to_numpy(op_A)
            diff_np = to_numpy(diff)
            steady_state_m_ind_np = to_numpy(steady_state_m_ind)

            evolved = expm(dt * op_A_np) @ diff_np
            return asarray(evolved) + asarray(steady_state_m_ind_np)

        else:
            # Fallback to scipy.solve_ivp for other integrators
            logger.debug(f"Using scipy.solve_ivp with method='{self.integrator}'.")

            def rhs_numpy(t, y):
                # Ensure input is backed by numpy if using scipy.solve_ivp
                y_backend = asarray(y)
                dy = self._calculate_d_m_ind_dt(y_backend, backend_E_noind)
                return to_numpy(dy)

            sol = solve_ivp(
                fun=rhs_numpy,
                t_span=(0, dt),
                y0=to_numpy(backend_m_ind),
                method=self.integrator,
                t_eval=[dt],
                dense_output=False,
            )

            if not sol.success:
                logger.warning(
                    f"solve_ivp integrator '{self.integrator}' failed with "
                    f"status {sol.status}: {sol.message}"
                )

            result = sol.y[:, -1]
            return asarray(result)

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        # This operation requires solving a linear system, which is most
        # robustly done with the dense matrix form of the operator.
        op_A = asarray(self.m_ind_to_E_df_matrix)
        vec_b = -asarray(E_coeffs_noind[1])
        return xp.linalg.solve(op_A, vec_b)
