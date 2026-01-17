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

from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.simulation.geometry import Geometry
from pynamit.primitives.basis import Basis
from pynamit.math.constants import mu0
from pynamit.utils import asarray, use_jax, xp, to_numpy, tensor_pinv
from pynamit.simulation.toroidal import ToroidalSystemMatrices
from pynamit.simulation.geometry_utils import to_dense

logger = logging.getLogger(__name__)


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
        
        # New: Toroidal Potential State Variable for "full_induction"
        self.psi: Optional[np.ndarray] = None
        self.d_psi_dt: Optional[np.ndarray] = None

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(
            basis, grid_basis, mainfield, settings, PFAC_matrix, solution_basis=self.solution_basis
        )

        # Initialize Toroidal System Matrices if in full_induction mode
        self.toroidal_matrices: Optional[ToroidalSystemMatrices] = None
        if self.dynamics_mode == "full_induction":
            self.toroidal_matrices = ToroidalSystemMatrices(
                basis=self.solution_basis, 
                grid=self.geometry.grid, 
                b_field=self.geometry.b_field,
                RI=self.RI
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
        # New: Track driver derivative for dynamic mode
        self.dt_jr_driver: Optional[Field] = None
        
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
        self.dynamics_mode = getattr(settings, "dynamics_mode", "legacy")
        print(f"DEBUG: State initialized with dynamics_mode={self.dynamics_mode}")
        
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
        """Operator mapping wind coefficients to E coefficients.
        
        Calculates M such that E_coeffs = M @ u_coeffs.
        E = u x B.
        Logic: v = u x B.
        Geometry.bu provides the cross-product tensor B_x such that v = B_x @ u.
        bu has shape (2, 2, Spatial...).
        """
        bu = asarray(self.geometry.bu)
        G_raw = asarray(self.geometry.basis.get_vector_basis_matrix(self.geometry.grid))
        
        # 1. Normalize shapes
        
        # bu: (2, 2, Spatial...) -> (2, 2, N_grid)
        if bu.ndim == 4: # (2, 2, Lat, Lon)
            bu_flat = bu.reshape(2, 2, -1)
        elif bu.ndim == 3: # (2, 2, Grid)
            bu_flat = bu
        else:
            raise ValueError(f"Unexpected bu shape: {bu.shape}")
            
        n_grid = bu_flat.shape[2]
        
        # G: (Component, Spatial..., Coeffs)
        # We need (Component, N_grid, Coeffs).
        # And Component dimension must match bu (2).
        if G_raw.ndim == 4: # SH: (Comp, Lat, Lon, Coeffs)
            G_flat = G_raw.reshape(G_raw.shape[0], n_grid, -1)
        elif G_raw.ndim == 3: # CS: (Comp, Grid, Coeffs)
            G_flat = G_raw
        else:
             raise ValueError(f"Unexpected G shape: {G_raw.shape}")
             
        # Handle Component Dimension Mismatch (3 vs 2)
        # Geometry.bu is 2x2 (assumes horizontal u).
        # If G has 3 components (r, th, ph), slice to (th, ph).
        if G_flat.shape[0] == 3 and bu_flat.shape[1] == 2:
             # Assume components are (r, th, ph). Take (th, ph).
             G_flat = G_flat[1:, :, :]
        elif G_flat.shape[0] != bu_flat.shape[1]:
             raise ValueError(f"Component mismatch: bu {bu_flat.shape[1]}, G {G_flat.shape[0]}")
             
        # 2. Compute Matrix Product: M = bu @ G
        # bu: (i, j, p). G: (j, p, c). Result: (i, p, c).
        # Elementwise on grid (p). Matrix product on components (j).
        M_grid = xp.einsum("ijp,jpc->ipc", bu_flat, G_flat, optimize=True)
        
        n_coeffs = M_grid.shape[2]
        
        # 3. Flatten for Projection (2*N_grid, N_coeffs)
        # Component Major: i=0... then i=1...
        # M_grid (2, N_grid, Coeffs) -> (2*N_grid, Coeffs)
        M_flat = M_grid.reshape(2 * n_grid, n_coeffs)

        # 4. Projection
        P_matrix = self.geometry.projection_matrix
        
        if hasattr(P_matrix, "dot"):
            res_flat = P_matrix.dot(M_flat)
        else:
            res_flat = asarray(P_matrix) @ M_flat
            
        return res_flat.reshape(2, -1, n_coeffs)

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

    @cached_property
    def dt_jr_problem(self) -> LeastSquaresProblem:
        """The least-squares problem definition for `dt_jr` (Dynamic).

        Delegates to ToroidalSystemMatrices.build_least_squares_problem()
        with parameters from the current state.
        """
        logger.info("Defining new least-squares problem for dt_jr.")
        if self.toroidal_matrices is None:
            raise RuntimeError("Toroidal matrices required for dt_jr problem.")

        return self.toroidal_matrices.build_least_squares_problem(
            jr_map_operator=self.geometry.jr_map_sim,
            constraint_scaling=1000.0,
            regularization_lambda=0.0,
        )




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
        """Physical Resistance (Resistivity) tensor on the spatial grid."""
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
        
        # Robust Shape Handling
        # Flatten spatial dimensions to (S, Tensor1, Tensor2, N_points)
        # b_stacked shape: (S, T1, T2, Spatial...)
        if b_stacked.ndim > 4: # e.g. (S, T1, T2, Lat, Lon)
            s, t1, t2 = b_stacked.shape[:3]
            b_flat = b_stacked.reshape(s, t1, t2, -1)
        else: # (S, T1, T2, Grid)
            b_flat = b_stacked
            
        # eta_stacked shape: (S, Spatial...)
        if eta_stacked.ndim > 2:
            s_eta = eta_stacked.shape[0]
            eta_flat = eta_stacked.reshape(s_eta, -1)
        else:
            eta_flat = eta_stacked
            
        # Contract species (s) and grid points (k)
        # b_flat: (s, i, j, k)
        # eta_flat: (s, k)
        # output: (i, j, k) -> (T1, T2, N_points)
        M_flat = xp.einsum("sijk,sk->ijk", b_flat, eta_flat, optimize=True)
        
        # Reshape output back to original spatialdims if needed?
        # Operators usually expect flattened grid for vector ops.
        # But if SH legacy expects (3, 3, Lat, Lon)...
        # LinearMap expects flattened.
        
        # If geometry uses 2D grid, we might need to reshape back.
        # But LinearMap usually wraps flat arrays or handles reshaping internally.
        # Let's verify what `LinearMap` expects?
        # It expects `(Matrix, ...)`?
        # Wait, M is tensor field.
        # If downstream uses M.
        # m_total_coeffs -> ?
        return M_flat

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
        """Unified operator mapping potential coefficients to electric field (E) coefficients."""
        potential_type = "m_imp" if mapping_type == "poloidal" else "m_ind"
        
        # Check for Br specifically
        if G_X_to_JS is getattr(self.geometry, "G_Br_to_JS", None) and G_X_to_JS is not None:
            potential_type = "Br"
            
        # Pass eta fields if available for analytic mode
        etaP_field = getattr(self, "etaP", None)
        etaH_field = getattr(self, "etaH", None)

        return self.geometry.get_conductivity_operator(
            mode=self.mode,
            potential_type=potential_type,
            eta_grid=self.M_total_on_grid,
            etaP=etaP_field,
            etaH=etaH_field
        )
        

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

        # This tensor maps E-field coefficients (or grid values) to
        # the difference in E_apex at conjugate points.
        # Shape: (2, Mask, 2, L) or (2, Mask, L) wrapped in ConstraintOperator
        op_obj = self.geometry.E_coeffs_to_E_apex_ll_diff
        
        if op_obj is None:
            return None

        # Extract underlying tensor if wrapped
        if hasattr(op_obj, "tensor"):
             outer_t = asarray(op_obj.tensor)
        else:
             outer_t = asarray(op_obj)
        
        if outer_t.ndim == 4:
             # Legacy SH: (2, Mask, Dim2, Coeffs). Presumes Dim2=2.
             n_mask, n_in = outer_t.shape[1], outer_t.shape[3]
             op_outer = as_linear_map(outer_t.reshape(2 * n_mask, 2 * n_in))
        elif outer_t.ndim == 3:
             # CS: (2, Mask, Coeffs)
             n_mask, n_coeffs = outer_t.shape[1], outer_t.shape[2]
             op_outer = as_linear_map(outer_t.reshape(2 * n_mask, n_coeffs))
        else:
             raise ValueError(f"Unexpected constraint tensor shape: {outer_t.shape}")

        # Inner: m_imp -> E-field
        op_inner = self.m_imp_to_E_coeffs
        if op_inner is None:
            return None

        # Composition: Constraint @ (m_imp -> E)
        return op_outer @ op_inner

    # ----- Solver Setup and Execution -----
    @cached_property
    def m_imp_problem(self) -> LeastSquaresProblem:
        """The least-squares problem definition for `m_imp`.

        Delegates to PoloidalSystemMatrices.build_least_squares_problem()
        with parameters from the current state.
        """
        logger.info("Defining new least-squares problem for m_imp.")

        # Determine E-field constraint operator based on mode
        # In full_induction mode, IH coupling is handled by the global solution
        E_constraint_op = None
        if (
            self.connect_hemispheres
            and self.dynamics_mode != "full_induction"
            and self.E_map_constraint_operator is not None
        ):
            E_constraint_op = self.E_map_constraint_operator

        return self.geometry.poloidal_matrices.build_least_squares_problem(
            jr_map_operator=self.geometry.jr_map_sim,
            E_constraint_operator=E_constraint_op,
            connect_hemispheres=(E_constraint_op is not None),
            ih_constraint_scaling=self.ih_constraint_scaling,
            regularization_lambda=self.m_imp_regularization_lambda,
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

        if (
            self.connect_hemispheres 
            and self.dynamics_mode != "full_induction"
            and self.E_map_constraint_operator is not None
        ):

            # Op is now basis-consistent (SH or Grid) thanks to Geometry.
            # It is wrapped in ConstraintOperator to handle rank differences.
            E_map_op = self.geometry.E_coeffs_to_E_apex_ll_diff
            E_direct_input = asarray(E_direct_coeffs)

            if hasattr(E_map_op, "apply"):
                 b_E = E_map_op.apply(E_direct_input)
            else:
                 # Fallback for raw arrays (if somehow not wrapped)
                 E_map_op = asarray(E_map_op)
                 if E_map_op.ndim == 4:
                     b_E = -xp.tensordot(E_map_op, E_direct_input, axes=([2, 3], [0, 1]))
                 else:
                     b_E = -xp.tensordot(E_map_op, E_direct_input, axes=([2], [0]))
            
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

    def _ensure_basis(self, field: Field, field_type: str = "scalar") -> Field:
        """Ensure the field is represented in the solution basis."""
        if field.basis is self.solution_basis or field.basis == self.solution_basis:
             return field
             
        # Handle projection to CS/Nodal basis
        if hasattr(self.solution_basis, "kind") and self.solution_basis.kind == "CS":
             grid = self.geometry.grid
             # Evaluate on grid
             v1, v2, v3 = field.evaluate(self.geometry.RI, grid.theta, grid.phi)
             
             if field_type == "scalar":
                  return Field.from_coefficients(
                      self.solution_basis, 
                      coeffs=asarray(v1).flatten(), 
                      field_type="scalar"
                  )
             elif field_type == "tangential":
                  # u (wind): theta, phi components
                  v2_flat = asarray(v2).flatten()
                  v3_flat = asarray(v3).flatten()
                  new_coeffs = xp.stack([v2_flat, v3_flat], axis=0)
                  return Field.from_coefficients(
                      self.solution_basis, 
                      coeffs=new_coeffs, 
                      field_type="tangential"
                  )
        
        return field

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
 
            # Check for derivatives
            current_deriv = None
            if key == "jr" and hasattr(input_manager, "get_entry_with_derivative"):
                # This assumes InputManager has been updated to provide derivatives methods.
                # We need interpolation to get correct derivatives from sparse time points
                _, current_deriv = input_manager.get_entry_with_derivative(key, time, interpolation=True)

            storage_base = input_manager.get_storage_basis(key)
            if key == "conductance":
                conductance_updated = True
                f_etaP = Field.from_coefficients(storage_base, coeffs=updated_input["etaP"])
                f_etaH = Field.from_coefficients(storage_base, coeffs=updated_input["etaH"])
                self.etaP = self._ensure_basis(f_etaP, "scalar")
                self.etaH = self._ensure_basis(f_etaH, "scalar")
            elif key == "jr":
                f_jr = Field.from_coefficients(storage_base, coeffs=updated_input["jr"])
                self.jr = self._ensure_basis(f_jr, "scalar")
                if current_deriv is not None:
                     f_dt_jr = Field.from_coefficients(storage_base, coeffs=current_deriv["jr"])
                     self.dt_jr_driver = self._ensure_basis(f_dt_jr, "scalar")
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None.")
                f_Br = Field.from_coefficients(storage_base, coeffs=updated_input["Br"])
                self.Br = self._ensure_basis(f_Br, "scalar")
            elif key == "u":
                f_u = Field.from_coefficients(
                    storage_base,
                    coeffs=updated_input["u"].reshape((2, -1)),
                    field_type="tangential",
                )
                self.u = self._ensure_basis(f_u, "tangential")

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
        
        # DYNAMIC MODES: Handle Toroidal Induction
        if self.dynamics_mode == "full_induction":
            return self._calculate_dynamic_state(E_direct, jr_coeffs)
            
        # LEGACY MODE
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def _calculate_dynamic_state(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for state in full induction mode (Toroidal + Poloidal).
        
        Solves the partitioned system:
        L * dt_jr_gap = K - L * dt_jr_driver
        """
        if self.toroidal_matrices is None:
             raise RuntimeError("Toroidal matrices not initialized.")
             
        # 1. Compute Source Vector K (RHS)
        # K = - Integral [ Y * S_known ]
        # S_known excludes toroidal d/dt term.
        # Ideally, we implement a helper to get S_known explicitly.
        # For now, let's assume we can compute the FULL RHS as usual but 
        # we need to subtract the toroidal part or construct it purely.
        
        # Actually, the user spec says:
        # "Reuse the existing calc_RHS logic but filter out the toroidal terms"
        # Since we don't have a `calc_RHS` method exactly here (it's embedded in `evolve_m_ind`),
        # we need to be careful.
        
        # Wait, `calculate_noind_coeffs` is usually called by `dynamics.py` BEFORE evolution.
        # But for the dynamic solver, we need `dt_jr` (which is `x`).
        # `dt_jr` drives the evolution of `psi`.
        
        # Let's align with the `State` class flow.
        # Normally `calculate_noind_coeffs` returns (E_total, m_imp).
        # In Dynamic mode, `m_imp` should be consistent with `psi` (if we define psi as toroidal potential leading to current).
        # Or, `m_imp` is the POLOIDAL part of the potential response.
        
        # User Spec: "Update Toroidal Field (psi)... Update Poloidal Field (P)..."
        # "P^(n+1) = P^n + dt * T[dt_jr]"
        # This implies we are integrating `m_imp` (which is P) in time too!
        
        # So in this function we should primarily be solving for `dt_jr` (the rate).
        # However, `calculate_noind_coeffs` expects to return STATIC COEFFICIENTS for the current timestep step.
        # The integration happens in `dynamics.py` / `TimeStepper`.
        
        # So, we should return the CURRENT E-field and m_imp.
        # But `m_imp` is now a state variable, evolved by `TimeStepper`.
        # So here we just return the current values from `self.m_imp` (if we store it) or `self.psi`.
        
        # If `m_imp` is state, we don't solve for it statically here.
        # We just return the stored value.
        
        # But we DO need to calculate `dt_jr` so the TimeStepper can use it.
        # Let's store `dt_jr` as a side effect or return it?
        # `dynamics.py` calls this method.
        # We might need to refactor `dynamics.py` significantly.
        # For now, let's implement the solver step here as a helper `_solve_dt_jr`.
        # And `calculate_noind_coeffs` just assembles E from current psi/m_imp.
        
        # If we assume `self.psi` contains Toroidal potential T, and maybe we add `self.m_imp` as Poloidal potential P state?
        # Legacy code calculates `m_imp` statically from `jr`.
        # Dynamic code evolves `m_imp`.
        # So we need to store `m_imp` in `self`.
        
        if self.psi is None:
             # Initialize if first run
             n = self.solution_basis.index_length
             self.psi = xp.zeros(n)
             # We should also have self.m_imp_state
             
        # Compute E from current potentials
        # E = E_direct + E(psi) + E(m_imp)
        # Note: E(psi) is Toroidal E from toroidal potential T.
        # E(m_imp) is Poloidal E from poloidal potential P.
        
        # For now, to satisfy the API:
        # We treat `m_imp` here as the Poloidal Potential P.
        # Return E_total, m_imp.
        
        # But we need to update `dt_jr` for the NEXT step.
        # Computing `dt_jr` requires the CURRENT E-field (including induced!).
        # But `calculate_noind_coeffs` is supposed to return "No Induced" part?
        # In legacy: "No Induced" = Direct + Imposed (Static).
        # In dynamic: "No Induced" might mean "Everything except self-induction of m_ind"?
        # Or does it mean "Everything known at time t"?
        
        # Let's assume this method calculates E based on the CURRENT state variables.
        # For dynamic mode, m_imp is state.
        # So we construct E_noind = E_direct.
        # E_total = E_direct + E_psi + E_m_imp + E_m_ind?
        # Wait, m_ind is handled by `evolve_m_ind` in legacy.
        # The user spec says "Disable ApexMapping... coupling handled by global solution".
        # And "m_ind" might be subsumed by the new physics or kept separate?
        # "Specifically, ensure K includes only the poloidal contribution..."
        
        # E-field calculation and solver step
        n = self.solution_basis.index_length
    
        if self.psi is None:
             # Initialize if first run
             self.psi = xp.zeros(n)
             self.d_psi_dt = xp.zeros(n)

        # 1. Solve for dt_jr
        # We need the current TOTAL E-field (excluding toroidal induction term d_psi/dt to be solved?)
        # Wait, the Master Equation is: dt_jr = - (1/mu B^2) * CurlCurl E . B
        # E = E_potential + E_induced.
        # E_induced = d_A / dt. A = A_pol + A_tor.
        # But here we are separating the Toroidal Induction part?
        # The spec says: "Filter out the toroidal contribution to d/dt(r x b_S)".
        # This implies K includes everything EXCEPT the term L*dt_jr is modelling.
        
        # Calculate E_current from known potentials
        u_coeffs = 0 if self.u is None else asarray(self.u.coeffs)
        E_u = self._apply_operator(self.u_coeffs_to_E_coeffs, u_coeffs, (2, n))
        
        # Imposed Poloidal Field (m_imp) from *current* state
        # Note: in dynamic mode, m_imp is likely evolved or derived.
        # For now, let's assume we use the legacy m_imp solution as the "Base Poloidal Field".
        # Or if m_imp is a state variable, we use self.m_imp_state.
        
        # Let's assume for this step that `m_imp` is calculated statically consistent with `jr`.
        # (Mixed mode: Static Poloidal, Dynamic Toroidal).
        # E_poloidal = E(m_imp)
        m_imp_curr = self._solve_for_m_imp(jr_coeffs, E_u) # Poloidal part
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp_curr, (2, n))
        
        # Total Known E (Poloidal + Motional)
        E_known = E_u + E_imp
        
        # 2. Solve for dt_jr
        dt_jr = self.solve_dt_jr(E_known)
        
        # 3. Update Toroidal State (psi)
        # This function returns E-coeffs, not evolves state.
        # Evolution happens in `evolve_m_ind`.
        # But we need to return the E-field corresponding to the *rates*?
        # No, `calculate_noind_coeffs` returns E for the main loop.
        
        # Wait, `TimeStepper` needs `d_y / dt`.
        # If `psi` is the state, we need `d_psi / dt`.
        # dt_jr gives us `d_psi / dt` !
        # J_tor = Curl Curl (r Psi). J_r term? 
        # Actually dt_jr is related to dt_psi.
        # Toroidal field is defined by scalar T (or psi). 
        # Curl(T r) = - r x grad T.
        # Curl Curl (T r) = ...
        # Relationship: j_r is defined by Poloidal, not Toroidal field.
        # Wait, j_r = - Delta_S P / r.
        # Toroidal field T has NO radial current.
        # So why are we solving for dt_jr?
        # "Master Equation... identity for the radial current density jr".
        # This equation governs the evolution of `jr` (which comes from Poloidal field P)
        # due to Induction (which involves Toroidal field).
        
        # Ah, `dt_jr` updates `jr` (Poloidal Source).
        # This allows calculating `P` (Poloidal Potential m_imp).
        # What about `psi`? "Update Toroidal Field... Derive d_psi from d_jr".
        # Equation (26) inverse?
        
        # Okay, so finding `dt_jr` gives us everything.
        # We store `dt_jr` or `d_psi_dt` for the integrator.
        
        # Here we just return the E-field of the CURRENT state.
        # E_total = E_known + E_psi.
        # E_psi comes from `psi`.
        # We need an operator `psi_to_E`.
        # Toroidal E-field E_T = - dA_T/dt = - d/dt Curl(r psi)? 
        # No, usually E = -grad Phi - dA/dt.
        # If psi is Toroidal Potential for B? B = Curl(r psi).
        # Then E? 
        # Let's assume standard VSH E-field structure.
        # If we have `psi` (Toroidal B potential or Poloidal E potential?),
        # let's assume `psi` represents the Toroidal Magnetic Field T.
        # Then the Electric Field is Poloidal?
        # E = - grad V - dA/dt.
        # This is getting deep into physics definitions not fully in the snippet.
        
        # 3. Handle Scaling: Derive d_psi_dt from dt_jr
        # jr = (R/mu0) * Laplacian(psi). So psi = inv(m_imp_to_jr) * jr.
        op_m_to_jr = as_linear_map(self.geometry.m_imp_to_jr)
        # For SH, this is a diagonal operator. For general, we use pinv.
        # Use dense pinv for robustness on the small spectral system.
        m_to_jr_dense = to_dense(op_m_to_jr)
        jr_to_m_dense = tensor_pinv(m_to_jr_dense)
        self.d_psi_dt = asarray(jr_to_m_dense @ dt_jr)
        
        # 4. Return Total E-field and current Toroidal state
        # psi represents the Toroidal Potential (m_imp Equivalent)
        # Use spectral mode=None because E_known is always in spectral space (projected).
        op_m_imp_to_E = self.geometry.get_potential_to_E_operator("m_imp", mode=None)
        E_psi = asarray(op_m_imp_to_E.matmat(self.psi)).reshape(E_known.shape)
        
        E_total = E_known + E_psi
        
        return E_total, self.psi

    def solve_dt_jr(self, E_known: np.ndarray) -> np.ndarray:
        """Solve constrained system for dt_jr.

        Uses ToroidalSystemMatrices.compute_rhs_physics() for the physics RHS
        and assembles the constraint RHS from driver data.
        """
        problem = self.dt_jr_problem  # Cached problem definition

        # Get driver coefficients if available
        dt_jr_driver_coeffs = None
        if self.dt_jr_driver is not None:
            dt_jr_driver_coeffs = to_numpy(self.dt_jr_driver.coeffs)

        # Term 1 (Physics): K - L * dt_jr_driver
        # Delegate to toroidal_matrices
        E_coeffs = asarray(E_known)
        rhs_1 = self.toroidal_matrices.compute_forcing_vector(E_coeffs, dt_jr_driver_coeffs)

        # Term 2 (Constraint): Unified Apex Current Mapping (Driver + IH)
        # We want (jr_map_sim @ x) = rhs.
        # rhs = j_apex_driver_rate in HL, 0 in Gap (IH difference).
        constraint_weight = 1000.0
        n_grid = self.geometry.grid.size
        rhs_2 = np.zeros(n_grid)

        if dt_jr_driver_coeffs is not None:
            # 1. Get transformation: radial to apex
            rad_to_apx, _ = self.geometry._apex_mapper.get_transformation_matrices(
                self.geometry.b_field
            )

            # 2. Map driver to apex current rate
            # j_apex = (rad_to_apx) * j_r
            driver_jr_grid = self.geometry.basis.evaluate(
                dt_jr_driver_coeffs, self.geometry.grid, "scalar"
            )
            driver_j_apex = rad_to_apx * to_numpy(driver_jr_grid)

            # 3. Apply to HL region only
            hl_mask = ~self.geometry.ll_mask
            rhs_2[hl_mask] = driver_j_apex[hl_mask]

        # Scale for penalty
        rhs_2 = constraint_weight * rhs_2

        # Solve
        rhs_entries = [rhs_1, rhs_2]
        solver = self.m_imp_solver
        solution = solver.solve(problem, rhs_entries)

        if solution is None:
            return xp.zeros(self.solution_basis.index_length)

        return asarray(solution)

    def evolve_psi(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Evolve psi forward in time."""
        # Simple Euler: psi_new = psi + dt * d_psi_dt
        if self.d_psi_dt is None:
             return psi
            
        new_psi = asarray(psi) + dt * asarray(self.d_psi_dt)
        return new_psi

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
            # Op is ConstraintOperator (Rank agnostic)
            E_map_op = self.geometry.E_coeffs_to_E_apex_ll_diff
            
            # Use apply() which handles reshaping and contraction internally
            # E_direct_dense has shape (N, Batch=N) (from to_dense presumably)
            # Or (2, L, N) if previously reshaped?
            # Let's trust E_direct_dense valid shape for now, but be careful.
            
            # If E_direct_dense was reshaped to (2, n, n) previously (line 968),
            # it implies n is NOT index_length? Or index_length = 2 * something?
            # Assuming ConstraintOperator handles (N, Batch) or (2, L, Batch).
            

            if hasattr(E_map_op, "apply"):
                 term = E_map_op.apply(E_direct_dense)
            else:
                 # Fallback logic if somehow raw array
                 E_map_op = asarray(E_map_op)
                 if E_map_op.ndim == 4:
                      # Manually reshape dense input if needed?
                      # Assume E_direct_dense is (2, L, N)
                      term = -xp.tensordot(E_map_op, E_direct_dense, axes=([2, 3], [0, 1]))
                 else:
                      term = -xp.tensordot(E_map_op, E_direct_dense, axes=([2], [0]))
            
            b_E_block = term
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
