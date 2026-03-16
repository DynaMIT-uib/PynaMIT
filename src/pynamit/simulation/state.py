"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required
for simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import hashlib
import logging
import os
import time
from typing import Optional, Tuple, Any, List, Dict, Literal, Callable, TYPE_CHECKING

import numpy as np
from functools import cached_property

from pynamit.primitives.field import Field
from pynamit.primitives.basis import is_cs_basis
from pynamit.primitives.field_spec import FieldSpec
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.integration import (
    EulerIntegrator,
    ExponentialIntegrator,
    ScipySolveIVPIntegrator,
)
from pynamit.math.constants import mu0

from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.simulation.core import (
    CoupledOperators,
    DtAlphaConstraintSystem,
    StateDiagnostics,
    StateConstraints,
    StateInduction,
)
from pynamit.simulation.induction import ToroidalSystemMatrices
from pynamit.simulation.induction.toroidal_closure import ToroidalRMBoundaryOperators
from pynamit.simulation.spatial import Geometry
from pynamit.primitives.basis import Basis
from pynamit.utils import asarray, xp, to_numpy
from pynamit.simulation.spatial import to_dense, canonicalize_vector_basis_matrix

if TYPE_CHECKING:
    from pynamit.simulation.induction.poloidal_solver import MImpFeedbackSystem
from pynamit.simulation.settings import (
    DynamicsMode,
    DynamicsSettings,
    IntegratorKind,
    LLConstraintMode,
    SimulationMode,
    StabilizationPolicy,
)
from pynamit.simulation.input import decode_conductance_representation_to_grids

logger = logging.getLogger(__name__)


def _timed_solve(label: str, solver: LeastSquaresSolver, *args: Any, **kwargs: Any) -> np.ndarray:
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


def _timed_structured_solve(
    label: str, solve_system: Any, solver: LeastSquaresSolver, *args: Any, **kwargs: Any
) -> np.ndarray:
    """Optionally time structured subproblem solves when timing is enabled."""
    kwargs.setdefault("warning_label", label)
    if os.getenv("PYNAMIT_TIMING_SOLVES", "").strip() in ("", "0"):
        return solve_system.solve(solver, *args, **kwargs)
    t0 = time.perf_counter()
    out = solve_system.solve(solver, *args, **kwargs)
    dt = time.perf_counter() - t0
    solver_name = getattr(solver, "solver", "unknown")
    print(f"TIMING solve[{label}] ({solver_name}): {dt:.3f}s", flush=True)
    return out


def _available_memory_bytes() -> Optional[int]:
    """Best-effort estimate of currently available host memory in bytes."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * avail_pages
    except (ValueError, OSError, AttributeError):
        return None


class State:
    """Manages the ionospheric electrodynamic state.

    This class encapsulates the physical state (e.g., potentials,
    currents), handles the construction of all necessary numerical
    operators based on the provided geometry and settings, and
    orchestrates the time evolution of the system. It uses a Geometry
    object to manage the underlying grid and mappings.
    """

    _STABILIZATION_CONTEXTS: tuple[str, ...] = (
        "legacy_m_imp_feedback",
        "legacy_scalar_steady_state",
        "full_induction_toroidal",
        "full_induction_coupled_steady_state",
    )
    _AUTO_REGULARIZED_CONTEXTS: frozenset[str] = frozenset(
        {"full_induction_toroidal", "full_induction_coupled_steady_state"}
    )
    _AUTO_STEADY_STATE_REGULARIZATION_CONTEXTS: frozenset[str] = frozenset(
        {"full_induction_coupled_steady_state"}
    )
    _STEADY_STATE_REGULARIZATION_CONTEXTS: frozenset[str] = frozenset(
        {"legacy_scalar_steady_state", "full_induction_coupled_steady_state"}
    )

    def __init__(
        self,
        basis: Basis,
        mainfield: Any,
        grid_basis: Any,
        settings: Any,
        PFAC_matrix: Optional[np.ndarray] = None,
        solution_space: Optional[Any] = None,
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
        solution_space : Any, optional
            The basis for solution variables.
        """
        self.basis = basis
        self.solution_space = solution_space if solution_space is not None else basis
        self._init_settings(settings)

        # Toroidal magnetic scalar (inductive), analogous to m_imp but time-evolved.
        self.psi: Optional[np.ndarray] = None
        self.d_psi_dt: Optional[np.ndarray] = None
        # Imposed toroidal magnetic scalar from driver (quasi-static baseline).
        self.m_imp_imposed: Optional[np.ndarray] = None
        self._imposed_toroidal_dirty: bool = True
        self._coupled_null_basis: Optional[np.ndarray] = None
        self._coupled_null_threshold: Optional[float] = None
        self._coupled_null_signature: Optional[tuple[int, int, float, str]] = None
        self._coupled_null_warned: bool = False

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(
            basis, grid_basis, mainfield, settings, PFAC_matrix, solution_space=self.solution_space
        )

        self.constraints = StateConstraints(
            geometry=self.geometry,
            solution_space=self.solution_space,
            dynamics_mode=self.dynamics_mode,
            connect_hemispheres=self.connect_hemispheres,
            apply_psi_gauge=self.apply_psi_gauge,
            apply_m_ind_gauge=self.apply_m_ind_gauge,
            apply_m_imp_gauge=self.apply_m_imp_gauge,
        )

        # Initialize Toroidal System Matrices if in full_induction mode
        self.toroidal_matrices: Optional[ToroidalSystemMatrices] = None
        if self.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            closure_basis = self.geometry.pfac_closure_basis
            toroidal_preconditioner = self.get_effective_least_squares_preconditioner(
                "full_induction_toroidal"
            )
            self.toroidal_matrices = ToroidalSystemMatrices(
                basis=self.solution_space,
                grid=self.geometry.grid,
                b_field=self.geometry.b_field,
                RI=self.RI,
                RM=self.RM,
                closure_derivative_basis=closure_basis,
                rhs_derivative_basis=closure_basis,
                radial_derivative_basis=closure_basis,
            )
            try:
                self.toroidal_matrices.configure_toroidal_solver(
                    solver=self.solver_type,
                    preconditioner=toroidal_preconditioner,
                    tolerance=1e-13,
                )
            except ValueError:
                logger.warning(
                    "Invalid least_squares_solver=%r for toroidal solver; "
                    "falling back to 'normal_eq'.",
                    self.solver_type,
                )
                self.toroidal_matrices.configure_toroidal_solver(
                    solver="normal_eq", preconditioner=toroidal_preconditioner, tolerance=1e-13
                )
            if self.mode == SimulationMode.CS_DOMINANT:
                logger.info(
                    "Using SH auxiliary closure basis for toroidal operator assembly "
                    "in cs_dominant/full_induction."
                )

        # The solver is configured here but remains stateless.
        self.m_imp_solver = LeastSquaresSolver(
            solver=self.solver_type,
            preconditioner=self.get_effective_least_squares_preconditioner(
                "legacy_m_imp_feedback"
            ),
        )

        # Initialize state variables
        self.u: Optional[Field] = None
        self.Br: Optional[Field] = None
        self.jr: Optional[Field] = None
        # Full-induction driver-rate representation in toroidal magnetic space.
        self.dt_m_imp_driver: Optional[Field] = None

        self.etaP: Optional[Field] = None
        self.etaH: Optional[Field] = None

        # State tracking
        self.previous_input_data = {}
        self.previous_input_time: Dict[str, float] = {}

        # Invalidate all caches
        self._invalidate_caches()

    @property
    def poloidal_matrices(self) -> Any:
        """The poloidal system matrices."""
        return self.geometry.poloidal_matrices

    @cached_property
    def induction(self) -> StateInduction:
        """Induction/coupled orchestration helper layered on top of `State`."""
        return StateInduction(
            self,
            timed_solve=_timed_solve,
            timed_structured_solve=_timed_structured_solve,
            available_memory_bytes=_available_memory_bytes,
        )

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        normalized_settings = DynamicsSettings.coerce(settings)
        self.settings = normalized_settings

        self.Nmax = int(normalized_settings.Nmax)
        self.Mmax = int(normalized_settings.Mmax)
        self.Ncs = int(normalized_settings.Ncs)
        self.solver_type = normalized_settings.least_squares_solver
        self.preconditioner = normalized_settings.least_squares_preconditioner
        self.stabilization_policy = normalized_settings.stabilization_policy
        self.integrator = normalized_settings.integrator
        self.m_imp_regularization_lambda = normalized_settings.m_imp_regularization_lambda
        self.steady_state_regularization_lambda = (
            normalized_settings.steady_state_regularization_lambda
        )
        self.RI = normalized_settings.RI
        self.RM = normalized_settings.RM
        self.ih_constraint_scaling = normalized_settings.ih_constraint_scaling
        self.ll_constraint_mode = normalized_settings.ll_constraint_mode
        self.apply_psi_gauge = bool(normalized_settings.apply_psi_gauge)
        self.induction_null_diagnostics = bool(normalized_settings.induction_null_diagnostics)
        self.induction_null_svd_rtol = float(normalized_settings.induction_null_svd_rtol)
        self.induction_null_warn_ratio = float(normalized_settings.induction_null_warn_ratio)
        self.apply_m_ind_gauge = bool(normalized_settings.apply_m_ind_gauge)
        self.apply_m_imp_gauge = bool(normalized_settings.apply_m_imp_gauge)
        self.magnetospheric_shielding = bool(normalized_settings.magnetospheric_shielding)
        self.conductance_interpolation_mode = normalized_settings.conductance_interpolation_mode
        self.conductance_interpolation_floor = float(
            normalized_settings.conductance_interpolation_floor
        )
        self.toroidal_regularization_lambda = normalized_settings.toroidal_regularization_lambda
        self.dense_full_operators = bool(normalized_settings.dense_full_operators)
        self.exponential_solver = normalized_settings.exponential_solver
        self.connect_hemispheres = bool(normalized_settings.connect_hemispheres)
        self.dynamics_mode = normalized_settings.dynamics_mode
        self.toroidal_weighting = normalized_settings.toroidal_weighting
        self.poloidal_weighting = normalized_settings.poloidal_weighting

        # Mode Handling
        self.mode = normalized_settings.simulation_mode

        # CS-dominant full-induction is sensitive when SH truncation approaches
        # the CS native Nyquist range; warn early to keep operational bandwidth safe.
        if (
            self.dynamics_mode == DynamicsMode.FULL_INDUCTION
            and self.mode == SimulationMode.CS_DOMINANT
            and self.Ncs > 0
        ):
            spectral_ratio = max(self.Nmax, self.Mmax) / float(self.Ncs)
            if spectral_ratio > 0.7:
                logger.warning(
                    "CS dominant full_induction is operating near Nyquist: "
                    "max(Nmax,Mmax)/Ncs=%.2f (Nmax=%d, Mmax=%d, Ncs=%d). "
                    "For robust toroidal radial forcing, prefer <= 0.6.",
                    spectral_ratio,
                    self.Nmax,
                    self.Mmax,
                    self.Ncs,
                )

        if self.integrator == IntegratorKind.EXPONENTIAL:
            self.poloidal_integrator = ExponentialIntegrator()
        elif self.integrator == IntegratorKind.EULER:
            self.poloidal_integrator = EulerIntegrator()
        else:
            # Assume it's a scipy method (DOP853, RK45, etc.)
            self.poloidal_integrator = ScipySolveIVPIntegrator(method=self.integrator)

    def _normalize_stabilization_context(self, context: str) -> str:
        """Return one canonical stabilization context identifier."""
        normalized = str(context).strip().lower()
        if normalized not in self._STABILIZATION_CONTEXTS:
            raise ValueError(
                f"Unknown stabilization context {context!r}. "
                f"Valid options: {list(self._STABILIZATION_CONTEXTS)!r}."
            )
        return normalized

    def get_effective_least_squares_preconditioner(self, context: str) -> Optional[str]:
        """Return the effective preconditioner policy for one solve context."""
        normalized = self._normalize_stabilization_context(context)
        if self.stabilization_policy == StabilizationPolicy.PRECONDITIONED:
            return self.preconditioner
        if self.stabilization_policy == StabilizationPolicy.REGULARIZED:
            return None
        if normalized in self._AUTO_REGULARIZED_CONTEXTS:
            return None
        return self.preconditioner

    def get_effective_steady_state_regularization_lambda(self, context: str) -> float:
        """Return the effective shared steady-state regularization for one context."""
        normalized = self._normalize_stabilization_context(context)
        if normalized not in self._STEADY_STATE_REGULARIZATION_CONTEXTS:
            return 0.0
        if self.stabilization_policy == StabilizationPolicy.PRECONDITIONED:
            return 0.0
        if self.stabilization_policy == StabilizationPolicy.REGULARIZED:
            return float(self.steady_state_regularization_lambda)
        if normalized not in self._AUTO_STEADY_STATE_REGULARIZATION_CONTEXTS:
            return 0.0
        return float(self.steady_state_regularization_lambda)

    def _create_u_to_E_operator(self) -> np.ndarray:
        """Operator mapping wind coefficients to E coefficients.

        Calculates M such that E_coeffs = M @ u_coeffs.
        E = u x B.
        Logic: v = u x B.
        Geometry.bu provides the cross-product tensor B_x such that v = B_x @ u.
        bu has shape (2, 2, Spatial...).
        """
        bu = asarray(self.geometry.bu)
        G_raw = canonicalize_vector_basis_matrix(
            self.geometry.basis.get_vector_basis_matrix(self.geometry.grid),
            basis_index_length=self.geometry.basis.index_length,
        )

        # 1. Normalize shapes

        # bu is expected as flattened grid tensor: (2, 2, N_grid)
        if bu.ndim != 3:
            raise ValueError(
                f"Unexpected bu shape {bu.shape}; expected canonical flattened (2, 2, N_grid)."
            )
        bu_flat = bu

        n_grid = bu_flat.shape[2]

        # G_raw: (Component, N_grid, PotentialType, Coeffs)
        # Flatten potential/coefficient axes for linear algebra below.
        G_flat = G_raw.reshape(G_raw.shape[0], G_raw.shape[1], -1)
        if G_flat.shape[1] != n_grid:
            raise ValueError(
                f"Grid size mismatch between bu ({n_grid}) and vector basis ({G_flat.shape[1]})."
            )

        if G_flat.shape[0] != bu_flat.shape[1]:
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

    @cached_property
    def u_coeffs_to_E_coeffs(self) -> np.ndarray:
        """Operator mapping wind coefficients to E coefficients.

        Built lazily because many runs (including toroidal benchmarks) have no
        wind forcing and do not require this expensive projection.
        """
        return self._create_u_to_E_operator()

    def _invalidate_caches(self) -> None:
        """Invalidate all conductance-dependent cached properties."""
        # The bundled m_imp feedback system depends on conductance only if we
        # include the legacy E-field interhemispheric constraint. In
        # full_induction or when IH coupling is disabled, it is conductance-
        # independent.
        invalidate_m_imp = (
            self.connect_hemispheres
            and self.dynamics_mode != DynamicsMode.FULL_INDUCTION
            and self.geometry.E_coeffs_to_E_apex_ll_diff is not None
        )

        invalidate_attrs = [
            "M_total_on_grid",
            "m_ind_to_E_coeffs",
            "m_imp_to_E_coeffs",
            "toroidal_to_E_coeffs",
            "Br_to_E_coeffs",
            "E_map_constraint_operator",
            "m_ind_to_E_df_matrix",
            "E_coeffs_to_E_df_matrix",
            "coupled_induction_tensor",
            "coupled_induction_operator_sparse",
            "coupled_induction_matrix_dense",
            "coupled_induction_blocks_dense",
            "coupled_preconditioner",
        ]

        if invalidate_m_imp:
            invalidate_attrs.extend(["m_imp_feedback_system"])

        for attr in invalidate_attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self._operator_linear_map_cache: Dict[
            Tuple[int, Tuple[int, ...], Tuple[int, ...]], Any
        ] = {}
        self._coupled_steady_state_column_scale_cache: Dict[Tuple[bool, int], np.ndarray] = {}
        self._coupled_null_basis = None
        self._coupled_null_threshold = None
        self._coupled_null_signature = None
        self._coupled_null_warned = False
        if "diagnostics" in self.__dict__:
            self.diagnostics.reset_stability_warnings()

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

    @cached_property
    def diagnostics(self) -> StateDiagnostics:
        """Runtime diagnostics helper derived from the live state."""
        return StateDiagnostics(self)

    def get_coupled_stability_report(
        self, *, source: Literal["dense", "sparse", "auto"] = "dense"
    ) -> Dict[str, float]:
        """Return spectral stability report for the coupled full-induction operator."""
        return self.diagnostics.get_coupled_stability_report(source=source)

    def get_toroidal_driver_balance_report(self) -> Dict[str, Any]:
        """Return LL-compatibility diagnostics for live toroidal forcing channels."""
        return self.diagnostics.get_toroidal_driver_balance_report()

    def get_magnetospheric_boundary_report(self) -> Dict[str, Any]:
        """Return induced boundary diagnostics at ``R_M``."""
        return self.diagnostics.get_magnetospheric_boundary_report()

    def get_effective_ll_constraint_mode(self) -> LLConstraintMode:
        """Resolve the configured LL compatibility policy for the active dynamics mode."""
        mode = LLConstraintMode(self.ll_constraint_mode)
        if mode != LLConstraintMode.AUTO:
            return mode
        if not self.connect_hemispheres:
            return LLConstraintMode.OFF
        if self.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            return LLConstraintMode.HARD
        return LLConstraintMode.SOFT

    @cached_property
    def dt_alpha_constraint_system(self) -> DtAlphaConstraintSystem:
        """Bundle active LL compatibility handling for the full-induction ``dt_alpha`` solve."""
        return self.constraints.build_dt_alpha_constraint_system(
            ll_mode=self.get_effective_ll_constraint_mode(),
            soft_scaling=float(self.ih_constraint_scaling),
        )

    @cached_property
    def toroidal_rm_boundary_operators(self) -> ToroidalRMBoundaryOperators:
        """Explicit toroidal boundary operator at ``R_M``.

        This captures the physically unambiguous source-side part:
        - field-line mapping of ``dt_alpha`` to the incoming normal current at ``R_M``,
        - divergent closure current on the ``R_M`` boundary,
        - and the induced toroidal scalar just below ``R_M``.

        It is not a runtime lock operator, because no downward continuation or
        Green response back to ``R_I`` is implied here.
        """
        n_sol = int(self.solution_space.index_length)
        if self.RM in (None, 0):
            zeros = np.zeros((n_sol, n_sol), dtype=float)
            return ToroidalRMBoundaryOperators(alpha_to_boundary_psi_rm=zeros)
        rm_ops = self.poloidal_matrices.toroidal_rm_closure_operators
        lift = self.poloidal_matrices.lift_closure_scalar_output_operator_to_solution
        return ToroidalRMBoundaryOperators(
            alpha_to_boundary_psi_rm=np.asarray(
                lift(rm_ops.alpha_to_boundary_psi_rm_coeff), dtype=float
            )
        )

    @cached_property
    def poloidal_rm_boundary_operators(self):
        """Induced poloidal boundary operators just above ``R_M``."""
        return self.poloidal_matrices.poloidal_rm_boundary_operators

    def _get_dt_alpha_driver_coeffs(self) -> Optional[np.ndarray]:
        """Return ``dt_alpha`` driver from the imposed toroidal-source derivative."""
        if self.dynamics_mode != DynamicsMode.FULL_INDUCTION or self.dt_m_imp_driver is None:
            return None
        dt_m_imp = np.asarray(asarray(self.dt_m_imp_driver.coeffs).reshape(-1))
        m_imp_to_jr = as_linear_map(self.poloidal_matrices.m_imp_to_jr)
        jr_to_alpha = as_linear_map(self.toroidal_matrices.jr_to_alpha_coeff_operator)
        return asarray(jr_to_alpha.matvec(m_imp_to_jr.matvec(dt_m_imp))).reshape(-1)

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
        if b_stacked.ndim > 4:  # e.g. (S, T1, T2, Lat, Lon)
            s, t1, t2 = b_stacked.shape[:3]
            b_flat = b_stacked.reshape(s, t1, t2, -1)
        else:  # (S, T1, T2, Grid)
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

    def _create_E_coeffs_operator(self, potential_type: str) -> Optional[LinearMap]:
        """Unified operator mapping magnetic-scalar coefficients to E coefficients.

        Parameters
        ----------
        potential_type : str
            Magnetic scalar type: "m_imp", "m_ind", or "Br".
        """
        # Pass eta fields if available for analytic mode
        etaP_field = getattr(self, "etaP", None)
        etaH_field = getattr(self, "etaH", None)

        op = self.geometry.get_potential_to_E_coeffs_operator(
            mode=self.mode,
            potential_type=potential_type,
            eta_grid=self.M_total_on_grid,
            etaP=etaP_field,
            etaH=etaH_field,
        )
        if op is None:
            return None
        return as_linear_map(op).with_spaces(
            domain_space=f"{potential_type}_coeffs",
            codomain_space="E_coeffs",
        )

    @cached_property
    def m_ind_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping m_ind coefficients to E coefficients."""
        return self._create_E_coeffs_operator("m_ind")

    @cached_property
    def m_imp_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping m_imp coefficients to E coefficients."""
        return self._create_E_coeffs_operator("m_imp")

    @cached_property
    def toroidal_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping dynamic toroidal magnetic scalar (psi) to E coefficients."""
        op = self._create_E_coeffs_operator("psi")
        if op is None:
            return self.m_imp_to_E_coeffs
        return op

    @cached_property
    def Br_to_E_coeffs(self) -> Optional[LinearMap]:
        """Operator mapping Br coefficients to E coefficients."""
        return self._create_E_coeffs_operator("Br")

    @cached_property
    def E_map_constraint_operator(self) -> Optional[LinearMap]:
        """Operator enforcing E-field mapping at low latitudes."""
        # Tensor shape is canonical (2, n_mask, 2, n_coeffs).
        op_obj = self.geometry.E_coeffs_to_E_apex_ll_diff

        if op_obj is None:
            return None

        # Extract underlying tensor if wrapped
        if hasattr(op_obj, "tensor"):
            outer_t = asarray(op_obj.tensor)
        else:
            outer_t = asarray(op_obj)

        if outer_t.ndim != 4 or outer_t.shape[2] != 2:
            raise ValueError(
                "Constraint tensor must have canonical shape (2, n_mask, 2, n_coeffs), "
                f"got {outer_t.shape}."
            )
        n_mask, n_in = outer_t.shape[1], outer_t.shape[3]
        op_outer = as_linear_map(outer_t.reshape(2 * n_mask, 2 * n_in)).with_spaces(
            domain_space="E_coeffs",
            codomain_space="E_ll_constraint",
        )

        # Inner: m_imp -> E-field
        op_inner = self.m_imp_to_E_coeffs
        if op_inner is None:
            return None

        # Composition: Constraint @ (m_imp -> E)
        return op_outer @ op_inner

    # ----- Solver Setup and Execution -----
    @cached_property
    def m_imp_feedback_system(self) -> "MImpFeedbackSystem":
        """Bundled reduced ``m_imp`` feedback solve definition."""
        return self.induction.build_m_imp_feedback_system()

    @property
    def m_imp_problem(self) -> LeastSquaresProblem:
        """Compatibility view of the bundled ``m_imp`` least-squares problem."""
        return self.m_imp_feedback_system.problem

    @property
    def m_imp_preconditioner(self) -> Optional[LinearMap]:
        """Compatibility view of the bundled ``m_imp`` preconditioner."""
        return self.m_imp_feedback_system.preconditioner

    @cached_property
    def coupled_preconditioner(self) -> Optional[LinearMap]:
        """Preconditioner for the coupled (2N, 2N) induction system.

        Uses the standard LeastSquaresProblem/LeastSquaresSolver construction.
        """
        return self.coupled_operators.build_coupled_preconditioner()

    def _build_imposed_toroidal_baseline(
        self, jr_coeffs: Optional[np.ndarray], E_direct_coeffs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Build imposed toroidal baseline `m_imp` from external driver inputs.

        Full-induction path: direct `jr -> m_imp` map in solution space.
        Legacy path: constrained least-squares solve for imposed baseline.
        """
        return self.induction.build_imposed_toroidal_baseline(jr_coeffs, E_direct_coeffs)

    def _map_dt_jr_driver_to_dt_m_imp(self, dt_jr_coeffs: np.ndarray) -> np.ndarray:
        """Map driver derivative ``dt_jr`` to toroidal driver derivative ``dt_m_imp``."""
        return self.induction.map_dt_jr_driver_to_dt_m_imp(dt_jr_coeffs)

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
        if field.basis is self.solution_space or field.basis == self.solution_space:
            return field

        # Handle projection to CS/Nodal basis
        if is_cs_basis(self.solution_space):
            grid = self.geometry.grid
            # Evaluate on grid
            v1, v2, v3 = field.evaluate(self.geometry.RI, grid.theta, grid.phi)

            if field_type == "scalar":
                return Field.from_coefficients(
                    self.solution_space, coeffs=asarray(v1).flatten(), field_type="scalar"
                )
            elif field_type == "tangential":
                # u (wind): theta, phi components
                v2_flat = asarray(v2).flatten()
                v3_flat = asarray(v3).flatten()
                new_coeffs = xp.stack([v2_flat, v3_flat], axis=0)
                return Field.from_coefficients(
                    self.solution_space, coeffs=new_coeffs, field_type="tangential"
                )

        return field

    def _decode_conductance_input_to_eta_coeffs(
        self, *, storage_spec: FieldSpec, updated_input: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode conductance input representation and return ``(etaP, etaH)`` coeffs.

        Supports:
        - legacy resistivity interpolation: ``etaP``, ``etaH`` (pass-through)
        - conductivity interpolation: ``SigmaP``, ``SigmaH``
        - log-conductivity interpolation: ``logSigmaP``, ``logSigmaH``
        """
        if "etaP" in updated_input and "etaH" in updated_input:
            return (
                np.asarray(updated_input["etaP"]).reshape(-1),
                np.asarray(updated_input["etaH"]).reshape(-1),
            )

        grid = self.geometry.grid
        r_eval = self.geometry.RI
        sigma_floor = float(max(self.conductance_interpolation_floor, 0.0))
        target_shape = tuple(np.asarray(grid.theta).shape)

        def _eval_scalar(coeffs: np.ndarray) -> np.ndarray:
            field = Field.from_coefficients(storage_spec, coeffs=coeffs)
            vals, _, _ = field.evaluate(r_eval, grid.theta, grid.phi)
            return np.asarray(vals, dtype=float).reshape(target_shape)

        _, _, etaP_grid, etaH_grid = decode_conductance_representation_to_grids(
            data=updated_input,
            eval_scalar_coeffs_to_grid=_eval_scalar,
            sigma_floor=sigma_floor,
            logger=logger,
        )
        etaP_grid = np.asarray(etaP_grid, dtype=float).reshape(-1)
        etaH_grid = np.asarray(etaH_grid, dtype=float).reshape(-1)

        etaP_coeffs = np.asarray(storage_spec.from_grid_values(etaP_grid, grid, "scalar")).reshape(
            -1
        )
        etaH_coeffs = np.asarray(storage_spec.from_grid_values(etaH_grid, grid, "scalar")).reshape(
            -1
        )
        return etaP_coeffs, etaH_coeffs

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

            prev_data_for_key = self.previous_input_data.get(key)
            prev_time_for_key = self.previous_input_time.get(key)

            # Update cache and proceed
            updated_input = current_data

            # Check for derivatives
            current_deriv = None
            if key == "jr" and hasattr(input_manager, "get_entry_with_derivative"):
                # This assumes InputManager has been updated to provide derivatives methods.
                # We need interpolation to get correct derivatives from sparse time points
                _, current_deriv = input_manager.get_entry_with_derivative(
                    key, time, interpolation=True
                )
                # Fallback: finite-difference derivative from the previously seen jr input.
                if (
                    current_deriv is None
                    and prev_data_for_key is not None
                    and prev_time_for_key is not None
                    and "jr" in prev_data_for_key
                    and np.isfinite(prev_time_for_key)
                    and time > prev_time_for_key
                ):
                    dt = float(time - prev_time_for_key)
                    jr_prev = np.asarray(prev_data_for_key["jr"])
                    jr_curr = np.asarray(updated_input["jr"])
                    if jr_prev.shape == jr_curr.shape and dt > 0.0:
                        current_deriv = {"jr": (jr_curr - jr_prev) / dt}

            storage_spec = input_manager.timeseries.get_storage_spec(key)
            if key == "conductance":
                conductance_updated = True
                etaP_coeffs, etaH_coeffs = self._decode_conductance_input_to_eta_coeffs(
                    storage_spec=storage_spec, updated_input=updated_input
                )
                f_etaP = Field.from_coefficients(storage_spec, coeffs=etaP_coeffs)
                f_etaH = Field.from_coefficients(storage_spec, coeffs=etaH_coeffs)
                self.etaP = self._ensure_basis(f_etaP, "scalar")
                self.etaH = self._ensure_basis(f_etaH, "scalar")
            elif key == "jr":
                f_jr = Field.from_coefficients(storage_spec, coeffs=updated_input["jr"])
                self.jr = self._ensure_basis(f_jr, "scalar")
                # Driver changed: rebuild imposed toroidal baseline on next use.
                self._imposed_toroidal_dirty = True
                if current_deriv is not None:
                    if self.dynamics_mode == DynamicsMode.FULL_INDUCTION:
                        f_dt_jr = Field.from_coefficients(storage_spec, coeffs=current_deriv["jr"])
                        dt_jr_solution = self._ensure_basis(f_dt_jr, "scalar")
                        dt_m_imp_coeffs = self._map_dt_jr_driver_to_dt_m_imp(
                            dt_jr_coeffs=asarray(dt_jr_solution.coeffs)
                        )
                        self.dt_m_imp_driver = Field.from_coefficients(
                            self.solution_space,
                            coeffs=asarray(dt_m_imp_coeffs),
                            field_type="scalar",
                        )
                    else:
                        self.dt_m_imp_driver = None
                else:
                    self.dt_m_imp_driver = None
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None.")
                f_Br = Field.from_coefficients(storage_spec, coeffs=updated_input["Br"])
                self.Br = self._ensure_basis(f_Br, "scalar")
            elif key == "u":
                f_u = Field.from_coefficients(
                    storage_spec,
                    coeffs=updated_input["u"].reshape((2, -1)),
                    field_type="tangential",
                )
                self.u = self._ensure_basis(f_u, "tangential")

            # Persist latest snapshot/time for finite-difference derivative estimates.
            self.previous_input_data[key] = current_data
            self.previous_input_time[key] = float(time)

        if conductance_updated:
            logger.info("Conductance updated: invalidating caches and problem definition.")
            self._invalidate_caches()

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
        return self.induction._calculate_total_E_field(E_direct_coeffs, jr_coeffs)

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction effects."""
        return self.induction.calculate_noind_coeffs()

    def _calculate_dynamic_state(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble non-inductive forcing and update toroidal residual rate."""
        return self.induction._calculate_dynamic_state(E_direct_coeffs, jr_coeffs)

    def solve_dt_psi(self, E_known: np.ndarray) -> np.ndarray:
        """Solve constrained system for dpsi/dt."""
        return self.induction.solve_dt_psi(E_known)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total E-field coefficients."""
        return self.induction.calculate_ind_coeffs(m_ind)

    def calculate_psi_E_coeffs(self, psi: np.ndarray) -> np.ndarray:
        """Map inductive toroidal residual psi to E-field coefficients."""
        return self.induction.calculate_psi_E_coeffs(psi)

    # ----- Time Evolution -----

    @cached_property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix mapping m_ind to div-free E-field."""
        return self.induction.build_m_ind_to_E_df_matrix()

    @cached_property
    def E_coeffs_to_E_df_matrix(self) -> np.ndarray:
        """Operator extracting toroidal potential (E_df) from vector coefficients."""
        return self.induction.build_E_coeffs_to_E_df_matrix()

    def get_induction_operator(self) -> "LinearMap":
        """Get matrix-free induction operator (m_ind -> E_df).

        Returns a LinearMap for matrix-free steady-state computation.
        More efficient than building the dense matrix for large systems.
        """
        return self.induction.get_induction_operator()

    # _build_m_ind_to_E_df_matrix refactored to PoloidalSystemMatrices.build_induction_matrix

    # _calculate_d_m_ind_dt refactored to PoloidalSystemMatrices.compute_rates

    def _apply_state_linear_operator(
        self, operator: Any, state: np.ndarray, output_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Apply a state-space linear operator to a flattened/stacked state."""
        return self.induction.apply_state_linear_operator(
            operator, state, output_shape=output_shape
        )

    def _evolve_linear_state(
        self,
        y: np.ndarray,
        dt: float,
        *,
        linear_operator: Optional[Any] = None,
        forcing: Optional[np.ndarray] = None,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        steady_state: Optional[np.ndarray] = None,
        exponential_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Shared linear-state evolution for legacy and full-induction paths."""
        return self.induction.evolve_linear_state(
            y,
            dt,
            linear_operator=linear_operator,
            forcing=forcing,
            rates_func=rates_func,
            steady_state=steady_state,
            exponential_kwargs=exponential_kwargs,
        )

    def _solve_linear_steady_state(
        self,
        *,
        linear_operator: Any,
        forcing: np.ndarray,
        solution_shape: Tuple[int, ...],
        solver: Optional[str] = None,
        preconditioner: Optional[LinearMap] = None,
    ) -> np.ndarray:
        """Solve a linear steady-state system `A x = -forcing` for arbitrary state shape."""
        return self.induction.solve_linear_steady_state(
            linear_operator=linear_operator,
            forcing=forcing,
            solution_shape=solution_shape,
            solver=solver,
            preconditioner=preconditioner,
        )

    def build_coupled_forcing(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Build coupled forcing tensor ``K`` for ``[psi, m_ind]`` dynamics."""
        return self.induction.build_coupled_forcing(E_coeffs_noind)

    def solve_steady_state_model_variables(
        self, E_coeffs_noind: np.ndarray, *, update_state: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Compute steady-state initialization for current dynamics mode."""
        return self.induction.solve_steady_state_model_variables(
            E_coeffs_noind, update_state=update_state
        )

    def evolve_model_variables(
        self,
        m_ind: np.ndarray,
        dt: float,
        E_coeffs_noind: np.ndarray,
        *,
        steady_state_m_ind: Optional[np.ndarray] = None,
        steady_state_psi: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Advance model variables by one time step.

        Returns ``(psi, m_ind)`` where ``psi`` is ``None`` for legacy mode.
        """
        return self.induction.evolve_model_variables(
            m_ind,
            dt,
            E_coeffs_noind,
            steady_state_m_ind=steady_state_m_ind,
            steady_state_psi=steady_state_psi,
            psi=psi,
        )

    # -------------------------------------------------------------------------
    # Coupled Exponential Integrator
    # -------------------------------------------------------------------------

    @cached_property
    def coupled_operators(self) -> CoupledOperators:
        """Internal coupled-operator assembly/exposure helper."""
        return CoupledOperators(self)

    def get_coupled_induction_tensor(self) -> np.ndarray:
        """Build the coupled tensor ``L_coupled`` with shape ``(2, N, 2, N)``."""
        return self.coupled_operators.get_coupled_induction_tensor()

    @cached_property
    def coupled_induction_tensor(self) -> np.ndarray:
        """Default coupled induction tensor (delegates to get_coupled_induction_tensor)."""
        return self.coupled_operators.get_coupled_induction_tensor()

    @cached_property
    def coupled_induction_operator_sparse(self) -> "LinearMap":
        """Cached matrix-free coupled operator for non-exponential stepping."""
        solver = self.solver_type if self.solver_type in ("lsmr", "cgls") else "lsmr"
        return self.coupled_operators.get_coupled_induction_operator(
            matrix_free=True, solver=solver
        )

    @cached_property
    def coupled_induction_matrix_dense(self) -> np.ndarray:
        """Cached dense coupled operator matrix with shape ``(2N, 2N)``."""
        N = self.solution_space.index_length
        return asarray(self.coupled_induction_tensor).reshape(2 * N, 2 * N)

    @cached_property
    def coupled_induction_blocks_dense(self) -> Dict[str, np.ndarray]:
        """Cached dense coupled blocks keyed by physical role."""
        return self.coupled_operators.get_coupled_induction_blocks(source="dense")

    def _densify_linear_operator(self, operator: Any, n_total: int) -> np.ndarray:
        """Convert a linear operator to dense ``(2N, 2N)``."""
        return self.coupled_operators._densify_linear_operator(operator, n_total)

    def get_coupled_induction_matrix(
        self, source: Literal["dense", "sparse", "auto"] = "auto", flatten: bool = True
    ) -> np.ndarray:
        """Expose coupled operator matrix in dense form."""
        return self.coupled_operators.get_coupled_induction_matrix(source=source, flatten=flatten)

    def get_coupled_induction_blocks(
        self, source: Literal["dense", "sparse", "auto"] = "auto"
    ) -> Dict[str, np.ndarray]:
        """Expose coupled block matrices keyed by physical role."""
        return self.coupled_operators.get_coupled_induction_blocks(source=source)

    def get_coupled_operator_for_steady_state(self, *, solver: Optional[str] = None) -> Any:
        """Return coupled operator used by steady-state coupled solve."""
        return self.coupled_operators.get_coupled_operator_for_steady_state(solver=solver)

    def get_coupled_operator_for_time_integration(
        self, *, use_dense: Optional[bool] = None
    ) -> Any:
        """Return the full-space coupled operator assembled for time stepping."""
        return self.coupled_operators.get_coupled_operator_for_time_integration(
            use_dense=use_dense
        )

    def get_coupled_reduced_time_integration_system(
        self, *, use_dense: Optional[bool] = None
    ) -> Any:
        """Return the gauge-reduced coupled system used by runtime time stepping."""
        return self.coupled_operators.get_coupled_reduced_time_integration_system(
            use_dense=use_dense
        )

    def get_m_ind_reduced_system(self, *, linear_operator: Any | None = None) -> Any:
        """Return the gauge-reduced scalar system used by legacy ``m_ind`` stepping."""
        return self.constraints.get_m_ind_reduced_system(linear_operator=linear_operator)

    def get_m_imp_reduced_system(self, *, linear_operator: Any | None = None) -> Any:
        """Return the gauge-reduced scalar system used by imposed ``m_imp`` solves."""
        return self.constraints.get_m_imp_reduced_system(linear_operator=linear_operator)

    def get_m_imp_from_jr_matrix(self, input_basis: Optional[Any] = None) -> np.ndarray:
        """Expose dense linear map from input `jr` coefficients to imposed `m_imp`."""
        return self.coupled_operators.get_m_imp_from_jr_matrix(input_basis=input_basis)

    def get_external_forcing_matrices(
        self, input_basis_jr: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """Expose dense rate maps from `u` and `jr` into the coupled system."""
        return self.coupled_operators.get_external_forcing_matrices(input_basis_jr=input_basis_jr)

    def get_coupled_induction_operator(
        self,
        dt_psi_from_psi: Any = None,
        dt_psi_from_m_ind: Any = None,
        dt_m_ind_from_psi: Any = None,
        dt_m_ind_from_m_ind: Any = None,
        matrix_free: bool = False,
        solver: str = "lsmr",
    ) -> "LinearMap":
        """Build coupled operator for ``y=[psi, m_ind]`` dynamics."""
        return self.coupled_operators.get_coupled_induction_operator(
            dt_psi_from_psi=dt_psi_from_psi,
            dt_psi_from_m_ind=dt_psi_from_m_ind,
            dt_m_ind_from_psi=dt_m_ind_from_psi,
            dt_m_ind_from_m_ind=dt_m_ind_from_m_ind,
            matrix_free=matrix_free,
            solver=solver,
        )

    def _update_coupled_null_basis(self, L_flat: np.ndarray) -> None:
        """Build/update cached near-null basis for coupled-operator diagnostics."""
        if not self.induction_null_diagnostics:
            return
        m = L_flat.shape[0]
        signature = self._get_coupled_null_signature(L_flat)
        if signature == self._coupled_null_signature and self._coupled_null_basis is not None:
            return

        _, svals, vt = np.linalg.svd(np.asarray(L_flat), full_matrices=False)
        if svals.size == 0:
            self._coupled_null_basis = np.zeros((m, 0), dtype=float)
            self._coupled_null_threshold = 0.0
            self._coupled_null_signature = signature
            self._coupled_null_warned = False
            return

        threshold = float(self.induction_null_svd_rtol) * float(svals[0])
        mask = svals < threshold
        basis = vt[mask, :].T if np.any(mask) else np.zeros((m, 0), dtype=vt.dtype)

        self._coupled_null_basis = basis
        self._coupled_null_threshold = threshold
        self._coupled_null_signature = signature
        self._coupled_null_warned = False
        logger.info(
            "Coupled null diagnostic: s_max=%.3e, s_min=%.3e, near_null=%d (rtol=%.1e).",
            float(svals[0]),
            float(svals[-1]),
            int(mask.sum()),
            float(self.induction_null_svd_rtol),
        )

    def _check_forcing_null_projection(self, K_flat: np.ndarray) -> None:
        """Warn when forcing projects strongly onto coupled near-null subspace."""
        if not self.induction_null_diagnostics or self._coupled_null_basis is None:
            return
        V = self._coupled_null_basis
        if V.shape[1] == 0:
            return

        k = np.asarray(K_flat).reshape(-1)
        k_norm = np.linalg.norm(k)
        if k_norm <= 0:
            return
        proj = V @ (V.T @ k)
        ratio = float(np.linalg.norm(proj) / k_norm)

        if ratio >= self.induction_null_warn_ratio and not self._coupled_null_warned:
            logger.warning(
                "Coupled forcing projects strongly onto near-null modes: ratio=%.3f (warn>=%.3f).",
                ratio,
                float(self.induction_null_warn_ratio),
            )
            self._coupled_null_warned = True

    def _get_coupled_null_signature(self, L_flat: np.ndarray) -> tuple[int, int, float, str]:
        """Return a stable fingerprint for the dense coupled operator."""
        dense = np.ascontiguousarray(np.asarray(L_flat, dtype=float))
        digest = hashlib.blake2b(dense.view(np.uint8), digest_size=16).hexdigest()
        return (
            int(dense.shape[0]),
            int(dense.shape[1]),
            float(self.induction_null_svd_rtol),
            digest,
        )

    def run_coupled_null_diagnostics(self, linear_operator: Any, forcing_flat: Any) -> None:
        """Inspect coupled operator/forcing for strong near-null excitation."""
        if not self.induction_null_diagnostics:
            return

        forcing = np.asarray(to_numpy(forcing_flat), dtype=float).reshape(-1)
        n_total = int(forcing.size)
        if n_total == 0:
            return

        avail_bytes = _available_memory_bytes()
        dense_bytes = int(n_total) * int(n_total) * np.dtype(float).itemsize
        estimated_peak_bytes = 6 * dense_bytes
        if avail_bytes is not None and estimated_peak_bytes > int(0.25 * avail_bytes):
            logger.warning(
                "Skipping coupled null diagnostic: estimated SVD memory %.2f GiB exceeds "
                "25%% of available memory %.2f GiB.",
                estimated_peak_bytes / float(1024**3),
                avail_bytes / float(1024**3),
            )
            return

        try:
            L_flat = np.asarray(
                self._densify_linear_operator(linear_operator, n_total), dtype=float
            ).reshape(n_total, n_total)
            self._update_coupled_null_basis(L_flat)
            self._check_forcing_null_projection(forcing)
        except np.linalg.LinAlgError as exc:
            logger.warning("Skipping coupled null diagnostic: SVD failed (%s).", exc)
        except Exception as exc:
            logger.warning("Skipping coupled null diagnostic: %s.", exc)
