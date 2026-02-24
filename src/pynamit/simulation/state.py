"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required
for simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import logging
import os
import time
import warnings
from typing import Optional, Tuple, Any, List, Dict, Literal, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from functools import cached_property

from pynamit.primitives.field import Field
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.integration import EulerIntegrator, ExponentialIntegrator, Integrator, ScipySolveIVPIntegrator

from pynamit.math.linear_map import as_linear_map, LinearMap
from pynamit.simulation.geometry import Geometry
from pynamit.simulation.coupled_solver import CoupledSteadyStateSolver, CoupledOperatorAPI
from pynamit.primitives.basis import Basis
from pynamit.math.constants import mu0
from pynamit.utils import asarray, use_jax, xp, to_numpy, tensor_pinv
from pynamit.simulation.toroidal import ToroidalSystemMatrices
from pynamit.simulation.geometry_utils import to_dense, canonicalize_vector_basis_matrix
from pynamit.simulation.state_constraints import StateConstraints

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
        
        # Toroidal magnetic scalar (inductive), analogous to m_imp but time-evolved.
        self.psi: Optional[np.ndarray] = None
        self.d_psi_dt: Optional[np.ndarray] = None
        # Imposed toroidal magnetic scalar from driver (quasi-static baseline).
        self.m_imp_imposed: Optional[np.ndarray] = None
        self._imposed_toroidal_dirty: bool = True
        self._coupled_null_basis: Optional[np.ndarray] = None
        self._coupled_null_threshold: Optional[float] = None
        self._coupled_null_warned: bool = False
        self._coupled_stability_warned_keys: set[Tuple[Any, ...]] = set()

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(
            basis, grid_basis, mainfield, settings, PFAC_matrix, solution_basis=self.solution_basis
        )

        self.constraints = StateConstraints(
            geometry=self.geometry,
            solution_basis=self.solution_basis,
            dynamics_mode=self.dynamics_mode,
            connect_hemispheres=self.connect_hemispheres,
            magnetospheric_toroidal_lock=self.magnetospheric_toroidal_lock,
            apply_psi_gauge=self.apply_psi_gauge,
            apply_m_ind_gauge=self.apply_m_ind_gauge,
        )

        # Initialize Toroidal System Matrices if in full_induction mode
        self.toroidal_matrices: Optional[ToroidalSystemMatrices] = None
        if self.dynamics_mode == "full_induction":
            closure_basis = self.solution_basis
            if getattr(self.mode, "value", self.mode) == "cs_dominant":
                from pynamit.spherical_harmonics.sh_basis import SHBasis

                closure_basis = SHBasis(self.Nmax, self.Mmax)
            self.toroidal_matrices = ToroidalSystemMatrices(
                basis=self.solution_basis, 
                grid=self.geometry.grid, 
                b_field=self.geometry.b_field,
                RI=self.RI,
                closure_derivative_basis=closure_basis,
                forcing_derivative_basis=closure_basis,
                radial_derivative_basis=closure_basis,
            )
            try:
                self.toroidal_matrices.configure_dtjr_solver(
                    solver=self.solver_type,
                    preconditioner=self.preconditioner,
                    tolerance=1e-13,
                )
            except ValueError:
                logger.warning(
                    "Invalid least_squares_solver=%r for toroidal dt_jr; falling back to 'svd'.",
                    self.solver_type,
                )
                self.toroidal_matrices.configure_dtjr_solver(
                    solver="normal_eq",
                    preconditioner=self.preconditioner,
                    tolerance=1e-13,
                )
            if getattr(self.mode, "value", self.mode) == "cs_dominant":
                logger.info(
                    "Using SH auxiliary closure basis for toroidal operator assembly "
                    "in cs_dominant/full_induction."
                )

        # The solver is configured here but remains stateless.
        self.m_imp_solver = LeastSquaresSolver(
            solver=self.solver_type, preconditioner=self.preconditioner
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

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        self.Nmax = int(getattr(settings, "Nmax", 0))
        self.Mmax = int(getattr(settings, "Mmax", 0))
        self.Ncs = int(getattr(settings, "Ncs", 0))
        self.solver_type = getattr(settings, "least_squares_solver", "cgls")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.static_preconditioner = getattr(settings, "static_preconditioner", False)
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.apply_psi_gauge = bool(getattr(settings, "apply_psi_gauge", True))
        self.induction_null_diagnostics = False
        self.induction_null_svd_rtol = 1e-8
        self.induction_null_warn_ratio = 0.5
        self.apply_m_ind_gauge = bool(getattr(settings, "apply_m_ind_gauge", True))
        self.apply_m_imp_gauge = bool(getattr(settings, "apply_m_imp_gauge", True))
        self.magnetospheric_toroidal_lock = bool(
            getattr(settings, "magnetospheric_toroidal_lock", False)
        )
        self.conductance_interpolation_mode = str(
            getattr(settings, "conductance_interpolation_mode", "legacy_eta_linear")
        )
        self.conductance_interpolation_floor = float(
            max(getattr(settings, "conductance_interpolation_floor", 1e-3), 0.0)
        )
        self.toroidal_regularization_lambda = getattr(settings, "toroidal_regularization_lambda", 0.0)
        self.dense_full_operators = bool(getattr(settings, "dense_full_operators", False))
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.dynamics_mode = getattr(settings, "dynamics_mode", "legacy")
        self.toroidal_weighting = getattr(settings, "toroidal_weighting", "none")
        self.poloidal_weighting = getattr(settings, "poloidal_weighting", "none")

        # Mode Handling
        from pynamit.simulation.settings import SimulationMode
        if hasattr(settings, "simulation_mode"):
            self.mode = settings.simulation_mode
        else:
            # Legacy Fallback
            pure = getattr(settings, "pure_spectral", False)
            self.mode = (
                SimulationMode.PURE_SPECTRAL if pure else SimulationMode.SPECTRAL_TRANSFORM_CS
            )

        # Map mode to legacy flags for internal checks (if any remain)
        self.pure_spectral = (self.mode == SimulationMode.PURE_SPECTRAL)

        # Default to regularization for CS_DOMINANT to handle equatorial singularity in electrostatic problem
        if self.mode == SimulationMode.CS_DOMINANT:
            if self.m_imp_regularization_lambda == 0.0:
                self.m_imp_regularization_lambda = 1e-4

        # Robust defaults for induction feedback loop stability
        # Equatorial singularity (Br=0) makes the toroidal problem ill-conditioned.
        # Quadratic weighting by Br and Tikhonov regularization are essential for stability.
        if self.dynamics_mode == "full_induction":
            if self.toroidal_weighting == "none":
                self.toroidal_weighting = "quadratic"
            if self.toroidal_regularization_lambda == 0.0:
                # Backward-compatible fallback for old serialized settings that
                # still carry lambda=0. Keep this light but stability-safe.
                self.toroidal_regularization_lambda = 1e-10
            if self.poloidal_weighting == "none":
                self.poloidal_weighting = "quadratic"

        # CS-dominant full-induction is sensitive when SH truncation approaches
        # the CS native Nyquist range; warn early to keep operational bandwidth safe.
        if (
            self.dynamics_mode == "full_induction"
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

        if self.integrator == "exponential":
             self.poloidal_integrator = ExponentialIntegrator()
        elif self.integrator == "euler":
             self.poloidal_integrator = EulerIntegrator()
        else:
             # Assume it's a scipy method (DOP853, RK45, etc.)
             self.poloidal_integrator = ScipySolveIVPIntegrator(method=self.integrator)

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
        # m_imp_problem depends on conductance only if we include the E-field
        # interhemispheric constraint (legacy path). In full_induction or when
        # IH coupling is disabled, it is conductance-independent.
        invalidate_m_imp = (
            self.connect_hemispheres
            and self.dynamics_mode != "full_induction"
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
            invalidate_attrs.extend(["m_imp_problem", "m_imp_preconditioner"])

        for attr in invalidate_attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self._operator_linear_map_cache: Dict[
            Tuple[int, Tuple[int, ...], Tuple[int, ...]], Any
        ] = {}
        self._coupled_steady_state_column_scale_cache: Dict[
            Tuple[bool, int], np.ndarray
        ] = {}
        self._coupled_null_basis = None
        self._coupled_null_threshold = None
        self._coupled_null_warned = False
        self._coupled_stability_warned_keys.clear()

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
            jr_map_operator=self.constraints.induction_constraint_operator_hard,
            constraint_scaling=0.0,
            regularization_lambda=self.toroidal_regularization_lambda,
            weighting=self.toroidal_weighting,
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
        if ll_mask is None:
            raise RuntimeError(
                "LL mask is required for full_induction with connect_hemispheres=True."
            )
        row_mask = np.asarray(ll_mask, dtype=bool).reshape(-1)

        op_lm = as_linear_map(op)
        if row_mask.size != op_lm.shape[0]:
            raise RuntimeError(
                "LL mask size mismatch: "
                f"mask={int(row_mask.size)} rows={int(op_lm.shape[0])}."
            )
        if not np.any(row_mask):
            raise RuntimeError(
                "LL mask contains no active rows for full_induction constraints."
            )

        if hasattr(op, "tocsr"):
            return op.tocsr()[row_mask]
        return np.ascontiguousarray(to_dense(op_lm)[row_mask, :])

    def _orthonormalize_columns(self, A: np.ndarray, rtol: float) -> np.ndarray:
        """Return an orthonormal basis for the column space of A."""
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.size == 0:
            n_rows = A.shape[0] if A.ndim == 2 else 0
            return np.zeros((n_rows, 0), dtype=float)
        u, s, _ = np.linalg.svd(A, full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            return np.zeros((A.shape[0], 0), dtype=float)
        thresh = max(float(rtol), 0.0) * float(s[0])
        keep = s > thresh
        if not np.any(keep):
            return np.zeros((A.shape[0], 0), dtype=float)
        return np.ascontiguousarray(u[:, keep])

    def _m_orthonormalize_columns(
        self,
        A: np.ndarray,
        metric: np.ndarray,
        rtol: float,
    ) -> np.ndarray:
        """Return an ``M``-orthonormal basis spanning ``col(A)``.

        The concentration eigenmodes are naturally orthogonal in the magnetic
        energy metric ``M``. We preserve that inner product so split amplitudes
        are represented consistently as ``q^T M x``.
        """
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.size == 0:
            n_rows = A.shape[0] if A.ndim == 2 else 0
            return np.zeros((n_rows, 0), dtype=float)

        M = np.asarray(metric, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != A.shape[0]:
            return self._orthonormalize_columns(A, rtol=rtol)

        G = 0.5 * ((A.T @ M @ A) + (A.T @ M @ A).T)
        try:
            evals, evecs = np.linalg.eigh(G)
        except np.linalg.LinAlgError:
            return self._orthonormalize_columns(A, rtol=rtol)

        if evals.size == 0:
            return np.zeros((A.shape[0], 0), dtype=float)

        order = np.argsort(evals)[::-1]
        evals = np.asarray(evals[order], dtype=float)
        evecs = np.asarray(evecs[:, order], dtype=float)
        max_eval = float(np.max(evals))
        if not np.isfinite(max_eval) or max_eval <= 0:
            return np.zeros((A.shape[0], 0), dtype=float)

        thresh = max(float(rtol), 0.0) * max_eval
        keep = evals > thresh
        if not np.any(keep):
            return np.zeros((A.shape[0], 0), dtype=float)

        scale = np.sqrt(np.maximum(evals[keep], 0.0))
        Q = A @ (evecs[:, keep] / scale.reshape(1, -1))
        return np.ascontiguousarray(Q)

    @staticmethod
    def _normalize_constraint_rows(C: np.ndarray) -> np.ndarray:
        """Row-normalize constraint matrix and drop zero rows."""
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] == 0:
            return np.zeros((0, C.shape[1] if C.ndim == 2 else 0), dtype=float)
        row_norm = np.linalg.norm(C, axis=1)
        keep = row_norm > 0
        if not np.any(keep):
            return np.zeros((0, C.shape[1]), dtype=float)
        C_use = C[keep] / row_norm[keep].reshape(-1, 1)
        return np.ascontiguousarray(C_use)





    def _analyze_coupled_stability(
        self,
        L_flat: np.ndarray,
        *,
        label: str,
        unstable_tol: float = 1e-10,
    ) -> Dict[str, float]:
        """Analyze coupled-operator spectrum and warn on unstable modes."""
        arr = np.asarray(L_flat, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    "Coupled stability analysis requires a square matrix, "
                    f"got {arr.shape}."
                )

        eigvals = np.linalg.eigvals(arr)
        real = np.real(eigvals)
        max_real = float(np.max(real)) if real.size > 0 else 0.0
        min_real = float(np.min(real)) if real.size > 0 else 0.0
        n_pos = int(np.sum(real > float(unstable_tol)))
        n_total = int(real.size)

        report = {
            "max_real": max_real,
            "min_real": min_real,
            "positive_real_count": float(n_pos),
            "n_eigs": float(n_total),
        }

        if max_real > float(unstable_tol):
            key = (
                label,
                arr.shape[0],
                round(max_real, 9),
                round(min_real, 9),
                n_pos,
            )
            if key not in self._coupled_stability_warned_keys:
                msg = (
                    "Coupled full-induction operator has unstable eigenmodes "
                    f"(label={label}, max Re(lambda)={max_real:.3e}, "
                    f"positive modes={n_pos}/{n_total}). "
                    "Explicit Euler integration is expected to be unstable for this operator."
                )
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self._coupled_stability_warned_keys.add(key)
        return report

    def get_coupled_stability_report(
        self,
        *,
        source: Literal["dense", "sparse", "auto"] = "dense",
        use_pinning: Optional[bool] = None,
    ) -> Dict[str, float]:
        """Return spectral stability report for the coupled full-induction operator."""
        if use_pinning is None:
            use_pinning = self.apply_psi_gauge
        L_flat = np.asarray(
            self.get_coupled_induction_matrix(
                source=source,
                flatten=True,
                use_pinning=use_pinning,
            )
        )
        return self._analyze_coupled_stability(
            L_flat,
            label=f"{source}:pinning={int(bool(use_pinning))}",
        )

    def _build_dt_jr_constraint_rhs(self, dt_jr_driver_coeffs: Optional[np.ndarray]) -> np.ndarray:
        """Build hard-constraint RHS for residual dt_jr solve."""
        constraint_op = self.constraints.induction_constraint_operator_hard
        if constraint_op is None:
            return xp.zeros(0)
        constraint_lm = as_linear_map(constraint_op)
        n_rows = int(constraint_lm.shape[0])
        if n_rows <= 0:
            return xp.zeros(0)
        if dt_jr_driver_coeffs is None:
            return xp.zeros(n_rows)

        driver = np.asarray(dt_jr_driver_coeffs).reshape(-1)
        bundle = self.constraints.induction_constraint_bundle_hard
        if bundle is None:
            if constraint_lm.shape[1] != driver.size and float(np.linalg.norm(driver)) == 0.0:
                driver = np.zeros(int(constraint_lm.shape[1]), dtype=float)
            if constraint_lm.shape[1] != driver.size:
                raise RuntimeError(
                    "Constraint RHS assembly mismatch for non-bundled constraints: "
                    "constraint operator columns do not match driver dimension."
                )
            return -asarray(constraint_lm.matvec(driver))

        if bundle is not None and bundle["C_total"].shape[1] != driver.size:
            n_cols = int(bundle["C_total"].shape[1])
            if float(np.linalg.norm(driver)) == 0.0:
                # Zero driver is representation-invariant for the hard-constraint RHS.
                driver = np.zeros(n_cols, dtype=float)

        if (
            bundle is not None
            and bundle["C_total"].shape[0] == n_rows
            and bundle["C_total"].shape[1] == driver.size
        ):
            C_ll = bundle["C_ll"]
            C_hl = bundle["C_hl"]
            rhs_ll = -C_ll @ driver if C_ll.shape[0] > 0 else np.zeros(0, dtype=float)
            rhs_hl = np.zeros(C_hl.shape[0], dtype=float)
            return asarray(np.concatenate([rhs_ll, rhs_hl]))

        raise RuntimeError(
            "Constraint RHS assembly mismatch: bundle rows/cols do not match "
            "constraint operator and driver dimensions."
        )

    def _project_to_hl_modes(self, values: np.ndarray) -> np.ndarray:
        """Project coefficient-space vector onto the HL mode subspace."""
        vec = np.asarray(values).reshape(-1)
        bundle = self.constraints.induction_constraint_bundle_hard
        if bundle is None:
            return asarray(vec)

        Q_hl = np.asarray(bundle.get("Q_hl", np.zeros((vec.size, 0), dtype=float)))
        M_metric = np.asarray(bundle.get("Q_metric", np.eye(vec.size, dtype=float)))
        if (
            Q_hl.ndim == 2
            and Q_hl.shape[0] == vec.size
            and Q_hl.shape[1] > 0
        ):
            if M_metric.ndim == 2 and M_metric.shape == (vec.size, vec.size):
                vec = Q_hl @ (Q_hl.T @ (M_metric @ vec))
            else:
                vec = Q_hl @ (Q_hl.T @ vec)
        return asarray(vec)

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

        return self.geometry.get_potential_to_E_coeffs_operator(
            mode=self.mode,
            potential_type=potential_type,
            eta_grid=self.M_total_on_grid,
            etaP=etaP_field,
            etaH=etaH_field
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

        # Keep lhs/rhs apex operator basis-consistent for legacy m_imp solves.
        jr_map_operator = self.geometry.get_jr_operator(self.jr.basis if self.jr else None)

        return self.geometry.poloidal_matrices.build_least_squares_problem(
            jr_map_operator=jr_map_operator,
            E_constraint_operator=E_constraint_op,
            connect_hemispheres=(E_constraint_op is not None),
            ih_constraint_scaling=self.ih_constraint_scaling,
            regularization_lambda=self.m_imp_regularization_lambda,
            use_pinning=(getattr(self.solution_basis, "kind", "") in ("CS", "GRID")),
            weighting=self.poloidal_weighting,
        )

    @cached_property
    def m_imp_preconditioner(self) -> Optional[LinearMap]:
        """Preconditioner for the m_imp least-squares problem."""
        logger.info("Building new preconditioner for m_imp solver.")
        return self.m_imp_solver.build_preconditioner(problem=self.m_imp_problem, num_scenarios=1)

    @cached_property
    def dt_jr_preconditioner(self) -> Optional[LinearMap]:
        """Preconditioner for the dt_jr (toroidal) least-squares problem."""
        return None

    @cached_property
    def coupled_preconditioner(self) -> Optional[LinearMap]:
        """Preconditioner for the coupled (2N, 2N) induction system.

        Uses the standard LeastSquaresProblem/LeastSquaresSolver construction.
        """
        if self.preconditioner is None:
            return None

        N = self.solution_basis.index_length
        L = self.coupled_induction_tensor
        L_map = as_linear_map(asarray(L).reshape(2 * N, 2 * N))
        problem = LeastSquaresProblem(
            A=[L_map],
            solution_shape=(2 * N,),
            data_shapes=[(2 * N,)],
        )
        solver = LeastSquaresSolver(solver=self.solver_type, preconditioner=self.preconditioner)
        return solver.build_preconditioner(
            problem=problem,
            preconditioner_type=self.preconditioner,
            num_scenarios=1,
        )

    def _build_imposed_toroidal_baseline(
        self,
        jr_coeffs: Optional[np.ndarray],
        E_direct_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build imposed toroidal baseline `m_imp` from external driver inputs.

        Full-induction path: direct `jr -> m_imp` map in solution space.
        Legacy path: constrained least-squares solve for imposed baseline.
        """
        n = self.solution_basis.index_length
        if self.dynamics_mode == "full_induction":
            if jr_coeffs is None:
                return xp.zeros(n)
            input_basis = self.jr.basis if self.jr is not None else None
            m_imp_from_jr = np.asarray(self.get_m_imp_from_jr_matrix(input_basis=input_basis))
            jr_vec = np.asarray(jr_coeffs).reshape(-1)
            return self.constraints.apply_m_imp_gauge_projection(m_imp_from_jr @ jr_vec)

        # Legacy mode: when IH E-constraint is active we still solve m_imp even if
        # jr is unavailable, because the E-constraint contributes RHS information.
        use_e_constraint = self.connect_hemispheres and self.E_map_constraint_operator is not None
        if jr_coeffs is None and not use_e_constraint:
            return xp.zeros(n)

        problem = self.m_imp_problem
        preconditioner = self.m_imp_preconditioner

        rhs_entries: List[Optional[Any]] = [None] * problem.num_data_terms
        if jr_coeffs is not None:
            op_rhs = self.geometry.get_jr_operator(self.jr.basis if self.jr else None)
            rhs_entries[0] = as_linear_map(op_rhs).matvec(asarray(jr_coeffs).reshape(-1))

        if use_e_constraint:
            if E_direct_coeffs is None:
                raise ValueError(
                    "E_direct_coeffs is required for imposed baseline solve with IH E-constraint."
                )
            E_map_op = self.geometry.E_coeffs_to_E_apex_ll_diff
            E_direct_input = asarray(E_direct_coeffs)

            if hasattr(E_map_op, "apply"):
                b_E = E_map_op.apply(E_direct_input)
            else:
                raise TypeError(
                    "E_coeffs_to_E_apex_ll_diff must provide an 'apply' method "
                    "(ConstraintOperator)."
                )
            rhs_entries[1] = self.ih_constraint_scaling * xp.reshape(b_E, (-1,))

        solution = _timed_solve(
            "state.m_imp",
            self.m_imp_solver,
            problem=problem,
            rhs=rhs_entries,
            preconditioner=preconditioner,
        )
        if solution is None:
            solution = xp.zeros(n)
        return self.constraints.apply_m_imp_gauge_projection(solution)

    def _map_dt_jr_driver_to_dt_m_imp(
        self,
        dt_jr_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Map driver derivative ``dt_jr`` to toroidal driver derivative ``dt_m_imp``."""
        dt_jr_vec = np.asarray(dt_jr_coeffs).reshape(-1)
        m_imp_from_jr = np.asarray(self.get_m_imp_from_jr_matrix(input_basis=self.solution_basis))
        if m_imp_from_jr.ndim != 2 or m_imp_from_jr.shape[1] != dt_jr_vec.size:
            raise RuntimeError(
                "dt_jr -> dt_m_imp mapping dimension mismatch: "
                f"map={m_imp_from_jr.shape}, driver={dt_jr_vec.shape}."
            )
        dt_m_imp = m_imp_from_jr @ dt_jr_vec
        dt_m_imp = self._project_to_hl_modes(dt_m_imp)
        return self.constraints.apply_m_imp_gauge_projection(dt_m_imp)

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

    def _decode_conductance_input_to_eta_coeffs(
        self,
        *,
        storage_base: Any,
        updated_input: dict,
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
        sigma_floor_safe = max(sigma_floor, np.finfo(float).tiny)

        if "SigmaP" in updated_input and "SigmaH" in updated_input:
            f_sigmaP = Field.from_coefficients(storage_base, coeffs=updated_input["SigmaP"])
            f_sigmaH = Field.from_coefficients(storage_base, coeffs=updated_input["SigmaH"])
            sigmaP_grid, _, _ = f_sigmaP.evaluate(r_eval, grid.theta, grid.phi)
            sigmaH_grid, _, _ = f_sigmaH.evaluate(r_eval, grid.theta, grid.phi)
        elif "logSigmaP" in updated_input and "logSigmaH" in updated_input:
            f_log_sigmaP = Field.from_coefficients(storage_base, coeffs=updated_input["logSigmaP"])
            f_log_sigmaH = Field.from_coefficients(storage_base, coeffs=updated_input["logSigmaH"])
            log_sigmaP_grid, _, _ = f_log_sigmaP.evaluate(r_eval, grid.theta, grid.phi)
            log_sigmaH_grid, _, _ = f_log_sigmaH.evaluate(r_eval, grid.theta, grid.phi)
            sigmaP_grid = np.exp(np.asarray(log_sigmaP_grid)) - sigma_floor_safe
            sigmaH_grid = np.exp(np.asarray(log_sigmaH_grid)) - sigma_floor_safe
        else:
            raise KeyError(
                "Unsupported conductance input representation. Expected "
                "('etaP','etaH'), ('SigmaP','SigmaH'), or "
                "('logSigmaP','logSigmaH')."
            )

        sigmaP_grid = np.asarray(sigmaP_grid, dtype=float).reshape(-1)
        sigmaH_grid = np.asarray(sigmaH_grid, dtype=float).reshape(-1)
        if np.any(sigmaP_grid < 0.0):
            logger.warning(
                "Negative Pedersen conductance encountered after interpolation; "
                "clipping to nonnegative."
            )
            sigmaP_grid = np.maximum(sigmaP_grid, 0.0)
        if np.any(sigmaH_grid < 0.0):
            logger.warning(
                "Negative Hall conductance encountered after interpolation; "
                "clipping to nonnegative (Hall sign is geometry-driven)."
            )
            sigmaH_grid = np.maximum(sigmaH_grid, 0.0)

        denom = sigmaP_grid * sigmaP_grid + sigmaH_grid * sigmaH_grid + sigma_floor_safe * sigma_floor_safe
        etaP_grid = sigmaP_grid / denom
        etaH_grid = sigmaH_grid / denom

        etaP_coeffs = np.asarray(
            storage_base.from_grid_values(etaP_grid, grid, "scalar")
        ).reshape(-1)
        etaH_coeffs = np.asarray(
            storage_base.from_grid_values(etaH_grid, grid, "scalar")
        ).reshape(-1)
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
                _, current_deriv = input_manager.get_entry_with_derivative(key, time, interpolation=True)
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

            storage_base = input_manager.get_storage_basis(key)
            if key == "conductance":
                conductance_updated = True
                etaP_coeffs, etaH_coeffs = self._decode_conductance_input_to_eta_coeffs(
                    storage_base=storage_base,
                    updated_input=updated_input,
                )
                f_etaP = Field.from_coefficients(storage_base, coeffs=etaP_coeffs)
                f_etaH = Field.from_coefficients(storage_base, coeffs=etaH_coeffs)
                self.etaP = self._ensure_basis(f_etaP, "scalar")
                self.etaH = self._ensure_basis(f_etaH, "scalar")
            elif key == "jr":
                f_jr = Field.from_coefficients(storage_base, coeffs=updated_input["jr"])
                self.jr = self._ensure_basis(f_jr, "scalar")
                # Driver changed: rebuild imposed toroidal baseline on next use.
                self._imposed_toroidal_dirty = True
                if current_deriv is not None:
                    if self.dynamics_mode == "full_induction":
                        f_dt_jr = Field.from_coefficients(storage_base, coeffs=current_deriv["jr"])
                        dt_jr_solution = self._ensure_basis(f_dt_jr, "scalar")
                        dt_m_imp_coeffs = self._map_dt_jr_driver_to_dt_m_imp(
                            dt_jr_coeffs=asarray(dt_jr_solution.coeffs),
                        )
                        self.dt_m_imp_driver = Field.from_coefficients(
                            self.solution_basis,
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
                f_Br = Field.from_coefficients(storage_base, coeffs=updated_input["Br"])
                self.Br = self._ensure_basis(f_Br, "scalar")
            elif key == "u":
                f_u = Field.from_coefficients(
                    storage_base,
                    coeffs=updated_input["u"].reshape((2, -1)),
                    field_type="tangential",
                )
                self.u = self._ensure_basis(f_u, "tangential")

            # Persist latest snapshot/time for finite-difference derivative estimates.
            self.previous_input_data[key] = current_data
            self.previous_input_time[key] = float(time)

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
        m_imp = self._build_imposed_toroidal_baseline(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction effects."""

        E_shape = (2, self.solution_basis.index_length)
        if self.u is None:
            E_direct = xp.zeros(E_shape)
        else:
            E_direct = self._apply_operator(
                self.u_coeffs_to_E_coeffs, asarray(self.u.coeffs), E_shape
            )
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
        """Assemble non-inductive forcing and update toroidal residual rate."""
        n = self.solution_basis.index_length
    
        if self.psi is None:
            # Initialize dynamic residual state/rate on first call.
            self.psi = xp.zeros(n)
            self.d_psi_dt = xp.zeros(n)

        # E_direct already contains external forcing terms (wind + imposed Br terms).
        E_external = asarray(E_direct_coeffs)

        # Build/refresh imposed toroidal baseline m_imp (quasi-static driver component).
        imposed_refreshed = False
        if jr_coeffs is None:
            m_imp_curr = xp.zeros(n)
            self.m_imp_imposed = asarray(m_imp_curr)
            self._imposed_toroidal_dirty = False
            imposed_refreshed = True
        elif self.m_imp_imposed is None or self._imposed_toroidal_dirty:
            m_imp_curr = self._build_imposed_toroidal_baseline(jr_coeffs, E_external)
            self.m_imp_imposed = asarray(m_imp_curr)
            self._imposed_toroidal_dirty = False
            imposed_refreshed = True
        else:
            m_imp_curr = asarray(self.m_imp_imposed)
            self.m_imp_imposed = asarray(m_imp_curr)

        # Net LL closure is enforced inside the hard toroidal constraint system.
        # Avoid post-hoc psi adjustments that can move the state off-manifold.

        E_imposed_toroidal = self._apply_operator(self.m_imp_to_E_coeffs, m_imp_curr, (2, n))
        E_noninductive = E_external + E_imposed_toroidal

        # Solve directly for dpsi/dt for non-inductive forcing. Coupled
        # self-feedback from (psi, m_ind) is handled in coupled operator blocks.
        self.d_psi_dt = self.solve_dpsi_dt(E_noninductive)

        # Return only non-inductive E coefficients here; inductive contributions
        # from (psi, m_ind) are applied in the coupled dynamics path.
        return E_noninductive, asarray(m_imp_curr)

    def solve_dt_jr(self, E_known: np.ndarray) -> np.ndarray:
        """Solve constrained system for dt_jr.

        Uses ToroidalSystemMatrices.compute_forcing_vector() for the physics RHS
        and assembles the constraint RHS from driver data.
        """
        dt_jr_driver_coeffs = self._get_dt_jr_driver_coeffs()

        # Term 1 (Physics): K - L * dt_jr_driver
        # Driver remains an external forcing. Hard LL symmetry is enforced on the
        # total derivative via the constraint RHS below.
        # Delegate to toroidal_matrices.
        E_coeffs = asarray(E_known)
        rhs_1 = self.toroidal_matrices.compute_forcing_vector(E_coeffs, dt_jr_driver_coeffs)

        # Term 2 (Constraint): hard LL symmetry constraints on residual solve.
        # We want (C_ll @ x) = rhs_ll for x = dt_jr_residual.
        # Constraint RHS for residual solve:
        # For x = d(jr_ind)/dt and d(jr_tot)/dt = d(jr_imp)/dt + x:
        # enforce LL coupling as C_LL x = -C_LL d(jr_imp)/dt.
        constraint_op = self.constraints.induction_constraint_operator_hard
        rhs_2 = self._build_dt_jr_constraint_rhs(dt_jr_driver_coeffs)

        solution = self.toroidal_matrices.solve_dt_jr_superposed(
            rhs_physics=rhs_1,
            rhs_constraint=rhs_2,
            jr_map_operator=constraint_op,
            weighting=self.toroidal_weighting,
            regularization_lambda=self.toroidal_regularization_lambda,
            penalty_operator=None,
            penalty_scaling=0.0,
            hinv_rtol=0.0,
        )

        if solution is None:
            raise RuntimeError("Toroidal superposed dt_jr solve returned no solution.")

        return asarray(solution)

    def solve_dpsi_dt(self, E_known: np.ndarray) -> np.ndarray:
        """Solve constrained system for dpsi/dt."""
        dt_jr_driver_coeffs = self._get_dt_jr_driver_coeffs()

        E_coeffs = asarray(E_known)
        constraint_op = self.constraints.induction_constraint_operator_hard
        rhs_2 = self._build_dt_jr_constraint_rhs(dt_jr_driver_coeffs)
        solution = self.toroidal_matrices.solve_dpsi_dt_superposed_joint_er(
            E_coeffs=E_coeffs,
            dt_jr_driver_coeffs=dt_jr_driver_coeffs,
            rhs_constraint=rhs_2,
            jr_map_operator=constraint_op,
            m_imp_to_jr_operator=self.poloidal_matrices.m_imp_to_jr,
            weighting=self.toroidal_weighting,
            regularization_lambda=self.toroidal_regularization_lambda,
            penalty_operator=None,
            penalty_scaling=0.0,
            hinv_rtol=0.0,
            use_pinning=self.apply_psi_gauge,
        )
        if solution is None:
            raise RuntimeError("Toroidal superposed dpsi/dt solve returned no solution.")
        return asarray(solution)

    def _get_dt_jr_driver_coeffs(self) -> Optional[np.ndarray]:
        """Return ``dt_jr`` driver mapped into the solution basis, then HL-projected."""
        if self.dynamics_mode != "full_induction" or self.dt_m_imp_driver is None:
            return None
        dt_m_imp = np.asarray(asarray(self.dt_m_imp_driver.coeffs).reshape(-1))
        dt_m_imp = np.asarray(self._project_to_hl_modes(dt_m_imp)).reshape(-1)
        m_imp_to_jr = as_linear_map(self.poloidal_matrices.m_imp_to_jr)
        return asarray(m_imp_to_jr.matvec(dt_m_imp)).reshape(-1)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total E-field coefficients."""
        E_shape = (2, self.solution_basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, asarray(m_ind), E_shape)
        if self.dynamics_mode == "full_induction":
            # In full-induction, imposed toroidal baseline is handled separately
            # via m_imp_imposed; the m_ind contribution should not trigger an
            # additional m_imp feedback solve here.
            return E_direct_ind, xp.zeros(self.solution_basis.index_length)
        return self._calculate_total_E_field(E_direct_ind, None)

    def calculate_psi_E_coeffs(self, psi: np.ndarray) -> np.ndarray:
        """Map inductive toroidal residual psi to E-field coefficients."""
        E_shape = (2, self.solution_basis.index_length)
        return self._apply_operator(self.toroidal_to_E_coeffs, asarray(psi), E_shape)

    # ----- Time Evolution -----

    @cached_property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix mapping m_ind to div-free E-field."""
        return self.poloidal_matrices.build_induction_matrix(
            problem=self.m_imp_problem,
            solver=self.m_imp_solver,
            E_map_constraint_operator=self.geometry.E_coeffs_to_E_apex_ll_diff,
            ih_constraint_scaling=self.ih_constraint_scaling,
            connect_hemispheres=(self.connect_hemispheres and self.dynamics_mode != "full_induction"),
            m_ind_to_E_operator=self.m_ind_to_E_coeffs,
            m_imp_to_E_operator=self.m_imp_to_E_coeffs,
        )

    @cached_property
    def E_coeffs_to_E_df_matrix(self) -> np.ndarray:
        """Operator extracting toroidal potential (E_df) from vector coefficients."""
        N = self.solution_basis.index_length
        kind = getattr(self.solution_basis, "kind", "")

        if kind == "SH":
            zeros = np.zeros((N, N))
            eye = np.eye(N)
            return asarray(np.hstack([zeros, eye]))

        if kind in ("CS", "GRID"):
            P = self.solution_basis.construct_projection_matrix(self.geometry.grid)
            if P.ndim != 4 or P.shape[0] != 2 or P.shape[2] != 2:
                raise ValueError(
                    "Projection matrix must have canonical shape (2, n_coeffs, 2, n_grid), "
                    f"got {getattr(P, 'shape', None)}."
                )
            # P shape: (2, N_coeffs, 2, N_grid). Toroidal block is P[1].
            return asarray(P[1].reshape(N, 2 * P.shape[3]))

        # Fallback: build by probing basis extraction
        M = np.zeros((N, 2 * N))
        for i in range(2 * N):
            e_i = np.zeros(2 * N)
            e_i[i] = 1.0
            coeffs = e_i.reshape(2, N)
            M[:, i] = asarray(self.solution_basis.get_toroidal_potential_coeffs(coeffs))
        return asarray(M)

    def get_induction_operator(self) -> "LinearMap":
        """Get matrix-free induction operator (m_ind -> E_df).

        Returns a LinearMap for matrix-free steady-state computation.
        More efficient than building the dense matrix for large systems.
        """
        return self.poloidal_matrices.get_induction_operator(
            problem=self.m_imp_problem,
            solver=self.m_imp_solver,
            preconditioner=self.m_imp_preconditioner,
            E_map_constraint_operator=self.geometry.E_coeffs_to_E_apex_ll_diff,
            ih_constraint_scaling=self.ih_constraint_scaling,
            connect_hemispheres=(self.connect_hemispheres and self.dynamics_mode != "full_induction"),
            m_ind_to_E_operator=self.m_ind_to_E_coeffs,
            m_imp_to_E_operator=self.m_imp_to_E_coeffs,
        )

    # _build_m_ind_to_E_df_matrix refactored to PoloidalSystemMatrices.build_induction_matrix

    # _calculate_d_m_ind_dt refactored to PoloidalSystemMatrices.compute_rates

    def _apply_state_linear_operator(
        self,
        operator: Any,
        state: np.ndarray,
        output_shape: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """Apply a state-space linear operator to a flattened/stacked state."""
        state_arr = asarray(state)
        state_shape = tuple(state_arr.shape)
        state_flat = state_arr.reshape(-1)

        if hasattr(operator, "matvec"):
            out_flat = asarray(operator.matvec(state_flat)).reshape(-1)
        else:
            op_arr = asarray(operator)
            if op_arr.ndim == 4:
                if state_arr.ndim != 2:
                    raise ValueError(
                        "4D coupled operator requires 2D state shaped (n_state, n_coeffs)."
                    )
                out_arr = asarray(
                    xp.einsum("ijkl,kl->ij", op_arr, state_arr, optimize=True)
                )
                return out_arr.reshape(output_shape or state_shape)
            out_flat = asarray(op_arr).reshape(state_flat.size, state_flat.size) @ state_flat

        return asarray(out_flat).reshape(output_shape or state_shape)

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
        y_arr = asarray(y)
        y_shape = tuple(y_arr.shape)
        forcing_arr = (
            xp.zeros_like(y_arr)
            if forcing is None
            else asarray(forcing).reshape(y_shape)
        )

        if isinstance(self.poloidal_integrator, ExponentialIntegrator):
            if linear_operator is None:
                raise ValueError("Exponential integration requires linear_operator.")

            n_total = int(y_arr.size)
            if hasattr(linear_operator, "matvec"):
                L_dense = self._densify_linear_operator(linear_operator, n_total)
            else:
                L_arr = asarray(linear_operator)
                if L_arr.ndim == 4:
                    L_dense = asarray(L_arr).reshape(n_total, n_total)
                else:
                    L_dense = asarray(L_arr).reshape(n_total, n_total)

            step_kwargs: Dict[str, Any] = dict(exponential_kwargs or {})
            forcing_flat = None
            if forcing is not None:
                forcing_flat = np.asarray(asarray(forcing_arr), dtype=float).reshape(n_total)
            steady_state_flat = None
            if steady_state is not None:
                steady_state_flat = np.asarray(steady_state, dtype=float).reshape(n_total)
            if forcing_flat is None and steady_state_flat is None:
                raise ValueError(
                    "Exponential integration requires either forcing or steady_state."
                )
            y_next_flat = self.poloidal_integrator.step(
                y=np.asarray(y_arr, dtype=float).reshape(n_total),
                dt=float(dt),
                linear_operator=np.asarray(L_dense, dtype=float),
                forcing=forcing_flat,
                steady_state=steady_state_flat,
                **step_kwargs,
            )
            return asarray(y_next_flat).reshape(y_shape)

        if rates_func is None:
            if linear_operator is None:
                raise ValueError("Either rates_func or linear_operator must be provided.")

            def default_rates_func(y_curr: np.ndarray, _t: float) -> np.ndarray:
                y_curr_arr = asarray(y_curr).reshape(y_shape)
                rates = self._apply_state_linear_operator(
                    linear_operator,
                    y_curr_arr,
                    output_shape=y_shape,
                )
                return asarray(rates + forcing_arr)

            rates = default_rates_func
        else:
            rates = rates_func

        return asarray(
            self.poloidal_integrator.step(
                y=y_arr,
                dt=dt,
                rates_func=rates,
            )
        ).reshape(y_shape)

    def _solve_linear_steady_state(
        self,
        *,
        linear_operator: Any,
        forcing: np.ndarray,
        solution_shape: Tuple[int, ...],
        solver: Optional[str] = None,
        preconditioner: Optional[LinearMap] = None,
        use_pinning: Optional[bool] = None,
    ) -> np.ndarray:
        """Solve a linear steady-state system `A x = -forcing` for arbitrary state shape."""
        rhs = asarray(forcing).reshape(-1)
        n_total = int(np.prod(solution_shape))

        # Coupled (psi, m_ind) steady-state path with gauge elimination.
        if (
            len(solution_shape) == 2
            and int(solution_shape[0]) == 2
            and int(solution_shape[1]) == self.solution_basis.index_length
        ):
            if use_pinning is None:
                use_pinning = self.apply_psi_gauge
            if solver is None:
                solver = self.solver_type

            coupled_operator = linear_operator
            using_default_coupled_operator = False
            if coupled_operator is None:
                using_default_coupled_operator = True
                coupled_operator = self.get_coupled_operator_for_steady_state(
                    solver=solver,
                    use_pinning=use_pinning,
                )
            steady_solver = CoupledSteadyStateSolver(
                n_scalar=self.solution_basis.index_length,
                apply_m_ind_gauge=self.apply_m_ind_gauge,
                preconditioner_type=self.preconditioner,
                psi_gauge_row_builder=self.constraints.get_psi_gauge_row,
                m_ind_gauge_row_builder=self.constraints.get_m_ind_gauge_row,
                timed_solve=_timed_solve,
                column_scale_cache=self._coupled_steady_state_column_scale_cache,
                solver_tolerance=float(getattr(self.m_imp_solver, "tolerance", 1e-13)),
                steady_state_regularization_lambda=1e-10,
            )
            column_scale_cache_key = None
            if using_default_coupled_operator:
                column_scale_cache_key = (
                    bool(use_pinning),
                    int(self.solution_basis.index_length),
                )
            y_ss_flat = steady_solver.solve(
                coupled_operator=coupled_operator,
                forcing_flat=rhs,
                solver=solver,
                preconditioner=preconditioner,
                use_pinning=bool(use_pinning),
                column_scale_cache_key=column_scale_cache_key,
            )
            return asarray(y_ss_flat).reshape(solution_shape)

        # Single-state steady-state path (legacy m_ind).
        if linear_operator is None:
            raise ValueError("Single-state steady-state solve requires linear_operator.")

        vec_b = -rhs
        induction_obj = linear_operator
        if not hasattr(induction_obj, "matvec"):
            induction_obj = asarray(induction_obj).reshape(n_total, n_total)
        induction_op = as_linear_map(induction_obj)

        equality_operator = None
        equality_rhs = None
        if self.apply_m_ind_gauge:
            gauge_row = np.asarray(self.constraints.get_m_ind_gauge_row(n_total), dtype=float)
            if gauge_row.ndim == 1:
                gauge_row = gauge_row.reshape(1, -1)
            if gauge_row.ndim == 2 and gauge_row.shape[1] == n_total and gauge_row.shape[0] > 0:
                equality_operator = gauge_row
                equality_rhs = np.zeros(gauge_row.shape[0], dtype=float)

        ls_problem = LeastSquaresProblem(
            A=[induction_op],
            solution_shape=solution_shape,
            data_shapes=[solution_shape],
        )
        ls_solver = LeastSquaresSolver(
            solver=(solver or "lsmr"),
            tolerance=1e-10,
        )
        solve_kwargs: Dict[str, Any] = {
            "preconditioner": preconditioner if equality_operator is None else None,
        }
        if equality_operator is not None:
            solve_kwargs["equality_operator"] = equality_operator
            solve_kwargs["equality_rhs"] = equality_rhs

        sol = _timed_solve(
            "state.steady_state_single",
            ls_solver,
            ls_problem,
            [vec_b],
            **solve_kwargs,
        )
        return asarray(sol).reshape(solution_shape)

    def build_coupled_forcing(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Build coupled forcing tensor ``K`` for ``[psi, m_ind]`` dynamics."""
        scale = self.poloidal_matrices.E_df_to_d_m_ind_dt
        E_noind_field = self.poloidal_matrices.solution_basis.get_toroidal_potential_coeffs(
            E_coeffs_noind
        )
        k1 = asarray(scale * E_noind_field)
        if self.d_psi_dt is not None:
            k0 = asarray(self.d_psi_dt)
        else:
            k0 = xp.zeros_like(k1)
        return xp.stack([k0, k1])

    def solve_steady_state_model_variables(
        self,
        E_coeffs_noind: np.ndarray,
        *,
        update_state: bool = True,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Compute steady-state initialization for current dynamics mode."""
        N = self.solution_basis.index_length
        if self.dynamics_mode == "full_induction":
            K = self.build_coupled_forcing(E_coeffs_noind)
            y_ss = self._solve_linear_steady_state(
                linear_operator=None,
                forcing=K,
                solution_shape=(2, N),
                solver=self.solver_type,
                use_pinning=self.apply_psi_gauge,
            )
            psi = asarray(y_ss[0])
            m_ind = asarray(y_ss[1])
            if update_state:
                self.psi = psi
            return psi, m_ind

        k_legacy = asarray(self.poloidal_matrices.solution_basis.get_toroidal_potential_coeffs(E_coeffs_noind))
        m_ss = self._solve_linear_steady_state(
            linear_operator=self.m_ind_to_E_df_matrix,
            forcing=k_legacy,
            solution_shape=(N,),
            solver=self.solver_type,
        )
        return None, asarray(m_ss)

    def evolve_model_variables(
        self,
        m_ind: np.ndarray,
        dt: float,
        E_coeffs_noind: np.ndarray,
        *,
        steady_state_m_ind: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Advance model variables by one time step.

        Returns ``(psi, m_ind)`` where ``psi`` is ``None`` for legacy mode.
        """
        if self.dynamics_mode == "full_induction":
            if psi is None:
                if self.psi is None:
                    self.psi = xp.zeros((self.solution_basis.index_length,))
                psi = self.psi

            y = xp.stack([asarray(psi), asarray(m_ind)])
            K = self.build_coupled_forcing(E_coeffs_noind)
            if isinstance(self.poloidal_integrator, ExponentialIntegrator):
                use_pinning = self.apply_psi_gauge
                N = self.solution_basis.index_length
                m = 2 * N

                # Dense expm on a (2N x 2N) matrix can require several matrix-sized
                # work buffers; guard against host OOM before allocation.
                avail_bytes = _available_memory_bytes()
                if avail_bytes is not None:
                    matrix_bytes = int(m) * int(m) * np.dtype(float).itemsize
                    estimated_peak_bytes = int(8 * matrix_bytes)
                    if estimated_peak_bytes > int(0.80 * avail_bytes):
                        need_gib = estimated_peak_bytes / float(1024 ** 3)
                        avail_gib = avail_bytes / float(1024 ** 3)
                        raise MemoryError(
                            "Coupled exponential step would likely exceed available memory: "
                            f"need ~{need_gib:.2f} GiB, available ~{avail_gib:.2f} GiB. "
                            "Reduce resolution or use a non-exponential integrator for this run."
                        )

                coupled_dynamics_operator = self.get_coupled_operator_for_time_integration(
                    use_dense=True,
                    use_pinning=use_pinning,
                )
                L_dense = np.asarray(
                    self._densify_linear_operator(coupled_dynamics_operator, m),
                    dtype=float,
                ).reshape(m, m)

                forcing_flat = np.asarray(K).reshape(m)
                if self.induction_null_diagnostics:
                    diag_dense = None
                    if self._coupled_null_basis is None or self._coupled_null_basis.shape[0] != m:
                        if m <= 2000:
                            diag_dense = np.asarray(
                                self._densify_linear_operator(coupled_dynamics_operator, m)
                            )
                        if diag_dense is not None:
                            self._update_coupled_null_basis(np.asarray(diag_dense))
                    self._check_forcing_null_projection(np.asarray(forcing_flat))

                y_new = self._evolve_linear_state(
                    y=np.asarray(y).reshape(m),
                    dt=float(dt),
                    linear_operator=L_dense,
                    forcing=np.asarray(K).reshape(m),
                    exponential_kwargs={
                        "max_step_scale": 10.0,
                        "max_substeps": 32768,
                    },
                ).reshape(2, N)
            else:
                coupled_operator = self.get_coupled_operator_for_time_integration(
                    use_dense=self.dense_full_operators,
                    use_pinning=self.apply_psi_gauge,
                )
                y_new = self._evolve_linear_state(
                    y=y,
                    dt=dt,
                    linear_operator=coupled_operator,
                    forcing=K,
                )
            psi_new = asarray(y_new[0])
            m_ind_new = asarray(y_new[1])
            self.psi = psi_new
            return psi_new, m_ind_new

        use_dense_rate_operator = bool(
            self.dense_full_operators or isinstance(self.poloidal_integrator, ExponentialIntegrator)
        )
        forcing = None
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]]
        if use_dense_rate_operator:
            scale = self.poloidal_matrices.E_df_to_d_m_ind_dt
            linear_operator = scale * asarray(self.m_ind_to_E_df_matrix)
            E_noind_field = self.poloidal_matrices.solution_basis.get_toroidal_potential_coeffs(
                E_coeffs_noind
            )
            forcing = asarray(scale * E_noind_field)
            rates_func = None
        else:
            linear_operator = None

            def rates_func(y, t):
                return self.poloidal_matrices.compute_rates(
                    m_ind=y,
                    t=t,
                    E_coeffs_noind=E_coeffs_noind,
                    induction_matrix=None,
                    m_ind_to_E_operator=self.m_ind_to_E_coeffs,
                    problem=self.m_imp_problem,
                    solver=self.m_imp_solver,
                    preconditioner=self.m_imp_preconditioner,
                    E_map_constraint_operator=self.geometry.E_coeffs_to_E_apex_ll_diff,
                    ih_constraint_scaling=self.ih_constraint_scaling,
                    connect_hemispheres=(
                        self.connect_hemispheres and self.dynamics_mode != "full_induction"
                    ),
                    m_imp_to_E_operator=self.m_imp_to_E_coeffs,
                )

        if isinstance(self.poloidal_integrator, ExponentialIntegrator) and linear_operator is None:
            scale = self.poloidal_matrices.E_df_to_d_m_ind_dt
            linear_operator = scale * self.m_ind_to_E_df_matrix

        m_ind_new = self._evolve_linear_state(
            y=asarray(m_ind),
            dt=dt,
            linear_operator=linear_operator,
            forcing=forcing,
            rates_func=rates_func,
            steady_state=asarray(steady_state_m_ind) if steady_state_m_ind is not None else None,
        )
        m_ind_new = self.constraints.apply_m_ind_gauge_projection(m_ind_new)
        return None, asarray(m_ind_new)

    # -------------------------------------------------------------------------
    # Coupled Exponential Integrator
    # -------------------------------------------------------------------------

    @cached_property
    def coupled_operator_api(self) -> CoupledOperatorAPI:
        """Internal coupled-operator assembly/exposure helper."""
        return CoupledOperatorAPI(self)

    def get_coupled_induction_tensor(self, use_pinning: Optional[bool] = None) -> np.ndarray:
        """Build the coupled tensor ``L_coupled`` with shape ``(2, N, 2, N)``."""
        return self.coupled_operator_api.get_coupled_induction_tensor(use_pinning=use_pinning)

    @cached_property
    def coupled_induction_tensor(self) -> np.ndarray:
        """Default coupled induction tensor (delegates to get_coupled_induction_tensor)."""
        return self.coupled_operator_api.get_coupled_induction_tensor(use_pinning=self.apply_psi_gauge)

    @cached_property
    def coupled_induction_operator_sparse(self) -> "LinearMap":
        """Cached matrix-free coupled operator for non-exponential stepping."""
        solver = self.solver_type if self.solver_type in ("lsmr", "cgls") else "lsmr"
        return self.coupled_operator_api.get_coupled_induction_operator(
            matrix_free=True,
            solver=solver,
            use_pinning=self.apply_psi_gauge,
        )

    @cached_property
    def coupled_induction_matrix_dense(self) -> np.ndarray:
        """Cached dense coupled operator matrix with shape ``(2N, 2N)``."""
        N = self.solution_basis.index_length
        return asarray(self.coupled_induction_tensor).reshape(2 * N, 2 * N)

    @cached_property
    def coupled_induction_blocks_dense(self) -> Dict[str, np.ndarray]:
        """Cached dense coupled blocks keyed by physical role."""
        return self.coupled_operator_api.get_coupled_induction_blocks(
            source="dense",
            use_pinning=self.apply_psi_gauge,
        )

    def _densify_linear_operator(self, operator: Any, n_total: int) -> np.ndarray:
        """Convert a linear operator to dense ``(2N, 2N)``."""
        return self.coupled_operator_api._densify_linear_operator(operator, n_total)

    def get_coupled_induction_matrix(
        self,
        source: Literal["dense", "sparse", "auto"] = "auto",
        flatten: bool = True,
        use_pinning: Optional[bool] = None,
    ) -> np.ndarray:
        """Expose coupled operator matrix in dense form."""
        return self.coupled_operator_api.get_coupled_induction_matrix(
            source=source,
            flatten=flatten,
            use_pinning=use_pinning,
        )

    def get_coupled_induction_blocks(
        self,
        source: Literal["dense", "sparse", "auto"] = "auto",
        use_pinning: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """Expose coupled block matrices keyed by physical role."""
        return self.coupled_operator_api.get_coupled_induction_blocks(
            source=source,
            use_pinning=use_pinning,
        )

    def get_coupled_operator_for_steady_state(
        self,
        *,
        solver: Optional[str] = None,
        use_pinning: Optional[bool] = None,
    ) -> Any:
        """Return coupled operator used by steady-state coupled solve."""
        return self.coupled_operator_api.get_coupled_operator_for_steady_state(
            solver=solver,
            use_pinning=use_pinning,
        )

    def get_coupled_operator_for_time_integration(
        self,
        *,
        use_dense: Optional[bool] = None,
        use_pinning: Optional[bool] = None,
    ) -> Any:
        """Return coupled operator used by non-exponential full-induction stepping."""
        return self.coupled_operator_api.get_coupled_operator_for_time_integration(
            use_dense=use_dense,
            use_pinning=use_pinning,
        )

    def _get_hl_projection_matrix(self, n_coeffs: int) -> np.ndarray:
        """Return dense projector used by `_project_to_hl_modes`."""
        return self.coupled_operator_api.get_hl_projection_matrix(n_coeffs)

    def get_m_imp_from_jr_matrix(self, input_basis: Optional[Any] = None) -> np.ndarray:
        """Expose dense linear map from input `jr` coefficients to imposed `m_imp`."""
        return self.coupled_operator_api.get_m_imp_from_jr_matrix(input_basis=input_basis)

    def get_external_forcing_matrices(self, input_basis_jr: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """Expose dense forcing maps for coupled rates from `u` and `jr`."""
        return self.coupled_operator_api.get_external_forcing_matrices(
            input_basis_jr=input_basis_jr
        )

    def get_coupled_induction_operator(
        self,
        dtpsi_from_psi: Any = None,
        dtpsi_from_mind: Any = None,
        dmind_from_psi: Any = None,
        dmind_from_mind: Any = None,
        matrix_free: bool = False,
        solver: str = "lsmr",
        use_pinning: Optional[bool] = None,
    ) -> "LinearMap":
        """Build coupled operator for ``y=[psi, m_ind]`` dynamics."""
        return self.coupled_operator_api.get_coupled_induction_operator(
            dtpsi_from_psi=dtpsi_from_psi,
            dtpsi_from_mind=dtpsi_from_mind,
            dmind_from_psi=dmind_from_psi,
            dmind_from_mind=dmind_from_mind,
            matrix_free=matrix_free,
            solver=solver,
            use_pinning=use_pinning,
        )

    def _update_coupled_null_basis(self, L_flat: np.ndarray) -> None:
        """Build/update cached near-null basis for coupled-operator diagnostics."""
        if not self.induction_null_diagnostics:
            return
        m = L_flat.shape[0]
        if self._coupled_null_basis is not None and self._coupled_null_basis.shape[0] == m:
            return

        _, svals, vt = np.linalg.svd(np.asarray(L_flat), full_matrices=False)
        if svals.size == 0:
            self._coupled_null_basis = np.zeros((m, 0), dtype=float)
            self._coupled_null_threshold = 0.0
            return

        threshold = float(self.induction_null_svd_rtol) * float(svals[0])
        mask = svals < threshold
        basis = vt[mask, :].T if np.any(mask) else np.zeros((m, 0), dtype=vt.dtype)

        self._coupled_null_basis = basis
        self._coupled_null_threshold = threshold
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
