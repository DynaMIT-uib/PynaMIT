"""Toroidal system assembly for the dt_jr evolution equation.

The assembled operator has the form:
    L_dtjr * dt_jr = forcing_dtjr
with
    L_dtjr = C + M0 + M1 @ radial_closure_dtjr

Design note for CS-dominant full induction:
    - The toroidal closure operator is assembled in one auxiliary SH basis
      on the same grid.
    - CS remains the state/grid representation basis.

This keeps eliminated radial-structure identities and coupled toroidal
blocks in one consistent closure basis while preserving CS-centric
state representation.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Tuple, Optional, Callable

import numpy as np
import scipy.sparse
from functools import cached_property

from pynamit.utils import to_numpy, asarray, xp, tensor_pinv
from pynamit.simulation.geometry_utils import to_dense
from pynamit.math.linear_map import as_linear_map
from pynamit.math.constants import mu0
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.spherical_harmonics import sh_operators
from pynamit.simulation.toroidal_closure import ToroidalClosureProjector

logger = logging.getLogger(__name__)


@dataclass
class _BrConstraintSolver:
    """Regularized weak-form solver for ``Br * x = rhs`` in the scalar basis.

    Solves the grid relation
        ``Br * x = rhs``
    in a projected weak form over scalar basis coefficients, using the
    regularized weighted minimum-norm problem
        ``argmin_c ||W^(1/2)(diag(Br) G c - rhs)||^2 + floor^2 ||W^(1/2) G c||^2``
    with ``x = G c``.

    This single helper is reused for all ``1/Br``-type inversions (e.g. the
    ideal-closure ``E_r`` reconstruction and ``j_r -> alpha`` mapping) so the
    equatorial branch-selection policy stays consistent.
    """

    G: np.ndarray
    Br: np.ndarray
    weights: np.ndarray
    floor: float
    pinv_symmetric: Callable[[np.ndarray, float], np.ndarray]
    default_pinv_rcond: Callable[[Any], float]

    def __post_init__(self) -> None:
        self.G = np.asarray(self.G, dtype=float)
        self.Br = np.asarray(self.Br, dtype=float).reshape(-1)
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        self.floor = float(max(self.floor, 0.0))

        if self.G.ndim != 2:
            self.G = self.G.reshape(self.G.shape[0], -1)
        n_grid = int(self.G.shape[0])
        if self.Br.size != n_grid:
            raise ValueError(
                "BrConstraintSolver size mismatch: "
                f"Br={self.Br.shape}, G={self.G.shape}."
            )
        if self.weights.size != n_grid:
            raise ValueError(
                "BrConstraintSolver weight size mismatch: "
                f"weights={self.weights.shape}, G={self.G.shape}."
            )
        self.weights = np.maximum(self.weights, 0.0)
        if float(np.sum(self.weights)) <= 0.0:
            self.weights = np.ones(n_grid, dtype=float)

    @cached_property
    def constraint_operator_grid(self) -> np.ndarray:
        """Return the explicit grid constraint operator ``diag(Br) @ G``."""
        return self.Br[:, None] * self.G

    @cached_property
    def regularized_normal_matrix(self) -> np.ndarray:
        """Return ``G^T W (Br^2 + floor^2) G``."""
        w_scale = self.weights * (self.Br * self.Br + self.floor * self.floor)
        lhs = (self.G.T * w_scale) @ self.G
        return 0.5 * (lhs + lhs.T)

    @cached_property
    def rhs_lift_operator(self) -> np.ndarray:
        """Return ``G^T W Br`` mapping grid rhs to coefficient-space RHS."""
        w_br = self.weights * self.Br
        return self.G.T * w_br

    @cached_property
    def rhs_to_coeff_operator(self) -> np.ndarray:
        """Return cached map from grid RHS to scalar coefficients."""
        lhs = self.regularized_normal_matrix
        rcond = float(self.default_pinv_rcond(lhs.shape))
        lhs_pinv = self.pinv_symmetric(lhs, rcond=rcond)
        return lhs_pinv @ self.rhs_lift_operator

    @cached_property
    def rhs_to_grid_operator(self) -> np.ndarray:
        """Return cached map from grid RHS to grid ``E_r``."""
        return self.G @ self.rhs_to_coeff_operator


class ToroidalSystemMatrices:
    """Assembles system matrices for Toroidal Induction.

    Handles the construction of:
    - Mass Matrix (C)
    - Advection Coupling Matrices (M0, M1)
    - dt_jr Radial-Closure Operator

    Parameters
    ----------
    basis : Any
        The spectral basis (SHBasis).
    grid : Any
        The integration grid (from Geometry).
    b_field : Any
        The background magnetic field evaluated on the grid.
        Components used throughout this class are from this background
        field: ``Br = b_field.vec.r`` and
        ``B_s = (b_field.vec.theta, b_field.vec.phi)``.
    RI : float
        Radius of the ionosphere.
    """

    def __init__(
        self,
        basis: Any,
        grid: Any,
        b_field: Any,
        RI: float,
        closure_derivative_basis: Optional[Any] = None,
        forcing_derivative_basis: Optional[Any] = None,
        radial_derivative_basis: Optional[Any] = None,
        dtjr_solver: str = "svd",
        dtjr_preconditioner: Optional[str] = None,
        dtjr_tolerance: float = 1e-13,
    ):
        self.basis = basis
        self.grid = grid
        self.b_field = b_field
        self.RI = RI
        # Derivative operators:
        # - closure_derivative_basis/forcing_derivative_basis: primary derivative
        #   basis for toroidal closure assembly.
        # - radial_derivative_basis: optional override for Er/radial-closure
        #   derivative chains.
        #
        # In cs_dominant full-induction we can set all three to one auxiliary SH
        # basis to keep the full toroidal closure assembly basis-consistent.
        base_closure_basis = basis if closure_derivative_basis is None else closure_derivative_basis
        self.closure_derivative_basis = base_closure_basis
        self.forcing_derivative_basis = (
            base_closure_basis if forcing_derivative_basis is None else forcing_derivative_basis
        )
        self.radial_derivative_basis = (
            self.forcing_derivative_basis if radial_derivative_basis is None else radial_derivative_basis
        )
        self._cs_derivative_operator_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        
        self.is_cs = (getattr(self.basis, "kind", "") == "CS")
        if not self.is_cs:
            self.gaunt_engine = GauntEngine(basis)
        # Cache for jr -> psi inverses (true Moore-Penrose, keyed by operator id).
        self._jr_to_psi_cache: dict[tuple[int, bool], np.ndarray] = {}
        # Cache for explicit gauge projector applied after MP inversion.
        self._psi_gauge_projector_cache: dict[tuple[int, bool, str], np.ndarray] = {}
        # Cache for weighted dt_jr least-squares problems.
        self._dtjr_problem_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        # Cached linear maps for repeated dt_jr solves.
        self._dtjr_unconstrained_map_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._dtalpha_unconstrained_map_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._dtjr_constrained_maps_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        # Cached linear maps for repeated direct dpsi/dt solves.
        self._dpsi_problem_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._dpsi_unconstrained_map_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._dpsi_constrained_maps_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._dtalpha_er_joint_problem_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._dtalpha_er_schur_bundle_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._dtalpha_to_dpsi_map_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self.configure_dtjr_solver(
            solver=dtjr_solver,
            preconditioner=dtjr_preconditioner,
            tolerance=dtjr_tolerance,
        )

    def _build_auxiliary_toroidal_matrices(self, closure_basis: Any) -> "ToroidalSystemMatrices":
        """Build auxiliary toroidal assembler in closure basis."""
        return ToroidalSystemMatrices(
            basis=closure_basis,
            grid=self.grid,
            b_field=self.b_field,
            RI=self.RI,
            closure_derivative_basis=closure_basis,
            forcing_derivative_basis=closure_basis,
            radial_derivative_basis=closure_basis,
            dtjr_solver=self.dtjr_solver,
            dtjr_preconditioner=self.dtjr_preconditioner,
            dtjr_tolerance=self.dtjr_tolerance,
        )

    @cached_property
    def _toroidal_closure_projector(self) -> ToroidalClosureProjector:
        """Closure-basis projector for toroidal assembly components."""
        return ToroidalClosureProjector(
            state_basis=self.basis,
            closure_basis=self.closure_derivative_basis,
            grid=self.grid,
            build_auxiliary_assembler=self._build_auxiliary_toroidal_matrices,
        )

    def configure_dtjr_solver(
        self,
        *,
        solver: str,
        preconditioner: Optional[str],
        tolerance: float,
    ) -> None:
        """Configure solver policy for dt_jr least-squares solves."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        if solver not in LeastSquaresSolver.VALID_SOLVERS:
            raise ValueError(
                f"Invalid dt_jr solver '{solver}'. Valid options: "
                f"{LeastSquaresSolver.VALID_SOLVERS}."
            )
        if preconditioner is not None and preconditioner not in LeastSquaresSolver.VALID_PRECONDITIONERS:
            raise ValueError(
                f"Invalid dt_jr preconditioner '{preconditioner}'. Valid options: "
                f"{LeastSquaresSolver.VALID_PRECONDITIONERS}."
            )
        self.dtjr_solver = str(solver)
        self.dtjr_preconditioner = preconditioner
        self.dtjr_tolerance = float(max(tolerance, 1e-15))
        # Cached linear maps depend on the solve policy.
        self._dtjr_unconstrained_map_cache.clear()
        self._dtalpha_unconstrained_map_cache.clear()
        self._dtjr_constrained_maps_cache.clear()
        self._dpsi_unconstrained_map_cache.clear()
        self._dpsi_constrained_maps_cache.clear()
        self._dtalpha_er_joint_problem_cache.clear()
        self._dtalpha_er_schur_bundle_cache.clear()
        self._dtalpha_to_dpsi_map_cache.clear()

    def _dtjr_solver_signature(self) -> tuple[Any, ...]:
        """Return cache signature for the configured dt_jr solver policy."""
        return (
            self.dtjr_solver,
            self.dtjr_preconditioner,
            float(self.dtjr_tolerance),
        )

    def _build_cs_grid_derivative_operators(
        self, deriv_basis: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build grid-space operators (D_theta, D_phi_scaled, Laplacian) for CS."""
        if deriv_basis is self.basis:
            D_th = self.basis.get_G(self.grid, derivative="theta")
            D_ph = self.basis.get_G(self.grid, derivative="phi")
            # Build Laplacian from the discrete vector identities to keep the
            # native CS path aligned with the same div/grad operators used
            # elsewhere (mimetic-consistent discrete calculus).
            L_grid = self._build_cs_mimetic_laplacian(self.basis)
            if scipy.sparse.issparse(D_th):
                D_th = D_th.toarray()
            if scipy.sparse.issparse(D_ph):
                D_ph = D_ph.toarray()
            if scipy.sparse.issparse(L_grid):
                L_grid = L_grid.toarray()
            return np.asarray(D_th), np.asarray(D_ph), np.asarray(L_grid)

        # Spectral-on-grid derivative backend:
        #   D = G_deriv @ P
        #   Lap = G @ Lap_coeff @ P
        G = np.asarray(to_dense(deriv_basis.get_G(self.grid)))
        G_th = np.asarray(to_dense(deriv_basis.get_G(self.grid, derivative="theta")))
        G_ph = np.asarray(to_dense(deriv_basis.get_G(self.grid, derivative="phi")))
        P = np.asarray(to_dense(deriv_basis.construct_scalar_projection_matrix(self.grid)))
        L_coeff = np.asarray(to_dense(deriv_basis.get_laplacian_operator(r=1.0)))

        D_th = G_th @ P
        D_ph = G_ph @ P
        L_grid = G @ (L_coeff @ P)
        return np.asarray(D_th), np.asarray(D_ph), np.asarray(L_grid)

    @property
    def _use_auxiliary_closure_basis(self) -> bool:
        """Whether to assemble toroidal closure through an auxiliary basis."""
        return bool(self.is_cs and self._toroidal_closure_projector.uses_auxiliary_basis)

    @cached_property
    def _auxiliary_closure_matrices(self) -> "ToroidalSystemMatrices":
        """Auxiliary toroidal assembler for closure-basis projection."""
        if not self._use_auxiliary_closure_basis:
            raise RuntimeError("Auxiliary closure matrices requested without auxiliary basis.")
        return self._toroidal_closure_projector.auxiliary_assembler

    @cached_property
    def _state_to_aux_scalar_map(self) -> np.ndarray:
        """Map state scalar coefficients to auxiliary closure coefficients."""
        if not self._use_auxiliary_closure_basis:
            raise RuntimeError("State->aux map requested without auxiliary basis.")
        return self._toroidal_closure_projector.state_to_closure_scalar_map

    @cached_property
    def _aux_to_state_scalar_map(self) -> np.ndarray:
        """Map auxiliary closure coefficients back to state coefficients."""
        if not self._use_auxiliary_closure_basis:
            raise RuntimeError("Aux->state map requested without auxiliary basis.")
        return self._toroidal_closure_projector.closure_to_state_scalar_map

    def _project_aux_square_operator_to_state(self, op_aux: np.ndarray) -> np.ndarray:
        """Project a square operator from auxiliary basis to state basis."""
        return self._toroidal_closure_projector.project_square_operator_to_state(op_aux)

    def _build_cs_mimetic_laplacian(self, cs_basis: Any) -> np.ndarray:
        """Assemble a CS Laplacian from discrete div/grad operators.

        ``cs_basis.get_gradient_operator`` maps ``phi -> -grad(phi)`` and
        ``get_vector_divergence_operator`` maps tangential vectors to
        ``div(v)``. Therefore:
            div(grad(phi)) = -div( (-grad(phi)) ) = -Div @ Grad_op
        and the scalar Laplacian operator is ``-Div @ Grad_op``.
        """
        if hasattr(cs_basis, "get_mimetic_laplacian_operator"):
            return np.asarray(cs_basis.get_mimetic_laplacian_operator(grid=self.grid, r=1.0))
        grad_op = np.asarray(to_dense(cs_basis.get_gradient_operator(r=1.0)))
        div_op = np.asarray(to_dense(cs_basis.get_vector_divergence_operator(self.grid)))
        lap_op = -(div_op @ grad_op)
        return np.asarray(lap_op)

    def _get_cs_grid_derivative_operators(
        self, deriv_basis: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached CS derivative operators for a specific analysis basis."""
        key = id(deriv_basis)
        cached = self._cs_derivative_operator_cache.get(key)
        if cached is not None:
            return cached
        ops = self._build_cs_grid_derivative_operators(deriv_basis)
        self._cs_derivative_operator_cache[key] = ops
        return ops

    @property
    def cs_grid_derivative_operators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CS derivative operators used in L_dtjr assembly."""
        if not self.is_cs:
            raise RuntimeError("cs_grid_derivative_operators is only valid for CS basis.")
        return self._get_cs_grid_derivative_operators(self.closure_derivative_basis)

    @property
    def cs_forcing_derivative_operators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CS derivative operators used in E->dt_jr forcing assembly."""
        if not self.is_cs:
            raise RuntimeError("cs_forcing_derivative_operators is only valid for CS basis.")
        return self._get_cs_grid_derivative_operators(self.forcing_derivative_basis)

    @property
    def cs_radial_derivative_operators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return derivative operators used for radial-closure terms on CS runs.

        This intentionally supports an auxiliary basis (typically SH) so that
        radial-structure closure terms can be evaluated in a band-limited
        spherical representation while horizontal operators remain CS-native.
        """
        if not self.is_cs:
            raise RuntimeError("cs_radial_derivative_operators is only valid for CS basis.")
        return self._get_cs_grid_derivative_operators(self.radial_derivative_basis)

    @cached_property
    def cs_laplacian_inverse(self) -> np.ndarray:
        """Return gauge-fixed inverse Laplacian for CS advection assembly.

        The scalar CS Laplacian has a constant null mode. We remove that mode
        explicitly and invert only the mean-zero subspace, then project the
        inverse back to the same subspace.
        """
        if not self.is_cs:
            raise RuntimeError("cs_laplacian_inverse is only valid for CS basis.")
        if hasattr(self.basis, "get_mimetic_laplacian_pinv"):
            return np.asarray(self.basis.get_mimetic_laplacian_pinv(grid=self.grid, r=1.0))

        _, _, lap = self.cs_grid_derivative_operators
        lap = np.asarray(lap)
        if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
            lap = lap.reshape(lap.shape[0], -1)
        n = int(lap.shape[0])

        ones = np.ones((n, 1), dtype=lap.dtype)
        proj = np.eye(n, dtype=lap.dtype) - (ones @ ones.T) / float(n)

        lap_proj = proj @ lap @ proj
        rcond = self._default_pinv_rcond(lap_proj.shape)
        lap_proj_pinv = self._pinv_symmetric(lap_proj, rcond=rcond)
        return proj @ lap_proj_pinv @ proj

    @staticmethod
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        """Return weighted quantile for ``values`` at probability ``q``."""
        vals = np.asarray(values, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if vals.size == 0:
            return 0.0
        if vals.size != w.size or not np.all(np.isfinite(w)) or np.sum(w) <= 0:
            return float(np.quantile(vals, q))

        order = np.argsort(vals)
        v = vals[order]
        ww = np.maximum(w[order], 0.0)
        wsum = float(np.sum(ww))
        if wsum <= 0.0:
            return float(np.quantile(vals, q))
        cdf = np.cumsum(ww) / wsum
        idx = int(np.searchsorted(cdf, np.clip(float(q), 0.0, 1.0), side="left"))
        idx = int(np.clip(idx, 0, v.size - 1))
        return float(v[idx])

    @staticmethod
    def _matrix_sqrt_psd(mat: np.ndarray, floor: float = 0.0) -> np.ndarray:
        """Return a symmetric PSD square root ``S`` with ``S.T @ S ~= mat``."""
        arr = np.asarray(mat, dtype=float)
        sym = 0.5 * (arr + arr.T)
        evals, evecs = np.linalg.eigh(sym)
        if floor > 0.0:
            evals = np.where(evals > floor, evals, 0.0)
        else:
            evals = np.where(evals > 0.0, evals, 0.0)
        return (evecs * np.sqrt(evals)) @ evecs.T

    def _build_physics_sqrt_weight(self, n_rows: int, weighting: str) -> Optional[np.ndarray]:
        """Build consistent sqrt-weight operator for toroidal physics residuals.

        ``LeastSquaresProblem`` interprets provided weights as *sqrt* weights.
        We therefore build weights so that effective residual weights are:
            - grid-native rows: ``diag(q * w_base^2)``
            - spectral rows: ``G^T diag(q * w_base^2) G``
        where ``w_base`` follows the configured strategy:
            - ``linear``: ``sqrt(|Br|)``
            - ``quadratic``: ``|Br|``
        """
        if weighting == "none":
            return None

        br = to_numpy(self.b_field.vec.r).flatten()
        w_base = np.abs(br) if weighting == "quadratic" else np.sqrt(np.abs(br))
        if np.max(w_base) > 0:
            w_base = w_base / np.max(w_base)

        q_weights = None
        if hasattr(self.grid, "weights") and self.grid.weights is not None:
            q_weights = np.asarray(to_numpy(self.grid.weights)).flatten()
            if q_weights.size != w_base.size:
                q_weights = None

        if q_weights is None:
            q_weights = np.ones_like(w_base)
        q_weights = np.maximum(q_weights, 0.0)

        if n_rows == w_base.size:
            # Grid-native residual rows.
            return np.sqrt(q_weights) * w_base

        # Spectral residual rows: build effective metric in coefficient space,
        # then provide its matrix square-root to LeastSquaresProblem.
        G_scalar = np.asarray(to_dense(self.basis.get_G(self.grid)))
        w_eff = q_weights * (w_base**2)
        W_coeff = (G_scalar.T * w_eff) @ G_scalar
        W_coeff = 0.5 * (W_coeff + W_coeff.T)
        return self._matrix_sqrt_psd(W_coeff)

    @cached_property
    def inverse_radial_field_floor(self) -> float:
        """Automatic pseudo-inverse floor for ``1/Br`` stabilization.

        We regularize near magnetic equator where ``Br -> 0`` by selecting a
        robust lower-tail scale from ``|Br|``. This avoids extreme amplification
        of small E-field discrepancies in forcing/closure terms.
        """
        br = np.abs(np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1))
        finite = br[np.isfinite(br) & (br > 0.0)]
        if finite.size == 0:
            return 0.0

        bmax = float(np.max(finite))
        eps_floor = np.finfo(float).eps * bmax

        if hasattr(self.grid, "weights") and self.grid.weights is not None:
            weights = np.asarray(to_numpy(self.grid.weights)).reshape(-1)
            if weights.size == br.size:
                w = np.maximum(weights[np.isfinite(br) & (br > 0.0)], 0.0)
            else:
                w = np.ones_like(finite)
        else:
            w = np.ones_like(finite)

        q05 = self._weighted_quantile(finite, w, 0.05)
        q50 = self._weighted_quantile(finite, w, 0.50)
        floor = max(float(q05), eps_floor)
        if q50 > 0.0:
            floor = min(floor, 0.25 * float(q50))

        return float(max(floor, eps_floor))

    @staticmethod
    def _default_pinv_rcond(shape: tuple[int, ...] | list[int] | np.ndarray | None = None) -> float:
        """Return a deterministic pseudo-inverse cutoff based on machine precision."""
        if shape is None:
            return float(np.finfo(float).eps)
        if isinstance(shape, np.ndarray):
            dims = tuple(int(v) for v in shape.reshape(-1))
        else:
            dims = tuple(int(v) for v in shape)
        dim_max = max(dims) if len(dims) > 0 else 1
        return float(np.finfo(float).eps * max(dim_max, 1))

    @staticmethod
    def _pinv_symmetric(a: np.ndarray, rcond: float) -> np.ndarray:
        """Robust pseudoinverse for symmetric matrices via ``eigh``."""
        a_np = np.asarray(a)
        if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
            return np.linalg.pinv(a_np, rcond=max(float(rcond), 0.0))

        a_sym = 0.5 * (a_np + a_np.T.conj())
        rcond = max(float(rcond), 0.0)
        try:
            eigvals, eigvecs = np.linalg.eigh(a_sym)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(a_sym, rcond=rcond)

        max_abs = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            return np.zeros_like(a_sym)

        cutoff = rcond * max_abs
        inv_eigvals = np.where(np.abs(eigvals) > cutoff, 1.0 / eigvals, 0.0)
        return (eigvecs * inv_eigvals) @ eigvecs.T.conj()

    def _build_reduced_constraint_right_inverse(
        self,
        CQ: np.ndarray,
        *,
        pinv_rcond: float,
    ) -> np.ndarray:
        """Build min-norm right-inverse for ``CQ z = d``.

        Returns ``Z_d`` with shape ``(k, m)`` such that
            z = Z_d d
        is the minimum-norm solution in the reduced coordinates.
        """
        CQ_np = np.asarray(CQ)
        if CQ_np.ndim != 2:
            CQ_np = CQ_np.reshape(CQ_np.shape[0], -1)
        if CQ_np.shape[0] == 0 or CQ_np.shape[1] == 0:
            return np.zeros((CQ_np.shape[1], CQ_np.shape[0]), dtype=CQ_np.dtype)
        if not np.all(np.isfinite(CQ_np)):
            raise ValueError("Reduced constraint matrix contains non-finite values.")

        CCt = CQ_np @ CQ_np.T
        CCt_pinv = self._pinv_symmetric(CCt, pinv_rcond)
        return CQ_np.T @ CCt_pinv

    @cached_property
    def inverse_radial_field(self) -> np.ndarray:
        """Compute stabilized pseudo-inverse ``1/Br`` on the grid.

        Uses ``Br / (Br^2 + floor^2)`` with an automatic floor derived from the
        lower tail of ``|Br|`` to reduce near-equatorial singular amplification.
        """
        Br = np.asarray(to_numpy(self.b_field.vec.r))
        floor = float(self.inverse_radial_field_floor)
        if floor <= 0.0:
            return asarray(1.0 / Br)
        inv = Br / (Br * Br + floor * floor)
        return asarray(inv)

    @cached_property
    def _br_constraint_solver(self) -> _BrConstraintSolver:
        """Build cached weak-form solver for ``Br * x = rhs``."""
        G = np.asarray(to_dense(self.basis.get_G(self.grid)))
        Br = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        n_grid = int(G.shape[0])
        if hasattr(self.grid, "weights") and self.grid.weights is not None:
            weights = np.asarray(to_numpy(self.grid.weights)).reshape(-1)
            if weights.size != n_grid:
                weights = np.ones(n_grid, dtype=float)
        else:
            weights = np.ones(n_grid, dtype=float)

        return _BrConstraintSolver(
            G=G,
            Br=Br,
            weights=weights,
            floor=float(self.inverse_radial_field_floor),
            pinv_symmetric=self._pinv_symmetric,
            default_pinv_rcond=self._default_pinv_rcond,
        )

    @cached_property
    def er_rhs_to_grid_operator(self) -> np.ndarray:
        """Return projected linear map ``rhs -> E_r`` for ideal closure.

        Delegates to ``_BrConstraintSolver`` so the regularized constraint
        semantics remain explicit in one place.
        """
        return asarray(self._br_constraint_solver.rhs_to_grid_operator)

    @cached_property
    def inertia_matrix(self) -> np.ndarray:
        """Construct Inertia Matrix C.
        
        C_lm,l'm' = mu0 * Integral [ Y_lm * (|B0|^2 / B0r) * Y_l'm' ]
        """
        logger.info("Building Toroidal Inertia Matrix...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(self._project_aux_square_operator_to_state(aux.inertia_matrix))
        
        B2 = self.b_field.magnitude**2
        factor = B2 * self.inverse_radial_field
        
        if self.is_cs:
            # Native CS collocation form.
            return xp.diag(mu0 * factor)

        # SH/grid-transform path: build multiplication operator directly from
        # projection/evaluation matrices in coefficient space.
        # For constant factors this yields a scalar-identity (up to quadrature
        # precision), which is the expected inertia behavior.
        G = np.asarray(to_dense(self.basis.get_G(self.grid)))
        P = np.asarray(to_dense(self.projection_matrix))
        factor_vec = np.asarray(to_numpy(factor)).reshape(-1)
        if factor_vec.size != G.shape[0]:
            raise ValueError(
                "Inertia factor/grid size mismatch: "
                f"factor={factor_vec.shape}, G={G.shape}."
            )
        M_factor = P @ (factor_vec[:, None] * G)
        return mu0 * asarray(M_factor)

    @cached_property
    def inertia_matrix_alpha(self) -> np.ndarray:
        """Construct alpha-space inertia matrix.

        For unknown ``dt_alpha`` (with ``dt_jr = Br * dt_alpha``), the inertia
        block is:
            ``C_alpha = mu0 * <Y, |B0|^2 Y>``.
        This avoids explicit ``1/Br`` factors in the solved physics operator.
        """
        logger.info("Building Toroidal Alpha-Space Inertia Matrix...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(self._project_aux_square_operator_to_state(aux.inertia_matrix_alpha))

        B2 = self.b_field.magnitude**2
        factor = np.asarray(to_numpy(B2)).reshape(-1)

        if self.is_cs:
            return asarray(np.diag(mu0 * factor))

        G = np.asarray(to_dense(self.basis.get_G(self.grid)))
        P = np.asarray(to_dense(self.projection_matrix))
        if factor.size != G.shape[0]:
            raise ValueError(
                "Alpha inertia factor/grid size mismatch: "
                f"factor={factor.shape}, G={G.shape}."
            )
        M_factor = P @ (factor[:, None] * G)
        return mu0 * asarray(M_factor)

    @cached_property
    def dtjr_radial_closure_operator(self) -> np.ndarray:
        """Construct the dt_jr radial-closure operator.

        Maps ``dt_jr`` to the radial-derivative closure term in the toroidal
        physics operator.

        Decomposition:
        - ``br_divergence_scale_term``: ``(d_r Br / Br) * f``
        - ``br_weighted_horizontal_advection_term``: ``-(1/R) * (1/Br) * (B_s · grad f)``
        - ``inv_br_gradient_correction_term``: ``-(1/R) * (B_s · grad(1/Br)) * f``
        """
        logger.info("Building dt_jr radial-closure operator...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(self._project_aux_square_operator_to_state(aux.dtjr_radial_closure_operator))

        if self.is_cs:
            inv_Rb = 1.0 / self.RI
            br_grid = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
            btheta_grid = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
            bphi_grid = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
            inv_br_grid = np.asarray(to_numpy(self.inverse_radial_field)).reshape(-1)

            # Horizontal advection of the state variable uses the configured
            # primary derivative basis.
            D_th_h, D_ph_h, _ = self.cs_grid_derivative_operators
            # Radial-closure coefficients can use a dedicated derivative basis
            # (defaults to the same primary basis).
            D_th_r, D_ph_r, _ = self.cs_radial_derivative_operators

            # div_Omega(B_s) = d/dtheta B_theta + cot(theta) B_theta + d/dphi B_phi
            theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
            cot_th = 1.0 / np.tan(theta_rad)
            horizontal_field_divergence = (D_th_r @ btheta_grid) + cot_th * btheta_grid + (D_ph_r @ bphi_grid)

            radial_field_radial_derivative = -inv_Rb * (2.0 * br_grid + horizontal_field_divergence)
            inv_br_gradient_theta = D_th_r @ inv_br_grid
            inv_br_gradient_phi = D_ph_r @ inv_br_grid

            inv_br_gradient_correction_factor = -inv_Rb * (
                btheta_grid * inv_br_gradient_theta + bphi_grid * inv_br_gradient_phi
            )
            inv_br_advection_scale = -inv_Rb * inv_br_grid

            horizontal_B_dot_grad_operator = (
                np.diag(btheta_grid) @ D_th_h
            ) + (np.diag(bphi_grid) @ D_ph_h)

            dtjr_closure_grid_operator = (
                np.diag(radial_field_radial_derivative * inv_br_grid)
                + np.diag(inv_br_gradient_correction_factor)
                + (np.diag(inv_br_advection_scale) @ horizontal_B_dot_grad_operator)
            )
            return np.asarray(dtjr_closure_grid_operator)

        # SH implementation
        inv_Rb = 1.0 / self.RI
        br_grid = to_numpy(self.b_field.vec.r).flatten()
        btheta_grid = to_numpy(self.b_field.vec.theta).flatten()
        bphi_grid = to_numpy(self.b_field.vec.phi).flatten()

        scalar_projection_matrix = to_dense(self.projection_matrix)
        scalar_evaluation_matrix = to_dense(self.basis.get_G(self.grid))
        gradient_theta_operator = to_dense(self.basis.get_G(self.grid, derivative="theta"))
        gradient_phi_operator = to_dense(self.basis.get_G(self.grid, derivative="phi"))

        btheta_coeffs = scalar_projection_matrix @ btheta_grid
        bphi_coeffs = scalar_projection_matrix @ bphi_grid

        theta_rad = np.deg2rad(to_numpy(self.grid.theta)).flatten()
        cot_th = 1.0 / np.tan(theta_rad)
        horizontal_field_divergence = (
            gradient_theta_operator @ btheta_coeffs
        ) + cot_th * btheta_grid + (gradient_phi_operator @ bphi_coeffs)

        radial_field_radial_derivative = -inv_Rb * (2.0 * br_grid + horizontal_field_divergence)
        inv_br_grid = to_numpy(self.inverse_radial_field).flatten()

        inv_br_coeffs = self.basis.from_grid_values(inv_br_grid, self.grid, "scalar")
        inv_br_gradient_theta = gradient_theta_operator @ inv_br_coeffs
        inv_br_gradient_phi = gradient_phi_operator @ inv_br_coeffs

        br_divergence_scale_term = (
            radial_field_radial_derivative * inv_br_grid
        )[:, None] * scalar_evaluation_matrix

        inv_br_gradient_correction_factor = -inv_Rb * (
            btheta_grid * inv_br_gradient_theta + bphi_grid * inv_br_gradient_phi
        )
        inv_br_gradient_correction_term = (
            inv_br_gradient_correction_factor
        )[:, None] * scalar_evaluation_matrix

        inv_br_advection_scale = -inv_Rb * inv_br_grid
        br_weighted_horizontal_advection_term = inv_br_advection_scale[:, None] * (
            btheta_grid[:, None] * gradient_theta_operator
            + bphi_grid[:, None] * gradient_phi_operator
        )

        dtjr_closure_grid_operator = (
            br_divergence_scale_term
            + br_weighted_horizontal_advection_term
            + inv_br_gradient_correction_term
        )

        # Project grid-operator back to coefficient space.
        return asarray(scalar_projection_matrix @ dtjr_closure_grid_operator)

    @cached_property
    def dtalpha_radial_closure_operator(self) -> np.ndarray:
        """Construct radial-closure map from ``dt_alpha`` to ``d_r(dt_jr)``.

        Using ``dt_jr = Br * dt_alpha`` and
            ``d_r(dt_jr) = (d_r Br) * dt_alpha - (1/R) * (B_s · grad(dt_alpha))``.
        """
        logger.info("Building dt_alpha radial-closure operator...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(
                self._project_aux_square_operator_to_state(aux.dtalpha_radial_closure_operator)
            )

        inv_Rb = 1.0 / self.RI
        br_grid = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        btheta_grid = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        bphi_grid = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)

        if self.is_cs:
            D_th_h, D_ph_h, _ = self.cs_grid_derivative_operators
            D_th_r, D_ph_r, _ = self.cs_radial_derivative_operators

            theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
            cot_th = 1.0 / np.tan(theta_rad)
            horizontal_field_divergence = (
                (D_th_r @ btheta_grid)
                + cot_th * btheta_grid
                + (D_ph_r @ bphi_grid)
            )
            radial_field_radial_derivative = -inv_Rb * (2.0 * br_grid + horizontal_field_divergence)

            horizontal_B_dot_grad_operator = (
                np.diag(btheta_grid) @ D_th_h
            ) + (np.diag(bphi_grid) @ D_ph_h)

            dtalpha_closure_grid_operator = (
                np.diag(radial_field_radial_derivative)
                - inv_Rb * horizontal_B_dot_grad_operator
            )
            return asarray(dtalpha_closure_grid_operator)

        scalar_projection_matrix = np.asarray(to_dense(self.projection_matrix))
        scalar_evaluation_matrix = np.asarray(to_dense(self.basis.get_G(self.grid)))
        gradient_theta_operator = np.asarray(
            to_dense(self.basis.get_G(self.grid, derivative="theta"))
        )
        gradient_phi_operator = np.asarray(
            to_dense(self.basis.get_G(self.grid, derivative="phi"))
        )

        btheta_coeffs = scalar_projection_matrix @ btheta_grid
        bphi_coeffs = scalar_projection_matrix @ bphi_grid

        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        cot_th = 1.0 / np.tan(theta_rad)
        horizontal_field_divergence = (
            gradient_theta_operator @ btheta_coeffs
        ) + cot_th * btheta_grid + (gradient_phi_operator @ bphi_coeffs)

        radial_field_radial_derivative = -inv_Rb * (2.0 * br_grid + horizontal_field_divergence)
        br_divergence_scale_term = (
            radial_field_radial_derivative
        )[:, None] * scalar_evaluation_matrix
        horizontal_advection_term = -inv_Rb * (
            btheta_grid[:, None] * gradient_theta_operator
            + bphi_grid[:, None] * gradient_phi_operator
        )
        dtalpha_closure_grid_operator = br_divergence_scale_term + horizontal_advection_term
        return asarray(scalar_projection_matrix @ dtalpha_closure_grid_operator)

    @cached_property
    def E_to_dtjr_forcing_matrix(self) -> np.ndarray:
        """Build matrix mapping E coefficients to dt_jr forcing.

        Since ``compute_dtjr_forcing_from_E`` is linear in ``E_coeffs``, we can
        represent it as:
            forcing = E_to_dtjr_forcing @ E_coeffs.flatten()

        Returns
        -------
        np.ndarray
            Matrix of shape ``(N, 2*N)`` mapping flattened ``E_coeffs`` to forcing.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            F_aux = np.asarray(aux.E_to_dtjr_forcing_matrix)
            return asarray(
                self._toroidal_closure_projector.project_vector_forcing_operator_to_state(F_aux)
            )

        if self.is_cs:
            return asarray(self._build_cs_E_to_dtjr_forcing_matrix())

        if (not self.is_cs) and (getattr(self.basis, "kind", "") == "SH"):
            return asarray(self._build_sh_E_to_dtjr_forcing_matrix())

        N = self.basis.index_length
        # E_coeffs has shape (2, N) - [poloidal, toroidal] potentials
        # Build matrix by applying compute_dtjr_forcing_from_E to each basis vector.
        
        E_to_dtjr_forcing = np.zeros((N, 2 * N))
        
        for i in range(2 * N):
            # Create basis vector
            e_i = np.zeros(2 * N)
            e_i[i] = 1.0
            E_i = e_i.reshape(2, N)
            
            # Apply the linear map
            forcing_i = self.compute_dtjr_forcing_from_E(E_i)
            E_to_dtjr_forcing[:, i] = to_numpy(forcing_i)
        
        return asarray(E_to_dtjr_forcing)

    @cached_property
    def E_to_er_rhs_matrix(self) -> np.ndarray:
        """Return grid map ``E_coeffs_flat -> rhs_Er = -(B_s . E_s)``."""
        N = self.basis.index_length
        G_vec = np.asarray(to_dense(self.basis.get_vector_basis_matrix(self.grid)))
        if G_vec.ndim != 4 or G_vec.shape[0] != 2 or G_vec.shape[2] != 2 or G_vec.shape[3] != N:
            raise ValueError(
                "Unexpected vector basis shape for E->Er RHS map: "
                f"{G_vec.shape}, expected (2, N_grid, 2, {N})."
            )
        V_th = np.hstack([G_vec[0, :, 0, :], G_vec[0, :, 1, :]])
        V_ph = np.hstack([G_vec[1, :, 0, :], G_vec[1, :, 1, :]])
        Bth = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        Bph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        if Bth.size != V_th.shape[0] or Bph.size != V_ph.shape[0]:
            raise ValueError("Grid field sizes are inconsistent in E->Er RHS map assembly.")
        return asarray(-((Bth[:, None] * V_th) + (Bph[:, None] * V_ph)))

    @cached_property
    def Er_grid_to_dtjr_forcing_matrix(self) -> np.ndarray:
        """Return grid map ``E_r(grid) -> dt_jr forcing(coeffs)``.

        This isolates the ``E_r``-dependent subset of the toroidal forcing
        assembly. It enables joint ``(dt_alpha, E_r)`` least-squares solves
        without double-counting terms already embedded in ``L_alpha``.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            K_aux = np.asarray(aux.Er_grid_to_dtjr_forcing_matrix)
            c2s = np.asarray(self._toroidal_closure_projector.closure_to_state_scalar_map)
            return asarray(c2s @ K_aux)

        Rb = float(self.RI)
        inv_Rb = 1.0 / Rb
        inv_Rb2 = 1.0 / (Rb**2)

        P = np.asarray(to_dense(self.projection_matrix))
        Br = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        Bth = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        Bph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)

        if self.is_cs:
            D_th_r, D_ph_r, L_grid_r = self.cs_radial_derivative_operators
            D_th_r = np.asarray(D_th_r)
            D_ph_r = np.asarray(D_ph_r)
            L_grid_r = np.asarray(L_grid_r)
            n_grid = int(D_th_r.shape[0])
            if (
                Br.size != n_grid
                or Bth.size != n_grid
                or Bph.size != n_grid
                or L_grid_r.shape[0] != n_grid
            ):
                raise ValueError("Grid field sizes are inconsistent in CS Er forcing map assembly.")

            S_from_Er = (
                (Br[:, None] * (inv_Rb + inv_Rb2)) * L_grid_r
                + (2.0 * inv_Rb2 * Bth[:, None]) * D_th_r
                + (2.0 * inv_Rb2 * Bph[:, None]) * D_ph_r
            )
            return asarray(P @ S_from_Er)

        if getattr(self.basis, "kind", "") == "SH":
            G = np.asarray(to_dense(self.basis.get_G(self.grid)))
            G_th = np.asarray(to_dense(self.basis.get_G(self.grid, derivative="theta")))
            G_ph = np.asarray(to_dense(self.basis.get_G(self.grid, derivative="phi")))
            l_arr = np.asarray(to_numpy(self.basis.n)).reshape(-1)
            ll1 = l_arr * (l_arr + 1.0)
            P_Er = P
            dEr_th_op = G_th @ P_Er
            dEr_ph_op = G_ph @ P_Er
            lap_Er_op = G @ ((-ll1)[:, None] * P_Er)
            S_from_Er = (
                (Br[:, None] * (inv_Rb + inv_Rb2)) * lap_Er_op
                + (2.0 * inv_Rb2 * Bth[:, None]) * dEr_th_op
                + (2.0 * inv_Rb2 * Bph[:, None]) * dEr_ph_op
            )
            return asarray(P @ S_from_Er)

        raise NotImplementedError(
            "Er_grid_to_dtjr_forcing_matrix is not implemented for basis "
            f"{getattr(self.basis, 'kind', type(self.basis).__name__)}."
        )

    def compute_forcing_vector_split(
        self,
        E_coeffs: np.ndarray,
        dt_jr_driver_coeffs: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return split toroidal forcing ``(rhs_no_er, er_rhs_grid)``.

        The full coefficient-space physics RHS is represented as
            ``rhs = rhs_no_er + K_er @ e_r``,
        where ``e_r`` is constrained by the ideal-closure relation
            ``Br * e_r = er_rhs_grid``.

        ``rhs_no_er`` already includes any driver subtraction ``-L * dt_jr_driver``.
        """
        E_flat = np.asarray(E_coeffs).reshape(-1)
        forcing_total = np.asarray(to_numpy(self.compute_dtjr_forcing_from_E(E_coeffs))).reshape(-1)
        er_rhs_grid = np.asarray(to_numpy(self.E_to_er_rhs_matrix @ E_flat)).reshape(-1)
        er_grid_particular = np.asarray(to_numpy(self.er_rhs_to_grid_operator @ er_rhs_grid)).reshape(-1)
        forcing_er_particular = np.asarray(
            to_numpy(self.Er_grid_to_dtjr_forcing_matrix @ er_grid_particular)
        ).reshape(-1)

        rhs_no_er = forcing_total - forcing_er_particular
        if dt_jr_driver_coeffs is not None:
            dtjr_operator = np.asarray(to_numpy(self.dtjr_physics_operator))
            rhs_no_er = rhs_no_er - dtjr_operator @ np.asarray(to_numpy(dt_jr_driver_coeffs)).reshape(-1)

        return asarray(rhs_no_er), asarray(er_rhs_grid)

    def _build_sh_E_to_dtjr_forcing_matrix(self) -> np.ndarray:
        """Build analytic SH forcing map ``E_coeffs -> forcing_dtjr``.

        This path keeps the derivative chain linear in coefficient space as far
        as possible (notably for ``div(E)`` / ``grad(div(E))``), and only uses
        grid projection where field products require it.
        """
        N = self.basis.index_length

        # Scalar basis matrices on the simulation grid.
        G = np.asarray(to_dense(self.basis.get_G(self.grid)))
        G_th = np.asarray(to_dense(self.basis.get_G(self.grid, derivative="theta")))
        G_ph = np.asarray(to_dense(self.basis.get_G(self.grid, derivative="phi")))
        P = np.asarray(to_dense(self.projection_matrix))

        # Vector basis tensor: (2, N_grid, 2, N_coeff).
        G_vec = np.asarray(to_dense(self.basis.get_vector_basis_matrix(self.grid)))
        if G_vec.ndim != 4 or G_vec.shape[0] != 2 or G_vec.shape[2] != 2 or G_vec.shape[3] != N:
            raise ValueError(
                "Unexpected vector basis shape for SH forcing map: "
                f"{G_vec.shape}, expected (2, N_grid, 2, {N})."
            )
        n_grid = int(G_vec.shape[1])

        # Map flattened E coefficients [pol; tor] to tangential components.
        V_th = np.hstack([G_vec[0, :, 0, :], G_vec[0, :, 1, :]])  # (N_grid, 2N)
        V_ph = np.hstack([G_vec[1, :, 0, :], G_vec[1, :, 1, :]])  # (N_grid, 2N)

        # Geometry and metric factors on grid.
        Br = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        Bth = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        Bph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)

        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        inv_sin2 = 1.0 / (sin_th_safe**2)

        if (
            Br.size != n_grid
            or Bth.size != n_grid
            or Bph.size != n_grid
            or cot_th.size != n_grid
            or inv_sin2.size != n_grid
        ):
            raise ValueError("Grid field sizes are inconsistent in SH forcing map assembly.")

        # SH scalar Laplacian factor at r=1: Delta_Omega Y = -l(l+1) Y.
        l_arr = np.asarray(to_numpy(self.basis.n)).reshape(-1)
        ll1 = l_arr * (l_arr + 1.0)

        # Helper projections for component derivatives / Laplacians.
        P_V_th = P @ V_th  # (N, 2N)
        P_V_ph = P @ V_ph  # (N, 2N)
        dEth_ph_op = G_ph @ P_V_th
        dEph_ph_op = G_ph @ P_V_ph
        lap_Eth_op = G @ ((-ll1)[:, None] * P_V_th)
        lap_Eph_op = G @ ((-ll1)[:, None] * P_V_ph)

        # Solve Br * Er = -(B_theta E_theta + B_phi E_phi) in weak form.
        rhs_Er_op = -((Bth[:, None] * V_th) + (Bph[:, None] * V_ph))
        Er_op = np.asarray(to_numpy(self.er_rhs_to_grid_operator)) @ rhs_Er_op
        P_Er = P @ Er_op
        dEr_th_op = G_th @ P_Er
        dEr_ph_op = G_ph @ P_Er
        lap_Er_op = G @ ((-ll1)[:, None] * P_Er)

        # Exact SH divergence in coefficient space, then evaluate gradients on grid.
        div_coeff_op = np.hstack([np.diag(ll1), np.zeros((N, N), dtype=float)])  # (N, 2N)
        div_E_op = G @ div_coeff_op
        grad_div_E_th_op = G_th @ div_coeff_op
        grad_div_E_ph_op = G_ph @ div_coeff_op

        # Vector Laplacian components (same formulas as the grid path).
        vec_lap_Eth_op = lap_Eth_op - (inv_sin2[:, None] * V_th) - (2.0 * cot_th[:, None] * dEph_ph_op)
        vec_lap_Eph_op = lap_Eph_op - (inv_sin2[:, None] * V_ph) + (2.0 * cot_th[:, None] * dEth_ph_op)

        # Assemble S_known operator: S = S_op @ E_coeffs_flat.
        Rb = float(self.RI)
        inv_Rb = 1.0 / Rb
        inv_Rb2 = 1.0 / (Rb**2)

        radial_field_coupling_op = Br[:, None] * (
            inv_Rb * div_E_op + (inv_Rb + inv_Rb2) * lap_Er_op
        )
        tangential_bracket_theta_op = inv_Rb2 * (
            2.0 * dEr_th_op + grad_div_E_th_op - 2.0 * V_th - vec_lap_Eth_op
        )
        tangential_bracket_phi_op = inv_Rb2 * (
            2.0 * dEr_ph_op + grad_div_E_ph_op - 2.0 * V_ph - vec_lap_Eph_op
        )
        tangential_field_coupling_op = (
            Bth[:, None] * tangential_bracket_theta_op
            + Bph[:, None] * tangential_bracket_phi_op
        )
        S_op = radial_field_coupling_op + tangential_field_coupling_op

        # Project back to scalar coefficients.
        return asarray(P @ S_op)

    def _build_cs_E_to_dtjr_forcing_matrix(self) -> np.ndarray:
        """Build analytic CS forcing map ``E_coeffs -> forcing_dtjr``.

        This mirrors ``compute_dtjr_forcing_from_E`` algebra in one direct
        linear assembly pass, avoiding per-column basis probing.
        """
        N = self.basis.index_length

        # Horizontal derivatives/laplacian use the configured forcing
        # derivative basis.
        D_th_h, D_ph_h, L_grid_h = self.cs_forcing_derivative_operators
        D_th_h = np.asarray(D_th_h)
        D_ph_h = np.asarray(D_ph_h)
        L_grid_h = np.asarray(L_grid_h)
        # Radial-closure derivatives (Er and related closures) may use an
        # explicit radial derivative basis on the same grid.
        D_th_r, D_ph_r, L_grid_r = self.cs_radial_derivative_operators
        D_th_r = np.asarray(D_th_r)
        D_ph_r = np.asarray(D_ph_r)
        L_grid_r = np.asarray(L_grid_r)
        P = np.asarray(to_dense(self.projection_matrix))

        # Vector basis tensor: (2, N_grid, 2, N_coeff).
        G_vec = np.asarray(to_dense(self.basis.get_vector_basis_matrix(self.grid)))
        if G_vec.ndim != 4 or G_vec.shape[0] != 2 or G_vec.shape[2] != 2 or G_vec.shape[3] != N:
            raise ValueError(
                "Unexpected vector basis shape for CS forcing map: "
                f"{G_vec.shape}, expected (2, N_grid, 2, {N})."
            )
        n_grid = int(G_vec.shape[1])

        # Map flattened E coefficients [pol; tor] to tangential components.
        V_th = np.hstack([G_vec[0, :, 0, :], G_vec[0, :, 1, :]])  # (N_grid, 2N)
        V_ph = np.hstack([G_vec[1, :, 0, :], G_vec[1, :, 1, :]])  # (N_grid, 2N)

        # Geometry and metric factors on grid.
        Br = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        Bth = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        Bph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)

        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        inv_sin2 = 1.0 / (sin_th_safe**2)

        if (
            Br.size != n_grid
            or Bth.size != n_grid
            or Bph.size != n_grid
            or cot_th.size != n_grid
            or inv_sin2.size != n_grid
        ):
            raise ValueError("Grid field sizes are inconsistent in CS forcing map assembly.")

        # Solve Br * Er = -(B_theta E_theta + B_phi E_phi) in weak form.
        rhs_Er_op = -((Bth[:, None] * V_th) + (Bph[:, None] * V_ph))
        Er_op = np.asarray(to_numpy(self.er_rhs_to_grid_operator)) @ rhs_Er_op

        # Derivative / Laplacian chain.
        # - Er-related operators are radial-closure terms.
        # - E_s horizontal operators remain CS-native.
        dEr_th_op = D_th_r @ Er_op
        dEr_ph_op = D_ph_r @ Er_op
        lap_Er_op = L_grid_r @ Er_op

        dEth_th_op = D_th_h @ V_th
        dEth_ph_op = D_ph_h @ V_th
        dEph_ph_op = D_ph_h @ V_ph
        lap_Eth_op = L_grid_h @ V_th
        lap_Eph_op = L_grid_h @ V_ph

        # D_ph_h already carries 1/sin(theta) scaling on CS basis.
        div_E_s_op = dEth_th_op + (cot_th[:, None] * V_th) + dEph_ph_op
        grad_div_E_th_op = D_th_h @ div_E_s_op
        grad_div_E_ph_op = D_ph_h @ div_E_s_op

        vec_lap_Eth_op = lap_Eth_op - (inv_sin2[:, None] * V_th) - (2.0 * cot_th[:, None] * dEph_ph_op)
        vec_lap_Eph_op = lap_Eph_op - (inv_sin2[:, None] * V_ph) + (2.0 * cot_th[:, None] * dEth_ph_op)

        # Assemble S_known operator: S = S_op @ E_coeffs_flat.
        Rb = float(self.RI)
        inv_Rb = 1.0 / Rb
        inv_Rb2 = 1.0 / (Rb**2)

        radial_field_coupling_op = Br[:, None] * (
            inv_Rb * div_E_s_op + (inv_Rb + inv_Rb2) * lap_Er_op
        )
        tangential_bracket_theta_op = inv_Rb2 * (
            2.0 * dEr_th_op + grad_div_E_th_op - 2.0 * V_th - vec_lap_Eth_op
        )
        tangential_bracket_phi_op = inv_Rb2 * (
            2.0 * dEr_ph_op + grad_div_E_ph_op - 2.0 * V_ph - vec_lap_Eph_op
        )
        tangential_field_coupling_op = (
            Bth[:, None] * tangential_bracket_theta_op
            + Bph[:, None] * tangential_bracket_phi_op
        )
        S_op = radial_field_coupling_op + tangential_field_coupling_op

        return asarray(P @ S_op)

    def compute_dtjr_forcing_from_E(self, E_coeffs: np.ndarray) -> np.ndarray:
        """Compute dt_jr forcing from known E-field coefficients.

        Computes:
            forcing_lm = Projection(S_known)
        where ``S_known`` is derived from Faraday and Gauss identities after
        eliminating radial derivatives of ``E``.

        Notes
        -----
        The sign convention is inherited from the basis projection operator.
        In this code path we apply the basis projection directly:
            ``K = P @ S_known``.
        """
        if self.is_cs:
            E_flat = np.asarray(E_coeffs).reshape(-1)
            return asarray(self.E_to_dtjr_forcing_matrix @ E_flat)

        if (not self.is_cs) and (getattr(self.basis, "kind", "") == "SH"):
            E_flat = np.asarray(E_coeffs).reshape(-1)
            return asarray(self.E_to_dtjr_forcing_matrix @ E_flat)

        # 1. Setup constants
        Rb = self.RI
        inv_Rb = 1.0 / Rb
        inv_Rb2 = 1.0 / (Rb**2)
        
        # 2. Get evaluation and projection matrices (Cached)
        G = to_dense(self.basis.get_G(self.grid))
        G_th = to_dense(self.basis.get_G(self.grid, derivative="theta"))
        G_ph = to_dense(self.basis.get_G(self.grid, derivative="phi"))
        P = to_dense(self.projection_matrix)

        # 3. Evaluate E_S and Er on grid
        # E_coeffs contains [Poloidal, Toroidal] potentials
        Eth_grid, Eph_grid = self.basis.evaluate(E_coeffs, self.grid, vector_type="tangential")
        
        # Solve Br * Er = -(B_s . E_s) in weak form.
        Br = to_numpy(self.b_field.vec.r)
        Bth = to_numpy(self.b_field.vec.theta)
        Bph = to_numpy(self.b_field.vec.phi)
        rhs_Er = -(Bth * Eth_grid + Bph * Eph_grid)
        Er_grid = np.asarray(to_numpy(self.er_rhs_to_grid_operator)) @ np.asarray(rhs_Er).reshape(-1)
        
        # 4. Helpers for grid derivatives via spectral projection
        def get_derivs(f_val):
            c = P @ f_val
            return G_th @ c, G_ph @ c

        def get_laplacian(f_val):
            c = P @ f_val
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
        # S_known = Br [ 1/Rb div_E_S + (1/Rb + 1/Rb^2) lap_Er ] + B_s · bracket
        radial_field_coupling_term = Br * (
            inv_Rb * div_E_S + (inv_Rb + inv_Rb2) * lap_Er
        )

        tangential_bracket_theta = inv_Rb2 * (
            2.0 * dEr_th + grad_div_E_th - 2.0 * Eth_grid - vec_lap_Eth
        )
        tangential_bracket_phi = inv_Rb2 * (
            2.0 * dEr_ph + grad_div_E_ph - 2.0 * Eph_grid - vec_lap_Eph
        )
        tangential_field_coupling_term = (
            Bth * tangential_bracket_theta + Bph * tangential_bracket_phi
        )

        S_known = radial_field_coupling_term + tangential_field_coupling_term
        
        # 7. Project forcing scalar field to basis coefficients.
        #
        # Use the basis projection operator directly so both SH and CS follow the
        # same coefficient convention:
        #   K = P @ S_known
        #
        # For SH with exact quadrature this is equivalent to the integral form.
        # For CS nodal bases this avoids an extra area-weight scaling that can
        # otherwise suppress forcing as resolution increases.
        K = P @ S_known
        
        return asarray(K)

    @cached_property
    def advection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Construct Advection Matrices M0 and M1.
        
        M0_lm,l'm' = Integrate[ Y_lm * (B0s . grad Y_l'm') * (2 mu0 / (l'(l'+1))) ]
        M1_lm,l'm' = Integrate[ Y_lm * (B0s . grad Y_l'm') * (-mu0 Rb / (l'(l'+1))) ]
        
        Both share the integral structure:
           Int = Integrate[ Y_lm * (B0s . grad Y_l'm') ]
        
        And then scaling depends on column index l'.
        """
        logger.info("Building Advection Matrices M0 and M1...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            M0_aux, M1_aux = aux.advection_matrices
            M0 = self._project_aux_square_operator_to_state(np.asarray(M0_aux))
            M1 = self._project_aux_square_operator_to_state(np.asarray(M1_aux))
            return asarray(M0), asarray(M1)
        
        if self.is_cs:
            # CS Implementation:
            # M0 = 2 mu0 * (Laplacian^-1) @ advection_matrix
            # M1 = -mu0 Rb * (Laplacian^-1) @ advection_matrix

            # 1. Build advection operator directly in collocation form.
            D_th, D_ph, _ = self.cs_grid_derivative_operators
            
            B_theta = to_numpy(self.b_field.vec.theta).flatten()
            B_phi = to_numpy(self.b_field.vec.phi).flatten()
            
            horizontal_B_dot_grad_operator = (
                np.diag(B_theta) @ D_th
            ) + (np.diag(B_phi) @ D_ph)

            horizontal_advection_matrix = horizontal_B_dot_grad_operator
            
            # 2. Gauge-fixed Laplacian inverse (mean-zero subspace)
            L_inv = self.cs_laplacian_inverse

            # Apply Inverse Laplacian
            Inv_Lap_Op = -L_inv
            
            # Combine
            scale_M0 = 2.0 * mu0
            scale_M1 = -mu0 * self.RI
            
            M0 = scale_M0 * (Inv_Lap_Op @ horizontal_advection_matrix)
            M1 = scale_M1 * (Inv_Lap_Op @ horizontal_advection_matrix)
            
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
        horizontal_B_dot_grad_operator = (B_theta[:, None] * G_th) + (B_phi[:, None] * G_ph)
        
        # Integrate against Y_i
        # Matrix = G^T @ W @ horizontal_B_dot_grad_operator
        # Optimize: (G^T * w) @ horizontal_B_dot_grad_operator
        weighted_horizontal_advection_matrix = (G.T * weights) @ horizontal_B_dot_grad_operator
        
        # 2. Apply Column Scalings
        l_arr = to_numpy(self.basis.n).flatten()
        ll1 = l_arr * (l_arr + 1.0)
        # Avoid division by zero for l=0 (monopole has no stiffness/current)
        ll1_inv = np.zeros_like(ll1)
        ll1_inv[ll1 > 0] = 1.0 / ll1[ll1 > 0]
        
        scale_M0 = 2.0 * mu0 * ll1_inv
        scale_M1 = -mu0 * self.RI * ll1_inv
        
        M0 = weighted_horizontal_advection_matrix * scale_M0[None, :]
        M1 = weighted_horizontal_advection_matrix * scale_M1[None, :]
        
        return asarray(M0), asarray(M1)

    @cached_property
    def dtjr_physics_operator(self) -> np.ndarray:
        """Assemble the dt_jr physics operator ``L = C + M0 + M1 @ closure``."""
        M0, M1 = self.advection_matrices
        closure = to_numpy(self.dtjr_radial_closure_operator)
        C = to_numpy(self.inertia_matrix)

        coupling_from_closure = M1 @ closure
        return asarray(C + M0 + coupling_from_closure)

    @cached_property
    def dtalpha_physics_operator(self) -> np.ndarray:
        """Assemble the alpha-space physics operator ``L_alpha``.

        Unknown is ``dt_alpha``. The assembled operator corresponds to:
            ``(C_alpha + M0 @ T_alpha_to_jr + M1 @ D1_alpha) @ dt_alpha = K``.
        """
        M0, M1 = self.advection_matrices
        C_alpha = np.asarray(to_numpy(self.inertia_matrix_alpha))
        alpha_to_jr = np.asarray(to_numpy(self.alpha_to_jr_coeff_operator))
        closure_alpha = np.asarray(to_numpy(self.dtalpha_radial_closure_operator))

        coupling_from_jr = np.asarray(to_numpy(M0)) @ alpha_to_jr
        coupling_from_closure = np.asarray(to_numpy(M1)) @ closure_alpha
        return asarray(C_alpha + coupling_from_jr + coupling_from_closure)

    @cached_property
    def _dtalpha_grid_residual_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R_grid, A_grid)`` for residuals ``A_grid x - R_grid b``.

        Unknown ``x`` is ``dt_alpha`` in coefficient space, with
        ``A_grid = R_grid @ L_alpha``.
        """
        L_alpha = np.asarray(to_numpy(self.dtalpha_physics_operator))
        R_grid = to_dense(self.basis.get_G(self.grid))
        if scipy.sparse.issparse(R_grid):
            R_grid = R_grid.toarray()
        R_grid = np.asarray(R_grid)

        if R_grid.ndim != 2:
            R_grid = R_grid.reshape(R_grid.shape[0], -1)
        if R_grid.shape[1] != L_alpha.shape[0]:
            raise ValueError(
                "Grid residual map shape mismatch: "
                f"R_grid={R_grid.shape}, L_alpha={L_alpha.shape}."
            )

        A_grid = R_grid @ L_alpha
        return R_grid, A_grid

    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Get projection matrix from Basis (Grid -> Coeffs)."""
        return self.basis.construct_scalar_projection_matrix(self.grid)

    @cached_property
    def alpha_to_jr_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``alpha`` to ``jr``.

        For ``alpha = jr / Br`` and static background field,
            ``jr = Br * alpha`` pointwise on the grid.
        In coefficient space this is assembled in weak form:
            ``T_alpha_to_jr = P @ diag(Br_grid) @ G``.
        """
        G = np.asarray(to_dense(self.basis.get_G(self.grid)))
        P = np.asarray(to_dense(self.projection_matrix))
        Br = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        if Br.size != G.shape[0]:
            raise ValueError(
                "alpha_to_jr assembly mismatch: "
                f"Br={Br.shape}, G={G.shape}."
            )
        return asarray(P @ (Br[:, None] * G))

    @cached_property
    def jr_to_alpha_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``jr`` to ``alpha``.

        For ``Br * alpha = jr`` we use the same weak-form minimum-norm branch
        selector as the ``E_r`` closure:
            ``argmin_alpha ||W^(1/2)(Br*alpha - jr)||^2 + floor^2 ||W^(1/2)alpha||^2``.
        The returned coefficient-space map composes the cached grid-``jr`` to
        coefficient-``alpha`` map with the basis evaluation matrix for ``jr``.
        """
        br_solver = self._br_constraint_solver
        G = np.asarray(br_solver.G)
        if G.ndim != 2:
            G = G.reshape(G.shape[0], -1)
        T_grid_to_alpha = np.asarray(br_solver.rhs_to_coeff_operator)
        if T_grid_to_alpha.ndim != 2 or T_grid_to_alpha.shape[1] != G.shape[0]:
            raise ValueError(
                "jr_to_alpha assembly mismatch: "
                f"T_grid_to_alpha={T_grid_to_alpha.shape}, G={G.shape}."
            )
        return asarray(T_grid_to_alpha @ G)

    def _get_cs_psi_gauge_rows(self, n_coeff: int, use_pinning: bool) -> np.ndarray:
        """Return hard psi gauge rows for direct dpsi solves.

        Policy:
            - CS basis: optional mean-zero hard row when ``use_pinning`` is true.
            - SH/other bases: no hard psi gauge rows.
        """
        if not (self.is_cs and bool(use_pinning)):
            return np.zeros((0, int(n_coeff)), dtype=float)

        if hasattr(self.basis, "get_scalar_gauge_constraint_matrix"):
            row = np.asarray(
                self.basis.get_scalar_gauge_constraint_matrix(
                    n_coeff=int(n_coeff),
                    mode="mean_zero",
                )
            )
            if row.ndim == 1:
                row = row.reshape(1, -1)
            if row.ndim == 2 and row.shape[1] == int(n_coeff) and row.shape[0] > 0:
                return row.astype(float, copy=False)

        row = np.ones((1, int(n_coeff)), dtype=float)
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row = row / norm
        return row

    # -------------------------------------------------------------------------
    # Least-Squares Problem Construction (aligned with PoloidalSystemMatrices)
    # -------------------------------------------------------------------------

    def build_least_squares_problem(
        self,
        jr_map_operator: np.ndarray,
        constraint_scaling: float = 1.0,
        regularization_lambda: float = 0.0,
        weighting: str = "none",
    ) -> "LeastSquaresProblem":
        """Build the least-squares problem for dt_jr (toroidal induction).

        The problem structure is:
            minimize || A @ dt_jr - b ||^2

        Where A consists of:
            1. Physics equation: L_dtjr @ dt_jr = forcing_dtjr
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
        weighting : str
            Weighting strategy for handling equatorial singularity (Br -> 0).
            Options: "none", "linear" (sqrt|Br|), "quadratic" (|Br|).

        Returns
        -------
        LeastSquaresProblem
            The assembled least-squares problem.
        """
        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import as_linear_map

        operators = []
        data_shapes = []
        sqrt_weights = []

        # 1. Physics equation: L_dtjr @ dt_jr = forcing_dtjr
        op_L = as_linear_map(self.dtjr_physics_operator)
        operators.append(op_L)
        data_shapes.append((op_L.shape[0],))

        # Calculate sqrt-weights for physics equation (Br-based weighting).
        physics_weight = self._build_physics_sqrt_weight(op_L.shape[0], weighting)
        sqrt_weights.append(physics_weight)

        # 2. Apex current constraint (interhemispheric + driver matching)
        op_constraint = as_linear_map(jr_map_operator)
        op_constraint = op_constraint.with_scaling(constraint_scaling)
        operators.append(op_constraint)
        data_shapes.append((op_constraint.shape[0],))
        sqrt_weights.append(None)  # No weighting for constraint

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
            sqrt_weights=sqrt_weights,
            regularization_matrices=reg_ops,
            regularization_weights=reg_weights,
        )

    def compute_forcing_vector(
        self,
        E_coeffs: np.ndarray,
        dt_jr_driver_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute RHS for the physics equation term.

        RHS = forcing - L_dtjr @ dt_jr_driver
        
        Where forcing is computed from E-field and L_dtjr @ dt_jr_driver accounts
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
        # Compute forcing from E-field
        forcing = self.compute_dtjr_forcing_from_E(E_coeffs)
        forcing = to_numpy(forcing)

        # Subtract driver contribution if present
        if dt_jr_driver_coeffs is not None:
            dtjr_operator = to_numpy(self.dtjr_physics_operator)
            L_driver = dtjr_operator @ to_numpy(dt_jr_driver_coeffs)
            forcing = forcing - L_driver

        return asarray(forcing)

    def _get_dtjr_problem_bundle(
        self,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
    ) -> dict[str, Any]:
        """Build/fetch weighted LS bundle for unknown ``dt_alpha``."""
        cache_key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
        )
        cached = self._dtjr_problem_cache.get(cache_key)
        if cached is not None:
            return cached

        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import as_linear_map, diagonal_linear_map

        dtalpha_operator = np.asarray(to_numpy(self.dtalpha_physics_operator))
        n_coeff = dtalpha_operator.shape[0]
        R_grid, A_grid = self._dtalpha_grid_residual_maps
        alpha_to_jr = np.asarray(to_numpy(self.alpha_to_jr_coeff_operator))
        op_A_grid = as_linear_map(A_grid)
        physics_weight = self._build_physics_sqrt_weight(op_A_grid.shape[0], weighting)

        operators = [op_A_grid]
        data_shapes = [(op_A_grid.shape[0],)]
        sqrt_weights = [physics_weight]

        if penalty_operator is not None and penalty_scaling > 0:
            op_penalty = as_linear_map(penalty_operator).with_scaling(penalty_scaling)
            operators.append(op_penalty)
            data_shapes.append((op_penalty.shape[0],))
            sqrt_weights.append(None)

        # Use explicit data rows for Tikhonov regularization so lambda remains
        # in absolute physical units (avoid automatic rescaling in
        # LeastSquaresProblem.regularization_weights).
        if regularization_lambda > 0:
            op_reg = diagonal_linear_map(np.ones(n_coeff)).with_scaling(
                float(np.sqrt(max(regularization_lambda, 0.0)))
            )
            operators.append(op_reg)
            data_shapes.append((op_reg.shape[0],))
            sqrt_weights.append(None)

        problem = LeastSquaresProblem(
            A=operators,
            solution_shape=n_coeff,
            data_shapes=data_shapes,
            sqrt_weights=sqrt_weights,
        )
        bundle = {
            "problem": problem,
            "R_grid": np.asarray(R_grid),
            "alpha_to_jr": np.asarray(alpha_to_jr),
            "n_coeff": int(n_coeff),
            "grid_rows": int(op_A_grid.shape[0]),
        }
        self._dtjr_problem_cache[cache_key] = bundle
        return bundle

    def _resolve_dtjr_rcond(self, n_coeff: int, hinv_rtol: float) -> float:
        """Resolve pseudoinverse cutoff used by constrained elimination."""
        if hinv_rtol > 0:
            return max(float(hinv_rtol), 0.0)
        rcond = self._default_pinv_rcond((n_coeff, n_coeff))
        logger.info(
            "Auto hard-solve rtol (default pseudoinverse cutoff): %.3e",
            float(rcond),
        )
        return float(max(rcond, 0.0))

    @staticmethod
    def _coeff_rhs_to_grid_rhs(R_grid: np.ndarray, rhs_coeffs: np.ndarray) -> np.ndarray:
        """Map coefficient-space RHS columns to grid-space RHS columns."""
        rhs_arr = np.asarray(to_numpy(rhs_coeffs))
        if rhs_arr.ndim == 1:
            return R_grid @ rhs_arr.reshape(-1, 1)
        rhs_2d = rhs_arr.reshape(rhs_arr.shape[0], -1)
        return R_grid @ rhs_2d

    def _get_dpsi_problem_bundle(
        self,
        *,
        m_imp_to_jr_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
    ) -> dict[str, Any]:
        """Build/fetch weighted LS bundle for unknown ``dpsi/dt``.

        Residual physics is kept in ``dt_alpha``-grid space:
            ``A_grid_psi * dpsi_dt - R_grid * rhs_physics``.
        """
        cache_key = (
            id(m_imp_to_jr_operator),
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
        )
        cached = self._dpsi_problem_cache.get(cache_key)
        if cached is not None:
            return cached

        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import as_linear_map

        m_to_jr = np.asarray(to_dense(as_linear_map(m_imp_to_jr_operator)))
        if m_to_jr.ndim != 2:
            m_to_jr = m_to_jr.reshape(m_to_jr.shape[0], -1)

        n_coeff = int(m_to_jr.shape[1])
        if m_to_jr.shape[0] != n_coeff:
            raise ValueError(
                "m_imp_to_jr operator must be square in coefficient space for direct dpsi solve: "
                f"shape={m_to_jr.shape}."
            )

        jr_to_alpha = np.asarray(to_numpy(self.jr_to_alpha_coeff_operator))
        if jr_to_alpha.shape != (n_coeff, n_coeff):
            raise ValueError(
                "jr_to_alpha size mismatch in direct dpsi bundle: "
                f"T={jr_to_alpha.shape}, n={n_coeff}."
            )
        psi_to_alpha = jr_to_alpha @ m_to_jr

        R_grid, _ = self._dtalpha_grid_residual_maps
        L_alpha = np.asarray(to_numpy(self.dtalpha_physics_operator))
        A_grid_psi = np.asarray(R_grid) @ (L_alpha @ psi_to_alpha)

        op_A_grid = as_linear_map(A_grid_psi)
        physics_weight = self._build_physics_sqrt_weight(op_A_grid.shape[0], weighting)

        operators = [op_A_grid]
        data_shapes = [(op_A_grid.shape[0],)]
        sqrt_weights = [physics_weight]

        if penalty_operator is not None and penalty_scaling > 0:
            op_penalty_base = as_linear_map(penalty_operator)
            if int(op_penalty_base.shape[1]) != n_coeff:
                raise ValueError(
                    "Penalty operator width mismatch for direct dpsi solve: "
                    f"penalty={op_penalty_base.shape}, n={n_coeff}."
                )
            op_penalty = (op_penalty_base @ as_linear_map(psi_to_alpha)).with_scaling(
                float(penalty_scaling)
            )
            operators.append(op_penalty)
            data_shapes.append((op_penalty.shape[0],))
            sqrt_weights.append(None)

        # Keep regularization in absolute dt_alpha units:
        #   lambda * ||dt_alpha||^2 = lambda * ||T_psi_to_alpha dpsi||^2.
        if regularization_lambda > 0:
            op_reg = as_linear_map(psi_to_alpha).with_scaling(
                float(np.sqrt(max(regularization_lambda, 0.0)))
            )
            operators.append(op_reg)
            data_shapes.append((op_reg.shape[0],))
            sqrt_weights.append(None)

        problem = LeastSquaresProblem(
            A=operators,
            solution_shape=n_coeff,
            data_shapes=data_shapes,
            sqrt_weights=sqrt_weights,
        )
        bundle = {
            "problem": problem,
            "R_grid": np.asarray(R_grid),
            "n_coeff": n_coeff,
            "m_to_jr": np.asarray(m_to_jr),
        }
        self._dpsi_problem_cache[cache_key] = bundle
        return bundle

    def _solve_dpsi_problem(
        self,
        *,
        rhs_physics_coeffs: np.ndarray,
        m_imp_to_jr_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        equality_operator: Any = None,
        equality_rhs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve weighted direct dpsi LS problem with optional exact equalities."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        bundle = self._get_dpsi_problem_bundle(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        problem = bundle["problem"]
        R_grid = np.asarray(bundle["R_grid"])
        n_coeff = int(bundle["n_coeff"])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)

        rhs_grid = self._coeff_rhs_to_grid_rhs(R_grid, rhs_physics_coeffs)
        rhs_grid_arr = np.asarray(rhs_grid)
        if rhs_grid_arr.ndim == 1:
            rhs_grid_arr = rhs_grid_arr.reshape(-1, 1)
        n_scenarios = int(rhs_grid_arr.shape[1])
        rhs_terms = [rhs_grid_arr]
        for term_index in range(1, int(problem.num_data_terms)):
            n_rows = int(problem.A[term_index].num_rows)
            rhs_terms.append(np.zeros((n_rows, n_scenarios), dtype=rhs_grid_arr.dtype))

        solver = LeastSquaresSolver(
            solver=self.dtjr_solver,
            tolerance=max(rcond, self.dtjr_tolerance),
            preconditioner=self.dtjr_preconditioner,
        )
        preconditioner = None
        preconditioner_type = self.dtjr_preconditioner
        if preconditioner_type is None and self.dtjr_solver in ("cg", "normal"):
            # Direct dpsi solves are often substantially stiffer than dt_jr solves.
            # Use a deterministic Jacobi preconditioner by default for normal/CG.
            preconditioner_type = "jacobi"
        if equality_operator is None and preconditioner_type is not None:
            preconditioner = solver.build_preconditioner(
                problem=problem,
                preconditioner_type=preconditioner_type,
                num_scenarios=n_scenarios,
                pinv_rcond=rcond,
            )
        sol = solver.solve(
            problem,
            rhs_terms,
            preconditioner=preconditioner,
            equality_operator=equality_operator,
            equality_rhs=equality_rhs,
            elimination_rcond=rcond,
        )
        return np.asarray(sol)

    def _solve_dtjr_problem(
        self,
        rhs_physics_coeffs: np.ndarray,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        equality_operator: Any = None,
        equality_rhs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve weighted dt_jr LS problem with optional exact equalities."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        bundle = self._get_dtjr_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        problem = bundle["problem"]
        R_grid = bundle["R_grid"]
        n_coeff = int(bundle["n_coeff"])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)

        rhs_grid = self._coeff_rhs_to_grid_rhs(R_grid, rhs_physics_coeffs)
        rhs_grid_arr = np.asarray(rhs_grid)
        if rhs_grid_arr.ndim == 1:
            rhs_grid_arr = rhs_grid_arr.reshape(-1, 1)
        n_scenarios = int(rhs_grid_arr.shape[1])
        rhs_terms = [rhs_grid_arr]
        for term_index in range(1, int(problem.num_data_terms)):
            n_rows = int(problem.A[term_index].num_rows)
            rhs_terms.append(np.zeros((n_rows, n_scenarios), dtype=rhs_grid_arr.dtype))

        solver = LeastSquaresSolver(
            solver=self.dtjr_solver,
            tolerance=max(rcond, self.dtjr_tolerance),
            preconditioner=self.dtjr_preconditioner,
        )
        preconditioner = None
        if equality_operator is None and self.dtjr_preconditioner is not None:
            preconditioner = solver.build_preconditioner(
                problem=problem,
                preconditioner_type=self.dtjr_preconditioner,
                num_scenarios=n_scenarios,
                pinv_rcond=rcond,
            )
        sol = solver.solve(
            problem,
            rhs_terms,
            preconditioner=preconditioner,
            equality_operator=equality_operator,
            equality_rhs=equality_rhs,
            elimination_rcond=rcond,
        )
        return np.asarray(sol)

    def _get_unconstrained_dtjr_map_cached(
        self,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Return cached dense map ``rhs_physics -> dt_jr`` (unconstrained)."""
        bundle = self._get_dtjr_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        n_coeff = int(bundle["n_coeff"])
        alpha_to_jr = np.asarray(bundle["alpha_to_jr"])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)
        key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            self._dtjr_solver_signature(),
        )
        cached = self._dtjr_unconstrained_map_cache.get(key)
        if cached is not None:
            return cached
        alpha_map = self._get_unconstrained_dtalpha_map_cached(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        x_map = alpha_to_jr @ alpha_map
        self._dtjr_unconstrained_map_cache[key] = x_map
        return x_map

    def _get_unconstrained_dtalpha_map_cached(
        self,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Return cached dense map ``rhs_physics -> dt_alpha`` (unconstrained)."""
        bundle = self._get_dtjr_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        n_coeff = int(bundle["n_coeff"])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)
        key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            self._dtjr_solver_signature(),
        )
        cached = self._dtalpha_unconstrained_map_cache.get(key)
        if cached is not None:
            return cached
        rhs_physics_basis = np.eye(n_coeff, dtype=float)
        alpha_map = np.asarray(
            self._solve_dtjr_problem(
                rhs_physics_coeffs=rhs_physics_basis,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        ).reshape(n_coeff, n_coeff)
        self._dtalpha_unconstrained_map_cache[key] = alpha_map
        return alpha_map

    def _get_constrained_dtjr_maps(
        self,
        *,
        jr_map_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Return cached constrained maps for one-shot hard-constrained solves.

        Returns dense matrices:
            ``M_phys_alpha``: physics RHS -> constrained homogeneous ``dt_alpha`` response
            ``M_corr_alpha``: constraint RHS -> constrained inhomogeneous ``dt_alpha`` response
            ``M_phys``: physics RHS -> constrained homogeneous response
            ``M_corr``: constraint RHS -> constrained inhomogeneous response
            ``C``: primary hard constraint operator used by RHS ``d``

        The constrained solution is represented as:
            x = M_phys @ rhs_physics + M_corr @ rhs_constraint
        with hard equalities enforced on ``C x``.
        """
        C = np.asarray(to_dense(as_linear_map(jr_map_operator)))
        if C.ndim != 2:
            C = C.reshape(C.shape[0], -1)
        n_coeff = int(C.shape[1])
        alpha_to_jr = np.asarray(to_numpy(self.alpha_to_jr_coeff_operator))
        if alpha_to_jr.shape != (n_coeff, n_coeff):
            raise ValueError(
                "alpha_to_jr size mismatch in constrained dt_jr maps: "
                f"T={alpha_to_jr.shape}, n={n_coeff}."
            )
        m_constraints = int(C.shape[0])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)
        key = (
            id(jr_map_operator),
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            self._dtjr_solver_signature(),
        )
        cached = self._dtjr_constrained_maps_cache.get(key)
        if cached is not None:
            return cached

        C_alpha = C @ alpha_to_jr

        M_phys_alpha = np.asarray(
            self._solve_dtjr_problem(
                rhs_physics_coeffs=np.eye(n_coeff, dtype=float),
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                equality_operator=C_alpha,
                equality_rhs=np.zeros(m_constraints, dtype=float),
            )
        ).reshape(n_coeff, n_coeff)
        M_phys = alpha_to_jr @ M_phys_alpha

        if m_constraints > 0:
            M_corr_alpha = np.asarray(
                self._solve_dtjr_problem(
                    rhs_physics_coeffs=np.zeros((n_coeff, m_constraints), dtype=float),
                    weighting=weighting,
                    regularization_lambda=regularization_lambda,
                    penalty_operator=penalty_operator,
                    penalty_scaling=penalty_scaling,
                    hinv_rtol=hinv_rtol,
                    equality_operator=C_alpha,
                    equality_rhs=np.eye(m_constraints, dtype=float),
                )
            ).reshape(n_coeff, m_constraints)
            M_corr = alpha_to_jr @ M_corr_alpha
        else:
            M_corr_alpha = np.zeros((n_coeff, 0), dtype=float)
            M_corr = np.zeros((n_coeff, 0), dtype=float)

        maps = {
            "C": C,
            "C_alpha": C_alpha,
            "M_phys_alpha": np.asarray(M_phys_alpha),
            "M_corr_alpha": np.asarray(M_corr_alpha),
            "M_phys": np.asarray(M_phys),
            "M_corr": np.asarray(M_corr),
        }
        self._dtjr_constrained_maps_cache[key] = maps
        return maps

    def _get_unconstrained_dpsi_map_cached(
        self,
        *,
        m_imp_to_jr_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Return cached dense map ``rhs_physics -> dpsi/dt`` (unconstrained)."""
        bundle = self._get_dpsi_problem_bundle(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        n_coeff = int(bundle["n_coeff"])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)
        key = (
            id(m_imp_to_jr_operator),
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            bool(use_pinning),
            self._dtjr_solver_signature(),
        )
        cached = self._dpsi_unconstrained_map_cache.get(key)
        if cached is not None:
            return cached

        gauge_rows = self._get_cs_psi_gauge_rows(n_coeff, use_pinning)
        equality_operator = gauge_rows if gauge_rows.shape[0] > 0 else None
        equality_rhs = np.zeros(gauge_rows.shape[0], dtype=float) if gauge_rows.shape[0] > 0 else None

        rhs_physics_basis = np.eye(n_coeff, dtype=float)
        dpsi_map = np.asarray(
            self._solve_dpsi_problem(
                rhs_physics_coeffs=rhs_physics_basis,
                m_imp_to_jr_operator=m_imp_to_jr_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                equality_operator=equality_operator,
                equality_rhs=equality_rhs,
            )
        ).reshape(n_coeff, n_coeff)
        self._dpsi_unconstrained_map_cache[key] = dpsi_map
        return dpsi_map

    def _get_joint_dtalpha_er_problem_bundle(
        self,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
    ) -> dict[str, Any]:
        """Build/fetch joint LS bundle for unknowns ``[dt_alpha, E_r_coeff]``.

        Physics residual is posed in the same weighted grid residual space as the
        standard ``dt_alpha`` solve, while the ideal ``E_r`` closure is added as
        explicit weighted rows in the same least-squares problem.
        """
        cache_key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
        )
        cached = self._dtalpha_er_joint_problem_cache.get(cache_key)
        if cached is not None:
            return cached

        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import as_linear_map, diagonal_linear_map

        dtjr_bundle = self._get_dtjr_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        R_grid = np.asarray(dtjr_bundle["R_grid"])
        n_alpha = int(dtjr_bundle["n_coeff"])
        A_grid = np.asarray(self._dtalpha_grid_residual_maps[1])
        alpha_to_jr = np.asarray(dtjr_bundle["alpha_to_jr"])

        er_solver = self._br_constraint_solver
        G_er = np.asarray(er_solver.G)
        n_er = int(G_er.shape[1])
        n_grid = int(G_er.shape[0])
        if R_grid.shape[0] != n_grid:
            raise ValueError(
                "Joint dtalpha/Er bundle grid-size mismatch: "
                f"R_grid={R_grid.shape}, G_er={G_er.shape}."
            )

        K_er_grid = np.asarray(to_numpy(self.Er_grid_to_dtjr_forcing_matrix))
        if K_er_grid.shape[1] != n_grid or K_er_grid.shape[0] != n_alpha:
            raise ValueError(
                "Er forcing map size mismatch in joint bundle: "
                f"K_er_grid={K_er_grid.shape}, n_alpha={n_alpha}, n_grid={n_grid}."
            )
        K_er_coeff = K_er_grid @ G_er

        # Physics residual rows (weighted by the usual toroidal weighting policy).
        A_phys_joint = np.hstack([A_grid, -(R_grid @ K_er_coeff)])
        op_phys = as_linear_map(A_phys_joint)
        physics_weight = self._build_physics_sqrt_weight(op_phys.shape[0], weighting)

        operators = [op_phys]
        data_shapes = [(op_phys.shape[0],)]
        sqrt_weights = [physics_weight]

        # Weighted ideal-closure rows for E_r coefficients:
        #   sqrt(w) * (diag(Br) G e - rhs) = 0
        # plus weak regularization floor:
        #   floor * sqrt(w) * (G e) = 0
        sqrt_w = np.sqrt(np.maximum(np.asarray(er_solver.weights).reshape(-1), 0.0))
        C_er_grid = np.asarray(er_solver.constraint_operator_grid)
        A_er_closure = np.hstack(
            [
                np.zeros((n_grid, n_alpha), dtype=float),
                sqrt_w[:, None] * C_er_grid,
            ]
        )
        operators.append(as_linear_map(A_er_closure))
        data_shapes.append((n_grid,))
        sqrt_weights.append(None)

        has_er_reg_row = float(er_solver.floor) > 0.0
        if has_er_reg_row:
            A_er_reg = np.hstack(
                [
                    np.zeros((n_grid, n_alpha), dtype=float),
                    (float(er_solver.floor) * sqrt_w)[:, None] * G_er,
                ]
            )
            operators.append(as_linear_map(A_er_reg))
            data_shapes.append((n_grid,))
            sqrt_weights.append(None)

        # Optional penalty on dt_alpha only.
        if penalty_operator is not None and penalty_scaling > 0:
            P_dtalpha = np.asarray(to_dense(as_linear_map(penalty_operator)))
            if P_dtalpha.ndim != 2:
                P_dtalpha = P_dtalpha.reshape(P_dtalpha.shape[0], -1)
            if P_dtalpha.shape[1] != n_alpha:
                raise ValueError(
                    "Penalty operator width mismatch in joint dtalpha/Er bundle: "
                    f"penalty={P_dtalpha.shape}, n_alpha={n_alpha}."
                )
            A_penalty_joint = np.hstack(
                [
                    float(penalty_scaling) * P_dtalpha,
                    np.zeros((P_dtalpha.shape[0], n_er), dtype=float),
                ]
            )
            operators.append(as_linear_map(A_penalty_joint))
            data_shapes.append((P_dtalpha.shape[0],))
            sqrt_weights.append(None)

        # Tikhonov on dt_alpha only, in absolute dt_alpha units.
        if regularization_lambda > 0:
            reg_scale = float(np.sqrt(max(regularization_lambda, 0.0)))
            A_reg_joint = np.hstack(
                [
                    reg_scale * np.eye(n_alpha, dtype=float),
                    np.zeros((n_alpha, n_er), dtype=float),
                ]
            )
            operators.append(as_linear_map(A_reg_joint))
            data_shapes.append((n_alpha,))
            sqrt_weights.append(None)

        problem = LeastSquaresProblem(
            A=operators,
            solution_shape=(n_alpha + n_er),
            data_shapes=data_shapes,
            sqrt_weights=sqrt_weights,
        )

        bundle = {
            "problem": problem,
            "R_grid": np.asarray(R_grid),
            "alpha_to_jr": np.asarray(alpha_to_jr),
            "n_alpha": int(n_alpha),
            "n_er": int(n_er),
            "sqrt_w_er": np.asarray(sqrt_w),
            "K_er_coeff": np.asarray(K_er_coeff),
            "has_er_reg_row": bool(has_er_reg_row),
        }
        self._dtalpha_er_joint_problem_cache[cache_key] = bundle
        return bundle

    def _solve_joint_dtalpha_er_problem(
        self,
        *,
        rhs_physics_no_er_coeffs: np.ndarray,
        er_rhs_grid: np.ndarray,
        weighting: str,
        regularization_lambda: float,
        jr_map_operator: Any = None,
        rhs_constraint: Optional[np.ndarray] = None,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve joint ``(dt_alpha, E_r_coeff)`` LS problem."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        bundle = self._get_joint_dtalpha_er_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        problem = bundle["problem"]
        R_grid = np.asarray(bundle["R_grid"])
        n_alpha = int(bundle["n_alpha"])
        n_er = int(bundle["n_er"])
        sqrt_w_er = np.asarray(bundle["sqrt_w_er"]).reshape(-1)
        has_er_reg_row = bool(bundle.get("has_er_reg_row", False))
        n_joint = n_alpha + n_er
        rcond = self._resolve_dtjr_rcond(n_joint, hinv_rtol)

        rhs_phys_grid = self._coeff_rhs_to_grid_rhs(R_grid, rhs_physics_no_er_coeffs)
        rhs_phys_grid = np.asarray(rhs_phys_grid).reshape(-1)
        rhs_er = np.asarray(to_numpy(er_rhs_grid)).reshape(-1)
        if rhs_er.size != sqrt_w_er.size:
            raise ValueError(
                "Joint dtalpha/Er RHS size mismatch: "
                f"er_rhs={rhs_er.shape}, sqrt_w={sqrt_w_er.shape}."
            )

        rhs_terms = [rhs_phys_grid, sqrt_w_er * rhs_er]

        # E_r regularization row RHS (if present) is zero.
        if has_er_reg_row:
            rhs_terms.append(np.zeros(rhs_er.size, dtype=float))

        # Remaining rows (penalty / dtalpha regularization) also have zero RHS.
        while len(rhs_terms) < int(problem.num_data_terms):
            n_rows = int(problem.A[len(rhs_terms)].num_rows)
            rhs_terms.append(np.zeros(n_rows, dtype=float))

        equality_operator = None
        equality_rhs = None
        if jr_map_operator is not None:
            C = np.asarray(to_dense(as_linear_map(jr_map_operator)))
            if C.ndim != 2:
                C = C.reshape(C.shape[0], -1)
            alpha_to_jr = np.asarray(bundle["alpha_to_jr"])
            C_alpha = C @ alpha_to_jr
            C_joint = np.hstack([C_alpha, np.zeros((C_alpha.shape[0], n_er), dtype=float)])
            equality_operator = C_joint
            rhs_c = np.zeros(C_joint.shape[0], dtype=float) if rhs_constraint is None else np.asarray(
                to_numpy(rhs_constraint)
            ).reshape(-1)
            if rhs_c.size != C_joint.shape[0]:
                raise ValueError(
                    "Joint dtalpha/Er constraint RHS mismatch: "
                    f"rhs={rhs_c.shape}, C={C_joint.shape}."
                )
            equality_rhs = rhs_c

        solver = LeastSquaresSolver(
            solver=self.dtjr_solver,
            tolerance=max(rcond, self.dtjr_tolerance),
            preconditioner=self.dtjr_preconditioner,
        )
        preconditioner = None
        if equality_operator is None and self.dtjr_preconditioner is not None:
            # Joint problem supports preconditioners, but we keep this conservative:
            # reuse the generic builder only when no equality elimination is used.
            preconditioner = solver.build_preconditioner(
                problem=problem,
                preconditioner_type=self.dtjr_preconditioner,
                num_scenarios=1,
                pinv_rcond=rcond,
            )

        sol = np.asarray(
            solver.solve(
                problem,
                rhs_terms,
                preconditioner=preconditioner,
                equality_operator=equality_operator,
                equality_rhs=equality_rhs,
                elimination_rcond=rcond,
            )
        ).reshape(-1)

        if sol.size != n_joint:
            raise RuntimeError(
                "Joint dtalpha/Er solve returned unexpected solution size: "
                f"{sol.shape} (expected {n_joint})."
            )
        return sol[:n_alpha], sol[n_alpha:]

    def _get_joint_dtalpha_er_schur_bundle(
        self,
        *,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
    ) -> dict[str, Any]:
        """Build/fetch Schur-ready dense blocks for joint ``(dt_alpha, E_r)`` solve."""
        cache_key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
        )
        cached = self._dtalpha_er_schur_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        dtjr_bundle = self._get_dtjr_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        R_grid = np.asarray(dtjr_bundle["R_grid"], dtype=float)
        alpha_to_jr = np.asarray(dtjr_bundle["alpha_to_jr"], dtype=float)
        n_alpha = int(dtjr_bundle["n_coeff"])
        A_phys = np.asarray(self._dtalpha_grid_residual_maps[1], dtype=float)

        er_solver = self._br_constraint_solver
        G_er = np.asarray(er_solver.G, dtype=float)
        C_er = np.asarray(er_solver.constraint_operator_grid, dtype=float)
        sqrt_w_er = np.sqrt(np.maximum(np.asarray(er_solver.weights, dtype=float).reshape(-1), 0.0))
        n_grid = int(G_er.shape[0])
        n_er = int(G_er.shape[1])
        if A_phys.shape[0] != n_grid or R_grid.shape[0] != n_grid:
            raise ValueError(
                "Joint dtalpha/Er Schur bundle grid-size mismatch: "
                f"A_phys={A_phys.shape}, R_grid={R_grid.shape}, G_er={G_er.shape}."
            )

        K_er_grid = np.asarray(to_numpy(self.Er_grid_to_dtjr_forcing_matrix), dtype=float)
        if K_er_grid.shape != (n_alpha, n_grid):
            raise ValueError(
                "Er forcing map size mismatch in Schur bundle: "
                f"K_er_grid={K_er_grid.shape}, expected ({n_alpha}, {n_grid})."
            )
        K_er_coeff = K_er_grid @ G_er
        B_phys = np.asarray(R_grid @ K_er_coeff, dtype=float)

        physics_sqrt_weight = self._build_physics_sqrt_weight(A_phys.shape[0], weighting)
        if physics_sqrt_weight is None:
            w_phys = np.ones(A_phys.shape[0], dtype=float)
        else:
            w_phys_arr = np.asarray(physics_sqrt_weight, dtype=float)
            if w_phys_arr.ndim != 1 or w_phys_arr.size != A_phys.shape[0]:
                raise NotImplementedError(
                    "Schur-condensed joint dtalpha/Er solve currently requires "
                    "grid-row diagonal physics weights."
                )
            w_phys = w_phys_arr * w_phys_arr

        P_dtalpha = None
        if penalty_operator is not None and penalty_scaling > 0.0:
            P_dtalpha = np.asarray(to_dense(as_linear_map(penalty_operator)), dtype=float)
            if P_dtalpha.ndim != 2:
                P_dtalpha = P_dtalpha.reshape(P_dtalpha.shape[0], -1)
            if P_dtalpha.shape[1] != n_alpha:
                raise ValueError(
                    "Penalty operator width mismatch in Schur bundle: "
                    f"penalty={P_dtalpha.shape}, n_alpha={n_alpha}."
                )
            P_dtalpha = float(penalty_scaling) * P_dtalpha

        reg_scale = float(np.sqrt(max(regularization_lambda, 0.0)))
        floor_er = float(er_solver.floor)
        has_er_reg = bool(floor_er > 0.0)

        Aw = A_phys.T * w_phys
        Bw = B_phys.T * w_phys
        Haa = Aw @ A_phys
        Hae = -(Aw @ B_phys)
        Hea = Hae.T
        Hee = Bw @ B_phys

        if P_dtalpha is not None:
            Haa = Haa + P_dtalpha.T @ P_dtalpha
        if reg_scale > 0.0:
            Haa = Haa + (reg_scale * reg_scale) * np.eye(n_alpha, dtype=float)

        Cw = sqrt_w_er[:, None] * C_er
        Hee = Hee + Cw.T @ Cw
        if has_er_reg:
            Rer = (floor_er * sqrt_w_er)[:, None] * G_er
            Hee = Hee + Rer.T @ Rer
        else:
            Rer = None

        bundle = {
            "n_alpha": n_alpha,
            "n_er": n_er,
            "R_grid": R_grid,
            "alpha_to_jr": alpha_to_jr,
            "sqrt_w_er": np.asarray(sqrt_w_er, dtype=float),
            "Haa": 0.5 * (np.asarray(Haa, dtype=float) + np.asarray(Haa, dtype=float).T),
            "Hae": np.asarray(Hae, dtype=float),
            "Hea": np.asarray(Hea, dtype=float),
            "Hee": 0.5 * (np.asarray(Hee, dtype=float) + np.asarray(Hee, dtype=float).T),
            "G_ba": np.asarray(Aw, dtype=float),
            "G_be": np.asarray(-Bw, dtype=float),
            "G_d": np.asarray(Cw.T, dtype=float),
            "A_phys": np.asarray(A_phys, dtype=float),
            "B_phys": np.asarray(B_phys, dtype=float),
            "Cw": np.asarray(Cw, dtype=float),
            "Rer": None if Rer is None else np.asarray(Rer, dtype=float),
            "has_er_reg": has_er_reg,
        }
        self._dtalpha_er_schur_bundle_cache[cache_key] = bundle
        return bundle

    def _solve_joint_dtalpha_er_problem_schur(
        self,
        *,
        rhs_physics_no_er_coeffs: np.ndarray,
        er_rhs_grid: np.ndarray,
        weighting: str,
        regularization_lambda: float,
        jr_map_operator: Any = None,
        rhs_constraint: Optional[np.ndarray] = None,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve joint ``(dt_alpha, E_r_coeff)`` via Schur condensation."""
        bundle = self._get_joint_dtalpha_er_schur_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        n_alpha = int(bundle["n_alpha"])
        n_er = int(bundle["n_er"])
        R_grid = np.asarray(bundle["R_grid"])
        sqrt_w_er = np.asarray(bundle["sqrt_w_er"]).reshape(-1)

        rhs_phys_grid = self._coeff_rhs_to_grid_rhs(R_grid, rhs_physics_no_er_coeffs)
        rhs_phys_grid = np.asarray(rhs_phys_grid, dtype=float).reshape(-1)
        rhs_er = np.asarray(to_numpy(er_rhs_grid), dtype=float).reshape(-1)
        if rhs_er.size != sqrt_w_er.size:
            raise ValueError(
                "Schur joint solve Er RHS size mismatch: "
                f"rhs_er={rhs_er.shape}, sqrt_w_er={sqrt_w_er.shape}."
            )
        d_weighted = sqrt_w_er * rhs_er

        rcond = self._resolve_dtjr_rcond(n_alpha + n_er, hinv_rtol)
        rcond_eff = max(float(rcond), float(self.dtjr_tolerance))
        Haa = np.asarray(bundle["Haa"])
        Hae = np.asarray(bundle["Hae"])
        Hea = np.asarray(bundle["Hea"])
        Hee = np.asarray(bundle["Hee"])
        G_ba = np.asarray(bundle["G_ba"])
        G_be = np.asarray(bundle["G_be"])
        G_d = np.asarray(bundle["G_d"])

        Hee_pinv = self._pinv_symmetric(Hee, rcond=rcond_eff)
        ga = G_ba @ rhs_phys_grid
        ge = G_be @ rhs_phys_grid + G_d @ d_weighted

        schur_left = Hae @ Hee_pinv
        Hred = Haa - schur_left @ Hea
        Hred = 0.5 * (Hred + Hred.T)
        gred = ga - schur_left @ ge

        if jr_map_operator is None:
            dtalpha = self._pinv_symmetric(Hred, rcond=rcond_eff) @ gred
        else:
            C = np.asarray(to_dense(as_linear_map(jr_map_operator)), dtype=float)
            if C.ndim != 2:
                C = C.reshape(C.shape[0], -1)
            alpha_to_jr = np.asarray(bundle["alpha_to_jr"])
            C_alpha = C @ alpha_to_jr
            rhs_c = (
                np.zeros(C_alpha.shape[0], dtype=float)
                if rhs_constraint is None
                else np.asarray(to_numpy(rhs_constraint), dtype=float).reshape(-1)
            )
            if rhs_c.size != C_alpha.shape[0]:
                raise ValueError(
                    "Schur joint solve constraint RHS mismatch: "
                    f"rhs={rhs_c.shape}, C_alpha={C_alpha.shape}."
                )
            if C_alpha.shape[0] == 0:
                dtalpha = self._pinv_symmetric(Hred, rcond=rcond_eff) @ gred
            else:
                KKT = np.block(
                    [
                        [Hred, C_alpha.T],
                        [C_alpha, np.zeros((C_alpha.shape[0], C_alpha.shape[0]), dtype=float)],
                    ]
                )
                rhs_kkt = np.concatenate([gred, rhs_c])
                # Use a symmetric pseudoinverse directly to select the same
                # minimum-norm KKT branch as the explicit constrained LS path
                # in near-singular cases.
                sol = self._pinv_symmetric(KKT, rcond=rcond_eff) @ rhs_kkt
                dtalpha = np.asarray(sol[:n_alpha], dtype=float)

        er_coeffs = Hee_pinv @ (ge - Hea @ np.asarray(dtalpha).reshape(-1))
        return np.asarray(dtalpha).reshape(-1), np.asarray(er_coeffs).reshape(-1)

    def _get_constrained_dpsi_maps(
        self,
        *,
        jr_map_operator: Any,
        m_imp_to_jr_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return cached constrained maps for one-shot hard-constrained dpsi solves.

        Returns dense matrices:
            ``M_phys``: physics RHS -> constrained homogeneous response
            ``M_corr``: constraint RHS -> constrained inhomogeneous response
            ``C``: primary hard-constraint rows on ``dt_jr``.
        """
        C = np.asarray(to_dense(as_linear_map(jr_map_operator)))
        if C.ndim != 2:
            C = C.reshape(C.shape[0], -1)
        n_coeff = int(C.shape[1])
        m_constraints = int(C.shape[0])
        rcond = self._resolve_dtjr_rcond(n_coeff, hinv_rtol)
        key = (
            id(jr_map_operator),
            id(m_imp_to_jr_operator),
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            bool(use_pinning),
            self._dtjr_solver_signature(),
        )
        cached = self._dpsi_constrained_maps_cache.get(key)
        if cached is not None:
            return cached

        bundle = self._get_dpsi_problem_bundle(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        m_to_jr = np.asarray(bundle["m_to_jr"])
        if m_to_jr.shape != (n_coeff, n_coeff):
            raise ValueError(
                "m_to_jr size mismatch in constrained dpsi maps: "
                f"map={m_to_jr.shape}, n={n_coeff}."
            )

        C_psi = C @ m_to_jr
        gauge_rows = self._get_cs_psi_gauge_rows(n_coeff, use_pinning)
        if gauge_rows.shape[0] > 0:
            C_eq = np.vstack([C_psi, gauge_rows])
        else:
            C_eq = C_psi
        m_total = int(C_eq.shape[0])

        M_phys = np.asarray(
            self._solve_dpsi_problem(
                rhs_physics_coeffs=np.eye(n_coeff, dtype=float),
                m_imp_to_jr_operator=m_imp_to_jr_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                equality_operator=C_eq,
                equality_rhs=np.zeros(m_total, dtype=float),
            )
        ).reshape(n_coeff, n_coeff)

        if m_constraints > 0:
            rhs_eq = np.zeros((m_total, m_constraints), dtype=float)
            rhs_eq[:m_constraints, :] = np.eye(m_constraints, dtype=float)
            M_corr = np.asarray(
                self._solve_dpsi_problem(
                    rhs_physics_coeffs=np.zeros((n_coeff, m_constraints), dtype=float),
                    m_imp_to_jr_operator=m_imp_to_jr_operator,
                    weighting=weighting,
                    regularization_lambda=regularization_lambda,
                    penalty_operator=penalty_operator,
                    penalty_scaling=penalty_scaling,
                    hinv_rtol=hinv_rtol,
                    equality_operator=C_eq,
                    equality_rhs=rhs_eq,
                )
            ).reshape(n_coeff, m_constraints)
        else:
            M_corr = np.zeros((n_coeff, 0), dtype=float)

        maps = {
            "C": C,
            "M_phys": np.asarray(M_phys),
            "M_corr": np.asarray(M_corr),
        }
        self._dpsi_constrained_maps_cache[key] = maps
        return maps

    def solve_dt_jr_physics(
        self,
        rhs_physics: np.ndarray,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Solve the unconstrained physics system for dt_jr."""
        M = self._get_unconstrained_dtjr_map_cached(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        rhs = np.asarray(to_numpy(rhs_physics)).reshape(-1)
        return asarray(M @ rhs)

    def _build_unconstrained_dtjr_map(
        self,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Build dense map ``rhs_physics -> dt_jr`` for unconstrained physics solve."""
        return asarray(
            self._get_unconstrained_dtjr_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        )

    def solve_dt_jr_superposed(
        self,
        rhs_physics: np.ndarray,
        rhs_constraint: np.ndarray,
        jr_map_operator: Any,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Solve dt_jr in one-shot constrained form.

        This is the primary runtime path:
            minimize ||A x - rhs_physics|| subject to C x = rhs_constraint
        """
        if jr_map_operator is None:
            # No hard constraints: return unconstrained physics solve.
            return self.solve_dt_jr_physics(
                rhs_physics=rhs_physics,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )

        maps = self._get_constrained_dtjr_maps(
            jr_map_operator=jr_map_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        rhs_p = np.asarray(to_numpy(rhs_physics)).reshape(-1)
        rhs_c = np.asarray(to_numpy(rhs_constraint)).reshape(-1)
        x = maps["M_phys"] @ rhs_p
        if maps["M_corr"].shape[1] > 0:
            x = x + maps["M_corr"] @ rhs_c
        return asarray(x.reshape(-1))

    def solve_dpsi_dt_superposed(
        self,
        rhs_physics: np.ndarray,
        rhs_constraint: np.ndarray,
        jr_map_operator: Any,
        m_imp_to_jr_operator: Any,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Solve for ``dpsi/dt`` via one-shot constrained ``dt_alpha`` solve.

        This is the monolithic toroidal path used at runtime:
        1) solve ``dt_alpha`` from physics + hard constraints,
        2) map ``dt_alpha -> dpsi`` via a single linear conversion map.
        """
        rhs_p = np.asarray(to_numpy(rhs_physics)).reshape(-1)
        dtalpha_to_dpsi = self._get_dtalpha_to_dpsi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )

        if jr_map_operator is None:
            M_alpha = self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha = M_alpha @ rhs_p
            return asarray((dtalpha_to_dpsi @ dtalpha).reshape(-1))

        maps = self._get_constrained_dtjr_maps(
            jr_map_operator=jr_map_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        rhs_c = np.asarray(to_numpy(rhs_constraint)).reshape(-1)
        dtalpha = maps["M_phys_alpha"] @ rhs_p
        if maps["M_corr_alpha"].shape[1] > 0:
            dtalpha = dtalpha + maps["M_corr_alpha"] @ rhs_c
        dpsi = dtalpha_to_dpsi @ dtalpha
        return asarray(dpsi.reshape(-1))

    def solve_dpsi_dt_superposed_joint_er(
        self,
        *,
        E_coeffs: np.ndarray,
        dt_jr_driver_coeffs: Optional[np.ndarray],
        rhs_constraint: np.ndarray,
        jr_map_operator: Any,
        m_imp_to_jr_operator: Any,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Solve ``dpsi/dt`` via joint ``(dt_alpha, E_r)`` least-squares.

        This keeps the standard ``dt_alpha`` physics operator ``L_alpha`` but
        avoids pre-eliminating the ideal-closure ``E_r`` in the forcing. Instead
        ``E_r`` is solved simultaneously with ``dt_alpha`` through explicit
        weighted closure rows.
        """
        rhs_no_er, er_rhs_grid = self.compute_forcing_vector_split(
            E_coeffs=E_coeffs,
            dt_jr_driver_coeffs=dt_jr_driver_coeffs,
        )
        dtalpha, _er_coeffs = self._solve_joint_dtalpha_er_problem_schur(
            rhs_physics_no_er_coeffs=rhs_no_er,
            er_rhs_grid=er_rhs_grid,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            jr_map_operator=jr_map_operator,
            rhs_constraint=rhs_constraint,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        dtalpha_to_dpsi = self._get_dtalpha_to_dpsi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )
        dpsi = dtalpha_to_dpsi @ np.asarray(dtalpha).reshape(-1)
        return asarray(dpsi.reshape(-1))

    # -------------------------------------------------------------------------
    # Time Evolution Logic
    # -------------------------------------------------------------------------

    def _get_jr_to_psi_dense(
        self,
        m_imp_to_jr_operator: Any,
        use_pinning: Optional[bool] = None,
    ) -> np.ndarray:
        """Return cached true MP inverse for ``jr -> psi`` (dense)."""
        cache_key = (id(m_imp_to_jr_operator), False)
        cached = self._jr_to_psi_cache.get(cache_key)
        if cached is not None:
            return cached

        op_m_to_jr = as_linear_map(m_imp_to_jr_operator)
        m_to_jr_dense = to_dense(op_m_to_jr)
        jr_to_psi = tensor_pinv(m_to_jr_dense, n_leading_flattened=1)

        self._jr_to_psi_cache[cache_key] = jr_to_psi
        return jr_to_psi

    def _get_psi_gauge_projector_dense(
        self,
        m_imp_to_jr_operator: Any,
        use_pinning: Optional[bool] = None,
    ) -> np.ndarray:
        """Return explicit gauge projector applied after MP inversion.

        The projector enforces a scalar gauge (mean-zero for CS) by removing
        null-space content:
            psi_gauged = P_gauge @ psi
        where ``P_gauge`` acts only along ``null(m_to_jr)`` so ``m_to_jr`` is
        unchanged by the correction.
        """
        if use_pinning is None:
            use_pinning = self.is_cs

        gauge_mode = "mean_zero"
        cache_key = (id(m_imp_to_jr_operator), bool(use_pinning), gauge_mode)
        cached = self._psi_gauge_projector_cache.get(cache_key)
        if cached is not None:
            return cached

        op_m_to_jr = as_linear_map(m_imp_to_jr_operator)
        m_to_jr_dense = to_dense(op_m_to_jr)
        n = int(m_to_jr_dense.shape[1])
        identity = np.eye(n, dtype=m_to_jr_dense.dtype)

        if not use_pinning:
            self._psi_gauge_projector_cache[cache_key] = identity
            return identity

        if self.is_cs and hasattr(self.basis, "get_scalar_gauge_projector_for_operator"):
            gauge_projector = np.asarray(
                self.basis.get_scalar_gauge_projector_for_operator(
                    m_to_jr_dense,
                    mode=gauge_mode,
                    rcond=self._default_pinv_rcond(m_to_jr_dense.shape),
                )
            )
            self._psi_gauge_projector_cache[cache_key] = gauge_projector
            return gauge_projector

        gauge_row = None
        if hasattr(self.basis, "get_scalar_gauge_constraint_matrix"):
            gauge_row = np.asarray(
                self.basis.get_scalar_gauge_constraint_matrix(
                    n_coeff=n,
                    mode=gauge_mode,
                )
            )
        if gauge_row is None:
            # Generic gauge: enforce weighted grid-mean potential = 0.
            if not hasattr(self.basis, "get_G"):
                raise RuntimeError(
                    "Scalar gauge projector requested, but basis does not provide "
                    "get_scalar_gauge_constraint_matrix() or get_G(grid)."
                )
            g_mat = np.asarray(to_dense(as_linear_map(self.basis.get_G(self.grid))))
            if g_mat.ndim != 2 or g_mat.shape[1] != n:
                raise RuntimeError(
                    "Failed to build generic scalar gauge row from basis.get_G(grid)."
                )
            if hasattr(self.grid, "weights") and self.grid.weights is not None:
                w = np.asarray(to_numpy(self.grid.weights)).reshape(-1)
                if w.size != g_mat.shape[0]:
                    raise RuntimeError(
                        "Grid weights size mismatch while building generic scalar gauge row."
                    )
                w = np.maximum(w, 0.0)
                w_sum = float(np.sum(w))
                if not np.isfinite(w_sum) or w_sum <= 0.0:
                    raise RuntimeError(
                        "Non-positive grid weights sum while building generic scalar gauge row."
                    )
                w = w / w_sum
            else:
                w = np.full(g_mat.shape[0], 1.0 / max(g_mat.shape[0], 1), dtype=float)
            gauge_row = (w @ g_mat).reshape(1, -1)
        if gauge_row.ndim == 1:
            gauge_row = gauge_row.reshape(1, -1)
        gauge_row = gauge_row.astype(m_to_jr_dense.dtype, copy=False)

        # Fast path for CS: constant potential is the expected Laplacian null mode.
        z_const = np.ones((n, 1), dtype=m_to_jr_dense.dtype)
        rel_const_null = np.linalg.norm(m_to_jr_dense @ z_const) / max(
            np.linalg.norm(m_to_jr_dense) * np.linalg.norm(z_const), 1e-30
        )
        if rel_const_null < 1e-6:
            null_basis = z_const
        else:
            _, s_vals, vh = np.linalg.svd(m_to_jr_dense, full_matrices=False)
            if s_vals.size == 0:
                null_basis = np.zeros((n, 0), dtype=m_to_jr_dense.dtype)
            else:
                svd_rtol = np.finfo(float).eps * max(m_to_jr_dense.shape)
                null_mask = s_vals <= svd_rtol * s_vals[0]
                null_basis = (
                    vh[null_mask].T
                    if np.any(null_mask)
                    else np.zeros((n, 0), dtype=m_to_jr_dense.dtype)
                )

        gauge_projector = identity
        if null_basis.shape[1] > 0:
            gauge_on_null = gauge_row @ null_basis
            if np.linalg.norm(gauge_on_null) > 0:
                gauge_on_null_pinv = np.linalg.pinv(gauge_on_null)
                gauge_projector = identity - (null_basis @ gauge_on_null_pinv @ gauge_row)

        self._psi_gauge_projector_cache[cache_key] = gauge_projector
        return gauge_projector

    def _get_dtalpha_to_dpsi_map_cached(
        self,
        *,
        m_imp_to_jr_operator: Any,
        use_pinning: bool,
    ) -> np.ndarray:
        """Return cached dense map ``dt_alpha -> dpsi/dt``.

        The map is assembled as:
            ``M = P_gauge @ (jr_to_psi @ alpha_to_jr)``
        so the toroidal evolution solve can remain in ``dt_alpha`` space and
        only convert to ``dpsi`` once at the end.
        """
        key = (
            id(m_imp_to_jr_operator),
            bool(use_pinning),
        )
        cached = self._dtalpha_to_dpsi_map_cache.get(key)
        if cached is not None:
            return cached

        alpha_to_jr = np.asarray(to_numpy(self.alpha_to_jr_coeff_operator))
        jr_to_psi = np.asarray(
            self._get_jr_to_psi_dense(m_imp_to_jr_operator, use_pinning=False)
        )
        gauge_projector = np.asarray(
            self._get_psi_gauge_projector_dense(
                m_imp_to_jr_operator,
                use_pinning=use_pinning,
            )
        )
        map_dtalpha_to_dpsi = gauge_projector @ (jr_to_psi @ alpha_to_jr)
        self._dtalpha_to_dpsi_map_cache[key] = map_dtalpha_to_dpsi
        return map_dtalpha_to_dpsi

    def compute_rates(
        self,
        dt_jr: np.ndarray,
        m_imp_to_jr_operator: Any,
        use_pinning: Optional[bool] = None,
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
             Typically ``state.poloidal_matrices.m_imp_to_jr``.
        use_pinning : bool, optional
             Whether to apply scalar gauge fixing. Defaults to True for CS basis.
             
        Returns
        -------
        np.ndarray
             d(psi)/dt coefficients.
        """
        # Invert operator with true MP inverse, then apply explicit gauge fix.
        jr_to_m_dense = self._get_jr_to_psi_dense(m_imp_to_jr_operator, use_pinning=False)
        gauge_projector = self._get_psi_gauge_projector_dense(
            m_imp_to_jr_operator, use_pinning=use_pinning
        )
        d_psi_dt = gauge_projector @ (jr_to_m_dense @ asarray(dt_jr))
        return asarray(d_psi_dt)

    def build_psi_dynamics_matrix(
        self,
        psi_to_E_operator: np.ndarray,
        m_imp_to_jr_operator: Any,
        jr_map_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Build the linear operator: psi → d(psi)/dt.

        This constructs the full chain:
            psi → E_psi → K → dt_jr → d(psi)/dt

        Each step is linear, so the composition is also linear:
            d(psi)/dt = L_psi_psi @ psi + (other contributions)

        Parameters
        ----------
        psi_to_E_operator : np.ndarray
            Operator mapping psi to E_coeffs contribution.
            Shape (2*N, N) or (2, N, N).
        m_imp_to_jr_operator : Any
            Operator mapping potential psi to jr.
        jr_map_operator : Any, optional
            Operator mapping jr to hard linear constraints.
        regularization_lambda : float
            Tikhonov regularization weight for the system inversion.
        penalty_operator : Any, optional
            Additional soft penalty operator applied to dt_jr.
        penalty_scaling : float, optional
            Scaling for the additional soft penalty operator.
        use_pinning : bool
            Whether to apply scalar gauge fixing (for CSBasis compatibility).

        Returns
        -------
        np.ndarray
            Matrix L_psi_psi of shape (N, N).
        """
        from pynamit.simulation.geometry_utils import to_dense

        N = self.basis.index_length

        # Step 1: map E coefficients to dt_jr forcing.
        E_to_dtjr_forcing = to_numpy(self.E_to_dtjr_forcing_matrix)  # (N, 2*N)

        # Step 2: psi → E_psi (reshape if needed)
        # NOTE: psi_to_E is post-resistivity (physical E). No additional scaling here.
        if hasattr(psi_to_E_operator, "to_dense"):
             psi_to_E = psi_to_E_operator.to_dense()
        else:
             psi_to_E = to_numpy(psi_to_E_operator)
        if psi_to_E.ndim == 3:  # shape (2, N, N)
            psi_to_E = psi_to_E.reshape(2 * N, N)
        
        # Step 3: forcing_dtjr -> dt_alpha via one-shot constrained solve maps.
        if jr_map_operator is not None:
            maps = self._get_constrained_dtjr_maps(
                jr_map_operator=jr_map_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha_from_K = maps["M_phys_alpha"]
        else:
            dtalpha_from_K = self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )

        # Step 4: dt_alpha -> dpsi conversion (gauge-aware).
        dtalpha_to_dpsi = self._get_dtalpha_to_dpsi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )
        dpsi_from_K = dtalpha_to_dpsi @ dtalpha_from_K

        # Final chain: psi -> E -> forcing_dtjr -> dt_alpha -> dpsi/dt.
        L_psi_psi = (dpsi_from_K @ E_to_dtjr_forcing) @ psi_to_E

        return asarray(L_psi_psi)

    def get_psi_dynamics_operator(
        self,
        psi_to_E_operator: Any,
        m_imp_to_jr_operator: Any,
        jr_map_operator: Any = None,
        solver: str = "lsmr",
        solver_tol: float = 1e-10,
        dense: bool = False,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> "LinearMap":
        """Get linear operator ``psi -> dpsi/dt``.

        Matrix-free and dense paths share one direct ``dpsi`` solve map, so both
        use identical hard-constraint/gauge semantics.
        """
        from pynamit.math.linear_map import LinearMap, as_linear_map

        # API compatibility: solver options are accepted but not used in this path.
        _ = (solver, solver_tol)
        N = self.basis.index_length

        if dense or jr_map_operator is not None:
            L_dense = self.build_psi_dynamics_matrix(
                psi_to_E_operator,
                m_imp_to_jr_operator,
                jr_map_operator=jr_map_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                use_pinning=use_pinning,
            )
            return as_linear_map(L_dense)

        if isinstance(psi_to_E_operator, LinearMap):
            psi_to_E_op = psi_to_E_operator
        else:
            psi_to_E_arr = to_numpy(psi_to_E_operator)
            if psi_to_E_arr.ndim == 3:
                psi_to_E_arr = psi_to_E_arr.reshape(2 * N, N)
            psi_to_E_op = as_linear_map(psi_to_E_arr)

        E_to_dtjr_forcing_op = as_linear_map(np.asarray(to_numpy(self.E_to_dtjr_forcing_matrix)))
        dtalpha_from_K_op = as_linear_map(
            self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        )
        dtalpha_to_dpsi_op = as_linear_map(
            self._get_dtalpha_to_dpsi_map_cached(
                m_imp_to_jr_operator=m_imp_to_jr_operator,
                use_pinning=use_pinning,
            )
        )

        def matvec(x):
            y = psi_to_E_op.matvec(asarray(x).reshape(-1))
            y = E_to_dtjr_forcing_op.matvec(y)
            y = dtalpha_from_K_op.matvec(y)
            y = dtalpha_to_dpsi_op.matvec(y)
            return asarray(y)

        def rmatvec(x):
            y = dtalpha_to_dpsi_op.rmatvec(asarray(x).reshape(-1))
            y = dtalpha_from_K_op.rmatvec(y)
            y = E_to_dtjr_forcing_op.rmatvec(y)
            y = psi_to_E_op.rmatvec(y)
            return asarray(y)

        def to_dense_func():
            return self.build_psi_dynamics_matrix(
                psi_to_E_operator,
                m_imp_to_jr_operator,
                jr_map_operator=jr_map_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                use_pinning=use_pinning,
            )

        return LinearMap(
            shape=(N, N),
            dtype=np.float64,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _to_dense=to_dense_func,
            source=None,
        )
