"""Toroidal system assembly for the dt_alpha evolution equation.

The current live assembled operator has the form:
    L_dtalpha * dt_alpha = rhs_toroidal

The field-line feedback part is assembled in the dt_alpha-native psi rewrite:
    mass_dtalpha * dt_alpha
    + A_raw * ((1/R) * dt_psi + d_r(dt_psi))
    = rhs_toroidal
where ``dt_psi`` is understood as the toroidal magnetic-potential response
induced by ``dt_alpha`` through the static ``jr <-> psi`` relation.

Important convention note:
    ``A_raw`` is assembled in unit-sphere angular-derivative form
    ``B0s · grad_Omega(.)``, not directly as ``B0s · grad_S(.)``. Since
    ``grad_Omega = R * grad_S``, the feedback block naturally appears as
    ``A_raw @ ((1/R) * dt_psi + d_r(dt_psi))``. This is equivalent to the
    surface-gradient form ``(B0s · grad_S)(dt_psi + R * d_r(dt_psi))``.

    The toroidal magnetic scalar ``psi`` uses the same sign convention as the
    imposed toroidal scalar ``m_imp`` and the basis toroidal vector convention
    ``Curl(T r) = -r x Grad T``. Equivalently,
        ``jr = -(R / mu0) * Delta_S(potential)``
    or
        ``jr = -(1 / (mu0 * R)) * Delta_Omega(potential)``
    depending on whether one writes the surface Laplacian on the radius-``R``
    sphere or the unit-sphere Laplacian. The basis operators implement the same
    negative-semidefinite scalar Laplacian in their native coordinates. For SH
    modes this means
        ``potential_lm = +(mu0 * R) / (l(l+1)) * jr_lm`` for ``l >= 1``.

Design note for CS-dominant full induction:
    - The toroidal closure operator is assembled in one auxiliary SH basis
      on the same grid.
    - CS remains the state/grid representation basis.

This keeps eliminated radial-structure identities and coupled toroidal
blocks in one consistent closure basis while preserving CS-centric
state representation.

Important closure note:
    The canonical full-induction shell closure is now ``radial_shell`` with
    an injected shell-gap response model. Older tangential closures remain
    only as internal benchmark/shadow implementations for diagnostics and
    equivalent-response construction. A distinct explicit bulk
    radial-shell BVP based directly on ``Q_I - Delta_Omega E_r,I`` is not
    assembled in this module yet; instead the runtime uses condensed response
    operators supplied through ``RadialShellResponseModel``.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import numpy as np
import scipy.sparse
from functools import cached_property

from pynamit.primitives.basis import is_cs_basis, is_sh_basis
from pynamit.utils import to_numpy, asarray, tensor_pinv
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.math.constants import mu0
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.simulation.induction.operator_utils import coerce_dense_operator_matrix
from pynamit.simulation.induction.radial_shell_response import (
    RadialShellResponseModel,
)
from pynamit.simulation.induction.toroidal_closure import ToroidalClosureProjector

logger = logging.getLogger(__name__)


class ToroidalSystemMatrices:
    """Assembles system matrices for Toroidal Induction.

    Handles the construction of:
    - alpha-space mass matrix (``mass_dtalpha``)
    - raw field-line advection operator (``fieldline_advection_operator_raw``)
    - alpha-space to toroidal-potential map (``alpha_to_psi_coeff_operator``)
    - alpha-space radial closure (``radial_closure_dtalpha``)

    Parameters
    ----------
    basis : Any
        The spectral basis (SHBasis).
    grid : Any
        The integration grid (from Geometry).
    b_field : Any
        The background magnetic field evaluated on the grid.
        Components used throughout this class are from this background
        field: ``B0r = b_field.vec.r`` and
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
        RM: Optional[float] = None,
        closure_mode: str = "radial_shell",
        radial_shell_response_model: Optional[RadialShellResponseModel] = None,
        closure_derivative_basis: Optional[Any] = None,
        rhs_derivative_basis: Optional[Any] = None,
        radial_derivative_basis: Optional[Any] = None,
        toroidal_solver: str = "normal_eq",
        toroidal_preconditioner: Optional[str] = None,
        toroidal_tolerance: float = 1e-13,
    ):
        self.basis = basis
        self.grid = grid
        self.b_field = b_field
        self.RI = RI
        self.RM = RM
        self.closure_mode = str(closure_mode)
        self.radial_shell_response_model = radial_shell_response_model
        # Derivative operators:
        # - closure_derivative_basis/rhs_derivative_basis: primary derivative
        #   basis for toroidal closure assembly.
        # - radial_derivative_basis: optional override for Er/radial-closure
        #   derivative chains.
        #
        # In cs_dominant full-induction we can set all three to one auxiliary SH
        # basis to keep the full toroidal closure assembly basis-consistent.
        base_closure_basis = (
            basis if closure_derivative_basis is None else closure_derivative_basis
        )
        self.closure_derivative_basis = base_closure_basis
        self.rhs_derivative_basis = (
            base_closure_basis if rhs_derivative_basis is None else rhs_derivative_basis
        )
        self.radial_derivative_basis = (
            self.rhs_derivative_basis
            if radial_derivative_basis is None
            else radial_derivative_basis
        )
        self._cs_derivative_operator_cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        self.is_cs = is_cs_basis(self.basis)
        if not self.is_cs:
            self.gaunt_engine = GauntEngine(basis)
        # Cache for explicit gauge projector applied after MP inversion.
        self._psi_gauge_projector_cache: dict[tuple[int, bool, str], np.ndarray] = {}
        # Cache for weighted toroidal least-squares problems.
        self._toroidal_problem_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._dtalpha_unconstrained_map_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._dtalpha_constrained_maps_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._dtalpha_to_dt_psi_map_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self.configure_toroidal_solver(
            solver=toroidal_solver,
            preconditioner=toroidal_preconditioner,
            tolerance=toroidal_tolerance,
        )

    def _build_auxiliary_toroidal_matrices(self, closure_basis: Any) -> "ToroidalSystemMatrices":
        """Build auxiliary toroidal assembler in closure basis."""
        return ToroidalSystemMatrices(
            basis=closure_basis,
            grid=self.grid,
            b_field=self.b_field,
            RI=self.RI,
            RM=self.RM,
            closure_mode=self.closure_mode,
            radial_shell_response_model=self.radial_shell_response_model,
            closure_derivative_basis=closure_basis,
            rhs_derivative_basis=closure_basis,
            radial_derivative_basis=closure_basis,
            toroidal_solver=self.toroidal_solver,
            toroidal_preconditioner=self.toroidal_preconditioner,
            toroidal_tolerance=self.toroidal_tolerance,
        )

    @staticmethod
    def full_radial_shell_response_requirements() -> dict[str, Any]:
        """Return the missing ingredients for a true radial-shell closure.

        The current runtime toroidal forcing state is stored only as tangential
        shell electric coefficients ``E_coeffs`` with shape ``(2, N)``.
        A first-principles radial-shell closure cannot be reconstructed from
        those coefficients alone: it also needs either the shell traces

            ``E_r|_{R_I^+}`` and ``Q_I = d_r(r div_Omega(E_S))|_{R_I^+}``

        or an equivalent response operator that maps the live forcing state to

            ``-(Q_I - Delta_Omega E_r,I) / R_I^2``.
        """
        return {
            "available": False,
            "known_runtime_forcing": "tangential_shell_E_coeffs_only",
            "known_runtime_forcing_shape": "(2, N)",
            "required_shell_traces": [
                "E_r at r = R_I^+",
                "Q_I = d_r(r div_Omega(E_S)) at r = R_I^+",
            ],
            "required_equivalent_operator": (
                "A response operator for -(Q_I - Delta_Omega E_r,I) / R_I^2 from the "
                "live toroidal forcing state."
            ),
            "reason": (
                "A true radial-shell closure is not uniquely determined by the current "
                "runtime data, because the runtime carries only tangential shell electric "
                "coefficients."
            ),
        }

    def _require_radial_shell_response_model(self) -> RadialShellResponseModel:
        """Return the configured full radial-shell response model or raise."""
        if self.radial_shell_response_model is None:
            requirements = self.full_radial_shell_response_requirements()
            raise NotImplementedError(
                "toroidal_closure_mode='radial_shell' requires a concrete "
                "RadialShellResponseModel. "
                f"Requirements: {requirements!r}"
            )
        return self.radial_shell_response_model

    @cached_property
    def _toroidal_closure_projector(self) -> ToroidalClosureProjector:
        """Closure-basis projector for toroidal assembly components."""
        return ToroidalClosureProjector(
            state_basis=self.basis,
            closure_basis=self.closure_derivative_basis,
            grid=self.grid,
            build_auxiliary_assembler=self._build_auxiliary_toroidal_matrices,
        )

    @cached_property
    def solver(self) -> "ToroidalSolver":
        """Helper exposing solve/orchestration routines built on toroidal operators."""
        from pynamit.simulation.induction.toroidal_solver import ToroidalSolver

        return ToroidalSolver(self)

    def configure_toroidal_solver(
        self, *, solver: str, preconditioner: Optional[str], tolerance: float
    ) -> None:
        """Configure solver policy for toroidal least-squares solves."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        if solver not in LeastSquaresSolver.VALID_SOLVERS:
            raise ValueError(
                f"Invalid toroidal solver '{solver}'. Valid options: "
                f"{LeastSquaresSolver.VALID_SOLVERS}."
            )
        if (
            preconditioner is not None
            and preconditioner not in LeastSquaresSolver.VALID_PRECONDITIONERS
        ):
            raise ValueError(
                f"Invalid toroidal preconditioner '{preconditioner}'. Valid options: "
                f"{LeastSquaresSolver.VALID_PRECONDITIONERS}."
            )
        self.toroidal_solver = str(solver)
        self.toroidal_preconditioner = preconditioner
        self.toroidal_tolerance = float(max(tolerance, 1e-15))
        # Cached linear maps depend on the solve policy.
        self._dtalpha_unconstrained_map_cache.clear()
        self._dtalpha_constrained_maps_cache.clear()
        self._dtalpha_to_dt_psi_map_cache.clear()

    def _toroidal_solver_signature(self) -> tuple[Any, ...]:
        """Return cache signature for the configured toroidal solver policy."""
        return (self.toroidal_solver, self.toroidal_preconditioner, float(self.toroidal_tolerance))

    def _build_cs_grid_derivative_operators(
        self, deriv_basis: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build grid-space operators (D_theta, D_phi_scaled, Laplacian) for CS."""
        if deriv_basis is self.basis:
            D_th = self.basis.get_evaluation_matrix(self.grid, derivative="theta")
            D_ph = self.basis.get_evaluation_matrix(self.grid, derivative="phi")
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
        G = np.asarray(to_dense(deriv_basis.get_evaluation_matrix(self.grid)))
        G_th = np.asarray(
            to_dense(deriv_basis.get_evaluation_matrix(self.grid, derivative="theta"))
        )
        G_ph = np.asarray(to_dense(deriv_basis.get_evaluation_matrix(self.grid, derivative="phi")))
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
        """Return CS derivative operators used in toroidal operator assembly."""
        if not self.is_cs:
            raise RuntimeError("cs_grid_derivative_operators is only valid for CS basis.")
        return self._get_cs_grid_derivative_operators(self.closure_derivative_basis)

    @property
    def cs_rhs_derivative_operators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CS derivative operators used in toroidal RHS assembly."""
        if not self.is_cs:
            raise RuntimeError("cs_rhs_derivative_operators is only valid for CS basis.")
        return self._get_cs_grid_derivative_operators(self.rhs_derivative_basis)

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
        """Return gauge-fixed inverse Laplacian for CS mapping assembly.

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
            - ``linear``: ``sqrt(|B0r|)``
            - ``quadratic``: ``|B0r|``
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

        if n_rows == 2 * w_base.size:
            # Stacked tangential full rows: apply the same grid-native weight to
            # both tangential scalar components.
            row_weight = np.sqrt(q_weights) * w_base
            return np.concatenate([row_weight, row_weight])

        # Spectral residual rows: build effective metric in coefficient space,
        # then provide its matrix square-root to LeastSquaresProblem.
        G_scalar = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)))
        w_eff = q_weights * (w_base**2)
        W_coeff = (G_scalar.T * w_eff) @ G_scalar
        W_coeff = 0.5 * (W_coeff + W_coeff.T)
        return self._matrix_sqrt_psd(W_coeff)

    @cached_property
    def inverse_radial_field_floor(self) -> float:
        """Automatic pseudo-inverse floor for ``1/B0r`` stabilization.

        We regularize near magnetic equator where ``B0r -> 0`` by selecting a
        robust lower-tail scale from ``|B0r|``. This avoids extreme amplification
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
    def _default_pinv_rcond(
        shape: tuple[int, ...] | list[int] | np.ndarray | None = None,
    ) -> float:
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
        self, CQ: np.ndarray, *, pinv_rcond: float
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
        """Compute stabilized pseudo-inverse ``1/B0r`` on the grid.

        Uses ``B0r / (B0r^2 + floor^2)`` with an automatic floor derived from the
        lower tail of ``|B0r|`` to reduce near-equatorial singular amplification.
        """
        B0r = np.asarray(to_numpy(self.b_field.vec.r))
        floor = float(self.inverse_radial_field_floor)
        if floor <= 0.0:
            return asarray(1.0 / B0r)
        inv = B0r / (B0r * B0r + floor * floor)
        return asarray(inv)

    @cached_property
    def mass_dtalpha(self) -> np.ndarray:
        """Construct tangentially projected alpha-space inertia matrix.

        For unknown ``dt_alpha`` (with ``dt_jr = B0r * dt_alpha``), the inertia
        block is:
            ``C_alpha = mu0 * <Y, |B0s|^2 Y>``.
        This belongs to the current tangentially projected toroidal closure and
        should not be read as a generic first-principles radial-shell mass term.
        """
        logger.info("Building Toroidal Alpha-Space Inertia Matrix...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(self._project_aux_square_operator_to_state(aux.mass_dtalpha))

        B2 = self.b_field.magnitude**2
        Br2 = self.b_field.vec.r**2
        factor = np.asarray(to_numpy(B2 - Br2)).reshape(-1)

        if self.is_cs:
            return asarray(np.diag(mu0 * factor))

        G = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)))
        P = np.asarray(to_dense(self.projection_matrix))
        if factor.size != G.shape[0]:
            raise ValueError(
                f"Alpha inertia factor/grid size mismatch: factor={factor.shape}, G={G.shape}."
            )
        M_factor = P @ (factor[:, None] * G)
        M_factor = 0.5 * (M_factor + M_factor.T)
        return mu0 * asarray(M_factor)

    @cached_property
    def radial_closure_dtalpha(self) -> np.ndarray:
        """Construct radial-closure map from ``dt_alpha`` to ``d_r(dt_jr)``.

        Using ``dt_jr = B0r * dt_alpha`` and
            ``d_r(dt_jr) = (d_r B0r) * dt_alpha - (1/R) * (B_s · grad(dt_alpha))``.
        """
        logger.info("Building dt_alpha radial-closure operator...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(self._project_aux_square_operator_to_state(aux.radial_closure_dtalpha))

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
                (D_th_r @ btheta_grid) + cot_th * btheta_grid + (D_ph_r @ bphi_grid)
            )
            radial_field_radial_derivative = -inv_Rb * (
                2.0 * br_grid + horizontal_field_divergence
            )

            horizontal_B_dot_grad_operator = (np.diag(btheta_grid) @ D_th_h) + (
                np.diag(bphi_grid) @ D_ph_h
            )

            dtalpha_closure_grid_operator = (
                np.diag(radial_field_radial_derivative) - inv_Rb * horizontal_B_dot_grad_operator
            )
            return asarray(dtalpha_closure_grid_operator)

        scalar_projection_matrix = np.asarray(to_dense(self.projection_matrix))
        scalar_evaluation_matrix = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid))
        )
        gradient_theta_operator = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta"))
        )
        gradient_phi_operator = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi"))
        )

        btheta_coeffs = scalar_projection_matrix @ btheta_grid
        bphi_coeffs = scalar_projection_matrix @ bphi_grid

        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        cot_th = 1.0 / np.tan(theta_rad)
        horizontal_field_divergence = (
            (gradient_theta_operator @ btheta_coeffs)
            + cot_th * btheta_grid
            + (gradient_phi_operator @ bphi_coeffs)
        )

        radial_field_radial_derivative = -inv_Rb * (2.0 * br_grid + horizontal_field_divergence)
        br_divergence_scale_term = (radial_field_radial_derivative)[
            :, None
        ] * scalar_evaluation_matrix
        horizontal_advection_term = -inv_Rb * (
            btheta_grid[:, None] * gradient_theta_operator
            + bphi_grid[:, None] * gradient_phi_operator
        )
        dtalpha_closure_grid_operator = br_divergence_scale_term + horizontal_advection_term
        return asarray(scalar_projection_matrix @ dtalpha_closure_grid_operator)

    @cached_property
    def toroidal_rhs_from_E_operator(self) -> np.ndarray:
        """Build matrix mapping E coefficients to toroidal RHS coefficients.

        Since ``compute_toroidal_rhs_from_E`` is linear in ``E_coeffs``, we can
        represent it as:
            rhs = E_to_rhs @ E_coeffs.flatten()

        Returns
        -------
        np.ndarray
            Matrix of shape ``(N, 2*N)`` mapping flattened ``E_coeffs`` to RHS coefficients.

        Notes
        -----
        This is the current Er-free tangential RHS operator used by the live
        projected-shell toroidal closure. It is not an explicit implementation
        of the radial-shell response operator ``Q_I - Delta_Omega E_r,I``.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            if self.closure_mode == "radial_shell":
                F_aux = np.asarray(aux.full_radial_shell_rhs_from_E_operator)
            elif self.closure_mode == "tangential_full":
                F_aux = np.asarray(aux.tangential_full_rhs_from_E_operator)
            else:
                F_aux = np.asarray(aux.toroidal_rhs_from_E_operator)
            return asarray(
                self._toroidal_closure_projector.project_vector_rhs_operator_to_state(F_aux)
            )

        if self.closure_mode == "radial_shell":
            return asarray(self.full_radial_shell_rhs_from_E_operator)

        if self.closure_mode == "tangential_full":
            return asarray(self.tangential_full_rhs_from_E_operator)

        if self.is_cs:
            return asarray(self._build_cs_toroidal_rhs_from_E_operator())

        if (not self.is_cs) and is_sh_basis(self.basis):
            return asarray(self._build_sh_toroidal_rhs_from_E_operator())

        N = self.basis.index_length
        # E_coeffs has shape (2, N) - [poloidal, toroidal] potentials
        # Build matrix by applying compute_toroidal_rhs_from_E to each basis vector.
        E_to_rhs = np.zeros((N, 2 * N))

        for i in range(2 * N):
            # Create basis vector
            e_i = np.zeros(2 * N)
            e_i[i] = 1.0
            E_i = e_i.reshape(2, N)

            # Apply the linear map
            rhs_i = self.compute_toroidal_rhs_from_E(E_i)
            E_to_rhs[:, i] = to_numpy(rhs_i)

        return asarray(E_to_rhs)

    @cached_property
    def _grid_metric_terms(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(theta_rad, cot(theta), n_grid)`` on the toroidal grid."""
        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        return theta_rad, cot_th, int(theta_rad.size)

    @cached_property
    def _background_field_grid_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return background field components ``(B0r, B0th, B0ph, n_grid)`` on the grid."""
        B0r = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        _, _, n_grid = self._grid_metric_terms
        if B0r.size != n_grid or B0th.size != n_grid or B0ph.size != n_grid:
            raise ValueError(
                "Grid field sizes are inconsistent in toroidal assembly: "
                f"B0r={B0r.size}, B0th={B0th.size}, B0ph={B0ph.size}, n_grid={n_grid}."
            )
        return B0r, B0th, B0ph, n_grid

    @cached_property
    def _vector_basis_component_maps(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return dense maps from flattened E coefficients to tangential components."""
        N = self.basis.index_length
        G_vec = np.asarray(to_dense(self.basis.get_vector_basis_matrix(self.grid)))
        if G_vec.ndim != 4 or G_vec.shape[0] != 2 or G_vec.shape[2] != 2 or G_vec.shape[3] != N:
            raise ValueError(
                "Unexpected vector basis shape for toroidal RHS map: "
                f"{G_vec.shape}, expected (2, N_grid, 2, {N})."
            )
        n_grid = int(G_vec.shape[1])
        V_th = np.hstack([G_vec[0, :, 0, :], G_vec[0, :, 1, :]])
        V_ph = np.hstack([G_vec[1, :, 0, :], G_vec[1, :, 1, :]])
        return V_th, V_ph, n_grid

    @cached_property
    def _rhs_scalar_derivative_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dense scalar derivative operators acting on grid-sampled scalars."""
        if self.is_cs:
            D_th, D_ph, _ = self.cs_rhs_derivative_operators
            return np.asarray(D_th, dtype=float), np.asarray(D_ph, dtype=float)

        P = np.asarray(to_dense(self.projection_matrix), dtype=float)
        G_th = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta"))
        )
        G_ph = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi")))
        return np.asarray(G_th @ P, dtype=float), np.asarray(G_ph @ P, dtype=float)

    def _apply_direct_toroidal_rhs_operator(self, E_coeffs: np.ndarray) -> np.ndarray:
        """Apply the cached direct RHS operator for SH/CS backends."""
        E_flat = np.asarray(E_coeffs).reshape(-1)
        return np.asarray(to_numpy(self.toroidal_rhs_from_E_operator @ E_flat)).reshape(-1)

    def _compute_generic_toroidal_rhs_from_E(self, E_coeffs: np.ndarray) -> np.ndarray:
        """Fallback RHS evaluation for non-SH/non-CS bases."""
        inv_Rb2 = 1.0 / (float(self.RI) ** 2)
        G_th = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta"))
        )
        G_ph = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi")))
        P = np.asarray(to_dense(self.projection_matrix))

        Eth_grid, Eph_grid = self.basis.evaluate(E_coeffs, self.grid, vector_type="tangential")
        B0th = np.asarray(to_numpy(self.b_field.vec.theta))
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi))

        def get_derivs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            coeffs = P @ values
            return G_th @ coeffs, G_ph @ coeffs

        _, dEth_ph = get_derivs(Eth_grid)
        dEph_th, _ = get_derivs(Eph_grid)
        _, cot_th, _ = self._grid_metric_terms
        curlE = dEph_th + cot_th * Eph_grid - dEth_ph
        dcurl_th, dcurl_ph = get_derivs(curlE)
        S_known = inv_Rb2 * (B0th * dcurl_ph - B0ph * dcurl_th)
        return np.asarray(to_numpy(P @ S_known)).reshape(-1)

    def _assemble_toroidal_rhs_from_tangential_maps(
        self, *, V_th: np.ndarray, V_ph: np.ndarray, D_th: np.ndarray, D_ph: np.ndarray
    ) -> np.ndarray:
        """Assemble the current inductive tangential toroidal RHS map.

        This is the forcing block used by the live tangential toroidal closure.
        Define the surface-curl scalar

            ``c = curl_Omega(E_S)``
                ``= d_theta(E_phi) + cot(theta) E_phi - d_phi^*(E_theta)``.

        The current forcing object is then

            ``f_ind(E_S)``
                ``= (1/R_I^2) * [ B0theta * d_phi^*(c) - B0phi * d_theta(c) ]``.

        Equivalently, if ``omega_E = hat(r)·curl(E) = c / R_I``, then this is

            ``f_ind(E_S) = B0S · curl(omega_E * hat(r))``.

        So the live toroidal forcing depends only on the inductive
        ``curl_Omega(E_S)`` content of the shell tangential electric field. In
        particular, pure curl-free shell electric fields do not drive this
        forcing block. This is why the operator is a plausible forcing-side
        object for the tangential projected closure, and why it should not be
        conflated with shell-current continuity formulas built from the
        instantaneous current ``K_S = Sigma E_S``.

        Relative to the exact projected Maxwell driver

            ``B0S · [ -curl(curl(E)) ]_S``,

        this block keeps only the contribution from the radial curl scalar
        ``hat(r)·curl(E)``. The complementary contribution from the tangential
        part of ``curl(E)`` is omitted in this reduced forcing operator.

        It should still be treated as a tangential closure block, not as a
        direct implementation of the radial-shell response operator.
        """
        P = np.asarray(to_dense(self.projection_matrix), dtype=float)
        _, cot_th, n_grid = self._grid_metric_terms
        _, B0th, B0ph, _ = self._background_field_grid_components
        if V_th.shape[0] != n_grid or V_ph.shape[0] != n_grid:
            raise ValueError(
                "Tangential component map height mismatch in toroidal RHS assembly: "
                f"V_th={V_th.shape}, V_ph={V_ph.shape}, n_grid={n_grid}."
            )

        curlE_op = (D_th @ V_ph) + (cot_th[:, None] * V_ph) - (D_ph @ V_th)
        dth_curlE_op = D_th @ curlE_op
        dph_curlE_op = D_ph @ curlE_op
        inv_Rb2 = 1.0 / (float(self.RI) ** 2)
        S_op = inv_Rb2 * ((B0th[:, None] * dph_curlE_op) - (B0ph[:, None] * dth_curlE_op))
        return asarray(P @ S_op)

    def _assemble_tangential_full_rhs_from_tangential_maps(
        self, *, V_th: np.ndarray, V_ph: np.ndarray, D_th: np.ndarray, D_ph: np.ndarray
    ) -> np.ndarray:
        """Assemble stacked projected/perpendicular tangential forcing operators.

        This keeps both scalar contractions of the same Er-free tangential
        driver field that underlies the current projected RHS construction:

            ``B0S · F_t(E)``
            ``B0S_perp · F_t(E)``.
        """
        P = np.asarray(to_dense(self.projection_matrix), dtype=float)
        _, cot_th, n_grid = self._grid_metric_terms
        _, B0th, B0ph, _ = self._background_field_grid_components
        if V_th.shape[0] != n_grid or V_ph.shape[0] != n_grid:
            raise ValueError(
                "Tangential component map height mismatch in full tangential RHS assembly: "
                f"V_th={V_th.shape}, V_ph={V_ph.shape}, n_grid={n_grid}."
            )

        curlE_op = (D_th @ V_ph) + (cot_th[:, None] * V_ph) - (D_ph @ V_th)
        dth_curlE_op = D_th @ curlE_op
        dph_curlE_op = D_ph @ curlE_op
        inv_Rb2 = 1.0 / (float(self.RI) ** 2)
        force_theta_op = inv_Rb2 * dph_curlE_op
        force_phi_op = -inv_Rb2 * dth_curlE_op

        projected_grid_op = (B0th[:, None] * force_theta_op) + (B0ph[:, None] * force_phi_op)
        perpendicular_grid_op = (-B0ph[:, None] * force_theta_op) + (
            B0th[:, None] * force_phi_op
        )
        projected_coeff_op = P @ projected_grid_op
        perpendicular_coeff_op = P @ perpendicular_grid_op
        return asarray(np.vstack([projected_coeff_op, perpendicular_coeff_op]))

    def _build_sh_toroidal_rhs_from_E_operator(self) -> np.ndarray:
        """Build SH toroidal RHS map ``E_coeffs -> rhs``.

        Uses the Er-free one-sided curlcurl projection:
            c = curl_Omega(E_S)
            S = (1/R^2) * [ B_theta * D_phi(c) - B_phi * D_theta(c) ]
            K = P @ S
        with ``D_phi = (1/sin(theta)) d/dphi``.
        """
        V_th, V_ph, _ = self._vector_basis_component_maps
        D_th, D_ph = self._rhs_scalar_derivative_operators
        return self._assemble_toroidal_rhs_from_tangential_maps(
            V_th=V_th, V_ph=V_ph, D_th=D_th, D_ph=D_ph
        )

    def _build_cs_toroidal_rhs_from_E_operator(self) -> np.ndarray:
        """Build CS toroidal RHS map ``E_coeffs -> rhs``.

        Uses the same Er-free one-sided curlcurl projection as the SH path:
            c = curl_Omega(E_S)
            S = (1/R^2) * [ B_theta * D_phi(c) - B_phi * D_theta(c) ]
            K = P @ S
        where CS ``D_phi`` already includes ``1/sin(theta)`` scaling.
        """
        V_th, V_ph, _ = self._vector_basis_component_maps
        D_th, D_ph = self._rhs_scalar_derivative_operators
        return self._assemble_toroidal_rhs_from_tangential_maps(
            V_th=V_th, V_ph=V_ph, D_th=D_th, D_ph=D_ph
        )

    @cached_property
    def full_radial_shell_rhs_from_E_operator(self) -> np.ndarray:
        """Build the explicit full radial-shell RHS map ``E_coeffs -> rhs``.

        This path is only available when a concrete ``RadialShellResponseModel``
        is injected. The model is expected to represent the shell scalar
        response

            ``-(Q_I - Delta_Omega E_r,I) / R_I^2``

        in the live toroidal coefficient space.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_rhs_operator(self)
        if op is None:
            n = int(self.basis.index_length)
            e_to_rhs = np.zeros((n, 2 * n), dtype=float)
            for i in range(2 * n):
                e_i = np.zeros(2 * n, dtype=float)
                e_i[i] = 1.0
                rhs_i = model.compute_rhs(self, e_i.reshape(2, n))
                e_to_rhs[:, i] = np.asarray(rhs_i, dtype=float).reshape(-1)
            return asarray(e_to_rhs)

        n = int(self.basis.index_length)
        dense = np.asarray(
            coerce_dense_operator_matrix(op, n_cols=2 * n),
            dtype=float,
        )
        if dense.shape != (n, 2 * n):
            raise ValueError(
                "Full radial-shell response operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {2 * n})."
            )
        return asarray(dense)

    @cached_property
    def full_radial_shell_known_source_from_E_operator(self) -> np.ndarray:
        """Return the known upper-side shell-current source map ``E_coeffs -> dt_jr^+``.

        The exact radial-shell scalar balance can be written as

            ``mu0 * dt_jr^+ = -(Q_I - Delta_Omega E_r,I) / R_I^2``.

        The existing full radial-shell RHS operator is the coefficient-space
        representation of the left-hand side ``mu0 * dt_jr^+``. This property
        exposes the underlying shell-current source map directly:

            ``dt_jr^+ = (1 / mu0) * rhs``.

        This is the natural ``g_surf``-level operator for the known forcing
        branch of the radial-shell closure.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_known_source_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, 2 * n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        if dense.shape != (n, 2 * n):
            raise ValueError(
                "Full radial-shell known-source operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {2 * n})."
        )
        return asarray(dense)

    @cached_property
    def full_radial_shell_gamma_known_operator(self) -> np.ndarray:
        """Alias for the forcing-side condensed source operator ``Gamma_known``."""
        return asarray(np.asarray(self.full_radial_shell_known_source_from_E_operator, dtype=float))

    @cached_property
    def full_radial_shell_known_source_from_JS_operator(self) -> np.ndarray:
        """Return the current-first shell source map ``J_S,coeffs -> dt_jr^+``.

        Some explicit radial-shell forcing models are most naturally written in
        terms of the shell-current increment first,

            ``dtK_S,known -> dt_jr,known^+``,

        with any shell-electric interpretation supplied only as a constitutive
        adapter on top. This property exposes that primitive shell-current
        source operator when the configured response model provides it.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_known_source_from_js_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, 2 * n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        if dense.shape != (n, 2 * n):
            raise ValueError(
                "Full radial-shell known-source-from-JS operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {2 * n})."
            )
        return asarray(dense)

    @cached_property
    def full_radial_shell_gamma_known_from_JS_operator(self) -> np.ndarray:
        """Alias for the current-first condensed source operator ``Gamma_known^(J)``."""
        return asarray(np.asarray(self.full_radial_shell_known_source_from_JS_operator, dtype=float))

    @cached_property
    def full_radial_shell_q_trace_from_JS_operator(self) -> np.ndarray:
        """Return the current-first shell trace map ``J_S,coeffs -> q``.

        When a forcing model exposes a primitive shell-current source law, the
        exact shell inversion also determines the corresponding

            ``q = d_r U|_{R_I^+} - E_{r,I}``

        trace in the mean-zero shell gauge.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_q_trace_from_js_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, 2 * n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        if dense.shape != (n, 2 * n):
            raise ValueError(
                "Full radial-shell q-trace-from-JS operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {2 * n})."
            )
        return asarray(dense)

    @cached_property
    def full_radial_shell_d_q_from_JS_operator(self) -> np.ndarray:
        """Alias for the current-first trace operator ``D_q^(J)``."""
        return asarray(np.asarray(self.full_radial_shell_q_trace_from_JS_operator, dtype=float))

    @cached_property
    def full_radial_shell_q_trace_from_E_operator(self) -> np.ndarray:
        """Return the exact forcing-side shell trace map ``E_coeffs -> q``.

        With ideality adopted, the exact forcing-side radial-shell trace is

            ``q = d_r U|_{R_I^+} - E_{r,I}``

        and satisfies the one-curl shell identity

            ``R_I * dt_psi^+ = Pi0(q)``
            ``dt_jr^+ = -(1/mu0) * Delta_S(q)``.

        This property exposes the corresponding shell operator directly. By
        default, configured radial-shell response models may either assemble
        ``q`` explicitly or allow it to be recovered exactly from the known
        source operator via the shell inversion

            ``q = R_I * jr_to_psi * dt_jr^+``.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_q_trace_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, 2 * n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        if dense.shape != (n, 2 * n):
            raise ValueError(
                "Full radial-shell q-trace operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {2 * n})."
        )
        return asarray(dense)

    @cached_property
    def full_radial_shell_d_q_operator(self) -> np.ndarray:
        """Alias for the forcing-side trace operator ``D_q``."""
        return asarray(np.asarray(self.full_radial_shell_q_trace_from_E_operator, dtype=float))

    @cached_property
    def full_radial_shell_lambda_gap_operator(self) -> np.ndarray:
        """Return the shell operator ``Lambda_gap`` acting on ``chi = dt_psi^+``.

        In the unified shell-source view,

            ``Lambda_gap(chi) = dt_jr^+``

        with the exact shell relation

            ``Lambda_gap = -(R_I / mu0) * Delta_S``.

        Concrete radial-shell response models may expose this through the
        common gap/co-energy interface even when their forcing-side source is
        still built by a reduced or current-first runtime model.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_lambda_gap_operator(self)
        n = int(self.basis.index_length)
        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=n), dtype=float)
        if dense.shape != (n, n):
            raise ValueError(
                "Full radial-shell Lambda_gap operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {n})."
        )
        return asarray(dense)

    @cached_property
    def full_radial_shell_condensed_operators(self) -> Any:
        """Return the unified shell-level operator bundle for the active branch."""
        model = self._require_radial_shell_response_model()
        return model.build_condensed_operators(self)

    @cached_property
    def tangential_full_rhs_from_E_operator(self) -> np.ndarray:
        """Build stacked projected/perpendicular tangential RHS coefficients."""
        V_th, V_ph, _ = self._vector_basis_component_maps
        D_th, D_ph = self._rhs_scalar_derivative_operators
        return self._assemble_tangential_full_rhs_from_tangential_maps(
            V_th=V_th, V_ph=V_ph, D_th=D_th, D_ph=D_ph
        )

    def compute_toroidal_rhs_from_E(self, E_coeffs: np.ndarray) -> np.ndarray:
        """Compute toroidal RHS coefficients from known E-field coefficients.

        Computes:
            rhs_lm = Projection(S_known)
        where ``S_known`` is the live inductive forcing scalar built from the
        shell tangential electric field only. In the SH/CS implementations,
        this is the projected scalar

            ``c = curl_Omega(E_S)``
            ``S_known = (1/R_I^2) * [ B0theta * d_phi^*(c) - B0phi * d_theta(c) ]``.

        Equivalently, with ``omega_E = hat(r)·curl(E) = c / R_I``,

            ``S_known = B0S · curl(omega_E * hat(r))``.

        So the forcing uses only the curl/Faraday content of ``E_S`` after the
        adopted elimination of radial electric derivatives. It is the forcing
        object of the current tangential closure, not the radial-shell scalar
        ``-(Q_I - Delta_Omega E_r,I)/R^2``.

        In the exact projected Maxwell driver, there is an additional
        contribution from the tangential part of ``curl(E)``. This routine
        intentionally omits that complementary term and therefore implements
        the reduced tangential forcing block, not the full projected driver.

        Parameters
        ----------
        E_coeffs : np.ndarray
            Tangential electric-field potential coefficients.

        Notes
        -----
        The sign convention is inherited from the basis projection operator.
        In this code path we apply the basis projection directly:
            ``K = P @ S_known``.

        This routine belongs to the current tangentially projected toroidal
        closure path. It does not provide a direct implementation of the
        radial-shell scalar response ``-(Q_I - Delta_Omega E_r,I)/R^2``.
        """
        if self.closure_mode == "radial_shell":
            return self.compute_full_radial_shell_rhs_from_E(E_coeffs)

        if self.closure_mode == "tangential_full":
            op = np.asarray(self.tangential_full_rhs_from_E_operator, dtype=float)
            return np.asarray(op @ np.asarray(E_coeffs).reshape(-1)).reshape(-1)

        if self.is_cs or is_sh_basis(self.basis):
            rhs_e = self._apply_direct_toroidal_rhs_operator(E_coeffs)
        else:
            rhs_e = self._compute_generic_toroidal_rhs_from_E(E_coeffs)
        return asarray(rhs_e)

    def compute_full_radial_shell_rhs_from_E(self, E_coeffs: np.ndarray) -> np.ndarray:
        """Compute the full radial-shell RHS using an injected response model."""
        model = self._require_radial_shell_response_model()
        if model.build_rhs_operator(self) is not None:
            op = np.asarray(self.full_radial_shell_rhs_from_E_operator, dtype=float)
            return np.asarray(op @ np.asarray(E_coeffs).reshape(-1)).reshape(-1)
        return np.asarray(model.compute_rhs(self, E_coeffs), dtype=float).reshape(-1)

    @cached_property
    def fieldline_advection_operator_raw(self) -> np.ndarray:
        """Return raw field-line advection operator ``A_raw``.

        This is the weak-form discretization of
            ``B0s · grad_Omega(.) = R * B0s · grad_S(.)``
        before applying the inverse-Laplacian toroidal-potential map.
        The unit-sphere form is intentional; it is why the final feedback block
        carries explicit ``1/R`` factors in front of ``dt_psi``.
        """
        logger.info("Building raw field-line advection operator...")

        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(
                self._project_aux_square_operator_to_state(aux.fieldline_advection_operator_raw)
            )

        if self.is_cs:
            D_th, D_ph, _ = self.cs_grid_derivative_operators
            B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
            B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
            advection_raw = (np.diag(B0th) @ D_th) + (np.diag(B0ph) @ D_ph)
            return asarray(advection_raw)

        if not hasattr(self.grid, "weights"):
            raise RuntimeError("Grid weights required for field-line advection construction.")

        weights = np.asarray(to_numpy(self.grid.weights)).reshape(-1)
        G = np.asarray(to_numpy(self.basis.get_evaluation_matrix(self.grid)))
        G_th = np.asarray(
            to_numpy(self.basis.get_evaluation_matrix(self.grid, derivative="theta"))
        )
        G_ph = np.asarray(to_numpy(self.basis.get_evaluation_matrix(self.grid, derivative="phi")))
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        horizontal_B_dot_grad_operator = (B0th[:, None] * G_th) + (B0ph[:, None] * G_ph)
        return asarray((G.T * weights) @ horizontal_B_dot_grad_operator)

    @cached_property
    def dtalpha_operator(self) -> np.ndarray:
        """Assemble the current alpha-space physics operator ``L_alpha``.

        Unknown is ``dt_alpha``. The assembled operator is the dt_alpha-native
        psi rewrite:
            ``(mass_dtalpha + A_raw @ ((1/R) * T_alpha_to_psi + D_r_dt_psi_from_dtalpha)) @ dt_alpha = K``.

        This is the live tangentially projected toroidal closure operator.
        """
        if self.closure_mode == "radial_shell":
            return asarray(self.full_radial_shell_dtalpha_operator)

        if self.closure_mode == "tangential_full":
            return asarray(self.first_principles_projected_dtalpha_operator)

        mass_dtalpha = np.asarray(to_numpy(self.mass_dtalpha))
        toroidal_feedback_dtalpha = np.asarray(
            to_numpy(self.toroidal_potential_feedback_dtalpha_operator)
        )
        return asarray(mass_dtalpha + toroidal_feedback_dtalpha)

    @cached_property
    def radial_shell_mass_dtalpha_operator(self) -> np.ndarray:
        """Return the shell-current mass block ``mu0 * T_alpha_to_jr``.

        This is the universal left-hand mass term in the reduced shell source
        law,

            ``mu0 * B0r * dt_alpha = dt_jr,known^+ + dt_jr,induced^+``,

        expressed in coefficient space through the weak-form map

            ``mu0 * T_alpha_to_jr``.

        It is shared by all explicit radial-shell closures and is not itself a
        closure approximation.
        """
        return asarray(mu0 * np.asarray(to_numpy(self.alpha_to_jr_coeff_operator)))

    @cached_property
    def full_radial_shell_feedback_dtalpha_operator(self) -> np.ndarray:
        """Return induced dt_alpha feedback for the explicit radial-shell closure.

        A concrete radial-shell response model may provide a dense feedback map

            ``dt_alpha -> feedback(dt_alpha)``

        representing the induced shell scalar contribution in

            ``mu0 * B0r * dt_alpha = rhs_driver + feedback(dt_alpha)``.

        The scalar radial-shell closure therefore enters the left-hand side as

            ``mu0 * B0r * dt_alpha - feedback(dt_alpha)``.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_feedback_dtalpha_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=n), dtype=float)
        if dense.shape != (n, n):
            raise ValueError(
                "Full radial-shell dt_alpha feedback operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {n})."
            )
        return asarray(dense)

    @cached_property
    def full_radial_shell_induced_source_dtalpha_operator(self) -> np.ndarray:
        """Return the induced upper-side shell-current source ``dt_alpha -> dt_jr^+``.

        In the exact radial-shell connector equation,

            ``mu0 * dt_jr^+ = feedback(dt_alpha)``,

        where the configured :class:`RadialShellResponseModel` supplies the
        coefficient-space feedback block. This property exposes the
        corresponding source-level operator directly:

            ``dt_jr^+ = (1 / mu0) * feedback(dt_alpha)``.

        Together with ``full_radial_shell_known_source_from_E_operator``, this
        gives the radial-shell closure in the source form

            ``-(R_I / mu0) * Delta_S(dt_psi^+) = dt_jr,known^+ + dt_jr,induced^+``.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_induced_source_dtalpha_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=n), dtype=float)
        if dense.shape != (n, n):
            raise ValueError(
                "Full radial-shell induced-source operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {n})."
            )
        return asarray(dense)

    @cached_property
    def full_radial_shell_induced_q_trace_dtalpha_operator(self) -> np.ndarray:
        """Return the induced shell trace map ``dt_alpha -> q = d_r U - E_r``.

        This mirrors the forcing-side ``q`` exposure, but for the induced
        branch. Once the induced upper-side shell-current source is known, the
        exact shell inversion determines the mean-zero induced trace through

            ``q_induced = R_I * jr_to_psi * dt_jr,induced^+``.
        """
        model = self._require_radial_shell_response_model()
        op = model.build_induced_q_trace_from_dtalpha_operator(self)
        n = int(self.basis.index_length)
        if op is None:
            return asarray(np.zeros((n, n), dtype=float))

        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=n), dtype=float)
        if dense.shape != (n, n):
            raise ValueError(
                "Full radial-shell induced q-trace operator has wrong shape: "
                f"{dense.shape}, expected ({n}, {n})."
            )
        return asarray(dense)

    @cached_property
    def full_radial_shell_dtalpha_operator(self) -> np.ndarray:
        """Return the explicit radial-shell dt_alpha closure operator.

        This assembles the scalar radial-shell left-hand side

            ``mu0 * B0r * dt_alpha - feedback(dt_alpha)``.

        Any nonlocal induced feedback is delegated to the configured
        :class:`RadialShellResponseModel`.
        """
        mass_dtalpha = np.asarray(self.radial_shell_mass_dtalpha_operator, dtype=float)
        feedback_dtalpha = np.asarray(self.full_radial_shell_feedback_dtalpha_operator, dtype=float)
        return asarray(mass_dtalpha - feedback_dtalpha)

    @cached_property
    def full_tangential_balance_residual_grid_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """Return grid residual maps for the exact tangential vector shell balance.

        For

            ``X = dt_psi + R_I * d_r(dt_psi)``

        the exact tangential vector identity is

            ``mu0 * dt_alpha * B0S = grad_S(X)``.

        This property returns the grid-space residual operators for the theta
        and phi-star components of

            ``mu0 * dt_alpha * B0S - grad_S(X)``.
        """
        G = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)), dtype=float)
        G_th = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta")),
            dtype=float,
        )
        G_ph = np.asarray(
            to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi")),
            dtype=float,
        )
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        x_from_dtalpha = np.asarray(to_numpy(self.alpha_to_psi_coeff_operator), dtype=float) + (
            float(self.RI) * np.asarray(to_numpy(self.radial_closure_dt_psi_from_dtalpha), dtype=float)
        )
        inv_Rb = 1.0 / float(self.RI)

        residual_theta = (mu0 * B0th[:, None] * G) - (inv_Rb * (G_th @ x_from_dtalpha))
        residual_phi = (mu0 * B0ph[:, None] * G) - (inv_Rb * (G_ph @ x_from_dtalpha))
        return asarray(residual_theta), asarray(residual_phi)

    @cached_property
    def first_principles_projected_dtalpha_operator_grid(self) -> np.ndarray:
        """Return grid map for the exact projected tangential shell closure.

        This is the scalar operator obtained by contracting the full
        tangential-vector residual with ``B0S``:

            ``B0S · (mu0 * dt_alpha * B0S - grad_S(X))``.
        """
        residual_theta, residual_phi = self.full_tangential_balance_residual_grid_operators
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        return asarray((B0th[:, None] * residual_theta) + (B0ph[:, None] * residual_phi))

    @cached_property
    def first_principles_projected_dtalpha_operator(self) -> np.ndarray:
        """Return coefficient-space projected tangential operator from first principles."""
        P = np.asarray(to_dense(self.projection_matrix), dtype=float)
        return asarray(P @ np.asarray(self.first_principles_projected_dtalpha_operator_grid, dtype=float))

    @cached_property
    def first_principles_perpendicular_dtalpha_operator_grid(self) -> np.ndarray:
        """Return grid map for the tangential component dropped by projection.

        This contracts the exact tangential-vector residual with the in-shell
        vector perpendicular to ``B0S``,

            ``B0S_perp = (-B0phi, B0theta)``.

        If this operator is active on the solved ``dt_alpha``, the projected
        scalar closure is discarding nontrivial tangential-balance content.
        """
        residual_theta, residual_phi = self.full_tangential_balance_residual_grid_operators
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        return asarray((-B0ph[:, None] * residual_theta) + (B0th[:, None] * residual_phi))

    @cached_property
    def first_principles_perpendicular_dtalpha_operator(self) -> np.ndarray:
        """Return coefficient-space operator for the dropped perpendicular component."""
        P = np.asarray(to_dense(self.projection_matrix), dtype=float)
        return asarray(
            P @ np.asarray(self.first_principles_perpendicular_dtalpha_operator_grid, dtype=float)
        )

    @cached_property
    def physics_residual_coeff_operator(self) -> np.ndarray:
        """Return coefficient-space physics residual operator for the active closure."""
        if self.closure_mode == "tangential_full":
            return asarray(
                np.vstack(
                    [
                        np.asarray(self.first_principles_projected_dtalpha_operator, dtype=float),
                        np.asarray(self.first_principles_perpendicular_dtalpha_operator, dtype=float),
                    ]
                )
            )
        return asarray(np.asarray(to_numpy(self.dtalpha_operator), dtype=float))

    @cached_property
    def physics_rhs_lift_operator(self) -> np.ndarray:
        """Return dense map from physics RHS coefficients to least-squares row RHS."""
        R_grid = to_dense(self.basis.get_evaluation_matrix(self.grid))
        if scipy.sparse.issparse(R_grid):
            R_grid = R_grid.toarray()
        R_grid = np.asarray(R_grid)

        if R_grid.ndim != 2:
            R_grid = R_grid.reshape(R_grid.shape[0], -1)
        rhs_cols = int(np.asarray(self.toroidal_rhs_from_E_operator, dtype=float).shape[0])
        if self.closure_mode == "tangential_full":
            if (2 * R_grid.shape[1]) != rhs_cols:
                raise ValueError(
                    "Tangential-full RHS lift mismatch: "
                    f"R_grid={R_grid.shape}, rhs_cols={rhs_cols}."
                )
            zeros = np.zeros_like(R_grid)
            return asarray(np.block([[R_grid, zeros], [zeros, R_grid]]))

        if R_grid.shape[1] != rhs_cols:
            raise ValueError(
                "Physics RHS lift mismatch: "
                f"R_grid={R_grid.shape}, rhs_cols={rhs_cols}."
            )
        return asarray(R_grid)

    @cached_property
    def physics_residual_row_operator(self) -> np.ndarray:
        """Return least-squares row operator for the active toroidal physics block."""
        row_lift = np.asarray(self.physics_rhs_lift_operator, dtype=float)
        residual_coeff = np.asarray(self.physics_residual_coeff_operator, dtype=float)
        if row_lift.shape[1] != residual_coeff.shape[0]:
            raise ValueError(
                "Physics row operator shape mismatch: "
                f"row_lift={row_lift.shape}, residual_coeff={residual_coeff.shape}."
            )
        return asarray(row_lift @ residual_coeff)

    @cached_property
    def projection_matrix(self) -> np.ndarray:
        """Get projection matrix from Basis (Grid -> Coeffs)."""
        return self.basis.construct_scalar_projection_matrix(self.grid)

    @cached_property
    def alpha_to_jr_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``alpha`` to ``jr``.

        For ``alpha = jr / B0r`` and static background field,
            ``jr = B0r * alpha`` pointwise on the grid.
        In coefficient space this is assembled in weak form:
            ``T_alpha_to_jr = P @ diag(Br_grid) @ G``.
        """
        G = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)))
        P = np.asarray(to_dense(self.projection_matrix))
        B0r = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        if B0r.size != G.shape[0]:
            raise ValueError(f"alpha_to_jr assembly mismatch: B0r={B0r.shape}, G={G.shape}.")
        return asarray(P @ (B0r[:, None] * G))

    @cached_property
    def jr_to_psi_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``dt_jr`` to ``dt_psi``.

        This uses the same sign convention as ``m_imp`` and the basis toroidal
        vector convention ``Curl(T r) = -r x Grad T``:
            ``jr = -(R / mu0) * Delta_S(psi)``
        or equivalently
            ``jr = -(1 / (mu0 * R)) * Delta_Omega(psi)``.
        On the mean-zero SH subspace this gives
            ``psi_lm = +(mu0 * R) / (l(l+1)) * jr_lm``
        for ``l >= 1``.

        This relation is physics-fixed by the toroidal magnetic convention
        ``Curl(T r) = -r x Grad(T)`` and should therefore remain invariant if
        the generic surface Helmholtz df sign is changed elsewhere.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(
                self._project_aux_square_operator_to_state(aux.jr_to_psi_coeff_operator)
            )

        if self.is_cs:
            return asarray(float(-mu0 * self.RI) * np.asarray(self.cs_laplacian_inverse))

        if is_sh_basis(self.basis):
            l_arr = np.asarray(to_numpy(self.basis.n)).reshape(-1).astype(float)
            laplacian_eigenvalues = l_arr * (l_arr + 1.0)
            inverse_laplacian_eigenvalues = np.zeros_like(laplacian_eigenvalues)
            mask_nonzero_modes = laplacian_eigenvalues > 0
            inverse_laplacian_eigenvalues[mask_nonzero_modes] = (
                -1.0 / laplacian_eigenvalues[mask_nonzero_modes]
            )
            return asarray(np.diag(float(-mu0 * self.RI) * inverse_laplacian_eigenvalues))

        lap = np.asarray(to_dense(self.basis.get_laplacian_operator(self.RI)))
        if lap.ndim != 2:
            lap = lap.reshape(lap.shape[0], -1)
        lap_pinv = tensor_pinv(lap, n_leading_flattened=1)
        return asarray(float(-mu0 / self.RI) * lap_pinv)

    @cached_property
    def alpha_to_psi_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``dt_alpha`` to ``dt_psi``."""
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            raw = np.asarray(
                self._project_aux_square_operator_to_state(aux._alpha_to_psi_coeff_operator_raw)
            )
            return asarray(raw)
        raw = np.asarray(to_numpy(self._alpha_to_psi_coeff_operator_raw))
        return asarray(raw)

    @cached_property
    def _alpha_to_psi_coeff_operator_raw(self) -> np.ndarray:
        """Raw open-boundary coefficient-space map from ``dt_alpha`` to ``dt_psi``."""
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(
                self._project_aux_square_operator_to_state(aux._alpha_to_psi_coeff_operator_raw)
            )
        jr_to_psi = np.asarray(to_numpy(self.jr_to_psi_coeff_operator))
        alpha_to_jr = np.asarray(to_numpy(self.alpha_to_jr_coeff_operator))
        return asarray(jr_to_psi @ alpha_to_jr)

    @cached_property
    def radial_closure_dt_psi_from_dtalpha(self) -> np.ndarray:
        """Map ``dt_alpha`` to ``d_r(dt_psi)``.

        From
            ``dt_psi = T_jr_to_psi @ dt_jr``
        and
            ``dt_jr = B0r * dt_alpha``,
        the one-sided radial derivative is
            ``d_r(dt_psi) = (1/R) * T_alpha_to_psi @ dt_alpha
                            + T_jr_to_psi @ d_r(dt_jr)``.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            raw = np.asarray(
                self._project_aux_square_operator_to_state(aux.radial_closure_dt_psi_from_dtalpha)
            )
            return asarray(raw)

        alpha_to_psi = np.asarray(to_numpy(self._alpha_to_psi_coeff_operator_raw))
        jr_to_psi = np.asarray(to_numpy(self.jr_to_psi_coeff_operator))
        radial_closure_dtalpha = np.asarray(to_numpy(self.radial_closure_dtalpha))
        inv_Rb = 1.0 / float(self.RI)
        raw = inv_Rb * alpha_to_psi + (jr_to_psi @ radial_closure_dtalpha)
        return asarray(raw)

    @cached_property
    def toroidal_potential_feedback_dtalpha_operator(self) -> np.ndarray:
        """Return the dt_alpha feedback block written in toroidal-potential form.

        This is the exact hybrid rewrite of the field-line coupling term:
            ``A_raw @ ((1/R) * dt_psi + d_r(dt_psi))``.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(
                self._project_aux_square_operator_to_state(
                    aux.toroidal_potential_feedback_dtalpha_operator
                )
            )

        advection_raw = np.asarray(to_numpy(self.fieldline_advection_operator_raw))
        alpha_to_psi = np.asarray(to_numpy(self.alpha_to_psi_coeff_operator))
        radial_closure_dtpsi = np.asarray(to_numpy(self.radial_closure_dt_psi_from_dtalpha))
        inv_Rb = 1.0 / float(self.RI)
        return asarray(advection_raw @ ((inv_Rb * alpha_to_psi) + radial_closure_dtpsi))

    @cached_property
    def jr_to_alpha_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``jr`` to ``alpha``.

        For ``B0r * alpha = jr`` we use a weak-form minimum-norm branch selector:
            ``argmin_alpha ||W^(1/2)(B0r*alpha - jr)||^2 + floor^2 ||W^(1/2)alpha||^2``.
        The returned coefficient-space map is
            ``T_jr_to_alpha = (H^+) @ (G^T W B0r) @ G``
        where
            ``H = G^T W (B0r^2 + floor^2) G``.
        """
        G = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)))
        if G.ndim != 2:
            G = G.reshape(G.shape[0], -1)
        n_grid = int(G.shape[0])

        B0r = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        if B0r.size != n_grid:
            raise ValueError(f"jr_to_alpha assembly mismatch: B0r={B0r.shape}, G={G.shape}.")

        if hasattr(self.grid, "weights") and self.grid.weights is not None:
            weights = np.asarray(to_numpy(self.grid.weights)).reshape(-1)
            if weights.size != n_grid:
                weights = np.ones(n_grid, dtype=float)
        else:
            weights = np.ones(n_grid, dtype=float)
        weights = np.maximum(weights, 0.0)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(n_grid, dtype=float)

        floor = float(max(self.inverse_radial_field_floor, 0.0))
        w_scale = weights * (B0r * B0r + floor * floor)
        lhs = (G.T * w_scale) @ G
        lhs = 0.5 * (lhs + lhs.T)
        rcond = float(self._default_pinv_rcond(lhs.shape))
        lhs_pinv = self._pinv_symmetric(lhs, rcond=rcond)

        rhs_lift = G.T * (weights * B0r)
        t_grid_to_alpha = lhs_pinv @ rhs_lift
        return asarray(t_grid_to_alpha @ G)

    def _get_cs_psi_gauge_rows(self, n_coeff: int, apply_psi_gauge: bool) -> np.ndarray:
        """Return hard psi gauge rows for direct dpsi solves.

        Policy:
            - CS basis: optional mean-zero hard row when ``apply_psi_gauge`` is true.
            - SH/other bases: no hard psi gauge rows.
        """
        if not (self.is_cs and bool(apply_psi_gauge)):
            return np.zeros((0, int(n_coeff)), dtype=float)

        if hasattr(self.basis, "get_scalar_gauge_constraint_matrix"):
            row = np.asarray(
                self.basis.get_scalar_gauge_constraint_matrix(
                    n_coeff=int(n_coeff), mode="mean_zero"
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

    def solve_dt_psi_superposed(
        self,
        rhs_physics: np.ndarray,
        rhs_constraint: np.ndarray,
        constraint_operator: Any,
        m_imp_to_jr_operator: Any,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        penalty_rhs: Any = None,
        hinv_rtol: float = 0.0,
        apply_psi_gauge: bool = False,
    ) -> np.ndarray:
        """Solve for ``dpsi/dt`` via one-shot constrained ``dt_alpha`` solve."""
        return self.solver.solve_dt_psi_superposed(
            rhs_physics=rhs_physics,
            rhs_constraint=rhs_constraint,
            constraint_operator=constraint_operator,
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            penalty_rhs=penalty_rhs,
            hinv_rtol=hinv_rtol,
            apply_psi_gauge=apply_psi_gauge,
        )

    def build_dtalpha_from_toroidal_rhs_matrix(
        self,
        *,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> np.ndarray:
        """Build dense map ``toroidal_rhs -> dt_alpha``."""
        return self.solver.build_dtalpha_from_toroidal_rhs_matrix(
            constraint_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )

    # -------------------------------------------------------------------------
    # Time Evolution Logic
    # -------------------------------------------------------------------------

    def _get_psi_gauge_projector_dense(
        self, m_imp_to_jr_operator: Any, apply_psi_gauge: Optional[bool] = None
    ) -> np.ndarray:
        """Return explicit gauge projector applied after MP inversion."""
        return self.solver._get_psi_gauge_projector_dense(
            m_imp_to_jr_operator=m_imp_to_jr_operator, apply_psi_gauge=apply_psi_gauge
        )

    def _get_dtalpha_to_dt_psi_map_cached(
        self, *, m_imp_to_jr_operator: Any, apply_psi_gauge: bool
    ) -> np.ndarray:
        """Return cached dense map ``dt_alpha -> dpsi/dt``."""
        return self.solver._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator, apply_psi_gauge=apply_psi_gauge
        )

    def build_dt_psi_from_toroidal_rhs_matrix(
        self,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        apply_psi_gauge: bool = False,
    ) -> np.ndarray:
        """Build dense map ``toroidal_rhs -> dpsi/dt``."""
        return self.solver.build_dt_psi_from_toroidal_rhs_matrix(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            constraint_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
            apply_psi_gauge=apply_psi_gauge,
        )

    def build_psi_dynamics_matrix(
        self,
        psi_to_E_operator: np.ndarray,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        apply_psi_gauge: bool = False,
    ) -> np.ndarray:
        """Build the linear operator: psi → d(psi)/dt."""
        return self.solver.build_psi_dynamics_matrix(
            psi_to_E_operator=psi_to_E_operator,
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            constraint_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
            apply_psi_gauge=apply_psi_gauge,
        )

    def get_psi_dynamics_operator(
        self,
        psi_to_E_operator: Any,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        dense: bool = False,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        apply_psi_gauge: bool = False,
    ) -> "LinearMap":
        """Get linear operator ``psi -> dpsi/dt``."""
        return self.solver.get_psi_dynamics_operator(
            psi_to_E_operator=psi_to_E_operator,
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            constraint_operator=constraint_operator,
            dense=dense,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
            apply_psi_gauge=apply_psi_gauge,
        )
