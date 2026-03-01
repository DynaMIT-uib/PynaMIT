"""Toroidal system assembly for the dt_alpha evolution equation.

The assembled operator has the form:
    L_dtalpha * dt_alpha = rhs_toroidal

The field-line feedback part is assembled in the dt_alpha-native psi rewrite:
    mass_dtalpha * dt_alpha
    + A_raw * ((1/R) * dt_psi + d_r(dt_psi))
    = rhs_toroidal
where ``dt_psi`` is understood as the toroidal magnetic-potential response
induced by ``dt_alpha`` through the static ``jr <-> psi`` relation.

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
from typing import Any, Optional

import numpy as np
import scipy.sparse
from functools import cached_property

from pynamit.utils import to_numpy, asarray, tensor_pinv
from pynamit.simulation.geometry_utils import to_dense
from pynamit.math.linear_map import as_linear_map
from pynamit.math.constants import mu0
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.spherical_harmonics import sh_operators
from pynamit.simulation.toroidal_closure import ToroidalClosureProjector

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
        # Derivative operators:
        # - closure_derivative_basis/rhs_derivative_basis: primary derivative
        #   basis for toroidal closure assembly.
        # - radial_derivative_basis: optional override for Er/radial-closure
        #   derivative chains.
        #
        # In cs_dominant full-induction we can set all three to one auxiliary SH
        # basis to keep the full toroidal closure assembly basis-consistent.
        base_closure_basis = basis if closure_derivative_basis is None else closure_derivative_basis
        self.closure_derivative_basis = base_closure_basis
        self.rhs_derivative_basis = (
            base_closure_basis if rhs_derivative_basis is None else rhs_derivative_basis
        )
        self.radial_derivative_basis = (
            self.rhs_derivative_basis if radial_derivative_basis is None else radial_derivative_basis
        )
        self._cs_derivative_operator_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        
        self.is_cs = (getattr(self.basis, "kind", "") == "CS")
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
            closure_derivative_basis=closure_basis,
            rhs_derivative_basis=closure_basis,
            radial_derivative_basis=closure_basis,
            toroidal_solver=self.toroidal_solver,
            toroidal_preconditioner=self.toroidal_preconditioner,
            toroidal_tolerance=self.toroidal_tolerance,
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

    def configure_toroidal_solver(
        self,
        *,
        solver: str,
        preconditioner: Optional[str],
        tolerance: float,
    ) -> None:
        """Configure solver policy for toroidal least-squares solves."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        if solver not in LeastSquaresSolver.VALID_SOLVERS:
            raise ValueError(
                f"Invalid toroidal solver '{solver}'. Valid options: "
                f"{LeastSquaresSolver.VALID_SOLVERS}."
            )
        if preconditioner is not None and preconditioner not in LeastSquaresSolver.VALID_PRECONDITIONERS:
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
        return (
            self.toroidal_solver,
            self.toroidal_preconditioner,
            float(self.toroidal_tolerance),
        )

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
        G_th = np.asarray(to_dense(deriv_basis.get_evaluation_matrix(self.grid, derivative="theta")))
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
        """Construct alpha-space inertia matrix.

        For unknown ``dt_alpha`` (with ``dt_jr = B0r * dt_alpha``), the inertia
        block is:
            ``C_alpha = mu0 * <Y, |B0s|^2 Y>``.
        This avoids explicit ``1/B0r`` factors in the solved physics operator.
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
                "Alpha inertia factor/grid size mismatch: "
                f"factor={factor.shape}, G={G.shape}."
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
            return asarray(
                self._project_aux_square_operator_to_state(aux.radial_closure_dtalpha)
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
        scalar_evaluation_matrix = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)))
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
    def toroidal_rhs_from_E_operator(self) -> np.ndarray:
        """Build matrix mapping E coefficients to toroidal RHS coefficients.

        Since ``compute_toroidal_rhs_from_E`` is linear in ``E_coeffs``, we can
        represent it as:
            rhs = E_to_rhs @ E_coeffs.flatten()

        Returns
        -------
        np.ndarray
            Matrix of shape ``(N, 2*N)`` mapping flattened ``E_coeffs`` to RHS coefficients.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            F_aux = np.asarray(aux.toroidal_rhs_from_E_operator)
            return asarray(
                self._toroidal_closure_projector.project_vector_rhs_operator_to_state(F_aux)
            )

        if self.is_cs:
            return asarray(self._build_cs_toroidal_rhs_from_E_operator())

        if (not self.is_cs) and (getattr(self.basis, "kind", "") == "SH"):
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

    def _normalize_tangential_grid_vector(
        self,
        value: Any,
        *,
        name: str,
    ) -> np.ndarray:
        """Return tangential grid vector with canonical shape ``(2, n_grid)``."""
        n_grid = int(np.asarray(to_numpy(self.grid.theta)).reshape(-1).size)
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"{name} must have two components (theta, phi).")
            comp_th = np.asarray(to_numpy(value[0])).reshape(-1)
            comp_ph = np.asarray(to_numpy(value[1])).reshape(-1)
            arr = np.vstack([comp_th, comp_ph])
        else:
            arr = np.asarray(to_numpy(value))
            if arr.ndim == 1:
                if arr.size == 2 * n_grid:
                    arr = arr.reshape(2, n_grid)
                else:
                    raise ValueError(
                        f"{name} 1D input must have size 2*n_grid={2*n_grid}, got {arr.size}."
                    )
            elif arr.ndim == 2:
                if arr.shape == (2, n_grid):
                    pass
                elif arr.shape == (n_grid, 2):
                    arr = arr.T
                else:
                    raise ValueError(
                        f"{name} must have shape (2, n_grid) or (n_grid, 2), got {arr.shape}."
                    )
            else:
                raise ValueError(f"{name} must be tuple/list or 2D array, got ndim={arr.ndim}.")

        if arr.shape != (2, n_grid):
            raise ValueError(f"{name} normalized shape mismatch: got {arr.shape}, expected (2, {n_grid}).")
        return np.asarray(arr, dtype=float)

    def _build_twist_rate_known_rhs_operator(self) -> np.ndarray:
        """Build dense map ``[u, dr_u]_grid -> toroidal RHS coefficients``.

        Returns a matrix ``K_u`` with shape ``(N_coeff, 4*N_grid)`` such that
            ``rhs_u = K_u @ [u_theta, u_phi, dr_u_theta, dr_u_phi]``.
        """
        B0r = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        inv_Rb = 1.0 / float(self.RI)

        n_grid = int(theta_rad.size)
        if B0r.size != n_grid or B0th.size != n_grid or B0ph.size != n_grid:
            raise ValueError(
                "Grid field sizes are inconsistent in u-known RHS map assembly: "
                f"B0r={B0r.size}, B0th={B0th.size}, B0ph={B0ph.size}, n_grid={n_grid}."
            )

        if self.is_cs:
            D_th, D_ph, _ = self.cs_rhs_derivative_operators
            D_th = np.asarray(D_th, dtype=float)
            D_ph = np.asarray(D_ph, dtype=float)
            div_u_theta = D_th + np.diag(cot_th)
            div_u_phi = D_ph
        else:
            P = np.asarray(to_dense(self.projection_matrix), dtype=float)
            G_th = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta")))
            G_ph = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi")))
            div_u_theta = (G_th @ P) + np.diag(cot_th)
            div_u_phi = G_ph @ P

        # S_u = (B0r/Rb) div(u) + B0th*(-u_th/Rb - dr_u_th) + B0ph*(-u_ph/Rb - dr_u_ph)
        br_div_scale = np.diag(B0r * inv_Rb)
        s_u_th = (br_div_scale @ div_u_theta) + np.diag(-inv_Rb * B0th)
        s_u_ph = (br_div_scale @ div_u_phi) + np.diag(-inv_Rb * B0ph)
        s_dr_u_th = np.diag(-B0th)
        s_dr_u_ph = np.diag(-B0ph)
        s_op = np.hstack([s_u_th, s_u_ph, s_dr_u_th, s_dr_u_ph])

        P = np.asarray(to_dense(self.projection_matrix), dtype=float)
        return asarray(P @ s_op)

    @cached_property
    def twist_rate_known_rhs_operator(self) -> np.ndarray:
        """Dense map ``[u, dr_u]_grid -> toroidal RHS coefficients``."""
        return self._build_twist_rate_known_rhs_operator()

    def _compute_twist_rate_known_rhs_coeffs(
        self,
        *,
        twist_rate_known_grid: Any,
        dr_twist_rate_known_grid: Any,
    ) -> np.ndarray:
        """Project ``u``-known contribution to toroidal RHS coefficient space.

        Implements:
            S_u = (B0r/Rb) div_Omega(u)
                  + B0th * (-(1/Rb) u_th - dr_u_th)
                  + B0ph * (-(1/Rb) u_ph - dr_u_ph)
            K_u = P @ S_u
        """
        u_known = self._normalize_tangential_grid_vector(twist_rate_known_grid, name="twist_rate_known_grid")
        dr_twist_rate_known = self._normalize_tangential_grid_vector(dr_twist_rate_known_grid, name="dr_twist_rate_known_grid")
        n_grid = u_known.shape[1]
        rhs_op = np.asarray(to_numpy(self.twist_rate_known_rhs_operator))
        if rhs_op.shape[1] != 4 * n_grid:
            raise ValueError(
                "u-known RHS operator width mismatch: "
                f"K_u={rhs_op.shape}, n_grid={n_grid}."
            )
        stacked = np.concatenate(
            [
                np.asarray(u_known[0]).reshape(-1),
                np.asarray(u_known[1]).reshape(-1),
                np.asarray(dr_twist_rate_known[0]).reshape(-1),
                np.asarray(dr_twist_rate_known[1]).reshape(-1),
            ],
            axis=0,
        )
        return asarray(rhs_op @ stacked)

    def _build_sh_toroidal_rhs_from_E_operator(self) -> np.ndarray:
        """Build SH toroidal RHS map ``E_coeffs -> rhs``.

        Uses the Er-free one-sided curlcurl projection:
            c = curl_Omega(E_S)
            S = (1/R^2) * [ B_theta * D_phi(c) - B_phi * D_theta(c) ]
            K = P @ S
        with ``D_phi = (1/sin(theta)) d/dphi``.
        """
        N = self.basis.index_length
        P = np.asarray(to_dense(self.projection_matrix))
        G_th = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta")))
        G_ph = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi")))

        # Vector basis tensor: (component, grid, potential_type, coeff)
        G_vec = np.asarray(to_dense(self.basis.get_vector_basis_matrix(self.grid)))
        if G_vec.ndim != 4 or G_vec.shape[0] != 2 or G_vec.shape[2] != 2 or G_vec.shape[3] != N:
            raise ValueError(
                "Unexpected vector basis shape for SH RHS map: "
                f"{G_vec.shape}, expected (2, N_grid, 2, {N})."
            )
        n_grid = int(G_vec.shape[1])

        # Map flattened E coefficients [pol; tor] to tangential components.
        V_th = np.hstack([G_vec[0, :, 0, :], G_vec[0, :, 1, :]])  # (N_grid, 2N)
        V_ph = np.hstack([G_vec[1, :, 0, :], G_vec[1, :, 1, :]])  # (N_grid, 2N)

        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        if B0th.size != n_grid or B0ph.size != n_grid or cot_th.size != n_grid:
            raise ValueError("Grid field sizes are inconsistent in SH RHS map assembly.")

        # c = curl_Omega(E_S) = d_theta E_phi + cot(theta) E_phi - d_phi E_theta
        dEth_ph_op = G_ph @ (P @ V_th)
        dEph_th_op = G_th @ (P @ V_ph)
        curlE_op = dEph_th_op + (cot_th[:, None] * V_ph) - dEth_ph_op

        dth_curlE_op = G_th @ (P @ curlE_op)
        dph_curlE_op = G_ph @ (P @ curlE_op)

        inv_Rb2 = 1.0 / (float(self.RI) ** 2)
        S_op = inv_Rb2 * (
            (B0th[:, None] * dph_curlE_op) - (B0ph[:, None] * dth_curlE_op)
        )
        return asarray(P @ S_op)

    def _build_cs_toroidal_rhs_from_E_operator(self) -> np.ndarray:
        """Build CS toroidal RHS map ``E_coeffs -> rhs``.

        Uses the same Er-free one-sided curlcurl projection as the SH path:
            c = curl_Omega(E_S)
            S = (1/R^2) * [ B_theta * D_phi(c) - B_phi * D_theta(c) ]
            K = P @ S
        where CS ``D_phi`` already includes ``1/sin(theta)`` scaling.
        """
        N = self.basis.index_length

        # Horizontal derivatives use the configured toroidal RHS derivative basis.
        D_th_h, D_ph_h, _ = self.cs_rhs_derivative_operators
        D_th_h = np.asarray(D_th_h)
        D_ph_h = np.asarray(D_ph_h)
        P = np.asarray(to_dense(self.projection_matrix))

        # Vector basis tensor: (2, N_grid, 2, N_coeff).
        G_vec = np.asarray(to_dense(self.basis.get_vector_basis_matrix(self.grid)))
        if G_vec.ndim != 4 or G_vec.shape[0] != 2 or G_vec.shape[2] != 2 or G_vec.shape[3] != N:
            raise ValueError(
                "Unexpected vector basis shape for CS RHS map: "
                f"{G_vec.shape}, expected (2, N_grid, 2, {N})."
            )
        n_grid = int(G_vec.shape[1])

        # Map flattened E coefficients [pol; tor] to tangential components.
        V_th = np.hstack([G_vec[0, :, 0, :], G_vec[0, :, 1, :]])  # (N_grid, 2N)
        V_ph = np.hstack([G_vec[1, :, 0, :], G_vec[1, :, 1, :]])  # (N_grid, 2N)

        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        theta_rad = np.deg2rad(np.asarray(to_numpy(self.grid.theta)).reshape(-1))
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        if B0th.size != n_grid or B0ph.size != n_grid or cot_th.size != n_grid:
            raise ValueError("Grid field sizes are inconsistent in CS RHS map assembly.")

        # c = curl_Omega(E_S) = d_theta E_phi + cot(theta) E_phi - d_phi E_theta
        dEth_ph_op = D_ph_h @ V_th
        dEph_th_op = D_th_h @ V_ph
        curlE_op = dEph_th_op + (cot_th[:, None] * V_ph) - dEth_ph_op

        dth_curlE_op = D_th_h @ curlE_op
        dph_curlE_op = D_ph_h @ curlE_op

        inv_Rb2 = 1.0 / (float(self.RI) ** 2)
        S_op = inv_Rb2 * (
            (B0th[:, None] * dph_curlE_op) - (B0ph[:, None] * dth_curlE_op)
        )
        return asarray(P @ S_op)

    def compute_toroidal_rhs_from_E(
        self,
        E_coeffs: np.ndarray,
        *,
        twist_rate_known_grid: Optional[Any] = None,
        dr_twist_rate_known_grid: Optional[Any] = None,
        allow_missing_dr_twist_rate_known: bool = False,
    ) -> np.ndarray:
        """Compute toroidal RHS coefficients from known E-field coefficients.

        Computes:
            rhs_lm = Projection(S_known)
        where ``S_known`` is derived from Faraday and Gauss identities after
        eliminating radial derivatives of ``E``.

        Optional additive known-term support:
            ``K_total = K_E + K_u``
        where ``K_u`` uses ``twist_rate_known_grid`` and ``dr_twist_rate_known_grid`` as
        externally provided one-sided closure inputs.

        Parameters
        ----------
        E_coeffs : np.ndarray
            Tangential electric-field potential coefficients.
        twist_rate_known_grid : optional
            Known tangential ``u`` on the grid (theta/phi components), as
            ``(u_theta, u_phi)`` or array shape ``(2, n_grid)``.
        dr_twist_rate_known_grid : optional
            One-sided radial derivative of ``twist_rate_known_grid`` with matching shape.
        allow_missing_dr_twist_rate_known : bool, optional
            If ``True`` and ``twist_rate_known_grid`` is provided without
            ``dr_twist_rate_known_grid``, use ``dr_twist_rate_known = 0``.

        Notes
        -----
        The sign convention is inherited from the basis projection operator.
        In this code path we apply the basis projection directly:
            ``K = P @ S_known``.
        """
        if twist_rate_known_grid is None and dr_twist_rate_known_grid is not None:
            raise ValueError(
                "dr_twist_rate_known_grid was provided without twist_rate_known_grid. "
                "Provide both, or neither."
            )
        if twist_rate_known_grid is not None and dr_twist_rate_known_grid is None:
            if not allow_missing_dr_twist_rate_known:
                raise ValueError(
                    "twist_rate_known_grid was provided without dr_twist_rate_known_grid. "
                    "Provide dr_twist_rate_known_grid explicitly, or set "
                    "allow_missing_dr_twist_rate_known=True to assume dr_twist_rate_known=0."
                )
            n_grid = int(np.asarray(to_numpy(self.grid.theta)).reshape(-1).size)
            dr_twist_rate_known_grid = np.zeros((2, n_grid), dtype=float)

        rhs_u = None
        if twist_rate_known_grid is not None:
            rhs_u = np.asarray(
                to_numpy(
                    self._compute_twist_rate_known_rhs_coeffs(
                        twist_rate_known_grid=twist_rate_known_grid,
                        dr_twist_rate_known_grid=dr_twist_rate_known_grid,
                    )
                )
            ).reshape(-1)

        if self.is_cs:
            E_flat = np.asarray(E_coeffs).reshape(-1)
            rhs_e = np.asarray(to_numpy(self.toroidal_rhs_from_E_operator @ E_flat)).reshape(-1)
            if rhs_u is not None:
                rhs_e = rhs_e + rhs_u
            return asarray(rhs_e)

        if (not self.is_cs) and (getattr(self.basis, "kind", "") == "SH"):
            E_flat = np.asarray(E_coeffs).reshape(-1)
            rhs_e = np.asarray(to_numpy(self.toroidal_rhs_from_E_operator @ E_flat)).reshape(-1)
            if rhs_u is not None:
                rhs_e = rhs_e + rhs_u
            return asarray(rhs_e)

        # Generic fallback (non-SH/non-CS): apply the same Er-free RHS
        # formula used in the specialized builders.
        inv_Rb2 = 1.0 / (float(self.RI) ** 2)

        # 1. Get evaluation and projection matrices (Cached)
        G_th = to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="theta"))
        G_ph = to_dense(self.basis.get_evaluation_matrix(self.grid, derivative="phi"))
        P = to_dense(self.projection_matrix)

        # 2. Evaluate E_S on grid
        # E_coeffs contains [Poloidal, Toroidal] potentials
        Eth_grid, Eph_grid = self.basis.evaluate(E_coeffs, self.grid, vector_type="tangential")

        B0th = to_numpy(self.b_field.vec.theta)
        B0ph = to_numpy(self.b_field.vec.phi)

        # 3. Helpers for grid derivatives via spectral projection
        def get_derivs(f_val):
            c = P @ f_val
            return G_th @ c, G_ph @ c

        # 4. Compute unit-sphere curl and its derivatives on grid
        _, dEth_ph = get_derivs(Eth_grid)
        dEph_th, _ = get_derivs(Eph_grid)
        theta_rad = np.deg2rad(to_numpy(self.grid.theta)).flatten()
        sin_th = np.sin(theta_rad)
        sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
        cot_th = np.cos(theta_rad) / sin_th_safe
        curlE = dEph_th + cot_th * Eph_grid - dEth_ph
        dcurl_th, dcurl_ph = get_derivs(curlE)

        S_known = inv_Rb2 * (B0th * dcurl_ph - B0ph * dcurl_th)
        rhs_e = np.asarray(to_numpy(P @ S_known)).reshape(-1)
        if rhs_u is not None:
            rhs_e = rhs_e + rhs_u
        return asarray(rhs_e)

    @cached_property
    def fieldline_advection_operator_raw(self) -> np.ndarray:
        """Return raw field-line advection operator ``A_raw``.

        This is the weak-form discretization of
            ``B0s · grad_Omega(.)``
        before applying the inverse-Laplacian toroidal-potential map.
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
        G_th = np.asarray(to_numpy(self.basis.get_evaluation_matrix(self.grid, derivative="theta")))
        G_ph = np.asarray(to_numpy(self.basis.get_evaluation_matrix(self.grid, derivative="phi")))
        B0th = np.asarray(to_numpy(self.b_field.vec.theta)).reshape(-1)
        B0ph = np.asarray(to_numpy(self.b_field.vec.phi)).reshape(-1)
        horizontal_B_dot_grad_operator = (B0th[:, None] * G_th) + (B0ph[:, None] * G_ph)
        return asarray((G.T * weights) @ horizontal_B_dot_grad_operator)

    @cached_property
    def dtalpha_operator(self) -> np.ndarray:
        """Assemble the alpha-space physics operator ``L_alpha``.

        Unknown is ``dt_alpha``. The assembled operator is the dt_alpha-native
        psi rewrite:
            ``(mass_dtalpha + A_raw @ ((1/R) * T_alpha_to_psi + D_r_dt_psi_from_dtalpha)) @ dt_alpha = K``.
        """
        mass_dtalpha = np.asarray(to_numpy(self.mass_dtalpha))
        toroidal_feedback_dtalpha = np.asarray(
            to_numpy(self.toroidal_potential_feedback_dtalpha_operator)
        )
        return asarray(mass_dtalpha + toroidal_feedback_dtalpha)

    @cached_property
    def _dtalpha_grid_residual_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R_grid, A_grid)`` for residuals ``A_grid x - R_grid b``.

        Unknown ``x`` is ``dt_alpha`` in coefficient space, with
        ``A_grid = R_grid @ L_alpha``.
        """
        L_alpha = np.asarray(to_numpy(self.dtalpha_operator))
        R_grid = to_dense(self.basis.get_evaluation_matrix(self.grid))
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

        For ``alpha = jr / B0r`` and static background field,
            ``jr = B0r * alpha`` pointwise on the grid.
        In coefficient space this is assembled in weak form:
            ``T_alpha_to_jr = P @ diag(Br_grid) @ G``.
        """
        G = np.asarray(to_dense(self.basis.get_evaluation_matrix(self.grid)))
        P = np.asarray(to_dense(self.projection_matrix))
        B0r = np.asarray(to_numpy(self.b_field.vec.r)).reshape(-1)
        if B0r.size != G.shape[0]:
            raise ValueError(
                "alpha_to_jr assembly mismatch: "
                f"B0r={B0r.shape}, G={G.shape}."
            )
        return asarray(P @ (B0r[:, None] * G))

    @cached_property
    def jr_to_psi_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``dt_jr`` to ``dt_psi``.

        In code convention,
            ``jr = (R / mu0) * Laplacian(psi)``,
        so
            ``psi = mu0 * R * Delta_Omega^{-1}(jr)``
        on the mean-zero scalar subspace.
        """
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(self._project_aux_square_operator_to_state(aux.jr_to_psi_coeff_operator))

        if self.is_cs:
            return asarray(float(mu0 * self.RI) * np.asarray(self.cs_laplacian_inverse))

        basis_kind = getattr(self.basis, "kind", "")
        if basis_kind == "SH":
            l_arr = np.asarray(to_numpy(self.basis.n)).reshape(-1).astype(float)
            laplacian_eigenvalues = l_arr * (l_arr + 1.0)
            inverse_laplacian_eigenvalues = np.zeros_like(laplacian_eigenvalues)
            mask_nonzero_modes = laplacian_eigenvalues > 0
            inverse_laplacian_eigenvalues[mask_nonzero_modes] = (
                -1.0 / laplacian_eigenvalues[mask_nonzero_modes]
            )
            return asarray(np.diag(float(mu0 * self.RI) * inverse_laplacian_eigenvalues))

        lap = np.asarray(to_dense(self.basis.get_laplacian_operator(self.RI)))
        if lap.ndim != 2:
            lap = lap.reshape(lap.shape[0], -1)
        lap_pinv = tensor_pinv(lap, n_leading_flattened=1)
        return asarray(float(mu0 / self.RI) * lap_pinv)

    @cached_property
    def alpha_to_psi_coeff_operator(self) -> np.ndarray:
        """Coefficient-space map from ``dt_alpha`` to ``dt_psi``."""
        if self._use_auxiliary_closure_basis:
            aux = self._auxiliary_closure_matrices
            return asarray(
                self._project_aux_square_operator_to_state(aux.alpha_to_psi_coeff_operator)
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
            return asarray(
                self._project_aux_square_operator_to_state(
                    aux.radial_closure_dt_psi_from_dtalpha
                )
            )

        alpha_to_psi = np.asarray(to_numpy(self.alpha_to_psi_coeff_operator))
        jr_to_psi = np.asarray(to_numpy(self.jr_to_psi_coeff_operator))
        radial_closure_dtalpha = np.asarray(to_numpy(self.radial_closure_dtalpha))
        inv_Rb = 1.0 / float(self.RI)
        return asarray(inv_Rb * alpha_to_psi + (jr_to_psi @ radial_closure_dtalpha))

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
            raise ValueError(
                "jr_to_alpha assembly mismatch: "
                f"B0r={B0r.shape}, G={G.shape}."
            )

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

    def _get_toroidal_problem_bundle(
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
        cached = self._toroidal_problem_cache.get(cache_key)
        if cached is not None:
            return cached

        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.linear_map import as_linear_map, diagonal_linear_map

        dtalpha_operator = np.asarray(to_numpy(self.dtalpha_operator))
        n_coeff = dtalpha_operator.shape[0]
        R_grid, A_grid = self._dtalpha_grid_residual_maps
        op_A_grid = as_linear_map(A_grid)
        physics_weight = self._build_physics_sqrt_weight(op_A_grid.shape[0], weighting)

        operators = [op_A_grid]
        data_shapes = [(op_A_grid.shape[0],)]
        sqrt_weights = [physics_weight]

        if penalty_operator is not None and penalty_scaling > 0:
            op_penalty = as_linear_map(penalty_operator) * penalty_scaling
            operators.append(op_penalty)
            data_shapes.append((op_penalty.shape[0],))
            sqrt_weights.append(None)

        # Use explicit data rows for Tikhonov regularization so lambda remains
        # in absolute physical units (avoid automatic rescaling in
        # LeastSquaresProblem.regularization_weights).
        if regularization_lambda > 0:
            op_reg = diagonal_linear_map(np.ones(n_coeff)) * float(
                np.sqrt(max(regularization_lambda, 0.0))
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
            "n_coeff": int(n_coeff),
            "grid_rows": int(op_A_grid.shape[0]),
        }
        self._toroidal_problem_cache[cache_key] = bundle
        return bundle

    def _resolve_toroidal_rcond(self, n_coeff: int, hinv_rtol: float) -> float:
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

    def _solve_toroidal_problem(
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
        """Solve weighted toroidal LS problem in ``dt_alpha`` with optional equalities."""
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        bundle = self._get_toroidal_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        problem = bundle["problem"]
        R_grid = bundle["R_grid"]
        n_coeff = int(bundle["n_coeff"])
        rcond = self._resolve_toroidal_rcond(n_coeff, hinv_rtol)

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
            solver=self.toroidal_solver,
            tolerance=max(rcond, self.toroidal_tolerance),
            preconditioner=self.toroidal_preconditioner,
        )
        preconditioner = None
        if equality_operator is None and self.toroidal_preconditioner is not None:
            preconditioner = solver.build_preconditioner(
                problem=problem,
                preconditioner_type=self.toroidal_preconditioner,
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
        bundle = self._get_toroidal_problem_bundle(
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
        )
        n_coeff = int(bundle["n_coeff"])
        rcond = self._resolve_toroidal_rcond(n_coeff, hinv_rtol)
        key = (
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            self._toroidal_solver_signature(),
        )
        cached = self._dtalpha_unconstrained_map_cache.get(key)
        if cached is not None:
            return cached
        rhs_physics_basis = np.eye(n_coeff, dtype=float)
        alpha_map = np.asarray(
            self._solve_toroidal_problem(
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

    def _get_constrained_dtalpha_maps(
        self,
        *,
        alpha_map_operator: Any,
        weighting: str,
        regularization_lambda: float,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Return cached constrained maps for direct ``dt_alpha`` constraints.

        Returns dense matrices:
            ``M_phys_dtalpha``: physics RHS -> constrained homogeneous ``dt_alpha`` response
            ``M_corr_dtalpha``: constraint RHS -> constrained inhomogeneous ``dt_alpha`` response
            ``C_dtalpha``: hard-constraint operator in solve space

        The constrained solution is represented as:
            ``dt_alpha = M_phys_dtalpha @ rhs_physics + M_corr_dtalpha @ rhs_constraint``
        with hard equalities enforced on ``C_dtalpha @ dt_alpha``.
        """
        C_dtalpha = np.asarray(to_dense(as_linear_map(alpha_map_operator)))
        if C_dtalpha.ndim != 2:
            C_dtalpha = C_dtalpha.reshape(C_dtalpha.shape[0], -1)
        n_coeff = int(C_dtalpha.shape[1])
        m_constraints = int(C_dtalpha.shape[0])
        rcond = self._resolve_toroidal_rcond(n_coeff, hinv_rtol)
        key = (
            id(alpha_map_operator),
            weighting,
            float(regularization_lambda),
            id(penalty_operator) if penalty_operator is not None else 0,
            float(penalty_scaling),
            float(rcond),
            self._toroidal_solver_signature(),
        )
        cached = self._dtalpha_constrained_maps_cache.get(key)
        if cached is not None:
            return cached

        M_phys_dtalpha = np.asarray(
            self._solve_toroidal_problem(
                rhs_physics_coeffs=np.eye(n_coeff, dtype=float),
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
                equality_operator=C_dtalpha,
                equality_rhs=np.zeros(m_constraints, dtype=float),
            )
        ).reshape(n_coeff, n_coeff)

        if m_constraints > 0:
            M_corr_dtalpha = np.asarray(
                self._solve_toroidal_problem(
                    rhs_physics_coeffs=np.zeros((n_coeff, m_constraints), dtype=float),
                    weighting=weighting,
                    regularization_lambda=regularization_lambda,
                    penalty_operator=penalty_operator,
                    penalty_scaling=penalty_scaling,
                    hinv_rtol=hinv_rtol,
                    equality_operator=C_dtalpha,
                    equality_rhs=np.eye(m_constraints, dtype=float),
                )
            ).reshape(n_coeff, m_constraints)
        else:
            M_corr_dtalpha = np.zeros((n_coeff, 0), dtype=float)

        maps = {
            "C_dtalpha": C_dtalpha,
            "M_phys_dtalpha": np.asarray(M_phys_dtalpha),
            "M_corr_dtalpha": np.asarray(M_corr_dtalpha),
        }
        self._dtalpha_constrained_maps_cache[key] = maps
        return maps

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
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Solve for ``dpsi/dt`` via one-shot constrained ``dt_alpha`` solve.

        This is the monolithic toroidal path used at runtime:
        1) solve ``dt_alpha`` from physics + hard constraints,
        2) map ``dt_alpha -> dpsi`` via a single linear conversion map.
        """
        rhs_p = np.asarray(to_numpy(rhs_physics)).reshape(-1)
        dtalpha_to_dt_psi = self._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )

        if constraint_operator is None:
            M_alpha = self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha = M_alpha @ rhs_p
            return asarray((dtalpha_to_dt_psi @ dtalpha).reshape(-1))

        maps = self._get_constrained_dtalpha_maps(
            alpha_map_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
        )
        rhs_c = np.asarray(to_numpy(rhs_constraint)).reshape(-1)
        dtalpha = maps["M_phys_dtalpha"] @ rhs_p
        if maps["M_corr_dtalpha"].shape[1] > 0:
            dtalpha = dtalpha + maps["M_corr_dtalpha"] @ rhs_c
        dt_psi = dtalpha_to_dt_psi @ dtalpha
        return asarray(dt_psi.reshape(-1))

    # -------------------------------------------------------------------------
    # Time Evolution Logic
    # -------------------------------------------------------------------------

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
            if not hasattr(self.basis, "get_evaluation_matrix"):
                raise RuntimeError(
                    "Scalar gauge projector requested, but basis does not provide "
                    "get_scalar_gauge_constraint_matrix() or get_evaluation_matrix(grid)."
                )
            g_mat = np.asarray(to_dense(as_linear_map(self.basis.get_evaluation_matrix(self.grid))))
            if g_mat.ndim != 2 or g_mat.shape[1] != n:
                raise RuntimeError(
                    "Failed to build generic scalar gauge row from basis.get_evaluation_matrix(grid)."
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

    def _get_dtalpha_to_dt_psi_map_cached(
        self,
        *,
        m_imp_to_jr_operator: Any,
        use_pinning: bool,
    ) -> np.ndarray:
        """Return cached dense map ``dt_alpha -> dpsi/dt``.

        The map is assembled as:
            ``M = P_gauge @ alpha_to_psi``
        so the toroidal evolution solve can remain in ``dt_alpha`` space and
        only convert to ``dpsi`` once at the end.
        """
        key = (
            id(m_imp_to_jr_operator),
            bool(use_pinning),
        )
        cached = self._dtalpha_to_dt_psi_map_cache.get(key)
        if cached is not None:
            return cached

        alpha_to_psi = np.asarray(to_numpy(self.alpha_to_psi_coeff_operator))
        gauge_projector = np.asarray(
            self._get_psi_gauge_projector_dense(
                m_imp_to_jr_operator,
                use_pinning=use_pinning,
            )
        )
        map_dtalpha_to_dt_psi = gauge_projector @ alpha_to_psi
        self._dtalpha_to_dt_psi_map_cache[key] = map_dtalpha_to_dt_psi
        return map_dtalpha_to_dt_psi

    def build_dt_psi_from_toroidal_rhs_matrix(
        self,
        m_imp_to_jr_operator: Any,
        constraint_operator: Any = None,
        weighting: str = "none",
        regularization_lambda: float = 0.0,
        penalty_operator: Any = None,
        penalty_scaling: float = 0.0,
        hinv_rtol: float = 0.0,
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Build dense map ``toroidal_rhs -> dpsi/dt``.

        This is the ``K -> dpsi`` block used by all toroidal linear chains.
        """
        if constraint_operator is not None:
            maps = self._get_constrained_dtalpha_maps(
                alpha_map_operator=constraint_operator,
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
            dtalpha_from_K = maps["M_phys_dtalpha"]
        else:
            dtalpha_from_K = self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )

        # Step 4: dt_alpha -> dpsi conversion (gauge-aware).
        dtalpha_to_dt_psi = self._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            use_pinning=use_pinning,
        )
        return asarray(dtalpha_to_dt_psi @ dtalpha_from_K)

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
        use_pinning: bool = False,
    ) -> np.ndarray:
        """Build the linear operator: psi → d(psi)/dt.

        This constructs the full chain:
            psi → E_psi → K → dpsi/dt

        Each step is linear, so the composition is also linear:
            d(psi)/dt = L_psi_psi @ psi + (other contributions)
        """
        from pynamit.simulation.geometry_utils import to_dense

        N = self.basis.index_length

        # Step 1: map E coefficients to toroidal RHS coefficients.
        E_to_rhs = to_numpy(self.toroidal_rhs_from_E_operator)  # (N, 2*N)

        # Step 2: psi → E_psi (reshape if needed)
        # NOTE: psi_to_E is post-resistivity (physical E). No additional scaling here.
        if hasattr(psi_to_E_operator, "to_dense"):
             psi_to_E = psi_to_E_operator.to_dense()
        else:
             psi_to_E = to_numpy(psi_to_E_operator)
        if psi_to_E.ndim == 3:  # shape (2, N, N)
            psi_to_E = psi_to_E.reshape(2 * N, N)

        dt_psi_from_K = self.build_dt_psi_from_toroidal_rhs_matrix(
            m_imp_to_jr_operator=m_imp_to_jr_operator,
            constraint_operator=constraint_operator,
            weighting=weighting,
            regularization_lambda=regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            hinv_rtol=hinv_rtol,
            use_pinning=use_pinning,
        )

        # Final chain: psi -> E -> toroidal_rhs -> dt_alpha -> dpsi/dt.
        L_psi_psi = (dt_psi_from_K @ E_to_rhs) @ psi_to_E

        return asarray(L_psi_psi)

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
        use_pinning: bool = False,
    ) -> "LinearMap":
        """Get linear operator ``psi -> dpsi/dt``.

        Matrix-free and dense paths share one direct ``dpsi`` solve map, so both
        use identical hard-constraint/gauge semantics.
        """
        from pynamit.math.linear_map import LinearMap, as_linear_map

        N = self.basis.index_length

        if dense or constraint_operator is not None:
            L_dense = self.build_psi_dynamics_matrix(
                psi_to_E_operator,
                m_imp_to_jr_operator,
                constraint_operator=constraint_operator,
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

        E_to_rhs_op = as_linear_map(np.asarray(to_numpy(self.toroidal_rhs_from_E_operator)))
        dtalpha_from_K_op = as_linear_map(
            self._get_unconstrained_dtalpha_map_cached(
                weighting=weighting,
                regularization_lambda=regularization_lambda,
                penalty_operator=penalty_operator,
                penalty_scaling=penalty_scaling,
                hinv_rtol=hinv_rtol,
            )
        )
        dtalpha_to_dt_psi_op = as_linear_map(
            self._get_dtalpha_to_dt_psi_map_cached(
                m_imp_to_jr_operator=m_imp_to_jr_operator,
                use_pinning=use_pinning,
            )
        )

        def matvec(x):
            y = psi_to_E_op.matvec(asarray(x).reshape(-1))
            y = E_to_rhs_op.matvec(y)
            y = dtalpha_from_K_op.matvec(y)
            y = dtalpha_to_dt_psi_op.matvec(y)
            return asarray(y)

        def rmatvec(x):
            y = dtalpha_to_dt_psi_op.rmatvec(asarray(x).reshape(-1))
            y = dtalpha_from_K_op.rmatvec(y)
            y = E_to_rhs_op.rmatvec(y)
            y = psi_to_E_op.rmatvec(y)
            return asarray(y)

        def to_dense_func():
            return self.build_psi_dynamics_matrix(
                psi_to_E_operator,
                m_imp_to_jr_operator,
                constraint_operator=constraint_operator,
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
