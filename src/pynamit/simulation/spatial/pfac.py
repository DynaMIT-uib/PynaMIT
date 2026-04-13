"""PFAC (Poloidal Field-Aligned Current) integration module.

This module handles the computation of magnetospheric coupling through
radial integration of field-aligned currents. The T_to_Ve operator maps
external toroidal potential to poloidal potential.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import xarray as xr

from pynamit.math.constants import mu0
from pynamit.primitives.basis import is_cs_like_basis, is_sh_basis
from pynamit.primitives.grid import Grid
from pynamit.simulation.settings import MainfieldKind
from pynamit.utils import tensor_pinv
from pynamit.simulation.spatial.geometry_utils import to_dense, get_radial_shift_diagonal

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToroidalRMClosureOperators:
    """Cached operators for RM normal-current termination diagnostics.

    These operators capture the unambiguous parts of the dynamic toroidal
    boundary closure problem:
    - map ionospheric ``alpha`` coefficients to the incoming normal current at
      ``R_M`` along field lines,
    - solve the surface-divergence closure potential ``chi`` from
      ``Delta_S chi = j_n(R_M)``,
    - and recover the resulting divergent horizontal closure current on the
      ``R_M`` boundary.

    The electromagnetic reaction of that closure current below ``R_M`` is a
    separate operator and is intentionally not folded into these diagnostics.
    """

    alpha_to_normal_current_rm_grid: np.ndarray
    alpha_to_normal_current_rm_coeff: np.ndarray
    alpha_to_closure_potential_rm_coeff: np.ndarray
    alpha_to_boundary_psi_rm_coeff: np.ndarray
    alpha_to_divergent_closure_current_rm_grid: np.ndarray


@dataclass(frozen=True)
class PFACRadialSplitOperators:
    """Dynamic PFAC return split relative to a chosen evaluation radius."""

    open_internal: np.ndarray
    open_external: np.ndarray
    effective_internal: np.ndarray
    effective_external: np.ndarray


class PFACIntegrator:
    """Handles PFAC (Poloidal Field-Aligned Current) computation.

    This class encapsulates the radial integration needed to compute
    the T_to_Ve operator, which maps external toroidal potential to
    poloidal potential at the ionospheric boundary.

    Parameters
    ----------
    basis : Basis
        The spectral (SH) basis for physics computations.
    solution_space : Basis
        The basis used for solution variables (may differ from basis).
    mainfield : Any
        The main magnetic field model.
    RI : float
        Ionosphere radius in meters.
    RM : float or None
        Magnetosphere radius in meters, or None if no boundary.
    FAC_integration_steps : array-like
        Radial steps for FAC integration.
    ignore_PFAC : bool
        If True, return zero operator (radial field approximation).
    magnetospheric_shielding : bool
        If True, include RM shielding-style roundtrip coupling for induced
        poloidal pathways. Explicit imposed boundary channels remain closed.
    """

    _T_TO_VE_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
    _TOROIDAL_RM_CLOSURE_CACHE: dict[tuple[Any, ...], ToroidalRMClosureOperators] = {}

    def __init__(
        self,
        basis: "Basis",
        solution_space: "Basis",
        mainfield: Any,
        RI: float,
        RM: Optional[float],
        FAC_integration_steps: Any,
        ignore_PFAC: bool = False,
        magnetospheric_shielding: bool = True,
    ) -> None:
        self.basis = basis
        self.solution_space = solution_space
        self.mainfield = mainfield
        self.RI = RI
        self.RM = RM
        self.FAC_integration_steps = FAC_integration_steps
        self.ignore_PFAC = ignore_PFAC
        self.magnetospheric_shielding = bool(magnetospheric_shielding)

    def get_coupling_factors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute magnetospheric coupling factors.

        Returns
        -------
        br_rm_to_ri_shift : ndarray
            Diagonal radial shift operator mapping RM response to RI.
        br_ri_to_rm_shift : ndarray
            Diagonal radial shift operator mapping RI response to RM.
        rm_roundtrip_denominator : ndarray
            Closure denominator ``1 - br_rm_to_ri_shift * br_ri_to_rm_shift``.
            The locked transfer factor is ``1 / rm_roundtrip_denominator``.

        Raises
        ------
        ValueError
            If RM is None (no magnetosphere boundary).
        """
        if self.RM is None:
            raise ValueError("Cannot compute coupling factors without RM.")

        br_rm_to_ri_shift = get_radial_shift_diagonal(
            self.basis, self.RM, self.RI, kind="external"
        )
        br_ri_to_rm_shift = get_radial_shift_diagonal(
            self.basis, self.RI, self.RM, kind="internal"
        )
        rm_roundtrip = br_rm_to_ri_shift * br_ri_to_rm_shift
        rm_roundtrip_denominator = 1.0 - rm_roundtrip
        return br_rm_to_ri_shift, br_ri_to_rm_shift, rm_roundtrip_denominator

    def _get_solution_space_mean_zero_projector(self, n_coeff: int) -> Optional[np.ndarray]:
        """Return validated solution-space mean-zero projector when available."""
        projector_builder = getattr(self.solution_space, "get_mean_zero_projector", None)
        if projector_builder is None:
            return None

        projector = np.asarray(projector_builder(n_coeff=n_coeff), dtype=float)
        expected_shape = (n_coeff, n_coeff)
        if projector.shape != expected_shape:
            raise ValueError(
                "Solution-space mean-zero projector has invalid shape "
                f"{projector.shape}; expected {expected_shape}."
            )
        return projector

    def _build_pfac_source_backbone(
        self, G_Ve_to_JS_closure: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the cached-independent PFAC source backbone."""
        n_sol = int(self.solution_space.index_length)
        n_sh = int(self.basis.index_length)
        fac_steps = np.asarray(self.FAC_integration_steps, dtype=float)
        delta_k = np.diff(fac_steps)
        rks = fac_steps[:-1] + 0.5 * delta_k
        G_inv_sh = tensor_pinv(G_Ve_to_JS_closure, n_leading_flattened=2, rtol=1e-12).reshape(
            n_sh, -1
        )
        L_sol = to_dense(self.solution_space.get_laplacian_operator(self.RI))
        m_imp_to_jr_sol_op = -(self.RI / mu0) * L_sol
        if is_cs_like_basis(self.solution_space):
            P_sol = self._get_solution_space_mean_zero_projector(n_sol)
            if P_sol is not None:
                m_imp_to_jr_sol_op = m_imp_to_jr_sol_op @ P_sol
        return G_inv_sh, np.asarray(m_imp_to_jr_sol_op, dtype=float), fac_steps, delta_k, rks

    def _map_closure_operator_to_solution(
        self, closure_operator: np.ndarray, grid: Grid
    ) -> np.ndarray:
        """Map a closure-basis operator back to solution coefficients."""
        if self.solution_space is self.basis or is_sh_basis(self.solution_space):
            mapped = np.asarray(closure_operator, dtype=float)
        else:
            E_sh = to_dense(self.basis.get_evaluation_matrix(grid))
            E_sol = to_dense(self.solution_space.get_evaluation_matrix(grid))
            P_sol = tensor_pinv(E_sol, rtol=1e-12)
            mapped = np.asarray((P_sol @ E_sh) @ closure_operator, dtype=float)

        if is_cs_like_basis(self.solution_space):
            P_g = self._get_solution_space_mean_zero_projector(int(self.solution_space.index_length))
            if P_g is not None:
                mapped = np.asarray(P_g @ mapped, dtype=float)
        return mapped

    def compute_T_to_Ve(
        self, G_Ve_to_JS_closure: np.ndarray, grid: Grid, *, rm_boundary_mode: str = "closed"
    ) -> xr.DataArray:
        """Construct the T_to_Ve operator by integrating radially.

        Physics:
          1. Internal potential m_imp produces radial current jr via Laplacian.
          2. jr maps to sheet current JS(rk) in the magnetospheric layers.
          3. Integrated JS produces poloidal potential Ve at ionospheric boundary.

        Implementation:
          We use a 'Spectral Accelerated Kernel'. For SH basis, this stays in
          coefficient space. For grid bases (CS), it maps to SH for propagation.

        Parameters
        ----------
        G_Ve_to_JS_closure : np.ndarray
            Closure-basis induction operator used for PFAC integration.
        grid : Grid
            The spatial grid for evaluation.

        Returns
        -------
        xr.DataArray
            The T_to_Ve operator mapping toroidal to poloidal potential.
        """
        n_sol = self.solution_space.index_length
        n_sh = self.basis.index_length
        T_to_Ve = xr.DataArray(np.zeros((n_sol, n_sol)), dims=("current_pot", "field_pot"))

        if self.mainfield.kind == MainfieldKind.RADIAL or self.ignore_PFAC:
            return T_to_Ve

        fac_steps = np.asarray(self.FAC_integration_steps, dtype=float)
        fac_steps_key = tuple(np.round(fac_steps, 6).tolist())
        rm_mode = str(rm_boundary_mode).lower()
        if rm_mode not in {"open", "closed"}:
            raise ValueError(
                f"Invalid rm_boundary_mode {rm_boundary_mode!r}; expected 'open' or 'closed'."
            )
        cache_key = (
            getattr(self.basis, "kind", None),
            int(getattr(self.basis, "index_length", -1)),
            int(getattr(self.basis, "Nmax", -1)),
            int(getattr(self.basis, "Mmax", -1)),
            getattr(self.solution_space, "kind", None),
            int(getattr(self.solution_space, "index_length", -1)),
            int(getattr(self.solution_space, "N", -1)),
            int(getattr(grid, "hash", id(grid))),
            float(self.RI),
            float(self.RM) if self.RM is not None else None,
            getattr(self.mainfield, "kind", None),
            int(getattr(self.mainfield, "epoch", 0)),
            float(getattr(self.mainfield, "B0", 0.0) or 0.0),
            fac_steps_key,
            rm_mode,
        )
        cached = PFACIntegrator._T_TO_VE_CACHE.get(cache_key)
        if cached is not None:
            return xr.DataArray(cached.copy(), dims=("current_pot", "field_pot"))

        # Radial steps
        rk_steps = fac_steps
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        # Integration backbone (spectral space)
        G_inv_sh = tensor_pinv(G_Ve_to_JS_closure, n_leading_flattened=2, rtol=1e-12).reshape(
            n_sh, -1
        )

        # Source operator factor in solution-basis coefficients:
        #   jr = -(RI/mu0) * Laplacian(m_imp)
        # from the toroidal magnetic identity ``Curl(T r) = -r x Grad(T)``.
        # This is a physics-fixed magnetic relation, not a generic Helmholtz
        # representation choice, so it must stay unchanged when the internal
        # cf/df signs are flipped.
        L_sol = to_dense(self.solution_space.get_laplacian_operator(self.RI))
        m_imp_to_jr_sol_op = -(self.RI / mu0) * L_sol
        if is_cs_like_basis(self.solution_space):
            P_sol = self._get_solution_space_mean_zero_projector(n_sol)
            if P_sol is not None:
                m_imp_to_jr_sol_op = m_imp_to_jr_sol_op @ P_sol

        # Is this a pure spectral simulation?
        is_pure_sh = self.solution_space is self.basis or is_sh_basis(self.solution_space)

        # Accumulator (spectral result: Ve_sh_coeffs / m_imp_coeffs)
        T_accum = np.zeros((n_sh, n_sol))

        for i, rk in enumerate(rks):
            step_mat = self._compute_integration_step(
                rk, grid, G_inv_sh, m_imp_to_jr_sol_op, n_sol, n_sh, rm_boundary_mode=rm_mode
            )
            T_accum += Delta_k[i] * step_mat

        # Map back to result basis
        if is_pure_sh:
            T_to_Ve.values = T_accum
        else:
            # Map integrated SH coefficients back to solver coefficients
            # (hybrid CS/SH architecture). This is a basis conversion, not a
            # convention change in the underlying PFAC physics.
            E_sh = to_dense(self.basis.get_evaluation_matrix(grid))
            E_sol = to_dense(self.solution_space.get_evaluation_matrix(grid))
            P_sol = tensor_pinv(E_sol, rtol=1e-12)
            T_to_Ve.values = (P_sol @ E_sh) @ T_accum

        if is_cs_like_basis(self.solution_space):
            P_g = self._get_solution_space_mean_zero_projector(n_sol)
            if P_g is not None:
                T_to_Ve.values = P_g @ T_to_Ve.values

        PFACIntegrator._T_TO_VE_CACHE[cache_key] = T_to_Ve.values.copy()
        return T_to_Ve

    def compute_T_to_Ve_radius_split(
        self,
        eval_radius: float,
        G_Ve_to_JS_closure: np.ndarray,
        grid: Grid,
    ) -> PFACRadialSplitOperators:
        """Return the dynamic PFAC ``T -> V_e`` split at an arbitrary radius.

        The split is defined relative to ``eval_radius``:

        - sources below ``eval_radius`` contribute to the internal branch;
        - sources above ``eval_radius`` contribute to the external branch;
        - the ``R_M`` closure/shielding reaction contributes through the
          effective branch assembled from the same harmonic roundtrip used by
          the runtime at ``R_I``.
        """
        n_sol = int(self.solution_space.index_length)
        zeros = np.zeros((n_sol, n_sol), dtype=float)

        if self.mainfield.kind == MainfieldKind.RADIAL or self.ignore_PFAC:
            return PFACRadialSplitOperators(
                open_internal=zeros,
                open_external=zeros,
                effective_internal=zeros,
                effective_external=zeros,
            )

        r_eval = float(eval_radius)
        tol = max(1e-12 * max(abs(self.RI), abs(r_eval), abs(self.RM or r_eval), 1.0), 1e-9)
        if r_eval < float(self.RI) - tol:
            raise ValueError(
                f"PFAC radius split requires eval_radius >= RI; got {eval_radius!r} < {self.RI!r}."
            )
        if self.RM is not None and r_eval > float(self.RM) + tol:
            raise ValueError(
                f"PFAC radius split requires eval_radius <= RM; got {eval_radius!r} > {self.RM!r}."
            )

        # Exact boundary special case: the effective field at a locked RM boundary is zero.
        if self.RM is not None and abs(r_eval - float(self.RM)) <= tol:
            open_internal = np.asarray(
                self.compute_T_to_Ve_radius_split(float(self.RM) - 10.0 * tol, G_Ve_to_JS_closure, grid).open_internal,
                dtype=float,
            )
            open_external = np.zeros_like(open_internal)
            if self.magnetospheric_shielding:
                effective_internal = np.zeros_like(open_internal)
                effective_external = np.zeros_like(open_internal)
            else:
                effective_internal = np.asarray(open_internal, dtype=float)
                effective_external = np.asarray(open_external, dtype=float)
            return PFACRadialSplitOperators(
                open_internal=open_internal,
                open_external=open_external,
                effective_internal=effective_internal,
                effective_external=effective_external,
            )

        G_inv_sh, m_imp_to_jr_sol_op, _fac_steps, delta_k, rks = self._build_pfac_source_backbone(
            G_Ve_to_JS_closure
        )
        n_sh = int(self.basis.index_length)
        open_internal = np.zeros((n_sh, n_sol), dtype=float)
        open_external = np.zeros((n_sh, n_sol), dtype=float)
        effective_internal = np.zeros((n_sh, n_sol), dtype=float)
        effective_external = np.zeros((n_sh, n_sol), dtype=float)

        use_rm_boundary = (self.RM is not None) and bool(self.magnetospheric_shielding)
        if use_rm_boundary:
            S_ext_RM = get_radial_shift_diagonal(self.basis, self.RM, r_eval, kind="external")
            S_int_eval = get_radial_shift_diagonal(self.basis, r_eval, self.RM, kind="internal")
            factor_vec = -1.0 / (1.0 - S_ext_RM * S_int_eval)
        else:
            S_ext_RM = np.zeros(n_sh, dtype=float)
            factor_vec = -1.0 * np.ones(n_sh, dtype=float)

        for i, rk in enumerate(rks):
            theta_m, phi_m = self.mainfield.map_coords(self.RI, rk, grid.theta, grid.phi)
            m_grid = Grid(theta=theta_m, phi=phi_m)
            rk_b = self.mainfield.discretize(grid, rk)
            m_b = self.mainfield.discretize(m_grid, self.RI)
            M_source = to_dense(self.solution_space.get_evaluation_matrix(m_grid))
            m_imp_to_JS_rk = np.einsum(
                "ij,jk->ijk",
                [rk_b.vec.theta / m_b.vec.r, rk_b.vec.phi / m_b.vec.r],
                M_source @ m_imp_to_jr_sol_op,
                optimize=True,
            ).reshape(-1, n_sol)
            base_step = np.asarray(G_inv_sh @ m_imp_to_JS_rk, dtype=float)

            if rk < r_eval - tol:
                prop_internal = get_radial_shift_diagonal(self.basis, rk, r_eval, kind="internal")
                prop_external = np.zeros(n_sh, dtype=float)
            elif rk > r_eval + tol:
                prop_internal = np.zeros(n_sh, dtype=float)
                prop_external = get_radial_shift_diagonal(self.basis, rk, r_eval, kind="external")
            else:
                # Midpoint quadrature should not land exactly on the evaluation radius in practice.
                prop_internal = np.zeros(n_sh, dtype=float)
                prop_external = get_radial_shift_diagonal(self.basis, rk, r_eval, kind="external")

            open_internal += delta_k[i] * ((-prop_internal)[:, None] * base_step)
            open_external += delta_k[i] * ((-prop_external)[:, None] * base_step)

            if use_rm_boundary:
                s_int_rk_rm = get_radial_shift_diagonal(self.basis, rk, self.RM, kind="internal")
                effective_internal += delta_k[i] * ((factor_vec * prop_internal)[:, None] * base_step)
                effective_external += delta_k[i] * (
                    (factor_vec * (prop_external - S_ext_RM * s_int_rk_rm))[:, None] * base_step
                )
            else:
                effective_internal += delta_k[i] * ((-prop_internal)[:, None] * base_step)
                effective_external += delta_k[i] * ((-prop_external)[:, None] * base_step)

        return PFACRadialSplitOperators(
            open_internal=self._map_closure_operator_to_solution(open_internal, grid),
            open_external=self._map_closure_operator_to_solution(open_external, grid),
            effective_internal=self._map_closure_operator_to_solution(effective_internal, grid),
            effective_external=self._map_closure_operator_to_solution(effective_external, grid),
        )

    def compute_toroidal_rm_closure_operators(self, grid: Grid) -> ToroidalRMClosureOperators:
        """Build operators for the ``R_M`` normal-current closure of dynamic ``alpha``.

        The closure is defined by:
            ``j_n(R_M) = alpha(R_M) * B0r(R_M)``
            ``Delta_S chi(R_M) = j_n(R_M)``
            ``K_div(R_M) = -grad_S chi(R_M)``

        These operators do not include any downward electromagnetic reaction;
        they only describe the current system placed on the ``R_M`` boundary.
        """
        n_sol = int(self.solution_space.index_length)
        n_closure = int(self.basis.index_length)
        n_grid = int(np.asarray(grid.theta).size)

        if self.RM is None:
            return ToroidalRMClosureOperators(
                alpha_to_normal_current_rm_grid=np.zeros((n_grid, n_sol)),
                alpha_to_normal_current_rm_coeff=np.zeros((n_closure, n_sol)),
                alpha_to_closure_potential_rm_coeff=np.zeros((n_closure, n_sol)),
                alpha_to_boundary_psi_rm_coeff=np.zeros((n_closure, n_sol)),
                alpha_to_divergent_closure_current_rm_grid=np.zeros((2, n_grid, n_sol)),
            )

        cache_key = (
            getattr(self.basis, "kind", None),
            int(getattr(self.basis, "index_length", -1)),
            int(getattr(self.basis, "Nmax", -1)),
            int(getattr(self.basis, "Mmax", -1)),
            getattr(self.solution_space, "kind", None),
            int(getattr(self.solution_space, "index_length", -1)),
            int(getattr(self.solution_space, "N", -1)),
            int(getattr(grid, "hash", id(grid))),
            float(self.RI),
            float(self.RM),
            getattr(self.mainfield, "kind", None),
            int(getattr(self.mainfield, "epoch", 0)),
            float(getattr(self.mainfield, "B0", 0.0) or 0.0),
        )
        cached = PFACIntegrator._TOROIDAL_RM_CLOSURE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        theta_ri, phi_ri = self.mainfield.map_coords(self.RI, self.RM, grid.theta, grid.phi)
        footprint_grid = Grid(theta=theta_ri, phi=phi_ri)

        alpha_eval = np.asarray(
            to_dense(self.solution_space.get_evaluation_matrix(footprint_grid))
        )
        if alpha_eval.ndim != 2:
            alpha_eval = alpha_eval.reshape(alpha_eval.shape[0], -1)

        b_rm = self.mainfield.discretize(grid, self.RM)
        br_rm = np.asarray(b_rm.vec.r).reshape(-1)
        if br_rm.size != alpha_eval.shape[0]:
            raise ValueError(
                "RM normal-current assembly mismatch: "
                f"B0r(R_M)={br_rm.shape}, alpha_eval={alpha_eval.shape}."
            )

        alpha_to_normal_current_rm_grid = br_rm[:, None] * alpha_eval

        proj_closure = np.asarray(to_dense(self.basis.construct_scalar_projection_matrix(grid)))
        alpha_to_normal_current_rm_coeff = proj_closure @ alpha_to_normal_current_rm_grid

        lap_rm = np.asarray(to_dense(self.basis.get_laplacian_operator(self.RM)))
        if lap_rm.ndim != 2:
            lap_rm = lap_rm.reshape(lap_rm.shape[0], -1)
        lap_rm_pinv = np.asarray(tensor_pinv(lap_rm, n_leading_flattened=1))
        alpha_to_closure_potential_rm_coeff = lap_rm_pinv @ alpha_to_normal_current_rm_coeff
        alpha_to_boundary_psi_rm_coeff = (
            float(mu0) / float(self.RM)
        ) * alpha_to_closure_potential_rm_coeff

        grad_rm = np.asarray(to_dense(self.basis.get_gradient_matrix(grid)))
        if grad_rm.ndim != 3 or grad_rm.shape[2] != alpha_to_closure_potential_rm_coeff.shape[0]:
            raise ValueError(
                "RM closure gradient assembly mismatch: "
                f"grad={grad_rm.shape}, chi={alpha_to_closure_potential_rm_coeff.shape}."
            )
        alpha_to_divergent_closure_current_rm_grid = -(1.0 / float(self.RM)) * np.tensordot(
            grad_rm, alpha_to_closure_potential_rm_coeff, axes=([2], [0])
        )

        operators = ToroidalRMClosureOperators(
            alpha_to_normal_current_rm_grid=np.asarray(alpha_to_normal_current_rm_grid),
            alpha_to_normal_current_rm_coeff=np.asarray(alpha_to_normal_current_rm_coeff),
            alpha_to_closure_potential_rm_coeff=np.asarray(alpha_to_closure_potential_rm_coeff),
            alpha_to_boundary_psi_rm_coeff=np.asarray(alpha_to_boundary_psi_rm_coeff),
            alpha_to_divergent_closure_current_rm_grid=np.asarray(
                alpha_to_divergent_closure_current_rm_grid
            ),
        )
        PFACIntegrator._TOROIDAL_RM_CLOSURE_CACHE[cache_key] = operators
        return operators

    def _compute_integration_step(
        self,
        rk: float,
        grid: Grid,
        G_inv_sh: np.ndarray,
        m_imp_to_jr_sol_op: np.ndarray,
        n_sol: int,
        n_sh: int,
        *,
        rm_boundary_mode: str = "closed",
    ) -> np.ndarray:
        """Compute a single radial integration step.

        Parameters
        ----------
        rk : float
            Current radial position.
        grid : Grid
            Spatial grid.
        G_inv_sh : np.ndarray
            Inverse of spectral induction operator.
        m_imp_to_jr_sol_op : np.ndarray
            Source operator (Laplacian scaled).
        n_sol : int
            Solution basis size.
        n_sh : int
            Spectral basis size.

        Returns
        -------
        np.ndarray
            Contribution to T_to_Ve from this integration step.
        """
        # Coordinate mapping
        theta_m, phi_m = self.mainfield.map_coords(self.RI, rk, grid.theta, grid.phi)
        m_grid = Grid(theta=theta_m, phi=phi_m)
        rk_b = self.mainfield.discretize(grid, rk)
        m_b = self.mainfield.discretize(m_grid, self.RI)

        # Footprint evaluation of the driving potential (m_imp)
        M_source = to_dense(self.solution_space.get_evaluation_matrix(m_grid))

        # Source term mapping: m_imp_sol -> jr_grid -> JS_grid(rk)
        m_imp_to_JS_rk = np.einsum(
            "ij,jk->ijk",
            [rk_b.vec.theta / m_b.vec.r, rk_b.vec.phi / m_b.vec.r],
            M_source @ m_imp_to_jr_sol_op,
            optimize=True,
        ).reshape(-1, n_sol)

        # Radial propagation factors (exact spectral decay)
        prop_vec = get_radial_shift_diagonal(self.basis, rk, self.RI, kind="external")

        use_rm_boundary = (self.RM is not None) and (str(rm_boundary_mode).lower() == "closed")
        if use_rm_boundary:
            # Reflection and boundaries
            S_ext_RM = get_radial_shift_diagonal(self.basis, self.RM, self.RI, kind="external")
            S_int_rk = get_radial_shift_diagonal(self.basis, rk, self.RM, kind="internal")
            S_int_RI = get_radial_shift_diagonal(self.basis, self.RI, self.RM, kind="internal")
            prop_vec = prop_vec - S_ext_RM * S_int_rk
            factor_vec = -1.0 / (1.0 - S_ext_RM * S_int_RI)
        else:
            factor_vec = -1.0 * np.ones(n_sh)

        # Ve_coeffs += diag(Prop * Factor) @ (G_inv @ JS_grid)
        # Both Prop and Factor live in closure (SH) mode space, i.e. row scaling.
        step_mat = G_inv_sh @ m_imp_to_JS_rk
        step_mat = prop_vec[:, None] * step_mat

        if use_rm_boundary:
            step_mat = step_mat * factor_vec[:, None]
        else:
            step_mat = step_mat * -1.0

        return step_mat
