"""PFAC (Poloidal Field-Aligned Current) integration module.

This module handles the computation of magnetospheric coupling through
radial integration of field-aligned currents. The T_to_Ve operator maps
external toroidal potential to poloidal potential.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import xarray as xr

from pynamit.math.constants import mu0
from pynamit.primitives.grid import Grid
from pynamit.utils import tensor_pinv
from pynamit.simulation.spatial.geometry_utils import to_dense, get_radial_shift_diagonal

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis


logger = logging.getLogger(__name__)


class PFACIntegrator:
    """Handles PFAC (Poloidal Field-Aligned Current) computation.

    This class encapsulates the radial integration needed to compute
    the T_to_Ve operator, which maps external toroidal potential to
    poloidal potential at the ionospheric boundary.

    Parameters
    ----------
    basis : Basis
        The spectral (SH) basis for physics computations.
    solution_basis : Basis
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
    magnetospheric_poloidal_lock : bool
        If True, include RM shielding-style coupling between imposed Br and
        induced m_ind pathways. If False, keep those pathways superposed but uncoupled.
    lock_toroidal_source_channels : bool
        If True, apply the RM poloidal lock to toroidal-source channels
        (m_imp/psi PFAC branch) in addition to m_ind/Br pathways.
    """
    _T_TO_VE_CACHE: dict[tuple[Any, ...], np.ndarray] = {}

    def __init__(
        self,
        basis: "Basis",
        solution_basis: "Basis",
        mainfield: Any,
        RI: float,
        RM: Optional[float],
        FAC_integration_steps: Any,
        ignore_PFAC: bool = False,
        magnetospheric_poloidal_lock: bool = True,
        lock_toroidal_source_channels: bool = False,
    ) -> None:
        self.basis = basis
        self.solution_basis = solution_basis
        self.mainfield = mainfield
        self.RI = RI
        self.RM = RM
        self.FAC_integration_steps = FAC_integration_steps
        self.ignore_PFAC = ignore_PFAC
        self.magnetospheric_poloidal_lock = bool(magnetospheric_poloidal_lock)
        self.lock_toroidal_source_channels = bool(lock_toroidal_source_channels)

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

    def compute_T_to_Ve(
        self,
        G_Ve_to_JS_closure: np.ndarray,
        grid: Grid,
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
        n_sol = self.solution_basis.index_length
        n_sh = self.basis.index_length
        T_to_Ve = xr.DataArray(np.zeros((n_sol, n_sol)), dims=("current_pot", "field_pot"))

        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return T_to_Ve

        fac_steps = np.asarray(self.FAC_integration_steps, dtype=float)
        fac_steps_key = tuple(np.round(fac_steps, 6).tolist())
        cache_key = (
            getattr(self.basis, "kind", None),
            int(getattr(self.basis, "index_length", -1)),
            int(getattr(self.basis, "Nmax", -1)),
            int(getattr(self.basis, "Mmax", -1)),
            getattr(self.solution_basis, "kind", None),
            int(getattr(self.solution_basis, "index_length", -1)),
            int(getattr(self.solution_basis, "N", -1)),
            int(getattr(grid, "hash", id(grid))),
            float(self.RI),
            float(self.RM) if self.RM is not None else None,
            getattr(self.mainfield, "kind", None),
            int(getattr(self.mainfield, "epoch", 0)),
            float(getattr(self.mainfield, "B0", 0.0) or 0.0),
            fac_steps_key,
        )
        cached = PFACIntegrator._T_TO_VE_CACHE.get(cache_key)
        if cached is not None:
            return xr.DataArray(cached.copy(), dims=("current_pot", "field_pot"))

        # Radial steps
        rk_steps = fac_steps
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        # Integration backbone (spectral space)
        G_inv_sh = tensor_pinv(G_Ve_to_JS_closure, n_leading_flattened=2, rtol=1e-12).reshape(n_sh, -1)

        # Source operator factor in solution-basis coefficients: RI/mu0 * Laplacian
        # (built once and reused for all radial quadrature steps).
        L_sol = to_dense(self.solution_basis.get_laplacian_operator(self.RI))
        m_imp_to_jr_sol_op = (self.RI / mu0) * L_sol
        if getattr(self.solution_basis, "kind", "") in ("CS", "GRID"):
            if hasattr(self.solution_basis, "get_mean_zero_projector"):
                try:
                    P_sol = np.asarray(
                        self.solution_basis.get_mean_zero_projector(n_coeff=n_sol),
                        dtype=float,
                    )
                    if P_sol.shape == (n_sol, n_sol):
                        m_imp_to_jr_sol_op = m_imp_to_jr_sol_op @ P_sol
                except Exception:
                    pass

        # Is this a pure spectral simulation?
        is_pure_sh = (self.solution_basis is self.basis or self.solution_basis.kind == "SH")

        # Accumulator (spectral result: Ve_sh_coeffs / m_imp_coeffs)
        T_accum = np.zeros((n_sh, n_sol))

        for i, rk in enumerate(rks):
            step_mat = self._compute_integration_step(
                rk, grid, G_inv_sh, m_imp_to_jr_sol_op, n_sol, n_sh
            )
            T_accum += Delta_k[i] * step_mat

        # Map back to result basis
        if is_pure_sh:
            T_to_Ve.values = T_accum
        else:
            # Map integrated SH coefficients back to solver coefficients (hybrid)
            E_sh = to_dense(self.basis.get_evaluation_matrix(grid))
            E_sol = to_dense(self.solution_basis.get_evaluation_matrix(grid))
            P_sol = tensor_pinv(E_sol, rtol=1e-12)
            T_to_Ve.values = (P_sol @ E_sh) @ T_accum

        if getattr(self.solution_basis, "kind", "") in ("CS", "GRID"):
            if hasattr(self.solution_basis, "get_mean_zero_projector"):
                try:
                    P_g = np.asarray(
                        self.solution_basis.get_mean_zero_projector(n_coeff=n_sol),
                        dtype=float,
                    )
                    if P_g.shape == (n_sol, n_sol):
                        T_to_Ve.values = P_g @ T_to_Ve.values
                except Exception:
                    pass

        PFACIntegrator._T_TO_VE_CACHE[cache_key] = T_to_Ve.values.copy()
        return T_to_Ve

    def _compute_integration_step(
        self,
        rk: float,
        grid: Grid,
        G_inv_sh: np.ndarray,
        m_imp_to_jr_sol_op: np.ndarray,
        n_sol: int,
        n_sh: int,
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
        theta_m, phi_m = self.mainfield.map_coords(
            self.RI, rk, grid.theta, grid.phi
        )
        m_grid = Grid(theta=theta_m, phi=phi_m)
        rk_b = self.mainfield.discretize(grid, rk)
        m_b = self.mainfield.discretize(m_grid, self.RI)

        # Footprint evaluation of the driving potential (m_imp)
        M_source = to_dense(self.solution_basis.get_evaluation_matrix(m_grid))

        # Source term mapping: m_imp_sol -> jr_grid -> JS_grid(rk)
        m_imp_to_JS_rk = np.einsum(
            "ij,jk->ijk",
            [rk_b.vec.theta / m_b.vec.r, rk_b.vec.phi / m_b.vec.r],
            M_source @ m_imp_to_jr_sol_op,
            optimize=True
        ).reshape(-1, n_sol)

        # Radial propagation factors (exact spectral decay)
        prop_vec = get_radial_shift_diagonal(self.basis, rk, self.RI, kind="external")

        if self.RM is not None:
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

        if self.RM is not None:
            step_mat = step_mat * factor_vec[:, None]
        else:
            step_mat = step_mat * -1.0

        return step_mat
