"""Geometry module.

This module contains the Geometry class, which encapsulates the spatial
grid, basis evaluators, magnetic field properties, and interhemispheric
mappings.
"""

from __future__ import annotations
import logging
from typing import Optional, Any

import numpy as np
import xarray as xr

from pynamit.math.constants import mu0
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.math.tensor_operations import tensor_pinv
from pynamit.primitives.basis import is_grid_basis
from pynamit.spherical_harmonics.sh_basis import SHBasis

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
        basis: SHBasis,
        cs_basis: SHBasis,
        mainfield: Any,
        settings: Any,
        PFAC_matrix: Optional[xr.DataArray] = None,
    ) -> None:
        """Initialize the geometric context."""
        self.basis = basis
        self.settings = settings
        self.mainfield = mainfield

        # Store relevant settings
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.FAC_integration_steps = settings.FAC_integration_steps

        # Initialize core geometric objects
        self._init_evaluators(cs_basis)
        self._init_constraint_mappings()

        # Caches for expensive properties
        self._bP: Optional[np.ndarray] = None
        self._bH: Optional[np.ndarray] = None
        self._bu: Optional[np.ndarray] = None

        # Allow pre-computed PFAC matrix
        if PFAC_matrix is not None:
            self._T_to_Ve = PFAC_matrix
        else:
            self._T_to_Ve: Optional[xr.DataArray] = None

        self._G_m_ind_to_JS = None
        self._G_m_imp_to_JS = None

        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        if np.ndim(Ve_to_J_df_coeffs) == 1:
            self.G_Ve_to_JS = (
                (1.0 / self.RI) * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs
            )
        else:
            self.G_Ve_to_JS = (1.0 / self.RI) * np.tensordot(
                self.basis_evaluator.G_rxgrad, Ve_to_J_df_coeffs, axes=([2], [0])
            )

        self._G_helmholtz_pinv = None

    def tangential_to_helmholtz(self, vec: np.ndarray) -> np.ndarray:
        """Convert tangential vector field to Helmholtz coeffs."""
        return np.tensordot(self.G_helmholtz_pinv, vec, 2)

    @property
    def G_helmholtz_pinv(self) -> np.ndarray:
        """Pseudo-inverse for horizontal vector field projections."""
        if self._G_helmholtz_pinv is None:
            self._G_helmholtz_pinv = tensor_pinv(
                self.basis_evaluator.G_helmholtz, n_leading_flattened=2
            )
        return self._G_helmholtz_pinv

    def _init_evaluators(self, cs_basis: SHBasis) -> None:
        """Set up grid, basis evaluators, and field evaluators."""
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.basis_evaluator_zero_added = BasisEvaluator(
            SHBasis(self.settings.Nmax, self.settings.Mmax, Nmin=0), self.grid
        )
        self.b_evaluator = FieldEvaluator(self.mainfield, self.grid, self.RI)

        # Optional evaluators for the conjugate hemisphere
        self.cp_grid = self.cp_basis_evaluator = self.cp_b_evaluator = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(self.mainfield, self.cp_grid, self.RI)

    def _init_constraint_mappings(self) -> None:
        """Initialize geometric operators related to constraints."""
        kind = self.mainfield.kind
        if kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - 6371e3) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)

        self.jr_coeffs_to_j_apex = (
            self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        ).copy()
        self.E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres:
            # Modify jr constraint for interhemispheric connection
            jr_coeffs_to_j_apex_cp = (
                self.cp_b_evaluator.radial_to_apex.reshape((-1, 1)) * self.cp_basis_evaluator.G
            )
            self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]

            # Create E-field mapping difference operator for constraint
            E_coeffs_to_E_apex = np.einsum(
                "ijk,jklm->iklm",
                self.b_evaluator.horizontal_to_apex,
                self.basis_evaluator.G_helmholtz,
                optimize=True,
            )
            E_coeffs_to_E_apex_cp = np.einsum(
                "ijk,jklm->iklm",
                self.cp_b_evaluator.horizontal_to_apex,
                self.cp_basis_evaluator.G_helmholtz,
                optimize=True,
            )
            self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask]
            )

    @property
    def bP(self) -> np.ndarray:
        """Pedersen geometric factor for conductance tensor."""
        if self._bP is None:
            b_th, b_ph, b_r = self.b_evaluator.btheta, self.b_evaluator.bphi, self.b_evaluator.br
            self._bP = np.array(
                [[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]]
            )
        return self._bP

    @property
    def bH(self) -> np.ndarray:
        """Hall geometric factor for conductance tensor."""
        if self._bH is None:
            br = self.b_evaluator.br
            self._bH = np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])
        return self._bH

    @property
    def bu(self) -> np.ndarray:
        """Geometric factor for u x B electric field."""
        if self._bu is None:
            Br = self.b_evaluator.Br
            self._bu = -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])
        return self._bu

    @property
    def T_to_Ve(self) -> xr.DataArray:
        """Mapping external toroidal (T) to poloidal (Ve) potential."""
        if self._T_to_Ve is None:
            self._build_T_to_Ve()
        return self._T_to_Ve

    def _build_T_to_Ve(self) -> None:
        """Construct the T_to_Ve operator by integrating radially."""
        n = self.basis.index_length
        self._T_to_Ve = xr.DataArray(np.zeros((n, n)), dims=("i", "j"))
        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return
        if is_grid_basis(self.basis):
            raise NotImplementedError(
                "PFAC integration with CS calculation basis is not implemented in this branch."
            )

        rk_steps = np.asarray(self.FAC_integration_steps)
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        if np.any(rks < self.RI):
            raise ValueError(
                "All FAC integration steps must be outside the ionospheric boundary (RI)."
            )
        if self.RM is not None and np.any(rks > self.RM):
            raise ValueError(
                "All FAC integration steps must be inside the magnetospheric boundary (RM)."
            )

        JS_rk_to_Ve_rk = tensor_pinv(self.G_Ve_to_JS, n_leading_flattened=2, rtol=0)
        m_imp_to_jr_coeffs = self.RI / mu0 * self.basis.laplacian(self.RI)

        for i, rk in enumerate(rks):
            logger.debug("PFAC integration step %d/%d (rk=%s)", i + 1, rks.size, rk)
            theta_mapped, phi_mapped = self.mainfield.map_coords(
                self.RI, rk, self.grid.theta, self.grid.phi
            )
            mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
            rk_b_evaluator = FieldEvaluator(self.mainfield, self.grid, rk)
            mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
            mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)

            m_imp_to_jr_grid = mapped_basis_evaluator.scaled_G(m_imp_to_jr_coeffs)
            jr_to_JS_rk = np.array(
                [
                    rk_b_evaluator.Btheta / mapped_b_evaluator.Br,
                    rk_b_evaluator.Bphi / mapped_b_evaluator.Br,
                ]
            )
            m_imp_to_JS_rk = np.einsum("ij,jk->ijk", jr_to_JS_rk, m_imp_to_jr_grid, optimize=True)

            Ve_rk_to_Ve = self.basis.radial_shift_Ve(rk, self.RI).reshape((-1, 1, 1))
            if self.RM is not None:
                Ve_rk_to_Ve -= (
                    self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(rk, self.RM)
                ).reshape((-1, 1, 1))
                factor = -1.0 / (
                    1.0
                    - self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(self.RI, self.RM)
                )
            else:
                factor = -1.0

            JS_rk_to_Ve = JS_rk_to_Ve_rk * Ve_rk_to_Ve
            self._T_to_Ve += (
                Delta_k[i] * factor * np.tensordot(JS_rk_to_Ve, m_imp_to_JS_rk, axes=2)
            )

    # ----- G operators mapping to sheet current (JS) -----

    @property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        if self._G_m_imp_to_JS is None:
            G_T_to_JS = -1.0 / self.RI * self.basis_evaluator.G_grad * (self.RI / mu0)
            self._G_m_imp_to_JS = G_T_to_JS + np.tensordot(
                self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0])
            )
        return self._G_m_imp_to_JS

    @property
    def G_m_ind_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        if self._G_m_ind_to_JS is None:
            G = self.G_Ve_to_JS.copy()
            if self.RM is not None:
                if is_grid_basis(self.basis):
                    raise NotImplementedError(
                        "RM boundary coupling with CS calculation basis is not implemented."
                    )
                br_shift = self.basis.radial_shift_Ve(self.RM, self.RI)
                vi_shift = self.basis.radial_shift_Vi(self.RI, self.RM)
                den = 1.0 - br_shift * vi_shift
                self.G_Br_to_JS = self.G_Ve_to_JS * (-br_shift / den / self.m_ind_to_Br)
                G *= 1.0 + (br_shift * vi_shift / den)
            self._G_m_ind_to_JS = G
        return self._G_m_ind_to_JS
