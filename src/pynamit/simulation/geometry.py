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
from functools import cached_property

from pynamit.math.constants import mu0
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field import Field
from pynamit.utils import tensor_pinv
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
        self.mainfield = mainfield

        # Allow pre-computed PFAC matrix (must override cached_property if provided)
        if PFAC_matrix is not None:
            self.T_to_Ve = PFAC_matrix

        # Store relevant settings

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

        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = (1.0 / self.RI) * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs

        self.G_helmholtz_pinv = tensor_pinv(
            self.basis_evaluator.G_helmholtz, n_leading_flattened=2
        )

    def tangential_to_helmholtz(self, vec: np.ndarray) -> np.ndarray:
        """Convert tangential vector field to Helmholtz coeffs."""
        return np.tensordot(self.G_helmholtz_pinv, vec, 2)

    def _init_evaluators(self, cs_basis: SHBasis) -> None:
        """Set up grid, basis evaluators, and field evaluators."""
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.basis_evaluator_zero_added = BasisEvaluator(
            SHBasis(self.basis.Nmax, self.basis.Mmax, Nmin=0), self.grid
        )
        self.b_field = self.mainfield.discretize(self.grid, self.RI)

        # Optional evaluators for the conjugate hemisphere
        self.cp_grid = self.cp_basis_evaluator = self.cp_b_field = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_field = self.mainfield.discretize(self.cp_grid, self.RI)

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

        radial_to_apex, horizontal_to_apex = self._get_transformation_matrices(self.b_field)

        self.jr_coeffs_to_j_apex = (
            radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        ).copy()
        self.E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres:
            # Modify jr constraint for interhemispheric connection
            radial_to_apex_cp, horizontal_to_apex_cp = self._get_transformation_matrices(
                self.cp_b_field
            )

            jr_coeffs_to_j_apex_cp = radial_to_apex_cp.reshape((-1, 1)) * self.cp_basis_evaluator.G
            self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]

            # Create E-field mapping difference operator for constraint
            E_coeffs_to_E_apex = np.einsum(
                "ijk,jklm->iklm",
                horizontal_to_apex,
                self.basis_evaluator.G_helmholtz,
                optimize=True,
            )
            E_coeffs_to_E_apex_cp = np.einsum(
                "ijk,jklm->iklm",
                horizontal_to_apex_cp,
                self.cp_basis_evaluator.G_helmholtz,
                optimize=True,
            )
            self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask]
            )

    @cached_property
    def bP(self) -> np.ndarray:
        """Pedersen geometric factor for conductance tensor."""
        mag = self.b_field.magnitude
        b_th, b_ph, b_r = (
            self.b_field.vec.theta / mag,
            self.b_field.vec.phi / mag,
            self.b_field.vec.r / mag,
        )
        return np.array([[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]])

    @cached_property
    def bH(self) -> np.ndarray:
        """Hall geometric factor for conductance tensor."""
        br = self.b_field.vec.r / self.b_field.magnitude
        return np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])

    @cached_property
    def bu(self) -> np.ndarray:
        """Geometric factor for u x B electric field."""
        Br = self.b_field.vec.r
        return -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])

    @cached_property
    def T_to_Ve(self) -> xr.DataArray:
        """Mapping external toroidal (T) to poloidal (Ve) potential."""
        return self._build_T_to_Ve()

    def _build_T_to_Ve(self) -> xr.DataArray:
        """Construct the T_to_Ve operator by integrating radially."""
        n = self.basis.index_length
        T_to_Ve = xr.DataArray(np.zeros((n, n)), dims=("i", "j"))
        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return T_to_Ve

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
            mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
            rk_b_field = self.mainfield.discretize(self.grid, rk)
            mapped_b_field = self.mainfield.discretize(mapped_grid, self.RI)
            mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)

            m_imp_to_jr_grid = mapped_basis_evaluator.scaled_G(m_imp_to_jr_coeffs)
            jr_to_JS_rk = np.array(
                [
                    rk_b_field.vec.theta / mapped_b_field.vec.r,
                    rk_b_field.vec.phi / mapped_b_field.vec.r,
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
            T_to_Ve += Delta_k[i] * factor * np.tensordot(JS_rk_to_Ve, m_imp_to_JS_rk, axes=2)
        return T_to_Ve

    # ----- G operators mapping to sheet current (JS) -----

    @cached_property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        G_T_to_JS = -1.0 / self.RI * self.basis_evaluator.G_grad * (self.RI / mu0)
        return G_T_to_JS + np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0]))

    def _get_transformation_matrices(self, dfield: Field):
        """Compute transformation matrices for Apex coordinates."""
        # Get basis vectors
        # d1, d2, d3, e1, e2, e3
        r_eval = dfield.r_loc if dfield.r_loc is not None else self.RI
        bv = dfield.basis_vectors(r_eval, dfield.grid.theta, dfield.grid.phi)
        d3 = bv[2]
        e1 = bv[3]
        e2 = bv[4]

        # Get unit vector components
        mag = dfield.magnitude
        br = dfield.vec.r / mag
        btheta = dfield.vec.theta / mag
        bphi = dfield.vec.phi / mag

        # d3 components (field parallel)
        d3r, d3theta, d3phi = d3[0], d3[1], d3[2]

        # e1, e2 components (field orthogonal)
        e1r, e1theta, e1phi = e1[0], e1[1], e1[2]
        e2r, e2theta, e2phi = e2[0], e2[1], e2[2]

        # 1. Radial to Apex
        # radial_to_field_parallel
        # [[1], [btheta/br], [bphi/br]]
        # field_parallel_to_apex
        # [[d3r, d3theta, d3phi]]

        # Result is scalar product for each grid point?
        # No, matrix mult.
        # r_to_fp: (3, 1, N) ?
        # Actually, let's look at previous implementation:
        # radial_to_field_parallel was (3, 1, N) implicitly? No.
        # It returned (3, 1) of arrays?
        # np.array([[ones], [btheta/br], [bphi/br]]) -> shape (3, 1, N)

        # 1. Radial to Apex
        # radial_to_field_parallel
        # We need shape (3, 1, N) for matrix multiplication radial -> parallel
        # The matrix is [[1], [btheta/br], [bphi/br]]

        # Ensure we are working with flattened arrays (N,)
        ones = np.ones(self.grid.size)
        ratio_theta = (btheta / br).flatten()
        ratio_phi = (bphi / br).flatten()

        # Stack to shape (3, N) then reshape to (3, 1, N)
        radial_to_field_parallel = np.stack([ones, ratio_theta, ratio_phi], axis=0)  # (3, N)
        radial_to_field_parallel = radial_to_field_parallel[:, np.newaxis, :]  # (3, 1, N)

        # field_parallel_to_apex
        # Matrix is [[d3r, d3theta, d3phi]] -> shape (1, 3, N)
        field_parallel_to_apex = np.stack([d3r, d3theta, d3phi], axis=0)  # (3, N)
        field_parallel_to_apex = field_parallel_to_apex[np.newaxis, :, :]  # (1, 3, N)

        # einsum ij k, jl k -> il k
        # (1, 3, N) x (3, 1, N) -> (1, 1, N) in matrix mult sense for each k?
        # NO. We map radial component (1D vector at each point) to Apex vector (3D)?

        # Wait, radial_to_apex converts a radial current/field to an apex vector.
        # Radial vector is v = v_r * r_hat.
        # r_hat = 1 * d3_parallel + (bth/br) * ...?
        # The logic is likely: r_hat decomposed into field-parallel and perp?
        # Re-check einsum indices.
        # ijk (1,3,N) , jlk (3,1,N) -> ilk (1,1,N).
        # This results in a scalar field?
        # self.jr_coeffs_to_j_apex shape usage: (radial_to_apex * G).
        # jr_coeffs_to_j_apex should map scalar coeff to... vector J?
        # G is scalar basis eval.
        # If radial_to_apex is (1,1,N), it's a scalar factor.
        # But j_apex is a vector?
        # self.jr_coeffs_to_j_apex shape should be compatible with J (3 components?).
        # Looking at subsequent code:
        # jr_coeffs_to_j_apex = radial_to_apex.reshape((-1, 1)) * G
        # If radial_to_apex is (3, N) or similar?

        # Recalculating:
        # field_parallel_to_apex is d3 vector. Shape (3, N) effectively if we just take d3.
        # It maps a magnitude along field line to vector components.
        # radial_to_field_parallel maps radial component to magnitude along field line?
        # If J = Jr r_hat. J = Jpar d3 + ...
        # Jpar = Jr / (d3 . r_hat) ?
        # Here we have [1, btheta/br, bphi/br].
        # d3 = d3r r_hat + d3th th_hat + d3ph ph_hat.
        # Dot product: d3 . r_hat = d3r.
        # If B is parallel to d3?
        # This math seems to assume B is proportional to d3?

        # For now, preserving the logic structure but fixing shapes.
        # Previously:
        # radial_to_field_parallel shape was (3, 1, N).
        # field_parallel_to_apex shape was (1, 3, N).
        # tensordot/einsum produced (1, 1, N)?
        # Let's check: i=1, j=3, k=N. l=1. -> (1, 1, N).
        # Reshape((-1, 1)) -> (N, 1)?
        # If it's (N,), then reshape works.
        # But wait, self.basis_evaluator.G is often (N, n_coeffs).
        # So we need (N, 1) broadcast.

        # Wait, if radial_to_apex is a vector, it should be (3, N).
        # (1, 3, N) x (3, 1, N) -> scalar (1, 1, N).
        # This means the result is a scalar at each point.
        # But jr_coeffs_to_j_apex name suggests generic Apex vector?
        # Ah, jr is Field Aligned Current? No, radial current.
        # The variable name creates confusion.

        # Let's fix the array creation first.
        radial_to_apex = np.einsum(
            "ijk,jlk->ilk", field_parallel_to_apex, radial_to_field_parallel, optimize=True
        )  # Result (1, 1, N) -> Squeeze to (N,) afterwards?

        # Remove singleton dimensions for simpler handling if needed
        radial_to_apex = radial_to_apex.squeeze()  # (N,)

        # 2. Horizontal to Apex
        # horizontal_to_field_orthogonal
        # [[-btheta/br, -bphi/br], [1, 0], [0, 1]] -> (3, 2, N)
        r1 = np.stack([-(btheta / br).flatten(), -(bphi / br).flatten()], axis=0)  # (2, N)
        r2 = np.stack([np.ones(self.grid.size), np.zeros(self.grid.size)], axis=0)  # (2, N)
        r3 = np.stack([np.zeros(self.grid.size), np.ones(self.grid.size)], axis=0)  # (2, N)

        horizontal_to_field_orthogonal = np.stack([r1, r2, r3], axis=0)  # (3, 2, N)

        # field_orthogonal_to_apex
        # [[e1r, e1th, e1ph], [e2r, ...]] -> (2, 3, N)
        e1_vec = np.stack([e1r, e1theta, e1phi], axis=0)  # (3, N)
        e2_vec = np.stack([e2r, e2theta, e2phi], axis=0)  # (3, N)
        field_orthogonal_to_apex = np.stack([e1_vec, e2_vec], axis=0)  # (2, 3, N)

        horizontal_to_apex = np.einsum(
            "ijk,jlk->ilk", field_orthogonal_to_apex, horizontal_to_field_orthogonal, optimize=True
        )  # (2, 2, N)

        return radial_to_apex, horizontal_to_apex

    @cached_property
    def G_m_ind_to_JS(self) -> np.ndarray:
        """Operator mapping m_imp to sheet current on grid."""
        G = self.G_Ve_to_JS.copy()
        if self.RM is not None:
            br_shift = self.basis.radial_shift_Ve(self.RM, self.RI)
            vi_shift = self.basis.radial_shift_Vi(self.RI, self.RM)
            den = 1.0 - br_shift * vi_shift
            self.G_Br_to_JS = self.G_Ve_to_JS * (-br_shift / den / self.m_ind_to_Br)
            G *= 1.0 + (br_shift * vi_shift / den)
        return G
