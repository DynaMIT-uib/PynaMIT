"""Geometry module.

This module contains the Geometry class, which encapsulates spatial
grids, spherical transforms, magnetic field properties, and
interhemispheric mappings.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import numpy as np
import xarray as xr

from pynamit.math.constants import mu0
from pynamit.math import (
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    identity_linear_map,
    pointwise_matrix_linear_map,
    take_linear_map,
)
from pynamit.math.backend import to_numpy
from pynamit.sphere import Grid, SolidHarmonics, SurfaceOperators
from pynamit.sphere.spherical_transform import SphericalTransform, resolve_sqrt_weights
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.math.tensor_operations import weighted_tensor_pinv
from pynamit.simulation.config import setting_value
from pynamit.simulation.mainfield import is_dipole_kind
from pynamit.simulation import sheet_current as sheet_current_ops
from pynamit.simulation.sheet_current import coefficient_scale_values as _coefficient_scale_values
from pynamit.sphere import CSBasis, is_sh_basis

logger = logging.getLogger(__name__)


def _extended_scalar_basis_for_potential(basis, settings):
    """Return the scalar potential basis including the monopole."""
    del settings
    if is_sh_basis(basis):
        return basis.get_extended_basis()
    return basis


class Geometry:
    """Encapsulates the geometric setup for the ionospheric simulation.

    This class manages grids, basis and field evaluators, geometric
    factors derived from the main magnetic field, and interhemispheric
    mappings. It provides a clean interface for the main State class to
    access pre-computed geometric quantities.
    """

    def __init__(
        self,
        basis: SurfaceOperators,
        cs_basis: CSBasis,
        mainfield: Any,
        settings: Any,
        PFAC_matrix: Optional[xr.DataArray] = None,
        solid_harmonics: Optional[SolidHarmonics] = None,
    ) -> None:
        """Initialize the geometric context."""
        if not isinstance(basis, SurfaceOperators):
            raise TypeError("Geometry basis must implement SurfaceOperators.")
        if solid_harmonics is not None and not isinstance(solid_harmonics, SolidHarmonics):
            raise TypeError("solid_harmonics must be a SolidHarmonics object.")
        self.basis = basis
        self.solid_harmonics = solid_harmonics or (
            SolidHarmonics(basis) if is_sh_basis(basis) else None
        )
        self.settings = settings
        self.mainfield = mainfield
        self.cs_basis = cs_basis

        # Store relevant settings
        self.RI = setting_value(settings, "RI")
        rm = setting_value(settings, "RM")
        self.RM = None if rm == 0 else rm
        self.RM_shielding = bool(setting_value(settings, "RM_shielding", False))
        self.connect_hemispheres = bool(setting_value(settings, "connect_hemispheres"))
        self.latitude_boundary = setting_value(settings, "latitude_boundary")
        self.ignore_PFAC = bool(setting_value(settings, "ignore_PFAC"))
        self.FAC_integration_steps = setting_value(settings, "FAC_integration_steps")
        self.area_weighted_least_squares = bool(
            setting_value(settings, "area_weighted_least_squares", False)
        )

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

        self.surface_laplacian_operator = self.basis.get_surface_laplacian_operator(self.RI)
        self._helmholtz_curl_free_potential = None
        self.helmholtz_curl_free_potential_operator = (
            self.basis.get_helmholtz_curl_free_potential_operator()
        )
        self._helmholtz_divergence_free_potential = None
        self.helmholtz_divergence_free_potential_operator = (
            self.basis.get_helmholtz_divergence_free_potential_operator()
        )
        self.m_imp_to_jr_operator = self.RI / mu0 * self.surface_laplacian_operator
        self.m_ind_to_Br_operator = -(self.RI**2) * self.surface_laplacian_operator
        self._m_imp_to_jr = None
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI
        self._m_ind_to_Br = None
        if self.solid_harmonics is None:
            raise NotImplementedError(
                f"{type(self.basis).__name__} requires solid harmonics for sheet-current coupling."
            )

        self._horizontal_solid_projection_is_identity = (
            self.solid_harmonics.basis.coefficients_are_compatible_with(self.basis)
        )
        self.horizontal_to_solid_harmonic_operator = (
            self._build_horizontal_to_solid_harmonic_operator()
        )
        self._horizontal_to_solid_harmonic = None
        self.solid_harmonic_to_horizontal_operator = (
            self._build_solid_harmonic_to_horizontal_operator()
        )
        self._solid_harmonic_to_horizontal = None
        self.poloidal_to_boundary_potential_jump_factor_operator = diagonal_linear_map(
            self.solid_harmonics.poloidal_to_boundary_potential_jump_factor
        )
        self._poloidal_to_boundary_potential_jump_factor = None
        self.horizontal_to_boundary_potential_jump_factor_operator = (
            self.poloidal_to_boundary_potential_jump_factor_operator
            @ self.horizontal_to_solid_harmonic_operator
        )
        self._horizontal_to_boundary_potential_jump_factor = None

        self._helmholtz_analysis_matrix = None
        self._solid_transform_cache = {}

    @property
    def horizontal_solid_projection_is_identity(self) -> bool:
        """Return whether horizontal and solid coefficients match."""
        return self._horizontal_solid_projection_is_identity

    @property
    def horizontal_to_solid_harmonic(self) -> np.ndarray:
        """Return the explicit horizontal-to-solid matrix."""
        if self._horizontal_to_solid_harmonic is None:
            self._horizontal_to_solid_harmonic = np.asarray(
                self.horizontal_to_solid_harmonic_operator.to_matrix(backend="numpy")
            )
        return self._horizontal_to_solid_harmonic

    @property
    def solid_harmonic_to_horizontal(self) -> np.ndarray:
        """Return the explicit solid-to-horizontal matrix."""
        if self._solid_harmonic_to_horizontal is None:
            self._solid_harmonic_to_horizontal = np.asarray(
                self.solid_harmonic_to_horizontal_operator.to_matrix(backend="numpy")
            )
        return self._solid_harmonic_to_horizontal

    @property
    def helmholtz_curl_free_potential(self) -> np.ndarray:
        """Return the curl-free Helmholtz-potential selector."""
        if self._helmholtz_curl_free_potential is None:
            self._helmholtz_curl_free_potential = self.helmholtz_curl_free_potential_operator.array
        return self._helmholtz_curl_free_potential

    @property
    def helmholtz_divergence_free_potential(self) -> np.ndarray:
        """Return the divergence-free Helmholtz-potential selector."""
        if self._helmholtz_divergence_free_potential is None:
            self._helmholtz_divergence_free_potential = (
                self.helmholtz_divergence_free_potential_operator.array
            )
        return self._helmholtz_divergence_free_potential

    @property
    def m_imp_to_jr(self) -> np.ndarray:
        """Return the explicit imposed-potential to jr matrix."""
        if self._m_imp_to_jr is None:
            self._m_imp_to_jr = np.asarray(
                self.m_imp_to_jr_operator.to_matrix(backend="numpy")
            ).copy()
        return self._m_imp_to_jr

    @property
    def m_ind_to_Br(self) -> np.ndarray:
        """Return the explicit induced-potential to Br matrix."""
        if self._m_ind_to_Br is None:
            self._m_ind_to_Br = np.asarray(
                self.m_ind_to_Br_operator.to_matrix(backend="numpy")
            ).copy()
        return self._m_ind_to_Br

    @property
    def poloidal_to_boundary_potential_jump_factor(self) -> np.ndarray:
        """Return the explicit solid-harmonic jump-factor matrix."""
        if self._poloidal_to_boundary_potential_jump_factor is None:
            self._poloidal_to_boundary_potential_jump_factor = np.asarray(
                self.poloidal_to_boundary_potential_jump_factor_operator.to_matrix(backend="numpy")
            ).copy()
        return self._poloidal_to_boundary_potential_jump_factor

    @property
    def horizontal_to_boundary_potential_jump_factor(self) -> np.ndarray:
        """Return the explicit horizontal-to-jump-factor matrix."""
        if self._horizontal_to_boundary_potential_jump_factor is None:
            self._horizontal_to_boundary_potential_jump_factor = np.asarray(
                self.horizontal_to_boundary_potential_jump_factor_operator.to_matrix(
                    backend="numpy"
                )
            ).copy()
        return self._horizontal_to_boundary_potential_jump_factor

    @property
    def jr_coeffs_to_j_apex(self) -> np.ndarray:
        """Return the explicit radial-current to apex-current matrix."""
        if self._jr_coeffs_to_j_apex is None:
            self._jr_coeffs_to_j_apex = np.asarray(
                self.jr_coeffs_to_j_apex_operator.to_matrix(backend="numpy")
            ).copy()
        return self._jr_coeffs_to_j_apex

    def tangential_to_helmholtz(self, vec: np.ndarray) -> np.ndarray:
        """Convert tangential vector field to Helmholtz coeffs."""
        coeffs = np.tensordot(self.helmholtz_analysis_matrix, vec, 2)
        projector = getattr(self.basis, "project_helmholtz_mean_free", None)
        return projector(coeffs) if callable(projector) else coeffs

    @property
    def helmholtz_analysis_matrix(self) -> np.ndarray:
        """Matrix mapping gridded vectors to Helmholtz coefficients."""
        if self._helmholtz_analysis_matrix is None:
            self._helmholtz_analysis_matrix = weighted_tensor_pinv(
                self.spherical_transform.helmholtz_coeffs_to_gridded_vector,
                sqrt_weights=self.grid_sqrt_weights(vector=True),
                n_leading_flattened=2,
            )
        return self._helmholtz_analysis_matrix

    def _init_evaluators(self, cs_basis: CSBasis) -> None:
        """Set up grid, spherical transforms, and field evaluators."""
        self.grid = Grid(
            theta=cs_basis.arr_theta, phi=cs_basis.arr_phi, area_weights=cs_basis.unit_area
        )
        self.spherical_transform = SphericalTransform(
            self.basis, self.grid, area_weighted=self.area_weighted_least_squares
        )
        self.spherical_transform_zero_added = SphericalTransform(
            _extended_scalar_basis_for_potential(self.basis, self.settings),
            self.grid,
            area_weighted=self.area_weighted_least_squares,
        )
        if self.solid_harmonics is None:
            self.solid_harmonic_transform = None
        elif self.solid_harmonics.basis is self.basis:
            self.solid_harmonic_transform = self.spherical_transform
        else:
            self.solid_harmonic_transform = SphericalTransform(
                self.solid_harmonics.basis,
                self.grid,
                area_weighted=self.area_weighted_least_squares,
            )
        self.b_evaluator = FieldEvaluator(self.mainfield, self.grid, self.RI)

        # Optional evaluators for the conjugate hemisphere
        self.cp_grid = self.cp_spherical_transform = self.cp_b_evaluator = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_spherical_transform = SphericalTransform(
                self.basis, self.cp_grid, area_weighted=self.area_weighted_least_squares
            )
            self.cp_b_evaluator = FieldEvaluator(self.mainfield, self.cp_grid, self.RI)

    def grid_sqrt_weights(self, *, vector=False):
        """Return grid sqrt weights when area weighting is enabled."""
        return resolve_sqrt_weights(
            self.grid, area_weighted=self.area_weighted_least_squares, vector=vector
        )

    def _build_horizontal_to_solid_harmonic_operator(self):
        """Project horizontal coefficients into the SH radial space.

        The horizontal basis owns surface operators. ``SolidHarmonics``
        owns the radial laws and wraps the SH basis used for their
        angular coefficients. For the CS horizontal path this is a grid
        least-squares projection from CS nodal values to those SH
        coefficients; for the SH path it is the identity.
        """
        if self._horizontal_solid_projection_is_identity:
            return identity_linear_map((self.basis.index_length,))
        solid_to_grid = self.solid_harmonic_transform.scalar_coeffs_to_grid
        horizontal_to_grid = self.spherical_transform.scalar_coeffs_to_grid
        grid_to_solid = weighted_tensor_pinv(
            solid_to_grid, sqrt_weights=self.grid_sqrt_weights(), n_leading_flattened=1
        )
        return as_linear_map(
            np.asarray(grid_to_solid @ horizontal_to_grid),
            input_shape=(self.basis.index_length,),
            output_shape=(self.solid_harmonics.basis.index_length,),
        )

    def _build_solid_harmonic_to_horizontal_operator(self):
        """Project solid-harmonic coefficients to horizontal space."""
        if self._horizontal_solid_projection_is_identity:
            return identity_linear_map((self.basis.index_length,))
        horizontal_to_grid = self.spherical_transform.scalar_coeffs_to_grid
        solid_to_grid = self.solid_harmonic_transform.scalar_coeffs_to_grid
        grid_to_horizontal = weighted_tensor_pinv(
            horizontal_to_grid, sqrt_weights=self.grid_sqrt_weights(), n_leading_flattened=1
        )
        return as_linear_map(
            np.asarray(grid_to_horizontal @ solid_to_grid),
            input_shape=(self.solid_harmonics.basis.index_length,),
            output_shape=(self.basis.index_length,),
        )

    def _horizontal_to_solid_harmonic_matrix(self):
        """Return an explicit horizontal-to-solid map when needed."""
        if self._horizontal_solid_projection_is_identity:
            return None
        return self.horizontal_to_solid_harmonic

    def solid_transform_for(self, transform: SphericalTransform) -> SphericalTransform:
        """Return a solid transform for ``transform.target``."""
        if self.solid_harmonics.basis.coefficients_are_compatible_with(transform.source):
            return transform
        cache_key = (
            getattr(self.solid_harmonics.basis, "signature", id(self.solid_harmonics.basis)),
            transform.target.signature,
            self.area_weighted_least_squares,
        )
        if cache_key not in self._solid_transform_cache:
            self._solid_transform_cache[cache_key] = SphericalTransform(
                self.solid_harmonics.basis,
                transform.target,
                area_weighted=self.area_weighted_least_squares,
            )
        return self._solid_transform_cache[cache_key]

    def m_ind_to_gridded_sheet_current(
        self,
        transform: Optional[SphericalTransform] = None,
        *,
        solid_transform: Optional[SphericalTransform] = None,
    ) -> np.ndarray:
        """Map induced-potential coefficients to sheet current."""
        if solid_transform is None:
            solid_transform = (
                self.solid_harmonic_transform
                if transform is None
                else self.solid_transform_for(transform)
            )
        return sheet_current_ops.m_ind_to_gridded_sheet_current(
            self.solid_harmonics,
            solid_transform,
            radius=self.RI,
            boundary_radius=self.RM,
            boundary_shielding=self.RM_shielding,
            horizontal_to_solid_harmonic=self._horizontal_to_solid_harmonic_matrix(),
        )

    def m_imp_to_gridded_sheet_current(
        self,
        transform: Optional[SphericalTransform] = None,
        *,
        solid_transform: Optional[SphericalTransform] = None,
    ) -> np.ndarray:
        """Map imposed-potential coefficients to sheet current."""
        transform = self.spherical_transform if transform is None else transform
        if solid_transform is None:
            solid_transform = self.solid_transform_for(transform)
        return sheet_current_ops.m_imp_to_gridded_sheet_current(
            self.solid_harmonics,
            transform,
            solid_transform=solid_transform,
            horizontal_to_solid_harmonic=self._horizontal_to_solid_harmonic_matrix(),
            T_to_Ve=self.T_to_Ve.values,
        )

    def _solid_to_horizontal_coefficients(self, values):
        """Map solid-harmonic coefficient rows to horizontal rows."""
        if self._horizontal_solid_projection_is_identity:
            return values
        return np.tensordot(self.solid_harmonic_to_horizontal, values, axes=([1], [0]))

    def _init_constraint_mappings(self) -> None:
        """Initialize geometric operators related to constraints."""
        kind = self.mainfield.kind
        if is_dipole_kind(kind):
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - 6371e3) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)

        self._jr_coeffs_to_j_apex = None
        self.jr_coeffs_to_j_apex_operator = self._build_jr_coeffs_to_j_apex_operator()
        self._E_coeffs_to_E_apex_ll_diff = None
        self.E_coeffs_to_E_apex_ll_diff_operator = None

        if self.connect_hemispheres:
            # Modify jr constraint for interhemispheric connection
            cp_operator = self._build_jr_coeffs_to_j_apex_operator(
                transform=self.cp_spherical_transform,
                evaluator=self.cp_b_evaluator,
                output_scale=self.ll_mask,
            )
            self.jr_coeffs_to_j_apex_operator = self.jr_coeffs_to_j_apex_operator - cp_operator

            e_to_apex = self._build_E_coeffs_to_E_apex_operator(output_mask=self.ll_mask)
            e_to_apex_cp = self._build_E_coeffs_to_E_apex_operator(
                transform=self.cp_spherical_transform,
                evaluator=self.cp_b_evaluator,
                output_mask=self.ll_mask,
            )
            self.E_coeffs_to_E_apex_ll_diff_operator = e_to_apex - e_to_apex_cp

    def _build_jr_coeffs_to_j_apex_operator(
        self, *, transform=None, evaluator=None, output_scale=None
    ):
        """Return radial-current coefficients mapped to apex current."""
        transform = self.spherical_transform if transform is None else transform
        evaluator = self.b_evaluator if evaluator is None else evaluator
        scale = np.asarray(evaluator.radial_to_apex)
        if output_scale is not None:
            scale = scale * np.asarray(output_scale)
        scale_operator = diagonal_linear_map(
            scale.reshape(-1),
            input_shape=(transform.target.size,),
            output_shape=(transform.target.size,),
        )
        return scale_operator @ transform.scalar_coeffs_to_grid_operator

    def _build_horizontal_grid_to_apex_operator(
        self, *, evaluator=None, grid=None, output_mask=None
    ) -> LinearMap:
        """Return horizontal grid vectors mapped to apex components."""
        evaluator = self.b_evaluator if evaluator is None else evaluator
        grid = self.grid if grid is None else grid
        if output_mask is None:
            indices = np.arange(grid.size)
        else:
            mask = np.asarray(output_mask, dtype=bool).reshape(-1)
            if mask.shape != (grid.size,):
                raise ValueError("output_mask must match the evaluator grid size.")
            indices = np.flatnonzero(mask)

        apex = np.asarray(evaluator.horizontal_to_apex)[:, :, indices]
        n_grid = int(grid.size)
        apex_rotation = pointwise_matrix_linear_map(apex)
        if indices.size == n_grid and np.array_equal(indices, np.arange(n_grid)):
            return apex_rotation

        grid_selection = take_linear_map((2, n_grid), indices, axis=1, dtype=apex.dtype)
        return apex_rotation @ grid_selection

    def _build_E_coeffs_to_E_apex_operator(
        self, *, transform=None, evaluator=None, output_mask=None
    ) -> LinearMap:
        """Return Helmholtz E coefficients mapped to apex components."""
        transform = self.spherical_transform if transform is None else transform
        return (
            self._build_horizontal_grid_to_apex_operator(
                evaluator=evaluator, grid=transform.target, output_mask=output_mask
            )
            @ transform.helmholtz_coeffs_to_gridded_vector_operator
        )

    @property
    def E_coeffs_to_E_apex_ll_diff(self) -> Optional[np.ndarray]:
        """Return explicit low-latitude E-apex difference tensor."""
        operator = self.E_coeffs_to_E_apex_ll_diff_operator
        if operator is None:
            return None
        if self._E_coeffs_to_E_apex_ll_diff is None:
            self._E_coeffs_to_E_apex_ll_diff = to_numpy(operator.array)
        return self._E_coeffs_to_E_apex_ll_diff

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

        solid_poloidal_to_gridded_sheet_current = (
            sheet_current_ops.poloidal_to_gridded_sheet_current(
                self.solid_harmonics, self.solid_harmonic_transform
            )
        )
        sheet_current_rk_to_solid_poloidal_rk = weighted_tensor_pinv(
            solid_poloidal_to_gridded_sheet_current,
            sqrt_weights=self.grid_sqrt_weights(vector=True),
            n_leading_flattened=2,
            rtol=0,
        )
        for i, rk in enumerate(rks):
            logger.debug("PFAC integration step %d/%d (rk=%s)", i + 1, rks.size, rk)
            theta_mapped, phi_mapped = self.mainfield.map_coords(
                self.RI, rk, self.grid.theta, self.grid.phi
            )
            mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
            rk_b_evaluator = FieldEvaluator(self.mainfield, self.grid, rk)
            mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
            mapped_spherical_transform = SphericalTransform(self.basis, mapped_grid)

            m_imp_to_jr_grid = mapped_spherical_transform.contract_scalar_coeffs_to_grid(
                self.m_imp_to_jr_operator
            )
            jr_to_sheet_current_rk = np.array(
                [
                    rk_b_evaluator.Btheta / mapped_b_evaluator.Br,
                    rk_b_evaluator.Bphi / mapped_b_evaluator.Br,
                ]
            )
            m_imp_to_sheet_current_rk = np.einsum(
                "ij,jk->ijk", jr_to_sheet_current_rk, m_imp_to_jr_grid, optimize=True
            )

            regular_poloidal_rk_to_ri = _coefficient_scale_values(
                self.solid_harmonics.regular_reference_shift(rk, self.RI)
            ).reshape((-1, 1, 1))
            if self.RM is not None:
                regular_poloidal_rk_to_ri -= (
                    _coefficient_scale_values(
                        self.solid_harmonics.regular_reference_shift(self.RM, self.RI)
                    )
                    * _coefficient_scale_values(
                        self.solid_harmonics.irregular_reference_shift(rk, self.RM)
                    )
                ).reshape((-1, 1, 1))
                factor = -1.0 / (
                    1.0
                    - _coefficient_scale_values(
                        self.solid_harmonics.regular_reference_shift(self.RM, self.RI)
                    )
                    * _coefficient_scale_values(
                        self.solid_harmonics.irregular_reference_shift(self.RI, self.RM)
                    )
                )
            else:
                factor = -1.0

            sheet_current_rk_to_solid_poloidal = (
                sheet_current_rk_to_solid_poloidal_rk * regular_poloidal_rk_to_ri
            )
            if np.ndim(factor) == 0:
                sheet_current_rk_to_solid_poloidal *= factor
            else:
                sheet_current_rk_to_solid_poloidal *= np.asarray(factor).reshape((-1, 1, 1))
            sheet_current_rk_to_horizontal_poloidal = self._solid_to_horizontal_coefficients(
                sheet_current_rk_to_solid_poloidal
            )
            self._T_to_Ve += Delta_k[i] * np.tensordot(
                sheet_current_rk_to_horizontal_poloidal, m_imp_to_sheet_current_rk, axes=2
            )

    def Br_to_gridded_sheet_current(
        self,
        transform: Optional[SphericalTransform] = None,
        *,
        solid_transform: Optional[SphericalTransform] = None,
    ) -> Optional[np.ndarray]:
        """Map boundary-Br coefficients to sheet current."""
        if self.RM is None:
            return None
        if solid_transform is None:
            solid_transform = (
                self.solid_harmonic_transform
                if transform is None
                else self.solid_transform_for(transform)
            )
        return sheet_current_ops.Br_to_gridded_sheet_current(
            self.solid_harmonics,
            solid_transform,
            radius=self.RI,
            boundary_radius=self.RM,
            horizontal_to_solid_harmonic=self._horizontal_to_solid_harmonic_matrix(),
        )
