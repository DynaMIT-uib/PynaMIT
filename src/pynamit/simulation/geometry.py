"""Run-invariant spatial and magnetic context for simulations."""

from __future__ import annotations
from functools import cached_property
import logging
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from pynamit.math.constants import RE, mu0
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
from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.math.tensor_operations import weighted_tensor_pinv
from pynamit.simulation.config import SimulationConfig
from pynamit.geomagnetism import MainField
from pynamit.simulation.electrodynamics import ionospheric_closure, magnetic_boundary
from pynamit.sphere import CSBasis, is_sh_basis

logger = logging.getLogger(__name__)


def build_main_field(config: SimulationConfig) -> MainField:
    """Build the background field selected by a simulation config."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("build_main_field requires a SimulationConfig.")
    return MainField(
        kind=config.main_field_kind,
        epoch=config.main_field_epoch,
        ionosphere_height_km=(config.RI - RE) * 1e-3,
        B0=config.main_field_B0,
    )


class SimulationGeometry:
    """Run-invariant spatial context for one ionospheric simulation.

    The geometry owns grids, transforms, background-field factors,
    boundary maps, and interhemispheric mappings. It contains no mutable
    forcing coefficients or persistence-shaped objects.
    """

    def __init__(
        self,
        horizontal_basis: SurfaceOperators,
        cs_basis: CSBasis,
        main_field: MainField,
        config: SimulationConfig,
        pfac_matrix: Optional[ArrayLike] = None,
        solid_harmonics: Optional[SolidHarmonics] = None,
    ) -> None:
        """Initialize the geometric context."""
        if not isinstance(horizontal_basis, SurfaceOperators):
            raise TypeError("SimulationGeometry horizontal_basis must implement SurfaceOperators.")
        if not isinstance(config, SimulationConfig):
            raise TypeError("SimulationGeometry requires a validated SimulationConfig.")
        if solid_harmonics is not None and not isinstance(solid_harmonics, SolidHarmonics):
            raise TypeError("solid_harmonics must be a SolidHarmonics object.")
        self.horizontal_basis = horizontal_basis
        self.solid_harmonics = (
            solid_harmonics
            if solid_harmonics is not None
            else (SolidHarmonics(horizontal_basis) if is_sh_basis(horizontal_basis) else None)
        )
        if self.solid_harmonics is None:
            raise NotImplementedError(
                f"{type(self.horizontal_basis).__name__} requires solid harmonics for JS coupling."
            )
        self.main_field = main_field

        # Store the configuration values used by geometric construction.
        self.RI = config.RI
        self.RM = config.RM
        self.magnetic_boundary_shielding = config.magnetic_boundary_shielding
        self.enable_interhemispheric_coupling = config.enable_interhemispheric_coupling
        self.interhemispheric_coupling_latitude = config.interhemispheric_coupling_latitude
        self.enable_pfac_coupling = config.enable_pfac_coupling
        self.fac_integration_radii = config.fac_integration_radii
        self.area_weighted_least_squares = config.area_weighted_least_squares

        # Initialize the model grid and magnetic-field evaluators.
        self._init_spatial_context(cs_basis)

        # Build surface and magnetic-boundary maps.
        self._init_surface_operators()

        # Build optional interhemispheric constraint geometry.
        self._init_constraint_mappings()

        # Restore a persisted PFAC map or build it on first access.
        self._init_pfac_matrix(pfac_matrix)
        self._solid_transform_cache = {}

    def _init_surface_operators(self) -> None:
        """Compile surface and magnetic-boundary coefficient maps."""
        self.surface_laplacian_operator = self.horizontal_basis.get_surface_laplacian_operator(
            self.RI
        )
        self.helmholtz_curl_free_potential_operator = (
            self.horizontal_basis.get_helmholtz_curl_free_potential_operator()
        )
        self.helmholtz_divergence_free_potential_operator = (
            self.horizontal_basis.get_helmholtz_divergence_free_potential_operator()
        )
        self.m_imp_to_jr_operator = self.RI / mu0 * self.surface_laplacian_operator
        self.m_ind_to_Br_operator = -(self.RI**2) * self.surface_laplacian_operator
        self.faraday_rate_scale = 1.0 / self.RI
        self.horizontal_solid_projection_is_identity = (
            self.solid_harmonics.basis.coefficients_are_compatible_with(self.horizontal_basis)
        )
        self.horizontal_to_solid_harmonic_operator = (
            self._build_horizontal_to_solid_harmonic_operator()
        )
        self.solid_harmonic_to_horizontal_operator = (
            self._build_solid_harmonic_to_horizontal_operator()
        )
        self.poloidal_to_boundary_potential_jump_factor_operator = diagonal_linear_map(
            self.solid_harmonics.poloidal_to_boundary_potential_jump_factor
        )
        self.horizontal_to_boundary_potential_jump_factor_operator = (
            self.poloidal_to_boundary_potential_jump_factor_operator
            @ self.horizontal_to_solid_harmonic_operator
        )

    def _init_pfac_matrix(self, pfac_matrix: Optional[ArrayLike]) -> None:
        """Validate and retain an optional persisted PFAC matrix."""
        self._pfac_matrix = None
        if pfac_matrix is None:
            return
        expected_shape = (self.horizontal_basis.index_length, self.horizontal_basis.index_length)
        matrix = np.asarray(pfac_matrix)
        if matrix.shape != expected_shape:
            raise ValueError(f"pfac_matrix must have shape {expected_shape}; got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("pfac_matrix must contain only finite values.")
        self._pfac_matrix = matrix.copy()
        self._pfac_matrix.flags.writeable = False

    @cached_property
    def horizontal_to_solid_harmonic(self) -> np.ndarray:
        """Return the explicit horizontal-to-solid matrix."""
        return np.asarray(self.horizontal_to_solid_harmonic_operator.to_matrix(backend="numpy"))

    @cached_property
    def solid_harmonic_to_horizontal(self) -> np.ndarray:
        """Return the explicit solid-to-horizontal matrix."""
        return np.asarray(self.solid_harmonic_to_horizontal_operator.to_matrix(backend="numpy"))

    @cached_property
    def helmholtz_curl_free_potential(self) -> np.ndarray:
        """Return the curl-free Helmholtz-potential selector."""
        return self.helmholtz_curl_free_potential_operator.array

    @cached_property
    def helmholtz_divergence_free_potential(self) -> np.ndarray:
        """Return the divergence-free Helmholtz-potential selector."""
        return self.helmholtz_divergence_free_potential_operator.array

    @cached_property
    def m_imp_to_jr(self) -> np.ndarray:
        """Return the explicit imposed-potential to jr matrix."""
        return np.asarray(self.m_imp_to_jr_operator.to_matrix(backend="numpy")).copy()

    @cached_property
    def m_ind_to_Br(self) -> np.ndarray:
        """Return the explicit induced-potential to Br matrix."""
        return np.asarray(self.m_ind_to_Br_operator.to_matrix(backend="numpy")).copy()

    @cached_property
    def poloidal_to_boundary_potential_jump_factor(self) -> np.ndarray:
        """Return the explicit solid-harmonic jump-factor matrix."""
        return np.asarray(
            self.poloidal_to_boundary_potential_jump_factor_operator.to_matrix(backend="numpy")
        ).copy()

    @cached_property
    def horizontal_to_boundary_potential_jump_factor(self) -> np.ndarray:
        """Return the explicit horizontal-to-jump-factor matrix."""
        return np.asarray(
            self.horizontal_to_boundary_potential_jump_factor_operator.to_matrix(backend="numpy")
        ).copy()

    @cached_property
    def radial_current_constraint_matrix(self) -> np.ndarray:
        """Return the explicit radial-current constraint matrix."""
        return np.asarray(
            self.radial_current_constraint_operator.to_matrix(backend="numpy")
        ).copy()

    def tangential_to_helmholtz(self, vec: np.ndarray) -> np.ndarray:
        """Convert tangential vector field to Helmholtz coeffs."""
        coeffs = np.tensordot(self.helmholtz_analysis_matrix, vec, 2)
        projector = getattr(self.horizontal_basis, "project_helmholtz_mean_free", None)
        return projector(coeffs) if callable(projector) else coeffs

    @cached_property
    def helmholtz_analysis_matrix(self) -> np.ndarray:
        """Matrix mapping gridded vectors to Helmholtz coefficients."""
        return weighted_tensor_pinv(
            self.horizontal_transform.helmholtz_coeffs_to_gridded_vector,
            sqrt_weights=self.model_grid_sqrt_weights(vector=True),
            n_leading_flattened=2,
        )

    def _init_spatial_context(self, cs_basis: CSBasis) -> None:
        """Set up grid, transforms, and background-field evaluators."""
        self.model_grid = Grid(
            theta=cs_basis.arr_theta, phi=cs_basis.arr_phi, area_weights=cs_basis.unit_area
        )
        self.horizontal_transform = SphericalTransform(
            self.horizontal_basis, self.model_grid, area_weighted=self.area_weighted_least_squares
        )
        if self.solid_harmonics.basis is self.horizontal_basis:
            self.solid_harmonic_transform = self.horizontal_transform
        else:
            self.solid_harmonic_transform = SphericalTransform(
                self.solid_harmonics.basis,
                self.model_grid,
                area_weighted=self.area_weighted_least_squares,
            )
        self.main_field_evaluation = MagneticFieldEvaluation(
            self.main_field, self.model_grid, self.RI
        )

        # Optional evaluators for the conjugate hemisphere
        self.conjugate_grid = self.conjugate_horizontal_transform = (
            self.conjugate_main_field_evaluation
        ) = None
        if self.enable_interhemispheric_coupling and self.main_field.kind != "radial":
            cp_theta, cp_phi = self.main_field.conjugate_coordinates(
                self.RI, self.model_grid.theta, self.model_grid.phi
            )
            self.conjugate_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.conjugate_horizontal_transform = SphericalTransform(
                self.horizontal_basis,
                self.conjugate_grid,
                area_weighted=self.area_weighted_least_squares,
            )
            self.conjugate_main_field_evaluation = MagneticFieldEvaluation(
                self.main_field, self.conjugate_grid, self.RI
            )

    def model_grid_sqrt_weights(self, *, vector=False):
        """Return model-grid weights for area-weighted analysis."""
        return resolve_sqrt_weights(
            self.model_grid, area_weighted=self.area_weighted_least_squares, vector=vector
        )

    def _build_horizontal_to_solid_harmonic_operator(self):
        """Project horizontal coefficients into the SH radial space.

        The horizontal basis owns surface operators. ``SolidHarmonics``
        owns the radial laws and wraps the SH basis used for their
        angular coefficients. For the CS horizontal path this is a grid
        least-squares projection from CS nodal values to those SH
        coefficients; for the SH path it is the identity.
        """
        if self.horizontal_solid_projection_is_identity:
            return identity_linear_map((self.horizontal_basis.index_length,))
        solid_to_grid = self.solid_harmonic_transform.scalar_coeffs_to_grid
        horizontal_to_grid = self.horizontal_transform.scalar_coeffs_to_grid
        grid_to_solid = weighted_tensor_pinv(
            solid_to_grid, sqrt_weights=self.model_grid_sqrt_weights(), n_leading_flattened=1
        )
        return as_linear_map(
            np.asarray(grid_to_solid @ horizontal_to_grid),
            input_shape=(self.horizontal_basis.index_length,),
            output_shape=(self.solid_harmonics.basis.index_length,),
        )

    def _build_solid_harmonic_to_horizontal_operator(self):
        """Project solid-harmonic coefficients to horizontal space."""
        if self.horizontal_solid_projection_is_identity:
            return identity_linear_map((self.horizontal_basis.index_length,))
        horizontal_to_grid = self.horizontal_transform.scalar_coeffs_to_grid
        solid_to_grid = self.solid_harmonic_transform.scalar_coeffs_to_grid
        grid_to_horizontal = weighted_tensor_pinv(
            horizontal_to_grid, sqrt_weights=self.model_grid_sqrt_weights(), n_leading_flattened=1
        )
        return as_linear_map(
            np.asarray(grid_to_horizontal @ solid_to_grid),
            input_shape=(self.solid_harmonics.basis.index_length,),
            output_shape=(self.horizontal_basis.index_length,),
        )

    def _horizontal_to_solid_harmonic_matrix(self):
        """Return an explicit horizontal-to-solid map when needed."""
        if self.horizontal_solid_projection_is_identity:
            return None
        return self.horizontal_to_solid_harmonic

    def solid_harmonic_transform_for(self, transform: SphericalTransform) -> SphericalTransform:
        """Return a solid transform for ``transform.target``."""
        if self.solid_harmonics.basis.coefficients_are_compatible_with(transform.source):
            return transform
        cache_key = (
            getattr(self.solid_harmonics.basis, "signature", id(self.solid_harmonics.basis)),
            (
                transform.target.analysis_signature
                if self.area_weighted_least_squares
                else transform.target.signature
            ),
            self.area_weighted_least_squares,
        )
        if cache_key not in self._solid_transform_cache:
            self._solid_transform_cache[cache_key] = SphericalTransform(
                self.solid_harmonics.basis,
                transform.target,
                area_weighted=self.area_weighted_least_squares,
            )
        return self._solid_transform_cache[cache_key]

    def m_ind_to_gridded_JS(
        self,
        transform: Optional[SphericalTransform] = None,
        *,
        solid_transform: Optional[SphericalTransform] = None,
    ) -> np.ndarray:
        """Map induced-potential coefficients to JS."""
        if solid_transform is None:
            solid_transform = (
                self.solid_harmonic_transform
                if transform is None
                else self.solid_harmonic_transform_for(transform)
            )
        return magnetic_boundary.m_ind_to_gridded_JS(
            self.solid_harmonics,
            solid_transform,
            radius=self.RI,
            boundary_radius=self.RM,
            boundary_shielding=self.magnetic_boundary_shielding,
            horizontal_to_solid_harmonic=self._horizontal_to_solid_harmonic_matrix(),
        )

    def m_imp_to_gridded_JS(
        self,
        transform: Optional[SphericalTransform] = None,
        *,
        solid_transform: Optional[SphericalTransform] = None,
    ) -> np.ndarray:
        """Map imposed-potential coefficients to JS."""
        transform = self.horizontal_transform if transform is None else transform
        if solid_transform is None:
            solid_transform = self.solid_harmonic_transform_for(transform)
        return magnetic_boundary.m_imp_to_gridded_JS(
            self.solid_harmonics,
            transform,
            solid_transform=solid_transform,
            horizontal_to_solid_harmonic=self._horizontal_to_solid_harmonic_matrix(),
            pfac_coupling_matrix=(
                None
                if self.main_field.kind == "radial" or not self.enable_pfac_coupling
                else self.pfac_coupling_matrix
            ),
        )

    def _solid_to_horizontal_coefficients(self, values):
        """Map solid-harmonic coefficient rows to horizontal rows."""
        if self.horizontal_solid_projection_is_identity:
            return values
        return np.tensordot(self.solid_harmonic_to_horizontal, values, axes=([1], [0]))

    def _init_constraint_mappings(self) -> None:
        """Initialize geometric operators related to constraints."""
        self.radial_current_constraint_operator = (
            self._build_radial_current_to_apex_current_operator()
        )
        self.interhemispheric_coupling_mask = None
        self.interhemispheric_electric_field_difference_operator = None

        if self.enable_interhemispheric_coupling and self.main_field.kind != "radial":
            magnetic_latitude = self.main_field.magnetic_latitude(
                self.RI, self.model_grid.theta, self.model_grid.phi
            )
            self.interhemispheric_coupling_mask = (
                np.abs(magnetic_latitude) < self.interhemispheric_coupling_latitude
            )

            # Compare mapped radial current at conjugate footpoints.
            conjugate_operator = self._build_radial_current_to_apex_current_operator(
                transform=self.conjugate_horizontal_transform,
                evaluator=self.conjugate_main_field_evaluation,
                output_scale=self.interhemispheric_coupling_mask,
            )
            self.radial_current_constraint_operator = (
                self.radial_current_constraint_operator - conjugate_operator
            )

            local_electric_field_to_apex = self._build_electric_field_to_apex_operator(
                output_mask=self.interhemispheric_coupling_mask
            )
            conjugate_electric_field_to_apex = self._build_electric_field_to_apex_operator(
                transform=self.conjugate_horizontal_transform,
                evaluator=self.conjugate_main_field_evaluation,
                output_mask=self.interhemispheric_coupling_mask,
            )
            self.interhemispheric_electric_field_difference_operator = (
                local_electric_field_to_apex - conjugate_electric_field_to_apex
            )

    def _build_radial_current_to_apex_current_operator(
        self, *, transform=None, evaluator=None, output_scale=None
    ):
        """Return radial-current coefficients mapped to apex current."""
        transform = self.horizontal_transform if transform is None else transform
        evaluator = self.main_field_evaluation if evaluator is None else evaluator
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
        evaluator = self.main_field_evaluation if evaluator is None else evaluator
        grid = self.model_grid if grid is None else grid
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

    def _build_electric_field_to_apex_operator(
        self, *, transform=None, evaluator=None, output_mask=None
    ) -> LinearMap:
        """Return Helmholtz E coefficients mapped to apex components."""
        transform = self.horizontal_transform if transform is None else transform
        return (
            self._build_horizontal_grid_to_apex_operator(
                evaluator=evaluator, grid=transform.target, output_mask=output_mask
            )
            @ transform.helmholtz_coeffs_to_gridded_vector_operator
        )

    @cached_property
    def interhemispheric_electric_field_difference_matrix(self) -> Optional[np.ndarray]:
        """Return explicit low-latitude E-apex difference tensor."""
        operator = self.interhemispheric_electric_field_difference_operator
        if operator is None:
            return None
        return to_numpy(operator.array)

    @cached_property
    def pedersen_geometry_tensor(self) -> np.ndarray:
        """Return the Pedersen part of the resistance tensor."""
        b_th, b_ph, b_r = (
            self.main_field_evaluation.unit_btheta,
            self.main_field_evaluation.unit_bphi,
            self.main_field_evaluation.unit_br,
        )
        return ionospheric_closure.pedersen_geometry_tensor(b_th, b_ph, b_r)

    @cached_property
    def hall_geometry_tensor(self) -> np.ndarray:
        """Return the Hall part of the resistance tensor."""
        return ionospheric_closure.hall_geometry_tensor(self.main_field_evaluation.unit_br)

    @cached_property
    def wind_motional_E_tensor(self) -> np.ndarray:
        """Map neutral wind to motional electric field pointwise."""
        return ionospheric_closure.wind_motional_E_tensor(self.main_field_evaluation.Br)

    @property
    def pfac_coupling_matrix(self) -> np.ndarray:
        """Return the PFAC toroidal-to-poloidal coupling matrix."""
        if self._pfac_matrix is None:
            self._build_pfac_matrix()
        return self._pfac_matrix

    def _pfac_boundary_response(self):
        """Return outer-boundary factors for the PFAC shell response."""
        if self.RM is None:
            return None, -1.0

        outer_regular_to_ionosphere = np.asarray(
            self.solid_harmonics.regular_reference_shift(self.RM, self.RI)
        )
        ionosphere_irregular_to_outer = np.asarray(
            self.solid_harmonics.irregular_reference_shift(self.RI, self.RM)
        )
        response_factor = -1.0 / (
            1.0 - outer_regular_to_ionosphere * ionosphere_irregular_to_outer
        )
        return outer_regular_to_ionosphere, response_factor

    def _pfac_integrand_at_radius(
        self,
        radius,
        gridded_JS_to_solid_poloidal,
        outer_regular_to_ionosphere,
        boundary_response_factor,
    ):
        """Return the imposed-to-poloidal PFAC response at one shell."""
        theta_footpoint, phi_footpoint = self.main_field.map_along_field_lines(
            r_dest=self.RI, r=radius, theta=self.model_grid.theta, phi=self.model_grid.phi
        )
        footpoint_grid = Grid(theta=theta_footpoint, phi=phi_footpoint)
        shell_field = MagneticFieldEvaluation(self.main_field, self.model_grid, radius)
        footpoint_field = MagneticFieldEvaluation(self.main_field, footpoint_grid, self.RI)
        footpoint_transform = SphericalTransform(self.horizontal_basis, footpoint_grid)

        m_imp_to_jr_grid = footpoint_transform.contract_scalar_coeffs_to_grid(
            self.m_imp_to_jr_operator
        )
        jr_to_JS = np.array(
            [shell_field.Btheta / footpoint_field.Br, shell_field.Bphi / footpoint_field.Br]
        )
        m_imp_to_JS = np.einsum("ij,jk->ijk", jr_to_JS, m_imp_to_jr_grid, optimize=True)

        shell_to_ionosphere = np.array(
            self.solid_harmonics.regular_reference_shift(radius, self.RI), copy=True
        ).reshape((-1, 1, 1))
        if self.RM is not None:
            shell_to_ionosphere -= (
                outer_regular_to_ionosphere
                * np.asarray(self.solid_harmonics.irregular_reference_shift(radius, self.RM))
            ).reshape((-1, 1, 1))

        JS_to_solid_poloidal = gridded_JS_to_solid_poloidal * shell_to_ionosphere
        if np.ndim(boundary_response_factor) == 0:
            JS_to_solid_poloidal *= boundary_response_factor
        else:
            JS_to_solid_poloidal *= np.asarray(boundary_response_factor).reshape((-1, 1, 1))
        JS_to_horizontal_poloidal = self._solid_to_horizontal_coefficients(JS_to_solid_poloidal)
        return np.tensordot(JS_to_horizontal_poloidal, m_imp_to_JS, axes=2)

    def _build_pfac_matrix(self) -> None:
        """Construct the PFAC coupling matrix by radial integration."""
        n = self.horizontal_basis.index_length
        pfac_matrix = np.zeros((n, n))
        if self.main_field.kind == "radial" or not self.enable_pfac_coupling:
            pfac_matrix.flags.writeable = False
            self._pfac_matrix = pfac_matrix
            return

        integration_radii = np.asarray(self.fac_integration_radii)
        radial_step_widths = np.diff(integration_radii)
        radial_midpoints = integration_radii[:-1] + 0.5 * radial_step_widths
        solid_poloidal_to_gridded_JS = magnetic_boundary.poloidal_to_gridded_JS(
            self.solid_harmonics, self.solid_harmonic_transform
        )
        gridded_JS_to_solid_poloidal = weighted_tensor_pinv(
            solid_poloidal_to_gridded_JS,
            sqrt_weights=self.model_grid_sqrt_weights(vector=True),
            n_leading_flattened=2,
            rtol=0,
        )
        outer_regular_to_ionosphere, boundary_response_factor = self._pfac_boundary_response()

        for i, radial_midpoint in enumerate(radial_midpoints):
            logger.debug(
                "PFAC integration step %d/%d (rk=%s)",
                i + 1,
                radial_midpoints.size,
                radial_midpoint,
            )
            pfac_matrix += radial_step_widths[i] * self._pfac_integrand_at_radius(
                radial_midpoint,
                gridded_JS_to_solid_poloidal,
                outer_regular_to_ionosphere,
                boundary_response_factor,
            )
        pfac_matrix.flags.writeable = False
        self._pfac_matrix = pfac_matrix

    def Br_to_gridded_JS(
        self,
        transform: Optional[SphericalTransform] = None,
        *,
        solid_transform: Optional[SphericalTransform] = None,
    ) -> Optional[np.ndarray]:
        """Map boundary-Br coefficients to JS."""
        if self.RM is None:
            return None
        if solid_transform is None:
            solid_transform = (
                self.solid_harmonic_transform
                if transform is None
                else self.solid_harmonic_transform_for(transform)
            )
        return magnetic_boundary.Br_to_gridded_JS(
            self.solid_harmonics,
            solid_transform,
            radius=self.RI,
            boundary_radius=self.RM,
            horizontal_to_solid_harmonic=self._horizontal_to_solid_harmonic_matrix(),
        )
