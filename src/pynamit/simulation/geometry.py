"""Run-invariant spatial and magnetic context for simulations."""

from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
from kompe import (
    GlobalCSBasis,
    SHBasis,
    SolidHarmonicOperators,
    SphericalGrid,
    SurfaceDifferentialBasis,
)
from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.math import (
    LinearMap,
    array_fingerprint,
    as_linear_map,
    dense_full_rank_least_squares_map,
    diagonal_linear_map,
    get_array_module,
    identity_linear_map,
    pointwise_matrix_linear_map,
    take_linear_map,
)
from kompe.spherical_transform import SphericalTransform, resolve_sqrt_weights
from numpy.typing import ArrayLike

from pynamit.geomagnetism import MagneticFieldEvaluation, MainField
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure, magnetic_boundary

logger = logging.getLogger(__name__)
_GAP_BR_RESPONSE_CACHE_VERSION = 1


def build_main_field(config: SimulationConfig) -> MainField:
    """Build the background field selected by a simulation config."""
    return MainField(
        kind=config.main_field_kind,
        epoch=config.main_field_epoch,
        ionosphere_height_km=(config.RI - EARTH_RADIUS_M) * 1e-3,
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
        horizontal_basis: SurfaceDifferentialBasis,
        cs_basis: GlobalCSBasis,
        main_field: MainField,
        config: SimulationConfig,
        gap_Br_response_matrix: ArrayLike | None = None,
        solid_harmonics: SolidHarmonicOperators | None = None,
        operator_cache=None,
    ) -> None:
        """Initialize the geometric context."""
        self.horizontal_basis = horizontal_basis
        self.solid_harmonics = (
            solid_harmonics
            if solid_harmonics is not None
            else (
                SolidHarmonicOperators(horizontal_basis)
                if isinstance(getattr(horizontal_basis, "root_basis", horizontal_basis), SHBasis)
                else None
            )
        )
        if self.solid_harmonics is None:
            raise NotImplementedError(
                f"{type(self.horizontal_basis).__name__} requires solid harmonics for JS coupling."
            )
        self.poloidal_basis = self.solid_harmonics.basis
        self.main_field = main_field
        self.operator_cache = operator_cache

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
        self._init_gap_Br_response_matrix(gap_Br_response_matrix)
        self._poloidal_transform_cache = {}

    def __repr__(self):
        """Summarize the simulation's fixed spatial context."""
        return (
            f"SimulationGeometry(horizontal_basis={self.horizontal_basis!r}, "
            f"poloidal_basis={self.poloidal_basis!r}, model_grid={self.model_grid!r}, "
            f"main_field={self.main_field.kind!r}, RI={self.RI:g}, RM={self.RM!r})"
        )

    def _init_surface_operators(self) -> None:
        """Compile surface and magnetic-boundary coefficient maps."""
        self.surface_laplacian_operator = self.horizontal_basis.surface_laplacian_operator(self.RI)
        self.poloidal_laplacian_operator = self.poloidal_basis.surface_laplacian_operator(self.RI)
        self.helmholtz_curl_free_potential_operator = (
            self.horizontal_basis.helmholtz_curl_free_potential_operator()
        )
        self.helmholtz_divergence_free_potential_operator = (
            self.horizontal_basis.helmholtz_divergence_free_potential_operator()
        )
        self.surface_gauge_operator = self._build_surface_gauge_operator()
        self.toroidal_potential_to_boundary_jr_operator = (
            self.RI / MU0 * self.surface_laplacian_operator
        )
        self.induced_poloidal_potential_to_Br_operator = (
            -(self.RI**2) * self.poloidal_laplacian_operator
        )
        poloidal_degree = self.poloidal_basis.n
        xp = get_array_module(poloidal_degree)
        poloidal_degree = xp.asarray(poloidal_degree)
        self.induced_Br_to_poloidal_potential_operator = diagonal_linear_map(
            1.0 / (poloidal_degree * (poloidal_degree + 1))
        )
        self.induced_poloidal_potential_faraday_rate_scale = 1.0 / self.RI
        self.surface_to_poloidal_operator = self._build_surface_to_poloidal_operator()
        self.poloidal_to_boundary_potential_jump_factor_operator = diagonal_linear_map(
            self.solid_harmonics.poloidal_to_boundary_potential_jump_factor
        )

    def _init_gap_Br_response_matrix(self, matrix: ArrayLike | None) -> None:
        """Validate and retain an optional boundary-jr to gap-Br map."""
        self._gap_Br_response_matrix = None
        if matrix is None:
            return
        expected_shape = (self.poloidal_basis.index_length, self.horizontal_basis.index_length)
        matrix = np.asarray(matrix)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"gap_Br_response_matrix must have shape {expected_shape}; got {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("gap_Br_response_matrix must contain only finite values.")
        self._gap_Br_response_matrix = matrix.copy()
        self._gap_Br_response_matrix.flags.writeable = False

    @property
    def helmholtz_analysis_operator(self) -> LinearMap:
        """Map gridded vectors to Helmholtz coefficients."""
        return self.horizontal_transform.helmholtz_analysis_operator

    def _init_spatial_context(self, cs_basis: GlobalCSBasis) -> None:
        """Set up grid, transforms, and background-field evaluators."""
        self.model_grid = cs_basis.mesh.cell_centers
        self.horizontal_transform = SphericalTransform(
            self.horizontal_basis, self.model_grid, area_weighted=self.area_weighted_least_squares
        )
        if self.poloidal_basis is self.horizontal_basis:
            self.poloidal_transform = self.horizontal_transform
        else:
            self.poloidal_transform = SphericalTransform(
                self.poloidal_basis,
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
            self.conjugate_grid = SphericalGrid(theta=cp_theta, phi=cp_phi)
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

    def _build_surface_gauge_operator(self) -> LinearMap | None:
        """Return a normalized zero-mean constraint if needed."""
        if self.horizontal_basis.scalar_fields_are_mean_free_by_construction():
            return None

        weights_source = self.horizontal_basis.scalar_mean_weights
        xp = get_array_module(weights_source)
        weights = xp.asarray(weights_source, dtype=float)
        expected_shape = (self.horizontal_basis.index_length,)
        if weights.shape != expected_shape:
            raise ValueError(
                f"Surface mean weights must have shape {expected_shape}; got {weights.shape}."
            )
        normalized_mean = self.horizontal_basis.index_length**0.5 * weights
        return as_linear_map(
            normalized_mean.reshape(1, -1), input_shape=expected_shape, output_shape=(1,)
        )

    def _build_surface_to_poloidal_operator(self):
        """Project surface coefficients into poloidal SH space.

        The horizontal basis owns ionospheric surface operators.
        The poloidal basis owns ``induced_Br`` and its radial
        continuation. For the CS surface path this map removes surface
        content that cannot be represented by the configured poloidal
        harmonics; for the SH path it is the identity. The projection is
        composed with the horizontal synthesis operator so a native CS
        identity remains structured rather than becoming a dense matrix.
        """
        if self.poloidal_basis.coefficients_are_compatible_with(self.horizontal_basis):
            return identity_linear_map((self.horizontal_basis.index_length,))
        poloidal_to_grid_matrix = self.poloidal_transform.scalar_synthesis_matrix
        grid_to_poloidal_operator = dense_full_rank_least_squares_map(
            poloidal_to_grid_matrix,
            sqrt_weights=self.model_grid_sqrt_weights(),
            input_shape=(self.model_grid.size,),
            output_shape=(self.poloidal_basis.index_length,),
        )
        return grid_to_poloidal_operator @ self.horizontal_transform.scalar_synthesis_operator

    def poloidal_transform_for(self, transform: SphericalTransform) -> SphericalTransform:
        """Return a poloidal transform for ``transform.grid``."""
        if self.poloidal_basis.coefficients_are_compatible_with(transform.basis):
            return transform
        cache_key = (
            self.poloidal_basis.signature,
            (
                transform.grid.analysis_signature
                if self.area_weighted_least_squares
                else transform.grid.signature
            ),
            self.area_weighted_least_squares,
        )
        if cache_key not in self._poloidal_transform_cache:
            self._poloidal_transform_cache[cache_key] = SphericalTransform(
                self.poloidal_basis, transform.grid, area_weighted=self.area_weighted_least_squares
            )
        return self._poloidal_transform_cache[cache_key]

    def induced_Br_to_gridded_JS_operator(
        self,
        transform: SphericalTransform | None = None,
        *,
        poloidal_transform: SphericalTransform | None = None,
    ) -> LinearMap:
        """Return the map from induced ``Br(RI)`` to sheet current."""
        if poloidal_transform is None:
            poloidal_transform = (
                self.poloidal_transform
                if transform is None
                else self.poloidal_transform_for(transform)
            )
        return magnetic_boundary.induced_Br_to_gridded_JS_operator(
            self.solid_harmonics,
            poloidal_transform,
            radius=self.RI,
            boundary_radius=self.RM,
            boundary_shielding=self.magnetic_boundary_shielding,
        )

    @cached_property
    def boundary_jr_to_gap_Br_operator(self) -> LinearMap:
        """Map boundary radial current to unshielded gap ``Br(RI)``."""
        return as_linear_map(
            self.boundary_jr_to_gap_Br_matrix,
            input_shape=(self.horizontal_basis.index_length,),
            output_shape=(self.poloidal_basis.index_length,),
        )

    @property
    def _active_boundary_jr_to_gap_Br_operator(self) -> LinearMap | None:
        """Return the gap response only when that coupling is active."""
        if self.main_field.kind == "radial" or not self.enable_pfac_coupling:
            return None
        return self.boundary_jr_to_gap_Br_operator

    @cached_property
    def boundary_jr_to_toroidal_potential_operator(self) -> LinearMap:
        """Return the gauge-fixed boundary-current inverse."""
        return MU0 / self.RI * self.horizontal_basis.mean_free_surface_poisson_operator(self.RI)

    def toroidal_potential_to_gridded_JS_operator(
        self,
        transform: SphericalTransform | None = None,
        *,
        poloidal_transform: SphericalTransform | None = None,
    ) -> LinearMap:
        """Return the private toroidal-potential current map."""
        transform = self.horizontal_transform if transform is None else transform
        gap_response = self._active_boundary_jr_to_gap_Br_operator
        if gap_response is not None and poloidal_transform is None:
            poloidal_transform = self.poloidal_transform_for(transform)
        return magnetic_boundary.toroidal_potential_to_gridded_JS_operator(
            self.solid_harmonics,
            transform,
            poloidal_transform=poloidal_transform,
            toroidal_potential_to_boundary_jr=(self.toroidal_potential_to_boundary_jr_operator),
            boundary_jr_to_gap_Br=gap_response,
        )

    def boundary_jr_to_gridded_JS_operator(
        self,
        transform: SphericalTransform | None = None,
        *,
        poloidal_transform: SphericalTransform | None = None,
    ) -> LinearMap:
        """Return the boundary-jr to total sheet-current map."""
        transform = self.horizontal_transform if transform is None else transform
        gap_response = self._active_boundary_jr_to_gap_Br_operator
        if gap_response is not None and poloidal_transform is None:
            poloidal_transform = self.poloidal_transform_for(transform)
        return magnetic_boundary.boundary_jr_to_gridded_JS_operator(
            self.solid_harmonics,
            transform,
            poloidal_transform=poloidal_transform,
            boundary_jr_to_toroidal_potential=(self.boundary_jr_to_toroidal_potential_operator),
            boundary_jr_to_gap_Br=gap_response,
        )

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
        scale_values = evaluator.radial_to_apex
        xp = get_array_module(scale_values, output_scale)
        scale = xp.asarray(scale_values)
        if output_scale is not None:
            scale = scale * xp.asarray(output_scale)
        scale_operator = diagonal_linear_map(
            scale.reshape(-1),
            input_shape=(transform.grid.size,),
            output_shape=(transform.grid.size,),
        )
        return scale_operator @ transform.scalar_synthesis_operator

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

        apex_values = np.asarray(evaluator.horizontal_to_apex)[:, :, indices]
        xp = get_array_module(apex_values)
        apex = xp.asarray(apex_values)
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
                evaluator=evaluator, grid=transform.grid, output_mask=output_mask
            )
            @ transform.helmholtz_synthesis_operator
        )

    @cached_property
    def interhemispheric_electric_field_difference_matrix(self) -> np.ndarray | None:
        """Return explicit low-latitude E-apex difference tensor."""
        operator = self.interhemispheric_electric_field_difference_operator
        if operator is None:
            return None
        return operator.array

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
    def boundary_jr_to_gap_Br_matrix(self) -> np.ndarray:
        """Return the unshielded gap-field response at the ionosphere.

        The matrix maps radial current at the upper ionospheric
        boundary to the poloidal radial magnetic field created by its
        field-aligned continuation through the gap. The result is the
        external-source field incident on the ionosphere, before the
        ionospheric shielding sheet current is applied.
        """
        if self._gap_Br_response_matrix is None:
            self._build_gap_Br_response_matrix()
        return self._gap_Br_response_matrix

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

    def _gap_Br_integrand_at_radius(
        self,
        radius,
        gridded_JS_to_poloidal_operator,
        outer_regular_to_ionosphere,
        boundary_response_factor,
    ):
        """Return one shell's shielding-potential response."""
        theta_footpoint, phi_footpoint = self.main_field.map_along_field_lines(
            r_dest=self.RI, r=radius, theta=self.model_grid.theta, phi=self.model_grid.phi
        )
        footpoint_grid = SphericalGrid(theta=theta_footpoint, phi=phi_footpoint)
        shell_field = MagneticFieldEvaluation(self.main_field, self.model_grid, radius)
        footpoint_field = MagneticFieldEvaluation(self.main_field, footpoint_grid, self.RI)
        footpoint_transform = SphericalTransform(
            self.horizontal_basis, footpoint_grid, use_persistent_evaluation_cache=False
        )

        jr_to_gridded_JS = pointwise_matrix_linear_map(
            np.array(
                [shell_field.Btheta / footpoint_field.Br, shell_field.Bphi / footpoint_field.Br]
            ).reshape(2, 1, self.model_grid.size)
        )

        poloidal_scale = np.array(
            self.solid_harmonics.regular_reference_shift(radius, self.RI), copy=True
        )
        if self.RM is not None:
            poloidal_scale -= outer_regular_to_ionosphere * np.asarray(
                self.solid_harmonics.irregular_reference_shift(radius, self.RM)
            )

        poloidal_scale *= boundary_response_factor
        integrand_operator = (
            diagonal_linear_map(poloidal_scale)
            @ gridded_JS_to_poloidal_operator
            @ jr_to_gridded_JS
            @ footpoint_transform.scalar_synthesis_operator
        )
        return np.asarray(integrand_operator.to_matrix(backend="numpy"))

    def _build_gap_Br_response_matrix(self) -> None:
        """Construct the gap-Br map by radial integration."""
        if self.main_field.kind == "radial" or not self.enable_pfac_coupling:
            matrix = np.zeros(
                (self.poloidal_basis.index_length, self.horizontal_basis.index_length)
            )
        elif self.operator_cache is None:
            matrix = self._compute_gap_Br_response_matrix()
        else:
            matrix = self.operator_cache.get_or_create(
                "gap_Br_response",
                self._gap_Br_response_cache_identity(),
                self._compute_gap_Br_response_matrix,
            )
        matrix.flags.writeable = False
        self._gap_Br_response_matrix = matrix

    def _gap_Br_response_cache_identity(self) -> dict:
        """Return the exact identity of the gap-Br response."""
        field_components = (
            self.main_field_evaluation.Br,
            self.main_field_evaluation.Btheta,
            self.main_field_evaluation.Bphi,
        )
        return {
            "algorithm": "boundary_jr_to_gap_Br_radial_integration",
            "version": _GAP_BR_RESPONSE_CACHE_VERSION,
            "input_quantity": "boundary_jr_at_RI",
            "output_quantity": "unshielded_gap_Br_at_RI",
            "horizontal_basis": self.horizontal_basis.signature,
            "poloidal_basis": self.poloidal_basis.signature,
            "model_grid_coordinates": self.model_grid.exact_coordinate_signature,
            "model_grid_area_weights": array_fingerprint(self.model_grid.area_weights),
            "main_field_kind": self.main_field.kind,
            "main_field_epoch": self.main_field.epoch,
            "main_field_on_model_grid": tuple(
                array_fingerprint(component) for component in field_components
            ),
            "ionosphere_radius": self.RI,
            "boundary_radius": self.RM,
            "integration_radii": array_fingerprint(self.fac_integration_radii),
            "area_weighted_least_squares": self.area_weighted_least_squares,
        }

    def _compute_gap_Br_response_matrix(self) -> np.ndarray:
        """Compute the physical gap-field response from shell maps."""
        shielding_potential_response = np.zeros(
            (self.poloidal_basis.index_length, self.horizontal_basis.index_length)
        )
        integration_radii = np.asarray(self.fac_integration_radii)
        radial_step_widths = np.diff(integration_radii)
        radial_midpoints = integration_radii[:-1] + 0.5 * radial_step_widths
        gridded_JS_to_poloidal_operator = (
            self.poloidal_transform.rhat_cross_gradient_analysis_operator(
                coefficient_scale=(
                    -np.asarray(self.solid_harmonics.poloidal_to_boundary_potential_jump_factor)
                    / MU0
                )
            )
        )
        outer_regular_to_ionosphere, boundary_response_factor = self._pfac_boundary_response()

        for i, radial_midpoint in enumerate(radial_midpoints):
            logger.debug(
                "Gap-field integration step %d/%d (rk=%s)",
                i + 1,
                radial_midpoints.size,
                radial_midpoint,
            )
            shielding_potential_response += radial_step_widths[
                i
            ] * self._gap_Br_integrand_at_radius(
                radial_midpoint,
                gridded_JS_to_poloidal_operator,
                outer_regular_to_ionosphere,
                boundary_response_factor,
            )

        # The integrated response above is the poloidal coefficient of
        # the ionospheric shielding field. If b_gap is the unshielded
        # external radial field incident from the gap, shielding gives
        # kappa = -D^-1 b_gap, hence b_gap = -D kappa.
        degree_factor = np.asarray(
            self.poloidal_basis.n * (self.poloidal_basis.n + 1), dtype=float
        )
        return -degree_factor[:, None] * shielding_potential_response

    def boundary_Br_to_gridded_JS_operator(
        self,
        transform: SphericalTransform | None = None,
        *,
        poloidal_transform: SphericalTransform | None = None,
    ) -> LinearMap | None:
        """Return the map from outer-boundary Br to sheet current."""
        if self.RM is None:
            return None
        if poloidal_transform is None:
            poloidal_transform = (
                self.poloidal_transform
                if transform is None
                else self.poloidal_transform_for(transform)
            )
        return magnetic_boundary.boundary_Br_to_gridded_JS_operator(
            self.solid_harmonics, poloidal_transform, radius=self.RI, boundary_radius=self.RM
        )
