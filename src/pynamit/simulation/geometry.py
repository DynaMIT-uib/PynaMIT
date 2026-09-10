"""Run-invariant spatial and magnetic context for simulations."""

from __future__ import annotations

from functools import cached_property, partial
from typing import Any

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
    get_backend,
    identity_linear_map,
    pointwise_component_map,
    take_linear_map,
)
from kompe.spherical_transform import SphericalTransform, resolve_sqrt_weights
from numpy.typing import ArrayLike

from pynamit.geomagnetism import MainField
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure, magnetic_boundary

_BOUNDARY_JR_TO_GAP_BR_CACHE_VERSION = 1
_GEOMETRY_SETTINGS = (
    "RI",
    "RM",
    "magnetic_boundary_shielding",
    "enable_pfac_coupling",
    "fac_integration_radii",
    "enable_interhemispheric_coupling",
    "interhemispheric_coupling_latitude",
    "area_weighted_least_squares",
    "main_field_kind",
    "main_field_epoch",
    "main_field_B0",
)


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

    Treat the shared numerical definitions as fixed. Use ``with_config``
    to change physical assumptions without mutating sibling experiments.
    Only spatial construction settings are retained. Time origin,
    integrator, and fitting policy belong to the consuming experiment.
    """

    def __init__(
        self,
        horizontal_basis: SurfaceDifferentialBasis,
        cs_basis: GlobalCSBasis,
        main_field: MainField,
        config: SimulationConfig,
        *,
        solid_harmonics: SolidHarmonicOperators,
        sh_basis=None,
        boundary_jr_to_gap_Br_matrix: ArrayLike | None = None,
        operator_cache=None,
    ) -> None:
        """Initialize the geometric context."""
        self.horizontal_basis = horizontal_basis
        self.cs_basis = cs_basis
        self.solid_harmonics = solid_harmonics
        self.poloidal_basis = self.solid_harmonics.basis
        self.sh_basis = self.poloidal_basis.with_mean_free(False) if sh_basis is None else sh_basis
        self.main_field = main_field
        self.operator_cache = operator_cache
        self._physical_settings = {name: getattr(config, name) for name in _GEOMETRY_SETTINGS}
        self.backend = get_backend()

        # Store the configuration values used by geometric construction.
        self.RI = config.RI
        self.RM = config.RM
        self.magnetic_boundary_shielding = config.magnetic_boundary_shielding
        self.enable_interhemispheric_coupling = config.enable_interhemispheric_coupling
        self.interhemispheric_coupling_latitude = config.interhemispheric_coupling_latitude
        self.enable_pfac_coupling = config.enable_pfac_coupling
        self.fac_integration_radii = config.fac_integration_radii
        self.area_weighted_least_squares = config.area_weighted_least_squares

        self.model_grid = cs_basis.mesh.cell_centers
        self.horizontal_transform = SphericalTransform(
            horizontal_basis, self.model_grid, area_weighted=config.area_weighted_least_squares
        )
        self._init_boundary_jr_to_gap_Br_matrix(boundary_jr_to_gap_Br_matrix)

    @classmethod
    def from_config(cls, config, *, operator_cache=None):
        """Construct numerical objects from convenience settings."""
        sh_basis = SHBasis(
            config.Nmax, config.Mmax, mean_free=False, operator_cache=operator_cache
        )
        cs_basis = GlobalCSBasis(config.Ncs)
        poloidal_basis = sh_basis.with_mean_free(True)
        return cls(
            cs_basis if config.horizontal_basis_kind == "CS" else poloidal_basis,
            cs_basis,
            build_main_field(config),
            config,
            sh_basis=sh_basis,
            solid_harmonics=SolidHarmonicOperators(poloidal_basis),
            operator_cache=operator_cache,
        )

    @classmethod
    def from_bases(cls, sh_basis, cs_basis, *, horizontal_basis=None, **settings):
        """Reuse Kompe bases and derive their simulation settings.

        ``sh_basis`` supplies poloidal harmonics and SH projections;
        ``cs_basis`` supplies the integration mesh and CS projections.
        Horizontal fields default to mean-free SH. Pass ``cs_basis``
        as ``horizontal_basis`` to use CS surface operators instead.
        The horizontal basis must support a Laplacian within its own
        coefficient space, as required by the MIT equations.
        Supplied bases define the in-memory mathematics. Saving checks
        separately whether the standard file recipe can reconstruct
        them. Supply time origin and solver choices when constructing
        InputPreparation or Simulation, not on the geometry.
        """
        if not isinstance(sh_basis, SurfaceDifferentialBasis) or not isinstance(
            sh_basis.root_basis, SHBasis
        ):
            raise TypeError("sh_basis must be an SH basis or its mean-free view.")
        if not isinstance(cs_basis, GlobalCSBasis):
            raise TypeError("cs_basis must be a GlobalCSBasis.")
        poloidal_basis = sh_basis.with_mean_free(True)
        if horizontal_basis is None:
            horizontal_basis = poloidal_basis
        if not isinstance(horizontal_basis, SurfaceDifferentialBasis):
            raise TypeError("horizontal_basis must provide surface differential operators.")
        derived = dict(
            Nmax=sh_basis.max_degree,
            Mmax=sh_basis.max_order,
            Ncs=cs_basis.cells_per_edge,
            horizontal_basis_kind=horizontal_basis.kind,
        )
        overlap = derived.keys() & settings.keys()
        if overlap:
            raise ValueError(
                f"Basis objects determine {sorted(overlap)}; do not also supply those settings."
            )
        nonspatial = settings.keys() - set(_GEOMETRY_SETTINGS)
        if nonspatial:
            raise ValueError(
                f"{sorted(nonspatial)} are experiment controls; supply them to "
                "InputPreparation.from_geometry or Simulation.from_geometry."
            )
        config = SimulationConfig(**derived, **settings)
        operator_cache = sh_basis.root_basis.operator_cache
        return cls(
            horizontal_basis,
            cs_basis,
            build_main_field(config),
            config,
            sh_basis=sh_basis.with_mean_free(False),
            solid_harmonics=SolidHarmonicOperators(poloidal_basis),
            operator_cache=operator_cache,
        )

    def with_config(self, config):
        """Reuse bases and share only compatible physical caches."""
        for name, value in self.basis_settings.items():
            if getattr(config, name) != value:
                raise ValueError(f"{name} is determined by the existing basis objects.")
        same_field = all(
            np.array_equal(getattr(config, name), self._physical_settings[name])
            for name in ("RI", "main_field_kind", "main_field_epoch", "main_field_B0")
        )
        if self.backend == get_backend() and all(
            np.array_equal(getattr(config, name), self._physical_settings[name])
            for name in _GEOMETRY_SETTINGS
        ):
            return self
        geometry = type(self)(
            self.horizontal_basis,
            self.cs_basis,
            self.main_field if same_field else build_main_field(config),
            config,
            sh_basis=self.sh_basis,
            solid_harmonics=self.solid_harmonics,
            operator_cache=self.operator_cache,
        )
        if (
            self.backend == geometry.backend
            and self.area_weighted_least_squares == geometry.area_weighted_least_squares
        ):
            geometry.horizontal_transform = self.horizontal_transform
        return geometry

    @property
    def physical_settings(self):
        """Return a snapshot of the fixed spatial settings."""
        return self._physical_settings.copy()

    @property
    def basis_settings(self):
        """Resolution and family defaults derived from actual bases.

        These defaults do not describe arbitrary subsets or alternative
        normalizations completely. Persistence checks those explicitly.
        """
        return dict(
            Nmax=self.sh_basis.max_degree,
            Mmax=self.sh_basis.max_order,
            Ncs=self.cs_basis.cells_per_edge,
            horizontal_basis_kind=self.horizontal_basis.kind,
        )

    def __repr__(self):
        """Summarize the simulation's fixed spatial context."""
        return (
            f"SimulationGeometry(horizontal_basis={self.horizontal_basis!r}, "
            f"poloidal_basis={self.poloidal_basis!r}, model_grid={self.model_grid!r}, "
            f"main_field={self.main_field.kind!r}, RI={self.RI:g}, RM={self.RM!r})"
        )

    @cached_property
    def u_coeffs_to_E_coeffs_operator(self):
        """Map wind to motional E independently of conductance."""
        return ionospheric_closure.wind_to_E_coeffs_operator(
            self.helmholtz_analysis_operator,
            self.wind_motional_E_tensor,
            self.horizontal_transform.helmholtz_synthesis_operator,
        )

    @cached_property
    def surface_laplacian_operator(self):
        """Surface Laplacian at the ionosphere."""
        return self.horizontal_basis.surface_laplacian_operator(self.RI)

    @cached_property
    def poloidal_laplacian_operator(self):
        """Laplacian in the radial magnetic-field coefficient space."""
        return self.poloidal_basis.surface_laplacian_operator(self.RI)

    @cached_property
    def helmholtz_curl_free_potential_operator(self):
        """Select the curl-free electric potential."""
        return self.horizontal_basis.helmholtz_curl_free_potential_operator()

    @cached_property
    def helmholtz_divergence_free_potential_operator(self):
        """Select the divergence-free electric potential."""
        return self.horizontal_basis.helmholtz_divergence_free_potential_operator()

    @cached_property
    def toroidal_potential_to_boundary_jr_operator(self):
        """Ampere-law radial current from toroidal potential."""
        return self.RI / MU0 * self.surface_laplacian_operator

    @cached_property
    def induced_poloidal_potential_to_Br_operator(self):
        """Radial magnetic field from private poloidal potential."""
        return -(self.RI**2) * self.poloidal_laplacian_operator

    @cached_property
    def induced_Br_to_poloidal_potential_operator(self):
        """Invert the radial-field degree factors."""
        poloidal_degree = self.poloidal_basis.n
        xp = get_array_module(poloidal_degree)
        poloidal_degree = xp.asarray(poloidal_degree)
        return diagonal_linear_map(1.0 / (poloidal_degree * (poloidal_degree + 1)))

    @property
    def induced_poloidal_potential_faraday_rate_scale(self):
        """Convert surface W to a poloidal-potential time derivative."""
        return 1.0 / self.RI

    @cached_property
    def poloidal_to_normalized_potential_jump_operator(self):
        """Jump in normalized magnetic potential across the sheet."""
        return diagonal_linear_map(
            self.solid_harmonics.poloidal_to_normalized_potential_jump_factors
        )

    def _init_boundary_jr_to_gap_Br_matrix(self, matrix: ArrayLike | None) -> None:
        """Validate and retain an optional boundary-jr to gap-Br map."""
        self._boundary_jr_to_gap_Br_matrix = None
        if matrix is None:
            return
        expected_shape = (
            self.poloidal_basis.coefficient_count,
            self.horizontal_basis.coefficient_count,
        )
        matrix = np.asarray(matrix)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"boundary_jr_to_gap_Br_matrix must have shape {expected_shape}; got {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("boundary_jr_to_gap_Br_matrix must contain only finite values.")
        self._boundary_jr_to_gap_Br_matrix = matrix.copy()
        self._boundary_jr_to_gap_Br_matrix.flags.writeable = False

    @property
    def helmholtz_analysis_operator(self) -> LinearMap:
        """Map gridded vectors to Helmholtz coefficients."""
        return self.horizontal_transform.helmholtz_analysis_operator

    @cached_property
    def poloidal_transform(self):
        """Evaluate and analyze poloidal harmonics on the model grid."""
        return self.horizontal_transform.with_basis(self.poloidal_basis)

    @cached_property
    def conjugate_grid(self):
        """Conjugate footpoints for interhemispheric constraints."""
        if self.enable_interhemispheric_coupling and self.main_field.kind != "radial":
            cp_theta, cp_phi = self.main_field.conjugate_coordinates(
                self.RI, self.model_grid.theta, self.model_grid.phi
            )
            return SphericalGrid(theta=cp_theta, phi=cp_phi)
        return None

    @cached_property
    def conjugate_horizontal_transform(self):
        """Transform on conjugate footpoints for coupled hemispheres."""
        if self.conjugate_grid is not None:
            return SphericalTransform(self.horizontal_basis, self.conjugate_grid)
        return None

    def model_grid_sqrt_weights(self, *, vector=False):
        """Return model-grid weights for area-weighted analysis."""
        return resolve_sqrt_weights(
            self.model_grid, area_weighted=self.area_weighted_least_squares, vector=vector
        )

    @cached_property
    def surface_to_poloidal_operator(self):
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
            return identity_linear_map((self.horizontal_basis.coefficient_count,))
        poloidal_synthesis = self.poloidal_transform.scalar_synthesis_array
        grid_to_poloidal_operator = dense_full_rank_least_squares_map(
            poloidal_synthesis,
            sqrt_weights=self.model_grid_sqrt_weights(),
            input_shape=(self.model_grid.size,),
            output_shape=(self.poloidal_basis.coefficient_count,),
        )
        return grid_to_poloidal_operator @ self.horizontal_transform.scalar_synthesis_operator

    def induced_Br_to_gridded_JS_operator(
        self, transform: SphericalTransform | None = None
    ) -> LinearMap:
        """Map induced Br to sheet current on the requested grid."""
        return magnetic_boundary.induced_Br_to_gridded_JS_operator(
            self.solid_harmonics,
            self.poloidal_transform if transform is None else transform,
            radius=self.RI,
            boundary_radius=self.RM,
            boundary_shielding=self.magnetic_boundary_shielding,
        )

    @cached_property
    def boundary_jr_to_gap_Br_operator(self) -> LinearMap:
        """Map boundary radial current to unshielded gap ``Br(RI)``."""
        return as_linear_map(
            self.boundary_jr_to_gap_Br_matrix,
            input_shape=(self.horizontal_basis.coefficient_count,),
            output_shape=(self.poloidal_basis.coefficient_count,),
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
        self, transform: SphericalTransform | None = None
    ) -> LinearMap:
        """Map toroidal potential to sheet current on this grid."""
        transform = (
            self.horizontal_transform
            if transform is None
            else transform.with_basis(self.horizontal_basis)
        )
        return magnetic_boundary.toroidal_potential_to_gridded_JS_operator(
            self.solid_harmonics,
            transform,
            toroidal_potential_to_boundary_jr=self.toroidal_potential_to_boundary_jr_operator,
            boundary_jr_to_gap_Br=self._active_boundary_jr_to_gap_Br_operator,
        )

    def boundary_jr_to_gridded_JS_operator(
        self, transform: SphericalTransform | None = None
    ) -> LinearMap:
        """Map boundary jr to sheet current on the transform's grid."""
        transform = (
            self.horizontal_transform
            if transform is None
            else transform.with_basis(self.horizontal_basis)
        )
        return magnetic_boundary.boundary_jr_to_gridded_JS_operator(
            self.solid_harmonics,
            transform,
            boundary_jr_to_toroidal_potential=self.boundary_jr_to_toroidal_potential_operator,
            boundary_jr_to_gap_Br=self._active_boundary_jr_to_gap_Br_operator,
        )

    @cached_property
    def interhemispheric_coupling_mask(self):
        """Select the coupled low-latitude region."""
        if self.enable_interhemispheric_coupling and self.main_field.kind != "radial":
            magnetic_latitude = self.main_field.magnetic_latitude(
                self.RI, self.model_grid.theta, self.model_grid.phi
            )
            return np.abs(magnetic_latitude) < self.interhemispheric_coupling_latitude
        return None

    @cached_property
    def radial_current_constraint_operator(self):
        """Compare apex-mapped currents at conjugate footpoints."""
        local = self._build_radial_current_to_apex_current_operator()
        if self.interhemispheric_coupling_mask is not None:
            conjugate_operator = self._build_radial_current_to_apex_current_operator(
                transform=self.conjugate_horizontal_transform,
                output_scale=self.interhemispheric_coupling_mask,
            )
            return local - conjugate_operator
        return local

    @cached_property
    def interhemispheric_electric_field_difference_operator(self):
        """Difference in mapped electric field at coupled footpoints."""
        if self.interhemispheric_coupling_mask is not None:
            local_electric_field_to_apex = self._build_electric_field_to_apex_operator(
                output_mask=self.interhemispheric_coupling_mask
            )
            conjugate_electric_field_to_apex = self._build_electric_field_to_apex_operator(
                transform=self.conjugate_horizontal_transform,
                output_mask=self.interhemispheric_coupling_mask,
            )
            return local_electric_field_to_apex - conjugate_electric_field_to_apex
        return None

    def _build_radial_current_to_apex_current_operator(self, *, transform=None, output_scale=None):
        """Return radial-current coefficients mapped to apex current."""
        transform = self.horizontal_transform if transform is None else transform
        scale_values = self.main_field.radial_to_apex_scale(transform.grid, self.RI)
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

    def _build_horizontal_grid_to_apex_operator(self, *, grid=None, output_mask=None) -> LinearMap:
        """Return horizontal grid vectors mapped to apex components."""
        grid = self.model_grid if grid is None else grid
        if output_mask is None:
            indices = np.arange(grid.size)
        else:
            mask = np.asarray(output_mask, dtype=bool).reshape(-1)
            if mask.shape != (grid.size,):
                raise ValueError("output_mask must match the grid size.")
            indices = np.flatnonzero(mask)

        apex_values = self.main_field.horizontal_to_apex_array(grid, self.RI)[:, :, indices]
        xp = get_array_module(apex_values)
        apex = xp.asarray(apex_values)
        n_grid = int(grid.size)
        apex_rotation = pointwise_component_map(apex)
        if indices.size == n_grid and np.array_equal(indices, np.arange(n_grid)):
            return apex_rotation

        grid_selection = take_linear_map((2, n_grid), indices, axis=1, dtype=apex.dtype)
        return apex_rotation @ grid_selection

    def _build_electric_field_to_apex_operator(
        self, *, transform=None, output_mask=None
    ) -> LinearMap:
        """Return Helmholtz E coefficients mapped to apex components."""
        transform = self.horizontal_transform if transform is None else transform
        return (
            self._build_horizontal_grid_to_apex_operator(
                grid=transform.grid, output_mask=output_mask
            )
            @ transform.helmholtz_synthesis_operator
        )

    @property
    def interhemispheric_electric_field_difference_array(self) -> Any | None:
        """Materialize the shaped low-latitude E-apex difference map."""
        operator = self.interhemispheric_electric_field_difference_operator
        if operator is None:
            return None
        return operator.to_array()

    @cached_property
    def pedersen_geometry_tensor(self) -> Any:
        """Return the Pedersen part of the resistance tensor."""
        b_r, b_th, b_ph = self.main_field.unit_vector(self.model_grid, self.RI)
        return ionospheric_closure.pedersen_geometry_tensor(b_th, b_ph, b_r)

    @cached_property
    def hall_geometry_tensor(self) -> Any:
        """Return the Hall part of the resistance tensor."""
        unit_br = self.main_field.unit_vector(self.model_grid, self.RI)[0]
        return ionospheric_closure.hall_geometry_tensor(unit_br)

    @cached_property
    def wind_motional_E_tensor(self) -> Any:
        """Map neutral wind to motional electric field pointwise."""
        Br = self.main_field.evaluate(self.model_grid, self.RI)[0]
        return ionospheric_closure.wind_motional_E_tensor(Br)

    @property
    def boundary_jr_to_gap_Br_matrix(self) -> np.ndarray:
        """Return the unshielded gap-field response at the ionosphere.

        The matrix maps radial current at the upper ionospheric
        boundary to the poloidal radial magnetic field created by its
        field-aligned continuation through the gap. The result is the
        external-source field incident on the ionosphere, before the
        ionospheric shielding sheet current is applied.
        """
        if self._boundary_jr_to_gap_Br_matrix is None:
            self._build_boundary_jr_to_gap_Br_matrix()
        return self._boundary_jr_to_gap_Br_matrix

    def _build_boundary_jr_to_gap_Br_matrix(self) -> None:
        """Construct the gap-Br map by radial integration."""
        if self.main_field.kind == "radial" or not self.enable_pfac_coupling:
            matrix = np.zeros(
                (self.poloidal_basis.coefficient_count, self.horizontal_basis.coefficient_count)
            )
        else:
            build_matrix = partial(
                magnetic_boundary.boundary_jr_to_gap_Br_matrix,
                self.main_field,
                self.horizontal_basis,
                self.poloidal_transform,
                self.solid_harmonics,
                ionosphere_radius=self.RI,
                integration_radii=self.fac_integration_radii,
                boundary_radius=self.RM,
            )
            matrix = (
                build_matrix()
                if self.operator_cache is None
                else self.operator_cache.get_or_create(
                    "gap_Br_response", self._boundary_jr_to_gap_Br_cache_identity(), build_matrix
                )
            )
        matrix.flags.writeable = False
        self._boundary_jr_to_gap_Br_matrix = matrix

    def _boundary_jr_to_gap_Br_cache_identity(self) -> dict:
        """Return the exact identity of the gap-Br response."""
        field_components = self.main_field.evaluate(self.model_grid, self.RI)
        return {
            "algorithm": "boundary_jr_to_gap_Br_radial_integration",
            "version": _BOUNDARY_JR_TO_GAP_BR_CACHE_VERSION,
            "input_quantity": "boundary_jr_at_RI",
            "output_quantity": "unshielded_gap_Br_at_RI",
            "horizontal_basis": self.horizontal_basis.signature,
            "poloidal_basis": self.poloidal_basis.signature,
            "model_grid_coordinates": self.model_grid.signature,
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

    def boundary_Br_to_gridded_JS_operator(
        self, transform: SphericalTransform | None = None
    ) -> LinearMap | None:
        """Map boundary Br to sheet current on the requested grid."""
        if self.RM is None:
            return None
        return magnetic_boundary.boundary_Br_to_gridded_JS_operator(
            self.solid_harmonics,
            self.poloidal_transform if transform is None else transform,
            radius=self.RI,
            boundary_radius=self.RM,
        )
