"""Simulation configuration normalization and serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from kompe.constants import EARTH_RADIUS_M
from kompe.math import (
    LINEAR_EVOLUTION_METHODS,
    LeastSquaresSolver,
    get_default_least_squares_solver,
)

from pynamit.coordinates import decimal_year
from pynamit.geomagnetism.main_field import (
    horizontal_coordinate_system_for_kind,
    normalize_main_field_kind,
)

SIMULATION_SCHEMA_VERSION = 4
INPUT_REMAPPING_KEYS = ("boundary_jr", "boundary_Br", "u", "E_neutral_wind", "Q_eff")
INPUT_REMAPPING_SETTINGS = tuple(f"{key}_remapping" for key in INPUT_REMAPPING_KEYS)
INTEGRATORS = {method.lower(): method for method in LINEAR_EVOLUTION_METHODS}


def _integer_setting(value: Any, *, name: str, minimum: int) -> int:
    """Return an integer setting without silently truncating it."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    integer = int(value)
    if integer != value or integer < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return integer


def _boolean_setting(value: Any, *, name: str) -> bool:
    """Return a boolean without accepting arbitrary truthy values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _normalize_integrator(value: Any) -> str:
    """Return a canonical built-in or SciPy integration method."""
    key = str(value).strip().lower()
    try:
        return INTEGRATORS[key]
    except KeyError as exc:
        raise ValueError(f"integrator must be one of {list(INTEGRATORS.values())}.") from exc


def _normalize_least_squares_solver(value: Any) -> str:
    """Return a supported least-squares solver name."""
    normalized = str(value).strip().lower()
    if normalized not in LeastSquaresSolver.VALID_SOLVERS:
        raise ValueError(
            f"least_squares_solver must be one of {list(LeastSquaresSolver.VALID_SOLVERS)}."
        )
    return normalized


def _normalize_least_squares_preconditioner(value: Any) -> str | None:
    """Return a supported preconditioner or ``None``."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized not in LeastSquaresSolver.VALID_PRECONDITIONERS:
        raise ValueError(
            "least_squares_preconditioner must be one of "
            f"{list(LeastSquaresSolver.VALID_PRECONDITIONERS)} or None."
        )
    return normalized


def _normalize_start_time(value: Any) -> str:
    """Return a canonical timezone-naive UTC start time."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("t0 must be a valid datetime-like value.") from exc
    if pd.isna(timestamp):
        raise ValueError("t0 must be a valid datetime-like value.")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat(sep=" ")


def default_fac_integration_radii(RI=EARTH_RADIUS_M + 110.0e3, RM=None):
    """Return the default radial samples used for FAC integration."""
    outer_radius = 4 * EARTH_RADIUS_M if RM is None else RM
    return np.logspace(np.log10(RI), np.log10(outer_radius), 11)


def dipole_fac_integration_radii(inner_radius, outer_radius, n_points):
    """Return FAC radii uniform in the legacy dipole latitude parameter.

    This preserves the established secant-squared spacing policy. The
    parameter controls radial-shell quadrature density; it is not the
    latitude trace of one physical field line.
    """
    inner_radius = float(inner_radius)
    outer_radius = float(outer_radius)
    if not np.isfinite(inner_radius) or not np.isfinite(outer_radius):
        raise ValueError("Dipole sampling radii must be finite.")
    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("Dipole sampling requires 0 < inner_radius < outer_radius.")
    try:
        point_count = float(n_points)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dipole sampling requires an integer point count.") from exc
    if not np.isfinite(point_count) or not point_count.is_integer() or point_count < 2:
        raise ValueError("Dipole sampling requires at least two integer points.")
    max_latitude = np.arccos(np.sqrt(inner_radius / outer_radius))
    magnetic_latitude = np.linspace(0.0, max_latitude, int(point_count))
    return inner_radius / np.cos(magnetic_latitude) ** 2


def normalize_horizontal_basis_kind(kind: str) -> str:
    """Normalize a simulation horizontal-basis kind."""
    normalized = str(kind).strip().upper()
    if normalized not in {"SH", "CS"}:
        raise ValueError("horizontal_basis_kind must be one of ['CS', 'SH'].")
    return normalized


def normalize_input_remapping(value, *, name="input_remapping"):
    """Select direct fitting or cubed-sphere sample remapping."""
    normalized = str(value).strip().lower()
    if normalized not in {"direct", "cs"}:
        raise ValueError(f"{name} must be 'direct' or 'CS'.")
    return "CS" if normalized == "cs" else "direct"


def resolve_input_remapping(settings, horizontal_basis_kind):
    """Resolve remapping independently of coefficient spaces."""
    default = "CS" if horizontal_basis_kind == "CS" else "direct"
    resolved = {}
    for key in INPUT_REMAPPING_KEYS:
        name = f"{key}_remapping"
        inherited = resolved["u_remapping"] if key == "Q_eff" else default
        resolved[name] = normalize_input_remapping(settings.get(name, inherited), name=name)
        if horizontal_basis_kind == "CS" and key != "boundary_Br" and resolved[name] != "CS":
            raise ValueError(f"{name} must be 'CS' when horizontal_basis_kind is 'CS'.")
    return resolved


def _zero_to_none(value):
    """Return ``None`` for the persisted zero sentinel."""
    if value is None:
        return None
    if np.ndim(value) == 0 and value == 0:
        return None
    return value


@dataclass(frozen=True)
class SimulationConfig:
    """Typed source of simulation settings and defaults.

    ``interhemispheric_electric_field_weight`` multiplies the conjugate
    electric-field residual before combining it with radial-current
    residuals. Its SI units are S/m, not a dimensionless probability or
    an exact constraint; its square weights the least-squares term.
    """

    Nmax: int = 20
    Mmax: int = 20
    Ncs: int = 30
    RI: float = EARTH_RADIUS_M + 110.0e3
    RM: float | None = None
    magnetic_boundary_shielding: bool = False
    interhemispheric_coupling_latitude: float = 50
    enable_pfac_coupling: bool = True
    enable_interhemispheric_coupling: bool = False
    fac_integration_radii: Any = None
    interhemispheric_electric_field_weight: float = 1e-5
    main_field_kind: str = "dipole"
    main_field_epoch: float | None = None
    main_field_B0: float | None = None
    boundary_jr_remapping: str | None = None
    boundary_Br_remapping: str | None = None
    conductance_basis: str | None = None
    u_remapping: str | None = None
    Q_eff_remapping: str | None = None
    E_neutral_wind_remapping: str | None = None
    horizontal_basis_kind: str = "SH"
    area_weighted_least_squares: bool = False
    t0: str = "2020-01-01 00:00:00"
    save_equilibria: bool = True
    integrator: str = "euler"
    least_squares_solver: str | None = None
    least_squares_tolerance: float = 1e-15
    least_squares_preconditioner: str | None = None
    reuse_preconditioner: bool = False
    toroidal_potential_regularization_lambda: float = 0.0

    def __post_init__(self):
        """Normalize settings after dataclass initialization."""
        self._normalize_resolution()
        self._normalize_radial_domain()
        self._normalize_coupling()
        object.__setattr__(self, "t0", _normalize_start_time(self.t0))
        self._normalize_main_field()
        self._normalize_bases()
        self._normalize_numerical_policy()

    def _normalize_resolution(self):
        """Normalize angular and cubed-sphere resolution."""
        object.__setattr__(self, "Nmax", _integer_setting(self.Nmax, name="Nmax", minimum=1))
        object.__setattr__(self, "Mmax", _integer_setting(self.Mmax, name="Mmax", minimum=0))
        object.__setattr__(self, "Ncs", _integer_setting(self.Ncs, name="Ncs", minimum=2))
        if self.Mmax > self.Nmax:
            raise ValueError("Mmax must be less than or equal to Nmax.")
        if self.Ncs % 2:
            raise ValueError("Ncs must be even for the cubed-sphere grid.")

    def _normalize_radial_domain(self):
        """Normalize radial bounds and FAC quadrature."""
        object.__setattr__(self, "RI", float(self.RI))
        object.__setattr__(
            self,
            "magnetic_boundary_shielding",
            _boolean_setting(self.magnetic_boundary_shielding, name="magnetic_boundary_shielding"),
        )
        if not np.isfinite(self.RI) or self.RI <= EARTH_RADIUS_M:
            raise ValueError(
                "RI must be finite and greater than Earth's reference radius EARTH_RADIUS_M."
            )
        if self.RM is not None:
            object.__setattr__(self, "RM", float(self.RM))
            if not np.isfinite(self.RM) or self.RM <= self.RI:
                raise ValueError("RM must be finite and greater than RI.")
        if self.magnetic_boundary_shielding and self.RM is None:
            raise ValueError(
                "magnetic_boundary_shielding requires a finite magnetospheric radius RM."
            )
        integration_radii = (
            default_fac_integration_radii(self.RI, self.RM)
            if self.fac_integration_radii is None
            else np.asarray(self.fac_integration_radii, dtype=float)
        )
        integration_radii = np.array(integration_radii, dtype=float, copy=True)
        if integration_radii.ndim != 1 or integration_radii.size < 2:
            raise ValueError("fac_integration_radii must be a one-dimensional radial grid.")
        if not np.all(np.isfinite(integration_radii)) or np.any(np.diff(integration_radii) <= 0.0):
            raise ValueError("fac_integration_radii must be finite and strictly increasing.")
        radius_tolerance = 1e-12 * self.RI
        if integration_radii[0] < self.RI - radius_tolerance:
            raise ValueError("fac_integration_radii must start at or outside RI.")
        if self.RM is not None and integration_radii[-1] > self.RM + 1e-12 * self.RM:
            raise ValueError("fac_integration_radii must end at or inside RM.")
        integration_radii.setflags(write=False)
        object.__setattr__(self, "fac_integration_radii", integration_radii)

    def _normalize_coupling(self):
        """Normalize magnetic coupling policy."""
        object.__setattr__(
            self,
            "interhemispheric_coupling_latitude",
            float(self.interhemispheric_coupling_latitude),
        )
        if (
            not np.isfinite(self.interhemispheric_coupling_latitude)
            or not 0.0 <= self.interhemispheric_coupling_latitude <= 90.0
        ):
            raise ValueError(
                "interhemispheric_coupling_latitude must be finite and between 0 and 90 degrees."
            )
        object.__setattr__(
            self,
            "enable_pfac_coupling",
            _boolean_setting(self.enable_pfac_coupling, name="enable_pfac_coupling"),
        )
        object.__setattr__(
            self,
            "enable_interhemispheric_coupling",
            _boolean_setting(
                self.enable_interhemispheric_coupling, name="enable_interhemispheric_coupling"
            ),
        )
        object.__setattr__(
            self,
            "interhemispheric_electric_field_weight",
            float(self.interhemispheric_electric_field_weight),
        )
        if (
            not np.isfinite(self.interhemispheric_electric_field_weight)
            or self.interhemispheric_electric_field_weight < 0.0
        ):
            raise ValueError(
                "interhemispheric_electric_field_weight must be finite and non-negative."
            )

    def _normalize_main_field(self):
        """Normalize the background magnetic-field specification."""
        object.__setattr__(
            self, "main_field_kind", normalize_main_field_kind(self.main_field_kind)
        )
        epoch = (
            decimal_year(pd.Timestamp(self.t0).to_pydatetime())
            if self.main_field_epoch is None
            else float(self.main_field_epoch)
        )
        object.__setattr__(self, "main_field_epoch", epoch)
        if not np.isfinite(self.main_field_epoch):
            raise ValueError("main_field_epoch must be finite.")
        if self.main_field_B0 is not None:
            object.__setattr__(self, "main_field_B0", float(self.main_field_B0))
            if not np.isfinite(self.main_field_B0) or self.main_field_B0 <= 0.0:
                raise ValueError("main_field_B0 must be finite and greater than zero.")
            if self.main_field_kind == "igrf":
                raise ValueError("main_field_B0 is not supported for the IGRF main-field model.")

    def _normalize_bases(self):
        """Normalize basis and sample-remapping choices."""
        horizontal_basis_kind = normalize_horizontal_basis_kind(self.horizontal_basis_kind)
        remapping = resolve_input_remapping(
            {
                name: getattr(self, name)
                for name in INPUT_REMAPPING_SETTINGS
                if getattr(self, name) is not None
            },
            horizontal_basis_kind,
        )
        object.__setattr__(self, "horizontal_basis_kind", horizontal_basis_kind)
        object.__setattr__(
            self,
            "conductance_basis",
            normalize_horizontal_basis_kind(
                horizontal_basis_kind if self.conductance_basis is None else self.conductance_basis
            ),
        )
        for name, value in remapping.items():
            object.__setattr__(self, name, value)

    def _normalize_numerical_policy(self):
        """Normalize time integration and least-squares policy."""
        object.__setattr__(
            self,
            "area_weighted_least_squares",
            _boolean_setting(self.area_weighted_least_squares, name="area_weighted_least_squares"),
        )
        object.__setattr__(
            self, "save_equilibria", _boolean_setting(self.save_equilibria, name="save_equilibria")
        )
        object.__setattr__(self, "integrator", _normalize_integrator(self.integrator))
        if self.least_squares_solver is None:
            default_solver = "lsmr" if self.horizontal_basis_kind == "CS" else "normal_pinv"
            object.__setattr__(
                self,
                "least_squares_solver",
                get_default_least_squares_solver(default=default_solver),
            )
        object.__setattr__(
            self,
            "least_squares_solver",
            _normalize_least_squares_solver(self.least_squares_solver),
        )
        object.__setattr__(
            self,
            "least_squares_preconditioner",
            _normalize_least_squares_preconditioner(self.least_squares_preconditioner),
        )
        if isinstance(self.least_squares_tolerance, (bool, np.bool_)):
            raise TypeError("least_squares_tolerance must be a finite non-negative scalar.")
        object.__setattr__(self, "least_squares_tolerance", float(self.least_squares_tolerance))
        if not np.isfinite(self.least_squares_tolerance) or self.least_squares_tolerance < 0.0:
            raise ValueError("least_squares_tolerance must be a finite non-negative scalar.")
        object.__setattr__(
            self,
            "reuse_preconditioner",
            _boolean_setting(self.reuse_preconditioner, name="reuse_preconditioner"),
        )
        object.__setattr__(
            self,
            "toroidal_potential_regularization_lambda",
            float(self.toroidal_potential_regularization_lambda),
        )
        if (
            not np.isfinite(self.toroidal_potential_regularization_lambda)
            or self.toroidal_potential_regularization_lambda < 0.0
        ):
            raise ValueError(
                "toroidal_potential_regularization_lambda must be finite and non-negative."
            )

    @property
    def horizontal_coordinate_system(self):
        """Return the horizontal frame implied by the main field."""
        return horizontal_coordinate_system_for_kind(self.main_field_kind)

    def to_attrs(self) -> dict[str, Any]:
        """Return canonical xarray attributes for persisted settings."""
        return {
            "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
            "Nmax": self.Nmax,
            "Mmax": self.Mmax,
            "Ncs": self.Ncs,
            "RI": self.RI,
            "RM": 0 if self.RM is None else self.RM,
            "magnetic_boundary_shielding": int(self.magnetic_boundary_shielding),
            "interhemispheric_coupling_latitude": self.interhemispheric_coupling_latitude,
            "enable_pfac_coupling": int(self.enable_pfac_coupling),
            "enable_interhemispheric_coupling": int(self.enable_interhemispheric_coupling),
            "fac_integration_radii": self.fac_integration_radii,
            "interhemispheric_electric_field_weight": self.interhemispheric_electric_field_weight,
            "main_field_kind": self.main_field_kind,
            "main_field_epoch": self.main_field_epoch,
            "main_field_B0": 0 if self.main_field_B0 is None else self.main_field_B0,
            "horizontal_coordinate_system": self.horizontal_coordinate_system,
            "boundary_jr_projection_basis": "SH"
            if self.boundary_jr_remapping == "direct"
            else "CS",
            "boundary_Br_projection_basis": "SH"
            if self.boundary_Br_remapping == "direct"
            else "CS",
            "conductance_projection_basis": self.conductance_basis,
            "u_projection_basis": "SH" if self.u_remapping == "direct" else "CS",
            "Q_eff_projection_basis": "SH" if self.Q_eff_remapping == "direct" else "CS",
            "E_neutral_wind_projection_basis": "SH"
            if self.E_neutral_wind_remapping == "direct"
            else "CS",
            "horizontal_basis_kind": self.horizontal_basis_kind,
            "area_weighted_least_squares": int(self.area_weighted_least_squares),
            "t0": self.t0,
            "save_equilibria": int(self.save_equilibria),
            "integrator": self.integrator,
            "least_squares_solver": self.least_squares_solver,
            "least_squares_tolerance": self.least_squares_tolerance,
            "least_squares_preconditioner": (
                "none"
                if self.least_squares_preconditioner is None
                else self.least_squares_preconditioner
            ),
            "reuse_preconditioner": int(self.reuse_preconditioner),
            "toroidal_potential_regularization_lambda": (
                self.toroidal_potential_regularization_lambda
            ),
        }

    def to_dataset(self) -> xr.Dataset:
        """Return canonical persisted settings as an xarray dataset."""
        return xr.Dataset(attrs=self.to_attrs())

    def to_kwargs(self) -> dict[str, Any]:
        """Return normalized constructor keyword arguments."""
        return {
            config_field.name: getattr(self, config_field.name) for config_field in fields(self)
        }

    @classmethod
    def from_geometry(cls, geometry, **settings):
        """Combine spatial objects with explicit experiment controls.

        The geometry contributes radii, background-field and coupling
        settings, and basis resolution. Other choices use the normal
        experiment defaults unless supplied here.
        """
        for name, value in geometry.basis_settings.items():
            if name in settings and settings[name] != value:
                raise ValueError(f"{name} is determined by the existing basis objects.")
        return cls(**(geometry.physical_settings | geometry.basis_settings | settings))

    def validate_geometry_for_storage(self, geometry):
        """Require the file recipe to reproduce the supplied bases.

        This is a serialization restriction, not a requirement for
        in-memory calculations with other coefficient spaces.
        """
        from kompe import GlobalCSBasis, SHBasis

        sh = SHBasis(self.Nmax, self.Mmax, mean_free=False)
        cs = GlobalCSBasis(self.Ncs)
        poloidal = sh.with_mean_free(True)
        expected = {
            "sh_basis": sh,
            "cs_basis": cs,
            "poloidal_basis": poloidal,
            "horizontal_basis": cs if self.horizontal_basis_kind == "CS" else poloidal,
        }
        for name, basis in expected.items():
            if not getattr(geometry, name).coefficients_are_compatible_with(basis):
                raise ValueError(
                    f"Cannot save {name}: the file format reconstructs standard Schmidt SH "
                    "truncations and global CS bases. This geometry can be used in memory."
                )

    @classmethod
    def from_settings(cls, settings: SimulationConfig | xr.Dataset | Mapping) -> SimulationConfig:
        """Read configuration from saved or in-memory settings.

        Xarray attributes take precedence over data variables. Use
        ``dataclasses.replace(config, ...)`` for deliberate changes.
        """
        if isinstance(settings, cls):
            return settings
        if isinstance(settings, xr.Dataset):
            settings = {**settings.data_vars, **settings.attrs}
        elif not isinstance(settings, Mapping):
            raise TypeError("settings must be a SimulationConfig, xarray Dataset, or mapping.")

        # Version-4 artifacts retain their historical projection-setting
        # names. Translate only at this settings/file boundary.
        settings = dict(settings)
        for key in INPUT_REMAPPING_KEYS:
            old_name, name = f"{key}_projection_basis", f"{key}_remapping"
            if name not in settings and old_name in settings:
                settings[name] = "direct" if settings[old_name] == "SH" else settings[old_name]
        if "conductance_basis" not in settings and "conductance_projection_basis" in settings:
            settings["conductance_basis"] = settings["conductance_projection_basis"]
        kwargs = {}
        for config_field in fields(cls):
            name = config_field.name
            value = settings.get(name, config_field.default)
            if isinstance(value, xr.DataArray):
                value = value.values
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            if name in {"RM", "main_field_B0"}:
                value = _zero_to_none(value)
            kwargs[name] = value

        config = cls(**kwargs)
        if "horizontal_coordinate_system" in settings:
            stored_frame = settings["horizontal_coordinate_system"]
            if stored_frame != config.horizontal_coordinate_system:
                raise ValueError(
                    "horizontal_coordinate_system does not match main_field_kind: "
                    f"stored={stored_frame!r}, expected={config.horizontal_coordinate_system!r}."
                )
        return config
