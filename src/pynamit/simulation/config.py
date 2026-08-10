"""Simulation configuration normalization and serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from kompe.constants import EARTH_RADIUS_M
from kompe.math import LeastSquaresSolver, get_default_least_squares_solver

from pynamit.geomagnetism.main_field import (
    decimal_year,
    horizontal_coordinate_system_for_kind,
    normalize_main_field_kind,
)

SIMULATION_SCHEMA_VERSION = 4
INDEPENDENT_PROJECTION_BASIS_KEYS = (
    "boundary_jr",
    "boundary_Br",
    "conductance",
    "u",
    "E_neutral_wind",
)
PROJECTION_BASIS_KEYS = INDEPENDENT_PROJECTION_BASIS_KEYS + ("Q_eff",)
PROJECTION_BASIS_SETTING_NAMES = tuple(f"{key}_projection_basis" for key in PROJECTION_BASIS_KEYS)
INTEGRATORS = {
    "euler": "euler",
    "exponential": "exponential",
    "rk23": "RK23",
    "rk45": "RK45",
    "dop853": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
    "lsoda": "LSODA",
}
_DERIVED_SETTING_NAMES = frozenset((*PROJECTION_BASIS_SETTING_NAMES, "fac_integration_radii"))

_MISSING = object()


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


def _plain_setting_value(value: Any) -> Any:
    """Return plain scalar values from xarray or NumPy wrappers."""
    values = getattr(value, "values", value)
    if getattr(values, "shape", None) == ():
        return values.item()
    return values


def setting_value(settings: Any, name: str, default: Any = _MISSING) -> Any:
    """Return one setting from a settings-like object."""
    attrs = getattr(settings, "attrs", None)
    if attrs is not None and name in attrs:
        return _plain_setting_value(attrs[name])
    if name in getattr(settings, "data_vars", {}):
        return _plain_setting_value(settings[name])
    if isinstance(settings, Mapping) and name in settings:
        return _plain_setting_value(settings[name])
    if hasattr(settings, name):
        return _plain_setting_value(getattr(settings, name))
    if default is not _MISSING:
        return default
    raise AttributeError(name)


def normalize_horizontal_basis_kind(kind: str) -> str:
    """Normalize a simulation horizontal-basis kind."""
    normalized = str(kind).strip().upper()
    if normalized not in {"SH", "CS"}:
        raise ValueError("horizontal_basis_kind must be one of ['CS', 'SH'].")
    return normalized


def normalize_projection_basis_kind(kind: str, *, name: str = "projection_basis") -> str:
    """Normalize an input projection-basis kind."""
    normalized = str(kind).strip().upper()
    if normalized not in {"SH", "CS"}:
        raise ValueError(f"{name} must be one of ['CS', 'SH'].")
    return normalized


def _projection_basis_kind(settings: Any, key: str, default: str) -> str:
    """Return normalized projection-basis setting for one input key."""
    name = f"{key}_projection_basis"
    return normalize_projection_basis_kind(setting_value(settings, name, default), name=name)


def resolve_projection_basis_settings(settings: Any, horizontal_basis_kind: str) -> dict[str, str]:
    """Return normalized input projection-basis settings."""
    horizontal_basis_kind = normalize_horizontal_basis_kind(horizontal_basis_kind)
    projection_settings = {
        f"{key}_projection_basis": _projection_basis_kind(settings, key, horizontal_basis_kind)
        for key in INDEPENDENT_PROJECTION_BASIS_KEYS
    }
    projection_settings["Q_eff_projection_basis"] = _projection_basis_kind(
        settings, "Q_eff", projection_settings["u_projection_basis"]
    )

    if horizontal_basis_kind == "CS":
        invalid = [name for name, value in projection_settings.items() if value != "CS"]
        if invalid:
            raise ValueError(
                ", ".join(invalid) + " must be 'CS' when horizontal_basis_kind is 'CS'."
            )

    return projection_settings


def _zero_to_none(value):
    """Return ``None`` for the persisted zero sentinel."""
    if value is None:
        return None
    if np.ndim(value) == 0 and value == 0:
        return None
    return value


def _values_equal(left, right) -> bool:
    """Return whether two normalized setting values are equal."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return left == right


def _setting_is_present(settings: Any, name: str) -> bool:
    """Return whether a setting is explicitly present."""
    attrs = getattr(settings, "attrs", None)
    if attrs is not None and name in attrs:
        return True
    if name in getattr(settings, "data_vars", {}):
        return True
    if isinstance(settings, Mapping) and name in settings:
        return True
    return hasattr(settings, name)


@dataclass(frozen=True)
class SimulationConfig:
    """Typed source of simulation settings and defaults."""

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
    boundary_jr_projection_basis: str | None = None
    boundary_Br_projection_basis: str | None = None
    conductance_projection_basis: str | None = None
    u_projection_basis: str | None = None
    Q_eff_projection_basis: str | None = None
    E_neutral_wind_projection_basis: str | None = None
    horizontal_basis_kind: str = "SH"
    area_weighted_least_squares: bool = False
    t0: str = "2020-01-01 00:00:00"
    save_equilibria: bool = True
    integrator: str = "euler"
    least_squares_solver: str | None = None
    least_squares_preconditioner: str | None = "pinv"
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
        """Normalize simulation and input-projection basis choices."""
        horizontal_basis_kind = normalize_horizontal_basis_kind(self.horizontal_basis_kind)
        projection_input = {
            name: getattr(self, name)
            for name in PROJECTION_BASIS_SETTING_NAMES
            if getattr(self, name) is not None
        }
        projection_settings = resolve_projection_basis_settings(
            projection_input, horizontal_basis_kind
        )
        object.__setattr__(self, "horizontal_basis_kind", horizontal_basis_kind)
        for name, value in projection_settings.items():
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
            object.__setattr__(self, "least_squares_solver", get_default_least_squares_solver())
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
    def stored_RM(self):
        """Return the persisted radius-magnetosphere setting."""
        return 0 if self.RM is None else self.RM

    @property
    def stored_main_field_B0(self):
        """Return the persisted main-field-strength setting."""
        return 0 if self.main_field_B0 is None else self.main_field_B0

    @property
    def stored_least_squares_preconditioner(self):
        """Return the storage-safe preconditioner setting."""
        return (
            "none"
            if self.least_squares_preconditioner is None
            else self.least_squares_preconditioner
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
            "RM": self.stored_RM,
            "magnetic_boundary_shielding": int(self.magnetic_boundary_shielding),
            "interhemispheric_coupling_latitude": self.interhemispheric_coupling_latitude,
            "enable_pfac_coupling": int(self.enable_pfac_coupling),
            "enable_interhemispheric_coupling": int(self.enable_interhemispheric_coupling),
            "fac_integration_radii": self.fac_integration_radii,
            "interhemispheric_electric_field_weight": self.interhemispheric_electric_field_weight,
            "main_field_kind": self.main_field_kind,
            "main_field_epoch": self.main_field_epoch,
            "main_field_B0": self.stored_main_field_B0,
            "horizontal_coordinate_system": self.horizontal_coordinate_system,
            "boundary_jr_projection_basis": self.boundary_jr_projection_basis,
            "boundary_Br_projection_basis": self.boundary_Br_projection_basis,
            "conductance_projection_basis": self.conductance_projection_basis,
            "u_projection_basis": self.u_projection_basis,
            "Q_eff_projection_basis": self.Q_eff_projection_basis,
            "E_neutral_wind_projection_basis": self.E_neutral_wind_projection_basis,
            "horizontal_basis_kind": self.horizontal_basis_kind,
            "area_weighted_least_squares": int(self.area_weighted_least_squares),
            "t0": self.t0,
            "save_equilibria": int(self.save_equilibria),
            "integrator": self.integrator,
            "least_squares_solver": self.least_squares_solver,
            "least_squares_preconditioner": self.stored_least_squares_preconditioner,
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
    def from_settings(cls, settings: Any, **overrides) -> SimulationConfig:
        """Build a normalized config from settings and overrides."""
        kwargs = {}
        for config_field in fields(cls):
            name = config_field.name
            default = None if name in _DERIVED_SETTING_NAMES else config_field.default
            value = setting_value(settings, name, default)
            if name in {"RM", "main_field_B0"}:
                value = _zero_to_none(value)
            kwargs[name] = value

        for name, explicit in overrides.items():
            if name not in kwargs:
                raise TypeError(f"Unknown simulation setting {name!r}.")
            if explicit is None:
                continue
            if _setting_is_present(settings, name):
                stored_config = cls(**kwargs)
                explicit_config = cls(**{**kwargs, name: explicit})
                if not _values_equal(getattr(stored_config, name), getattr(explicit_config, name)):
                    raise ValueError(f"{name} argument does not match settings.")
            kwargs[name] = explicit

        config = cls(**kwargs)
        if _setting_is_present(settings, "horizontal_coordinate_system"):
            stored_frame = setting_value(settings, "horizontal_coordinate_system")
            if stored_frame != config.horizontal_coordinate_system:
                raise ValueError(
                    "horizontal_coordinate_system does not match main_field_kind: "
                    f"stored={stored_frame!r}, expected={config.horizontal_coordinate_system!r}."
                )
        return config
