"""Simulation configuration normalization and serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import xarray as xr

from pynamit.math.constants import RE
from pynamit.math.least_squares_solver import get_default_least_squares_solver

INDEPENDENT_PROJECTION_BASIS_KEYS = ("jr", "Br", "conductance", "u")
PROJECTION_BASIS_KEYS = INDEPENDENT_PROJECTION_BASIS_KEYS + ("Q_eff",)
PROJECTION_BASIS_SETTING_NAMES = tuple(f"{key}_projection_basis" for key in PROJECTION_BASIS_KEYS)

_MISSING = object()


def default_fac_integration_steps():
    """Return the default radial samples used for FAC integration."""
    return np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)


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
    RI: float = RE + 110.0e3
    RM: float | None = None
    RM_shielding: bool = False
    latitude_boundary: float = 50
    ignore_PFAC: bool = False
    connect_hemispheres: bool = False
    FAC_integration_steps: Any = field(default_factory=default_fac_integration_steps)
    ih_constraint_scaling: float = 1e-5
    mainfield_kind: str = "dipole"
    mainfield_epoch: float = 2020.0
    mainfield_B0: float | None = None
    jr_projection_basis: str | None = None
    Br_projection_basis: str | None = None
    conductance_projection_basis: str | None = None
    u_projection_basis: str | None = None
    Q_eff_projection_basis: str | None = None
    horizontal_basis_kind: str = "SH"
    area_weighted_least_squares: bool = False
    t0: str = "2020-01-01 00:00:00"
    save_steady_states: bool = True
    integrator: str = "euler"
    least_squares_solver: str | None = None
    least_squares_preconditioner: str | None = "pinv"
    static_preconditioner: bool = False
    m_imp_regularization_lambda: float = 0.0

    def __post_init__(self):
        """Normalize settings after dataclass initialization."""
        object.__setattr__(self, "Nmax", int(self.Nmax))
        object.__setattr__(self, "Mmax", int(self.Mmax))
        object.__setattr__(self, "Ncs", int(self.Ncs))
        object.__setattr__(self, "RI", float(self.RI))
        object.__setattr__(self, "RM", _zero_to_none(self.RM))
        object.__setattr__(self, "RM_shielding", bool(self.RM_shielding))
        object.__setattr__(self, "latitude_boundary", float(self.latitude_boundary))
        object.__setattr__(self, "ignore_PFAC", bool(self.ignore_PFAC))
        object.__setattr__(self, "connect_hemispheres", bool(self.connect_hemispheres))
        if self.FAC_integration_steps is None:
            object.__setattr__(self, "FAC_integration_steps", default_fac_integration_steps())
        object.__setattr__(self, "ih_constraint_scaling", float(self.ih_constraint_scaling))
        object.__setattr__(self, "mainfield_kind", str(self.mainfield_kind))
        object.__setattr__(self, "mainfield_epoch", float(self.mainfield_epoch))
        object.__setattr__(self, "mainfield_B0", _zero_to_none(self.mainfield_B0))

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

        object.__setattr__(
            self, "area_weighted_least_squares", bool(self.area_weighted_least_squares)
        )
        object.__setattr__(self, "t0", str(self.t0))
        object.__setattr__(self, "save_steady_states", bool(self.save_steady_states))
        object.__setattr__(self, "integrator", str(self.integrator))
        if self.least_squares_solver is None:
            object.__setattr__(self, "least_squares_solver", get_default_least_squares_solver())
        object.__setattr__(self, "static_preconditioner", bool(self.static_preconditioner))
        object.__setattr__(
            self, "m_imp_regularization_lambda", float(self.m_imp_regularization_lambda)
        )

    @property
    def stored_RM(self):
        """Return the persisted radius-magnetosphere setting."""
        return 0 if self.RM is None else self.RM

    @property
    def stored_mainfield_B0(self):
        """Return the persisted main-field-strength setting."""
        return 0 if self.mainfield_B0 is None else self.mainfield_B0

    def to_attrs(self) -> dict[str, Any]:
        """Return canonical xarray attributes for persisted settings."""
        return {
            "Nmax": self.Nmax,
            "Mmax": self.Mmax,
            "Ncs": self.Ncs,
            "RI": self.RI,
            "RM": self.stored_RM,
            "RM_shielding": int(self.RM_shielding),
            "latitude_boundary": self.latitude_boundary,
            "ignore_PFAC": int(self.ignore_PFAC),
            "connect_hemispheres": int(self.connect_hemispheres),
            "FAC_integration_steps": self.FAC_integration_steps,
            "ih_constraint_scaling": self.ih_constraint_scaling,
            "mainfield_kind": self.mainfield_kind,
            "mainfield_epoch": self.mainfield_epoch,
            "mainfield_B0": self.stored_mainfield_B0,
            "jr_projection_basis": self.jr_projection_basis,
            "Br_projection_basis": self.Br_projection_basis,
            "conductance_projection_basis": self.conductance_projection_basis,
            "u_projection_basis": self.u_projection_basis,
            "Q_eff_projection_basis": self.Q_eff_projection_basis,
            "horizontal_basis_kind": self.horizontal_basis_kind,
            "area_weighted_least_squares": int(self.area_weighted_least_squares),
            "t0": self.t0,
            "save_steady_states": int(self.save_steady_states),
            "integrator": self.integrator,
            "least_squares_solver": self.least_squares_solver,
            "least_squares_preconditioner": self.least_squares_preconditioner,
            "static_preconditioner": int(self.static_preconditioner),
            "m_imp_regularization_lambda": self.m_imp_regularization_lambda,
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
    def from_settings(cls, settings: Any, **overrides) -> "SimulationConfig":
        """Build a normalized config from settings and overrides."""
        defaults = cls()
        kwargs = {}
        for config_field in fields(cls):
            name = config_field.name
            default = None if name in PROJECTION_BASIS_SETTING_NAMES else getattr(defaults, name)
            value = setting_value(settings, name, default)
            if name in {"RM", "mainfield_B0"}:
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

        return cls(**kwargs)
