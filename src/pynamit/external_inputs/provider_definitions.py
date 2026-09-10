"""Empirical input-provider definitions and coordinate semantics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pynamit.coordinates import CENTERED_DIPOLE, GEOCENTRIC_GEOGRAPHIC
from pynamit.external_inputs.coordinates import (
    LIBRARY_GEOGRAPHIC_110KM,
    PYNAMIT_CENTERED_DIPOLE_110KM,
    PYNAMIT_SPHERICAL_GEO_110KM,
    CoordinateConvention,
    _canonical_json,
    _freeze_mapping,
    _json_value,
)


@dataclass(frozen=True)
class InputProviderSpec:
    """Provider semantics independent of any particular sample grid."""

    key: str
    implementation: str
    sampling_policy: str
    request_coordinate_convention: CoordinateConvention
    output_coordinate_convention: CoordinateConvention
    fields: tuple[str, ...]
    request_coordinate_views: Mapping[str, CoordinateConvention] = field(default_factory=dict)
    request_vector_basis: str | None = None
    output_vector_basis: str | None = None
    derived_coordinates: Mapping[str, Any] = field(default_factory=dict)
    adapter_assumptions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the provider specification."""
        key = str(self.key).strip()
        implementation = str(self.implementation).strip()
        policy = str(self.sampling_policy).strip()
        fields = tuple(str(name).strip() for name in self.fields)
        if not key or not implementation:
            raise ValueError("Provider key and implementation cannot be empty.")
        if policy not in {"requested_positions", "provider_native_grid"}:
            raise ValueError(f"Unsupported sampling policy {policy!r}.")
        if not fields or any(not name for name in fields):
            raise ValueError("Provider fields must be non-empty.")

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "sampling_policy", policy)
        object.__setattr__(self, "fields", fields)
        coordinate_views = {
            str(name).strip(): convention
            for name, convention in self.request_coordinate_views.items()
        }
        if any(not name for name in coordinate_views) or not all(
            isinstance(convention, CoordinateConvention)
            for convention in coordinate_views.values()
        ):
            raise ValueError(
                "Provider coordinate views require named CoordinateConvention values."
            )
        object.__setattr__(self, "request_coordinate_views", MappingProxyType(coordinate_views))
        object.__setattr__(self, "derived_coordinates", _freeze_mapping(self.derived_coordinates))
        object.__setattr__(self, "adapter_assumptions", _freeze_mapping(self.adapter_assumptions))
        for name in ("request_vector_basis", "output_vector_basis"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                object.__setattr__(self, name, value or None)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InputProviderSpec:
        """Construct a provider specification from metadata."""
        return cls(
            key=str(payload["key"]),
            implementation=str(payload["implementation"]),
            sampling_policy=str(payload["sampling_policy"]),
            request_coordinate_convention=CoordinateConvention.from_dict(
                payload["request_coordinate_contract"]
            ),
            output_coordinate_convention=CoordinateConvention.from_dict(
                payload["output_coordinate_contract"]
            ),
            fields=tuple(payload["fields"]),
            request_coordinate_views={
                name: CoordinateConvention.from_dict(convention)
                for name, convention in payload.get("request_coordinate_views", {}).items()
            },
            request_vector_basis=payload.get("request_vector_basis"),
            output_vector_basis=payload.get("output_vector_basis"),
            derived_coordinates=payload.get("derived_coordinates", {}),
            adapter_assumptions=payload.get("adapter_assumptions", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serialized provider metadata."""
        result = {
            "key": self.key,
            "implementation": self.implementation,
            "sampling_policy": self.sampling_policy,
            "request_coordinate_contract": (self.request_coordinate_convention.to_dict()),
            "output_coordinate_contract": (self.output_coordinate_convention.to_dict()),
            "fields": list(self.fields),
            "derived_coordinates": _json_value(self.derived_coordinates),
            "adapter_assumptions": _json_value(self.adapter_assumptions),
        }
        if self.request_vector_basis is not None:
            result["request_vector_basis"] = self.request_vector_basis
        if self.output_vector_basis is not None:
            result["output_vector_basis"] = self.output_vector_basis
        if self.request_coordinate_views:
            result["request_coordinate_views"] = {
                name: convention.to_dict()
                for name, convention in self.request_coordinate_views.items()
            }
        return result

    @property
    def signature(self) -> str:
        """Return a stable provider-specification signature."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        """Hash by immutable provider semantics."""
        return hash(self.signature)


CONDUCTANCE_PROVIDER_SPEC = InputProviderSpec(
    key="conductance",
    implementation="lompe.conductance",
    sampling_policy="requested_positions",
    request_coordinate_convention=LIBRARY_GEOGRAPHIC_110KM,
    output_coordinate_convention=PYNAMIT_SPHERICAL_GEO_110KM,
    fields=("hall", "pedersen"),
    request_coordinate_views={"model": PYNAMIT_CENTERED_DIPOLE_110KM},
    derived_coordinates={
        "auroral_model": {
            CENTERED_DIPOLE: "originating_model_coordinates",
            GEOCENTRIC_GEOGRAPHIC: "apexpy_modified_apex_at_110_km",
        },
        "solar_zenith_angle": "geographic",
    },
    adapter_assumptions={
        "request_mapping": (
            "The physical sample grid is geocentric geographic at 110 km. "
            "Centered-dipole simulations additionally provide their model-grid view."
        ),
        "coordinate_selection": (
            "Centered-dipole requests evaluate Hardy in the model view at model_epoch "
            "and GEO requests derive modified-Apex coordinates at the full event time. "
            "Both evaluate EUV illumination in the paired physical GEO view and use a "
            "decimal-year dipole for magnetic local time."
        ),
    },
)

BOUNDARY_JR_PROVIDER_SPEC = InputProviderSpec(
    key="boundary_jr",
    implementation="pyamps.AMPS.get_upward_current",
    sampling_policy="requested_positions",
    request_coordinate_convention=LIBRARY_GEOGRAPHIC_110KM,
    output_coordinate_convention=PYNAMIT_SPHERICAL_GEO_110KM,
    fields=("jr",),
    request_coordinate_views={"model": PYNAMIT_CENTERED_DIPOLE_110KM},
    derived_coordinates={
        "magnetic_model": {
            CENTERED_DIPOLE: "originating_model_coordinates",
            GEOCENTRIC_GEOGRAPHIC: "apexpy_modified_apex_at_110_km",
        },
        "magnetic_local_time": {
            CENTERED_DIPOLE: "simulation_dipole_at_model_epoch",
            GEOCENTRIC_GEOGRAPHIC: "pyamps_at_event_decimal_year",
        },
    },
    adapter_assumptions={
        "request_mapping": (
            "The physical sample grid is geocentric geographic at 110 km. "
            "Centered-dipole simulations additionally provide their model-grid view."
        ),
        "coordinate_selection": (
            "Centered-dipole requests interpret AMPS in the simulation dipole frame "
            "at model_epoch. GEO requests derive IGRF modified-Apex coordinates at "
            "the full event time."
        ),
    },
)

NEUTRAL_WIND_PROVIDER_SPEC = InputProviderSpec(
    key="neutral_wind",
    implementation="pyhwm2014.hwm14_vectorized",
    sampling_policy="requested_positions",
    request_coordinate_convention=LIBRARY_GEOGRAPHIC_110KM,
    output_coordinate_convention=PYNAMIT_SPHERICAL_GEO_110KM,
    fields=("u_theta", "u_phi"),
    request_vector_basis="library_geographic_east_north",
    output_vector_basis="geocentric_spherical_theta_phi",
    adapter_assumptions={
        "request_mapping": (
            "PynaMIT spherical latitude/longitude are passed through numerically "
            "at the same nominal 110-km altitude."
        ),
        "vector_mapping": (
            "Library east/north components are identified with PynaMIT east/north "
            "components under the same spherical approximation."
        ),
    },
)

PROVIDER_SPECS = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            CONDUCTANCE_PROVIDER_SPEC,
            BOUNDARY_JR_PROVIDER_SPEC,
            NEUTRAL_WIND_PROVIDER_SPEC,
        )
    }
)
