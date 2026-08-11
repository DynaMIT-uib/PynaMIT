"""Immutable contracts and grids for external empirical inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.coordinates import CENTERED_DIPOLE, GEOCENTRIC_GEOGRAPHIC
from pynamit.geodesy import spherical_geo_to_library_geographic

_COORDINATE_IDENTITY_VERSION = 3
_COORDINATE_QUANTIZATION = "little_endian_float32"
_IONOSPHERE_ALTITUDE_KM = 110.0


def _json_value(value: Any) -> Any:
    """Return one recursively JSON-compatible value."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable recursively copied mapping."""
    if value is None:
        return MappingProxyType({})

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(subitem) for key, subitem in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(subitem) for subitem in item)
        if isinstance(item, np.generic):
            return item.item()
        return item

    return freeze(value)


def _canonical_json(value: Any) -> str:
    """Return stable compact JSON."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class ReferenceSurface:
    """Reference surface used to interpret horizontal coordinates."""

    kind: str
    radius_m: float | None = None
    altitude_km: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the reference surface."""
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("Reference-surface kind cannot be empty.")
        object.__setattr__(self, "kind", kind)

        if self.radius_m is not None:
            radius = float(self.radius_m)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("radius_m must be finite and positive.")
            object.__setattr__(self, "radius_m", radius)

        if self.altitude_km is not None:
            altitude = float(self.altitude_km)
            if not np.isfinite(altitude):
                raise ValueError("altitude_km must be finite.")
            object.__setattr__(self, "altitude_km", altitude)

        if self.radius_m is None and self.altitude_km is None:
            raise ValueError("Reference surface requires either radius_m or altitude_km.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceSurface:
        """Construct a reference surface from serialized metadata."""
        return cls(
            kind=str(payload["type"]),
            radius_m=payload.get("radius_m"),
            altitude_km=payload.get("altitude_km"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serialized reference-surface metadata."""
        result: dict[str, Any] = {"type": self.kind}
        if self.radius_m is not None:
            result["radius_m"] = self.radius_m
        if self.altitude_km is not None:
            result["altitude_km"] = self.altitude_km
        return result


@dataclass(frozen=True)
class CoordinateContract:
    """Semantic interpretation of horizontal coordinates."""

    coordinate_system: str
    angular_units: str
    latitude_definition: str
    longitude_definition: str
    longitude_wrap: str
    reference_surface: ReferenceSurface

    def __post_init__(self) -> None:
        """Validate and normalize the coordinate contract."""
        for name in (
            "coordinate_system",
            "angular_units",
            "latitude_definition",
            "longitude_definition",
            "longitude_wrap",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"Coordinate field {name!r} cannot be empty.")
            object.__setattr__(self, name, value)

        if self.angular_units != "degrees":
            raise ValueError("Only degree-valued coordinates are supported.")
        if self.longitude_definition != "east_positive":
            raise ValueError("Longitude must be east-positive.")
        if self.longitude_wrap != "[-180,180)":
            raise ValueError("Longitude must use the [-180, 180) convention.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CoordinateContract:
        """Construct a coordinate contract from metadata."""
        return cls(
            coordinate_system=str(payload["coordinate_system"]),
            angular_units=str(payload["angular_units"]),
            latitude_definition=str(payload["latitude_definition"]),
            longitude_definition=str(payload["longitude_definition"]),
            longitude_wrap=str(payload["longitude_wrap"]),
            reference_surface=ReferenceSurface.from_dict(payload["reference_surface"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serialized coordinate-contract metadata."""
        return {
            "coordinate_system": self.coordinate_system,
            "angular_units": self.angular_units,
            "latitude_definition": self.latitude_definition,
            "longitude_definition": self.longitude_definition,
            "longitude_wrap": self.longitude_wrap,
            "reference_surface": self.reference_surface.to_dict(),
        }

    @property
    def signature(self) -> str:
        """Return a stable contract signature."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def normalize(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return owned read-only normalized ordered coordinates."""
        latitude = np.asarray(lat, dtype=float).reshape(-1)
        longitude = np.asarray(lon, dtype=float).reshape(-1)
        if latitude.size == 0 or latitude.size != longitude.size:
            raise ValueError("Coordinates must be non-empty paired samples.")
        if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
            raise ValueError("Coordinates must be finite.")
        if np.any(latitude < -90.0) or np.any(latitude > 90.0):
            raise ValueError("Latitude must lie in [-90, 90] degrees.")

        longitude = (longitude + 180.0) % 360.0 - 180.0
        latitude = np.array(latitude, copy=True, order="C")
        longitude = np.array(longitude, copy=True, order="C")
        latitude.setflags(write=False)
        longitude.setflags(write=False)
        return latitude, longitude

    def coordinate_identity(self, lat: np.ndarray, lon: np.ndarray) -> str:
        """Hash the contract and normalized ordered coordinates."""
        latitude, longitude = self.normalize(lat, lon)
        header = {
            "identity_version": _COORDINATE_IDENTITY_VERSION,
            "coordinate_contract": self.to_dict(),
            "point_count": int(latitude.size),
            "ordering": "ordered_pairs",
            "coordinate_quantization": _COORDINATE_QUANTIZATION,
        }
        ordered_pairs = np.column_stack((latitude, longitude)).astype("<f4", copy=False)
        digest = hashlib.sha256()
        digest.update(_canonical_json(header).encode("utf-8"))
        digest.update(ordered_pairs.tobytes(order="C"))
        return digest.hexdigest()


PYNAMIT_SPHERICAL_GEO_110KM = CoordinateContract(
    coordinate_system=GEOCENTRIC_GEOGRAPHIC,
    angular_units="degrees",
    latitude_definition="geocentric",
    longitude_definition="east_positive",
    longitude_wrap="[-180,180)",
    reference_surface=ReferenceSurface(
        kind="sphere", radius_m=float(EARTH_RADIUS_M + _IONOSPHERE_ALTITUDE_KM * 1e3)
    ),
)

PYNAMIT_CENTERED_DIPOLE_110KM = CoordinateContract(
    coordinate_system=CENTERED_DIPOLE,
    angular_units="degrees",
    latitude_definition="centered_dipole",
    longitude_definition="east_positive",
    longitude_wrap="[-180,180)",
    reference_surface=ReferenceSurface(
        kind="sphere", radius_m=float(EARTH_RADIUS_M + _IONOSPHERE_ALTITUDE_KM * 1e3)
    ),
)

LIBRARY_GEOGRAPHIC_110KM = CoordinateContract(
    coordinate_system="library_geographic",
    angular_units="degrees",
    latitude_definition="numerical_identity_from_geocentric",
    longitude_definition="east_positive",
    longitude_wrap="[-180,180)",
    reference_surface=ReferenceSurface(
        kind="nominal_altitude", altitude_km=_IONOSPHERE_ALTITUDE_KM
    ),
)


@dataclass(frozen=True)
class ProviderSpec:
    """Provider semantics independent of any particular sample grid."""

    key: str
    implementation: str
    sampling_policy: str
    request_coordinate_contract: CoordinateContract
    output_coordinate_contract: CoordinateContract
    fields: tuple[str, ...]
    request_coordinate_views: Mapping[str, CoordinateContract] = field(default_factory=dict)
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
            str(name).strip(): contract for name, contract in self.request_coordinate_views.items()
        }
        if any(not name for name in coordinate_views) or not all(
            isinstance(contract, CoordinateContract) for contract in coordinate_views.values()
        ):
            raise ValueError("Provider coordinate views require named CoordinateContract values.")
        object.__setattr__(self, "request_coordinate_views", MappingProxyType(coordinate_views))
        object.__setattr__(self, "derived_coordinates", _freeze_mapping(self.derived_coordinates))
        object.__setattr__(self, "adapter_assumptions", _freeze_mapping(self.adapter_assumptions))
        for name in ("request_vector_basis", "output_vector_basis"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                object.__setattr__(self, name, value or None)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProviderSpec:
        """Construct a provider specification from metadata."""
        return cls(
            key=str(payload["key"]),
            implementation=str(payload["implementation"]),
            sampling_policy=str(payload["sampling_policy"]),
            request_coordinate_contract=CoordinateContract.from_dict(
                payload["request_coordinate_contract"]
            ),
            output_coordinate_contract=CoordinateContract.from_dict(
                payload["output_coordinate_contract"]
            ),
            fields=tuple(payload["fields"]),
            request_coordinate_views={
                name: CoordinateContract.from_dict(contract)
                for name, contract in payload.get("request_coordinate_views", {}).items()
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
            "request_coordinate_contract": (self.request_coordinate_contract.to_dict()),
            "output_coordinate_contract": (self.output_coordinate_contract.to_dict()),
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
                name: contract.to_dict()
                for name, contract in self.request_coordinate_views.items()
            }
        return result

    @property
    def signature(self) -> str:
        """Return a stable provider-specification signature."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        """Hash by immutable provider semantics."""
        return hash(self.signature)


CONDUCTANCE_PROVIDER_SPEC = ProviderSpec(
    key="conductance",
    implementation="lompe.conductance",
    sampling_policy="requested_positions",
    request_coordinate_contract=LIBRARY_GEOGRAPHIC_110KM,
    output_coordinate_contract=PYNAMIT_SPHERICAL_GEO_110KM,
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

BOUNDARY_JR_PROVIDER_SPEC = ProviderSpec(
    key="boundary_jr",
    implementation="pyamps.AMPS.get_upward_current",
    sampling_policy="requested_positions",
    request_coordinate_contract=LIBRARY_GEOGRAPHIC_110KM,
    output_coordinate_contract=PYNAMIT_SPHERICAL_GEO_110KM,
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

NEUTRAL_WIND_PROVIDER_SPEC = ProviderSpec(
    key="neutral_wind",
    implementation="pyhwm2014.hwm14_vectorized",
    sampling_policy="requested_positions",
    request_coordinate_contract=LIBRARY_GEOGRAPHIC_110KM,
    output_coordinate_contract=PYNAMIT_SPHERICAL_GEO_110KM,
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


@dataclass(frozen=True)
class SampleGrid:
    """Immutable ordered coordinates under one coordinate contract."""

    grid_id: str
    coordinate_contract: CoordinateContract
    lat: np.ndarray = field(repr=False, compare=False)
    lon: np.ndarray = field(repr=False, compare=False)
    sampling_geometry: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    coordinate_identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate, normalize, and own the sample-grid data."""
        grid_id = str(self.grid_id).strip()
        if not grid_id:
            raise ValueError("Sample-grid ID cannot be empty.")
        latitude, longitude = self.coordinate_contract.normalize(self.lat, self.lon)
        object.__setattr__(self, "grid_id", grid_id)
        object.__setattr__(self, "lat", latitude)
        object.__setattr__(self, "lon", longitude)
        object.__setattr__(self, "sampling_geometry", _freeze_mapping(self.sampling_geometry))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(
            self,
            "coordinate_identity",
            self.coordinate_contract.coordinate_identity(latitude, longitude),
        )

    @property
    def size(self) -> int:
        """Return the number of ordered sample positions."""
        return int(self.lat.size)

    def __hash__(self) -> int:
        """Hash by semantic ordered-grid identity."""
        return hash(self.coordinate_identity)

    @classmethod
    def from_dict(cls, grid_id: str, payload: Mapping[str, Any]) -> SampleGrid:
        """Construct and verify a serialized sample grid."""
        grid = cls(
            grid_id=grid_id,
            coordinate_contract=CoordinateContract.from_dict(payload["coordinate_contract"]),
            lat=np.asarray(payload["lat"]),
            lon=np.asarray(payload["lon"]),
            sampling_geometry=payload.get("sampling_geometry", {}),
            provenance=payload.get("provenance", {}),
        )
        recorded = payload.get("coordinate_identity")
        if recorded is not None and str(recorded) != grid.coordinate_identity:
            raise ValueError(f"Sample grid {grid_id!r} has a stale coordinate identity.")
        return grid

    def to_dict(self) -> dict[str, Any]:
        """Return serialized grid metadata and coordinates."""
        return {
            "coordinate_contract": self.coordinate_contract.to_dict(),
            "sampling_geometry": _json_value(self.sampling_geometry),
            "provenance": _json_value(self.provenance),
            "coordinate_identity": self.coordinate_identity,
            "lat": self.lat.tolist(),
            "lon": self.lon.tolist(),
        }


def _spherical_geo_to_library_110km(
    source_grid: SampleGrid, target_contract: CoordinateContract
) -> SampleGrid:
    """Create a library grid using the spherical identity map."""
    altitude = target_contract.reference_surface.altitude_km
    if altitude is None:
        raise ValueError("Library request contract requires altitude_km.")
    latitude, longitude, _ = spherical_geo_to_library_geographic(
        source_grid.lat, source_grid.lon, altitude
    )
    target_short = target_contract.signature[:12]
    return SampleGrid(
        grid_id=f"{source_grid.grid_id}--request-{target_short}",
        coordinate_contract=target_contract,
        lat=latitude,
        lon=longitude,
        sampling_geometry={"type": "mapped_ordered_points", "source_grid_id": source_grid.grid_id},
        provenance={
            "source_coordinate_identity": source_grid.coordinate_identity,
            "mapping": "numerical_lat_lon_identity_at_equal_nominal_altitude",
        },
    )


class ExternalInputRequest:
    """One ordered physical grid with explicit coordinate views."""

    def __init__(
        self,
        source_grid: SampleGrid,
        *,
        model_grid: SampleGrid | None = None,
        model_epoch: float | None = None,
    ):
        if source_grid.coordinate_contract != PYNAMIT_SPHERICAL_GEO_110KM:
            raise ValueError("External-input source_grid must be geocentric geographic.")
        model_grid = source_grid if model_grid is None else model_grid
        if model_grid.size != source_grid.size:
            raise ValueError("Model and geographic grids must contain the same ordered samples.")
        if model_grid.coordinate_contract not in {
            PYNAMIT_SPHERICAL_GEO_110KM,
            PYNAMIT_CENTERED_DIPOLE_110KM,
        }:
            raise ValueError("External-input model_grid uses an unsupported coordinate system.")
        if (
            model_grid.coordinate_contract == PYNAMIT_SPHERICAL_GEO_110KM
            and model_grid.coordinate_identity != source_grid.coordinate_identity
        ):
            raise ValueError("Geographic model and source grids must be identical.")
        if model_epoch is not None:
            model_epoch = float(model_epoch)
            if not np.isfinite(model_epoch):
                raise ValueError("model_epoch must be finite.")
        if (
            model_grid.coordinate_contract == PYNAMIT_CENTERED_DIPOLE_110KM
            and model_epoch is None
        ):
            raise ValueError("Centered-dipole model coordinates require model_epoch.")
        self.source_grid = source_grid
        self.model_grid = model_grid
        self.model_epoch = model_epoch
        self._request_grids: dict[str, SampleGrid] = {
            source_grid.coordinate_contract.signature: source_grid,
            model_grid.coordinate_contract.signature: model_grid,
        }

    @classmethod
    def from_geocentric_geo(
        cls,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        grid_id: str = "runtime-geocentric-grid",
        sampling_geometry: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ExternalInputRequest:
        """Construct a request from PynaMIT spherical-GEO positions."""
        return cls(
            SampleGrid(
                grid_id=grid_id,
                coordinate_contract=PYNAMIT_SPHERICAL_GEO_110KM,
                lat=lat,
                lon=lon,
                sampling_geometry={} if sampling_geometry is None else sampling_geometry,
                provenance={} if provenance is None else provenance,
            )
        )

    @classmethod
    def from_model_coordinates(
        cls,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        geographic_lat: np.ndarray,
        geographic_lon: np.ndarray,
        coordinate_system: str,
        model_epoch: float | None = None,
        grid_id: str = "runtime-model-grid",
        sampling_geometry: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ExternalInputRequest:
        """Construct model and GEO views of the same ordered samples."""
        coordinate_system = str(coordinate_system).strip().lower()
        try:
            model_contract = {
                CENTERED_DIPOLE: PYNAMIT_CENTERED_DIPOLE_110KM,
                GEOCENTRIC_GEOGRAPHIC: PYNAMIT_SPHERICAL_GEO_110KM,
            }[coordinate_system]
        except KeyError as exc:
            raise ValueError(
                "coordinate_system must be 'centered_dipole' or 'geocentric_geographic'."
            ) from exc

        source = SampleGrid(
            grid_id=grid_id,
            coordinate_contract=PYNAMIT_SPHERICAL_GEO_110KM,
            lat=geographic_lat,
            lon=geographic_lon,
            sampling_geometry={} if sampling_geometry is None else sampling_geometry,
            provenance={} if provenance is None else provenance,
        )
        if model_contract == PYNAMIT_SPHERICAL_GEO_110KM:
            model_identity = model_contract.coordinate_identity(lat, lon)
            if model_identity != source.coordinate_identity:
                raise ValueError("Geographic model coordinates must match geographic samples.")
            model_grid = source
        else:
            model_grid = SampleGrid(
                grid_id=f"{grid_id}--model",
                coordinate_contract=model_contract,
                lat=lat,
                lon=lon,
                sampling_geometry={"type": "coordinate_view", "source_grid_id": source.grid_id},
                provenance={
                    **({} if provenance is None else provenance),
                    "model_epoch": float(model_epoch) if model_epoch is not None else None,
                    "source_coordinate_identity": source.coordinate_identity,
                },
            )
        return cls(source, model_grid=model_grid, model_epoch=model_epoch)

    def grid_for(self, spec_or_contract: ProviderSpec | CoordinateContract) -> SampleGrid:
        """Return the cached request grid for a provider or contract."""
        contract = (
            spec_or_contract.request_coordinate_contract
            if isinstance(spec_or_contract, ProviderSpec)
            else spec_or_contract
        )
        cached = self._request_grids.get(contract.signature)
        if cached is not None:
            return cached

        if not (
            self.source_grid.coordinate_contract == PYNAMIT_SPHERICAL_GEO_110KM
            and contract == LIBRARY_GEOGRAPHIC_110KM
        ):
            raise ValueError(
                "No coordinate conversion is defined from "
                f"{self.source_grid.coordinate_contract.coordinate_system!r} "
                f"to {contract.coordinate_system!r}."
            )
        converted = _spherical_geo_to_library_110km(self.source_grid, contract)
        self._request_grids[contract.signature] = converted
        return converted


@dataclass(frozen=True)
class ProviderDataset:
    """Provider values bound to shared source and request grids."""

    spec: ProviderSpec
    source_grid: SampleGrid
    request_grid: SampleGrid
    values: Mapping[str, np.ndarray] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate grid bindings and own the provider values."""
        if self.source_grid.coordinate_contract != (self.spec.output_coordinate_contract):
            raise ValueError("Provider output and source-grid contracts differ.")
        if self.request_grid.coordinate_contract != (self.spec.request_coordinate_contract):
            raise ValueError("Provider request and request-grid contracts differ.")
        if self.source_grid.size != self.request_grid.size:
            raise ValueError("Source and request grids must have equal sizes.")

        supplied = set(self.values)
        required = set(self.spec.fields)
        if supplied != required:
            raise ValueError(
                f"Provider {self.spec.key!r} requires fields "
                f"{sorted(required)}, got {sorted(supplied)}."
            )

        owned: dict[str, np.ndarray] = {}
        for name in self.spec.fields:
            values = np.asarray(self.values[name]).reshape(-1)
            if values.size != self.source_grid.size:
                raise ValueError(
                    f"Provider field {name!r} has {values.size} values for "
                    f"{self.source_grid.size} positions."
                )
            values = np.array(values, copy=True, order="C")
            values.setflags(write=False)
            owned[name] = values
        object.__setattr__(self, "values", MappingProxyType(owned))

    def to_dict(self) -> dict[str, Any]:
        """Return serialized values referencing shared grids."""
        return {
            "source_grid_id": self.source_grid.grid_id,
            "request_grid_id": self.request_grid.grid_id,
            "values": {name: self.values[name].tolist() for name in self.spec.fields},
        }


@dataclass(frozen=True)
class FallbackCollection:
    """Immutable provider specs, shared grids, and cached datasets."""

    version: int
    event_time: str | None
    time: np.ndarray = field(repr=False, compare=False)
    grids: Mapping[str, SampleGrid]
    providers: Mapping[str, ProviderSpec]
    datasets: Mapping[str, Mapping[str, ProviderDataset]]

    def __post_init__(self) -> None:
        """Validate and freeze the fallback collection."""
        version = int(self.version)
        time = np.asarray(self.time, dtype=float).reshape(-1)
        if time.size == 0 or not np.all(np.isfinite(time)):
            raise ValueError("Fallback time must contain finite values.")
        time = np.array(time, copy=True, order="C")
        time.setflags(write=False)

        grids = dict(self.grids)
        providers = dict(self.providers)
        if set(providers) != set(self.datasets):
            missing = set(providers).symmetric_difference(self.datasets)
            raise ValueError(
                "Fallback provider/dataset keys differ: " + ", ".join(sorted(missing))
            )

        normalized: dict[str, Mapping[str, ProviderDataset]] = {}
        for provider_key, provider_datasets in self.datasets.items():
            spec = providers[provider_key]
            current = dict(provider_datasets)
            if not current:
                raise ValueError(f"Fallback provider {provider_key!r} has no datasets.")
            for source_grid_id, dataset in current.items():
                if dataset.spec != spec:
                    raise ValueError("Dataset uses another provider specification.")
                if source_grid_id != dataset.source_grid.grid_id:
                    raise ValueError("Dataset mapping key must be source_grid_id.")
                if grids.get(dataset.source_grid.grid_id) is not dataset.source_grid:
                    raise ValueError("Dataset source grid must be the collection's shared object.")
                if grids.get(dataset.request_grid.grid_id) is not dataset.request_grid:
                    raise ValueError("Dataset request grid must be the collection's shared object.")
            normalized[provider_key] = MappingProxyType(current)

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "grids", MappingProxyType(grids))
        object.__setattr__(self, "providers", MappingProxyType(providers))
        object.__setattr__(self, "datasets", MappingProxyType(normalized))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, expected_version: int
    ) -> FallbackCollection:
        """Construct a collection while sharing its grid objects."""
        version = int(payload.get("version", 0))
        if version != expected_version:
            raise ValueError(f"Expected fallback schema {expected_version}, got {version}.")

        grids = {
            str(grid_id): SampleGrid.from_dict(str(grid_id), grid_payload)
            for grid_id, grid_payload in payload.get("grids", {}).items()
        }
        providers: dict[str, ProviderSpec] = {}
        datasets: dict[str, dict[str, ProviderDataset]] = {}

        for provider_key, provider_payload in payload.get("providers", {}).items():
            provider_key = str(provider_key)
            spec = ProviderSpec.from_dict(provider_payload["spec"])
            if spec.key != provider_key:
                raise ValueError("Provider key and serialized spec key differ.")
            providers[provider_key] = spec
            provider_datasets: dict[str, ProviderDataset] = {}
            for dataset_payload in provider_payload.get("datasets", []):
                source_grid_id = str(dataset_payload["source_grid_id"])
                request_grid_id = str(dataset_payload["request_grid_id"])
                try:
                    source_grid = grids[source_grid_id]
                    request_grid = grids[request_grid_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Provider {provider_key!r} references an unknown grid."
                    ) from exc
                provider_datasets[source_grid_id] = ProviderDataset(
                    spec=spec,
                    source_grid=source_grid,
                    request_grid=request_grid,
                    values=dataset_payload["values"],
                )
            datasets[provider_key] = provider_datasets

        return cls(
            version=version,
            event_time=payload.get("event_time"),
            time=np.asarray(payload.get("time", [0.0])),
            grids=grids,
            providers=providers,
            datasets=datasets,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return normalized serialized collection data."""
        return {
            "version": self.version,
            "event_time": self.event_time,
            "time": self.time.tolist(),
            "grids": {grid_id: grid.to_dict() for grid_id, grid in self.grids.items()},
            "providers": {
                provider_key: {
                    "spec": spec.to_dict(),
                    "datasets": [
                        dataset.to_dict() for dataset in self.datasets[provider_key].values()
                    ],
                }
                for provider_key, spec in self.providers.items()
            },
        }

    @classmethod
    def read(cls, path, *, expected_version: int) -> FallbackCollection:
        """Read a JSON fallback collection."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_payload(payload, expected_version=expected_version)

    def write(self, path, *, indent: int | None = 2) -> None:
        """Write a JSON fallback collection."""
        path.write_text(
            json.dumps(self.to_payload(), indent=indent, sort_keys=True, ensure_ascii=False)
            + ("\n" if indent is not None else ""),
            encoding="utf-8",
        )
