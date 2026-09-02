"""Coordinates and sampling grids for external empirical inputs."""

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
class CoordinateConvention:
    """Semantic interpretation of horizontal coordinates."""

    coordinate_system: str
    angular_units: str
    latitude_definition: str
    longitude_definition: str
    longitude_wrap: str
    reference_surface: ReferenceSurface

    def __post_init__(self) -> None:
        """Validate and normalize the coordinate convention."""
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
    def from_dict(cls, payload: Mapping[str, Any]) -> CoordinateConvention:
        """Construct a coordinate convention from metadata."""
        return cls(
            coordinate_system=str(payload["coordinate_system"]),
            angular_units=str(payload["angular_units"]),
            latitude_definition=str(payload["latitude_definition"]),
            longitude_definition=str(payload["longitude_definition"]),
            longitude_wrap=str(payload["longitude_wrap"]),
            reference_surface=ReferenceSurface.from_dict(payload["reference_surface"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serialized coordinate-convention metadata."""
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
        """Return a stable convention signature."""
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
        """Hash the convention and normalized ordered coordinates."""
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


PYNAMIT_SPHERICAL_GEO_110KM = CoordinateConvention(
    coordinate_system=GEOCENTRIC_GEOGRAPHIC,
    angular_units="degrees",
    latitude_definition="geocentric",
    longitude_definition="east_positive",
    longitude_wrap="[-180,180)",
    reference_surface=ReferenceSurface(
        kind="sphere", radius_m=float(EARTH_RADIUS_M + _IONOSPHERE_ALTITUDE_KM * 1e3)
    ),
)

PYNAMIT_CENTERED_DIPOLE_110KM = CoordinateConvention(
    coordinate_system=CENTERED_DIPOLE,
    angular_units="degrees",
    latitude_definition="centered_dipole",
    longitude_definition="east_positive",
    longitude_wrap="[-180,180)",
    reference_surface=ReferenceSurface(
        kind="sphere", radius_m=float(EARTH_RADIUS_M + _IONOSPHERE_ALTITUDE_KM * 1e3)
    ),
)

LIBRARY_GEOGRAPHIC_110KM = CoordinateConvention(
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
class SampleGrid:
    """Immutable ordered coordinates under one coordinate convention."""

    grid_id: str
    coordinate_convention: CoordinateConvention
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
        latitude, longitude = self.coordinate_convention.normalize(self.lat, self.lon)
        object.__setattr__(self, "grid_id", grid_id)
        object.__setattr__(self, "lat", latitude)
        object.__setattr__(self, "lon", longitude)
        object.__setattr__(self, "sampling_geometry", _freeze_mapping(self.sampling_geometry))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(
            self,
            "coordinate_identity",
            self.coordinate_convention.coordinate_identity(latitude, longitude),
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
            coordinate_convention=CoordinateConvention.from_dict(payload["coordinate_contract"]),
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
            "coordinate_contract": self.coordinate_convention.to_dict(),
            "sampling_geometry": _json_value(self.sampling_geometry),
            "provenance": _json_value(self.provenance),
            "coordinate_identity": self.coordinate_identity,
            "lat": self.lat.tolist(),
            "lon": self.lon.tolist(),
        }


def _spherical_geo_to_library_110km(
    geographic_grid: SampleGrid, target_convention: CoordinateConvention
) -> SampleGrid:
    """Create a library grid using the spherical identity map."""
    altitude = target_convention.reference_surface.altitude_km
    if altitude is None:
        raise ValueError("Library request convention requires altitude_km.")
    latitude, longitude, _ = spherical_geo_to_library_geographic(
        geographic_grid.lat, geographic_grid.lon, altitude
    )
    target_short = target_convention.signature[:12]
    return SampleGrid(
        grid_id=f"{geographic_grid.grid_id}--request-{target_short}",
        coordinate_convention=target_convention,
        lat=latitude,
        lon=longitude,
        sampling_geometry={
            "type": "mapped_ordered_points",
            "geographic_grid_id": geographic_grid.grid_id,
        },
        provenance={
            "geographic_coordinate_identity": geographic_grid.coordinate_identity,
            "mapping": "numerical_lat_lon_identity_at_equal_nominal_altitude",
        },
    )


class ExternalInputCoordinates:
    """Coordinate views of the same ordered physical locations."""

    def __init__(
        self,
        geographic_grid: SampleGrid,
        *,
        model_grid: SampleGrid | None = None,
        model_epoch: float | None = None,
    ):
        if geographic_grid.coordinate_convention != PYNAMIT_SPHERICAL_GEO_110KM:
            raise ValueError("geographic_grid must use geocentric geographic coordinates.")
        model_grid = geographic_grid if model_grid is None else model_grid
        if model_grid.size != geographic_grid.size:
            raise ValueError("Model and geographic grids must contain the same ordered samples.")
        if model_grid.coordinate_convention not in {
            PYNAMIT_SPHERICAL_GEO_110KM,
            PYNAMIT_CENTERED_DIPOLE_110KM,
        }:
            raise ValueError("model_grid uses an unsupported coordinate system.")
        if (
            model_grid.coordinate_convention == PYNAMIT_SPHERICAL_GEO_110KM
            and model_grid.coordinate_identity != geographic_grid.coordinate_identity
        ):
            raise ValueError("Geographic model and geographic grids must be identical.")
        if model_epoch is not None:
            model_epoch = float(model_epoch)
            if not np.isfinite(model_epoch):
                raise ValueError("model_epoch must be finite.")
        if (
            model_grid.coordinate_convention == PYNAMIT_CENTERED_DIPOLE_110KM
            and model_epoch is None
        ):
            raise ValueError("Centered-dipole model coordinates require model_epoch.")
        self.geographic_grid = geographic_grid
        self.model_grid = model_grid
        self.model_epoch = model_epoch
        self._sample_grids: dict[str, SampleGrid] = {
            geographic_grid.coordinate_convention.signature: geographic_grid,
            model_grid.coordinate_convention.signature: model_grid,
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
    ) -> ExternalInputCoordinates:
        """Construct coordinate views from spherical-GEO positions."""
        return cls(
            SampleGrid(
                grid_id=grid_id,
                coordinate_convention=PYNAMIT_SPHERICAL_GEO_110KM,
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
    ) -> ExternalInputCoordinates:
        """Construct model and GEO views of the same ordered samples."""
        coordinate_system = str(coordinate_system).strip().lower()
        try:
            model_convention = {
                CENTERED_DIPOLE: PYNAMIT_CENTERED_DIPOLE_110KM,
                GEOCENTRIC_GEOGRAPHIC: PYNAMIT_SPHERICAL_GEO_110KM,
            }[coordinate_system]
        except KeyError as exc:
            raise ValueError(
                "coordinate_system must be 'centered_dipole' or 'geocentric_geographic'."
            ) from exc

        geographic_grid = SampleGrid(
            grid_id=grid_id,
            coordinate_convention=PYNAMIT_SPHERICAL_GEO_110KM,
            lat=geographic_lat,
            lon=geographic_lon,
            sampling_geometry={} if sampling_geometry is None else sampling_geometry,
            provenance={} if provenance is None else provenance,
        )
        if model_convention == PYNAMIT_SPHERICAL_GEO_110KM:
            model_identity = model_convention.coordinate_identity(lat, lon)
            if model_identity != geographic_grid.coordinate_identity:
                raise ValueError("Geographic model coordinates must match geographic samples.")
            model_grid = geographic_grid
        else:
            model_grid = SampleGrid(
                grid_id=f"{grid_id}--model",
                coordinate_convention=model_convention,
                lat=lat,
                lon=lon,
                sampling_geometry={
                    "type": "coordinate_view",
                    "geographic_grid_id": geographic_grid.grid_id,
                },
                provenance={
                    **({} if provenance is None else provenance),
                    "model_epoch": float(model_epoch) if model_epoch is not None else None,
                    "geographic_coordinate_identity": geographic_grid.coordinate_identity,
                },
            )
        return cls(geographic_grid, model_grid=model_grid, model_epoch=model_epoch)

    def sample_grid(self, convention: CoordinateConvention) -> SampleGrid:
        """Return coordinates under the requested convention."""
        cached = self._sample_grids.get(convention.signature)
        if cached is not None:
            return cached

        if not (
            self.geographic_grid.coordinate_convention == PYNAMIT_SPHERICAL_GEO_110KM
            and convention == LIBRARY_GEOGRAPHIC_110KM
        ):
            raise ValueError(
                "No coordinate conversion is defined from "
                f"{self.geographic_grid.coordinate_convention.coordinate_system!r} "
                f"to {convention.coordinate_system!r}."
            )
        converted = _spherical_geo_to_library_110km(self.geographic_grid, convention)
        self._sample_grids[convention.signature] = converted
        return converted
