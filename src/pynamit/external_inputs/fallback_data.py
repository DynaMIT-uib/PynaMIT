"""Bundled fallback values and their physical provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from pynamit.external_inputs.coordinates import SampleGrid
from pynamit.external_inputs.provider_definitions import InputProviderSpec

FALLBACK_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class ProviderSnapshot:
    """Provider values bound to geographic and library-request grids."""

    spec: InputProviderSpec
    geographic_grid: SampleGrid
    request_grid: SampleGrid
    values: Mapping[str, np.ndarray] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate grid bindings and own the provider values."""
        if self.geographic_grid.coordinate_convention != (self.spec.output_coordinate_convention):
            raise ValueError("Provider output and geographic-grid conventions differ.")
        if self.request_grid.coordinate_convention != (self.spec.request_coordinate_convention):
            raise ValueError("Provider request and request-grid conventions differ.")
        if self.geographic_grid.size != self.request_grid.size:
            raise ValueError("Geographic and request grids must have equal sizes.")

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
            if values.size != self.geographic_grid.size:
                raise ValueError(
                    f"Provider field {name!r} has {values.size} values for "
                    f"{self.geographic_grid.size} positions."
                )
            values = np.array(values, copy=True, order="C")
            values.setflags(write=False)
            owned[name] = values
        object.__setattr__(self, "values", MappingProxyType(owned))

    def to_dict(self) -> dict[str, Any]:
        """Return serialized values referencing shared grids."""
        return {
            "geographic_grid_id": self.geographic_grid.grid_id,
            "request_grid_id": self.request_grid.grid_id,
            "values": {name: self.values[name].tolist() for name in self.spec.fields},
        }


@dataclass(frozen=True)
class FallbackCollection:
    """Immutable fallback provider data and its physical conditions."""

    version: int
    event_time: str | None
    time: np.ndarray = field(repr=False, compare=False)
    grids: Mapping[str, SampleGrid]
    providers: Mapping[str, InputProviderSpec]
    datasets: Mapping[str, Mapping[str, ProviderSnapshot]]
    conditions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

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
        conditions = {
            str(provider_key): MappingProxyType(
                {
                    str(name): tuple(value) if isinstance(value, list) else value
                    for name, value in parameters.items()
                }
            )
            for provider_key, parameters in self.conditions.items()
        }
        if set(providers) != set(self.datasets):
            missing = set(providers).symmetric_difference(self.datasets)
            raise ValueError(
                "Fallback provider/dataset keys differ: " + ", ".join(sorted(missing))
            )

        normalized: dict[str, Mapping[str, ProviderSnapshot]] = {}
        for provider_key, provider_datasets in self.datasets.items():
            spec = providers[provider_key]
            current = dict(provider_datasets)
            if not current:
                raise ValueError(f"Fallback provider {provider_key!r} has no datasets.")
            for geographic_grid_id, dataset in current.items():
                if dataset.spec != spec:
                    raise ValueError("Dataset uses another provider specification.")
                if geographic_grid_id != dataset.geographic_grid.grid_id:
                    raise ValueError("Dataset mapping key must be geographic_grid_id.")
                if grids.get(dataset.geographic_grid.grid_id) is not dataset.geographic_grid:
                    raise ValueError(
                        "Dataset geographic grid must be the collection's shared object."
                    )
                if grids.get(dataset.request_grid.grid_id) is not dataset.request_grid:
                    raise ValueError(
                        "Dataset request grid must be the collection's shared object."
                    )
            normalized[provider_key] = MappingProxyType(current)

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "grids", MappingProxyType(grids))
        object.__setattr__(self, "providers", MappingProxyType(providers))
        object.__setattr__(self, "datasets", MappingProxyType(normalized))
        object.__setattr__(self, "conditions", MappingProxyType(conditions))

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
        providers: dict[str, InputProviderSpec] = {}
        datasets: dict[str, dict[str, ProviderSnapshot]] = {}

        for provider_key, provider_payload in payload.get("providers", {}).items():
            provider_key = str(provider_key)
            spec = InputProviderSpec.from_dict(provider_payload["spec"])
            if spec.key != provider_key:
                raise ValueError("Provider key and serialized spec key differ.")
            providers[provider_key] = spec
            provider_datasets: dict[str, ProviderSnapshot] = {}
            for dataset_payload in provider_payload.get("datasets", []):
                geographic_grid_id = str(dataset_payload["geographic_grid_id"])
                request_grid_id = str(dataset_payload["request_grid_id"])
                try:
                    geographic_grid = grids[geographic_grid_id]
                    request_grid = grids[request_grid_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Provider {provider_key!r} references an unknown grid."
                    ) from exc
                provider_datasets[geographic_grid_id] = ProviderSnapshot(
                    spec=spec,
                    geographic_grid=geographic_grid,
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
            conditions=payload.get("conditions", {}),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return normalized serialized collection data."""
        return {
            "version": self.version,
            "event_time": self.event_time,
            "conditions": {
                provider_key: dict(parameters)
                for provider_key, parameters in self.conditions.items()
            },
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
