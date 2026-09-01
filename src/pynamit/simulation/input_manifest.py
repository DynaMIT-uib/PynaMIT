"""File contract for reusable prepared-input packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.schema import INPUT_DATASET_KEYS, SIMULATION_ARTIFACT_NAMES
from pynamit.storage import ArtifactStore

INPUT_MANIFEST_FILENAME = "pynamit_input_manifest.json"
_INPUT_MANIFEST_VERSION = 7

_INPUT_PROJECTION_SETTING_KEYS = (
    "Nmax",
    "Mmax",
    "Ncs",
    "RI",
    "boundary_jr_projection_basis",
    "boundary_Br_projection_basis",
    "conductance_projection_basis",
    "u_projection_basis",
    "Q_eff_projection_basis",
    "E_neutral_wind_projection_basis",
    "horizontal_basis_kind",
    "area_weighted_least_squares",
)

_INPUT_MAINFIELD_SETTING_KEYS = ("main_field_kind", "main_field_epoch", "main_field_B0")
_INPUT_GEOMETRY_SETTING_KEYS = ("RM",) + _INPUT_MAINFIELD_SETTING_KEYS
_INPUT_DATASET_REQUIREMENT_KEYS = {"boundary_Br": ("RM",)}


def input_projection_settings(config_or_settings: Any) -> dict[str, Any]:
    """Return settings defining prepared input coefficient space."""
    config = SimulationConfig.from_settings(config_or_settings)
    return {name: getattr(config, name) for name in _INPUT_PROJECTION_SETTING_KEYS}


def input_geometry_settings(config_or_settings: Any) -> dict[str, Any]:
    """Return geometry settings stored with prepared inputs."""
    config = SimulationConfig.from_settings(config_or_settings)
    geometry = {name: getattr(config, name) for name in _INPUT_GEOMETRY_SETTING_KEYS}
    geometry["horizontal_coordinate_system"] = config.horizontal_coordinate_system
    geometry["input_time_origin"] = config.t0
    return geometry


def input_dataset_requirements(
    input_datasets: list[str] | tuple[str, ...],
) -> dict[str, list[str]]:
    """Return settings required by prepared datasets."""
    datasets = [str(key) for key in input_datasets]
    return {
        key: list(_INPUT_DATASET_REQUIREMENT_KEYS[key])
        for key in datasets
        if key in _INPUT_DATASET_REQUIREMENT_KEYS
    }


def prepared_input_contract(
    settings: Any, input_datasets: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Return the serialized contract for a prepared input package."""
    datasets = [str(key) for key in input_datasets]
    return {
        "coefficient_space": input_projection_settings(settings),
        "geometry": input_geometry_settings(settings),
        "input_datasets": datasets,
        "dataset_requirements": input_dataset_requirements(datasets),
    }


def _setting_json_value(value: Any) -> Any:
    """Return a JSON-serializable setting value."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def available_prepared_inputs(
    directory: str | Path, *, artifact_storage: str = "auto"
) -> tuple[str, ...]:
    """Return the input datasets stored in a prepared package."""
    directory = ArtifactStore.require_artifact_directory(directory, ("settings",))
    store = ArtifactStore(directory, preferred_dataset_storage=artifact_storage)
    artifacts = store.scan_artifacts(INPUT_DATASET_KEYS)
    return tuple(key for key in INPUT_DATASET_KEYS if key in artifacts)


def clear_prepared_input_package(
    directory: str | Path, *, artifact_storage: str = "auto"
) -> tuple[str, ...]:
    """Remove PynaMIT artifacts before rewriting an input package.

    Input setters replace matching time coordinates. Clearing first
    prevents stale forcing rows and obsolete input streams in a newly
    projected package.
    """
    directory = Path(directory).resolve()
    store = ArtifactStore(directory, preferred_dataset_storage=artifact_storage)
    artifact_names = tuple(sorted(store.scan_artifacts(SIMULATION_ARTIFACT_NAMES)))
    for name in artifact_names:
        store.remove_artifact(name)
    (directory / INPUT_MANIFEST_FILENAME).unlink(missing_ok=True)
    return artifact_names


def write_input_manifest(
    directory: str | Path,
    settings: Any,
    *,
    input_datasets: list[str] | tuple[str, ...],
    source: str,
    notes: list[str] | tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and return a prepared-input manifest."""
    config = SimulationConfig.from_settings(settings)
    manifest = {
        "kind": "pynamit_prepared_inputs",
        "version": _INPUT_MANIFEST_VERSION,
        "source": source,
        "input_contract": prepared_input_contract(config, input_datasets),
        "notes": list(notes),
        "metadata": dict(metadata or {}),
    }
    path = Path(directory) / INPUT_MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_setting_json_value) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_input_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Read a prepared-input manifest if present."""
    path = Path(directory).resolve() / INPUT_MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_input_contract(manifest: dict[str, Any], directory: str | Path) -> dict[str, Any]:
    """Return the contract from a supported prepared-input manifest."""
    if manifest.get("kind") != "pynamit_prepared_inputs":
        raise ValueError(
            f"{INPUT_MANIFEST_FILENAME} in {directory!s} is not a PynaMIT prepared-input manifest."
        )
    if manifest.get("version") != _INPUT_MANIFEST_VERSION:
        raise ValueError(
            f"{INPUT_MANIFEST_FILENAME} in {directory!s} uses unsupported version "
            f"{manifest.get('version')!r}; expected {_INPUT_MANIFEST_VERSION}."
        )

    stored_contract = manifest.get("input_contract")
    if not isinstance(stored_contract, dict):
        raise ValueError(
            f"{INPUT_MANIFEST_FILENAME} in {directory!s} has no valid input contract."
        )
    return stored_contract


def _validate_manifest_artifacts(
    directory: str | Path,
    manifest_inputs: list[str],
    available_inputs: list[str],
    *,
    allow_unlisted: bool,
) -> None:
    """Validate stored artifacts against the manifest's dataset list."""
    missing = [key for key in manifest_inputs if key not in available_inputs]
    unlisted = [key for key in available_inputs if key not in manifest_inputs]
    if not missing and (allow_unlisted or not unlisted):
        return

    details = []
    if missing:
        details.append(f"listed but missing: {missing}")
    if unlisted and not allow_unlisted:
        details.append(f"stored but not listed: {unlisted}")
    raise ValueError(
        f"Prepared input artifacts in {directory!s} do not match "
        f"{INPUT_MANIFEST_FILENAME}: " + "; ".join(details)
    )


def validate_input_manifest(
    directory: str | Path,
    settings: Any | None = None,
    *,
    available_inputs: list[str] | tuple[str, ...] | None = None,
    allow_unlisted: bool = False,
    require: bool = False,
) -> dict[str, Any] | None:
    """Validate a manifest against its settings and artifacts."""
    manifest = read_input_manifest(directory)
    if manifest is None:
        if require:
            raise ValueError(
                f"No {INPUT_MANIFEST_FILENAME} found in prepared input directory {directory!s}."
            )
        return None

    stored_contract = _validated_input_contract(manifest, directory)
    manifest_inputs = [str(key) for key in stored_contract.get("input_datasets", [])]
    if available_inputs is None:
        available_inputs = available_prepared_inputs(directory)
    available_inputs = [str(key) for key in available_inputs]
    _validate_manifest_artifacts(
        directory, manifest_inputs, available_inputs, allow_unlisted=allow_unlisted
    )

    if settings is not None:
        expected_contract = prepared_input_contract(settings, manifest_inputs)
        if stored_contract != expected_contract:
            raise ValueError(
                f"{INPUT_MANIFEST_FILENAME} in {directory!s} does not match the settings "
                "artifact input contract."
            )
    return manifest


def validate_prepared_input_compatibility(
    input_settings: Any,
    simulation_settings: Any,
    *,
    input_datasets: list[str] | tuple[str, ...] | None = None,
    input_directory: str | Path | None = None,
) -> None:
    """Raise if prepared coefficients cannot be used by a simulation."""
    input_config = SimulationConfig.from_settings(input_settings)
    simulation_config = SimulationConfig.from_settings(simulation_settings)
    input_projection = input_projection_settings(input_config)
    simulation_projection = input_projection_settings(simulation_config)
    mismatches = {
        name: (input_projection[name], simulation_projection[name])
        for name in _INPUT_PROJECTION_SETTING_KEYS
        if input_projection[name] != simulation_projection[name]
    }
    for name in _INPUT_MAINFIELD_SETTING_KEYS:
        input_value = getattr(input_config, name)
        simulation_value = getattr(simulation_config, name)
        if not np.array_equal(input_value, simulation_value):
            mismatches[name] = (input_value, simulation_value)
    if input_config.t0 != simulation_config.t0:
        mismatches["input_time_origin"] = (input_config.t0, simulation_config.t0)

    if input_datasets is not None:
        required = {
            name for keys in input_dataset_requirements(input_datasets).values() for name in keys
        }
        for name in sorted(required):
            input_value = getattr(input_config, name)
            simulation_value = getattr(simulation_config, name)
            if not np.array_equal(input_value, simulation_value):
                mismatches[name] = (input_value, simulation_value)

    if mismatches:
        prefix = "" if input_directory is None else f"Prepared inputs in {input_directory!s} "
        details = ", ".join(
            f"{name}: input={left!r}, simulation={right!r}"
            for name, (left, right) in sorted(mismatches.items())
        )
        raise ValueError(prefix + "do not match the simulation input contract: " + details)


__all__ = [
    "INPUT_MANIFEST_FILENAME",
    "available_prepared_inputs",
    "clear_prepared_input_package",
    "input_dataset_requirements",
    "input_geometry_settings",
    "input_projection_settings",
    "prepared_input_contract",
    "read_input_manifest",
    "validate_input_manifest",
    "validate_prepared_input_compatibility",
    "write_input_manifest",
]
