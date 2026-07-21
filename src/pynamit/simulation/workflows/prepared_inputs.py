"""Prepared-input packages for PynaMIT simulations.

The input package boundary is intentionally narrower than a full run
directory.  It stores projected input coefficients on a chosen grid,
basis, and Earth-fixed main-field coordinate system. The time origin
locates those coefficients in physical time; it is not part of the
spatial frame. Most evolution choices belong to the consuming run, but
boundary ``Br`` also declares the magnetospheric radius it requires.
"""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.external_inputs import (
    get_conductance_inputs,
    get_input_source,
    get_jr_inputs,
    get_wind_inputs,
)
from pynamit.geomagnetism import MainField, decimal_year
from pynamit.math.constants import RE
from pynamit.simulation.api import Simulation
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.schema import INPUT_DATASET_KEYS, RUN_ARTIFACT_NAMES
from pynamit.storage import ArtifactStore, FieldTimeSeries

INPUT_MANIFEST_FILENAME = "pynamit_input_manifest.json"
RUN_MANIFEST_FILENAME = "pynamit_run_manifest.json"
_INPUT_MANIFEST_VERSION = 3

_INPUT_PROJECTION_SETTING_KEYS = (
    "Nmax",
    "Mmax",
    "Ncs",
    "RI",
    "jr_projection_basis",
    "Br_projection_basis",
    "resistance_projection_basis",
    "u_projection_basis",
    "Q_eff_projection_basis",
    "E_source_projection_basis",
    "horizontal_basis_kind",
    "area_weighted_least_squares",
)

_RUN_SETTING_KEYS = (
    "RM",
    "magnetic_boundary_shielding",
    "interhemispheric_coupling_latitude",
    "enable_pfac_coupling",
    "enable_interhemispheric_coupling",
    "fac_integration_radii",
    "interhemispheric_electric_field_weight",
    "main_field_kind",
    "main_field_epoch",
    "main_field_B0",
    "save_steady_states",
    "integrator",
    "least_squares_solver",
    "least_squares_preconditioner",
    "reuse_preconditioner",
    "m_imp_regularization_lambda",
)

_INPUT_MAINFIELD_SETTING_KEYS = ("main_field_kind", "main_field_epoch", "main_field_B0")
_INPUT_GEOMETRY_SETTING_KEYS = ("RM",) + _INPUT_MAINFIELD_SETTING_KEYS
_INPUT_DATASET_REQUIREMENT_KEYS = {"Br": ("RM",)}
_DEFAULT_INPUT_TIME = _datetime.datetime(2001, 5, 12, 21, 45)


def _wind_to_model_coordinates(main_field, u_theta, u_phi, lat, lon):
    """Rotate geographic wind samples into model coordinates."""
    u_theta, u_phi = np.broadcast_arrays(np.asarray(u_theta), np.asarray(u_phi))
    lat = np.asarray(lat).reshape(-1)
    lon = np.asarray(lon).reshape(-1)
    if lat.size != lon.size or u_theta.shape[-1] != lat.size:
        raise ValueError("Wind coordinates must match the final wind-sample dimension.")

    model_lat, model_lon = main_field.geo_to_model_coordinates(lat, lon)
    vector_lat = np.broadcast_to(lat, u_theta.shape)
    vector_lon = np.broadcast_to(lon, u_theta.shape)
    _, _, model_east, model_north = main_field.geo_to_model_coordinates(
        vector_lat, vector_lon, east=u_phi, north=-u_theta
    )
    return -model_north, model_east, model_lat, model_lon


def _empirical_dipole_coordinates_for_model_grid(main_field, event_time, model_lat, model_lon):
    """Return event-dipole coordinates for positions on a model grid.

    The native Hardy and AMPS inputs are evaluated in centered-dipole
    magnetic coordinates tied to the event epoch. The simulation grid
    may instead use IGRF/GEO or a centered dipole from another epoch,
    so positions must pass through GEO before model evaluation.
    """
    geo_lat, geo_lon = main_field.model_to_geo_coordinates(model_lat, model_lon)
    empirical_dipole = MainField(
        kind="dipole",
        epoch=decimal_year(event_time),
        ionosphere_height_km=main_field.ionosphere_height_km,
    )
    return empirical_dipole.geo_to_model_coordinates(geo_lat, geo_lon, event_time=event_time)


def input_projection_settings(config_or_settings: Any) -> dict[str, Any]:
    """Return settings defining prepared input coefficient space."""
    config = (
        config_or_settings
        if isinstance(config_or_settings, SimulationConfig)
        else SimulationConfig.from_settings(config_or_settings)
    )
    return {name: getattr(config, name) for name in _INPUT_PROJECTION_SETTING_KEYS}


def input_geometry_settings(config_or_settings: Any) -> dict[str, Any]:
    """Return geometry settings stored with prepared inputs."""
    config = (
        config_or_settings
        if isinstance(config_or_settings, SimulationConfig)
        else SimulationConfig.from_settings(config_or_settings)
    )
    geometry = {name: getattr(config, name) for name in _INPUT_GEOMETRY_SETTING_KEYS}
    geometry["input_time_origin"] = config.t0
    return geometry


def input_dataset_requirements(
    input_datasets: list[str] | tuple[str, ...],
) -> dict[str, list[str]]:
    """Return run-setting requirements implied by prepared datasets."""
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


def _plain_json_value(value: Any) -> Any:
    """Return a JSON-serializable version of a setting value."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _settings_equal(left: Any, right: Any) -> bool:
    """Return whether two normalized setting values are equal."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return left == right


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_plain_json_value) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON sidecar if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _input_directory(directory: str | Path | None) -> str:
    """Return a resolved input-package directory."""
    if directory is None:
        return ArtifactStore.create_temporary_directory("simulation/inputs")
    return str(Path(directory).resolve())


def _run_directory(directory: str | Path | None) -> str:
    """Return a resolved output run directory."""
    if directory is None:
        return ArtifactStore.create_temporary_directory("simulation/runs")
    return str(Path(directory).resolve())


def _available_input_datasets(store: ArtifactStore) -> list[str]:
    """Return prepared input datasets present in one directory."""
    artifacts = store.scan_artifacts(INPUT_DATASET_KEYS)
    return [key for key in INPUT_DATASET_KEYS if key in artifacts]


def clear_prepared_input_package(
    directory: str | Path, *, artifact_storage: str = "auto"
) -> tuple[str, ...]:
    """Remove generated artifacts before rewriting an input package.

    Input setters only replace matching time coordinates. Clearing first
    prevents stale forcing rows and obsolete input streams in a newly
    projected package.
    """
    directory = Path(directory).resolve()
    store = ArtifactStore(directory, preferred_dataset_storage=artifact_storage)
    artifact_names = tuple(sorted(store.scan_artifacts(RUN_ARTIFACT_NAMES)))
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
    input_datasets = list(input_datasets)
    manifest = {
        "kind": "pynamit_prepared_inputs",
        "version": _INPUT_MANIFEST_VERSION,
        "source": source,
        "input_contract": prepared_input_contract(config, input_datasets),
        "notes": list(notes),
        "metadata": dict(metadata or {}),
    }
    _write_json(Path(directory) / INPUT_MANIFEST_FILENAME, manifest)
    return manifest


def read_input_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Read a prepared-input manifest if present."""
    return _read_json(Path(directory).resolve() / INPUT_MANIFEST_FILENAME)


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
    """Validate a prepared-input manifest against stored artifacts.

    When a manifest is present it is treated as the package contract:
    listed datasets must exist, and the serialized contract must match
    the settings artifact when supplied.  Extra stored artifacts are
    rejected by default; callers that explicitly select a subset may
    allow them.  Set ``require=True`` for code paths that should not
    accept legacy packages without a contract.
    """
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
        available_inputs = _available_input_datasets(ArtifactStore(directory))
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
    run_settings: Any,
    *,
    input_datasets: list[str] | tuple[str, ...] | None = None,
    input_directory: str | Path | None = None,
) -> None:
    """Raise if prepared input coefficients cannot be used by a run.

    The baseline check covers the coefficient/grid space, main-field
    definition, and physical input time origin. Boundary ``Br`` also
    needs the same ``RM``.  PFAC treatment, hemisphere coupling,
    low-latitude boundary, shielding, and integrator remain run choices.
    """
    input_config = SimulationConfig.from_settings(input_settings)
    run_config = SimulationConfig.from_settings(run_settings)
    input_projection = input_projection_settings(input_config)
    run_projection = input_projection_settings(run_config)
    mismatches = {}
    for name in _INPUT_PROJECTION_SETTING_KEYS:
        if input_projection[name] != run_projection[name]:
            mismatches[name] = (input_projection[name], run_projection[name])
    for name in _INPUT_MAINFIELD_SETTING_KEYS:
        input_value = getattr(input_config, name)
        run_value = getattr(run_config, name)
        if not _settings_equal(input_value, run_value):
            mismatches[name] = (input_value, run_value)
    if not _settings_equal(input_config.t0, run_config.t0):
        mismatches["input_time_origin"] = (input_config.t0, run_config.t0)

    if input_datasets is not None:
        required = {
            name for keys in input_dataset_requirements(input_datasets).values() for name in keys
        }
        for name in sorted(required):
            input_value = getattr(input_config, name)
            run_value = getattr(run_config, name)
            if not _settings_equal(input_value, run_value):
                mismatches[name] = (input_value, run_value)

    if mismatches:
        prefix = "" if input_directory is None else f"Prepared inputs in {input_directory!s} "
        details = ", ".join(
            f"{name}: input={left!r}, run={right!r}"
            for name, (left, right) in sorted(mismatches.items())
        )
        raise ValueError(prefix + "do not match the run input contract: " + details)


def _validate_and_select_prepared_inputs(
    input_directory: str | Path,
    *,
    artifact_storage: str,
    enabled_inputs: tuple[str, ...] | list[str] | None,
) -> tuple[str, ArtifactStore, Any, list[str], dict[str, Any]]:
    """Validate a package and return its selected input streams."""
    input_directory = ArtifactStore.require_artifact_directory(input_directory, ("settings",))
    input_store = ArtifactStore(input_directory, preferred_dataset_storage=artifact_storage)
    input_settings = input_store.load_dataset("settings")

    available_inputs = _available_input_datasets(input_store)
    allowed = None if enabled_inputs is None else set(enabled_inputs)
    unknown = set() if allowed is None else allowed - set(INPUT_DATASET_KEYS)
    if unknown:
        raise ValueError(f"Unknown input dataset key(s): {sorted(unknown)}.")
    if allowed is not None:
        missing_requested = sorted(allowed - set(available_inputs))
        if missing_requested:
            raise ValueError(
                f"Requested prepared input dataset(s) are not available in {input_directory!r}: "
                f"{missing_requested}."
            )
    manifest = validate_input_manifest(
        input_directory,
        input_settings,
        available_inputs=available_inputs,
        allow_unlisted=allowed is not None,
        require=True,
    )
    if manifest is None:
        raise RuntimeError("Prepared-input validation returned no manifest.")
    selected_inputs = (
        available_inputs
        if allowed is None
        else [key for key in available_inputs if key in allowed]
    )
    if not selected_inputs:
        raise ValueError(f"No prepared input datasets found in {input_directory!r}.")
    active_wind_forcings = sorted(set(selected_inputs) & {"u", "Q_eff"})
    if len(active_wind_forcings) > 1:
        raise ValueError(
            "Prepared input selection contains mutually exclusive wind-forcing "
            f"representations {active_wind_forcings}; enable only one of 'u' or "
            "'Q_eff' for a run."
        )
    return input_directory, input_store, input_settings, selected_inputs, manifest


def _copy_prepared_inputs(
    simulation: Simulation, input_store: ArtifactStore, selected_inputs: list[str]
) -> list[str]:
    """Copy validated input streams into a simulation-owned store."""
    series = FieldTimeSeries(
        simulation.run_data.schema.input_field_spaces, simulation.run_data.schema.input_variables
    )
    for key in selected_inputs:
        series.load(key, input_store)

    loaded = [key for key in INPUT_DATASET_KEYS if key in series.datasets]
    for key in INPUT_DATASET_KEYS:
        if key not in loaded:
            simulation.run_data.artifact_store.remove_artifact(key)
    for key in loaded:
        simulation.run_data.artifact_store.save_dataset(series.datasets[key].reset_index("i"), key)

    # Reload through the consuming run's artifact store. Lazy Zarr
    # arrays must not leave an active simulation dependent on the
    # preparation package.
    run_series = FieldTimeSeries(
        simulation.run_data.schema.input_field_spaces, simulation.run_data.schema.input_variables
    )
    run_series.load_all(simulation.run_data.artifact_store)
    simulation.run_data.input_series = run_series
    return loaded


def _validate_run_identity(
    run_directory: str | Path,
    *,
    input_directory: str,
    selected_inputs: list[str],
    input_manifest: dict[str, Any],
    evolution_policy: dict[str, Any],
) -> None:
    """Require a trajectory to keep one identity."""
    existing = _read_json(Path(run_directory) / RUN_MANIFEST_FILENAME)
    if existing is None:
        return
    if (
        not isinstance(existing, dict)
        or existing.get("kind") not in {"pynamit_run", "pynamit_paper_run"}
        or existing.get("version") != 2
    ):
        raise ValueError(
            f"Existing {RUN_MANIFEST_FILENAME} in {run_directory!s} does not describe "
            "a compatible resumable run. Use a new run directory."
        )

    existing_evolution = dict(existing.get("time_evolution", {}))
    existing_evolution.pop("final_time", None)
    expected = {
        "input_directory": str(Path(input_directory).resolve()),
        "enabled_inputs": selected_inputs,
        "input_manifest": input_manifest,
        "time_evolution": evolution_policy,
    }
    actual = {**existing, "time_evolution": existing_evolution}
    mismatches = [name for name, value in expected.items() if actual.get(name) != value]
    if mismatches:
        raise ValueError(
            f"Existing run in {run_directory!s} has a different trajectory identity "
            f"({', '.join(mismatches)}). Use a new run directory for a different "
            "input package, selection, or evolution policy."
        )


def load_prepared_inputs_into_simulation(
    simulation: Simulation,
    input_directory: str | Path,
    *,
    artifact_storage: str = "auto",
    enabled_inputs: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Validate and copy inputs into an existing simulation run."""
    input_directory, input_store, input_settings, selected_inputs, _ = (
        _validate_and_select_prepared_inputs(
            input_directory, artifact_storage=artifact_storage, enabled_inputs=enabled_inputs
        )
    )

    validate_prepared_input_compatibility(
        input_settings,
        simulation.run_data.config,
        input_datasets=selected_inputs,
        input_directory=input_directory,
    )
    return _copy_prepared_inputs(simulation, input_store, selected_inputs)


def prepare_pynamit_inputs(
    input_directory=None,
    *,
    final_time=100,
    Nmax=20,
    Mmax=20,
    Ncs=30,
    main_field_kind="dipole",
    main_field_epoch=2020,
    main_field_B0=None,
    use_wind=False,
    use_Q_eff=False,
    use_jr=True,
    jr_projection_basis=None,
    Br_projection_basis=None,
    resistance_projection_basis=None,
    u_projection_basis=None,
    Q_eff_projection_basis=None,
    jr_lambda=None,
    conductance_lambda=None,
    u_lambda=None,
    Q_eff_lambda=None,
    multi_data=False,
    artifact_storage="auto",
    horizontal_basis_kind="SH",
    area_weighted_least_squares=False,
):
    """Prepare default example inputs without evolving simulation.

    This is the split-out input-projection half of ``run_pynamit``.  It
    writes the projected input datasets plus a small manifest to
    ``input_directory`` and returns the ``Simulation`` instance that
    owns the package.

    ``main_field_*`` is part of the prepared-input contract because
    inputs may be projected in model magnetic coordinates or converted
    from FAC to ``jr`` using the main field.
    """
    if use_Q_eff and not use_wind:
        raise ValueError("use_Q_eff=True requires use_wind=True in prepare_pynamit_inputs.")

    event_time = _DEFAULT_INPUT_TIME
    input_directory = _input_directory(input_directory)
    clear_prepared_input_package(input_directory, artifact_storage=artifact_storage)
    simulation = Simulation(
        run_directory=input_directory,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=RE + 110.0e3,
        main_field_kind=main_field_kind,
        main_field_epoch=main_field_epoch,
        main_field_B0=main_field_B0,
        t0=event_time.isoformat(sep=" "),
        jr_projection_basis=jr_projection_basis,
        Br_projection_basis=Br_projection_basis,
        resistance_projection_basis=resistance_projection_basis,
        u_projection_basis=u_projection_basis,
        Q_eff_projection_basis=Q_eff_projection_basis,
        horizontal_basis_kind=horizontal_basis_kind,
        area_weighted_least_squares=area_weighted_least_squares,
        artifact_storage=artifact_storage,
        enable_pfac_coupling=False,
    )

    time = np.linspace(0, final_time, 4) if multi_data else None

    model_lat = simulation.geometry.model_grid.lat
    model_lon = simulation.geometry.model_grid.lon
    native_empirical_inputs = get_input_source() == "native"
    if native_empirical_inputs:
        query_lat, query_lon = _empirical_dipole_coordinates_for_model_grid(
            simulation.geometry.main_field, event_time, model_lat, model_lon
        )
    else:
        # Bundled fallback values are indexed directly by their stored
        # synthetic grid and are not an empirical coordinate model.
        query_lat, query_lon = model_lat, model_lon
    hall, pedersen, returned_lat, returned_lon = get_conductance_inputs(
        event_time, query_lat, query_lon, time
    )
    conductance_lat = model_lat if native_empirical_inputs else returned_lat
    conductance_lon = model_lon if native_empirical_inputs else returned_lon
    simulation.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        reg_lambda=conductance_lambda,
        time=time,
    )

    if use_jr:
        jr, returned_lat, returned_lon = get_jr_inputs(event_time, query_lat, query_lon, time)
        jr_lat = model_lat if native_empirical_inputs else returned_lat
        jr_lon = model_lon if native_empirical_inputs else returned_lon
        simulation.set_jr(jr, lat=jr_lat, lon=jr_lon, reg_lambda=jr_lambda, time=time)

    wind_inputs = get_wind_inputs(event_time, use_wind=use_wind, time=time)
    if wind_inputs is not None:
        u_theta, u_phi, u_lat, u_lon, weights = wind_inputs
        u_theta, u_phi, u_lat, u_lon = _wind_to_model_coordinates(
            simulation.geometry.main_field, u_theta, u_phi, u_lat, u_lon
        )
        if use_Q_eff:
            simulation.set_Q_eff_from_neutral_wind(
                u_theta=u_theta,
                u_phi=u_phi,
                lat=u_lat,
                lon=u_lon,
                sqrt_weights=weights,
                wind_reg_lambda=u_lambda,
                Q_eff_reg_lambda=Q_eff_lambda,
                time=time,
            )
        else:
            simulation.set_neutral_wind(
                u_theta=u_theta,
                u_phi=u_phi,
                lat=u_lat,
                lon=u_lon,
                sqrt_weights=weights,
                reg_lambda=u_lambda,
                time=time,
            )

    input_store = ArtifactStore(input_directory, preferred_dataset_storage=artifact_storage)
    notes = []
    if use_Q_eff:
        notes.append(
            "Q_eff was derived from neutral wind through the current model operators; "
            "prefer direct E_source inputs for externally prepared weighted winds."
        )
    write_input_manifest(
        input_directory,
        simulation.run_data.config,
        input_datasets=_available_input_datasets(input_store),
        source="pynamit.default_external_inputs",
        notes=notes,
        metadata={
            "external_input_source": get_input_source(),
            "multi_data": bool(multi_data),
            "projection_regularization": {
                "jr_lambda": jr_lambda,
                "conductance_lambda": conductance_lambda,
                "u_lambda": u_lambda,
                "Q_eff_lambda": Q_eff_lambda,
            },
        },
    )
    return simulation


def run_pynamit_from_inputs(
    input_directory,
    *,
    run_directory=None,
    enabled_inputs=None,
    final_time=100,
    sampling_step_interval=1,
    saving_sample_interval=200,
    dt=5e-4,
    RM=None,
    main_field_kind=None,
    fac_integration_radii=None,
    interhemispheric_electric_field_weight=None,
    enable_pfac_coupling=False,
    enable_interhemispheric_coupling=False,
    interhemispheric_coupling_latitude=50,
    steady_state_initialization=True,
    run_inductive=True,
    run_steady_state=True,
    integrator="euler",
    least_squares_solver=None,
    least_squares_preconditioner="pinv",
    reuse_preconditioner=False,
    m_imp_regularization_lambda=0.0,
    artifact_storage="auto",
    magnetic_boundary_shielding=False,
):
    """Run simulation from a prepared input package."""
    input_directory, input_store, input_settings, selected_inputs, input_manifest = (
        _validate_and_select_prepared_inputs(
            input_directory, artifact_storage=artifact_storage, enabled_inputs=enabled_inputs
        )
    )

    config_kwargs = SimulationConfig.from_settings(input_settings).to_kwargs()
    config_kwargs.update(
        {
            "enable_pfac_coupling": enable_pfac_coupling,
            "enable_interhemispheric_coupling": enable_interhemispheric_coupling,
            "interhemispheric_coupling_latitude": interhemispheric_coupling_latitude,
            "save_steady_states": run_steady_state,
            "integrator": integrator,
            "least_squares_solver": least_squares_solver,
            "least_squares_preconditioner": least_squares_preconditioner,
            "reuse_preconditioner": reuse_preconditioner,
            "m_imp_regularization_lambda": m_imp_regularization_lambda,
            "magnetic_boundary_shielding": magnetic_boundary_shielding,
        }
    )
    if RM is not None:
        config_kwargs["RM"] = RM
    if main_field_kind is not None:
        config_kwargs["main_field_kind"] = main_field_kind
    # PFAC integration is run geometry, not part of the prepared-input
    # coefficient contract. ``None`` deliberately asks SimulationConfig
    # to derive a radial grid for this run's RI/RM.
    config_kwargs["fac_integration_radii"] = fac_integration_radii
    if interhemispheric_electric_field_weight is not None:
        config_kwargs["interhemispheric_electric_field_weight"] = (
            interhemispheric_electric_field_weight
        )
    config = SimulationConfig(**config_kwargs)
    validate_prepared_input_compatibility(
        input_settings, config, input_datasets=selected_inputs, input_directory=input_directory
    )
    time_evolution = {
        "final_time": final_time,
        "dt": dt,
        "sampling_step_interval": sampling_step_interval,
        "saving_sample_interval": saving_sample_interval,
        "steady_state_initialization": steady_state_initialization,
        "run_inductive": run_inductive,
        "run_steady_state": run_steady_state,
    }
    evolution_policy = {
        name: value for name, value in time_evolution.items() if name != "final_time"
    }
    run_directory = _run_directory(run_directory)
    if Path(run_directory) == Path(input_directory):
        raise ValueError(
            "run_directory must differ from input_directory so the reusable "
            "prepared-input package remains immutable."
        )
    _validate_run_identity(
        run_directory,
        input_directory=input_directory,
        selected_inputs=selected_inputs,
        input_manifest=input_manifest,
        evolution_policy=evolution_policy,
    )
    simulation = Simulation.from_config(
        config, run_directory=run_directory, artifact_storage=artifact_storage
    )
    loaded_inputs = _copy_prepared_inputs(simulation, input_store, selected_inputs)

    _write_json(
        Path(run_directory) / RUN_MANIFEST_FILENAME,
        {
            "kind": "pynamit_run",
            "version": 2,
            "input_directory": str(Path(input_directory).resolve()),
            "enabled_inputs": loaded_inputs,
            "input_manifest": input_manifest,
            "run_settings": {
                name: _plain_json_value(getattr(simulation.config, name))
                for name in _RUN_SETTING_KEYS
            },
            "time_evolution": time_evolution,
        },
    )

    simulation.evolve_to_time(
        t=final_time,
        dt=dt,
        sampling_step_interval=sampling_step_interval,
        saving_sample_interval=saving_sample_interval,
        steady_state_initialization=steady_state_initialization,
        run_inductive=run_inductive,
        run_steady_state=run_steady_state,
    )
    return simulation


__all__ = [
    "INPUT_MANIFEST_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "input_projection_settings",
    "input_geometry_settings",
    "input_dataset_requirements",
    "prepared_input_contract",
    "clear_prepared_input_package",
    "write_input_manifest",
    "read_input_manifest",
    "validate_input_manifest",
    "validate_prepared_input_compatibility",
    "load_prepared_inputs_into_simulation",
    "prepare_pynamit_inputs",
    "run_pynamit_from_inputs",
]
