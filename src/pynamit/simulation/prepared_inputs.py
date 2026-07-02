"""Prepared-input packages for PynaMIT simulations.

The input package boundary is intentionally narrower than a full run
directory.  It stores projected input coefficients on a chosen grid,
basis, and main-field coordinate system.  For Kaiju/Geopack SM inputs
the coordinate time is the event time used for GEO-SM rotations.  Most
evolution choices belong to the consuming run, but boundary ``Br`` also
declares the magnetospheric radius it requires.
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
from pynamit.math.constants import RE
from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.dynamics import Dynamics

INPUT_MANIFEST_FILENAME = "pynamit_input_manifest.json"
RUN_MANIFEST_FILENAME = "pynamit_run_manifest.json"

INPUT_PROJECTION_SETTING_KEYS = (
    "Nmax",
    "Mmax",
    "Ncs",
    "RI",
    "jr_projection_basis",
    "Br_projection_basis",
    "conductance_projection_basis",
    "u_projection_basis",
    "Q_eff_projection_basis",
    "horizontal_basis_kind",
    "area_weighted_least_squares",
)

RUN_SETTING_KEYS = (
    "RM",
    "RM_shielding",
    "latitude_boundary",
    "ignore_PFAC",
    "connect_hemispheres",
    "FAC_integration_steps",
    "ih_constraint_scaling",
    "mainfield_kind",
    "mainfield_epoch",
    "mainfield_B0",
    "save_steady_states",
    "integrator",
    "least_squares_solver",
    "least_squares_preconditioner",
    "static_preconditioner",
    "m_imp_regularization_lambda",
)

INPUT_DATASET_KEYS = ("jr", "Br", "conductance", "u", "Q_eff", "E_source")
INPUT_MAINFIELD_SETTING_KEYS = ("mainfield_kind", "mainfield_epoch", "mainfield_B0")
INPUT_GEOMETRY_SETTING_KEYS = ("RM",) + INPUT_MAINFIELD_SETTING_KEYS
INPUT_DATASET_REQUIREMENT_KEYS = {"Br": ("RM",)}


def input_projection_settings(config_or_settings: Any) -> dict[str, Any]:
    """Return settings defining prepared input coefficient space."""
    config = (
        config_or_settings
        if isinstance(config_or_settings, SimulationConfig)
        else SimulationConfig.from_settings(config_or_settings)
    )
    return {name: getattr(config, name) for name in INPUT_PROJECTION_SETTING_KEYS}


def input_geometry_settings(config_or_settings: Any) -> dict[str, Any]:
    """Return geometry settings stored with prepared inputs."""
    config = (
        config_or_settings
        if isinstance(config_or_settings, SimulationConfig)
        else SimulationConfig.from_settings(config_or_settings)
    )
    geometry = {name: getattr(config, name) for name in INPUT_GEOMETRY_SETTING_KEYS}
    geometry["mainfield_coordinate_time"] = config.t0
    return geometry


def input_dataset_requirements(
    input_datasets: list[str] | tuple[str, ...],
) -> dict[str, list[str]]:
    """Return run-setting requirements implied by prepared datasets."""
    datasets = [str(key) for key in input_datasets]
    return {
        key: list(INPUT_DATASET_REQUIREMENT_KEYS[key])
        for key in datasets
        if key in INPUT_DATASET_REQUIREMENT_KEYS
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
        return IO.build_temporary_run_directory_in_directory("simulation/inputs")
    return IO.build_run_directory(directory)


def _run_directory(directory: str | Path | None) -> str:
    """Return a resolved output run directory."""
    if directory is None:
        return IO.build_temporary_run_directory_in_directory("simulation/runs")
    return IO.build_run_directory(directory)


def _available_input_datasets(io: IO) -> list[str]:
    """Return prepared input datasets present in one directory."""
    artifacts = io.scan_run_artifacts()
    return [key for key in INPUT_DATASET_KEYS if key in artifacts]


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
        "version": 1,
        "source": source,
        "input_projection_settings": input_projection_settings(config),
        "input_contract": prepared_input_contract(config, input_datasets),
        "input_datasets": input_datasets,
        "notes": list(notes),
        "metadata": dict(metadata or {}),
    }
    _write_json(Path(directory) / INPUT_MANIFEST_FILENAME, manifest)
    return manifest


def read_input_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Read a prepared-input manifest if present."""
    return _read_json(Path(directory).resolve() / INPUT_MANIFEST_FILENAME)


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
    if manifest.get("kind") != "pynamit_prepared_inputs":
        raise ValueError(
            f"{INPUT_MANIFEST_FILENAME} in {directory!s} is not a PynaMIT prepared-input manifest."
        )

    manifest_inputs = [str(key) for key in manifest.get("input_datasets", [])]
    if available_inputs is None:
        available_inputs = _available_input_datasets(IO(directory))
    available_inputs = [str(key) for key in available_inputs]

    missing = [key for key in manifest_inputs if key not in available_inputs]
    unlisted = [key for key in available_inputs if key not in manifest_inputs]
    if missing or (unlisted and not allow_unlisted):
        details = []
        if missing:
            details.append(f"listed but missing: {missing}")
        if unlisted and not allow_unlisted:
            details.append(f"stored but not listed: {unlisted}")
        raise ValueError(
            f"Prepared input artifacts in {directory!s} do not match "
            f"{INPUT_MANIFEST_FILENAME}: " + "; ".join(details)
        )

    if settings is not None:
        expected_contract = prepared_input_contract(settings, manifest_inputs)
        stored_contract = manifest.get("input_contract")
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
    definition, and main-field coordinate time.  Boundary ``Br`` also
    needs the same ``RM``.  PFAC treatment, hemisphere coupling,
    low-latitude boundary, shielding, and integrator remain run choices.
    """
    input_config = SimulationConfig.from_settings(input_settings)
    run_config = SimulationConfig.from_settings(run_settings)
    input_projection = input_projection_settings(input_config)
    run_projection = input_projection_settings(run_config)
    mismatches = {}
    for name in INPUT_PROJECTION_SETTING_KEYS:
        if input_projection[name] != run_projection[name]:
            mismatches[name] = (input_projection[name], run_projection[name])
    for name in INPUT_MAINFIELD_SETTING_KEYS:
        input_value = getattr(input_config, name)
        run_value = getattr(run_config, name)
        if not _settings_equal(input_value, run_value):
            mismatches[name] = (input_value, run_value)
    if not _settings_equal(input_config.t0, run_config.t0):
        mismatches["mainfield_coordinate_time"] = (input_config.t0, run_config.t0)

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


def load_prepared_inputs_into_dynamics(
    dynamics: Dynamics,
    input_directory: str | Path,
    *,
    artifact_storage: str = "auto",
    enabled_inputs: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Load prepared input datasets into an existing ``Dynamics``."""
    input_directory = IO.discover_run_directory(input_directory)
    input_io = IO(input_directory, preferred_dataset_storage=artifact_storage)
    input_settings = input_io.load_dataset("settings")
    if input_settings is None:
        raise ValueError(
            f"No settings dataset found in prepared input directory {input_directory!r}."
        )

    available_inputs = _available_input_datasets(input_io)
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
    validate_input_manifest(
        input_directory,
        input_settings,
        available_inputs=available_inputs,
        allow_unlisted=allowed is not None,
        require=True,
    )
    selected_inputs = (
        available_inputs
        if allowed is None
        else [key for key in available_inputs if key in allowed]
    )

    validate_prepared_input_compatibility(
        input_settings,
        dynamics.settings,
        input_datasets=selected_inputs,
        input_directory=input_directory,
    )

    timeseries = Timeseries(
        dynamics.schema.input_field_spaces,
        dynamics.schema.input_vars,
        area_weighted_least_squares=dynamics.config.area_weighted_least_squares,
    )
    timeseries.load_all(input_io)
    if allowed is not None:
        timeseries.datasets = {
            key: dataset for key, dataset in timeseries.datasets.items() if key in allowed
        }

    loaded = [key for key in INPUT_DATASET_KEYS if key in timeseries.datasets]
    if not loaded:
        raise ValueError(f"No prepared input datasets found in {input_directory!r}.")
    active_wind_sources = sorted(set(loaded) & {"u", "Q_eff", "E_source"})
    if len(active_wind_sources) > 1:
        raise ValueError(
            "Prepared input selection contains mutually exclusive wind/source "
            f"representations {active_wind_sources}; enable only one of 'u', "
            "'Q_eff', or 'E_source' for a run."
        )

    dynamics.input_timeseries = timeseries
    dynamics.data.input_timeseries = timeseries
    return loaded


def prepare_pynamit_inputs(
    input_directory=None,
    *,
    final_time=100,
    Nmax=20,
    Mmax=20,
    Ncs=30,
    mainfield_kind="dipole",
    mainfield_epoch=2020,
    mainfield_B0=None,
    use_wind=False,
    use_Q_eff=False,
    use_jr=True,
    jr_projection_basis=None,
    Br_projection_basis=None,
    conductance_projection_basis=None,
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
    """Prepare default example inputs without evolving dynamics.

    This is the split-out input-projection half of ``run_pynamit``.  It
    writes the projected input datasets plus a small manifest to
    ``input_directory`` and returns the ``Dynamics`` instance that owns
    the package.

    ``mainfield_*`` is part of the prepared-input contract because
    inputs may be projected in model magnetic coordinates or converted
    from FAC to ``jr`` using the main field.
    """
    if use_Q_eff and not use_wind:
        raise ValueError("use_Q_eff=True requires use_wind=True in prepare_pynamit_inputs.")

    input_directory = _input_directory(input_directory)
    dynamics = Dynamics(
        run_directory=input_directory,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=RE + 110.0e3,
        mainfield_kind=mainfield_kind,
        mainfield_epoch=mainfield_epoch,
        mainfield_B0=mainfield_B0,
        jr_projection_basis=jr_projection_basis,
        Br_projection_basis=Br_projection_basis,
        conductance_projection_basis=conductance_projection_basis,
        u_projection_basis=u_projection_basis,
        Q_eff_projection_basis=Q_eff_projection_basis,
        horizontal_basis_kind=horizontal_basis_kind,
        area_weighted_least_squares=area_weighted_least_squares,
        artifact_storage=artifact_storage,
        ignore_PFAC=True,
    )

    date = _datetime.datetime(2001, 5, 12, 21, 45)
    time = np.linspace(0, final_time, 4) if multi_data else None

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon
    hall, pedersen, conductance_lat, conductance_lon = get_conductance_inputs(
        date, conductance_lat, conductance_lon, time
    )
    dynamics.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        reg_lambda=conductance_lambda,
        time=time,
    )

    if use_jr:
        jr_lat = dynamics.state.geometry.grid.lat
        jr_lon = dynamics.state.geometry.grid.lon
        jr, jr_lat, jr_lon = get_jr_inputs(date, jr_lat, jr_lon, time)
        dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon, reg_lambda=jr_lambda, time=time)

    wind_inputs = get_wind_inputs(date, use_wind=use_wind, time=time)
    if wind_inputs is not None:
        u_theta, u_phi, u_lat, u_lon, weights = wind_inputs
        if use_Q_eff:
            dynamics.set_Q_eff_from_neutral_wind(
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
            dynamics.set_neutral_wind(
                u_theta=u_theta,
                u_phi=u_phi,
                lat=u_lat,
                lon=u_lon,
                sqrt_weights=weights,
                reg_lambda=u_lambda,
                time=time,
            )

    input_io = IO(input_directory, preferred_dataset_storage=artifact_storage)
    notes = []
    if use_Q_eff:
        notes.append(
            "Q_eff was derived from neutral wind through the current model operators; "
            "prefer direct E_source inputs for externally prepared weighted winds."
        )
    write_input_manifest(
        input_directory,
        dynamics.settings,
        input_datasets=_available_input_datasets(input_io),
        source="pynamit.default_external_inputs",
        notes=notes,
        metadata={"external_input_source": get_input_source(), "multi_data": bool(multi_data)},
    )
    return dynamics


def run_pynamit_from_inputs(
    input_directory,
    *,
    run_directory=None,
    enabled_inputs=None,
    final_time=100,
    plotsteps=200,
    dt=5e-4,
    RM=None,
    mainfield_kind=None,
    FAC_integration_steps=None,
    ih_constraint_scaling=None,
    ignore_PFAC=True,
    connect_hemispheres=False,
    latitude_boundary=50,
    steady_state_initialization=True,
    run_inductive=True,
    run_steady_state=True,
    integrator="euler",
    least_squares_solver=None,
    least_squares_preconditioner="pinv",
    static_preconditioner=False,
    m_imp_regularization_lambda=0.0,
    artifact_storage="auto",
    RM_shielding=False,
):
    """Run dynamics from a prepared input package."""
    input_directory = IO.discover_run_directory(input_directory)
    input_io = IO(input_directory, preferred_dataset_storage=artifact_storage)
    input_settings = input_io.load_dataset("settings")
    if input_settings is None:
        raise ValueError(
            f"No settings dataset found in prepared input directory {input_directory!r}."
        )

    config_kwargs = SimulationConfig.from_settings(input_settings).to_kwargs()
    config_kwargs.update(
        {
            "ignore_PFAC": ignore_PFAC,
            "connect_hemispheres": connect_hemispheres,
            "latitude_boundary": latitude_boundary,
            "save_steady_states": run_steady_state,
            "integrator": integrator,
            "least_squares_solver": least_squares_solver,
            "least_squares_preconditioner": least_squares_preconditioner,
            "static_preconditioner": static_preconditioner,
            "m_imp_regularization_lambda": m_imp_regularization_lambda,
            "RM_shielding": RM_shielding,
        }
    )
    if RM is not None:
        config_kwargs["RM"] = RM
    if mainfield_kind is not None:
        config_kwargs["mainfield_kind"] = mainfield_kind
    if FAC_integration_steps is not None:
        config_kwargs["FAC_integration_steps"] = FAC_integration_steps
    if ih_constraint_scaling is not None:
        config_kwargs["ih_constraint_scaling"] = ih_constraint_scaling
    run_directory = _run_directory(run_directory)
    dynamics = Dynamics(
        run_directory=run_directory, artifact_storage=artifact_storage, **config_kwargs
    )
    loaded_inputs = load_prepared_inputs_into_dynamics(
        dynamics, input_directory, artifact_storage=artifact_storage, enabled_inputs=enabled_inputs
    )
    for key in loaded_inputs:
        dataset = input_io.load_dataset(key)
        if dataset is not None:
            dynamics.io.save_dataset(dataset, key)

    _write_json(
        Path(run_directory) / RUN_MANIFEST_FILENAME,
        {
            "kind": "pynamit_run",
            "version": 1,
            "input_directory": str(Path(input_directory).resolve()),
            "enabled_inputs": loaded_inputs,
            "run_settings": {
                name: _plain_json_value(getattr(dynamics.config, name))
                for name in RUN_SETTING_KEYS
            },
            "time_evolution": {
                "final_time": final_time,
                "dt": dt,
                "plotsteps": plotsteps,
                "steady_state_initialization": steady_state_initialization,
                "run_inductive": run_inductive,
                "run_steady_state": run_steady_state,
            },
        },
    )

    dynamics.evolve_to_time(
        t=final_time,
        dt=dt,
        sampling_step_interval=1,
        saving_sample_interval=plotsteps,
        steady_state_initialization=steady_state_initialization,
        run_inductive=run_inductive,
        run_steady_state=run_steady_state,
    )
    return dynamics


__all__ = [
    "INPUT_MANIFEST_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "INPUT_PROJECTION_SETTING_KEYS",
    "INPUT_MAINFIELD_SETTING_KEYS",
    "INPUT_GEOMETRY_SETTING_KEYS",
    "INPUT_DATASET_REQUIREMENT_KEYS",
    "RUN_SETTING_KEYS",
    "input_projection_settings",
    "input_geometry_settings",
    "input_dataset_requirements",
    "prepared_input_contract",
    "write_input_manifest",
    "read_input_manifest",
    "validate_input_manifest",
    "validate_prepared_input_compatibility",
    "load_prepared_inputs_into_dynamics",
    "prepare_pynamit_inputs",
    "run_pynamit_from_inputs",
]
