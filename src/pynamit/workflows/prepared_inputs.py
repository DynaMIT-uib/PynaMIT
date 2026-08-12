"""Prepared-input packages for PynaMIT simulations.

An input package is intentionally narrower than a full simulation
directory. It stores projected input coefficients on a chosen grid,
basis, and explicit main-field coordinate system. The time origin
locates those coefficients in physical time; it is not part of the
spatial frame. Most evolution choices belong to the consuming
simulation, but ``boundary_Br`` also declares its required
magnetospheric radius.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.simulation import input_manifest as _input_manifest
from pynamit.simulation.api import Simulation
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.runner import SimulationRunner
from pynamit.simulation.schema import INPUT_DATASET_KEYS
from pynamit.storage import ArtifactStore, FieldTimeSeries
from pynamit.storage.field_time_series import TIME_TOLERANCE_SECONDS

SIMULATION_MANIFEST_FILENAME = "pynamit_simulation_manifest.json"
_SIMULATION_MANIFEST_VERSION = 4

_SIMULATION_SETTING_KEYS = (
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
    "save_equilibria",
    "integrator",
    "least_squares_solver",
    "least_squares_preconditioner",
    "reuse_preconditioner",
    "toroidal_potential_regularization_lambda",
)

def _plain_json_value(value: Any) -> Any:
    """Return a JSON-serializable version of a setting value."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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

    artifacts = input_store.scan_artifacts(INPUT_DATASET_KEYS)
    available_inputs = [key for key in INPUT_DATASET_KEYS if key in artifacts]
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
    manifest = _input_manifest.validate_input_manifest(
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
    active_wind_forcings = sorted(set(selected_inputs) & {"u", "Q_eff", "E_neutral_wind"})
    if len(active_wind_forcings) > 1:
        raise ValueError(
            "Prepared input selection contains mutually exclusive wind-forcing "
            f"representations {active_wind_forcings}; enable only one of 'u', "
            "'Q_eff', or 'E_neutral_wind' for a simulation."
        )
    return input_directory, input_store, input_settings, selected_inputs, manifest


def _copy_prepared_inputs(
    simulation: Simulation, input_store: ArtifactStore, selected_inputs: list[str]
) -> list[str]:
    """Copy validated input streams into a simulation-owned store."""
    series = FieldTimeSeries(
        simulation.data.schema.input_field_spaces, simulation.data.schema.input_variables
    )
    for key in selected_inputs:
        series.load(key, input_store)

    loaded = [key for key in INPUT_DATASET_KEYS if key in series.datasets]
    for key in INPUT_DATASET_KEYS:
        if key not in loaded:
            simulation.data.artifact_store.remove_artifact(key)
    for key in loaded:
        simulation.data.artifact_store.save_dataset(series.datasets[key].reset_index("i"), key)

    # Reload through the consuming simulation's store. Lazy Zarr arrays
    # must not leave an active simulation dependent on the preparation
    # package.
    simulation_series = FieldTimeSeries(
        simulation.data.schema.input_field_spaces, simulation.data.schema.input_variables
    )
    simulation_series.load_all(simulation.data.artifact_store)
    simulation.data.input_series = simulation_series
    return loaded


def _validate_simulation_identity(
    simulation_directory: str | Path,
    *,
    selected_inputs: list[str],
    input_manifest: dict[str, Any],
    simulation_settings: dict[str, Any],
    evolution_policy: dict[str, Any],
) -> bool:
    """Require a trajectory to keep one identity."""
    existing = _read_json(Path(simulation_directory) / SIMULATION_MANIFEST_FILENAME)
    if existing is None:
        return False
    if (
        not isinstance(existing, dict)
        or existing.get("kind") not in {"pynamit_simulation", "pynamit_paper_simulation"}
        or existing.get("version") != _SIMULATION_MANIFEST_VERSION
    ):
        raise ValueError(
            f"Existing {SIMULATION_MANIFEST_FILENAME} in {simulation_directory!s} does not describe "
            "a compatible resumable simulation. Use a new simulation directory."
        )

    existing_evolution = dict(existing.get("time_evolution", {}))
    existing_evolution.pop("final_time", None)
    expected = {
        "enabled_inputs": selected_inputs,
        "input_manifest": input_manifest,
        "simulation_settings": simulation_settings,
        "time_evolution": evolution_policy,
    }
    actual = {**existing, "time_evolution": existing_evolution}
    mismatches = [name for name, value in expected.items() if actual.get(name) != value]
    if mismatches:
        raise ValueError(
            f"Existing simulation in {simulation_directory!s} has a different trajectory identity "
            f"({', '.join(mismatches)}). Use a new simulation directory for a different "
            "input package, selection, or evolution policy."
        )
    return True


def _stored_simulation_outputs_reach(
    store: ArtifactStore, target_time: float, *, run_dynamic: bool, run_equilibrium: bool
) -> bool:
    """Return whether all requested persisted outputs reach a target."""
    requested_outputs = []
    if run_dynamic:
        requested_outputs.append("dynamic")
    if run_equilibrium:
        requested_outputs.append("equilibrium")
    if not requested_outputs:
        return False

    available = store.scan_artifacts(requested_outputs)
    if any(key not in available for key in requested_outputs):
        return False
    for key in requested_outputs:
        dataset = store.load_dataset(key)
        if dataset is None or "time" not in dataset.coords or dataset.sizes.get("time", 0) == 0:
            return False
        times = np.asarray(dataset.time.values, dtype=float)
        if not np.all(np.isfinite(times)):
            return False
        if float(np.max(times)) < float(target_time) - TIME_TOLERANCE_SECONDS:
            return False
    return True


def _validate_stored_simulation_settings(
    store: ArtifactStore, config: SimulationConfig, simulation_directory: str | Path
) -> None:
    """Require stored simulation settings to match the request."""
    stored_settings = store.load_dataset("settings")
    if stored_settings is None:
        raise ValueError(
            f"Existing {SIMULATION_MANIFEST_FILENAME} in {simulation_directory!s} has no settings artifact."
        )
    normalized_stored = SimulationConfig.from_settings(stored_settings).to_dataset()
    if not config.to_dataset().identical(normalized_stored):
        raise ValueError(
            f"Existing settings in {simulation_directory!s} do not match the requested simulation."
        )


def load_prepared_inputs_into_simulation(
    simulation: Simulation,
    input_directory: str | Path,
    *,
    artifact_storage: str = "auto",
    enabled_inputs: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Validate and copy inputs into an existing simulation."""
    input_directory, input_store, input_settings, selected_inputs, _ = (
        _validate_and_select_prepared_inputs(
            input_directory, artifact_storage=artifact_storage, enabled_inputs=enabled_inputs
        )
    )

    _input_manifest.validate_prepared_input_compatibility(
        input_settings,
        simulation.data.config,
        input_datasets=selected_inputs,
        input_directory=input_directory,
    )
    return _copy_prepared_inputs(simulation, input_store, selected_inputs)


def run_from_inputs(
    input_directory,
    *,
    simulation_directory=None,
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
    equilibrium_initialization=True,
    run_dynamic=True,
    run_equilibrium=True,
    integrator="euler",
    least_squares_solver=None,
    least_squares_preconditioner="pinv",
    reuse_preconditioner=False,
    toroidal_potential_regularization_lambda=0.0,
    artifact_storage="auto",
    operator_cache_directory=None,
    magnetic_boundary_shielding=False,
    skip_completed=False,
):
    """Run simulation from a prepared input package.

    With ``skip_completed=True``, a matching simulation returns ``None``
    before geometry construction when its requested outputs reach
    ``final_time``.
    This is intended for batch orchestration; callers requiring a live
    ``Simulation`` object should keep the default.
    """
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
            "save_equilibria": run_equilibrium,
            "integrator": integrator,
            "least_squares_solver": least_squares_solver,
            "least_squares_preconditioner": least_squares_preconditioner,
            "reuse_preconditioner": reuse_preconditioner,
            "toroidal_potential_regularization_lambda": toroidal_potential_regularization_lambda,
            "magnetic_boundary_shielding": magnetic_boundary_shielding,
        }
    )
    if RM is not None:
        config_kwargs["RM"] = RM
    if main_field_kind is not None:
        config_kwargs["main_field_kind"] = main_field_kind
    # PFAC integration is simulation geometry, not part of the
    # prepared-input contract. ``None`` asks SimulationConfig to derive
    # a radial grid for this simulation's RI/RM.
    config_kwargs["fac_integration_radii"] = fac_integration_radii
    if interhemispheric_electric_field_weight is not None:
        config_kwargs["interhemispheric_electric_field_weight"] = (
            interhemispheric_electric_field_weight
        )
    config = SimulationConfig(**config_kwargs)
    _input_manifest.validate_prepared_input_compatibility(
        input_settings, config, input_datasets=selected_inputs, input_directory=input_directory
    )
    if not isinstance(skip_completed, (bool, np.bool_)):
        raise ValueError("skip_completed must be a boolean value.")
    skip_completed = bool(skip_completed)
    options = SimulationRunner.normalize_evolution_options(
        config,
        t=final_time,
        dt=dt,
        sampling_step_interval=sampling_step_interval,
        saving_sample_interval=saving_sample_interval,
        quiet=False,
        equilibrium_initialization=equilibrium_initialization,
        run_dynamic=run_dynamic,
        run_equilibrium=run_equilibrium,
    )
    final_time = options.target_time
    dt = float(options.dt)
    sampling_step_interval = options.sampling_step_interval
    saving_sample_interval = options.saving_sample_interval
    equilibrium_initialization = options.equilibrium_initialization
    run_dynamic = options.run_dynamic
    run_equilibrium = options.run_equilibrium
    time_evolution = {
        "final_time": final_time,
        "dt": dt,
        "sampling_step_interval": sampling_step_interval,
        "saving_sample_interval": saving_sample_interval,
        "equilibrium_initialization": equilibrium_initialization,
        "run_dynamic": run_dynamic,
        "run_equilibrium": run_equilibrium,
    }
    evolution_policy = {
        name: value for name, value in time_evolution.items() if name != "final_time"
    }
    simulation_settings = {name: _plain_json_value(getattr(config, name)) for name in _SIMULATION_SETTING_KEYS}
    simulation_directory = (
        ArtifactStore.create_temporary_directory("simulations")
        if simulation_directory is None
        else str(Path(simulation_directory).resolve())
    )
    if Path(simulation_directory) == Path(input_directory):
        raise ValueError(
            "simulation_directory must differ from input_directory so the reusable "
            "prepared-input package remains immutable."
        )
    existing_simulation = _validate_simulation_identity(
        simulation_directory,
        selected_inputs=selected_inputs,
        input_manifest=input_manifest,
        simulation_settings=simulation_settings,
        evolution_policy=evolution_policy,
    )
    if existing_simulation:
        simulation_store = ArtifactStore(simulation_directory, preferred_dataset_storage=artifact_storage)
        _validate_stored_simulation_settings(simulation_store, config, simulation_directory)
        if skip_completed and _stored_simulation_outputs_reach(
            simulation_store, final_time, run_dynamic=run_dynamic, run_equilibrium=run_equilibrium
        ):
            print(
                f"Simulation output in {simulation_directory} already reaches "
                f"t={float(final_time):g} s; "
                "skipping.",
                flush=True,
            )
            return None

    simulation = Simulation.from_config(
        config,
        simulation_directory=simulation_directory,
        artifact_storage=artifact_storage,
        operator_cache_directory=operator_cache_directory,
    )
    existing_inputs = [
        key for key in INPUT_DATASET_KEYS if key in simulation.inputs
    ]
    loaded_inputs = (
        existing_inputs
        if existing_simulation and existing_inputs == selected_inputs
        else _copy_prepared_inputs(simulation, input_store, selected_inputs)
    )

    _write_json(
        Path(simulation_directory) / SIMULATION_MANIFEST_FILENAME,
        {
            "kind": "pynamit_simulation",
            "version": _SIMULATION_MANIFEST_VERSION,
            "input_directory": str(Path(input_directory).resolve()),
            "enabled_inputs": loaded_inputs,
            "input_manifest": input_manifest,
            "simulation_settings": simulation_settings,
            "time_evolution": time_evolution,
        },
    )

    simulation.evolve_to_time(
        t=final_time,
        dt=dt,
        sampling_step_interval=sampling_step_interval,
        saving_sample_interval=saving_sample_interval,
        equilibrium_initialization=equilibrium_initialization,
        run_dynamic=run_dynamic,
        run_equilibrium=run_equilibrium,
    )
    return simulation


__all__ = [
    "SIMULATION_MANIFEST_FILENAME",
    "load_prepared_inputs_into_simulation",
    "run_from_inputs",
]
