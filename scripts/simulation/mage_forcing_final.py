"""Run PynaMIT from projected MAGE/GAMERA/TIEGCM inputs.

The MAGE workflow is split at the expensive and resolution-dependent
boundaries:

1. ``mage_prepare_forcing.py`` converts native MAGE/TIEGCM output into a
   height-integrated HDF5 forcing file.
2. ``mage_project_inputs.py`` projects that forcing onto the PynaMIT
   basis for a chosen spherical-harmonic/cubed-sphere resolution.
3. This script only loads the projected input package and evolves the
   model.

Edit ``SETTINGS`` below for the run.  Projection regularization,
weighted-wind processing, and input-grid details intentionally stay in
``mage_project_inputs.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pynamit
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.workflows.mage import (
    DEFAULT_MMAX,
    DEFAULT_NCS,
    DEFAULT_NMAX,
    projection_directory_for_resolution,
    result_directory_for_resolution,
)
from pynamit.simulation.workflows.prepared_inputs import (
    RUN_MANIFEST_FILENAME,
    load_prepared_inputs_into_simulation,
)
from pynamit.storage import ArtifactStore

SCRIPT_DIR = Path(__file__).resolve().parent
LATITUDE_BOUNDARY = 35.0
DEFAULT_MAGE_RUN_ROOT = SCRIPT_DIR / "mage_runs" / "mage_2011_kaiju_direct_e"


@dataclass(frozen=True)
class MageForcingSettings:
    """Defaults intended to be edited for the MAGE run step."""

    nmax: int = DEFAULT_NMAX
    mmax: int = DEFAULT_MMAX
    ncs: int = DEFAULT_NCS
    input_directory: Path | None = None
    run_directory: Path | None = None
    magnetic_boundary_shielding: bool = False
    dt: float = 10.0
    final_time: float = 3600.0
    sampling_step_interval: int = 1
    saving_sample_interval: int = 1
    integrator: str = "exponential"
    m_imp_regularization_lambda: float = 0.0
    steady_state_initialization: bool = False
    save_steady_states: bool = False
    artifact_storage: str = "auto"


SETTINGS = MageForcingSettings()


def main(settings: MageForcingSettings = SETTINGS) -> None:
    """Run PynaMIT from an already projected MAGE input package."""
    input_directory = Path(
        settings.input_directory
        or projection_directory_for_resolution(
            settings.nmax, settings.mmax, settings.ncs, DEFAULT_MAGE_RUN_ROOT
        )
    ).expanduser()
    if not input_directory.exists():
        raise FileNotFoundError(
            f"Projected input package does not exist: {input_directory}. "
            "Run scripts/simulation/mage_project_inputs.py first."
        )

    input_directory = Path(
        ArtifactStore.require_artifact_directory(input_directory, ("settings",))
    )
    input_store = ArtifactStore(
        input_directory, preferred_dataset_storage=settings.artifact_storage
    )
    input_settings = input_store.load_dataset("settings")

    mage_metadata_path = input_directory / "mage_input_metadata.json"
    mage_metadata = {}
    if mage_metadata_path.exists():
        mage_metadata = json.loads(mage_metadata_path.read_text(encoding="utf-8"))

    config_kwargs = SimulationConfig.from_settings(input_settings).to_kwargs()
    config_kwargs.update(
        {
            "magnetic_boundary_shielding": settings.magnetic_boundary_shielding,
            "enable_pfac_coupling": True,
            "enable_interhemispheric_coupling": True,
            "interhemispheric_coupling_latitude": LATITUDE_BOUNDARY,
            "interhemispheric_electric_field_weight": 1e-5,
            "save_steady_states": settings.save_steady_states,
            "integrator": settings.integrator,
            "m_imp_regularization_lambda": settings.m_imp_regularization_lambda,
        }
    )

    print(f"Using projected input package: {input_directory}", flush=True)
    run_directory = Path(
        settings.run_directory
        or result_directory_for_resolution(
            settings.nmax, settings.mmax, settings.ncs, DEFAULT_MAGE_RUN_ROOT
        )
    ).expanduser()
    print(f"Writing run directory: {run_directory}", flush=True)
    if mage_metadata:
        print(f"Projected forcing file: {mage_metadata.get('forcing_h5', 'unknown')}", flush=True)
        print(f"Event time: {mage_metadata.get('event_time', config_kwargs['t0'])}", flush=True)
        print(
            "Projected main field: "
            f"{mage_metadata.get('main_field_kind', config_kwargs['main_field_kind'])}",
            flush=True,
        )
        print(f"Projected RM: {mage_metadata.get('RM_m', config_kwargs['RM'])} m", flush=True)
    print(f"Run main field: {config_kwargs['main_field_kind']}", flush=True)
    if config_kwargs["RM"] is None:
        print("Run RM: None", flush=True)
    else:
        print(f"Run RM: {config_kwargs['RM']:.6g} m", flush=True)
    print(f"Magnetic-boundary shielding: {settings.magnetic_boundary_shielding}", flush=True)
    print("Wind forcing: projected direct E_source", flush=True)
    print(f"Integrator: {settings.integrator}", flush=True)
    if settings.integrator == "exponential":
        print(
            "Warning: the exponential integrator builds a dense matrix exponential at "
            "each step. Monitor the line-based progress/RSS output on MAGE-size runs.",
            flush=True,
        )
    print(f"Steady-state initialization: {settings.steady_state_initialization}", flush=True)

    simulation = pynamit.Simulation(
        run_directory=run_directory, artifact_storage=settings.artifact_storage, **config_kwargs
    )
    state_size = int(simulation.geometry.magnetic_basis.index_length)
    dense_matrix_mib = state_size * state_size * 8.0 / 1024.0**2
    print(
        f"Induction coefficient count: {state_size}; one dense float64 operator "
        f"is ~{dense_matrix_mib:.1f} MiB before solver/expm workspace.",
        flush=True,
    )
    print("Loading projected input datasets into simulation.", flush=True)
    loaded_inputs = load_prepared_inputs_into_simulation(
        simulation, input_directory, artifact_storage=settings.artifact_storage
    )
    print(f"Loaded projected inputs: {', '.join(loaded_inputs)}", flush=True)

    run_manifest = {
        "kind": "mage_pynamit_run",
        "version": 1,
        "input_directory": str(input_directory),
        "loaded_inputs": loaded_inputs,
        "magnetic_boundary_shielding": settings.magnetic_boundary_shielding,
        "time_evolution": {
            "final_time": settings.final_time,
            "dt": settings.dt,
            "sampling_step_interval": settings.sampling_step_interval,
            "saving_sample_interval": settings.saving_sample_interval,
            "integrator": settings.integrator,
            "steady_state_initialization": settings.steady_state_initialization,
            "run_steady_state": settings.save_steady_states,
        },
    }
    Path(simulation.run_data.run_directory).mkdir(parents=True, exist_ok=True)
    (Path(simulation.run_data.run_directory) / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "Time evolution: "
        f"final_time={settings.final_time:g} s, dt={settings.dt:g} s, "
        f"sample every {settings.sampling_step_interval} step(s), "
        f"save every {settings.saving_sample_interval} sample(s)",
        flush=True,
    )
    simulation.evolve_to_time(
        settings.final_time,
        dt=settings.dt,
        sampling_step_interval=settings.sampling_step_interval,
        saving_sample_interval=settings.saving_sample_interval,
        steady_state_initialization=settings.steady_state_initialization,
        run_steady_state=settings.save_steady_states,
    )
    print("Time evolution complete", flush=True)


if __name__ == "__main__":
    main()
