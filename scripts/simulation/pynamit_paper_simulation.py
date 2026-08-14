"""Modern split version of the original PynaMIT paper simulation.

The input projection and the time evolution are intentionally separate:

``prepare_paper_inputs``
    Projects conductance, neutral wind, and AMPS radial current into a
    reusable input package.

``run_paper_simulation``
    Consumes that package.  The first phase uses conductance and wind
    only, imposes equilibrium, then loads the prepared radial current
    and continues the inductive evolution.

Edit ``SETTINGS`` below instead of passing command-line flags.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from kompe.constants import EARTH_RADIUS_M

import pynamit
from pynamit.simulation.config import dipole_fac_integration_radii
from pynamit.simulation.input_manifest import clear_prepared_input_package, write_input_manifest
from pynamit.simulation.schema import INPUT_DATASET_KEYS
from pynamit.workflows.prepared_inputs import (
    SIMULATION_MANIFEST_FILENAME,
    load_prepared_inputs_into_simulation,
    run_from_inputs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RI = EARTH_RADIUS_M + 110e3


@dataclass(frozen=True)
class PaperSimulationSettings:
    """Editable settings for the paper-style simulation."""

    input_directory: Path = SCRIPT_DIR / "paper_prepared" / "inputs"
    simulation_directory: Path = SCRIPT_DIR / "paper_simulations" / "pynamit_paper_simulation"
    date: dt.datetime = dt.datetime(2001, 6, 1, 0, 0)
    kp: float = 4.0
    nmax: int = 90
    mmax: int = 90
    ncs: int = 100
    simulation_time: float = 480.0
    interhemispheric_coupling_latitude: float = 45.0
    dt: float = 5e-4
    write_sample_interval: int = 200
    conductance_lambda: float = 0.001
    wind_lambda: float = 0.001
    boundary_jr_lambda: float | None = None
    area_weighted_least_squares: bool = False
    toroidal_potential_regularization_lambda: float = 0.0
    artifact_storage: str = "auto"
    prepare_inputs: bool = True
    run_simulation: bool = True


SETTINGS = PaperSimulationSettings()


def _json_value(value: Any) -> Any:
    """Return a JSON-serializable metadata value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepared_input_datasets(preparation: pynamit.InputPreparation) -> list[str]:
    """Return projected input artifacts present in ``preparation``."""
    artifacts = preparation.data.artifact_store.scan_artifacts(INPUT_DATASET_KEYS)
    return [key for key in INPUT_DATASET_KEYS if key in artifacts]


def prepare_paper_inputs(settings: PaperSimulationSettings = SETTINGS) -> Path:
    """Project paper inputs through the shared library adapters."""
    import dipole

    from pynamit.external_inputs import get_conductance_inputs, get_jr_inputs, get_wind_inputs

    input_directory = Path(settings.input_directory).expanduser()
    input_directory.mkdir(parents=True, exist_ok=True)
    clear_prepared_input_package(input_directory, artifact_storage=settings.artifact_storage)
    (input_directory / "paper_input_metadata.json").unlink(missing_ok=True)

    print(f"Writing paper input package: {input_directory}", flush=True)
    preparation = pynamit.InputPreparation(
        input_directory=input_directory,
        Nmax=settings.nmax,
        Mmax=settings.mmax,
        Ncs=settings.ncs,
        RI=RI,
        main_field_kind="igrf",
        artifact_storage=settings.artifact_storage,
        area_weighted_least_squares=settings.area_weighted_least_squares,
        t0=str(settings.date),
    )

    source_lat = preparation.model_grid.lat
    source_lon = preparation.model_grid.lon
    pedersen, hall, _, _ = get_conductance_inputs(
        settings.date, source_lat, source_lon, kp=settings.kp
    )
    preparation.set_conductance(
        pedersen=pedersen,
        hall=hall,
        lat=source_lat,
        lon=source_lon,
        reg_lambda=settings.conductance_lambda,
    )

    dipole_model = dipole.Dipole(preparation.geometry.main_field.epoch)
    boundary_jr, _, _ = get_jr_inputs(
        settings.date,
        source_lat,
        source_lon,
        amps_parameters=(400.0, 5.0, -5.0, float(dipole_model.tilt(settings.date)), 100.0),
        minlat=50.0,
    )
    preparation.set_boundary_jr(
        boundary_jr, lat=source_lat, lon=source_lon, reg_lambda=settings.boundary_jr_lambda
    )

    wind = get_wind_inputs(settings.date, lat=source_lat, lon=source_lon)
    if wind is None:
        raise RuntimeError("HWM14 returned no wind data.")
    u_theta, u_phi, _, _, sqrt_weights = wind
    preparation.set_neutral_wind(
        u_theta=u_theta,
        u_phi=u_phi,
        lat=source_lat,
        lon=source_lon,
        sqrt_weights=sqrt_weights,
        reg_lambda=settings.wind_lambda,
    )

    write_input_manifest(
        input_directory,
        preparation.data.config,
        input_datasets=_prepared_input_datasets(preparation),
        source="scripts.simulation.pynamit_paper_simulation",
        notes=[
            "Paper-style inputs use the shared Hardy, AMPS, and HWM adapters.",
            "The three providers are sampled on the simulation model grid.",
            "The simulation loads conductance/wind first, imposes equilibrium, then enables jr.",
        ],
        metadata={
            "date": settings.date.isoformat(),
            "kp": settings.kp,
            "projection_regularization": {
                "conductance_lambda": settings.conductance_lambda,
                "wind_lambda": settings.wind_lambda,
                "boundary_jr_lambda": settings.boundary_jr_lambda,
            },
        },
    )
    return input_directory


def run_paper_simulation(settings: PaperSimulationSettings = SETTINGS) -> pynamit.Simulation:
    """Run the paper-style simulation from projected inputs."""
    input_directory = Path(settings.input_directory).expanduser()
    simulation_directory = Path(settings.simulation_directory).expanduser()
    print(f"Using paper input package: {input_directory}", flush=True)
    print(f"Writing paper simulation: {simulation_directory}", flush=True)

    simulation = run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        enabled_inputs=("conductance", "u"),
        final_time=settings.simulation_time,
        dt=settings.dt,
        write_sample_interval=settings.write_sample_interval,
        main_field_kind="igrf",
        fac_integration_radii=dipole_fac_integration_radii(
            RI, RI / np.cos(np.deg2rad(69.0)) ** 2, n_points=70
        ),
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=settings.interhemispheric_coupling_latitude,
        interhemispheric_electric_field_weight=1e-5,
        initialize_from_equilibrium=False,
        run_dynamic=True,
        run_equilibrium=False,
        toroidal_potential_regularization_lambda=settings.toroidal_potential_regularization_lambda,
        artifact_storage=settings.artifact_storage,
    )

    print("Imposing wind/conductance equilibrium before enabling jr", flush=True)
    simulation.impose_equilibrium()

    loaded_inputs = load_prepared_inputs_into_simulation(
        simulation,
        input_directory,
        artifact_storage=settings.artifact_storage,
        enabled_inputs=("conductance", "u", "boundary_jr"),
    )

    final_time = 2.0 * float(settings.simulation_time)
    print(f"Continuing with jr enabled to t={final_time:g} s", flush=True)
    simulation.evolve_to_time(
        final_time,
        dt=settings.dt,
        sampling_step_interval=1,
        write_sample_interval=settings.write_sample_interval,
        initialize_from_equilibrium=False,
        run_dynamic=True,
        run_equilibrium=False,
    )

    manifest_path = Path(simulation.simulation_directory) / SIMULATION_MANIFEST_FILENAME
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.update(
        {
            "kind": "pynamit_paper_simulation",
            "paper_two_phase_simulation": {
                "phase_1_inputs": ["conductance", "u"],
                "phase_1_final_time": settings.simulation_time,
                "phase_2_inputs": loaded_inputs,
                "phase_2_final_time": final_time,
                "equilibrium_imposed_between_phases": True,
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return simulation


def main(settings: PaperSimulationSettings = SETTINGS) -> None:
    """Run the configured preparation and simulation workflow."""
    if settings.prepare_inputs:
        prepare_paper_inputs(settings)
    if settings.run_simulation:
        run_paper_simulation(settings)


if __name__ == "__main__":
    main()
