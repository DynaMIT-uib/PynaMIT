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
from pynamit.geomagnetism.main_field import decimal_year
from pynamit.simulation.config import dipole_fac_integration_radii
from pynamit.simulation.schema import INPUT_DATASET_KEYS
from pynamit.simulation.workflows.prepared_inputs import (
    RUN_MANIFEST_FILENAME,
    clear_prepared_input_package,
    load_prepared_inputs_into_simulation,
    run_pynamit_from_inputs,
    write_input_manifest,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RI = EARTH_RADIUS_M + 110e3


@dataclass(frozen=True)
class PaperSimulationSettings:
    """Defaults intended to be edited for the paper-style run."""

    input_directory: Path = SCRIPT_DIR / "paper_prepared" / "inputs"
    run_directory: Path = SCRIPT_DIR / "paper_runs" / "pynamit_paper_simulation"
    date: dt.datetime = dt.datetime(2001, 6, 1, 0, 0)
    kp: float = 4.0
    nmax: int = 90
    mmax: int = 90
    ncs: int = 100
    simulation_time: float = 480.0
    interhemispheric_coupling_latitude: float = 45.0
    dt: float = 5e-4
    saving_sample_interval: int = 200
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


def _prepared_input_datasets(simulation: pynamit.Simulation) -> list[str]:
    """Return projected input artifacts present in ``simulation``."""
    artifacts = simulation.run_data.artifact_store.scan_artifacts(INPUT_DATASET_KEYS)
    return [key for key in INPUT_DATASET_KEYS if key in artifacts]


def prepare_paper_inputs(settings: PaperSimulationSettings = SETTINGS) -> Path:
    """Project the paper-simulation inputs into a reusable package."""
    import apexpy
    import dipole
    import pyamps
    import pyhwm2014
    from lompe import conductance

    input_directory = Path(settings.input_directory).expanduser()
    input_directory.mkdir(parents=True, exist_ok=True)
    clear_prepared_input_package(input_directory, artifact_storage=settings.artifact_storage)
    (input_directory / "paper_input_metadata.json").unlink(missing_ok=True)

    print(f"Writing paper input package: {input_directory}", flush=True)
    simulation = pynamit.Simulation(
        run_directory=input_directory,
        Nmax=settings.nmax,
        Mmax=settings.mmax,
        Ncs=settings.ncs,
        RI=RI,
        main_field_kind="igrf",
        main_field_epoch=decimal_year(settings.date),
        artifact_storage=settings.artifact_storage,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=False,
        interhemispheric_coupling_latitude=settings.interhemispheric_coupling_latitude,
        area_weighted_least_squares=settings.area_weighted_least_squares,
        t0=str(settings.date),
    )

    conductance_lat = simulation.geometry.model_grid.lat
    conductance_lon = simulation.geometry.model_grid.lon
    hall, pedersen = conductance.hardy_EUV(
        conductance_lon, conductance_lat, settings.kp, settings.date, starlight=1, dipole=False
    )
    simulation.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        reg_lambda=settings.conductance_lambda,
    )

    jr_lat = simulation.geometry.model_grid.lat
    jr_lon = simulation.geometry.model_grid.lon
    dipole_model = dipole.Dipole(settings.date.year)
    apex = apexpy.Apex(refh=(RI - EARTH_RADIUS_M) * 1e-3, date=settings.date.year)
    mlat, mlon = apex.geo2apex(jr_lat, jr_lon, (RI - EARTH_RADIUS_M) * 1e-3)
    mlt = dipole_model.mlon2mlt(mlon, settings.date)
    amps = pyamps.AMPS(400, 5, -5, dipole_model.tilt(settings.date), 100, minlat=50)
    jr = amps.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
    jr[np.abs(jr_lat) < 50] = 0.0
    simulation.set_boundary_jr(jr, lat=jr_lat, lon=jr_lon, reg_lambda=settings.boundary_jr_lambda)

    hwm14 = pyhwm2014.HWM142D(
        alt=110.0,
        ap=[35, 35],
        glatlim=[-88.5, 88.5],
        glatstp=1.5,
        glonlim=[-180.0, 180.0],
        glonstp=3.0,
        option=6,
        verbose=False,
        ut=settings.date.hour + settings.date.minute / 60,
        day=settings.date.timetuple().tm_yday,
    )
    u_theta, u_phi = (-hwm14.Vwind.flatten(), hwm14.Uwind.flatten())
    u_lat, u_lon = np.meshgrid(hwm14.glatbins, hwm14.glonbins, indexing="ij")
    simulation.set_neutral_wind(
        u_theta=u_theta,
        u_phi=u_phi,
        lat=u_lat,
        lon=u_lon,
        sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
        reg_lambda=settings.wind_lambda,
    )

    write_input_manifest(
        input_directory,
        simulation.run_data.config,
        input_datasets=_prepared_input_datasets(simulation),
        source="scripts.simulation.pynamit_paper_simulation",
        notes=[
            "Paper-style inputs: Hardy EUV conductance, HWM14 wind, AMPS upward current.",
            "The run script loads conductance/wind first, imposes equilibrium, "
            "then enables jr for the second phase.",
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
    run_directory = Path(settings.run_directory).expanduser()
    print(f"Using paper input package: {input_directory}", flush=True)
    print(f"Writing paper run: {run_directory}", flush=True)

    simulation = run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        enabled_inputs=("conductance", "u"),
        final_time=settings.simulation_time,
        dt=settings.dt,
        saving_sample_interval=settings.saving_sample_interval,
        main_field_kind="igrf",
        fac_integration_radii=dipole_fac_integration_radii(
            RI, RI / np.cos(np.deg2rad(69.0)) ** 2, n_points=70
        ),
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=settings.interhemispheric_coupling_latitude,
        interhemispheric_electric_field_weight=1e-5,
        equilibrium_initialization=False,
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
        saving_sample_interval=settings.saving_sample_interval,
        equilibrium_initialization=False,
        run_dynamic=True,
        run_equilibrium=False,
    )

    manifest_path = Path(simulation.run_data.run_directory) / RUN_MANIFEST_FILENAME
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.update(
        {
            "kind": "pynamit_paper_run",
            "paper_two_phase_run": {
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
    """Run the configured prepare/run workflow."""
    if settings.prepare_inputs:
        prepare_paper_inputs(settings)
    if settings.run_simulation:
        run_paper_simulation(settings)


if __name__ == "__main__":
    main()
