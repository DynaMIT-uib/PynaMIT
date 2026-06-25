"""Modern split version of the original PynaMIT paper simulation.

The input projection and the time evolution are intentionally separate:

``prepare_paper_inputs``
    Projects conductance, neutral wind, and AMPS radial current into a
    reusable input package.

``run_paper_simulation``
    Consumes that package.  The first phase uses conductance and wind
    only, imposes steady state, then loads the prepared radial current
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

import pynamit
from pynamit.math.constants import RE
from pynamit.primitives.io import IO
from pynamit.simulation.mainfield import decimal_year
from pynamit.simulation.prepared_inputs import (
    INPUT_DATASET_KEYS,
    RUN_MANIFEST_FILENAME,
    load_prepared_inputs_into_dynamics,
    run_pynamit_from_inputs,
    write_input_manifest,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RI = RE + 110e3


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
    latitude_boundary: float = 45.0
    dt: float = 5e-4
    saving_sample_interval: int = 200
    conductance_lambda: float = 0.001
    wind_lambda: float = 0.001
    jr_lambda: float | None = None
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


def _prepared_input_datasets(dynamics: pynamit.Dynamics) -> list[str]:
    """Return projected input artifacts present in ``dynamics``."""
    artifacts = dynamics.io.scan_run_artifacts()
    return [key for key in INPUT_DATASET_KEYS if key in artifacts]


def _dipole_radial_sampling(settings: PaperSimulationSettings) -> np.ndarray:
    """Return the field-line sampling used for PFAC integration."""
    return RI / np.cos(np.deg2rad(np.r_[0:70:1])) ** 2


def prepare_paper_inputs(settings: PaperSimulationSettings = SETTINGS) -> Path:
    """Project the paper-simulation inputs into a reusable package."""
    from lompe import conductance
    import apexpy
    import dipole
    import pyamps
    import pyhwm2014

    input_directory = Path(settings.input_directory).expanduser()
    input_directory.mkdir(parents=True, exist_ok=True)

    print(f"Writing paper input package: {input_directory}", flush=True)
    dynamics = pynamit.Dynamics(
        run_directory=input_directory,
        Nmax=settings.nmax,
        Mmax=settings.mmax,
        Ncs=settings.ncs,
        RI=RI,
        mainfield_kind="igrf",
        mainfield_epoch=decimal_year(settings.date),
        artifact_storage=settings.artifact_storage,
        ignore_PFAC=True,
        connect_hemispheres=False,
        latitude_boundary=settings.latitude_boundary,
        t0=str(settings.date),
    )

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon
    hall, pedersen = conductance.hardy_EUV(
        conductance_lon, conductance_lat, settings.kp, settings.date, starlight=1, dipole=False
    )
    dynamics.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        reg_lambda=settings.conductance_lambda,
    )

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    dipole_model = dipole.Dipole(settings.date.year)
    apex = apexpy.Apex(refh=(RI - RE) * 1e-3, date=settings.date.year)
    mlat, mlon = apex.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
    mlt = dipole_model.mlon2mlt(mlon, settings.date)
    amps = pyamps.AMPS(400, 5, -5, dipole_model.tilt(settings.date), 100, minlat=50)
    jr = amps.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
    jr[np.abs(jr_lat) < 50] = 0.0
    dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon, reg_lambda=settings.jr_lambda)

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
    dynamics.set_neutral_wind(
        u_theta=u_theta,
        u_phi=u_phi,
        lat=u_lat,
        lon=u_lon,
        sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
        reg_lambda=settings.wind_lambda,
    )

    write_input_manifest(
        input_directory,
        dynamics.settings,
        input_datasets=_prepared_input_datasets(dynamics),
        source="scripts.simulation.pynamit_paper_simulation",
        notes=[
            "Paper-style inputs: Hardy EUV conductance, HWM14 wind, AMPS upward current.",
            "The run script loads conductance/wind first, imposes steady state, "
            "then enables jr for the second phase.",
        ],
    )
    metadata = {
        "kind": "pynamit_paper_input_metadata",
        "date": settings.date,
        "kp": settings.kp,
        "simulation_time": settings.simulation_time,
        "latitude_boundary": settings.latitude_boundary,
    }
    (input_directory / "paper_input_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return input_directory


def run_paper_simulation(settings: PaperSimulationSettings = SETTINGS) -> pynamit.Dynamics:
    """Run the paper-style simulation from projected inputs."""
    input_directory = Path(settings.input_directory).expanduser()
    run_directory = Path(settings.run_directory).expanduser()
    print(f"Using paper input package: {input_directory}", flush=True)
    print(f"Writing paper run: {run_directory}", flush=True)

    dynamics = run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        enabled_inputs=("conductance", "u"),
        final_time=settings.simulation_time,
        dt=settings.dt,
        plotsteps=settings.saving_sample_interval,
        mainfield_kind="igrf",
        FAC_integration_steps=_dipole_radial_sampling(settings),
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=settings.latitude_boundary,
        ih_constraint_scaling=1e-5,
        steady_state_initialization=False,
        run_inductive=True,
        run_steady_state=False,
        artifact_storage=settings.artifact_storage,
    )

    print("Imposing wind/conductance steady state before enabling jr", flush=True)
    dynamics.impose_steady_state()

    input_io = IO(str(input_directory), preferred_dataset_storage=settings.artifact_storage)
    loaded_inputs = load_prepared_inputs_into_dynamics(
        dynamics,
        input_directory,
        artifact_storage=settings.artifact_storage,
        enabled_inputs=("conductance", "u", "jr"),
    )
    for key in loaded_inputs:
        dataset = input_io.load_dataset(key)
        if dataset is not None:
            dynamics.io.save_dataset(dataset, key)

    final_time = 2.0 * float(settings.simulation_time)
    print(f"Continuing with jr enabled to t={final_time:g} s", flush=True)
    dynamics.evolve_to_time(
        final_time,
        dt=settings.dt,
        sampling_step_interval=1,
        saving_sample_interval=settings.saving_sample_interval,
        steady_state_initialization=False,
        run_inductive=True,
        run_steady_state=False,
    )

    manifest_path = Path(dynamics.run_directory) / RUN_MANIFEST_FILENAME
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
                "steady_state_imposed_between_phases": True,
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return dynamics


def main() -> None:
    """Run the configured prepare/run workflow."""
    if SETTINGS.prepare_inputs:
        prepare_paper_inputs(SETTINGS)
    if SETTINGS.run_simulation:
        run_paper_simulation(SETTINGS)


if __name__ == "__main__":
    main()
