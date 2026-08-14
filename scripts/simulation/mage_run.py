"""Run PynaMIT from a reusable projected MAGE input package.

The MAGE workflow has three explicit stages:

1. ``mage_prepare.py`` creates resolution-independent forcing.
2. ``mage_project.py`` creates one input package for each configured
   resolution.
3. This script creates one named simulation for each configured
   resolution.

Edit ``SETTINGS`` below for the simulation sweep. Completed simulations
are skipped, while interrupted simulations resume from their last saved
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pynamit.simulation.config import SimulationConfig, dipole_fac_integration_radii
from pynamit.storage import ArtifactStore
from pynamit.workflows.mage.projection import MAGE_MAIN_FIELD_KIND
from pynamit.workflows.prepared_inputs import run_from_inputs

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIRECTORY = SCRIPT_DIR / "mage_output" / "2011-10-24"
DEFAULT_RESOLUTIONS_DIRECTORY = CASE_DIRECTORY / "resolutions"


@dataclass(frozen=True)
class SimulationSweep:
    """Editable settings for a MAGE simulation sweep."""

    resolutions_directory: Path = DEFAULT_RESOLUTIONS_DIRECTORY
    resolutions: tuple[int, ...] = (20, 40, 60, 80)
    projection_name: str = "default"
    simulation_name: str = "default"
    cache_operators: bool = True
    magnetic_boundary_shielding: bool = False
    fac_integration_points: int = 40
    interhemispheric_coupling_latitude: float = 35.0
    interhemispheric_electric_field_weight: float = 1e-5
    dt: float = 10.0
    final_time: float | None = None
    sampling_step_interval: int = 1
    write_sample_interval: int = 1
    integrator: str = "exponential"
    toroidal_potential_regularization_lambda: float = 0.0
    initialize_from_equilibrium: bool = True
    run_equilibrium: bool = True
    artifact_storage: str = "auto"


SETTINGS = SimulationSweep()


@dataclass(frozen=True)
class _SimulationTarget:
    """Resolved inputs and outputs for one simulation in the sweep."""

    resolution_name: str
    projection_directory: Path
    simulation_directory: Path
    operator_cache_directory: Path | None
    final_time: float
    fac_integration_radii: np.ndarray


def _last_projected_input_time(input_store: ArtifactStore) -> float:
    """Return the final time covered by the projected MAGE inputs."""
    time = np.asarray(input_store.load_dataset("boundary_Br").time.values, dtype=float)
    if time.ndim != 1 or time.size == 0 or np.any(~np.isfinite(time)):
        raise RuntimeError(
            "Projected MAGE boundary-Br input must have a finite one-dimensional time axis."
        )
    if time[0] < 0.0 or np.any(np.diff(time) <= 0.0):
        raise RuntimeError(
            "Projected MAGE boundary-Br input times must be non-negative and increasing."
        )
    return float(time[-1])


def _build_simulation_targets(settings: SimulationSweep) -> tuple[_SimulationTarget, ...]:
    """Validate the full sweep and resolve its simulation targets."""
    resolutions = tuple(settings.resolutions)
    if not resolutions:
        raise ValueError("resolutions must contain at least one positive integer.")
    if any(
        isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0
        for resolution in resolutions
    ):
        raise ValueError("resolutions must contain only positive integers.")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must not contain duplicates.")

    simulation_name = settings.simulation_name.strip()
    if (
        not simulation_name
        or simulation_name in {".", ".."}
        or Path(simulation_name).name != simulation_name
    ):
        raise ValueError("simulation_name must be one non-empty directory name.")
    projection_name = settings.projection_name.strip()
    if (
        not projection_name
        or projection_name in {".", ".."}
        or Path(projection_name).name != projection_name
    ):
        raise ValueError("projection_name must be one non-empty directory name.")

    targets = []
    for resolution in resolutions:
        resolution_name = f"N{resolution}_M{resolution}_Ncs{resolution}"
        resolution_directory = Path(settings.resolutions_directory).expanduser() / resolution_name
        projection_directory = resolution_directory / "projections" / projection_name
        projection_directory = Path(
            ArtifactStore.require_artifact_directory(projection_directory, ("settings",))
        )
        input_store = ArtifactStore(
            projection_directory, preferred_dataset_storage=settings.artifact_storage
        )
        input_config = SimulationConfig.from_settings(input_store.load_dataset("settings"))
        if input_config.main_field_kind != MAGE_MAIN_FIELD_KIND:
            raise RuntimeError(
                f"MAGE projection {projection_directory} uses main_field_kind="
                f"{input_config.main_field_kind!r}; expected {MAGE_MAIN_FIELD_KIND!r}."
            )
        actual_resolution_name = f"N{input_config.Nmax}_M{input_config.Mmax}_Ncs{input_config.Ncs}"
        if actual_resolution_name != resolution_name:
            raise RuntimeError(
                f"Projection directory {projection_directory} contains "
                f"{actual_resolution_name}, not {resolution_name}."
            )
        if input_config.RM is None:
            raise RuntimeError("The projected boundary-Br input requires a finite RM.")
        final_time = (
            _last_projected_input_time(input_store)
            if settings.final_time is None
            else float(settings.final_time)
        )
        targets.append(
            _SimulationTarget(
                resolution_name=resolution_name,
                projection_directory=projection_directory,
                simulation_directory=resolution_directory / "simulations" / simulation_name,
                operator_cache_directory=(
                    resolution_directory / "operator_cache" if settings.cache_operators else None
                ),
                final_time=final_time,
                fac_integration_radii=dipole_fac_integration_radii(
                    input_config.RI, input_config.RM, settings.fac_integration_points
                ),
            )
        )
    return tuple(targets)


def main(settings: SimulationSweep = SETTINGS) -> None:
    """Run every configured MAGE projection."""
    targets = _build_simulation_targets(settings)
    print(
        "Magnetic-boundary shielding of induced_Br: "
        f"{'enabled' if settings.magnetic_boundary_shielding else 'disabled'}",
        flush=True,
    )
    if settings.integrator == "exponential":
        print(
            "Warning: the exponential integrator builds a dense matrix exponential at "
            "each step. Monitor progress and memory on MAGE-size simulations.",
            flush=True,
        )

    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] Using projected input package: "
            f"{target.projection_directory}",
            flush=True,
        )
        print(f"Writing simulation directory: {target.simulation_directory}", flush=True)
        print(f"Running through projected time t={target.final_time:g} s", flush=True)
        simulation = run_from_inputs(
            target.projection_directory,
            simulation_directory=target.simulation_directory,
            final_time=target.final_time,
            dt=settings.dt,
            sampling_step_interval=settings.sampling_step_interval,
            write_sample_interval=settings.write_sample_interval,
            fac_integration_radii=target.fac_integration_radii,
            enable_pfac_coupling=True,
            enable_interhemispheric_coupling=True,
            interhemispheric_coupling_latitude=settings.interhemispheric_coupling_latitude,
            interhemispheric_electric_field_weight=(
                settings.interhemispheric_electric_field_weight
            ),
            magnetic_boundary_shielding=settings.magnetic_boundary_shielding,
            initialize_from_equilibrium=settings.initialize_from_equilibrium,
            run_dynamic=True,
            run_equilibrium=settings.run_equilibrium,
            integrator=settings.integrator,
            toroidal_potential_regularization_lambda=settings.toroidal_potential_regularization_lambda,
            artifact_storage=settings.artifact_storage,
            operator_cache_directory=target.operator_cache_directory,
            skip_completed=True,
        )
        if simulation is not None:
            print(f"{target.resolution_name} time evolution complete", flush=True)


if __name__ == "__main__":
    main()
