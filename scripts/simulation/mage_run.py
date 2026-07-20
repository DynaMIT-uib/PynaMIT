"""Run PynaMIT from a reusable projected MAGE input package.

The MAGE workflow has three explicit stages:

1. ``mage_prepare.py`` creates resolution-independent forcing.
2. ``mage_project.py`` creates one input package per resolution.
3. This script creates any number of named runs from one projection.

Edit ``SETTINGS`` below for the run. Projection choices intentionally
remain in ``mage_project.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pynamit.simulation.config import SimulationConfig, dipole_fac_integration_radii
from pynamit.simulation.workflows.prepared_inputs import run_pynamit_from_inputs
from pynamit.storage import ArtifactStore

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIRECTORY = SCRIPT_DIR / "mage_cases" / "mage_2011_kaiju"
DEFAULT_PROJECTION_DIRECTORY = CASE_DIRECTORY / "projections" / "N50_M50_Ncs50"


@dataclass(frozen=True)
class RunSettings:
    """Defaults intended to be edited for a MAGE simulation run."""

    projection_directory: Path = DEFAULT_PROJECTION_DIRECTORY
    run_directory: Path | None = None
    run_name: str = "default"
    magnetic_boundary_shielding: bool = False
    fac_integration_points: int = 40
    interhemispheric_coupling_latitude: float = 35.0
    interhemispheric_electric_field_weight: float = 1e-5
    dt: float = 10.0
    final_time: float = 3600.0
    sampling_step_interval: int = 1
    saving_sample_interval: int = 1
    integrator: str = "exponential"
    m_imp_regularization_lambda: float = 0.0
    steady_state_initialization: bool = True
    run_steady_state: bool = True
    artifact_storage: str = "auto"


SETTINGS = RunSettings()


def main(settings: RunSettings = SETTINGS) -> None:
    """Run PynaMIT from an already projected MAGE input package."""
    projection_directory = Path(
        ArtifactStore.require_artifact_directory(
            Path(settings.projection_directory).expanduser(), ("settings",)
        )
    )

    input_store = ArtifactStore(
        projection_directory, preferred_dataset_storage=settings.artifact_storage
    )
    input_config = SimulationConfig.from_settings(input_store.load_dataset("settings"))
    if input_config.RM is None:
        raise RuntimeError("The projected boundary-Br input requires a finite RM.")
    fac_integration_radii = dipole_fac_integration_radii(
        input_config.RI, input_config.RM, settings.fac_integration_points
    )

    resolution = f"N{input_config.Nmax}_M{input_config.Mmax}_Ncs{input_config.Ncs}"
    if settings.run_directory is None:
        run_name = settings.run_name.strip()
        if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
            raise ValueError("run_name must be one non-empty directory name.")
        run_directory = CASE_DIRECTORY / "runs" / resolution / run_name
    else:
        run_directory = Path(settings.run_directory).expanduser()

    print(f"Using projected input package: {projection_directory}", flush=True)
    print(f"Writing run directory: {run_directory}", flush=True)
    if settings.integrator == "exponential":
        print(
            "Warning: the exponential integrator builds a dense matrix exponential at "
            "each step. Monitor progress and memory on MAGE-size runs.",
            flush=True,
        )

    run_pynamit_from_inputs(
        projection_directory,
        run_directory=run_directory,
        final_time=settings.final_time,
        dt=settings.dt,
        sampling_step_interval=settings.sampling_step_interval,
        saving_sample_interval=settings.saving_sample_interval,
        fac_integration_radii=fac_integration_radii,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=settings.interhemispheric_coupling_latitude,
        interhemispheric_electric_field_weight=settings.interhemispheric_electric_field_weight,
        magnetic_boundary_shielding=settings.magnetic_boundary_shielding,
        steady_state_initialization=settings.steady_state_initialization,
        run_inductive=True,
        run_steady_state=settings.run_steady_state,
        integrator=settings.integrator,
        m_imp_regularization_lambda=settings.m_imp_regularization_lambda,
        artifact_storage=settings.artifact_storage,
    )
    print("Time evolution complete", flush=True)


if __name__ == "__main__":
    main()
