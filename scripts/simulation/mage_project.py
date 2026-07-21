"""Project prepared MAGE/GAMERA/TIEGCM forcing into PynaMIT inputs.

Run ``mage_prepare.py`` first. Edit ``SETTINGS`` below to choose the
projection resolution and numerical regularization. The resulting input
package can be reused by any number of ``mage_run.py`` experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pynamit.simulation.workflows.mage_projection import project_inputs

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FORCING_PATH = SCRIPT_DIR / "mage_prepared" / "mage_prepared_forcing.h5"
CASE_DIRECTORY = SCRIPT_DIR / "mage_cases" / "mage_2011_kaiju"


@dataclass(frozen=True)
class ProjectionSettings:
    """Editable settings for one MAGE input projection."""

    forcing_path: Path = DEFAULT_FORCING_PATH
    projection_directory: Path | None = None
    dipole_B0: float | None = None
    boundary_radius: float | None = None
    nmax: int = 50
    mmax: int = 50
    ncs: int = 50
    max_steps: int | None = None
    br_lambda: float = 0.1
    conductance_lambda: float = 3.0
    jr_lambda: float = 0.1
    e_source_lambda: float = 0.1
    artifact_storage: str = "auto"


SETTINGS = ProjectionSettings()


def main(settings: ProjectionSettings = SETTINGS) -> None:
    """Project prepared MAGE forcing using the edited settings."""
    resolution = f"N{settings.nmax}_M{settings.mmax}_Ncs{settings.ncs}"
    projection_directory = (
        settings.projection_directory or CASE_DIRECTORY / "projections" / resolution
    )
    project_inputs(
        forcing_path=settings.forcing_path,
        projection_directory=projection_directory,
        dipole_B0_override=settings.dipole_B0,
        boundary_radius_override=settings.boundary_radius,
        nmax=settings.nmax,
        mmax=settings.mmax,
        ncs=settings.ncs,
        max_steps=settings.max_steps,
        br_lambda=settings.br_lambda,
        conductance_lambda=settings.conductance_lambda,
        jr_lambda=settings.jr_lambda,
        e_source_lambda=settings.e_source_lambda,
        artifact_storage=settings.artifact_storage,
    )


if __name__ == "__main__":
    main()
