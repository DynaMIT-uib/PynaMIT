"""Project prepared MAGE/GAMERA/TIEGCM forcing into PynaMIT inputs.

Run ``mage_prepare.py`` first. Edit ``SETTINGS`` below to choose
projection resolutions and numerical regularization. Each resulting
input package can be reused by any number of ``mage_run.py``
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pynamit.workflows.mage.projection import prepare_inputs

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIRECTORY = SCRIPT_DIR / "mage_output" / "2011-10-24"
DEFAULT_FORCING_PATH = CASE_DIRECTORY / "forcing.h5"
DEFAULT_RESOLUTIONS_DIRECTORY = CASE_DIRECTORY / "resolutions"


@dataclass(frozen=True)
class ProjectionSettings:
    """Editable settings for a MAGE input-projection sweep."""

    forcing_path: Path = DEFAULT_FORCING_PATH
    resolutions_directory: Path = DEFAULT_RESOLUTIONS_DIRECTORY
    resolutions: tuple[int, ...] = (20, 40, 60, 80)
    projection_name: str = "default"
    cache_operators: bool = True
    dipole_B0: float | None = None
    boundary_radius: float | None = None
    max_steps: int | None = None
    boundary_Br_lambda: float = 0.1
    conductance_lambda: float = 0.1
    boundary_jr_lambda: float = 0.1
    e_neutral_wind_lambda: float = 0.1
    artifact_storage: str = "auto"
    write_diagnostics: bool = True
    diagnostic_steps: tuple[int, ...] | None = None
    diagnostic_fields: tuple[str, ...] = ("etaP", "etaH", "SigmaP", "SigmaH", "jr", "Br")


SETTINGS = ProjectionSettings()


def main(settings: ProjectionSettings = SETTINGS) -> None:
    """Project prepared MAGE forcing at every configured resolution."""
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
    projection_name = settings.projection_name.strip()
    if (
        not projection_name
        or projection_name in {".", ".."}
        or Path(projection_name).name != projection_name
    ):
        raise ValueError("projection_name must be one non-empty directory name.")

    for index, resolution in enumerate(resolutions, start=1):
        resolution_name = f"N{resolution}_M{resolution}_Ncs{resolution}"
        resolution_directory = Path(settings.resolutions_directory).expanduser() / resolution_name
        projection_directory = resolution_directory / "projections" / projection_name
        operator_cache_directory = (
            resolution_directory / "operator_cache" if settings.cache_operators else None
        )
        print(
            f"[{index}/{len(resolutions)}] Projecting {resolution_name} to {projection_directory}",
            flush=True,
        )
        input_directory = prepare_inputs(
            forcing_path=settings.forcing_path,
            input_directory=projection_directory,
            dipole_B0_override=settings.dipole_B0,
            boundary_radius_override=settings.boundary_radius,
            nmax=resolution,
            mmax=resolution,
            ncs=resolution,
            max_steps=settings.max_steps,
            boundary_Br_lambda=settings.boundary_Br_lambda,
            conductance_lambda=settings.conductance_lambda,
            boundary_jr_lambda=settings.boundary_jr_lambda,
            e_neutral_wind_lambda=settings.e_neutral_wind_lambda,
            artifact_storage=settings.artifact_storage,
            operator_cache_directory=operator_cache_directory,
        )
        if settings.write_diagnostics:
            from pynamit.workflows.mage import write_input_projection_diagnostics

            write_input_projection_diagnostics(
                settings.forcing_path,
                input_directory,
                timesteps=settings.diagnostic_steps,
                fields=settings.diagnostic_fields,
                operator_cache_directory=operator_cache_directory,
            )


if __name__ == "__main__":
    main()
