"""Saved-run storage migration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from pynamit.primitives.io import DATAARRAY_ARTIFACTS, IO


@dataclass(frozen=True)
class RunStorageMigrationReport:
    """Summary of one explicit run-storage migration."""

    run_directory: str
    target_storage: str
    migrated_artifacts: tuple[str, ...]
    unchanged_artifacts: tuple[str, ...]


def _remove_artifact(path: Path) -> None:
    """Delete one persisted artifact after successful migration."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def migrate_run_storage(
    run_directory: str | Path, target_storage: str, *, print_info: bool = False
) -> RunStorageMigrationReport:
    """Convert one saved run directory between NetCDF and Zarr.

    Parameters
    ----------
    run_directory : str or Path
        Directory containing one persisted PynaMIT run.
    target_storage : {"netcdf", "zarr"}
        Target storage format for all detected artifacts.
    print_info : bool, optional
        Whether to print xarray read/write activity during migration.
    """
    normalized_target = str(target_storage).strip().lower()
    if normalized_target not in {"netcdf", "zarr"}:
        raise ValueError(
            f"target_storage must be one of {{'netcdf', 'zarr'}}, got {target_storage!r}."
        )
    if normalized_target == "zarr" and not IO.zarr_available():
        raise ImportError(
            "Zarr migration requested but the optional 'zarr' dependency is not installed."
        )

    resolved_directory = Path(IO.discover_run_directory(run_directory))
    io = IO(resolved_directory)
    artifacts = io.scan_run_artifacts()

    if "settings" not in artifacts:
        raise ValueError(
            f"Run directory {str(resolved_directory)!r} does not contain a settings artifact."
        )

    migrated_artifacts: list[str] = []
    unchanged_artifacts: list[str] = []

    for name in sorted(artifacts):
        storages = set(artifacts[name])
        if len(storages) > 1:
            raise ValueError(
                f"Artifact {name!r} exists as both NetCDF and Zarr in "
                f"{str(resolved_directory)!r}. Remove one copy before migrating."
            )

        source_storage = next(iter(storages))
        if source_storage == normalized_target:
            unchanged_artifacts.append(name)
            continue

        if name in DATAARRAY_ARTIFACTS:
            artifact = io.load_dataarray(name, print_info=print_info, storage=source_storage)
            if artifact is None:
                raise ValueError(
                    f"Could not load data array artifact {name!r} from "
                    f"{str(resolved_directory)!r}."
                )
            artifact.load()
            try:
                io.save_dataarray(artifact, name, print_info=print_info, storage=normalized_target)
            finally:
                close = getattr(artifact, "close", None)
                if callable(close):
                    close()
        else:
            artifact = io.load_dataset(name, print_info=print_info, storage=source_storage)
            if artifact is None:
                raise ValueError(
                    f"Could not load dataset artifact {name!r} from {str(resolved_directory)!r}."
                )
            artifact.load()
            try:
                io.save_dataset(artifact, name, print_info=print_info, storage=normalized_target)
            finally:
                close = getattr(artifact, "close", None)
                if callable(close):
                    close()

        _remove_artifact(io._path_for(name, storage=source_storage))
        migrated_artifacts.append(name)

    return RunStorageMigrationReport(
        run_directory=str(resolved_directory),
        target_storage=normalized_target,
        migrated_artifacts=tuple(migrated_artifacts),
        unchanged_artifacts=tuple(unchanged_artifacts),
    )
