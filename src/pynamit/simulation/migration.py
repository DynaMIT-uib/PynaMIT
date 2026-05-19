"""Saved-run storage migration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from pynamit.primitives.io import DATAARRAY_ARTIFACTS, IO


@dataclass(frozen=True)
class StorageMigrationReport:
    """Summary of one explicit storage migration."""

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


def _migrate_io_storage(
    io: IO, target_storage: str, *, print_info: bool = False
) -> StorageMigrationReport:
    """Convert all detected artifacts for one IO helper."""
    normalized_target = str(target_storage).strip().lower()
    if normalized_target not in {"netcdf", "zarr"}:
        raise ValueError(
            f"target_storage must be one of {{'netcdf', 'zarr'}}, got {target_storage!r}."
        )
    if normalized_target == "zarr" and not IO.zarr_available():
        raise ImportError(
            "Zarr migration requested but the optional 'zarr' dependency is not installed."
        )

    io.set_preferred_dataset_storage(normalized_target)
    artifacts = io.scan_run_artifacts()

    if "settings" not in artifacts:
        raise ValueError(f"No settings artifact found for {str(io.run_directory)!r}.")

    migrated_artifacts: list[str] = []
    unchanged_artifacts: list[str] = []

    for name in sorted(artifacts):
        storages = artifacts[name]
        if len(storages) > 1:
            raise ValueError(
                f"Artifact {name!r} exists as both NetCDF and Zarr for "
                f"{str(io.run_directory)!r}. Remove one copy before migrating."
            )

        source_storage = storages[0]
        if source_storage == normalized_target:
            unchanged_artifacts.append(name)
            continue

        if name in DATAARRAY_ARTIFACTS:
            artifact = io.load_dataarray(name, print_info=print_info, storage=source_storage)
            if artifact is None:
                raise ValueError(f"Could not load data array artifact {name!r}.")
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
                raise ValueError(f"Could not load dataset artifact {name!r}.")
            artifact.load()
            try:
                io.save_dataset(artifact, name, print_info=print_info, storage=normalized_target)
            finally:
                close = getattr(artifact, "close", None)
                if callable(close):
                    close()

        _remove_artifact(io._path_for(name, storage=source_storage))
        migrated_artifacts.append(name)

    return StorageMigrationReport(
        run_directory=str(io.run_directory),
        target_storage=normalized_target,
        migrated_artifacts=tuple(migrated_artifacts),
        unchanged_artifacts=tuple(unchanged_artifacts),
    )


def migrate_run_storage(
    run_directory: str | Path, target_storage: str, *, print_info: bool = False
) -> StorageMigrationReport:
    """Convert saved run-directory artifacts between NetCDF and Zarr."""
    io = IO(run_directory, preferred_dataset_storage=target_storage)
    return _migrate_io_storage(io, target_storage, print_info=print_info)
