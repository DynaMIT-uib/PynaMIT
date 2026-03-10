"""Persistence helper for one simulation run directory.

The IO layer owns artifact-path construction plus the read/write behavior for
one run directory. New artifacts prefer Zarr when the optional dependency is
available, while existing artifacts keep their established on-disk format.
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import os
import tempfile
from pathlib import Path

import xarray as xr

NETCDF_SUFFIX = ".ncdf"
ZARR_SUFFIX = ".zarr"
DATASET_STORAGE_KINDS = frozenset({"auto", "netcdf", "zarr"})
DATAARRAY_ARTIFACTS = frozenset({"PFAC_matrix"})
ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None


class IO:
    """Handle persisted artifacts for one simulation run."""

    def __init__(self, run_directory: str | os.PathLike[str] | None):
        """Initialize the IO helper.

        Parameters
        ----------
        run_directory : str or Path, optional
            Directory holding fixed artifact names like ``settings.zarr`` and
            ``state.zarr``.
        """
        self.run_directory = None if run_directory is None else str(Path(run_directory).resolve())

    @staticmethod
    def _timestamped_tempdir(*, root: Path | None = None, prefix: str) -> Path:
        """Create a unique run directory with a readable timestamp prefix."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if root is None:
            return Path(tempfile.mkdtemp(prefix=f"{prefix}{stamp}-"))
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{prefix}{stamp}-", dir=root))

    @staticmethod
    def build_temporary_run_directory() -> str:
        """Return a writable temporary run directory for one run."""
        return str(IO._timestamped_tempdir(prefix="pynamit-run-"))

    @staticmethod
    def build_run_directory(directory: str | os.PathLike[str]) -> str:
        """Return one explicit run directory path."""
        return str(Path(directory).resolve())

    @staticmethod
    def build_temporary_run_directory_in_directory(directory: str | os.PathLike[str]) -> str:
        """Return a unique writable run directory under one parent directory."""
        root = Path(directory).resolve()
        run_dir = IO._timestamped_tempdir(root=root, prefix="run-")
        return str(run_dir)

    @staticmethod
    def discover_run_directory(run_directory: str | os.PathLike[str]) -> str:
        """Return one run directory after verifying that settings exist."""
        root = Path(run_directory).resolve()
        settings_paths = [root / f"settings{ZARR_SUFFIX}", root / f"settings{NETCDF_SUFFIX}"]
        if not any(path.exists() for path in settings_paths):
            raise ValueError(f"No settings dataset found in run directory {str(root)!r}.")
        return str(root)

    @staticmethod
    def zarr_available() -> bool:
        """Return whether the optional ``zarr`` dependency is available."""
        return bool(ZARR_AVAILABLE)

    def _path_for(self, name: str, *, storage: str) -> Path:
        """Return the persisted path for one named artifact."""
        if self.run_directory is None:
            raise ValueError("No run directory configured. Cannot build file path.")
        if storage == "netcdf":
            suffix = NETCDF_SUFFIX
        elif storage == "zarr":
            suffix = ZARR_SUFFIX
        else:
            raise ValueError(
                f"Unsupported dataset storage kind {storage!r}. "
                f"Expected one of {sorted(DATASET_STORAGE_KINDS - {'auto'})}."
            )
        return Path(self.run_directory) / f"{name}{suffix}"

    def default_dataset_storage_kind(self, name: str) -> str:
        """Return the preferred storage format for one dataset artifact."""
        return "zarr" if self.zarr_available() else "netcdf"

    def get_dataset_storage_kind(self, name: str) -> str | None:
        """Return the on-disk storage kind for one dataset artifact, if any."""
        netcdf_path = self._path_for(name, storage="netcdf")
        zarr_path = self._path_for(name, storage="zarr")
        if zarr_path.exists():
            return "zarr"
        if netcdf_path.exists():
            return "netcdf"
        return None

    def _resolve_dataset_storage_kind(self, name: str, storage: str | None) -> str:
        """Choose the storage kind for one dataset save."""
        normalized = "auto" if storage is None else str(storage).strip().lower()
        if normalized not in DATASET_STORAGE_KINDS:
            raise ValueError(
                f"Unsupported dataset storage kind {storage!r}. "
                f"Expected one of {sorted(DATASET_STORAGE_KINDS)}."
            )

        if normalized == "auto":
            existing = self.get_dataset_storage_kind(name)
            if existing is not None:
                return existing
            normalized = self.default_dataset_storage_kind(name)

        if normalized == "zarr" and not self.zarr_available():
            raise ImportError(
                "Zarr storage requested but the optional 'zarr' dependency is not installed."
            )

        return normalized

    def _resolve_existing_dataset_storage_kind(self, name: str, storage: str | None) -> str | None:
        """Return the existing storage kind for one dataset load, if available."""
        normalized = "auto" if storage is None else str(storage).strip().lower()
        if normalized not in DATASET_STORAGE_KINDS:
            raise ValueError(
                f"Unsupported dataset storage kind {storage!r}. "
                f"Expected one of {sorted(DATASET_STORAGE_KINDS)}."
            )
        if normalized == "auto":
            return self.get_dataset_storage_kind(name)
        filename = self._path_for(name, storage=normalized)
        return normalized if filename.exists() else None

    def save_dataset(
        self,
        dataset,
        name,
        print_info=False,
        *,
        storage: str | None = None,
        append_dim: str | None = None,
    ):
        """Persist one Dataset using the configured storage backend."""
        storage_kind = self._resolve_dataset_storage_kind(name, storage)
        filename = self._path_for(name, storage=storage_kind)
        filename.parent.mkdir(parents=True, exist_ok=True)

        if storage_kind == "zarr":
            if append_dim is None:
                dataset.to_zarr(filename, mode="w")
            else:
                dataset.to_zarr(filename, append_dim=append_dim)
        else:
            tmp_filename = filename.with_suffix(filename.suffix + ".tmp")
            try:
                dataset.to_netcdf(tmp_filename)
                os.replace(tmp_filename, filename)
            except Exception as e:
                if tmp_filename.exists():
                    tmp_filename.unlink()
                raise e

        if print_info:
            suffix = "" if append_dim is None else f" (append_dim={append_dim!r})"
            print(f"Saved Dataset to {filename}{suffix}", flush=True)

    def load_dataset(self, name, print_info=False, *, storage: str | None = None):
        """Load one Dataset from the available storage backend."""
        storage_kind = self._resolve_existing_dataset_storage_kind(name, storage)
        if storage_kind is None:
            return None

        filename = self._path_for(name, storage=storage_kind)
        if print_info:
            print(f"Loading Dataset from {filename}", flush=True)

        if storage_kind == "zarr":
            return xr.open_zarr(filename)
        if filename.exists():
            return xr.load_dataset(filename)
        return None

    def load_dataarray(self, name, print_info=False, *, storage: str | None = None):
        """Load a DataArray from the available storage backend."""
        storage_kind = self._resolve_existing_dataset_storage_kind(name, storage=storage)
        if storage_kind is None:
            return None

        filename = self._path_for(name, storage=storage_kind)
        if print_info:
            print(f"Loading DataArray from {filename}", flush=True)

        if storage_kind == "zarr":
            dataset = xr.open_zarr(filename)
            data_var_names = list(dataset.data_vars)
            if len(data_var_names) != 1:
                raise ValueError(
                    f"Expected exactly one data variable in Zarr store {filename}, "
                    f"found {data_var_names}."
                )
            return dataset[data_var_names[0]]
        if filename.exists():
            return xr.load_dataarray(filename)
        return None

    def save_dataarray(self, dataarray, name, print_info=False, *, storage: str | None = None):
        """Save a DataArray using the configured storage backend."""
        storage_kind = self._resolve_dataset_storage_kind(name, storage=storage)
        filename = self._path_for(name, storage=storage_kind)
        filename.parent.mkdir(parents=True, exist_ok=True)

        if storage_kind == "zarr":
            dataarray.to_zarr(filename, mode="w")
        else:
            tmp_filename = filename.with_suffix(filename.suffix + ".tmp")

            try:
                dataarray.to_netcdf(tmp_filename)
                os.replace(tmp_filename, filename)
            except Exception as e:
                if tmp_filename.exists():
                    tmp_filename.unlink()
                raise e

        if print_info:
            print(f"Saved DataArray to {filename}", flush=True)
