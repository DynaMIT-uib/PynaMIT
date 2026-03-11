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

import numpy as np
import xarray as xr

NETCDF_SUFFIX = ".ncdf"
ZARR_SUFFIX = ".zarr"
DATASET_STORAGE_KINDS = frozenset({"auto", "netcdf", "zarr"})
DATAARRAY_ARTIFACTS = frozenset({"PFAC_matrix"})
RUN_ARTIFACTS = frozenset(
    {"settings", "PFAC_matrix", "jr", "Br", "conductance", "u", "state", "steady_state"}
)
ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None


class IO:
    """Handle persisted artifacts for one simulation run."""

    def __init__(
        self,
        run_directory: str | os.PathLike[str] | None,
        *,
        preferred_dataset_storage: str = "auto",
    ):
        """Initialize the IO helper.

        Parameters
        ----------
        run_directory : str or Path, optional
            Directory holding fixed artifact names like ``settings.zarr`` and
            ``state.zarr``.
        preferred_dataset_storage : {"auto", "netcdf", "zarr"}, optional
            Default storage format for new artifacts that do not yet exist.
        """
        self.run_directory = None if run_directory is None else str(Path(run_directory).resolve())
        self.preferred_dataset_storage = self._normalize_storage_kind(preferred_dataset_storage)

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

    @staticmethod
    def _normalize_storage_kind(storage: str) -> str:
        """Return normalized storage kind for explicit preferences."""
        normalized = str(storage).strip().lower()
        if normalized not in DATASET_STORAGE_KINDS:
            raise ValueError(
                f"Unsupported dataset storage kind {storage!r}. "
                f"Expected one of {sorted(DATASET_STORAGE_KINDS)}."
            )
        return normalized

    def set_preferred_dataset_storage(self, storage: str) -> None:
        """Update the default storage format for new artifacts."""
        self.preferred_dataset_storage = self._normalize_storage_kind(storage)

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

    @staticmethod
    def _requires_materialization(data) -> bool:
        """Return whether one xarray payload must be loaded before writing."""
        return not isinstance(data, np.ndarray)

    @classmethod
    def _prepare_dataset_for_zarr_write(cls, dataset: xr.Dataset) -> xr.Dataset:
        """Return a materialized dataset without stale Zarr chunk encodings."""
        if any(
            cls._requires_materialization(variable.data) for variable in dataset.data_vars.values()
        ):
            prepared = dataset.load()
        else:
            prepared = dataset.copy(deep=False)

        for variable_name in prepared.variables:
            variable = prepared[variable_name]
            variable.encoding = dict(variable.encoding)
            variable.encoding.pop("chunks", None)
            variable.encoding.pop("preferred_chunks", None)
        return prepared

    @classmethod
    def _prepare_dataarray_for_zarr_write(cls, dataarray: xr.DataArray) -> xr.DataArray:
        """Return a materialized data array without stale Zarr chunk encodings."""
        if cls._requires_materialization(dataarray.data):
            prepared = dataarray.load()
        else:
            prepared = dataarray.copy(deep=False)
        prepared.encoding = dict(prepared.encoding)
        prepared.encoding.pop("chunks", None)
        prepared.encoding.pop("preferred_chunks", None)
        return prepared

    def default_dataset_storage_kind(self, name: str) -> str:
        """Return the preferred storage format for one dataset artifact."""
        if self.preferred_dataset_storage != "auto":
            return self.preferred_dataset_storage
        return "zarr" if self.zarr_available() else "netcdf"

    def get_dataset_storage_kinds(self, name: str) -> tuple[str, ...]:
        """Return all on-disk storage kinds present for one artifact."""
        storages: list[str] = []
        if self._path_for(name, storage="zarr").exists():
            storages.append("zarr")
        if self._path_for(name, storage="netcdf").exists():
            storages.append("netcdf")
        return tuple(storages)

    def get_dataset_storage_kind(self, name: str) -> str | None:
        """Return the on-disk storage kind for one dataset artifact, if any."""
        storages = self.get_dataset_storage_kinds(name)
        if "zarr" in storages:
            return "zarr"
        if "netcdf" in storages:
            return "netcdf"
        return None

    def scan_run_artifacts(self) -> dict[str, tuple[str, ...]]:
        """Return known persisted artifacts present in this run directory."""
        artifacts: dict[str, tuple[str, ...]] = {}
        for name in RUN_ARTIFACTS:
            storages = self.get_dataset_storage_kinds(name)
            if storages:
                artifacts[name] = storages
        return artifacts

    def _resolve_dataset_storage_kind(self, name: str, storage: str | None) -> str:
        """Choose the storage kind for one dataset save."""
        normalized = "auto" if storage is None else str(storage)
        normalized = self._normalize_storage_kind(normalized)

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
        normalized = "auto" if storage is None else str(storage)
        normalized = self._normalize_storage_kind(normalized)
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
            dataset = self._prepare_dataset_for_zarr_write(dataset)
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
            dataarray = self._prepare_dataarray_for_zarr_write(dataarray)
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
