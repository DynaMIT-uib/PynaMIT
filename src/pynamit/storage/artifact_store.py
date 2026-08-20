"""Named artifact persistence for one directory.

The storage API uses fixed artifact names like
``settings.zarr`` and ``state.zarr``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

NETCDF_SUFFIX = ".ncdf"
ZARR_SUFFIX = ".zarr"
DATASET_STORAGE_KINDS = frozenset({"auto", "netcdf", "zarr"})
ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None
ZARR_WRITE_KWARGS = {"write_empty_chunks": True, "consolidated": False}
ZARR_OPEN_KWARGS = {"consolidated": False}
ZARR_READ_CONFIG = {"array.read_missing_chunks": False}


class ArtifactStore:
    """Handle persisted named artifacts for one directory."""

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        *,
        preferred_dataset_storage: str = "auto",
    ):
        """Bind named-artifact persistence to one directory.

        Parameters
        ----------
        directory : str or Path, optional
            Directory holding fixed artifact names.
        preferred_dataset_storage : {"auto", "netcdf", "zarr"}, optional
            Default storage format for new artifacts.
        """
        self.directory = None if directory is None else str(Path(directory).resolve())
        self.preferred_dataset_storage = self._normalize_storage_kind(preferred_dataset_storage)

    @staticmethod
    def create_temporary_directory(parent: str | os.PathLike[str] | None = None) -> str:
        """Create and return a unique writable artifact directory."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if parent is None:
            return tempfile.mkdtemp(prefix=f"pynamit-simulation-{stamp}-")
        parent_path = Path(parent).resolve()
        parent_path.mkdir(parents=True, exist_ok=True)
        return tempfile.mkdtemp(prefix=f"simulation-{stamp}-", dir=parent_path)

    @staticmethod
    def require_artifact_directory(directory: str | os.PathLike[str], required_names) -> str:
        """Verify named artifacts and return their directory."""
        if isinstance(required_names, str):
            raise TypeError("required_names must be a collection of artifact names, not a string.")
        names = tuple(required_names)
        if not names:
            raise ValueError("required_names must contain at least one artifact name.")

        store = ArtifactStore(directory)
        missing = [name for name in names if not store.get_dataset_storage_kinds(name)]
        if missing:
            raise ValueError(
                f"Artifact directory {store.directory!r} is missing required "
                f"artifact(s): {sorted(missing)}."
            )
        return store.directory

    @staticmethod
    def zarr_available() -> bool:
        """Return whether optional ``zarr`` is available."""
        return bool(ZARR_AVAILABLE)

    @staticmethod
    def _zarr_config_context():
        """Return a context that fails on missing zarr chunks."""
        if not ArtifactStore.zarr_available():
            return nullcontext()
        zarr = importlib.import_module("zarr")
        return zarr.config.set(ZARR_READ_CONFIG)

    @staticmethod
    def _normalize_storage_kind(storage: str) -> str:
        """Return a normalized storage kind or raise a clear error."""
        normalized = str(storage).strip().lower()
        if normalized not in DATASET_STORAGE_KINDS:
            raise ValueError(
                f"Unsupported dataset storage kind {storage!r}. "
                f"Expected one of {sorted(DATASET_STORAGE_KINDS)}."
            )
        return normalized

    def _path_for(self, name: str, *, storage: str) -> Path:
        """Return the persisted path for one named artifact."""
        if self.directory is None:
            raise ValueError("No artifact directory configured. Cannot build file path.")
        name = str(name)
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or Path(name).name != name
        ):
            raise ValueError(f"Artifact name must be one path-safe name, got {name!r}.")

        if storage == "netcdf":
            suffix = NETCDF_SUFFIX
        elif storage == "zarr":
            suffix = ZARR_SUFFIX
        else:
            raise ValueError(
                f"Unsupported dataset storage kind {storage!r}. "
                f"Expected one of {sorted(DATASET_STORAGE_KINDS - {'auto'})}."
            )
        return Path(self.directory) / f"{name}{suffix}"

    @staticmethod
    def _requires_materialization(data) -> bool:
        """Return whether xarray data must be loaded before writing."""
        return not isinstance(data, np.ndarray)

    @classmethod
    def _prepare_dataset_for_zarr_write(cls, dataset: xr.Dataset) -> xr.Dataset:
        """Return a materialized dataset ready for Zarr writes."""
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
        """Return a materialized data array ready for Zarr writes."""
        if cls._requires_materialization(dataarray.data):
            prepared = dataarray.load()
        else:
            prepared = dataarray.copy(deep=False)
        prepared.encoding = dict(prepared.encoding)
        prepared.encoding.pop("chunks", None)
        prepared.encoding.pop("preferred_chunks", None)
        return prepared

    @staticmethod
    def _remove_path(path: Path) -> None:
        """Remove one file or directory path if it exists."""
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    @classmethod
    def _write_zarr_atomically(cls, target: Path, write_fn: Callable[[Path], None]) -> None:
        """Write a complete Zarr store via a temporary directory."""
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_store = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
        try:
            write_fn(temp_store)
            if target.exists():
                cls._remove_path(target)
            os.replace(temp_store, target)
        except Exception:
            cls._remove_path(temp_store)
            raise

    @classmethod
    def _write_netcdf_atomically(cls, target: Path, write_fn: Callable[[Path], None]) -> None:
        """Write a complete NetCDF file via a temporary file."""
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.tmp-", suffix=target.suffix, dir=str(target.parent)
        )
        os.close(descriptor)
        temp_target = Path(temp_name)
        try:
            write_fn(temp_target)
            os.replace(temp_target, target)
        except Exception:
            cls._remove_path(temp_target)
            raise

    def default_dataset_storage_kind(self) -> str:
        """Return the preferred storage format for one new artifact."""
        if self.preferred_dataset_storage != "auto":
            return self.preferred_dataset_storage
        return "zarr" if self.zarr_available() else "netcdf"

    def get_dataset_storage_kinds(self, name: str) -> tuple[str, ...]:
        """Return all on-disk storage kinds present for one artifact."""
        if self.directory is None:
            return ()

        storages: list[str] = []
        if self._path_for(name, storage="zarr").exists():
            storages.append("zarr")
        if self._path_for(name, storage="netcdf").exists():
            storages.append("netcdf")
        return tuple(storages)

    def get_dataset_storage_kind(self, name: str) -> str | None:
        """Return the unique existing storage for one artifact."""
        storages = self.get_dataset_storage_kinds(name)
        if len(storages) > 1:
            raise ValueError(
                f"Artifact {name!r} has ambiguous storage representations "
                f"{list(storages)} in {self.directory!r}. Remove the stale representation."
            )
        return storages[0] if storages else None

    def existing_artifact_path(self, name: str) -> Path | None:
        """Return the preferred path for an existing artifact."""
        storage = self.get_dataset_storage_kind(name)
        return None if storage is None else self._path_for(name, storage=storage)

    def scan_artifacts(self, names) -> dict[str, tuple[str, ...]]:
        """Return requested artifact names present in this directory."""
        return {
            name: storages for name in names if (storages := self.get_dataset_storage_kinds(name))
        }

    def remove_artifact(self, name: str) -> None:
        """Remove all on-disk representations of one artifact."""
        for storage in ("zarr", "netcdf"):
            self._remove_path(self._path_for(name, storage=storage))

    def _resolve_dataset_storage_kind(self, name: str, storage: str | None) -> str:
        """Choose the storage kind for one dataset save."""
        normalized = "auto" if storage is None else str(storage)
        normalized = self._normalize_storage_kind(normalized)

        if normalized == "auto":
            existing = self.get_dataset_storage_kind(name)
            if existing is not None:
                normalized = existing
            else:
                normalized = self.default_dataset_storage_kind()

        if normalized == "zarr" and not self.zarr_available():
            raise ImportError(
                "Zarr storage requested but the optional 'zarr' dependency is not installed."
            )

        return normalized

    def _resolve_existing_dataset_storage_kind(self, name: str, storage: str | None) -> str | None:
        """Return existing storage for one dataset load."""
        if self.directory is None:
            return None

        normalized = "auto" if storage is None else str(storage)
        normalized = self._normalize_storage_kind(normalized)
        if normalized == "auto":
            normalized = self.get_dataset_storage_kind(name)
            if normalized is None:
                return None
        else:
            filename = self._path_for(name, storage=normalized)
            if not filename.exists():
                return None

        if normalized == "zarr" and not self.zarr_available():
            raise ImportError(
                "Zarr storage requested but the optional 'zarr' dependency is not installed."
            )
        return normalized

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
        if self.directory is None:
            raise ValueError("No artifact directory configured. Cannot save Dataset.")

        storage_kind = self._resolve_dataset_storage_kind(name, storage)
        filename = self._path_for(name, storage=storage_kind)
        filename.parent.mkdir(parents=True, exist_ok=True)

        if storage_kind == "zarr":
            dataset = self._prepare_dataset_for_zarr_write(dataset)
            if append_dim is None:
                self._write_zarr_atomically(
                    filename,
                    lambda temp_store: dataset.to_zarr(temp_store, mode="w", **ZARR_WRITE_KWARGS),
                )
            else:
                dataset.to_zarr(filename, append_dim=append_dim, **ZARR_WRITE_KWARGS)
        else:
            self._write_netcdf_atomically(filename, lambda temp_file: dataset.to_netcdf(temp_file))

        alternate_storage = "netcdf" if storage_kind == "zarr" else "zarr"
        self._remove_path(self._path_for(name, storage=alternate_storage))

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
            with self._zarr_config_context():
                return xr.open_zarr(filename, **ZARR_OPEN_KWARGS)
        return xr.load_dataset(filename)

    def load_dataarray(self, name, print_info=False, *, storage: str | None = None):
        """Load a DataArray from the available storage backend."""
        storage_kind = self._resolve_existing_dataset_storage_kind(name, storage=storage)
        if storage_kind is None:
            return None

        filename = self._path_for(name, storage=storage_kind)
        if print_info:
            print(f"Loading DataArray from {filename}", flush=True)

        if storage_kind == "zarr":
            with self._zarr_config_context():
                dataset = xr.open_zarr(filename, **ZARR_OPEN_KWARGS)
            data_var_names = list(dataset.data_vars)
            if len(data_var_names) != 1:
                raise ValueError(
                    f"Expected exactly one data variable in Zarr store {filename}, "
                    f"found {data_var_names}."
                )
            return dataset[data_var_names[0]]
        return xr.load_dataarray(filename)

    def save_dataarray(self, dataarray, name, print_info=False, *, storage: str | None = None):
        """Save a DataArray using the configured storage backend."""
        if self.directory is None:
            raise ValueError("No artifact directory configured. Cannot save DataArray.")

        storage_kind = self._resolve_dataset_storage_kind(name, storage=storage)
        filename = self._path_for(name, storage=storage_kind)
        filename.parent.mkdir(parents=True, exist_ok=True)

        if storage_kind == "zarr":
            dataarray = self._prepare_dataarray_for_zarr_write(dataarray)
            self._write_zarr_atomically(
                filename,
                lambda temp_store: dataarray.to_zarr(temp_store, mode="w", **ZARR_WRITE_KWARGS),
            )
        else:
            self._write_netcdf_atomically(
                filename, lambda temp_file: dataarray.to_netcdf(temp_file)
            )

        alternate_storage = "netcdf" if storage_kind == "zarr" else "zarr"
        self._remove_path(self._path_for(name, storage=alternate_storage))

        if print_info:
            print(f"Saved DataArray to {filename}", flush=True)
