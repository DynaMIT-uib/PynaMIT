"""Small NetCDF persistence helper.

The IO layer owns filename construction and atomic read/write behavior for one
simulation run directory.
"""

from __future__ import annotations

from datetime import datetime
import os
import tempfile
from pathlib import Path

import xarray as xr


class IO:
    """Handle NetCDF persistence for one simulation run."""

    def __init__(self, run_directory: str | os.PathLike[str] | None):
        """Initialize the IO helper.

        Parameters
        ----------
        run_directory : str or Path, optional
            Directory holding fixed artifact names like ``settings.ncdf``.
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
    def build_run_directory(
        directory: str | os.PathLike[str],
    ) -> str:
        """Return one explicit run directory path."""
        return str(Path(directory).resolve())

    @staticmethod
    def build_temporary_run_directory_in_directory(
        directory: str | os.PathLike[str],
    ) -> str:
        """Return a unique writable run directory under one parent directory."""
        root = Path(directory).resolve()
        run_dir = IO._timestamped_tempdir(root=root, prefix="run-")
        return str(run_dir)

    @staticmethod
    def discover_run_directory(run_directory: str | os.PathLike[str]) -> str:
        """Return one run directory after verifying that ``settings.ncdf`` exists."""
        root = Path(run_directory).resolve()
        settings_path = root / "settings.ncdf"
        if not settings_path.exists():
            raise ValueError(f"No settings dataset found in run directory {str(root)!r}.")
        return str(root)

    def _path_for(self, name: str) -> Path:
        """Return the NetCDF path for one named artifact."""
        if self.run_directory is None:
            raise ValueError("No run directory configured. Cannot build file path.")
        return Path(self.run_directory) / f"{name}.ncdf"

    def save_dataset(self, dataset, name, print_info=False):
        """Save a Dataset to NetCDF file."""
        filename = self._path_for(name)
        filename.parent.mkdir(parents=True, exist_ok=True)
        tmp_filename = filename.with_suffix(filename.suffix + ".tmp")

        try:
            dataset.to_netcdf(tmp_filename)
            os.replace(tmp_filename, filename)
        except Exception as e:
            if tmp_filename.exists():
                tmp_filename.unlink()
            raise e

        if print_info:
            print(f"Saved Dataset to {filename}", flush=True)

    def load_dataset(self, name, print_info=False):
        """Load a Dataset from NetCDF file."""
        filename = self._path_for(name)
        if filename.exists():
            if print_info:
                print(f"Loading Dataset from {filename}", flush=True)
            return xr.load_dataset(filename)
        return None

    def load_dataarray(self, name, print_info=False):
        """Load a DataArray from NetCDF file."""
        filename = self._path_for(name)
        if filename.exists():
            if print_info:
                print(f"Loading DataArray from {filename}", flush=True)
            return xr.load_dataarray(filename)
        return None

    def save_dataarray(self, dataarray, name, print_info=False):
        """Save a DataArray to NetCDF file."""
        filename = self._path_for(name)
        filename.parent.mkdir(parents=True, exist_ok=True)
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
