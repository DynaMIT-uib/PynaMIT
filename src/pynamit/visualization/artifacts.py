"""Xarray artifact helpers used by visualization workflows."""

import os
from pathlib import Path

import xarray as xr

from pynamit.primitives.io import IO


def resolve_xarray_artifact_path(path):
    """Resolve a base path to an existing zarr or NetCDF path."""
    path_string = os.fspath(path)
    root, ext = os.path.splitext(path_string)
    if ext in {".ncdf", ".zarr"}:
        candidates = [root + ".zarr", root + ".ncdf"]
    else:
        candidates = [path_string + ".zarr", path_string + ".ncdf", path_string]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path_string


def xarray_artifact_exists(path):
    """Return whether a zarr or NetCDF artifact exists for ``path``."""
    return os.path.exists(resolve_xarray_artifact_path(path))


def artifact_path(run_directory, name):
    """Return the conventional artifact base path for a run output."""
    return os.fspath(Path(run_directory) / name)


def load_dataset_artifact(path):
    """Load an xarray dataset from a zarr or NetCDF artifact."""
    resolved = resolve_xarray_artifact_path(path)
    if resolved.endswith(".zarr"):
        with IO._zarr_config_context():
            return xr.open_zarr(resolved)
    return xr.load_dataset(resolved)


def load_dataarray_artifact(path):
    """Load one xarray data array from a zarr or NetCDF artifact."""
    resolved = resolve_xarray_artifact_path(path)
    if resolved.endswith(".zarr"):
        with IO._zarr_config_context():
            dataset = xr.open_zarr(resolved)
        data_vars = list(dataset.data_vars)
        if len(data_vars) != 1:
            raise ValueError(
                f"Expected one data variable in {resolved!r}, found {data_vars}."
            )
        return dataset[data_vars[0]]
    return xr.load_dataarray(resolved)


__all__ = [
    "artifact_path",
    "load_dataarray_artifact",
    "load_dataset_artifact",
    "resolve_xarray_artifact_path",
    "xarray_artifact_exists",
]
