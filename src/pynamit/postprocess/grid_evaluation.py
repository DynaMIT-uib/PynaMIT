"""Shared grid-evaluation helpers for visualization code paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import xarray as xr

from pynamit.primitives.basis import is_sh_basis
from pynamit.primitives.grid import Grid
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.simulation.input import decode_conductance_representation_to_grids


def get_scalar_grid_evaluation_matrix(
    storage_basis: Any,
    grid: Grid,
    *,
    mean_free: Optional[bool] = None,
) -> np.ndarray:
    """Return the dense scalar evaluation matrix for ``storage_basis`` on ``grid``."""
    kwargs = {}
    if mean_free is not None and is_sh_basis(storage_basis):
        kwargs["mean_free"] = mean_free
    return np.asarray(to_dense(storage_basis.get_evaluation_matrix(grid, **kwargs)))


def get_tangential_grid_component_matrices(
    storage_basis: Any,
    grid: Grid,
    *,
    mean_free: Optional[bool] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense ``(theta, phi)`` tangential evaluation matrices on ``grid``."""
    kwargs = {}
    if mean_free is not None and is_sh_basis(storage_basis):
        kwargs["mean_free"] = mean_free
    vector_basis = np.asarray(to_dense(storage_basis.get_vector_basis_matrix(grid, **kwargs)))
    if vector_basis.ndim != 4 or vector_basis.shape[0] != 2 or vector_basis.shape[2] != 2:
        raise ValueError(
            "Unexpected vector basis tensor shape for tangential grid evaluation: "
            f"{vector_basis.shape}."
        )
    theta_matrix = np.hstack([vector_basis[0, :, 0, :], vector_basis[0, :, 1, :]])
    phi_matrix = np.hstack([vector_basis[1, :, 0, :], vector_basis[1, :, 1, :]])
    return np.asarray(theta_matrix), np.asarray(phi_matrix)


def load_netcdf_dataset(path: str | Path, *, engine: str = "netcdf4") -> xr.Dataset:
    """Load a NetCDF dataset with an explicit engine."""
    return xr.load_dataset(Path(path), engine=engine)


def load_netcdf_dataarray(path: str | Path, *, engine: str = "netcdf4") -> xr.DataArray:
    """Load a NetCDF data array with an explicit engine."""
    return xr.load_dataarray(Path(path), engine=engine)


def evaluate_scalar_coeffs_to_grid(
    coeffs: Optional[np.ndarray],
    storage_basis: Any,
    grid: Grid,
    target_shape: Tuple[int, ...],
    *,
    mean_free: Optional[bool] = None,
    evaluation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate scalar coefficients to a target grid."""
    if coeffs is None:
        return np.full(target_shape, np.nan)
    scalar_matrix = (
        np.asarray(evaluation_matrix)
        if evaluation_matrix is not None
        else get_scalar_grid_evaluation_matrix(storage_basis, grid, mean_free=mean_free)
    )
    values = scalar_matrix @ np.asarray(coeffs).reshape(-1)
    return np.asarray(values).reshape(target_shape)


def evaluate_tangential_coeffs_to_grid_components(
    coeffs: Optional[np.ndarray],
    storage_basis: Any,
    grid: Grid,
    target_shape: Tuple[int, ...],
    *,
    mean_free: Optional[bool] = None,
    theta_evaluation_matrix: Optional[np.ndarray] = None,
    phi_evaluation_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate tangential coefficients to grid components."""
    if coeffs is None:
        nan_grid = np.full(target_shape, np.nan)
        return nan_grid, nan_grid
    if theta_evaluation_matrix is None or phi_evaluation_matrix is None:
        theta_matrix, phi_matrix = get_tangential_grid_component_matrices(
            storage_basis,
            grid,
            mean_free=mean_free,
        )
    else:
        theta_matrix = np.asarray(theta_evaluation_matrix)
        phi_matrix = np.asarray(phi_evaluation_matrix)
    coeff_vector = np.asarray(coeffs).reshape(-1)
    return (
        np.asarray(theta_matrix @ coeff_vector).reshape(target_shape),
        np.asarray(phi_matrix @ coeff_vector).reshape(target_shape),
    )


def decode_conductance_entry_to_grids(
    entry: dict[str, np.ndarray],
    storage_basis: Any,
    grid: Grid,
    target_shape: Tuple[int, ...],
    *,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a conductance entry to ``(SigmaP, SigmaH, etaP, etaH)`` on a grid."""

    def _eval(coeffs: np.ndarray) -> np.ndarray:
        return evaluate_scalar_coeffs_to_grid(coeffs, storage_basis, grid, target_shape)

    return decode_conductance_representation_to_grids(
        data=entry,
        eval_scalar_coeffs_to_grid=_eval,
        sigma_floor=sigma_floor,
    )


def decode_conductance_dataset_to_grids(
    conductance_ds: xr.Dataset,
    *,
    t_idx: int,
    storage_basis: Any,
    grid: Grid,
    target_shape: Tuple[int, ...],
    sigma_floor: float,
    coeff_prefix: str = "SH_",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a conductance dataset snapshot to ``(SigmaP, SigmaH, etaP, etaH)`` on a grid."""
    entry: dict[str, np.ndarray] = {}
    for short_name in ("etaP", "etaH", "SigmaP", "SigmaH", "logSigmaP", "logSigmaH"):
        dataset_name = f"{coeff_prefix}{short_name}" if coeff_prefix else short_name
        if dataset_name in conductance_ds.data_vars:
            entry[short_name] = conductance_ds[dataset_name].isel(time=t_idx).values

    if not entry:
        raise KeyError(
            "Unsupported conductance dataset representation. Expected "
            f"{coeff_prefix}etaP/{coeff_prefix}etaH, "
            f"{coeff_prefix}SigmaP/{coeff_prefix}SigmaH, or "
            f"{coeff_prefix}logSigmaP/{coeff_prefix}logSigmaH."
        )

    return decode_conductance_entry_to_grids(
        entry,
        storage_basis,
        grid,
        target_shape,
        sigma_floor=sigma_floor,
    )
