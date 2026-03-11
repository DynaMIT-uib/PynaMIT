"""Helpers for batched projection inputs and projection control flow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class RegularGridAxes:
    """Regular-grid axes needed for the fast SH projection path."""

    n_theta: int
    n_phi: int
    theta: np.ndarray
    phi: np.ndarray
    trim_wrapped_lon_endpoint: bool = False

    @property
    def n_points(self) -> int:
        """Return the untrimmed number of grid points."""
        return self.n_theta * self.n_phi

    def reshape_scalar_batch(self, values: np.ndarray) -> np.ndarray:
        """Reshape scalar batches for fast-path grid projection."""
        batch = np.asarray(values).reshape(values.shape[0], self.n_theta, self.n_phi)
        if self.trim_wrapped_lon_endpoint:
            batch = batch[:, :, :-1]
        return batch

    def reshape_vector_batch(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape tangential batches for fast-path grid projection."""
        batch = np.asarray(values).reshape(values.shape[0], 2, self.n_theta, self.n_phi)
        if self.trim_wrapped_lon_endpoint:
            batch = batch[:, :, :, :-1]
        return batch[:, 0], batch[:, 1]


def _extract_tangential_components(
    raw_values: Any, n_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract two tangential components from common one-slice layouts."""
    if isinstance(raw_values, (tuple, list)) and len(raw_values) == 2:
        comp0, comp1 = raw_values
    else:
        arr = np.asarray(raw_values)
        if arr.ndim == 2 and arr.shape[0] == 2:
            comp0, comp1 = arr[0], arr[1]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            comp0, comp1 = arr[:, 0], arr[:, 1]
        elif arr.ndim == 3 and arr.shape[0] == 2:
            comp0, comp1 = arr[0], arr[1]
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            comp0, comp1 = arr[..., 0], arr[..., 1]
        else:
            return None

    comp0 = np.asarray(comp0)
    comp1 = np.asarray(comp1)
    if comp0.size != n_points or comp1.size != n_points:
        return None
    return comp0.reshape(-1), comp1.reshape(-1)


def _normalize_scalar_projection_batch(
    values: Any, *, n_points: int, n_times: int | None = None
) -> np.ndarray:
    """Return scalar data with canonical shape ``(T, N)``."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        if arr.size != n_points:
            raise ValueError(
                f"Scalar projection expects {n_points} points, got shape {arr.shape}."
            )
        if n_times not in (None, 1):
            raise ValueError(
                f"Scalar projection with n_times={n_times} must provide an explicit time axis, got shape {arr.shape}."
            )
        return arr.reshape(1, n_points)

    if arr.ndim == 2:
        if arr.shape[-1] == n_points:
            batch = arr
        elif arr.shape[0] == n_points:
            batch = arr.T
        else:
            raise ValueError(
                "Scalar projection expects shape (N,), (T, N), or (N, T), "
                f"got {arr.shape} for grid size {n_points}."
            )
        if n_times is not None and batch.shape[0] != n_times:
            raise ValueError(
                f"Scalar projection expects {n_times} time slices, got shape {arr.shape}."
            )
        return batch

    raise ValueError(
        "Scalar projection expects shape (N,), (T, N), or (N, T), "
        f"got {arr.shape} for grid size {n_points}."
    )


def _normalize_tangential_projection_batch(
    values: Any, *, n_points: int, n_times: int | None = None
) -> np.ndarray:
    """Return tangential data with canonical shape ``(T, 2, N)``."""
    if isinstance(values, (tuple, list)):
        single_slice = _extract_tangential_components(values, n_points)
        if single_slice is not None:
            if n_times not in (None, 1):
                raise ValueError(
                    f"Tangential projection expects {n_times} time slices, got a single slice."
                )
            return np.stack(single_slice, axis=0)[np.newaxis, ...]

        if n_times is not None and len(values) == n_times:
            rows = []
            for item in values:
                comps = _extract_tangential_components(item, n_points)
                if comps is None:
                    raise ValueError(
                        f"Unsupported tangential batch item layout: {type(item).__name__}."
                    )
                rows.append(np.stack(comps, axis=0))
            return np.asarray(rows)

    arr = np.asarray(values)
    if arr.ndim == 2:
        comps = _extract_tangential_components(arr, n_points)
        if comps is not None:
            if n_times not in (None, 1):
                raise ValueError(
                    f"Tangential projection expects {n_times} time slices, got shape {arr.shape}."
                )
            return np.stack(comps, axis=0)[np.newaxis, ...]
    elif arr.ndim == 3:
        if arr.shape[1] == 2 and arr.shape[2] == n_points:
            batch = arr
        elif arr.shape[0] == 2 and arr.shape[1] == n_points:
            batch = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] == n_points and arr.shape[2] == 2:
            batch = np.moveaxis(arr, 1, 0)
        else:
            batch = None
        if batch is not None:
            if n_times is not None and batch.shape[0] != n_times:
                raise ValueError(
                    f"Tangential projection expects {n_times} time slices, got shape {arr.shape}."
                )
            return batch
    elif arr.ndim == 4:
        if arr.shape[1] == 2:
            batch = arr.reshape(arr.shape[0], 2, -1)
        elif arr.shape[3] == 2:
            batch = np.moveaxis(arr, -1, 1).reshape(arr.shape[0], 2, -1)
        else:
            batch = None
        if batch is not None and batch.shape[2] == n_points:
            if n_times is not None and batch.shape[0] != n_times:
                raise ValueError(
                    f"Tangential projection expects {n_times} time slices, got shape {arr.shape}."
                )
            return batch

    raise ValueError(
        "Tangential projection expects shape (2, N), (T, 2, N), or (2, N, T), "
        f"got {arr.shape} for grid size {n_points}."
    )


def normalize_projection_input_batch(
    values: Any,
    *,
    vector_type: Literal["scalar", "tangential"],
    n_points: int,
    n_times: int | None = None,
) -> np.ndarray:
    """Return one canonical batch layout for projection inputs."""
    if vector_type == "scalar":
        return _normalize_scalar_projection_batch(values, n_points=n_points, n_times=n_times)
    if vector_type == "tangential":
        return _normalize_tangential_projection_batch(values, n_points=n_points, n_times=n_times)
    raise ValueError(f"Unknown vector_type: {vector_type!r}")


def detect_regular_grid_axes(input_grid: Any) -> RegularGridAxes | None:
    """Return regular-grid axes for fast SH projection, if available."""
    u_lat = np.unique(np.round(input_grid.lat, 6))
    u_lon = np.unique(np.round(input_grid.lon, 6))
    if u_lat.size * u_lon.size != input_grid.size:
        return None

    n_theta = u_lat.size
    n_phi = u_lon.size
    try:
        lat_2d = input_grid.lat.reshape(n_theta, n_phi)
        lon_2d = input_grid.lon.reshape(n_theta, n_phi)
    except ValueError:
        return None

    if not (np.allclose(lat_2d[:, 0:1], lat_2d) and np.allclose(lon_2d[0:1, :], lon_2d)):
        return None

    lat_1d = lat_2d[:, 0]
    lon_1d = lon_2d[0, :]
    trim_wrapped_lon_endpoint = False
    if lon_1d.size > 2:
        lon_step = np.diff(lon_1d)
        if np.allclose(lon_step, lon_step[0]) and np.isclose(
            np.abs(lon_1d[-1] - lon_1d[0]), 360.0
        ):
            trim_wrapped_lon_endpoint = True
            lon_1d = lon_1d[:-1]

    return RegularGridAxes(
        n_theta=n_theta,
        n_phi=n_phi,
        theta=np.deg2rad(90 - lat_1d),
        phi=np.deg2rad(lon_1d),
        trim_wrapped_lon_endpoint=trim_wrapped_lon_endpoint,
    )


def _extract_fast_weight_points(
    sqrt_weights: np.ndarray, *, n_points: int, vector_type: Literal["scalar", "tangential"]
) -> np.ndarray | None:
    """Extract point-wise weights for the fast path from common input layouts."""
    arr = np.asarray(sqrt_weights)

    if vector_type == "scalar":
        if arr.ndim == 1:
            return arr
        return arr.reshape(-1)

    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == n_points:
        if np.allclose(arr[0], arr[1]):
            return arr[0]
        return None
    if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[1] * arr.shape[2] == n_points:
        if np.allclose(arr[0], arr[1]):
            return arr[0].reshape(-1)
        return None
    if arr.ndim == 1 and arr.size == 2 * n_points:
        first, second = arr[:n_points], arr[n_points:]
        if np.allclose(first, second):
            return first
        return None
    if arr.size == n_points:
        return arr.reshape(-1)
    return None


def extract_fast_path_weights(
    sqrt_weights: np.ndarray | None,
    *,
    regular_axes: RegularGridAxes,
    vector_type: Literal["scalar", "tangential"],
) -> np.ndarray | None:
    """Return latitude-only fast-path weights, or ``None`` if unsupported."""
    if sqrt_weights is None:
        return None

    point_weights = _extract_fast_weight_points(
        sqrt_weights, n_points=regular_axes.n_points, vector_type=vector_type
    )
    if point_weights is None:
        return None
    if point_weights.size == regular_axes.n_theta:
        return point_weights

    weight_grid = point_weights.reshape(regular_axes.n_theta, regular_axes.n_phi)
    if regular_axes.trim_wrapped_lon_endpoint:
        weight_grid = weight_grid[:, :-1]
    if not np.allclose(weight_grid[:, 0:1], weight_grid):
        return None
    return weight_grid[:, 0]


def build_fast_projection_input(
    value_batch: np.ndarray,
    *,
    vector_type: Literal["scalar", "tangential"],
    regular_axes: RegularGridAxes,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert canonical batches into the fast-path grid layout."""
    if vector_type == "scalar":
        return regular_axes.reshape_scalar_batch(value_batch)
    return regular_axes.reshape_vector_batch(value_batch)


def as_coefficient_matrix(coefficients: np.ndarray, *, n_times: int) -> np.ndarray:
    """Normalize projected coefficients to shape ``(N_coeff, T)``."""
    coeff_matrix = np.asarray(coefficients)
    if coeff_matrix.ndim == 1:
        return coeff_matrix.reshape(-1, 1)
    return coeff_matrix.reshape(-1, n_times)


def project_batch_to_coefficients(
    value_batch: np.ndarray,
    *,
    input_grid: Any,
    vector_type: Literal["scalar", "tangential"],
    projection_basis: Any,
    target_basis: Any,
    target_grid: Any,
    target_mean_free: bool,
    sqrt_weights: np.ndarray | None = None,
    reg_lambda: float | None = None,
    pinv_rtol: float = 1e-15,
    enable_fast_path: bool = True,
) -> np.ndarray:
    """Project one canonical input batch to coefficient columns."""
    regular_axes = None
    if enable_fast_path and target_basis.supports_regular_grid_fast_path():
        regular_axes = detect_regular_grid_axes(input_grid)

    if regular_axes is not None:
        weights_1d = extract_fast_path_weights(
            sqrt_weights, regular_axes=regular_axes, vector_type=vector_type
        )
        if sqrt_weights is None or weights_1d is not None:
            fast_input = build_fast_projection_input(
                value_batch, vector_type=vector_type, regular_axes=regular_axes
            )
            coeff_block = target_basis.grid_to_basis_fast(
                fast_input,
                regular_axes.theta,
                phi=regular_axes.phi,
                weights=weights_1d,
                reg_lambda=reg_lambda,
                vector_type=vector_type,
            )
            return as_coefficient_matrix(coeff_block, n_times=value_batch.shape[0])

    coeff_block = projection_basis.project_to_basis(
        value_batch,
        input_grid,
        vector_type=vector_type,
        target_grid=target_grid,
        target_basis=target_basis,
        mean_free=target_mean_free,
        weights=sqrt_weights,
        reg_lambda=reg_lambda,
        pinv_rtol=pinv_rtol,
    )
    return as_coefficient_matrix(coeff_block, n_times=value_batch.shape[0])


def interpolate_then_project_batch(
    input_values: np.ndarray,
    *,
    input_grid: Any,
    vector_type: Literal["scalar", "tangential"],
    target_grid: Any,
    target_basis: Any,
    scalar_interpolator: Callable[[np.ndarray], np.ndarray],
    vector_interpolator: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    fit_kwargs: dict[str, Any],
) -> np.ndarray:
    """Interpolate each time slice to the target grid and fit once per slice."""
    n_points = int(getattr(input_grid, "size", 0))
    value_batch = normalize_projection_input_batch(
        input_values, vector_type=vector_type, n_points=n_points
    )

    coeff_columns = []
    project_kwargs = fit_kwargs.copy()
    project_kwargs["weights"] = None
    for time_index in range(value_batch.shape[0]):
        if vector_type == "scalar":
            grid_values = scalar_interpolator(value_batch[time_index])
        else:
            u_theta = value_batch[time_index, 0]
            u_phi = value_batch[time_index, 1]
            u_east = u_phi
            u_north = -u_theta
            u_r = np.zeros_like(u_north)
            u_east_int, u_north_int, _ = vector_interpolator(u_east, u_north, u_r)
            grid_values = np.vstack((-u_north_int, u_east_int))

        coeffs = target_basis.from_grid_values(
            grid_values, target_grid, vector_type, **project_kwargs
        )
        coeff_columns.append(np.asarray(coeffs).reshape(-1))

    if len(coeff_columns) == 1:
        return coeff_columns[0]
    return np.column_stack(coeff_columns)
