"""Time-series preparation helpers for visualization workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def datetime_index_to_epoch_ns(index):
    """Return epoch nanoseconds for any pandas datetime64 resolution."""
    return pd.DatetimeIndex(pd.to_datetime(index)).to_numpy(dtype="datetime64[ns]").astype("int64")


def resample_series_to_times(index, values, target_times):
    """Linearly interpolate a scalar series onto target datetimes."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    time_index = pd.DatetimeIndex(pd.to_datetime(index))
    target_index = pd.DatetimeIndex(pd.to_datetime(target_times))
    if values_arr.size != time_index.size:
        raise ValueError("index and values must have the same length.")

    finite_mask = np.isfinite(values_arr)
    if not np.any(finite_mask):
        return np.full(target_index.shape, np.nan, dtype=float)

    x_ns = datetime_index_to_epoch_ns(time_index)[finite_mask]
    y = values_arr[finite_mask]
    order = np.argsort(x_ns)
    x_ns = x_ns[order]
    y = y[order]
    x_unique, unique_idx = np.unique(x_ns, return_index=True)
    y_unique = y[unique_idx]
    target_ns = datetime_index_to_epoch_ns(target_index)
    left = float(x_unique[0])
    right = float(x_unique[-1])
    out = np.interp(target_ns, x_unique, y_unique, left=np.nan, right=np.nan)
    out[(target_ns < left) | (target_ns > right)] = np.nan
    return out.astype(float)


def resample_matrix_to_times(index, values, target_times):
    """Linearly interpolate each matrix row onto target datetimes."""
    values_arr = np.asarray(values, dtype=float)
    return np.vstack([resample_series_to_times(index, row, target_times) for row in values_arr])


def get_time_index_median_cadence_seconds(time_index):
    """Return the median positive datetime cadence in seconds."""
    time_ns = datetime_index_to_epoch_ns(time_index).astype(float)
    if time_ns.size < 2:
        return np.nan
    dt_seconds = np.diff(time_ns) * 1e-9
    dt_seconds = dt_seconds[np.isfinite(dt_seconds) & (dt_seconds > 0.0)]
    if dt_seconds.size == 0:
        return np.nan
    return float(np.nanmedian(dt_seconds))


def compute_centered_difference_series_at_times(
    source_index, source_values, target_times, half_window_points=1, cadence_seconds=None
):
    """Evaluate a centered finite difference on target datetimes."""
    target_index = pd.DatetimeIndex(pd.to_datetime(target_times))
    half_window_points = max(1, int(half_window_points))
    if cadence_seconds is None or not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        cadence_seconds = get_time_index_median_cadence_seconds(source_index)
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        return np.full(target_index.shape, np.nan, dtype=float)

    half_window_seconds = float(half_window_points) * float(cadence_seconds)
    delta = pd.to_timedelta(half_window_seconds, unit="s")
    left_values = resample_series_to_times(source_index, source_values, target_index - delta)
    right_values = resample_series_to_times(source_index, source_values, target_index + delta)
    return (right_values - left_values) / (2.0 * half_window_seconds)


def compute_centered_difference_matrix_at_times(
    source_index, source_values, target_times, half_window_points=1, cadence_seconds=None
):
    """Evaluate centered finite differences for each row of a matrix."""
    values_arr = np.asarray(source_values, dtype=float)
    return np.vstack(
        [
            compute_centered_difference_series_at_times(
                source_index,
                row,
                target_times,
                half_window_points=half_window_points,
                cadence_seconds=cadence_seconds,
            )
            for row in values_arr
        ]
    )


def compute_time_derivative_matrix(values_matrix, time_index, half_window_points=1):
    """Return same-grid centered derivatives along the last axis."""
    values_arr = np.asarray(values_matrix, dtype=float)
    time_ns = datetime_index_to_epoch_ns(time_index)
    if time_ns.size < 2:
        return np.full_like(values_arr, np.nan, dtype=float)

    time_seconds = (time_ns - time_ns[0]).astype(float) * 1e-9
    if values_arr.shape[-1] != time_seconds.size or time_seconds.size < 2:
        return np.full_like(values_arr, np.nan, dtype=float)
    if np.any(np.diff(time_seconds) <= 0.0):
        return np.full_like(values_arr, np.nan, dtype=float)

    n_times = time_seconds.size
    half_window_points = max(1, int(half_window_points))
    if n_times <= 2 * half_window_points:
        return np.full_like(values_arr, np.nan, dtype=float)

    derivative = np.full_like(values_arr, np.nan, dtype=float)
    center_idx = np.arange(half_window_points, n_times - half_window_points, dtype=int)
    left_idx = center_idx - half_window_points
    right_idx = center_idx + half_window_points
    dt = time_seconds[right_idx] - time_seconds[left_idx]
    valid_dt = np.isfinite(dt) & (dt > 0.0)
    if np.any(valid_dt):
        derivative[..., center_idx[valid_dt]] = (
            values_arr[..., right_idx[valid_dt]] - values_arr[..., left_idx[valid_dt]]
        ) / dt[valid_dt]
    return derivative


def vector_magnitude_from_component_series(component_values):
    """Return vector magnitude from one-dimensional component series."""
    component_arr = np.vstack(
        [np.asarray(values, dtype=float).reshape(-1) for values in component_values]
    )
    finite_any = np.any(np.isfinite(component_arr), axis=0)
    magnitude = np.sqrt(np.nansum(component_arr**2, axis=0))
    magnitude[~finite_any] = np.nan
    return magnitude


def vector_magnitude_preserve_shape(component_values):
    """Return vector magnitude, preserving component array shape."""
    component_arr = np.asarray([np.asarray(values, dtype=float) for values in component_values])
    finite_any = np.any(np.isfinite(component_arr), axis=0)
    magnitude = np.sqrt(np.nansum(component_arr**2, axis=0))
    magnitude[~finite_any] = np.nan
    return magnitude


__all__ = [
    "compute_centered_difference_matrix_at_times",
    "compute_centered_difference_series_at_times",
    "compute_time_derivative_matrix",
    "datetime_index_to_epoch_ns",
    "get_time_index_median_cadence_seconds",
    "resample_matrix_to_times",
    "resample_series_to_times",
    "vector_magnitude_from_component_series",
    "vector_magnitude_preserve_shape",
]
