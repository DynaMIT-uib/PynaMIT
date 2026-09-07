"""Time-series preparation helpers for result workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def datetime_at_index(times, index, *, start_time=None):
    """Return a saved timestamp using ordinary NumPy indexing."""
    values = np.asarray(times)
    value = values[index]
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.Timestamp(value)
    if start_time is None:
        raise ValueError("Numeric simulation times require the physical start_time.")
    return pd.Timestamp(start_time) + pd.to_timedelta(float(value), unit="s")


def time_index_from_dataset(dataset, *, start_time=None):
    """Convert saved datetimes or model seconds to a DatetimeIndex."""
    times = np.asarray(dataset.time.values)
    if np.issubdtype(times.dtype, np.datetime64) or times.size == 0:
        return pd.DatetimeIndex(times)
    if start_time is None:
        raise ValueError("Numeric simulation times require the physical start_time.")
    return pd.Timestamp(start_time) + pd.to_timedelta(times, unit="s")


def datetime_index_to_epoch_ns(index):
    """Return epoch nanoseconds for any pandas datetime64 resolution."""
    return pd.DatetimeIndex(pd.to_datetime(index)).to_numpy(dtype="datetime64[ns]").astype("int64")


def resample_to_times(index, values, target_times):
    """Interpolate along the last axis, preserving all leading axes.

    Missing samples are skipped per series. Duplicate timestamps use
    the first finite sample; values outside each finite span are NaN.
    Datetime conversion and ordering are shared across the whole batch.
    """
    values = np.asarray(values, dtype=float)
    source_ns = datetime_index_to_epoch_ns(index)
    target_ns = datetime_index_to_epoch_ns(target_times)
    if values.ndim == 0 or values.shape[-1] != source_ns.size:
        raise ValueError("The last values axis must match index.")
    shape = values.shape[:-1] + (target_ns.size,)
    if values.size == 0:
        return np.full(shape, np.nan)
    order = np.argsort(source_ns, kind="stable")
    # Subtract the origin before conversion to avoid losing subsecond
    # precision by interpolating large floating-point epoch values.
    x = (source_ns[order] - source_ns[order[0]]).astype(float)
    target = (target_ns - source_ns[order[0]]).astype(float)
    rows = values.reshape(-1, source_ns.size)[:, order]
    output = np.full((rows.shape[0], target.size), np.nan)
    for row, result in zip(rows, output, strict=True):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        coordinates, first = np.unique(x[finite], return_index=True)
        result[:] = np.interp(target, coordinates, row[finite][first], left=np.nan, right=np.nan)
    return output.reshape(shape)


def median_cadence_seconds(time_index):
    """Return the median positive datetime cadence in seconds."""
    time_ns = datetime_index_to_epoch_ns(time_index)
    if time_ns.size < 2:
        return np.nan
    dt_seconds = np.diff(time_ns).astype(float) * 1e-9
    dt_seconds = dt_seconds[np.isfinite(dt_seconds) & (dt_seconds > 0.0)]
    if dt_seconds.size == 0:
        return np.nan
    return float(np.nanmedian(dt_seconds))


def centered_difference_at_times(
    source_index, source_values, target_times, half_window_points=1, cadence_seconds=None
):
    """Evaluate a centered finite difference on target datetimes."""
    source_values = np.asarray(source_values, dtype=float)
    if source_values.ndim == 0 or source_values.shape[-1] != len(source_index):
        raise ValueError("The last values axis must match source_index.")
    target_index = pd.DatetimeIndex(pd.to_datetime(target_times))
    if isinstance(half_window_points, (bool, np.bool_)):
        raise ValueError("half_window_points must be a positive integer.")
    integer_window = int(half_window_points)
    if integer_window != half_window_points or integer_window < 1:
        raise ValueError("half_window_points must be a positive integer.")
    half_window_points = integer_window
    if cadence_seconds is None:
        cadence_seconds = median_cadence_seconds(source_index)
    else:
        cadence_seconds = float(cadence_seconds)
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
            raise ValueError("cadence_seconds must be finite and positive.")
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        return np.full(source_values.shape[:-1] + (target_index.size,), np.nan)

    half_window_seconds = float(half_window_points) * float(cadence_seconds)
    delta = pd.to_timedelta(half_window_seconds, unit="s")
    left_values = resample_to_times(source_index, source_values, target_index - delta)
    right_values = resample_to_times(source_index, source_values, target_index + delta)
    return (right_values - left_values) / (2.0 * half_window_seconds)


def compute_time_derivative_values(values, time_index, half_window_points=1):
    """Return same-grid centered derivatives along the last axis."""
    values_arr = np.asarray(values, dtype=float)
    time_ns = datetime_index_to_epoch_ns(time_index)
    if values_arr.ndim == 0 or values_arr.shape[-1] != time_ns.size:
        raise ValueError("The last values axis must match time_index.")
    if time_ns.size < 2:
        return np.full_like(values_arr, np.nan, dtype=float)

    time_seconds = (time_ns - time_ns[0]).astype(float) * 1e-9
    if np.any(np.diff(time_seconds) <= 0.0):
        raise ValueError("time_index must be strictly increasing.")

    n_times = time_seconds.size
    if isinstance(half_window_points, (bool, np.bool_)):
        raise ValueError("half_window_points must be a positive integer.")
    integer_window = int(half_window_points)
    if integer_window != half_window_points or integer_window < 1:
        raise ValueError("half_window_points must be a positive integer.")
    half_window_points = integer_window
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


def vector_magnitude(component_values):
    """Return magnitudes, preserving all non-component axes.

    NaN components are ignored, retaining the observational convention:
    a sample with no finite component has a NaN magnitude.
    """
    components = np.asarray(component_values, dtype=float)
    finite_any = np.any(np.isfinite(components), axis=0)
    return np.where(finite_any, np.sqrt(np.nansum(components**2, axis=0)), np.nan)


__all__ = [
    "centered_difference_at_times",
    "compute_time_derivative_values",
    "datetime_at_index",
    "datetime_index_to_epoch_ns",
    "median_cadence_seconds",
    "resample_to_times",
    "time_index_from_dataset",
    "vector_magnitude",
]
