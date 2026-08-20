"""Time-series preparation helpers for result workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks as _find_peaks
except ImportError:  # pragma: no cover - optional dependency fallback
    _find_peaks = None


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
    if isinstance(half_window_points, (bool, np.bool_)):
        raise ValueError("half_window_points must be a positive integer.")
    integer_window = int(half_window_points)
    if integer_window != half_window_points or integer_window < 1:
        raise ValueError("half_window_points must be a positive integer.")
    half_window_points = integer_window
    if cadence_seconds is None:
        cadence_seconds = get_time_index_median_cadence_seconds(source_index)
    else:
        cadence_seconds = float(cadence_seconds)
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
            raise ValueError("cadence_seconds must be finite and positive.")
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


def _find_prominent_peaks_numpy(values, min_prominence):
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    if values_arr.size == 0:
        return np.array([], dtype=int)

    peak_indices = []
    for idx, value in enumerate(values_arr):
        if not np.isfinite(value):
            continue
        left_value = values_arr[idx - 1] if idx > 0 else -np.inf
        right_value = values_arr[idx + 1] if idx + 1 < values_arr.size else -np.inf
        left_ok = (not np.isfinite(left_value)) or value >= left_value
        right_ok = (not np.isfinite(right_value)) or value >= right_value
        strictly_above_neighbor = (
            (np.isfinite(left_value) and value > left_value)
            or (np.isfinite(right_value) and value > right_value)
            or values_arr.size == 1
        )
        if not (left_ok and right_ok and strictly_above_neighbor):
            continue

        left_min = value
        scan_idx = idx - 1
        while scan_idx >= 0 and np.isfinite(values_arr[scan_idx]):
            if values_arr[scan_idx] > value:
                break
            left_min = min(left_min, values_arr[scan_idx])
            scan_idx -= 1

        right_min = value
        scan_idx = idx + 1
        while scan_idx < values_arr.size and np.isfinite(values_arr[scan_idx]):
            if values_arr[scan_idx] > value:
                break
            right_min = min(right_min, values_arr[scan_idx])
            scan_idx += 1

        prominence = value - max(left_min, right_min)
        if np.isfinite(prominence) and prominence >= min_prominence:
            peak_indices.append(idx)

    return np.asarray(peak_indices, dtype=int)


def _estimate_peak_prominence_numpy(values, peak_idx):
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    peak_idx = int(peak_idx)
    if peak_idx < 0 or peak_idx >= values_arr.size:
        return np.nan
    value = values_arr[peak_idx]
    if not np.isfinite(value):
        return np.nan

    left_min = value
    scan_idx = peak_idx - 1
    while scan_idx >= 0 and np.isfinite(values_arr[scan_idx]):
        if values_arr[scan_idx] > value:
            break
        left_min = min(left_min, values_arr[scan_idx])
        scan_idx -= 1

    right_min = value
    scan_idx = peak_idx + 1
    while scan_idx < values_arr.size and np.isfinite(values_arr[scan_idx]):
        if values_arr[scan_idx] > value:
            break
        right_min = min(right_min, values_arr[scan_idx])
        scan_idx += 1

    prominence = float(value - max(left_min, right_min))
    return prominence if np.isfinite(prominence) and prominence >= 0.0 else np.nan


def _global_peak_candidate(abs_values, valid, target_index):
    """Return the global finite maximum as a one-item candidate list."""
    peak_idx = int(np.nanargmax(np.where(valid, abs_values, -np.inf)))
    peak_value = float(abs_values[peak_idx])
    return [
        {
            "index": peak_idx,
            "abs_value": peak_value,
            "prominence": peak_value,
            "relative_value": 1.0,
            "time": pd.Timestamp(target_index[peak_idx]),
        }
    ]


def _finite_segment_peak_candidates(abs_values, min_prominence):
    """Find peaks independently in each contiguous finite segment."""
    candidate_indices = []
    prominence_by_index = {}
    finite_mask = np.isfinite(abs_values)
    segment_start = 0
    while segment_start < abs_values.size:
        while segment_start < abs_values.size and not finite_mask[segment_start]:
            segment_start += 1
        if segment_start >= abs_values.size:
            break
        segment_end = segment_start + 1
        while segment_end < abs_values.size and finite_mask[segment_end]:
            segment_end += 1

        segment_values = abs_values[segment_start:segment_end]
        if segment_values.size == 1:
            if segment_values[0] >= min_prominence:
                candidate_indices.append(segment_start)
                prominence_by_index[segment_start] = float(segment_values[0])
            segment_start = segment_end
            continue

        edge_floor = min(0.0, float(np.nanmin(segment_values)))
        padded_values = np.r_[edge_floor, segment_values, edge_floor]
        if _find_peaks is None:
            segment_peak_indices = _find_prominent_peaks_numpy(padded_values, min_prominence)
            segment_peak_prominences = np.array(
                [
                    _estimate_peak_prominence_numpy(padded_values, peak_idx)
                    for peak_idx in segment_peak_indices
                ],
                dtype=float,
            )
        else:
            segment_peak_indices, peak_properties = _find_peaks(
                padded_values, prominence=min_prominence
            )
            segment_peak_prominences = np.asarray(
                peak_properties.get(
                    "prominences", np.full(segment_peak_indices.shape, np.nan, dtype=float)
                ),
                dtype=float,
            )
        segment_peak_indices = segment_peak_indices - 1
        valid_peak_mask = (segment_peak_indices >= 0) & (
            segment_peak_indices < segment_values.size
        )
        original_peak_indices = (segment_start + segment_peak_indices[valid_peak_mask]).astype(int)
        candidate_indices.extend(original_peak_indices.tolist())
        for original_peak_idx, peak_prominence in zip(
            original_peak_indices, segment_peak_prominences[valid_peak_mask], strict=True
        ):
            if not np.isfinite(peak_prominence):
                peak_prominence = _estimate_peak_prominence_numpy(abs_values, original_peak_idx)
            if not np.isfinite(peak_prominence):
                peak_prominence = abs_values[original_peak_idx]
            prominence_by_index[original_peak_idx] = max(
                float(prominence_by_index.get(original_peak_idx, -np.inf)), float(peak_prominence)
            )
        segment_start = segment_end
    return candidate_indices, prominence_by_index


def _separated_peak_indices(
    candidate_indices,
    candidate_values,
    candidate_prominences,
    target_index,
    min_separation_seconds,
):
    """Keep the strongest peak within each minimum time separation."""
    candidate_times_ns = datetime_index_to_epoch_ns(target_index[candidate_indices]).astype(float)
    min_separation_ns = min_separation_seconds * 1e9
    order = np.lexsort((candidate_times_ns, -candidate_values, -candidate_prominences))
    kept_indices = []
    kept_times_ns = []
    for order_idx in order:
        peak_time_ns = candidate_times_ns[order_idx]
        if any(
            abs(peak_time_ns - kept_time_ns) <= min_separation_ns for kept_time_ns in kept_times_ns
        ):
            continue
        kept_indices.append(candidate_indices[order_idx])
        kept_times_ns.append(peak_time_ns)
    return kept_indices


def prominent_peak_candidates(
    values, time_index, min_separation_seconds=20.0, prominence_fraction=0.05
):
    """Return separated peaks from absolute time-series values."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    target_index = pd.DatetimeIndex(time_index)
    if values_arr.size != len(target_index):
        raise ValueError("values and time_index must have the same length.")
    min_separation_seconds = float(min_separation_seconds)
    prominence_fraction = float(prominence_fraction)
    if not np.isfinite(min_separation_seconds) or min_separation_seconds < 0.0:
        raise ValueError("min_separation_seconds must be finite and non-negative.")
    if not np.isfinite(prominence_fraction) or prominence_fraction < 0.0:
        raise ValueError("prominence_fraction must be finite and non-negative.")
    if values_arr.size == 0:
        return []

    valid = np.isfinite(values_arr)
    if not np.any(valid):
        return []

    abs_values = np.where(valid, np.abs(values_arr), np.nan)
    finite_abs = abs_values[np.isfinite(abs_values)]
    if finite_abs.size == 0:
        return []

    global_peak = float(np.nanmax(finite_abs))
    if not np.isfinite(global_peak):
        return []
    if global_peak <= np.finfo(float).tiny:
        return _global_peak_candidate(abs_values, valid, target_index)

    min_prominence = prominence_fraction * global_peak
    candidate_indices, candidate_prominence_by_index = _finite_segment_peak_candidates(
        abs_values, min_prominence
    )
    if not candidate_indices:
        return _global_peak_candidate(abs_values, valid, target_index)

    candidate_indices = np.array(sorted(set(candidate_indices)), dtype=int)
    candidate_values = abs_values[candidate_indices]
    candidate_prominences = np.array(
        [
            candidate_prominence_by_index.get(
                int(idx), _estimate_peak_prominence_numpy(abs_values, int(idx))
            )
            for idx in candidate_indices
        ],
        dtype=float,
    )
    candidate_prominences = np.where(
        np.isfinite(candidate_prominences), candidate_prominences, candidate_values
    )
    kept_indices = _separated_peak_indices(
        candidate_indices,
        candidate_values,
        candidate_prominences,
        target_index,
        min_separation_seconds,
    )
    if not kept_indices:
        return _global_peak_candidate(abs_values, valid, target_index)

    return [
        {
            "index": int(idx),
            "abs_value": float(abs_values[idx]),
            "prominence": float(
                candidate_prominence_by_index.get(
                    int(idx), _estimate_peak_prominence_numpy(abs_values, int(idx))
                )
            ),
            "relative_value": float(abs_values[idx] / global_peak),
            "time": pd.Timestamp(target_index[idx]),
        }
        for idx in sorted(kept_indices, key=lambda kept_idx: target_index[kept_idx])
    ]


def most_prominent_peak_abs_value_and_time(
    values, time_index, min_separation_seconds=20.0, prominence_fraction=0.05
):
    """Return the most prominent absolute peak value and timestamp."""
    candidates = prominent_peak_candidates(
        values,
        time_index,
        min_separation_seconds=min_separation_seconds,
        prominence_fraction=prominence_fraction,
    )
    if not candidates:
        return np.nan, None

    def peak_sort_key(candidate):
        prominence = float(candidate.get("prominence", np.nan))
        abs_value = float(candidate.get("abs_value", np.nan))
        if not np.isfinite(prominence):
            prominence = -np.inf
        if not np.isfinite(abs_value):
            abs_value = -np.inf
        peak_time = pd.Timestamp(candidate["time"]).value
        return (-prominence, -abs_value, peak_time)

    selected_peak = min(candidates, key=peak_sort_key)
    return float(selected_peak["abs_value"]), pd.Timestamp(selected_peak["time"])


def first_event_peak_abs_value_and_time(
    values,
    time_index,
    min_separation_seconds=20.0,
    prominence_fraction=0.05,
    noise_floor_fraction=0.20,
):
    """Return the first event-like absolute peak value and timestamp."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    target_index = pd.DatetimeIndex(time_index)
    if values_arr.size != len(target_index):
        raise ValueError("values and time_index must have the same length.")
    noise_floor_fraction = float(noise_floor_fraction)
    if not np.isfinite(noise_floor_fraction) or noise_floor_fraction < 0.0:
        raise ValueError("noise_floor_fraction must be finite and non-negative.")
    if values_arr.size == 0:
        return most_prominent_peak_abs_value_and_time(values_arr, target_index)

    abs_values = np.where(np.isfinite(values_arr), np.abs(values_arr), np.nan)
    finite_abs = abs_values[np.isfinite(abs_values)]
    if finite_abs.size == 0:
        return np.nan, None

    global_peak = float(np.nanmax(finite_abs))
    if not np.isfinite(global_peak) or global_peak <= np.finfo(float).tiny:
        return most_prominent_peak_abs_value_and_time(values_arr, target_index)

    noise_floor = noise_floor_fraction * global_peak
    above_floor = np.isfinite(abs_values) & (abs_values >= noise_floor)
    if not np.any(above_floor):
        return most_prominent_peak_abs_value_and_time(values_arr, target_index)

    first_above_idx = int(np.flatnonzero(above_floor)[0])
    candidates = prominent_peak_candidates(
        values_arr,
        target_index,
        min_separation_seconds=min_separation_seconds,
        prominence_fraction=prominence_fraction,
    )
    for candidate in candidates:
        candidate_idx = int(candidate["index"])
        if candidate_idx >= first_above_idx and float(candidate["abs_value"]) >= noise_floor:
            return float(candidate["abs_value"]), pd.Timestamp(candidate["time"])

    for candidate in candidates:
        candidate_idx = int(candidate["index"])
        if candidate_idx >= first_above_idx:
            return float(candidate["abs_value"]), pd.Timestamp(candidate["time"])

    return float(abs_values[first_above_idx]), pd.Timestamp(target_index[first_above_idx])


def local_peak_abs_value_and_time(
    values,
    time_index,
    center_time,
    half_window_seconds=100.0,
    min_separation_seconds=20.0,
    prominence_fraction=0.05,
    noise_floor_fraction=0.20,
):
    """Return a peak near ``center_time``, falling back globally."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    target_index = pd.DatetimeIndex(time_index)
    if values_arr.size != len(target_index):
        raise ValueError("values and time_index must have the same length.")
    if values_arr.size == 0:
        return np.nan, None
    if center_time is None:
        return first_event_peak_abs_value_and_time(
            values_arr,
            target_index,
            min_separation_seconds=min_separation_seconds,
            prominence_fraction=prominence_fraction,
            noise_floor_fraction=noise_floor_fraction,
        )

    center_time = pd.Timestamp(center_time)
    half_window_seconds = float(half_window_seconds)
    if not np.isfinite(half_window_seconds) or half_window_seconds < 0.0:
        raise ValueError("half_window_seconds must be finite and non-negative.")

    seconds_from_center = np.abs((target_index - center_time).total_seconds())
    valid = np.isfinite(values_arr) & (seconds_from_center <= half_window_seconds)
    if not np.any(valid):
        return first_event_peak_abs_value_and_time(
            values_arr,
            target_index,
            min_separation_seconds=min_separation_seconds,
            prominence_fraction=prominence_fraction,
            noise_floor_fraction=noise_floor_fraction,
        )

    local_values = np.where(valid, values_arr, np.nan)
    peak_value, peak_time = first_event_peak_abs_value_and_time(
        local_values,
        target_index,
        min_separation_seconds=min_separation_seconds,
        prominence_fraction=prominence_fraction,
        noise_floor_fraction=noise_floor_fraction,
    )
    if peak_time is not None and np.isfinite(peak_value):
        return peak_value, peak_time

    local_abs_values = np.where(valid, np.abs(values_arr), -np.inf)
    peak_idx = int(np.argmax(local_abs_values))
    peak_value = float(local_abs_values[peak_idx])
    if not np.isfinite(peak_value):
        return np.nan, None
    return peak_value, pd.Timestamp(target_index[peak_idx])


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
    "first_event_peak_abs_value_and_time",
    "get_time_index_median_cadence_seconds",
    "local_peak_abs_value_and_time",
    "most_prominent_peak_abs_value_and_time",
    "prominent_peak_candidates",
    "resample_matrix_to_times",
    "resample_series_to_times",
    "vector_magnitude_from_component_series",
    "vector_magnitude_preserve_shape",
]
