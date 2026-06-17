"""Tests for reusable visualization time-series helpers."""

import numpy as np
import pandas as pd
import pytest

from pynamit.visualization.time_series import (
    compute_centered_difference_matrix_at_times,
    compute_centered_difference_series_at_times,
    compute_time_derivative_matrix,
    datetime_index_to_epoch_ns,
    get_time_index_median_cadence_seconds,
    resample_matrix_to_times,
    resample_series_to_times,
    vector_magnitude_from_component_series,
    vector_magnitude_preserve_shape,
)


def test_datetime_index_to_epoch_ns_normalizes_resolution():
    """Datetime conversion should be independent of input resolution."""
    times = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[s]")

    epoch_ns = datetime_index_to_epoch_ns(times)

    assert epoch_ns.dtype == np.dtype("int64")
    np.testing.assert_array_equal(epoch_ns, np.array([1577836800000000000, 1577836801000000000]))


def test_resample_series_to_times_sorts_masks_duplicates_and_bounds():
    """Scalar resampling should match the notebook behavior."""
    index = pd.to_datetime(
        [
            "2020-01-01T00:01:00",
            "2020-01-01T00:00:00",
            "2020-01-01T00:01:00",
            "2020-01-01T00:02:00",
            "2020-01-01T00:04:00",
        ]
    )
    values = np.array([10.0, 0.0, np.nan, 20.0, 40.0])
    target = pd.to_datetime(
        [
            "2019-12-31T23:59:00",
            "2020-01-01T00:01:00",
            "2020-01-01T00:03:00",
            "2020-01-01T00:05:00",
        ]
    )

    resampled = resample_series_to_times(index, values, target)

    np.testing.assert_allclose(resampled, np.array([np.nan, 10.0, 30.0, np.nan]))


def test_resample_series_to_times_validates_length():
    """Length mismatches should fail close to the primitive."""
    with pytest.raises(ValueError, match="same length"):
        resample_series_to_times(pd.date_range("2020-01-01", periods=2), [1.0], [])


def test_resample_matrix_to_times_interpolates_rows():
    """Matrix resampling applies the scalar operation row-wise."""
    index = pd.date_range("2020-01-01", periods=3, freq="10s")
    values = np.array([[0.0, 10.0, 20.0], [0.0, 20.0, 40.0]])
    target = pd.to_datetime(["2020-01-01T00:00:05", "2020-01-01T00:00:15"])

    resampled = resample_matrix_to_times(index, values, target)

    np.testing.assert_allclose(resampled, np.array([[5.0, 15.0], [10.0, 30.0]]))


def test_time_index_median_cadence_uses_positive_steps():
    """Duplicate times should not create a zero cadence."""
    index = pd.to_datetime(
        [
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:10",
            "2020-01-01T00:00:10",
            "2020-01-01T00:00:25",
            "2020-01-01T00:00:40",
        ]
    )

    np.testing.assert_allclose(get_time_index_median_cadence_seconds(index), 15.0)
    assert np.isnan(get_time_index_median_cadence_seconds(index[:1]))


def test_centered_difference_series_uses_interpolated_window():
    """Centered differences should evaluate at target times."""
    index = pd.date_range("2020-01-01", periods=5, freq="10s")
    source_seconds = np.arange(5, dtype=float) * 10.0
    source_values = 2.0 * source_seconds
    target = pd.to_datetime(["2020-01-01T00:00:00", "2020-01-01T00:00:20"])

    derivative = compute_centered_difference_series_at_times(
        index, source_values, target, half_window_points=1
    )

    np.testing.assert_allclose(derivative, np.array([np.nan, 2.0]))


def test_centered_difference_matrix_applies_rows():
    """Matrix centered differences keep one row per source row."""
    index = pd.date_range("2020-01-01", periods=5, freq="10s")
    source_seconds = np.arange(5, dtype=float) * 10.0
    source_values = np.vstack([2.0 * source_seconds, -3.0 * source_seconds])
    target = pd.to_datetime(["2020-01-01T00:00:20"])

    derivative = compute_centered_difference_matrix_at_times(index, source_values, target)

    np.testing.assert_allclose(derivative, np.array([[2.0], [-3.0]]))


def test_compute_time_derivative_matrix_preserves_shape():
    """Same-grid derivatives should keep edge samples undefined."""
    index = pd.date_range("2020-01-01", periods=5, freq="10s")
    source_seconds = np.arange(5, dtype=float) * 10.0
    values = np.stack(
        [
            np.vstack([2.0 * source_seconds, -3.0 * source_seconds]),
            np.vstack([source_seconds**2, source_seconds + 5.0]),
        ]
    )

    derivative = compute_time_derivative_matrix(values, index, half_window_points=1)

    assert derivative.shape == values.shape
    np.testing.assert_allclose(
        derivative[0, :, 1:-1], np.array([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0]])
    )
    np.testing.assert_allclose(derivative[1, 0, 1:-1], np.array([20.0, 40.0, 60.0]))
    assert np.all(np.isnan(derivative[..., [0, -1]]))


def test_compute_time_derivative_matrix_rejects_invalid_time_axis():
    """Nonmonotonic or mismatched time axes should return NaNs."""
    invalid_index = pd.to_datetime(
        ["2020-01-01T00:00:00", "2020-01-01T00:00:10", "2020-01-01T00:00:05"]
    )

    derivative = compute_time_derivative_matrix(np.ones((2, 3)), invalid_index)
    mismatched = compute_time_derivative_matrix(np.ones((2, 4)), invalid_index)

    assert np.all(np.isnan(derivative))
    assert np.all(np.isnan(mismatched))


def test_vector_magnitude_helpers_preserve_nan_only_columns():
    """Magnitude should ignore partial NaNs but keep all-NaN samples."""
    magnitude = vector_magnitude_from_component_series(
        [np.array([3.0, np.nan, np.nan]), np.array([4.0, 5.0, np.nan])]
    )

    np.testing.assert_allclose(magnitude, np.array([5.0, 5.0, np.nan]))


def test_vector_magnitude_preserve_shape_keeps_grid_shape():
    """Grid-shaped component arrays should produce grid magnitudes."""
    first = np.array([[3.0, np.nan], [0.0, np.nan]])
    second = np.array([[4.0, 5.0], [0.0, np.nan]])

    magnitude = vector_magnitude_preserve_shape([first, second])

    np.testing.assert_allclose(magnitude, np.array([[5.0, 5.0], [0.0, np.nan]]))
