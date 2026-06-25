"""Tests for reusable visualization time-series helpers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pynamit.visualization.time_series import (
    compute_centered_difference_matrix_at_times,
    compute_centered_difference_series_at_times,
    compute_time_derivative_matrix,
    datetime_index_to_epoch_ns,
    first_event_peak_abs_value_and_time,
    get_time_index_median_cadence_seconds,
    local_peak_abs_value_and_time,
    prominent_peak_candidates,
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


def test_numeric_saved_times_use_mage_event_time_origin():
    """MAGE output seconds should be displayed from event t0."""
    from pynamit.visualization.run_fields import time_index_from_dataset

    dataset = xr.Dataset(coords={"time": np.array([0.0, 10.0, 20.0])})

    index = time_index_from_dataset(
        dataset, fallback_start_time=pd.Timestamp("2011-10-24 18:00:10")
    )

    expected = pd.DatetimeIndex(
        [
            "2011-10-24 18:00:10",
            "2011-10-24 18:00:20",
            "2011-10-24 18:00:30",
        ]
    )
    np.testing.assert_array_equal(index.values, expected.values)


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


def test_prominent_peak_candidates_separate_events():
    """Peak selection should prefer separated prominent peaks."""
    index = pd.date_range("2020-01-01", periods=8, freq="10s")
    values = np.array([0.0, 1.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.5])

    candidates = prominent_peak_candidates(
        values, index, min_separation_seconds=15.0, prominence_fraction=0.10
    )

    assert [candidate["index"] for candidate in candidates] == [1, 3, 5, 7]
    peak_value, peak_time = first_event_peak_abs_value_and_time(
        values,
        index,
        min_separation_seconds=15.0,
        prominence_fraction=0.10,
        noise_floor_fraction=0.20,
    )
    assert peak_value == 1.0
    assert peak_time == index[1]


def test_local_peak_abs_value_and_time_prefers_window():
    """Local peak selection should use the requested event window."""
    index = pd.date_range("2020-01-01", periods=8, freq="10s")
    values = np.array([0.0, 1.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.5])

    peak_value, peak_time = local_peak_abs_value_and_time(
        values,
        index,
        center_time=index[5],
        half_window_seconds=5.0,
        min_separation_seconds=15.0,
        prominence_fraction=0.10,
        noise_floor_fraction=0.20,
    )

    assert peak_value == 3.0
    assert peak_time == index[5]


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


def test_ground_dbdt_magnitude_differentiates_components_first():
    """Ground dB/dt magnitude is the magnitude of the dB/dt vector."""
    from pynamit.visualization.ground_figures import GroundFigureRenderer

    renderer = object.__new__(GroundFigureRenderer)
    renderer.spec = SimpleNamespace(dbdt_window_points=1)
    source_times = pd.date_range("2020-01-01", periods=3, freq="1s")
    target_times = pd.DatetimeIndex([source_times[1]])

    north = np.array([[0.0, 3.0, 6.0]])
    east = np.array([[0.0, 4.0, 8.0]])
    down = np.array([[0.0, 12.0, 24.0]])
    br_values = -down * 1e-9
    bh_values = np.stack([-north * 1e-9, east * 1e-9])

    magnitude = renderer.ground_matrix_at_times(
        "Magnitude", br_values, bh_values, source_times, target_times, quantity="dbdt"
    )

    np.testing.assert_allclose(magnitude, np.array([[13.0]]))


def test_station_dbdt_uses_supplied_simulation_cadence():
    """Measured dB/dt should use the common comparison cadence."""
    from pynamit.visualization.ground_figures import GroundFigureRenderer

    renderer = object.__new__(GroundFigureRenderer)
    renderer.spec = SimpleNamespace(
        ground_component="North", ground_quantity="dbdt", dbdt_window_points=1
    )
    measured_times = pd.date_range("2020-01-01", periods=21, freq="1s")
    measured = pd.DataFrame(
        {"North": np.arange(21, dtype=float) ** 3, "East": np.zeros(21), "Down": np.zeros(21)},
        index=measured_times,
    )

    values = renderer.station_values_at_times(
        measured, pd.DatetimeIndex([measured_times[10]]), dbdt_cadence_seconds=10.0
    )

    np.testing.assert_allclose(values, np.array([400.0]))


def test_reference_aligned_curve_centers_put_reference_sample_on_site():
    """Reference-line maps should intersect the station."""
    from pynamit.visualization.map_curves import reference_aligned_curve_centers

    site_lon = np.array([10.0])
    site_lat = np.array([60.0])
    normalized_time = np.array([0.0, 1.0])
    values = np.array([[1.0, 3.0]])
    curve_width = 10.0
    curve_height = 4.0
    value_scale = 2.0
    reference_position = 0.25

    curve_lon, curve_lat = reference_aligned_curve_centers(
        site_lon,
        site_lat,
        normalized_time,
        [{"label": "Measured", "series_key": "measured", "values": values}],
        curve_width_deg=curve_width,
        curve_height_deg=curve_height,
        value_scale=value_scale,
        reference_line={"position": reference_position},
    )

    reference_value = 1.5
    reference_lon = curve_lon + curve_width * (reference_position - 0.5)
    reference_lat = curve_lat + curve_height * (reference_value / value_scale)
    np.testing.assert_allclose(reference_lon, site_lon)
    np.testing.assert_allclose(reference_lat, site_lat)


def test_reference_aligned_curve_centers_keep_reference_x_on_site():
    """Reference time should align to site longitude."""
    from pynamit.visualization.map_curves import reference_aligned_curve_centers

    site_lon = np.array([-90.0, 45.0])
    site_lat = np.array([55.0, -35.0])
    normalized_time = np.array([0.0, 0.5, 1.0])
    center_values = np.array([[0.0, 2.0, 4.0], [6.0, 4.0, 2.0]])
    original_values = center_values.copy()
    curve_width = 12.0
    curve_height = 3.0
    value_scale = 2.0
    site_scale = np.array([1.0, 2.0])
    reference_position = 0.75

    curve_lon, curve_lat = reference_aligned_curve_centers(
        site_lon,
        site_lat,
        normalized_time,
        [{"label": "Inductive", "series_key": "inductive", "values": center_values}],
        curve_width_deg=curve_width,
        curve_height_deg=curve_height,
        value_scale=value_scale,
        site_curve_scale=site_scale,
        reference_line={"position": reference_position, "center_values": center_values},
    )

    reference_lon = curve_lon + curve_width * (reference_position - 0.5)
    reference_value = np.array([3.0, 3.0])
    reference_lat = curve_lat + curve_height * site_scale * (reference_value / value_scale)
    np.testing.assert_allclose(reference_lon, site_lon)
    np.testing.assert_allclose(reference_lat, site_lat)
    np.testing.assert_allclose(center_values, original_values)


def test_reference_aligned_curve_centers_cancel_drawn_reference_offset():
    """The center shift cancels the draw-time reference x offset."""
    from pynamit.visualization.map_curves import reference_aligned_curve_centers

    site_lon = np.array([-120.0, 5.0, 130.0])
    site_lat = np.array([40.0, 55.0, -25.0])
    normalized_time = np.array([0.0, 0.5, 1.0])
    values = np.array([[0.0, 2.0, 4.0], [3.0, 1.0, -1.0], [2.0, 2.0, 2.0]])
    curve_width = 14.0

    for reference_position in (0.0, 0.25, 0.75, 1.0):
        curve_lon, _ = reference_aligned_curve_centers(
            site_lon,
            site_lat,
            normalized_time,
            [{"label": "Measured", "series_key": "measured", "values": values}],
            curve_width_deg=curve_width,
            curve_height_deg=2.0,
            value_scale=4.0,
            reference_line={"position": reference_position},
        )

        drawn_reference_lon = curve_lon + curve_width * (reference_position - 0.5)
        np.testing.assert_allclose(drawn_reference_lon, site_lon)


def test_reference_aligned_curve_centers_prefers_measured_anchor():
    """Measured data should anchor reference placement."""
    from pynamit.visualization.map_curves import reference_aligned_curve_centers

    site_lon = np.array([10.0, 20.0])
    site_lat = np.array([50.0, 60.0])
    normalized_time = np.array([0.0, 1.0])
    measured = np.array([[10.0, 12.0], [20.0, 22.0]])
    inductive = np.array([[1.0, 3.0], [2.0, 4.0]])
    magnetostatic = np.array([[5.0, 7.0], [6.0, 8.0]])

    curve_lon, curve_lat = reference_aligned_curve_centers(
        site_lon,
        site_lat,
        normalized_time,
        [
            {"series_key": "measured", "values": measured},
            {"series_key": "inductive", "values": inductive},
            {"series_key": "magnetostatic", "values": magnetostatic},
        ],
        curve_width_deg=8.0,
        curve_height_deg=2.0,
        value_scale=10.0,
        reference_line={"position": 0.0},
    )

    np.testing.assert_allclose(curve_lon, site_lon + 4.0)
    np.testing.assert_allclose(curve_lat, site_lat - 2.0 * measured[:, 0] / 10.0)


def test_reference_aligned_curve_centers_averages_enabled_models():
    """Enabled simulation curves share the anchor."""
    from pynamit.visualization.map_curves import reference_aligned_curve_centers

    site_lon = np.array([10.0, 20.0])
    site_lat = np.array([50.0, 60.0])
    normalized_time = np.array([0.0, 1.0])
    inductive = np.array([[1.0, 3.0], [np.nan, 4.0]])
    magnetostatic = np.array([[5.0, 7.0], [6.0, np.nan]])

    curve_lon, curve_lat = reference_aligned_curve_centers(
        site_lon,
        site_lat,
        normalized_time,
        [
            {"series_key": "inductive", "values": inductive},
            {"series_key": "magnetostatic", "values": magnetostatic},
        ],
        curve_width_deg=8.0,
        curve_height_deg=2.0,
        value_scale=10.0,
        reference_line={"position": 0.5},
    )

    reference_values = np.array([4.0, 5.0])
    np.testing.assert_allclose(curve_lon, site_lon)
    np.testing.assert_allclose(curve_lat, site_lat - 2.0 * reference_values / 10.0)


def test_ground_plot_times_preserve_one_second_station_resolution():
    """Ground plots keep a 1 s measured-data grid."""
    from pynamit.visualization.ground_figures import GroundFigureRenderer

    renderer = object.__new__(GroundFigureRenderer)
    renderer.spec = SimpleNamespace(time_range=(0, 2), include_station_data=True)
    renderer.view = SimpleNamespace(
        n_time=3,
        time_index=pd.to_datetime(
            ["2020-01-01T00:00:00.004", "2020-01-01T00:00:10.000", "2020-01-01T00:00:20.019"]
        ),
    )

    plot_times = renderer.ground_plot_times()

    assert plot_times[0] == pd.Timestamp("2020-01-01T00:00:01")
    assert plot_times[-1] == pd.Timestamp("2020-01-01T00:00:20")
    assert len(plot_times) == 20
