"""Tests for reusable ground magnetometer station-data helpers."""

import numpy as np
import pandas as pd

from pynamit.magnetometers import (
    find_station_metadata,
    load_iaga2002_magnetometer_data,
    load_local_iaga2002_station_data,
    normalize_station_metadata,
    shift_station_datetime_index,
    station_has_complete_nonzero_components_at_times,
    station_source_time_window,
    station_window_has_nonzero_measurements,
)


def test_find_station_metadata_prefers_explicit_directory(tmp_path):
    """An explicit station directory defines the metadata source."""
    station_directory = tmp_path / "stations"
    station_directory.mkdir()
    metadata_path = station_directory / "stations_full_list.csv"
    metadata_path.write_text("IAGA,GEOLAT,GEOLON\naaa,60,190\n", encoding="utf-8")

    stations, source_path = find_station_metadata(
        tmp_path / "simulation", station_data_directory=station_directory
    )

    assert source_path == metadata_path
    assert stations["IAGA"].tolist() == ["AAA"]
    assert stations["GEOLON"].tolist() == [-170.0]


def test_load_local_station_data_applies_display_time_offset(tmp_path):
    """Station lookup and returned times share one explicit offset."""
    path = tmp_path / "aaa20200101vsec.sec"
    path.write_text(
        "\n".join(
            ["Format IAGA-2002", "DATE TIME AAAX AAAY AAAZ", "2020-01-01 23:59:58.000 1.0 2.0 3.0"]
        )
        + "\n",
        encoding="utf-8",
    )

    data = load_local_iaga2002_station_data(
        tmp_path, "AAA", "2020-01-02T00:00:03", data_time_offset_seconds=5.0
    )

    assert list(data.columns) == ["North", "East", "Down"]
    assert data.index.tolist() == [pd.Timestamp("2020-01-02T00:00:03")]
    np.testing.assert_array_equal(data.iloc[0], [1.0, 2.0, 3.0])


def test_normalize_station_metadata_uppercases_codes_and_wraps_longitude():
    """Station metadata should match notebook expectations."""
    stations = pd.DataFrame(
        {"IAGA": ["ipm", "res"], "GEOLAT": ["10.5", "-20"], "GEOLON": [190.0, -181.0]}
    )

    normalized = normalize_station_metadata(stations)

    assert normalized["IAGA"].tolist() == ["IPM", "RES"]
    np.testing.assert_allclose(normalized["GEOLAT"], np.array([10.5, -20.0]))
    np.testing.assert_allclose(normalized["GEOLON"], np.array([-170.0, 179.0]))


def test_station_time_offset_helpers_are_explicit():
    """Station time shifts should use an explicit offset."""
    index = pd.to_datetime(["2020-01-01T00:00:00", "2020-01-01T00:00:10"])

    shifted = shift_station_datetime_index(index, data_time_offset_seconds=5.0)
    source_start, source_end = station_source_time_window(
        "2020-01-01T00:00:05", "2020-01-01T00:00:15", data_time_offset_seconds=5.0
    )

    np.testing.assert_array_equal(shifted, index + pd.to_timedelta(5.0, unit="s"))
    assert source_start == pd.Timestamp("2020-01-01T00:00:00")
    assert source_end == pd.Timestamp("2020-01-01T00:00:10")


def test_station_availability_checks_require_all_nonzero_components():
    """Availability checks should reject all-zero components."""
    index = pd.date_range("2020-01-01", periods=3, freq="10s")
    data = pd.DataFrame(
        {"AAAX": [1.0, 2.0, 3.0], "AAAY": [4.0, 5.0, 6.0], "AAAZ": [7.0, 8.0, 9.0]}, index=index
    )

    assert station_window_has_nonzero_measurements(
        data, "AAA", index[0], index[-1], data_time_offset_seconds=0.0
    )
    assert station_has_complete_nonzero_components_at_times(data, "AAA", [index[1]])

    data["AAAY"] = 0.0
    assert not station_window_has_nonzero_measurements(data, "AAA", index[0], index[-1])
    assert not station_has_complete_nonzero_components_at_times(data, "AAA", [index[1]])


def test_load_iaga2002_magnetometer_data_converts_dhz_to_xyz(tmp_path):
    """IAGA2002 DHZ data should be converted to standardized XYZ."""
    path = tmp_path / "aaa20200101vsec.sec"
    path.write_text(
        "\n".join(
            [
                "Format IAGA-2002",
                "DATE TIME AAAD AAAH AAAZ",
                "2020-01-01 00:00:00.000 600.0 1000.0 5.0",
                "2020-01-01 00:00:01.000 0.0 99999.0 6.0",
            ]
        )
        + "\n"
    )

    data = load_iaga2002_magnetometer_data(path, "AAA")

    assert list(data.columns) == ["AAAX", "AAAY", "AAAZ"]
    assert data.index[0] == pd.Timestamp("2020-01-01T00:00:00")
    np.testing.assert_allclose(data["AAAX"].iloc[0], 1000.0 * np.cos(np.deg2rad(1.0)))
    np.testing.assert_allclose(data["AAAY"].iloc[0], 1000.0 * np.sin(np.deg2rad(1.0)))
    assert np.isnan(data["AAAX"].iloc[1])
