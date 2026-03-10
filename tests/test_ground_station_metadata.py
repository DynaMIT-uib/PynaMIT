import pandas as pd
import pytest

from pynamit.postprocess.ground_station import (
    load_iaga2002_station_data,
    normalize_station_metadata,
)


def test_normalize_station_metadata_wraps_longitude_and_uppercases_iaga():
    stations = pd.DataFrame(
        {
            "IAGA": ["abk", "AIA", "CMO"],
            "GEOLAT": [68.358, -65.245, 64.874],
            "GEOLON": [18.823, 295.742, 212.140],
        }
    )

    normalized = normalize_station_metadata(stations)

    assert list(normalized["IAGA"]) == ["ABK", "AIA", "CMO"]
    assert normalized["GEOLON"].tolist() == pytest.approx([18.823, -64.258, -147.86])


def _write_iaga_file(path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_iaga2002_station_data_supports_reported_xyzf_with_hdzf_sensor_orientation(tmp_path):
    filepath = tmp_path / "clf.sec"
    _write_iaga_file(
        filepath,
        """ Format                 IAGA-2002                                    |
 Reported               XYZF                                         |
 Sensor Orientation     HDZF                                         |
 DATE       TIME         CLFX     CLFY     CLFZ     CLFF
 2011-10-24 00:00:00     1.0      2.0      3.0      4.0
""",
    )

    data = load_iaga2002_station_data(filepath, "CLF")

    assert data is not None
    assert list(data.columns) == ["CLFX", "CLFY", "CLFZ"]
    assert data.iloc[0].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_load_iaga2002_station_data_converts_hdzf_to_xyz(tmp_path):
    filepath = tmp_path / "cmo.sec"
    _write_iaga_file(
        filepath,
        """ Format                 IAGA-2002                                    |
 Reported               HDZF                                         |
 Sensor Orientation     HDZF                                         |
 DATE       TIME         CMOD     CMOH     CMOZ     CMOF
 2011-10-24 00:00:00     600.0    10.0     3.0      4.0
""",
    )

    data = load_iaga2002_station_data(filepath, "CMO")

    assert data is not None
    assert list(data.columns) == ["CMOX", "CMOY", "CMOZ"]
    assert data.iloc[0].tolist() == pytest.approx([9.9984769516, 0.1745240644, 3.0])


def test_load_iaga2002_station_data_rejects_reported_column_mismatch(tmp_path):
    filepath = tmp_path / "bad.sec"
    _write_iaga_file(
        filepath,
        """ Format                 IAGA-2002                                    |
 Reported               XYZF                                         |
 Sensor Orientation     XYZF                                         |
 DATE       TIME         BADD     BADH     BADZ     BADF
 2011-10-24 00:00:00     600.0    10.0     3.0      4.0
""",
    )

    data = load_iaga2002_station_data(filepath, "BAD")

    assert data is None
