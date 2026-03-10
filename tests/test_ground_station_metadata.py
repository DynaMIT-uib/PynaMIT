import pandas as pd
import pytest

from pynamit.postprocess.ground_station import normalize_station_metadata


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
