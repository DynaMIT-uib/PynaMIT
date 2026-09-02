"""Tests for input projection-comparison plotting helpers."""

import numpy as np

from pynamit.coordinates import (
    local_time_longitude_to_geographic as canonical_local_time_conversion,
)


def test_local_time_longitude_to_geographic_rotates_noon_meridian():
    """Raw local-time noon maps to geographic noon."""
    lon = np.array([-180.0, 0.0, 90.0, 180.0])

    converted = canonical_local_time_conversion(
        lon, noon_longitude=-100.0, local_noon_longitude=0.0
    )

    np.testing.assert_allclose(converted, np.array([80.0, -100.0, -10.0, 80.0]))


def test_local_time_longitude_to_geographic_supports_nonzero_source_noon():
    """Nonzero geographic-grid noon can be converted."""
    converted = canonical_local_time_conversion(
        np.array([90.0]), noon_longitude=-100.0, local_noon_longitude=90.0
    )

    np.testing.assert_allclose(converted, np.array([-100.0]))
