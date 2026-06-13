"""Tests for input-vs-interpolated plotting helpers."""

import numpy as np

import pynamit.visualization as visualization
from pynamit.coordinates import (
    local_time_longitude_to_geographic as canonical_local_time_conversion,
)
from pynamit.visualization.input_vs_interpolated import (
    local_time_longitude_to_geographic,
)


def test_local_time_longitude_to_geographic_rotates_noon_meridian():
    """Raw local-time noon maps to geographic noon."""
    lon = np.array([-180.0, 0.0, 90.0, 180.0])

    converted = local_time_longitude_to_geographic(
        lon,
        noon_longitude=-100.0,
        local_noon_longitude=0.0,
    )

    np.testing.assert_allclose(converted, np.array([80.0, -100.0, -10.0, 80.0]))


def test_local_time_longitude_to_geographic_supports_nonzero_source_noon():
    """Nonzero source-grid noon can be converted."""
    converted = local_time_longitude_to_geographic(
        np.array([90.0]),
        noon_longitude=-100.0,
        local_noon_longitude=90.0,
    )

    np.testing.assert_allclose(converted, np.array([-100.0]))


def test_input_vs_interpolated_reexports_canonical_helpers():
    """Diagnostic modules re-export canonical coordinate helpers."""
    from pynamit.visualization import input_vs_interpolated as canonical

    assert canonical.local_time_longitude_to_geographic is canonical_local_time_conversion
    assert local_time_longitude_to_geographic is canonical_local_time_conversion
    assert visualization.local_time_longitude_to_geographic is canonical_local_time_conversion
