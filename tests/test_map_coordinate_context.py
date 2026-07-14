"""Tests for visualization map coordinate contexts."""

import datetime as dt
from types import SimpleNamespace

import cartopy.crs as ccrs
import numpy as np

from pynamit.coordinates import local_noon_longitude
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.pynameye import PynamEye


class FakeDipole:
    """Minimal dipole object for coordinate-context tests."""

    def mlt2mlon(self, mlt, time):
        """Map MLT to magnetic longitude."""
        del time
        return (float(mlt) - 12.0) * 15.0 + 40.0

    def mlon2mlt(self, mlon, time):
        """Map magnetic longitude to MLT."""
        del time
        return (12.0 + (np.asarray(mlon, dtype=float) - 40.0) / 15.0) % 24.0


class FakeApex:
    """Minimal apex object for coordinate-context tests."""

    def apex2geo(self, lat, lon, height):
        """Convert to a shifted geographic longitude."""
        return lat, np.asarray(lon, dtype=float) + 5.0, height


def test_geographic_context_matches_utc_local_time_helpers():
    """Geographic context reproduces UTC solar local time."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)
    context = MapCoordinateContext.geographic(reference_time)

    assert context.longitude_kind == "geographic"
    assert context.local_time_kind == "solar"
    assert context.label == "LT"
    assert context.noon_longitude == local_noon_longitude(reference_time)
    assert context.projection().equals(ccrs.PlateCarree(central_longitude=context.noon_longitude))
    np.testing.assert_allclose(
        context.local_time_grid_longitudes(), np.array([127.5, -142.5, -52.5, 37.5])
    )
    np.testing.assert_allclose(
        context.longitude_to_local_time(np.array([-97.5, -7.5])), np.array([12.0, 18.0])
    )


def test_magnetic_context_matches_dipole_mlt_conversion():
    """Magnetic context matches the dipole MLT convention."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)
    dipole = FakeDipole()
    context = MapCoordinateContext.magnetic(reference_time, dipole)
    mlon = np.array([40.0, 130.0, -140.0])

    assert context.longitude_kind == "magnetic"
    assert context.local_time_kind == "magnetic"
    assert context.label == "MLT"
    assert context.noon_longitude == 40.0
    np.testing.assert_allclose(
        context.longitude_to_local_time(mlon), dipole.mlon2mlt(mlon, reference_time)
    )


def test_apex_context_uses_geographic_longitude_for_global_maps():
    """Apex context converts magnetic noon into geographic longitude."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)
    context = MapCoordinateContext.magnetic(reference_time, FakeDipole(), apex=FakeApex())

    assert context.longitude_kind == "geographic"
    assert context.local_time_kind == "magnetic"
    assert context.label == "MLT"
    assert context.noon_longitude == 45.0
    assert context.longitude_to_local_time(45.0) == 12.0


def test_context_converts_source_local_time_longitude_to_plot_coordinate():
    """Context replaces ad hoc source-longitude rotations."""
    context = MapCoordinateContext.from_noon_longitude(
        -100.0, longitude_kind="geographic", local_time_kind="solar"
    )

    np.testing.assert_allclose(
        context.local_time_longitude_to_coordinate(
            np.array([-180.0, 0.0, 90.0]), local_noon_longitude=0.0
        ),
        np.array([80.0, -100.0, -10.0]),
    )


def test_pynameye_uses_distinct_global_and_magnetic_contexts():
    """PynamEye keeps global map and polar MLT contexts explicit."""
    eye = object.__new__(PynamEye)
    eye.time = dt.datetime(2011, 10, 24, 18, 30)
    eye.dp = FakeDipole()
    eye.apx = FakeApex()

    eye.settings = SimpleNamespace(main_field_kind="dipole")
    magnetic_context = eye.get_magnetic_coordinate_context()
    global_context = eye.get_global_coordinate_context()
    assert magnetic_context.longitude_kind == "magnetic"
    assert global_context == magnetic_context

    eye.settings = SimpleNamespace(main_field_kind="igrf")
    global_context = eye.get_global_coordinate_context()
    assert global_context.longitude_kind == "geographic"
    assert global_context.local_time_kind == "magnetic"
    assert global_context.noon_longitude == 45.0
    assert magnetic_context.noon_longitude == 40.0


def test_pynameye_uses_sm_noon_for_kaiju_dipole_context():
    """Kaiju dipole runs use SM longitude with noon at zero."""
    eye = object.__new__(PynamEye)
    eye.time = dt.datetime(2011, 10, 24, 18, 30)
    eye.dp = FakeDipole()
    eye.settings = SimpleNamespace(main_field_kind="kaiju_dipole")

    context = eye.get_magnetic_coordinate_context()

    assert context.longitude_kind == "magnetic"
    assert context.local_time_kind == "magnetic"
    assert context.noon_longitude == 0.0
