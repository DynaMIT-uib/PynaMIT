"""Tests for main-field model conventions."""

import datetime as dt
import numpy as np
import dipole
import pytest

from pynamit.math.constants import RE
from pynamit.simulation.kaiju_dipole import (
    kaiju_geopack_coefficients,
    kaiju_geopack_dipole,
    kaiju_geopack_sm,
)
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.mainfield import Mainfield, mainfield_from_config


def test_dipole_B0_override_preserves_epoch_alignment():
    """Dipole B0 changes magnitude without moving the pole."""
    epoch = 2011
    b0 = 29617.369174957275e-9

    mainfield = Mainfield(kind="dipole", epoch=epoch, B0=b0)
    reference = dipole.Dipole(epoch)

    np.testing.assert_allclose(mainfield.dpl.north_pole, reference.north_pole)
    assert mainfield.dpl.B0 == b0 * 1e9

    Br, Btheta, Bphi = mainfield.get_B(RE, 90.0, 0.0)
    np.testing.assert_allclose(Br, 0.0, atol=1e-20)
    np.testing.assert_allclose(Btheta, -b0)
    np.testing.assert_allclose(Bphi, 0.0, atol=0.0)


def test_kaiju_geopack_dipole_uses_embedded_degree_one_coefficients():
    """Kaiju dipole placement follows the Geopack degree-1 formula."""
    coefficients = kaiju_geopack_coefficients(2011.0)

    f2 = (2011.0 - 2010.0) / 5.0
    f1 = 1.0 - f2
    expected_g10 = f1 * -29496.57 + f2 * -29441.46
    expected_g11 = f1 * -1586.42 + f2 * -1501.77
    expected_h11 = f1 * 4944.26 + f2 * 4795.99
    expected_b0 = np.sqrt(expected_g10**2 + expected_g11**2 + expected_h11**2)
    expected_axis = -np.array([expected_g11, expected_h11, expected_g10]) / expected_b0

    assert coefficients.g10 == pytest.approx(expected_g10)
    assert coefficients.g11 == pytest.approx(expected_g11)
    assert coefficients.h11 == pytest.approx(expected_h11)
    np.testing.assert_allclose(coefficients.axis, expected_axis)


def test_kaiju_dipole_mainfield_B0_override_preserves_kaiju_alignment():
    """Kaiju dipole B0 changes magnitude without moving the pole."""
    epoch = 2011
    b0 = 29617.369174957275e-9

    mainfield = Mainfield(kind="kaiju_dipole", epoch=epoch, B0=b0)
    reference = kaiju_geopack_dipole(epoch)

    np.testing.assert_allclose(mainfield.dpl.north_pole, reference.north_pole)
    np.testing.assert_allclose(mainfield.dpl.axis, reference.axis)
    assert mainfield.dpl.B0 == b0 * 1e9

    Br, Btheta, Bphi = mainfield.get_B(RE, 90.0, 0.0)
    np.testing.assert_allclose(Br, 0.0, atol=1e-20)
    np.testing.assert_allclose(Btheta, -b0)
    np.testing.assert_allclose(Bphi, 0.0, atol=0.0)


def test_mainfield_coordinate_system_labels_are_explicit():
    """Main-field kinds advertise their horizontal coordinates."""
    assert Mainfield(kind="kaiju_dipole", epoch=2011).coordinate_system == "SM"
    assert Mainfield(kind="dipole", epoch=2011).coordinate_system == "centered_dipole_magnetic"
    assert Mainfield(kind="igrf", epoch=2011).coordinate_system == "geographic"
    assert Mainfield(kind="radial", epoch=2011).coordinate_system == "geographic"


def test_mainfield_from_config_uses_canonical_settings():
    """Saved-run and dynamics code share one main-field constructor."""
    config = SimulationConfig(
        RI=RE + 130.0e3,
        mainfield_kind="kaiju_dipole",
        mainfield_epoch=2011.5,
        mainfield_B0=29_000.0e-9,
    )

    mainfield = mainfield_from_config(config)

    assert mainfield.kind == "kaiju_dipole"
    assert mainfield.epoch == pytest.approx(2011.5)
    assert mainfield.hI == pytest.approx(130.0)
    assert mainfield.dpl.B0 == pytest.approx(29_000.0)


def test_kaiju_mainfield_geo_transform_requires_event_time():
    """SM coordinates are Sun-aligned, so a date/time is required."""
    mainfield = Mainfield(kind="kaiju_dipole", epoch=2011)

    with pytest.raises(ValueError, match="requires event_time"):
        mainfield.geo_to_model_coordinates(65.0, -30.0)


def test_mainfield_vector_transform_requires_east_and_north_pair():
    """Partial tangent-vector inputs are rejected instead of ignored."""
    mainfield = Mainfield(kind="dipole", epoch=2011)

    with pytest.raises(ValueError, match="east and north"):
        mainfield.geo_to_model_coordinates(65.0, -30.0, east=1.0)


def test_kaiju_mainfield_geo_transform_uses_geopack_sm():
    """Mainfield conversion delegates to the Kaiju SM transform."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    mainfield = Mainfield(kind="kaiju_dipole", epoch=2011)
    reference = kaiju_geopack_sm(event_time)
    lat = np.array([65.0, -50.0, 10.0])
    lon = np.array([-30.0, 120.0, 5.0])
    east = np.array([25.0, -10.0, 3.0])
    north = np.array([5.0, 40.0, -2.0])

    result = mainfield.geo_to_model_coordinates(lat, lon, east, north, event_time=event_time)
    expected = reference.geo2sm(lat, lon, east, north)

    np.testing.assert_allclose(result[0], expected[0])
    np.testing.assert_allclose(((result[1] - expected[1] + 180.0) % 360.0) - 180.0, 0.0)
    np.testing.assert_allclose(result[2], expected[2])
    np.testing.assert_allclose(result[3], expected[3])


def test_kaiju_mainfield_local_time_longitude_is_sm_longitude():
    """REMIX noon-based longitude is the kaiju_dipole SM longitude."""
    mainfield = Mainfield(kind="kaiju_dipole", epoch=2011)
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])

    result = mainfield.local_time_longitude_to_model_longitude(
        lon, dt.datetime(2011, 10, 24, 18, 0, 10)
    )

    np.testing.assert_allclose(result, np.array([-180.0, -90.0, 0.0, 90.0, -180.0]))


def test_kaiju_mainfield_alignment_metadata_is_sm_based():
    """Kaiju alignment metadata keeps SM noon at longitude zero."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    mainfield = Mainfield(kind="kaiju_dipole", epoch=2011)

    metadata = mainfield.alignment_metadata(event_time)

    assert metadata["mainfield_coordinate_system"] == "SM"
    assert metadata["dipole_alignment_model"] == "kaiju_geopack_centered_dipole"
    assert metadata["noon_mlon_deg"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        metadata["axis_geo_cartesian"], kaiju_geopack_sm(event_time).sm_to_geo_matrix[:, 2]
    )


def test_kaiju_sm_transform_uses_dipole_axis_as_z_axis():
    """Kaiju SM has the Geopack dipole axis as north pole."""
    sm = kaiju_geopack_sm(2011.0)

    np.testing.assert_allclose(sm.sm_to_geo_matrix[:, 2], sm.coefficients.axis, atol=1e-12)

    pole_lat, pole_lon = kaiju_geopack_dipole(2011.0).north_pole
    sm_lat, _ = sm.geo2sm(pole_lat, pole_lon)
    np.testing.assert_allclose(sm_lat, 90.0, atol=1e-10)


def test_kaiju_sm_transform_round_trips_coordinates_and_vectors():
    """Kaiju GEO-SM preserves coordinates and tangent vectors."""
    sm = kaiju_geopack_sm(2011.0)
    lat = np.array([65.0, -50.0, 10.0])
    lon = np.array([-30.0, 120.0, 5.0])
    east = np.array([25.0, -10.0, 3.0])
    north = np.array([5.0, 40.0, -2.0])

    sm_lat, sm_lon, sm_east, sm_north = sm.geo2sm(lat, lon, east, north)
    geo_lat, geo_lon, geo_east, geo_north = sm.sm2geo(sm_lat, sm_lon, sm_east, sm_north)

    np.testing.assert_allclose(geo_lat, lat, atol=1e-12)
    np.testing.assert_allclose(((geo_lon - lon + 180.0) % 360.0) - 180.0, 0.0, atol=1e-12)
    np.testing.assert_allclose(geo_east, east, atol=1e-12)
    np.testing.assert_allclose(geo_north, north, atol=1e-12)


def test_dipole_magnetic_latitude_trace_uses_geographic_conversion():
    """Magnetic-latitude traces are geographic, not parallels."""
    mainfield = Mainfield(kind="dipole", epoch=2011)
    magnetic_longitude = np.array([-120.0, 0.0, 120.0])
    magnetic_latitude = 65.0

    geo_lon, geo_lat = mainfield.magnetic_latitude_trace_to_geo(
        magnetic_latitude, magnetic_longitude
    )
    expected_lat, expected_lon = mainfield.dpl.mag2geo(
        np.full_like(magnetic_longitude, magnetic_latitude), magnetic_longitude
    )

    np.testing.assert_allclose(geo_lat, expected_lat)
    np.testing.assert_allclose(((geo_lon - expected_lon + 180.0) % 360.0) - 180.0, 0.0)


def test_radial_default_B0_is_in_tesla():
    """Radial field uses the same SI-unit B0 convention as dipole."""
    epoch = 2011
    reference_b0 = dipole.Dipole(epoch).B0 * 1e-9

    mainfield = Mainfield(kind="radial", epoch=epoch)

    Br, Btheta, Bphi = mainfield.get_B(RE, 90.0, 0.0)
    np.testing.assert_allclose(Br, reference_b0)
    np.testing.assert_allclose(Btheta, 0.0, atol=0.0)
    np.testing.assert_allclose(Bphi, 0.0, atol=0.0)
