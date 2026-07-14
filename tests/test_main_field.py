"""Tests for main-field model conventions."""

import datetime as dt

import dipole
import numpy as np
import pytest

from pynamit.geomagnetism import MainField
from pynamit.geomagnetism.kaiju_geopack import (
    KaijuGeopackSM,
    axis_lat_lon,
    kaiju_geopack_coefficients,
    kaiju_geopack_dipole,
    kaiju_geopack_sm,
)
from pynamit.math.constants import RE
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.geometry import build_main_field


def test_dipole_B0_override_preserves_epoch_alignment():
    """Dipole B0 changes magnitude without moving the pole."""
    epoch = 2011
    b0 = 29617.369174957275e-9

    main_field = MainField(kind="dipole", epoch=epoch, B0=b0)
    reference = dipole.Dipole(epoch)

    np.testing.assert_allclose(main_field.dipole.north_pole, reference.north_pole)
    assert main_field.dipole.B0 == b0 * 1e9

    Br, Btheta, Bphi = main_field.field_components(RE, 90.0, 0.0)
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


def test_axis_lat_lon_rejects_degenerate_axes():
    """Axis coordinates require a meaningful Cartesian direction."""
    with pytest.raises(ValueError, match="finite non-zero norm"):
        axis_lat_lon(np.zeros(3))
    with pytest.raises(ValueError, match="finite non-zero norm"):
        axis_lat_lon(np.array([np.nan, 0.0, 1.0]))


def test_kaiju_dipole_main_field_B0_override_preserves_kaiju_alignment():
    """Kaiju dipole B0 changes magnitude without moving the pole."""
    epoch = 2011
    b0 = 29617.369174957275e-9

    main_field = MainField(kind="kaiju_dipole", epoch=epoch, B0=b0)
    reference = kaiju_geopack_dipole(epoch)

    np.testing.assert_allclose(main_field.dipole.north_pole, reference.north_pole)
    np.testing.assert_allclose(main_field.dipole.axis, reference.axis)
    assert main_field.dipole.B0 == b0 * 1e9

    Br, Btheta, Bphi = main_field.field_components(RE, 90.0, 0.0)
    np.testing.assert_allclose(Br, 0.0, atol=1e-20)
    np.testing.assert_allclose(Btheta, -b0)
    np.testing.assert_allclose(Bphi, 0.0, atol=0.0)


def test_main_field_horizontal_coordinate_system_labels_are_explicit():
    """Main-field kinds advertise their horizontal coordinates."""
    assert MainField(kind="kaiju_dipole", epoch=2011).horizontal_coordinate_system == "SM"
    assert (
        MainField(kind="dipole", epoch=2011).horizontal_coordinate_system
        == "centered_dipole_magnetic"
    )
    assert MainField(kind="igrf", epoch=2011).horizontal_coordinate_system == "geographic"
    assert MainField(kind="radial", epoch=2011).horizontal_coordinate_system == "geographic"


def test_main_field_kind_uses_configuration_normalization():
    """Direct construction shares canonical model-name handling."""
    assert MainField(kind=" DIPOLE ", epoch=2011).kind == "dipole"

    with pytest.raises(ValueError, match="main_field_kind"):
        MainField(kind="unknown", epoch=2011)


def test_main_field_rejects_invalid_or_inapplicable_physical_parameters():
    """Direct construction enforces the same physical domain."""
    with pytest.raises(ValueError, match="epoch"):
        MainField(epoch=np.nan)
    with pytest.raises(ValueError, match="B0"):
        MainField(B0=0.0)
    with pytest.raises(ValueError, match="IGRF"):
        MainField(kind="igrf", B0=3e-5)


@pytest.mark.parametrize("kind", ["dipole", "radial"])
def test_main_field_components_preserve_broadcast_shape(kind):
    """Field components and inclination follow NumPy broadcasting."""
    main_field = MainField(kind=kind, epoch=2011)
    radius = np.array([[RE], [RE + 110e3]])
    theta = np.array([[35.0], [120.0]])
    longitude = np.array([[-90.0, 0.0, 90.0]])

    components = main_field.field_components(radius, theta, longitude)

    assert all(component.shape == (2, 3) for component in components)
    assert main_field.inclination_sine(radius, theta, longitude).shape == (2, 3)


def test_main_field_from_config_uses_canonical_settings():
    """Saved and live runs share one main-field constructor."""
    config = SimulationConfig(
        RI=RE + 130.0e3,
        main_field_kind="kaiju_dipole",
        main_field_epoch=2011.5,
        main_field_B0=29_000.0e-9,
    )

    main_field = build_main_field(config)

    assert main_field.kind == "kaiju_dipole"
    assert main_field.epoch == pytest.approx(2011.5)
    assert main_field.ionosphere_height_km == pytest.approx(130.0)
    assert main_field.dipole.B0 == pytest.approx(29_000.0)


def test_kaiju_main_field_geo_transform_requires_event_time():
    """SM coordinates are Sun-aligned, so a date/time is required."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011)

    with pytest.raises(ValueError, match="requires event_time"):
        main_field.geo_to_model_coordinates(65.0, -30.0)


def test_main_field_vector_transform_requires_east_and_north_pair():
    """Partial tangent-vector inputs are rejected instead of ignored."""
    main_field = MainField(kind="dipole", epoch=2011)

    with pytest.raises(ValueError, match="east and north"):
        main_field.geo_to_model_coordinates(65.0, -30.0, east=1.0)


def test_kaiju_main_field_geo_transform_uses_geopack_sm():
    """MainField conversion delegates to the Kaiju SM transform."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    main_field = MainField(kind="kaiju_dipole", epoch=2011)
    reference = kaiju_geopack_sm(event_time)
    lat = np.array([65.0, -50.0, 10.0])
    lon = np.array([-30.0, 120.0, 5.0])
    east = np.array([25.0, -10.0, 3.0])
    north = np.array([5.0, 40.0, -2.0])

    result = main_field.geo_to_model_coordinates(lat, lon, east, north, event_time=event_time)
    expected = reference.geo2sm(lat, lon, east, north)

    np.testing.assert_allclose(result[0], expected[0])
    np.testing.assert_allclose(((result[1] - expected[1] + 180.0) % 360.0) - 180.0, 0.0)
    np.testing.assert_allclose(result[2], expected[2])
    np.testing.assert_allclose(result[3], expected[3])


def test_kaiju_main_field_local_time_longitude_is_sm_longitude():
    """REMIX noon-based longitude is the kaiju_dipole SM longitude."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011)
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])

    result = main_field.local_time_longitude_to_model_longitude(
        lon, dt.datetime(2011, 10, 24, 18, 0, 10)
    )

    np.testing.assert_allclose(result, np.array([-180.0, -90.0, 0.0, 90.0, -180.0]))


def test_kaiju_main_field_alignment_metadata_is_sm_based():
    """Kaiju alignment metadata keeps SM noon at longitude zero."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    main_field = MainField(kind="kaiju_dipole", epoch=2011)

    metadata = main_field.alignment_metadata(event_time)

    assert metadata["main_field_horizontal_coordinate_system"] == "SM"
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


def test_kaiju_sm_transform_owns_an_immutable_rotation_matrix():
    """SM transforms cannot be changed through caller-owned arrays."""
    reference = kaiju_geopack_sm(2011.0)
    supplied_matrix = reference.geo_to_sm_matrix.copy()

    transform = KaijuGeopackSM(reference.epoch, reference.coefficients, supplied_matrix)
    supplied_matrix[0, 0] = 0.0

    np.testing.assert_allclose(transform.geo_to_sm_matrix, reference.geo_to_sm_matrix)
    assert not transform.geo_to_sm_matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        transform.geo_to_sm_matrix[0, 0] = 0.0


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 2)), "shape"),
        (np.full((3, 3), np.nan), "finite"),
        (np.diag([1.0, 1.0, 2.0]), "orthogonal"),
        (np.diag([1.0, 1.0, -1.0]), "proper rotation"),
    ],
)
def test_kaiju_sm_transform_rejects_invalid_rotation_matrices(matrix, message):
    """SM transforms validate rotation invariants at construction."""
    reference = kaiju_geopack_sm(2011.0)

    with pytest.raises(ValueError, match=message):
        KaijuGeopackSM(reference.epoch, reference.coefficients, matrix)


def test_dipole_magnetic_latitude_trace_uses_geographic_conversion():
    """Magnetic-latitude traces are geographic, not parallels."""
    main_field = MainField(kind="dipole", epoch=2011)
    magnetic_longitude = np.array([-120.0, 0.0, 120.0])
    magnetic_latitude = 65.0

    geo_lat, geo_lon = main_field.magnetic_latitude_trace_to_geographic(
        magnetic_latitude, magnetic_longitude
    )
    expected_lat, expected_lon = main_field.dipole.mag2geo(
        np.full_like(magnetic_longitude, magnetic_latitude), magnetic_longitude
    )

    np.testing.assert_allclose(geo_lat, expected_lat)
    np.testing.assert_allclose(((geo_lon - expected_lon + 180.0) % 360.0) - 180.0, 0.0)


def test_radial_default_B0_is_in_tesla():
    """Radial field uses the same SI-unit B0 convention as dipole."""
    epoch = 2011
    reference_b0 = dipole.Dipole(epoch).B0 * 1e-9

    main_field = MainField(kind="radial", epoch=epoch)

    Br, Btheta, Bphi = main_field.field_components(RE, 90.0, 0.0)
    np.testing.assert_allclose(Br, reference_b0)
    np.testing.assert_allclose(Btheta, 0.0, atol=0.0)
    np.testing.assert_allclose(Bphi, 0.0, atol=0.0)


def test_radial_basis_vectors_are_field_aligned_and_orthonormal():
    """Radial-field apex vectors follow the spherical convention."""
    main_field = MainField(kind="radial", epoch=2011)
    radius = np.array([RE, RE + 110e3])
    theta = np.array([30.0, 120.0])
    phi = np.array([-45.0, 80.0])

    d1, d2, d3, e1, e2, e3 = main_field.basis_vectors(radius, theta, phi)
    contravariant = np.stack((d1, d2, d3))
    covariant = np.stack((e1, e2, e3))
    duality = np.einsum("icn,jcn->ijn", contravariant, covariant)

    expected_identity = np.repeat(np.eye(3)[:, :, None], radius.size, axis=2)
    np.testing.assert_allclose(duality, expected_identity)
    np.testing.assert_allclose(d1, np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(d2, np.array([[0.0, 0.0], [-1.0, -1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(d3, np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def test_radial_undefined_magnetic_traces_are_nan_for_integer_inputs():
    """Undefined radial-field traces retain floating NaN values."""
    main_field = MainField(kind="radial", epoch=2011)

    assert np.all(np.isnan(main_field.magnetic_colatitude_at_longitude([0, 90])))
    geo_lat, geo_lon = main_field.magnetic_latitude_trace_to_geographic(
        60, magnetic_longitude=np.array([0, 90])
    )
    assert np.all(np.isnan(geo_lon))
    assert np.all(np.isnan(geo_lat))
