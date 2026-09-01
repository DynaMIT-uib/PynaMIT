"""Tests for main-field model conventions."""

import datetime as dt

import dipole
import numpy as np
import pytest
from kompe.constants import EARTH_RADIUS_M

from pynamit.coordinates import decimal_year, local_noon_longitude
from pynamit.geomagnetism import MainField
from pynamit.geomagnetism.kaiju_geopack import (
    KaijuGeopackMAG,
    KaijuGeopackSM,
    axis_lat_lon,
    kaiju_geopack_coefficients,
    kaiju_geopack_dipole,
    kaiju_geopack_mag,
    kaiju_geopack_sm,
)
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
    assert main_field.B0 == pytest.approx(b0)

    Br, Btheta, Bphi = main_field.field_components(EARTH_RADIUS_M, 90.0, 0.0)
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


def test_kaiju_geopack_coefficients_cover_all_of_calendar_2025():
    """Kaiju's final supported year does not end on January 1."""
    end_of_2025 = kaiju_geopack_coefficients(dt.datetime(2025, 12, 31, 23, 59, 59))

    assert 2025.0 < end_of_2025.epoch_value < 2026.0
    with pytest.raises(ValueError, match="before 2026"):
        kaiju_geopack_coefficients(dt.datetime(2026, 1, 1))


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
    assert main_field.B0 == pytest.approx(b0)

    geo_latitude, geo_longitude = kaiju_geopack_mag(epoch).mag2geo(0.0, 0.0)
    Br, Btheta, Bphi = main_field.field_components(
        EARTH_RADIUS_M, 90.0 - geo_latitude, geo_longitude
    )
    np.testing.assert_allclose(Br, 0.0, atol=1e-20)
    np.testing.assert_allclose(np.hypot(Btheta, Bphi), b0)


def test_main_field_horizontal_coordinate_system_labels_are_explicit():
    """Main-field kinds advertise their horizontal coordinates."""
    assert (
        MainField(kind="kaiju_dipole", epoch=2011).horizontal_coordinate_system
        == "geocentric_geographic"
    )
    assert MainField(kind="dipole", epoch=2011).horizontal_coordinate_system == "centered_dipole"
    assert (
        MainField(kind="igrf", epoch=2011).horizontal_coordinate_system == "geocentric_geographic"
    )
    assert (
        MainField(kind="radial", epoch=2011).horizontal_coordinate_system
        == "geocentric_geographic"
    )


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


@pytest.mark.parametrize("kind", ["dipole", "kaiju_dipole", "radial"])
def test_main_field_components_preserve_broadcast_shape(kind):
    """Field components and inclination follow NumPy broadcasting."""
    main_field = MainField(kind=kind, epoch=2011)
    radius = np.array([[EARTH_RADIUS_M], [EARTH_RADIUS_M + 110e3]])
    theta = np.array([[35.0], [120.0]])
    longitude = np.array([[-90.0, 0.0, 90.0]])

    components = main_field.field_components(radius, theta, longitude)

    assert all(component.shape == (2, 3) for component in components)
    assert main_field.inclination_sine(radius, theta, longitude).shape == (2, 3)


def test_main_field_from_config_uses_canonical_settings():
    """Saved and live simulations share one main-field constructor."""
    config = SimulationConfig(
        RI=EARTH_RADIUS_M + 130.0e3,
        main_field_kind="kaiju_dipole",
        main_field_epoch=2011.5,
        main_field_B0=29_000.0e-9,
    )

    main_field = build_main_field(config)

    assert main_field.kind == "kaiju_dipole"
    assert main_field.epoch == pytest.approx(2011.5)
    assert main_field.ionosphere_height_km == pytest.approx(130.0)
    assert main_field.dipole.B0 == pytest.approx(29_000.0)


def test_kaiju_main_field_model_coordinates_are_geographic():
    """Kaiju simulation coordinates are Earth-fixed GEO."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011)
    model_coordinates = main_field.geo_to_model_coordinates(65.0, -30.0)

    np.testing.assert_allclose(model_coordinates, (65.0, -30.0))


def test_main_field_vector_transform_requires_east_and_north_pair():
    """Partial tangent-vector inputs are rejected instead of ignored."""
    main_field = MainField(kind="dipole", epoch=2011)

    with pytest.raises(ValueError, match="east and north"):
        main_field.geo_to_model_coordinates(65.0, -30.0, east=1.0)


def test_dipole_model_geo_coordinates_and_vectors_round_trip():
    """Round-trip centered-dipole positions and tangent components."""
    main_field = MainField(kind="dipole", epoch=2011.5)
    model_lat = np.array([-75.0, -20.0, 35.0, 80.0])
    model_lon = np.array([-170.0, -30.0, 65.0, 150.0])
    model_east = np.array([1.0, -2.0, 3.0, -4.0])
    model_north = np.array([5.0, 6.0, -7.0, -8.0])

    geo = main_field.model_to_geo_coordinates(
        model_lat, model_lon, east=model_east, north=model_north
    )
    restored = main_field.geo_to_model_coordinates(geo[0], geo[1], east=geo[2], north=geo[3])

    np.testing.assert_allclose(restored[0], model_lat, atol=1e-12)
    np.testing.assert_allclose(restored[1], model_lon, atol=1e-12)
    np.testing.assert_allclose(restored[2], model_east, atol=1e-12)
    np.testing.assert_allclose(restored[3], model_north, atol=1e-12)


def test_kaiju_main_field_evaluates_dipole_in_mag_and_returns_geo_components():
    """Kaiju dipole physics is internal MAG with GEO output vectors."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011.0)
    reference = kaiju_geopack_mag(2011.0)
    lat = np.array([65.0, -50.0, 10.0])
    lon = np.array([-30.0, 120.0, 5.0])
    magnetic_latitude, magnetic_longitude = reference.geo2mag(lat, lon)
    magnetic_north, magnetic_radial = main_field.dipole.B(magnetic_latitude, EARTH_RADIUS_M * 1e-3)
    _, _, expected_east, expected_north = reference.mag2geo(
        magnetic_latitude,
        magnetic_longitude,
        east=np.zeros_like(magnetic_north),
        north=magnetic_north * 1e-9,
    )

    br, btheta, bphi = main_field.field_components(EARTH_RADIUS_M, 90.0 - lat, lon)

    np.testing.assert_allclose(br, magnetic_radial * 1e-9)
    np.testing.assert_allclose(btheta, -expected_north)
    np.testing.assert_allclose(bphi, expected_east)


def test_kaiju_main_field_mapping_and_conjugacy_are_returned_in_geo():
    """Analytic MAG operations preserve the GEO API boundary."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011.0)
    radius = np.array([2.0 * EARTH_RADIUS_M, 2.5 * EARTH_RADIUS_M])
    destination = EARTH_RADIUS_M
    magnetic_latitude = np.array([45.0, -50.0])
    magnetic_longitude = np.array([-30.0, 80.0])
    magnetic_coordinates = kaiju_geopack_mag(2011.0)
    latitude, longitude = magnetic_coordinates.mag2geo(magnetic_latitude, magnetic_longitude)

    theta_mapped, longitude_mapped = main_field.map_along_field_lines(
        destination, radius, 90.0 - latitude, longitude
    )
    mapped_latitude_mag, mapped_longitude_mag = magnetic_coordinates.geo2mag(
        90.0 - theta_mapped, longitude_mapped
    )
    expected_unsigned_latitude = 90.0 - np.rad2deg(
        np.arcsin(np.cos(np.deg2rad(magnetic_latitude)) * np.sqrt(destination / radius))
    )
    np.testing.assert_allclose(
        mapped_latitude_mag, np.sign(magnetic_latitude) * expected_unsigned_latitude
    )
    np.testing.assert_allclose(mapped_longitude_mag, magnetic_longitude)

    theta_conjugate, longitude_conjugate = main_field.conjugate_coordinates(
        radius, 90.0 - latitude, longitude
    )
    conjugate_latitude_mag, conjugate_longitude_mag = magnetic_coordinates.geo2mag(
        90.0 - theta_conjugate, longitude_conjugate
    )
    np.testing.assert_allclose(conjugate_latitude_mag, -magnetic_latitude)
    np.testing.assert_allclose(conjugate_longitude_mag, magnetic_longitude)


def test_kaiju_main_field_basis_vectors_are_dual_in_geo_components():
    """Rotating analytic MAG basis vectors preserves their duality."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011.0)
    radius = np.array([EARTH_RADIUS_M + 110e3, EARTH_RADIUS_M + 110e3])
    latitude = np.array([65.0, -55.0])
    longitude = np.array([-40.0, 100.0])

    d1, d2, d3, e1, e2, e3 = main_field.basis_vectors(radius, 90.0 - latitude, longitude)
    duality = np.einsum("icn,jcn->ijn", np.stack((d1, d2, d3)), np.stack((e1, e2, e3)))

    expected = np.repeat(np.eye(3)[:, :, None], radius.size, axis=2)
    np.testing.assert_allclose(duality, expected, atol=1e-12)


def test_kaiju_main_field_model_noon_is_geographic_noon():
    """The Kaiju GEO model centers global maps on geographic noon."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    main_field = MainField(kind="kaiju_dipole", epoch=decimal_year(event_time))

    assert main_field.local_noon_longitude(event_time) == local_noon_longitude(event_time)


def test_kaiju_main_field_alignment_metadata_distinguishes_geo_mag_and_sm():
    """Metadata distinguishes model GEO from internal MAG/source SM."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    main_field = MainField(kind="kaiju_dipole", epoch=2011)

    metadata = main_field.alignment_metadata(event_time)

    assert metadata["main_field_horizontal_coordinate_system"] == "geocentric_geographic"
    assert metadata["dipole_alignment_model"] == "kaiju_geopack_centered_dipole"
    assert metadata["noon_model_longitude_deg"] != pytest.approx(0.0)
    np.testing.assert_allclose(metadata["axis_geo_cartesian"], main_field.dipole.axis)
    np.testing.assert_allclose(metadata["dipole_mag_z_axis_geo_cartesian"], main_field.dipole.axis)
    np.testing.assert_allclose(
        metadata["dipole_sm_z_axis_geo_cartesian"],
        kaiju_geopack_sm(event_time).sm_to_geo_matrix[:, 2],
    )


def test_kaiju_mag_transform_matches_dipole_package_coordinates():
    """MAG rotation uses Kaiju's standard centered-dipole axes."""
    epoch = dt.datetime(2011, 10, 24, 18, 0, 10)
    transform = kaiju_geopack_mag(epoch)
    dipole_model = kaiju_geopack_dipole(epoch)
    lat = np.array([65.0, -50.0, 10.0])
    lon = np.array([-30.0, 120.0, 5.0])

    observed = transform.geo2mag(lat, lon)
    expected = dipole_model.geo2mag(lat, lon)

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(
        ((observed[1] - expected[1] + 180.0) % 360.0) - 180.0, 0.0, atol=1e-12
    )


def test_kaiju_main_field_magnetic_coordinate_api_round_trips_geo():
    """Public GEO/MAG conversions use the internal Kaiju frame."""
    main_field = MainField(kind="kaiju_dipole", epoch=2011.0)
    latitude = np.array([65.0, -50.0, 10.0])
    longitude = np.array([-30.0, 120.0, 5.0])

    magnetic_latitude, magnetic_longitude = main_field.geographic_to_magnetic_coordinates(
        latitude, longitude
    )
    geographic_latitude, geographic_longitude = main_field.magnetic_to_geographic_coordinates(
        magnetic_latitude, magnetic_longitude
    )

    np.testing.assert_allclose(geographic_latitude, latitude, atol=1e-12)
    np.testing.assert_allclose(
        ((geographic_longitude - longitude + 180.0) % 360.0) - 180.0, 0.0, atol=1e-12
    )


def test_kaiju_sm_differs_from_mag_only_by_timestamped_longitude():
    """SM and MAG share a dipole axis but not a longitude origin."""
    first_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    last_time = dt.datetime(2011, 10, 24, 19, 0, 0)
    mag = kaiju_geopack_mag(first_time)
    lat = np.array([65.0, -50.0, 10.0])
    lon = np.array([-30.0, 120.0, 5.0])

    offsets = []
    for event_time in (first_time, last_time):
        sm = kaiju_geopack_sm(event_time)
        geo_lat, geo_lon = sm.sm2geo(lat, lon)
        mag_lat, mag_lon = mag.geo2mag(geo_lat, geo_lon)
        np.testing.assert_allclose(mag_lat, lat, atol=1e-12)
        longitude_offset = ((mag_lon - lon + 180.0) % 360.0) - 180.0
        np.testing.assert_allclose(longitude_offset, longitude_offset[0], atol=1e-12)
        offsets.append(longitude_offset[0])

    assert abs(offsets[1] - offsets[0]) > 14.0


def test_kaiju_sm_transform_uses_dipole_axis_as_z_axis():
    """Kaiju SM has the Geopack dipole axis as north pole."""
    sm = kaiju_geopack_sm(2011.0)

    np.testing.assert_allclose(sm.sm_to_geo_matrix[:, 2], sm.coefficients.axis, atol=1e-12)

    pole_lat, pole_lon = kaiju_geopack_dipole(2011.0).north_pole
    sm_lat, _ = sm.geo2sm(pole_lat, pole_lon)
    np.testing.assert_allclose(sm_lat, 90.0, atol=1e-10)


def test_kaiju_transforms_normalize_timezone_aware_epochs_to_utc():
    """Equivalent instants must produce identical SM and MAG frames."""
    utc_time = dt.datetime(2011, 10, 24, 18, 0, 10, tzinfo=dt.timezone.utc)
    offset_time = dt.datetime(2011, 10, 24, 20, 0, 10, tzinfo=dt.timezone(dt.timedelta(hours=2)))

    utc_sm = kaiju_geopack_sm(utc_time)
    offset_sm = kaiju_geopack_sm(offset_time)
    utc_mag = kaiju_geopack_mag(utc_time)
    offset_mag = kaiju_geopack_mag(offset_time)

    assert utc_sm.epoch == dt.datetime(2011, 10, 24, 18, 0, 10)
    assert offset_sm.epoch == utc_sm.epoch
    np.testing.assert_allclose(offset_sm.geo_to_sm_matrix, utc_sm.geo_to_sm_matrix, atol=1e-15)
    np.testing.assert_allclose(offset_mag.geo_to_mag_matrix, utc_mag.geo_to_mag_matrix, atol=1e-15)
    assert decimal_year(offset_time) == pytest.approx(decimal_year(utc_time))


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


def test_kaiju_mag_transform_owns_an_immutable_rotation_matrix():
    """MAG transforms cannot be changed through caller-owned arrays."""
    reference = kaiju_geopack_mag(2011.0)
    supplied_matrix = reference.geo_to_mag_matrix.copy()

    transform = KaijuGeopackMAG(reference.epoch, reference.coefficients, supplied_matrix)
    supplied_matrix[0, 0] = 0.0

    np.testing.assert_allclose(transform.geo_to_mag_matrix, reference.geo_to_mag_matrix)
    assert not transform.geo_to_mag_matrix.flags.writeable


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

    assert main_field.B0 == pytest.approx(reference_b0)
    Br, Btheta, Bphi = main_field.field_components(EARTH_RADIUS_M, 90.0, 0.0)
    np.testing.assert_allclose(Br, reference_b0)
    np.testing.assert_allclose(Btheta, 0.0, atol=0.0)
    np.testing.assert_allclose(Bphi, 0.0, atol=0.0)


def test_radial_basis_vectors_are_field_aligned_and_orthonormal():
    """Radial-field apex vectors follow the spherical convention."""
    main_field = MainField(kind="radial", epoch=2011)
    radius = np.array([EARTH_RADIUS_M, EARTH_RADIUS_M + 110e3])
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

    geo_lat, geo_lon = main_field.magnetic_latitude_trace_to_geographic(
        60, magnetic_longitude=np.array([0, 90])
    )
    assert np.all(np.isnan(geo_lon))
    assert np.all(np.isnan(geo_lat))


def test_igrf_apex_forward_boundary_uses_spherical_identity(monkeypatch):
    """IGRF GEO-to-Apex keeps spherical positions."""
    captured = {}

    class FakeApex:
        refh = 110.0

        def geo2apex(self, latitude, longitude, height):
            captured["latitude"] = np.asarray(latitude)
            captured["longitude"] = np.asarray(longitude)
            captured["height"] = np.asarray(height)
            return np.asarray(latitude) + 1.0, np.asarray(longitude) + 2.0

    main_field = MainField(kind="igrf", epoch=2011, ionosphere_height_km=110.0)
    main_field.apex = FakeApex()
    latitude = np.array([0.0, 45.0, 70.0])
    longitude = np.array([-30.0, 20.0, 100.0])

    main_field.geographic_to_magnetic_coordinates(latitude, longitude)

    np.testing.assert_allclose(captured["latitude"], latitude)
    np.testing.assert_allclose(captured["longitude"], longitude)
    np.testing.assert_allclose(captured["height"], 110.0)


def test_igrf_magnetic_latitude_uses_spherical_radial_altitude():
    """Apex receives spherical radial altitude."""
    captured = {}

    class FakeApex:
        refh = 110.0

        def geo2apex(self, latitude, longitude, height):
            captured["latitude"] = np.asarray(latitude)
            captured["longitude"] = np.asarray(longitude)
            captured["height"] = np.asarray(height)
            return np.asarray(latitude), np.asarray(longitude)

    main_field = MainField(kind="igrf", epoch=2011, ionosphere_height_km=110.0)
    main_field.apex = FakeApex()
    radius = EARTH_RADIUS_M + np.array([100e3, 110e3, 120e3])
    latitude = np.array([0.0, 45.0, 80.0])
    longitude = np.array([-20.0, 0.0, 30.0])

    main_field.magnetic_latitude(radius, 90.0 - latitude, longitude)

    np.testing.assert_allclose(captured["latitude"], latitude)
    np.testing.assert_allclose(captured["longitude"], longitude)
    np.testing.assert_allclose(captured["height"], [100.0, 110.0, 120.0])


def test_igrf_apex_inverse_is_interpreted_as_spherical_geo():
    """Apex output is interpreted on the PynaMIT sphere."""

    class FakeApex:
        refh = 110.0

        def apex2geo(self, latitude, longitude, height):
            return (
                np.asarray(latitude) - 1.0,
                np.asarray(longitude) + 2.0,
                np.zeros_like(np.asarray(latitude), dtype=float),
            )

    main_field = MainField(kind="igrf", epoch=2011, ionosphere_height_km=110.0)
    main_field.apex = FakeApex()

    latitude, longitude = main_field.magnetic_to_geographic_coordinates(
        np.array([60.0]), np.array([179.0])
    )
    np.testing.assert_allclose(latitude, [59.0])
    np.testing.assert_allclose(longitude, [-179.0])
