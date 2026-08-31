"""Tests for external-input coordinate views and identities."""

import numpy as np
import pytest

from pynamit.external_inputs.coordinates import (
    LIBRARY_GEOGRAPHIC_110KM,
    PYNAMIT_CENTERED_DIPOLE_110KM,
    PYNAMIT_SPHERICAL_GEO_110KM,
    ExternalInputCoordinates,
    SampleGrid,
)
from pynamit.external_inputs.provider_definitions import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
)


def _coordinates(grid_id="geographic"):
    return ExternalInputCoordinates.from_geocentric_geo(
        np.array([-75.0, -20.0, 0.0, 45.0, 80.0]),
        np.array([-180.0, -30.0, 0.0, 90.0, 179.0]),
        grid_id=grid_id,
    )


def test_coordinate_views_reuse_one_converted_grid_object():
    """Equal provider contracts reuse one identity-mapped grid."""
    coordinates = _coordinates()
    hardy_grid = coordinates.sample_grid(CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention)
    amps_grid = coordinates.sample_grid(BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_convention)
    hwm_grid = coordinates.sample_grid(NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention)
    assert hardy_grid is amps_grid is hwm_grid
    assert hardy_grid.coordinate_convention is LIBRARY_GEOGRAPHIC_110KM
    np.testing.assert_array_equal(hardy_grid.lat, coordinates.geographic_grid.lat)
    np.testing.assert_array_equal(hardy_grid.lon, coordinates.geographic_grid.lon)


def test_model_coordinates_retain_centered_dipole_and_geographic_views():
    """Expose both required views of one ordered sample set."""
    model_lat = np.array([75.0, -60.0])
    model_lon = np.array([-20.0, 130.0])
    geo_lat = np.array([68.0, -52.0])
    geo_lon = np.array([70.0, -140.0])

    coordinates = ExternalInputCoordinates.from_model_coordinates(
        model_lat,
        model_lon,
        geographic_lat=geo_lat,
        geographic_lon=geo_lon,
        coordinate_system="centered_dipole",
        model_epoch=2001.5,
    )

    assert coordinates.model_grid.coordinate_convention is PYNAMIT_CENTERED_DIPOLE_110KM
    np.testing.assert_array_equal(coordinates.model_grid.lat, model_lat)
    np.testing.assert_array_equal(coordinates.model_grid.lon, model_lon)
    np.testing.assert_array_equal(coordinates.geographic_grid.lat, geo_lat)
    np.testing.assert_array_equal(coordinates.geographic_grid.lon, geo_lon)
    assert coordinates.model_epoch == pytest.approx(2001.5)
    assert coordinates.sample_grid(PYNAMIT_CENTERED_DIPOLE_110KM) is coordinates.model_grid


def test_centered_dipole_model_coordinates_require_epoch():
    """A magnetic coordinate view requires its axis epoch."""
    with pytest.raises(ValueError, match="require model_epoch"):
        ExternalInputCoordinates.from_model_coordinates(
            np.array([60.0]),
            np.array([10.0]),
            geographic_lat=np.array([50.0]),
            geographic_lon=np.array([20.0]),
            coordinate_system="centered_dipole",
        )


def test_geographic_model_coordinates_reuse_geographic_view():
    """A GEO model frame does not create a redundant coordinate grid."""
    lat = np.array([60.0, -40.0])
    lon = np.array([10.0, 120.0])
    coordinates = ExternalInputCoordinates.from_model_coordinates(
        lat, lon, geographic_lat=lat, geographic_lon=lon, coordinate_system="geocentric_geographic"
    )
    assert coordinates.model_grid is coordinates.geographic_grid


def test_geographic_model_coordinates_rejects_inconsistent_views():
    """Reject inconsistent GEO model and physical coordinates."""
    with pytest.raises(ValueError, match="must match geographic samples"):
        ExternalInputCoordinates.from_model_coordinates(
            np.array([60.0]),
            np.array([10.0]),
            geographic_lat=np.array([61.0]),
            geographic_lon=np.array([10.0]),
            coordinate_system="geocentric_geographic",
        )


def test_library_coordinate_mapping_is_numeric_identity():
    """The library mapping preserves spherical coordinate labels."""
    coordinates = _coordinates()
    provider_grid = coordinates.sample_grid(
        CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
    )
    np.testing.assert_array_equal(provider_grid.lat, coordinates.geographic_grid.lat)
    np.testing.assert_array_equal(provider_grid.lon, coordinates.geographic_grid.lon)
    assert provider_grid is not coordinates.geographic_grid
    assert provider_grid.coordinate_convention is LIBRARY_GEOGRAPHIC_110KM


def test_coordinate_identity_normalizes_longitude_and_preserves_order():
    """Equivalent longitudes match while reordered samples do not."""
    convention = PYNAMIT_SPHERICAL_GEO_110KM
    first = convention.coordinate_identity(np.array([10.0, 20.0]), np.array([180.0, 350.0]))
    equivalent = convention.coordinate_identity(np.array([10.0, 20.0]), np.array([-180.0, -10.0]))
    reordered = convention.coordinate_identity(np.array([20.0, 10.0]), np.array([-10.0, -180.0]))
    assert first == equivalent
    assert first != reordered


def test_coordinate_identity_ignores_float64_reconstruction_roundoff():
    """Sub-storage-precision differences identify the same grid."""
    convention = PYNAMIT_SPHERICAL_GEO_110KM
    lat = np.array([-40.14552, 13.086702597441118])
    lon = np.array([-20.99691648166619, 5.544013180231985])
    perturbed_lat = np.nextafter(lat, np.inf)
    perturbed_lon = np.nextafter(lon, -np.inf)

    assert not np.array_equal(lat, perturbed_lat)
    assert not np.array_equal(lon, perturbed_lon)
    assert convention.coordinate_identity(lat, lon) == convention.coordinate_identity(
        perturbed_lat, perturbed_lon
    )


def test_equal_arrays_under_different_conventions_are_different_grids():
    """Coordinate semantics are part of ordered-grid identity."""
    lat = np.array([10.0, 20.0])
    lon = np.array([0.0, 30.0])
    assert PYNAMIT_SPHERICAL_GEO_110KM.coordinate_identity(
        lat, lon
    ) != LIBRARY_GEOGRAPHIC_110KM.coordinate_identity(lat, lon)


def test_sample_grid_is_immutable_and_owns_arrays():
    """External mutation cannot alter a registered coordinate grid."""
    lat = np.array([10.0, 20.0])
    geometry = {"type": "sample_points"}
    grid = SampleGrid(
        grid_id="grid",
        coordinate_convention=PYNAMIT_SPHERICAL_GEO_110KM,
        lat=lat,
        lon=np.array([0.0, 30.0]),
        sampling_geometry=geometry,
    )
    lat[0] = -80.0
    geometry["type"] = "changed"
    assert grid.lat[0] == pytest.approx(10.0)
    assert grid.sampling_geometry["type"] == "sample_points"
    with pytest.raises(ValueError):
        grid.lat[0] = 0.0
    with pytest.raises(TypeError):
        grid.sampling_geometry["type"] = "changed"
