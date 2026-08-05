"""Tests for the explicit spherical library-interface approximation."""

import numpy as np
import pytest

from pynamit.geodesy import (
    library_geographic_to_spherical_geo,
    library_horizontal_to_spherical,
    spherical_geo_to_library_geographic,
)


def test_library_geographic_mapping_is_numeric_identity():
    """The adapter preserves spherical latitude/longitude labels."""
    latitude = np.array([-80.0, -45.0, 0.0, 45.0, 80.0])
    longitude = np.array([-180.0, -30.0, 0.0, 90.0, 350.0])

    mapped_lat, mapped_lon, height = spherical_geo_to_library_geographic(
        latitude, longitude, 110.0
    )

    np.testing.assert_allclose(mapped_lat, latitude)
    np.testing.assert_allclose(mapped_lon, [-180.0, -30.0, 0.0, 90.0, -10.0])
    np.testing.assert_allclose(height, 110.0)


def test_library_geographic_inverse_is_numeric_identity():
    """Library angles are interpreted on the PynaMIT sphere."""
    latitude, longitude = library_geographic_to_spherical_geo(
        np.array([-20.0, 30.0]), np.array([190.0, -200.0])
    )
    np.testing.assert_allclose(latitude, [-20.0, 30.0])
    np.testing.assert_allclose(longitude, [-170.0, 160.0])


def test_library_horizontal_components_use_spherical_sign_convention():
    """East maps to phi and north maps to negative theta."""
    theta, phi = library_horizontal_to_spherical(
        east=np.array([5.0, -10.0]), north=np.array([20.0, -30.0])
    )
    np.testing.assert_allclose(theta, [-20.0, 30.0])
    np.testing.assert_allclose(phi, [5.0, -10.0])


def test_library_mapping_rejects_invalid_coordinates():
    """The simple adapter still validates its numerical inputs."""
    with pytest.raises(ValueError, match="Latitude"):
        spherical_geo_to_library_geographic([91.0], [0.0], 110.0)
    with pytest.raises(ValueError, match="Altitude"):
        spherical_geo_to_library_geographic([0.0], [0.0], np.nan)
