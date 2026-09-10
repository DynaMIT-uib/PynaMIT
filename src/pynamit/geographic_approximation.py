"""Simple spherical-Earth adapters for geographic library interfaces.

PynaMIT's operators live on a geocentric spherical shell. Hardy,
ApexPy, AMPS, and HWM expose geographic/geodetic interfaces. PynaMIT
deliberately uses a spherical-Earth approximation at those boundaries:
latitude and longitude are passed through numerically, and the nominal
spherical altitude is passed as the library altitude.

These functions make that approximation explicit and central without
introducing an ellipsoidal geometry into the model.
"""

from __future__ import annotations

import numpy as np


def _normalize_lat_lon(latitude, longitude):
    """Return broadcast finite latitude and wrapped east longitude."""
    latitude, longitude = np.broadcast_arrays(
        np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    )
    if np.any(~np.isfinite(latitude)) or np.any(~np.isfinite(longitude)):
        raise ValueError("Latitude and longitude must be finite.")
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError("Latitude must lie in [-90, 90] degrees.")
    longitude = (longitude + 180.0) % 360.0 - 180.0
    return latitude, longitude


def spherical_geo_to_library_geographic(geocentric_latitude, longitude, altitude_km):
    """Return the library-facing geographic approximation.

    Numerical latitude/longitude are unchanged.  ``altitude_km`` is the
    nominal altitude above PynaMIT's spherical reference surface and is
    passed unchanged to the external library.
    """
    latitude, longitude = _normalize_lat_lon(geocentric_latitude, longitude)
    altitude = np.broadcast_to(np.asarray(altitude_km, dtype=float), latitude.shape)
    if np.any(~np.isfinite(altitude)):
        raise ValueError("Altitude must be finite.")
    return latitude, longitude, altitude


def library_geographic_to_spherical_geo(latitude, longitude):
    """Interpret library angles as PynaMIT spherical GEO angles."""
    return _normalize_lat_lon(latitude, longitude)


def library_horizontal_to_spherical(east, north):
    """Map library east/north to spherical theta/phi components."""
    east, north = np.broadcast_arrays(
        np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    )
    if np.any(~np.isfinite(east)) or np.any(~np.isfinite(north)):
        raise ValueError("Horizontal vector components must be finite.")
    return -north, east
