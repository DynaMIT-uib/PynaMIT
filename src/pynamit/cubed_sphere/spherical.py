"""Spherical Coordinate Utilities.

This module provides functions for converting between spherical and Cartesian coordinates, transforming coordinate systems, and other operations performed in spherical coordinates.
"""

import numpy as np

d2r = np.pi / 180
r2d = 180 / np.pi


def sph_to_car(sph, deg=True):
    """
    Convert from spherical to Cartesian coordinates.

    Converts a 3 x N array representing spherical coordinates to a 3 x N array of Cartesian coordinates.

    Parameters
    ----------
    sph : array-like
        A 3 x N array containing the spherical coordinates [r, colatitude, longitude].
    deg : bool, optional
        If True, the input angles are in degrees; otherwise in radians. Default is True.

    Returns
    -------
    ndarray
        A 3 x N array containing the Cartesian coordinates [x, y, z].
    """
    r, theta, phi = sph
    conv = 1.0 if not deg else d2r
    return np.vstack(
        (
            r * np.sin(theta * conv) * np.cos(phi * conv),
            r * np.sin(theta * conv) * np.sin(phi * conv),
            r * np.cos(theta * conv),
        )
    )


def car_to_sph(car, deg=True):
    """
    Convert from Cartesian to spherical coordinates.

    Converts a 3 x N array representing Cartesian coordinates to a 3 x N array of spherical coordinates.

    Parameters
    ----------
    car : array-like
        A 3 x N array containing the Cartesian coordinates [x, y, z].
    deg : bool, optional
        If True, the output angles are in degrees; otherwise in radians. Default is True.

    Returns
    -------
    ndarray
        A 3 x N array containing the spherical coordinates [r, colatitude, longitude].
    """
    x, y, z = car
    conv = 1.0 if not deg else r2d
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) * conv
    phi = ((np.arctan2(y, x) * 180 / np.pi) % 360) / 180 * np.pi * conv
    return np.vstack((r, theta, phi))


def sph_to_sph(lat, lon, x_lat, x_lon, z_lat, z_lon, deg=True):
    """
    Transform spherical coordinates from one system to another.

    Given input latitudes and longitudes, computes the corresponding coordinates in a new spherical system defined by new x and z axes.

    Parameters
    ----------
    lat : array-like
        Input latitudes.
    lon : array-like
        Input longitudes.
    x_lat : float
        Latitude of the new x-axis.
    x_lon : float
        Longitude of the new x-axis.
    z_lat : float
        Latitude of the new z-axis.
    z_lon : float
        Longitude of the new z-axis.
    deg : bool, optional
        If True, the input and output angles are in degrees; otherwise in radians. Default is True.

    Returns
    -------
    tuple of ndarray
        A tuple containing the transformed latitude and longitude in the new coordinate system.
    """
    lat, lon = lat.flatten(), lon.flatten()
    conv = 1.0 if not deg else d2r
    xyz = np.vstack(
        (
            np.cos(lat * conv) * np.cos(lon * conv),
            np.cos(lat * conv) * np.sin(lon * conv),
            np.sin(lat * conv),
        )
    )
    new_z = np.array(
        [
            np.cos(z_lat * conv) * np.cos(z_lon * conv),
            np.cos(z_lat * conv) * np.sin(z_lon * conv),
            np.sin(z_lat * conv),
        ]
    )
    new_x = np.array(
        [
            np.cos(x_lat * conv) * np.cos(x_lon * conv),
            np.cos(x_lat * conv) * np.sin(x_lon * conv),
            np.sin(x_lat * conv),
        ]
    )
    new_y = np.cross(new_z, new_x)
    new_x, new_y, new_z = new_x.flatten(), new_y.flatten(), new_z.flatten()
    if not np.isclose(np.linalg.norm(new_y), 1):
        raise ValueError("x and z coordinates do not define orthogonal directions")
    r = np.vstack((new_x, new_y, new_z))
    xyz = r.dot(xyz)
    _, colat, lon = car_to_sph(xyz, deg=deg)
    return (90 - colat, lon) if deg else (np.pi / 2 - colat, lon)


def enu_to_ecef(v, lon, lat, reverse=False):
    """
    Convert between ENU and ECEF coordinate systems.

    Converts an array of vector components from East-North-Up (ENU) to Earth-Centered Earth-Fixed (ECEF) coordinates, or vice versa if reverse is True.

    Parameters
    ----------
    v : array-like
        An N x 3 array of vector components in ENU (or ECEF if reverse is True).
    lon : array-like
        An array of longitudes in degrees.
    lat : array-like
        An array of latitudes in degrees.
    reverse : bool, optional
        If True, converts from ECEF to ENU; otherwise, from ENU to ECEF. Default is False.

    Returns
    -------
    ndarray
        An N x 3 array of vector components in the target coordinate system.
    """
    phi = lon * d2r
    theta = (90 - lat) * d2r
    unit_east = np.vstack((-np.sin(phi), np.cos(phi), np.zeros_like(phi))).T
    unit_north = np.vstack(
        (
            -np.cos(theta) * np.cos(phi),
            -np.cos(theta) * np.sin(phi),
            np.sin(theta),
        )
    ).T
    unit_up = np.vstack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    ).T
    enu_to_ecef_or_reverse = np.stack(
        (unit_east, unit_north, unit_up), axis=1 if reverse else 2
    )
    return np.einsum("nij, nj -> ni", enu_to_ecef_or_reverse, v)


def ecef_to_enu(v, lon, lat):
    """
    Convert vectors from ECEF to ENU coordinates.

    Convenience wrapper around enu_to_ecef() with the reverse flag set to True.

    Parameters
    ----------
    v : array-like
        An N x 3 array of vector components in ECEF coordinates.
    lon : array-like
        An array of longitudes in degrees.
    lat : array-like
        An array of latitudes in degrees.

    Returns
    -------
    ndarray
        An N x 3 array of vector components in ENU coordinates.
    """
    return enu_to_ecef(v, lon, lat, reverse=True)


def tangent_vector(lat1, lon1, lat2, lon2, degrees=True):
    """
    Calculate the tangential unit vector on a sphere.

    Computes a unit vector tangent to the sphere at the point (lat1, lon1) pointing towards (lat2, lon2).

    Parameters
    ----------
    lat1 : array-like
        Latitude(s) of the origin point (not colatitude).
    lon1 : array-like
        Longitude(s) of the origin point.
    lat2 : array-like
        Latitude(s) of the target point.
    lon2 : array-like
        Longitude(s) of the target point.
    degrees : bool, optional
        If True, inputs are in degrees; otherwise, in radians. Default is True.

    Returns
    -------
    tuple of ndarray
        A tuple (east, north) representing the eastward and northward components of the tangential unit vector.

    Raises
    ------
    ValueError
        If input arrays do not have the same shape or if the tangent is undefined (points nearly identical or antipodal).
    """
    if not (lat1.shape == lon1.shape == lat2.shape == lon2.shape):
        raise ValueError("tangent_vector: input coordinates do not have equal shapes")

    shape = lat1.shape

    # convert to radians if necessary, and flatten:
    if degrees:

        def converter(x):
            return x.flatten() * np.pi / 180.0

    else:

        def converter(x):
            return x.flatten()

    lat1, lon1, lat2, lon2 = list(map(converter, (lat1, lon1, lat2, lon2)))

    # ECEF position vectors:
    ecef_p1 = np.vstack(
        (
            np.cos(lat1) * np.cos(lon1),
            np.cos(lat1) * np.sin(lon1),
            np.sin(lat1),
        )
    )
    ecef_p2 = np.vstack(
        (
            np.cos(lat2) * np.cos(lon2),
            np.cos(lat2) * np.sin(lon2),
            np.sin(lat2),
        )
    )

    # Check if tangent is well defined:
    if np.any(np.isclose(np.sum((ecef_p1 * ecef_p2) ** 2, axis=0), 1.0)):
        points = np.isclose(np.sum((ecef_p1 * ecef_p2) ** 2, axis=0), 1.0).nonzero()[0]
        raise ValueError(
            "tangent_vector: input coordinates at nearly identical or antipodal points; tangent not defined\n flattened coordinates: %s"
            % points
        )

    # Non-tangential difference vector (3, N):
    dp = ecef_p2 - ecef_p1

    # Subtract normal part of the vectors to make tangential vector in ECEF coordinates:
    ecef_t = dp - np.sum(dp * ecef_p1, axis=0) * ecef_p1

    # Normalize:
    ecef_t = ecef_t / np.linalg.norm(ecef_t, axis=0)

    # Convert ecef_t to enu_t, by constructing N rotation matrices:
    R = np.dstack(
        (
            np.vstack((-np.sin(lon1), np.cos(lon1), 0 * lat1)).T,
            np.vstack(
                (
                    -np.cos(lon1) * np.sin(lat1),
                    -np.sin(lon1) * np.sin(lat1),
                    np.cos(lat1),
                )
            ).T,
            np.vstack(
                (
                    np.cos(lon1) * np.cos(lat1),
                    np.sin(lon1) * np.cos(lat1),
                    np.sin(lat1),
                )
            ).T,
        )
    )

    enu_t = np.einsum("lji, jl->il", R, ecef_t)
    # Third coordinate (up) is zero, since normal part was removed
    enu_t = enu_t[:2]

    # Extract east and north components, reshape to original shape and return stacked
    east = enu_t[0].reshape(shape)
    north = enu_t[1].reshape(shape)

    return east, north


def geo2local(lat, lon, Ae, An, lon0, lat0, inverse=False):
    """
    Convert geographic coordinates and ENU vector components to a local coordinate system.

    Transforms input geographic coordinates and east/north components to a local system where (lon0, lat0) defines the new pole.

    Parameters
    ----------
    lat : array-like
        Geographic latitudes in degrees.
    lon : array-like
        Geographic longitudes in degrees.
    Ae : array-like
        Eastward vector components.
    An : array-like
        Northward vector components.
    lon0 : float
        Longitude of the new pole in degrees.
    lat0 : float
        Latitude of the new pole in degrees.
    inverse : bool, optional
        If True, performs the inverse transformation (local to geographic). Default is False.

    Returns
    -------
    tuple of ndarray
        A tuple (local_lat, local_lon, Ae_local, An_local) with transformed coordinates and vectors.

    Raises
    ------
    Exception
        If input arrays have inconsistent shapes.
    """
    try:
        lat, lon, Ae, An = np.broadcast_arrays(lat, lon, Ae, An)
        shape = lat.shape
        lat, lon, Ae, An = (
            lat.flatten(),
            lon.flatten(),
            Ae.flatten(),
            An.flatten(),
        )
    except ValueError:
        raise Exception("Input arrays have inconsistent shapes")
    lat, lon = lat.flatten(), lon.flatten()
    # Make rotation matrix from geo to local
    Z = np.array(
        [
            np.cos(np.deg2rad(lat0)) * np.cos(np.deg2rad(lon0)),
            np.cos(np.deg2rad(lat0)) * np.sin(np.deg2rad(lon0)),
            np.sin(np.deg2rad(lat0)),
        ]
    )

    Zgeo_x_Z = np.cross(np.array([0, 0, 1]), Z)
    Y = Zgeo_x_Z / np.linalg.norm(Zgeo_x_Z)
    X = np.cross(Y, Z)

    # Rotation matrix from geographic to local (ECEF)
    Rgeo_to_local = np.vstack((X, Y, Z))

    if inverse:  # Transpose rotation matrix to get inverse operation
        Rgeo_to_local = Rgeo_to_local.T

    # Convert input to ECEF:
    colat = 90 - lat
    r_geo = sph_to_car(np.vstack((np.ones_like(colat), colat, lon)), deg=True)

    # Rotate:
    r_local = Rgeo_to_local.dot(r_geo)

    # Convert result back to spherical:
    _, colat_local, lon_local = car_to_sph(r_local, deg=True)

    A_geo_enu = np.vstack((Ae, An, np.zeros(Ae.size)))
    A = np.sqrt(Ae**2 + An**2)
    # Rotate normalized vectors to ecef
    A_geo_ecef = enu_to_ecef((A_geo_enu / A).T, lon, lat)
    A_local_ecef = Rgeo_to_local.dot(A_geo_ecef.T)
    A_local_enu = ecef_to_enu(A_local_ecef.T, lon_local, 90 - colat_local).T * A

    # Return coords and vector components:
    return (
        90 - colat_local.reshape(shape),
        lon_local.reshape(shape),
        A_local_enu[0].reshape(shape),
        A_local_enu[1].reshape(shape),
    )


if __name__ == "__main__":

    # TESTING ENU/ECEF CONVERSION:
    v = np.array([[1, 1, 0], [1, 0, 0]])
    lat = np.array([-90, 0])
    lon = np.array([0.0, 0])
    print(enu_to_ecef(v, lat, lon))

    v = (np.random.random((30, 3)) - 0.5) * 300
    lat = (np.random.random(30) - 0.5) * 180
    lon = np.random.random(30) * 360
    print(
        "This number should be small: ",
        np.max(enu_to_ecef(enu_to_ecef(v, lat, lon), lat, lon, reverse=True) - v) ** 2,
    )

    # Testing the geo2local function by comparing it to equivalent code in Dipole module
    from dipole import Dipole

    d = Dipole(2010)
    lat0, lon0 = d.north_pole

    # Test points:
    x, y, z = np.random.random((3, 1000)) * 2 - 1
    r = x**2 + y**2 + z**2
    x, y, z = x[r <= 1], y[r <= 1], z[r <= 1]
    r, colat, lon = car_to_sph(np.vstack((x, y, z)))

    # Test vector components
    Ae, An = np.random.random((2, sum(r <= 1)))

    newlat, newlon, neweast, newnorth = geo2local(90 - colat, lon, Ae, An, lon0, lat0)
    cdlat, cdlon, cdeast, cdnorth = d.geo2mag(90 - colat, lon, Ae=Ae, An=An)
    assert np.all(np.isclose(cdlat - newlat, 0))
    assert np.all(np.isclose(cdlon - newlon, 0))
    assert np.all(np.isclose(neweast - cdeast, 0))
    assert np.all(np.isclose(newnorth - cdnorth, 0))

    # Check that converting back works:
    lat2, lon2, Ae2, An2 = geo2local(
        newlat, newlon, neweast, newnorth, lon0, lat0, inverse=True
    )
    assert np.all(np.isclose(lon - lon2, 0))
    assert np.all(np.isclose(90 - colat - lat2, 0))
    assert np.all(np.isclose(Ae - Ae2, 0))
    assert np.all(np.isclose(An - An2, 0))
