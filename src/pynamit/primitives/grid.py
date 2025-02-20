"""Spatial Grid Utilities.

This module provides the Grid class for representing spatial grids used
in PynaMIT. It supports generating grid representations for the
evaluation of magnetic fields and simulation data in PynaMIT.
"""

import numpy as np


class Grid(object):
    """Two-dimensional coordinate grid representation.

    Attributes
    ----------
    lat : ndarray
        Flattened array of latitude values in degrees
    lon : ndarray
        Flattened array of longitude values in degrees
    theta : ndarray
        Flattened array of colatitude values in degrees
    phi : ndarray
        Flattened array of longitude values in degrees (same as lon)
    size : int
        Total number of grid points

    Notes
    -----
    All coordinate arrays are automatically broadcast to match shapes
    and flattened for internal storage.
    """

    def __init__(self, lat=None, lon=None, theta=None, phi=None):
        """Initialize the grid object from coordinate inputs.

        Parameters
        ----------
        lat : array-like, optional
            Geographic latitude coordinates in degrees.
        lon : array-like, optional
            Geographic longitude coordinates in degrees.
        theta : array-like, optional
            Spherical colatitude coordinates in degrees.
        phi : array-like, optional
            Spherical longitude coordinates in degrees.

        Raises
        ------
        ValueError
            If neither `lat`/`theta` or `lon`/`phi` coordinates are
            provided.

        Notes
        -----
        Either `lat` or `theta` must be provided, and either `lon` or
        `phi` must be provided.
        """
        if lat is not None:
            self.lat = lat
            self.theta = 90 - self.lat
        elif theta is not None:
            self.theta = theta
            self.lat = 90 - self.theta
        else:
            raise ValueError(
                "Latitude or theta must be provided to initialize the grid."
            )

        if lon is not None:
            self.lon = lon
            self.phi = lon
        elif phi is not None:
            self.phi = phi
            self.lon = phi
        else:
            raise ValueError(
                "Longitude or phi must be provided to initialize the grid."
            )

        self.lat, self.lon = np.broadcast_arrays(self.lat, self.lon)
        self.theta, self.phi = np.broadcast_arrays(self.theta, self.phi)

        self.lat = self.lat.flatten()
        self.lon = self.lon.flatten()
        self.theta = self.theta.flatten()
        self.phi = self.phi.flatten()

        self.size = self.lon.size
