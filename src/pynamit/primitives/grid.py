"""Grid coordinate system representation.

This module provides the Grid class for managing two-dimensional coordinate grids
in both geographic (lat/lon) and spherical (theta/phi) coordinate systems.

Classes
-------
Grid
    Two-dimensional coordinate grid with automatic coordinate conversion.
"""

import numpy as np

class Grid(object):
    """Two-dimensional coordinate grid representation.
    
    Manages coordinates in both geographic (latitude/longitude) and spherical 
    (colatitude/longitude) coordinate systems with automatic conversion between them.

    Parameters
    ----------
    lat : array-like, optional
        Geographic latitude coordinates in degrees
    lon : array-like, optional
        Geographic longitude coordinates in degrees
    theta : array-like, optional
        Spherical colatitude coordinates in degrees (0° at North pole)
    phi : array-like, optional
        Spherical longitude coordinates in degrees
        
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
    Must provide either lat or theta, and either lon or phi coordinates.
    All coordinate arrays are automatically broadcast to match shapes and
    flattened for internal storage.
    
    Raises
    ------
    ValueError
        If neither lat/theta or lon/phi coordinates are provided
    """

    def __init__(self, lat=None, lon=None, theta=None, phi=None):
        """
        Initialize the object for storing the two-dimensional grid. The
        grid is initialized from `lat` or `theta` and `lon` or `phi`
        coordinates.
        """
    
        if lat is not None:
            self.lat = lat
            self.theta = 90 - self.lat
        elif theta is not None:
            self.theta = theta
            self.lat = 90 - self.theta
        else:
            raise ValueError("Latitude or theta must be provided to initialize the grid.")

        if lon is not None:
            self.lon = lon
            self.phi = lon
        elif phi is not None:
            self.phi = phi
            self.lon = phi
        else:
            raise ValueError("Longitude or phi must be provided to initialize the grid.")

        self.lat, self.lon = np.broadcast_arrays(self.lat, self.lon)
        self.theta, self.phi = np.broadcast_arrays(self.theta, self.phi)

        self.lat = self.lat.flatten()
        self.lon = self.lon.flatten()
        self.theta = self.theta.flatten()
        self.phi = self.phi.flatten()

        self.size = self.lon.size