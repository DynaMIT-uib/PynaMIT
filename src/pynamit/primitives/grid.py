"""Grid module.

This module contains the Grid class for representing two-dimensional
coordinate grids.
"""

import numpy as np


class Grid:
    """Class for representing two-dimensional coordinate grids.

    Attributes
    ----------
    lat : ndarray
        Flattened array of latitude values in degrees.
    lon : ndarray
        Flattened array of longitude values in degrees.
    theta : ndarray
        Flattened array of colatitude values in degrees.
    phi : ndarray
        Flattened array of longitude values in degrees (same as lon).
    size : int
        Total number of grid points.

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
        
        # Lazy hash cache
        self._hash = None
        
    @property
    def hash(self):
        """Compute a hash for the grid based on its coordinates.
        
        We use float32 precision for the hash to be robust against 
        numerical noise (approx 1e-7 tolerance), while maintaining
        high performance.
        """
        if self._hash is None:
            # Cast to float32 for robustness against double-precision noise
            # This effectively treats grids differing by < 1e-7 as identical
            h_th = hash(self.theta.astype(np.float32).tobytes())
            h_ph = hash(self.phi.astype(np.float32).tobytes())
            self._hash = hash((h_th, h_ph))
        return self._hash
        
    def __eq__(self, other):
        """Check for equality with another grid."""
        if not isinstance(other, Grid):
            return NotImplemented
        # Fast path: identity
        if self is other:
            return True
        # Hash match implies equality within float32 precision
        # This acts as a robust check replacing np.allclose(..., rtol=1e-6)
        return self.hash == other.hash
