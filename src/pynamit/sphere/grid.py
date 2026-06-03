"""Grid module.

This module contains the Grid class for representing two-dimensional
coordinate grids.
"""

import hashlib

import numpy as np


class Grid(object):
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
    area_weights : ndarray, optional
        Flattened cell-area weights associated with the grid points.
    size : int
        Total number of grid points.

    Notes
    -----
    All coordinate arrays are automatically broadcast to match shapes
    and flattened for internal storage.
    """

    def __init__(self, lat=None, lon=None, theta=None, phi=None, area_weights=None):
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
        area_weights : array-like, optional
            Cell-area weights for weighted surface fits. If provided,
            the flattened shape must match the grid size.

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
        self._hash = None

        if area_weights is not None:
            self.area_weights = np.asarray(area_weights, dtype=float).flatten()
            if self.area_weights.shape != (self.size,):
                raise ValueError("area_weights must match the flattened grid size.")

    @staticmethod
    def _hash_coordinate(digest, values):
        """Hash one coordinate array at float32 precision."""
        array = np.ascontiguousarray(np.asarray(values, dtype="<f4").reshape(-1))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())

    @property
    def hash(self):
        """Deterministic hash for the flattened grid coordinates.

        Coordinates are quantized to float32 before hashing so grids
        that differ only by insignificant double-precision noise compare
        as equal.
        """
        if self._hash is None:
            digest = hashlib.blake2b(digest_size=16)
            self._hash_coordinate(digest, self.theta)
            self._hash_coordinate(digest, self.phi)
            self._hash = digest.hexdigest()
        return self._hash

    def same_as(self, other):
        """Return whether another grid has the same coordinates."""
        if self is other:
            return True
        if not isinstance(other, Grid):
            return False
        return self.hash == other.hash

    def __eq__(self, other):
        """Compare grids by their coordinate hashes."""
        if not isinstance(other, Grid):
            return NotImplemented
        return self.same_as(other)
