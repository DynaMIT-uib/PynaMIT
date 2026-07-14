"""Grid module.

This module contains the Grid class for representing two-dimensional
coordinate grids.
"""

import hashlib
from functools import cached_property

import numpy as np

from pynamit.sphere.core import SphericalRepresentation


class Grid(SphericalRepresentation):
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
        if (lat is None) == (theta is None):
            raise ValueError("Provide exactly one of latitude or theta.")
        if (lon is None) == (phi is None):
            raise ValueError("Provide exactly one of longitude or phi.")

        latitude = np.asarray(lat, dtype=float) if lat is not None else 90.0 - np.asarray(theta)
        longitude = (
            np.asarray(lon, dtype=float) if lon is not None else np.asarray(phi, dtype=float)
        )
        latitude, longitude = np.broadcast_arrays(latitude, longitude)

        self.lat = np.array(latitude, dtype=float, copy=True).reshape(-1)
        self.lon = np.array(longitude, dtype=float, copy=True).reshape(-1)
        if not np.all(np.isfinite(self.lat)) or not np.all(np.isfinite(self.lon)):
            raise ValueError("Grid coordinates must be finite.")
        if np.any(np.abs(self.lat) > 90.0):
            raise ValueError("Grid latitude must be between -90 and 90 degrees.")
        self.theta = 90.0 - self.lat
        self.phi = self.lon.copy()

        self.size = self.lon.size
        self._hash = None

        if area_weights is not None:
            self.area_weights = np.array(area_weights, dtype=float, copy=True).reshape(-1)
            if self.area_weights.shape != (self.size,):
                raise ValueError("area_weights must match the flattened grid size.")
            if not np.all(np.isfinite(self.area_weights)) or np.any(self.area_weights < 0.0):
                raise ValueError("area_weights must be finite and non-negative.")

        for array in (self.lat, self.lon, self.theta, self.phi):
            array.setflags(write=False)
        if hasattr(self, "area_weights"):
            self.area_weights.setflags(write=False)

        self.validate_metadata()

    @property
    def kind(self):
        """Short identifier for grid representations."""
        return "GRID"

    @property
    def index_names(self):
        """Names of grid-value indices."""
        return ("point",)

    @property
    def index_length(self):
        """Number of scalar grid values."""
        return self.size

    @property
    def index_arrays(self):
        """Point indices for scalar grid values."""
        return (np.arange(self.size),)

    @property
    def signature(self):
        """Return a stable signature for this grid."""
        return (type(self).__module__, type(self).__qualname__, self.hash)

    @property
    def coefficient_space_signature(self):
        """Return the grid-value compatibility signature."""
        return self.signature

    @cached_property
    def analysis_signature(self):
        """Return cache identity for coordinate-weighted analysis."""
        if not hasattr(self, "area_weights"):
            return (self.signature, None)
        weights = np.ascontiguousarray(self.area_weights, dtype="<f8")
        digest = hashlib.blake2b(weights.tobytes(), digest_size=16)
        return (self.signature, digest.hexdigest())

    @staticmethod
    def _hash_coordinate(digest, values):
        """Hash one coordinate array at float32 precision."""
        array = np.ascontiguousarray(np.asarray(values, dtype="<f4").reshape(-1))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())

    @classmethod
    def coordinate_hash(cls, theta, phi):
        """Return a hash for flattened spherical coordinates."""
        digest = hashlib.blake2b(digest_size=16)
        cls._hash_coordinate(digest, theta)
        cls._hash_coordinate(digest, phi)
        return digest.hexdigest()

    @property
    def hash(self):
        """Deterministic hash for the flattened grid coordinates.

        Coordinates are quantized to float32 before hashing so grids
        that differ only by insignificant double-precision noise compare
        as equal.
        """
        if self._hash is None:
            self._hash = self.coordinate_hash(self.theta, self.phi)
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
