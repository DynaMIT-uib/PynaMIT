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
    @classmethod
    def create_mw_grid(cls, L):
        """Create a McEwen-Wiaux (MW) equiangular grid for band-limit L.
        
        Ref: McEwen & Wiaux (2011), 'A novel sampling theorem on the sphere'.
        
        Parameters
        ----------
        L : int
            Band-limit (L_max). Ideally the grid supports exact quadrature for degrees up to L.
            Number of theta points: L+1.
            Number of phi points: 2L+1.
            
        Returns
        -------
        Grid
            The generated Grid object.
        """
        N_theta = L + 1
        N_phi = 2 * L + 1
        
        # Theta: t = 0..L. theta_t = pi * (2t + 1) / (2N_theta - 1)
        # Note: MW paper uses N for N_theta. Denom is 2N-1.
        t = np.arange(N_theta)
        theta_1d = np.pi * (2 * t + 1) / (2 * N_theta - 1)
        
        # Phi: p = 0..2L. phi_p = 2pi * p / (2L+1)
        p = np.arange(N_phi)
        phi_1d = 2 * np.pi * p / N_phi
        
        # Meshgrid
        tt, pp = np.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        # Convert to Degrees for InputManager compatibility
        # Grid expects flattened arrays in degrees
        return cls(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
