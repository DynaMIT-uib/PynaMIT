"""
Tools that are useful for spherical harmonic analysis.

Functions in this module:

- ``nterms``: function which calculates the number of terms in a real
  expansion of a poloidal (internal + external) and toroidal expansion.
- ``legendre``: calculate associated legendre functions - with option for
  Schmidt semi-normalization.
- ``get_G``: calculate matrix for evaluating surface spherical harmonics
  at given grid.

"""

import numpy as np
from pynamit.sha.helpers import SHkeys, legendre

class SHBasis(object):
    """ Class for spherical harmonic analysis.

    """

    def __init__(self, Nmax, Mmax):
        self.Nmax = Nmax
        self.Mmax = Mmax

        # make separate sets of spherical harmonic keys for cos and sin terms:
        self.cnm = SHkeys(self.Nmax, self.Mmax).setNmin(1).MleN()
        self.snm = SHkeys(self.Nmax, self.Mmax).setNmin(1).MleN().Mge(1)

        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))

        # Number of spherical harmonic coefficients
        self.Nshc = len(self.cnm.keys) + len(self.snm.keys)


    def get_G(self, grid, derivative = None, return_nm = False):
        """
        Calculate matrix that evaluates surface spherical harmonics using
        the terms contained in ``shkeys``, and at the locations defined by
        `lat` and `lon`.

        Parameters
        ----------
        lat : array
            Latitude in degrees. Must be broadcastable with `lon`.
        lon : array
            Longitude in degrees. Must be broadcastable with `lat`.
        N: int
            Maximum spherical harmonic degree.
        M: int
            Maximum spherical harmonic order.
        a : float, optional, default = 6371.2
            Reference radius.
        derivative : string, {None, 'phi', 'theta'}, default = None
            Set to 'phi' to get the matrix that gives the eastward
            gradient.
            Set to 'theta' to get the matrix that gives the southward
            gradient. Default gives surface SH (no derivative).

        Returns
        -------
        G : array
            ``N x M`` array, where ``N`` is the size inferred by
            broadcasting `lon` and `lat`, and ``M`` is the number of terms
            in the spherical harmonics inferred from ``shkeys``. The
            ``cos`` terms are given first, and ``sin`` terms after.
        
        """

        try: # broadcast lat and lon, and turn results into column vectors:
            lat, lon = np.broadcast_arrays(grid.lat, grid.lon)
            lat, lon = lat.flatten().reshape((-1, 1)), lon.flatten().reshape((-1, 1))
        except ValueError:
            raise Exception('get_G: could not brodcast lat and lon')

        ph, th = np.deg2rad(lon), np.deg2rad(90 - lat) # lon and colat in radians

        # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
        PdP = legendre(self.Nmax, self.Mmax, 90 - lat.flatten(), keys = self.cnm)
        Pc, dPc = np.split(PdP, 2, axis = 1)
        Ps      =  Pc[: , self.cnm.m.flatten() != 0]
        dPs     = dPc[: , self.cnm.m.flatten() != 0]

        if derivative is None:
            Gc = grid.RI * Pc * np.cos(ph * self.cnm.m)
            Gs = grid.RI * Ps * np.sin(ph * self.snm.m)
        elif derivative == 'phi':
            Gc = -Pc * self.cnm.m * np.sin(ph * self.cnm.m) / np.sin(th)
            Gs =  Ps * self.snm.m * np.cos(ph * self.snm.m) / np.sin(th) 
        elif derivative == 'theta':
            Gc = dPc * np.cos(ph * self.cnm.m)
            Gs = dPs * np.sin(ph * self.snm.m)
        else:
            raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

        return np.hstack((Gc, Gs))

    def evaluate_on_grid(self, coeffs, grid):
        """
        Evaluate the spherical harmonic coefficients on the grid.

        Parameters
        ----------
        coeffs : array
            Spherical harmonic coefficients.
        grid : pynamit.grid.grid
            Grid object.

        Returns
        -------
        array
            The values of the spherical harmonic coefficients on the grid.

        """

        G = self.get_G(grid)
        return G.dot(coeffs)