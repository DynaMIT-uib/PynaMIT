"""
Spherical harmonic basis.

This module contains the ``SHBasis`` class.

"""

import numpy as np
from pynamit.sha.helpers import SHKeys, legendre

class SHBasis(object):
    """ Spherical harmonic basis.

    Class to store information about a spherical harmonic basis and to
    generate matrices for evaluating the spherical harmonics at a given
    grid.

    """

    def __init__(self, Nmax, Mmax):
        self.Nmax = Nmax
        self.Mmax = Mmax

        # make separate sets of spherical harmonic keys for cos and sin terms:
        self.cnm = SHKeys(self.Nmax, self.Mmax).setNmin(1).MleN()
        self.snm = SHKeys(self.Nmax, self.Mmax).setNmin(1).MleN().Mge(1)

        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))

        # Number of spherical harmonic coefficients
        self.num_coeffs = len(self.cnm.keys) + len(self.snm.keys)


    def get_G(self, grid, derivative = None):
        """
        Calculate matrix that evaluates surface spherical harmonics at the
        latitudes and longitudes of the given `grid`, using the terms
        contained in the ``SHKeys`` of the ``SHBasis`` object.

        Parameters
        ----------
        grid: grid object
            Grid object containing the latitudes and longitudes where the
            spherical harmonics are to be evaluated.
        derivative : string, {None, 'phi', 'theta'}, default = None
            Set to 'phi' to get the matrix that gives the eastward
            gradient.
            Set to 'theta' to get the matrix that gives the southward
            gradient. Default gives surface SH (no derivative).

        Returns
        -------
        G : array
            ``N x M`` array, where ``N`` is the size inferred by
            broadcasting `grid.lon` and `grid.lat`, and ``M`` is the
            number of terms in the spherical harmonics inferred from
            the ``SHKeys`` of the ``SHBasis`` object. The ``cos`` terms
            are given first, and ``sin`` terms after.
        
        """

        # convert grid angular coordinates to radians and convert to (N, 1) arrays:
        ph, th = np.deg2rad(grid.lon).reshape((-1, 1)), np.deg2rad(90 - grid.lat).reshape((-1, 1)) 
        r = grid.r.reshape((-1, 1))

        # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
        PdP = legendre(self.Nmax, self.Mmax, grid.theta, keys = self.cnm)
        Pc, dPc = np.split(PdP, 2, axis = 1)
        Ps      =  Pc[: , self.cnm.m.flatten() != 0]
        dPs     = dPc[: , self.cnm.m.flatten() != 0]

        if derivative is None:
            # Warning: the variable grid.r is included, but this only evaluates to the radius-dependent harmonics at r = RI
            Gc = Pc * np.cos(ph * self.cnm.m)
            Gs = Ps * np.sin(ph * self.snm.m)
        elif derivative == 'phi':
            Gc = -Pc * self.cnm.m * np.sin(ph * self.cnm.m) / np.sin(th)
            Gs =  Ps * self.snm.m * np.cos(ph * self.snm.m) / np.sin(th)
        elif derivative == 'theta':
            Gc = dPc * np.cos(ph * self.cnm.m)
            Gs = dPs * np.sin(ph * self.snm.m)
        else:
            raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

        return np.hstack((Gc, Gs))