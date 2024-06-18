"""
Spherical harmonic basis.

This module contains the ``SHBasis`` class.

"""

import numpy as np
from pynamit.sha.helpers import SHKeys, legendre, schmidt_normalization_factors
from scipy.special import lpmv

class SHBasis(object):
    """ Spherical harmonic basis.

    Class to store information about a spherical harmonic basis and to
    generate matrices for evaluating the spherical harmonics at a given
    grid.

    """

    def __init__(self, Nmax, Mmax, Nmin = 1):
        self.Nmax = Nmax
        self.Mmax = Mmax

        # Make separate sets of spherical harmonic keys for cos and sin terms
        self.cnm = SHKeys(self.Nmax, self.Mmax).setNmin(Nmin).MleN()
        self.snm = SHKeys(self.Nmax, self.Mmax).setNmin(Nmin).MleN().Mge(1)

        # Make the Schmidt normalization factors for all terms
        self.schmidt_normalization_factors = schmidt_normalization_factors(list(self.cnm.keys))

        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))

        # Number of spherical harmonic coefficients
        self.num_coeffs = len(self.cnm.keys) + len(self.snm.keys)


    def get_G(self, grid, derivative = None):
        """
        Calculate matrix that evaluates surface spherical harmonics at
        unit radius and the latitudes and longitudes of the given `grid`,
        using the terms contained in the ``SHKeys`` of the ``SHBasis`` object.

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

        # Get the Legendre functions and their derivatives
        Pc, dPc = legendre(self.Nmax, self.Mmax, grid.theta, keys = self.cnm)
        Pc  *= self.schmidt_normalization_factors
        dPc *= self.schmidt_normalization_factors

        Ps  =  Pc[: , self.cnm.m.flatten() != 0]
        dPs = dPc[: , self.cnm.m.flatten() != 0]

        phi_rad = np.deg2rad(grid.lon).reshape((-1, 1))
        if derivative is None:
            Gc = Pc * np.cos(phi_rad * self.cnm.m)
            Gs = Ps * np.sin(phi_rad * self.snm.m)
        elif derivative == 'phi':
            theta_rad = np.deg2rad(grid.theta).reshape((-1, 1))
            Gc = -Pc * self.cnm.m * np.sin(phi_rad * self.cnm.m) / np.sin(theta_rad)
            Gs =  Ps * self.snm.m * np.cos(phi_rad * self.snm.m) / np.sin(theta_rad)
        elif derivative == 'theta':
            Gc = dPc * np.cos(phi_rad * self.cnm.m)
            Gs = dPs * np.sin(phi_rad * self.snm.m)
        else:
            raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

        return np.hstack((Gc, Gs))