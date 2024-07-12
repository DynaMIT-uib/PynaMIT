"""
Spherical harmonic basis.

This module contains the ``SHBasis`` class.

"""

import numpy as np
import pandas as pd
from pynamit.spherical_harmonics.helpers import SHKeys, schmidt_normalization_factors

class SHBasis(object):
    """ Spherical harmonic basis.

    Class to store information about a spherical harmonic basis and to
    generate matrices for evaluating the spherical harmonics at a given
    grid.

    """

    def __init__(self, Nmax, Mmax, Nmin = 1, schmidt_normalization = True):
        # Make a set of all spherical harmonic keys up to Nmax, Mmax
        all_keys = SHKeys(Nmax, Mmax)

        # Make separate sets of spherical harmonic keys for cos and sin
        # terms, and remove n < Nmin terms and m = 0 sin terms
        self.cnm = SHKeys(Nmax, Mmax).set_Nmin(Nmin)
        self.snm = SHKeys(Nmax, Mmax).set_Nmin(Nmin).set_Mmin(1)

        self.cnm_filter = [(key in self.cnm) for key in all_keys]
        self.snm_filter = [(key in self.snm) for key in all_keys]

        self.nm_tuples = list(all_keys)
        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))

        # Make the Schmidt normalization factors for the spherical harmonics
        self.schmidt_normalization = schmidt_normalization
        if self.schmidt_normalization:
            self.schmidt_factors = schmidt_normalization_factors(self.nm_tuples)

        # Set the general properties of the basis
        self.short_name = 'SH'

        self.index_names = ['n', 'm']
        self.index_length = len(self.cnm.keys) + len(self.snm.keys)
        self.indices = [self.n, self.m]

        self.minimum_phi_sampling = 2 * Mmax + 1

        self.caching = True


    def get_G(self, grid, derivative = None, cache_in = None, cache_out = False):
        """
        Calculate matrix that evaluates surface spherical harmonics at
        unit radius and the latitudes and longitudes of the given `grid`,
        using the terms contained in the ``SHKeys`` of the ``SHBasis``
        object.

        Optional Schmidt semi-normalization.

        Parameters
        ----------
        grid: grid object
            Grid object containing the latitudes and longitudes where the
            spherical harmonics are to be evaluated.
        derivative : string, {None, 'phi', 'theta'}, default = None
            Default gives the matrix that evaluates the spherical
            harmonics.  Set to 'phi' to get the matrix that gives the
            eastward derivative. Set to 'theta' to get the matrix that
            gives the southward derivative.

        Returns
        -------
        G : array
            ``N x M`` array, where ``N`` is the size inferred by
            broadcasting `grid.lon` and `grid.lat`, and ``M`` is the
            number of terms in the spherical harmonics inferred from
            the ``SHKeys`` of the ``SHBasis`` object. The ``cos`` terms
            are given first, and ``sin`` terms after.
        
        """

        # Convert the grid coordinates to radians
        phi = np.deg2rad(grid.phi)
        theta = np.deg2rad(grid.theta)

        # Get the Legendre functions and their derivatives
        if cache_in is not None:
            P_unnormalized = cache_in
        else:
            P_unnormalized = self.legendre(theta)

        if derivative == 'theta':
            dP_unnormalized = self.legendre_derivative(theta, P = P_unnormalized)

        if self.schmidt_normalization:
            P = P_unnormalized * self.schmidt_factors
            if derivative == 'theta':
                dP = dP_unnormalized * self.schmidt_factors

        if derivative is None:
            Gc = P[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = P[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        elif derivative == 'phi':
            Gc = -P[:, self.cnm_filter] * self.cnm.m * np.sin(phi.reshape((-1, 1)) * self.cnm.m) / np.sin(theta.reshape((-1, 1)))
            Gs =  P[:, self.snm_filter] * self.snm.m * np.cos(phi.reshape((-1, 1)) * self.snm.m) / np.sin(theta.reshape((-1, 1)))
        elif derivative == 'theta':
            Gc = dP[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = dP[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        else:
            raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

        if cache_out:
            return np.hstack((Gc, Gs)), P_unnormalized
        else:
            return np.hstack((Gc, Gs))


    def legendre(self, theta):
        """
        Calculate associated Legendre function ``P`` and its derivative.

        Algorithm from "Spacecraft Attitude Determination and Control" by
        James Richard Wertz.

        Parameters
        ----------
        theta : array, float
            Colatitude in radians.
        
        Returns
        -------
        P : array, float
            Legendre functions for all terms in the ``SHKeys`` of the
            ``SHBasis`` object, evaluated at `theta`. Shape is
            ``(n_theta, n_sh)``, where ``n_theta`` is the number of
            elements in `theta` and ``n_sh`` is the number of spherical
            harmonics.

        """

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Calculate the Legendre functions
        P  = np.empty((theta.size, len(self.nm_tuples)), dtype = np.float64)
        P[:, 0]  = np.ones_like(theta, dtype = np.float64)
        for nm in range(1, len(self.nm_tuples)):
            n, m = self.nm_tuples[nm]
            if n == m:
                P[:, nm]  = sin_theta * P[:, self.nm_tuples.index((n - 1, m - 1))]
            else:
                if n > m:
                    P[:, nm]  = cos_theta * P[:, self.nm_tuples.index((n - 1, m))]
                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, self.nm_tuples.index((n - 2, m))]

        return P


    def legendre_derivative(self, theta, P = None):
        """
        Calculate the derivative of the associated Legendre function ``P``
        with respect to colatitude.

        Algorithm from "Spacecraft Attitude Determination and Control" by
        James Richard Wertz.

        Parameters
        ----------
        theta : array, float
            Colatitude in radians.

        Returns
        -------
        dP : array, float
            Derivatives of the Legendre functions for all terms in the
            ``SHKeys`` of the ``SHBasis`` object, evaluated at `theta`.
            Shape is ``(n_theta, n_sh)``, where ``n_theta`` is the number
            of elements in `theta` and ``n_sh`` is the number of spherical
            harmonics.

        """

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if P is None:
            P = self.legendre(theta)

        # Calculate the derivatives of the Legendre functions
        dP = np.empty((theta.size, len(self.nm_tuples)), dtype = np.float64)
        dP[:, 0] = np.zeros_like(theta, dtype = np.float64)
        for nm in range(1, len(self.nm_tuples)):
            n, m = self.nm_tuples[nm]
            if n == m:
                dP[:, nm] = sin_theta * dP[:, self.nm_tuples.index((n - 1, m - 1))] + cos_theta * P[:, self.nm_tuples.index((n - 1, m - 1))]
            else:
                if n > m:
                    dP[:, nm] = cos_theta * dP[:, self.nm_tuples.index((n - 1, m))] - sin_theta * P[:, self.nm_tuples.index((n - 1, m))]
                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, self.nm_tuples.index((n - 2, m))]

        return dP


    def d_dr(self, r = 1.0):
        """
        Calculate the vector that represents the radial derivative of the
        spherical harmonics.

        """

        return -self.n / r


    def laplacian(self, r = 1.0):
        """
        Calculate the vector that represents the Laplacian of the
        spherical harmonics

        """

        return -self.n * (self.n + 1) / r**2


    @property
    def surface_discontinuity(self):
        """
        Calculate the discontinuity across the surface where the spherical
        harmonics are evaluated, assuming that the first-order radial
        derivative is continuous across the surface.

        """

        if not hasattr(self, '_surface_discontinuity'):
            self._surface_discontinuity = (2 * self.n + 1) / (self.n + 1)

        return self._surface_discontinuity


    def radial_shift(self, start, end):
        """
        Calculate the vector that represents a radial shift of the
        spherical harmonics.

        """

        return (end / start)**(self.n - 1)