"""
Spherical harmonic basis.

This module contains the ``SHBasis`` class.

"""

import numpy as np
from pynamit.sha.helpers import SHKeys, schmidt_normalization_factors

class SHBasis(object):
    """ Spherical harmonic basis.

    Class to store information about a spherical harmonic basis and to
    generate matrices for evaluating the spherical harmonics at a given
    grid.

    """

    def __init__(self, Nmax, Mmax, Nmin = 1, schmidt_normalization = True):
        self.Nmax = Nmax
        self.Mmax = Mmax

        # Make separate sets of spherical harmonic keys for cos and sin terms
        all_keys = SHKeys(self.Nmax, self.Mmax)
        self.all_keys_list = list(all_keys)

        self.cnm = SHKeys(self.Nmax, self.Mmax).setNmin(Nmin).MleN()
        self.snm = SHKeys(self.Nmax, self.Mmax).setNmin(Nmin).MleN().Mge(1)

        self.cnm_filter = [(key in self.cnm) for key in self.all_keys_list]
        self.snm_filter = [(key in self.snm) for key in self.all_keys_list]

        # Make the Schmidt normalization factors for the spherical harmonics
        self.schmidt_normalization = schmidt_normalization
        if self.schmidt_normalization:
            self.schmidt_normalization_factors = schmidt_normalization_factors(self.all_keys_list)

        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))

        # Number of spherical harmonic coefficients
        self.num_coeffs = len(self.cnm.keys) + len(self.snm.keys)


    def get_G(self, grid, derivative = None):
        """
        Calculate matrix that evaluates surface spherical harmonics at
        unit radius and the latitudes and longitudes of the given `grid`,
        using the terms contained in the ``SHKeys`` of the ``SHBasis`` object.

        Optional Schmidt semi-normalization.

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
        P = self.legendre(grid.theta)
        if derivative == 'theta':
            dP = self.legendre_derivative(grid.theta, P = P)

        if self.schmidt_normalization:
            P *= self.schmidt_normalization_factors
            if derivative == 'theta':
                dP *= self.schmidt_normalization_factors

        phi_rad = np.deg2rad(grid.lon).reshape((-1, 1))
        if derivative is None:
            Gc = P[:, self.cnm_filter] * np.cos(phi_rad * self.cnm.m)
            Gs = P[:, self.snm_filter] * np.sin(phi_rad * self.snm.m)
        elif derivative == 'phi':
            theta_rad = np.deg2rad(grid.theta).reshape((-1, 1))
            Gc = -P[:, self.cnm_filter] * self.cnm.m * np.sin(phi_rad * self.cnm.m) / np.sin(theta_rad)
            Gs =  P[:, self.snm_filter] * self.snm.m * np.cos(phi_rad * self.snm.m) / np.sin(theta_rad)
        elif derivative == 'theta':
            Gc = dP[:, self.cnm_filter] * np.cos(phi_rad * self.cnm.m)
            Gs = dP[:, self.snm_filter] * np.sin(phi_rad * self.snm.m)
        else:
            raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

        return np.hstack((Gc, Gs))

    def legendre(self, theta):
        """
        Calculate associated Legendre function ``P`` and its derivative.

        Algorithm from "Spacecraft Attitude Determination and Control" by
        James Richard Wertz.

        Parameters
        ----------
        theta : array, float
            Colatitude in degrees (shape is not preserved).
        
        Returns
        -------
        P : array, float
            Legendre functions for all terms in the ``SHKeys`` of the
            ``SHBasis`` object, evaluated at `theta`. Shape is
            ``(n_theta, n_sh)``, where ``n_theta`` is the number of
            elements in `theta` and ``n_sh`` is the number of spherical
            harmonics.

        """

        sinth = np.sin(np.deg2rad(theta))
        costh = np.cos(np.deg2rad(theta))

        # Calculate the Legendre functions
        P  = np.empty((theta.size, len(self.all_keys_list)), dtype = np.float64)
        P[:, 0]  = np.ones_like(theta, dtype = np.float64)
        for nm in range(1, len(self.all_keys_list)):
            n, m = self.all_keys_list[nm]
            if n == m:
                P[:, nm]  = sinth * P[:, self.all_keys_list.index((n - 1, m - 1))]
            else:
                if n > m:
                    P[:, nm]  = costh * P[:, self.all_keys_list.index((n - 1, m))]
                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, self.all_keys_list.index((n - 2, m))]

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
            Colatitude in degrees (shape is not preserved).

        Returns
        -------
        dP : array, float
            Derivatives of the Legendre functions for all terms in the
            ``SHKeys`` of the ``SHBasis`` object, evaluated at `theta`.
            Shape is ``(n_theta, n_sh)``, where ``n_theta`` is the number
            of elements in `theta` and ``n_sh`` is the number of spherical
            harmonics.

        """

        sinth = np.sin(np.deg2rad(theta))
        costh = np.cos(np.deg2rad(theta))

        if P is None:
            P = self.legendre(theta)

        # Calculate the derivatives of the Legendre functions
        dP = np.empty((theta.size, len(self.all_keys_list)), dtype = np.float64)
        dP[:, 0] = np.zeros_like(theta, dtype = np.float64)
        for nm in range(1, len(self.all_keys_list)):
            n, m = self.all_keys_list[nm]
            if n == m:
                dP[:, nm] = sinth * dP[:, self.all_keys_list.index((n - 1, m - 1))] + costh * P[:, self.all_keys_list.index((n - 1, m - 1))]
            else:
                if n > m:
                    dP[:, nm] = costh * dP[:, self.all_keys_list.index((n - 1, m))] - sinth * P[:, self.all_keys_list.index((n - 1, m))]
                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, self.all_keys_list.index((n - 2, m))]

        return dP