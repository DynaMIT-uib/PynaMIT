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

    def __init__(self, Nmax, Mmax, Nmin = 1):
        self.Nmax = Nmax
        self.Mmax = Mmax

        # Make separate sets of spherical harmonic keys for cos and sin terms
        all_cnm = SHKeys(self.Nmax, self.Mmax)
        self.all_cnm_list = list(all_cnm)
        self.cnm = all_cnm.setNmin(Nmin).MleN()
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
        Pc, dPc = self.legendre(self.Nmax, self.Mmax, grid.theta, keys = self.cnm)
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


    def legendre(self, Nmax, Mmax, theta, schmidt_normalization = False, keys = None):
        """
        Calculate associated Legendre function ``P`` and its derivative.

        Optional Schmidt semi-normalization.

        Algorithm from "Spacecraft Attitude Determination and Control" by
        James Richard Wertz.

        Parameters
        ----------
        nmax : int
            Highest spherical harmonic degree.
        mmax : int
            Highest spherical harmonic order.
        theta : array, float
            Colatitude in degrees (shape is not preserved).
        schmidt_normalization : bool, optional, default = False
            ``True`` if Schmidt semi-normalization is wanted, ``False``
            otherwise.
        keys : SHKeys, optional
            If this parameter is set, an array will be returned instead of
            a dict. The array will be ``(N, 2M)``, where ``N`` is the
            number of elements in `theta`, and ``M`` is the number of
            keys. The first ``M`` columns represents a matrix of ``P``
            values, and the last ``M`` columns represent values of
            ``dP/dtheta``.

        Returns
        -------
        P : dict
            If ``keys is None``, the dictionary of Legendre function
            evalulated at `theta` is returned. Dictionary keys are
            spherical harmonic wave number tuples ``(n, m)``, and values
            will have shape ``(N, 1)``, where ``N`` is the number of
            elements in `theta`. 
        dP : dict
            If ``keys is None``, the dictionary of Legendre function
            derivatives evaluated at `theta` is returned. Dictionary keys
            are spherical harmonic wave number tuples ``(n, m)``, and
            values will have shape ``(N, 1)``, where ``N`` is number of
            elements in theta.
        PdP : array
            If ``keys is not None``, ``PdP`` is returned instead of `P`
            and `dP`. `PdP` is an ``(N, 2M)`` array, where ``N`` is the
            number of elements in `theta`, and ``M`` is the number of
            keys. The first ``M`` columns represent a matrix of ``P``
            values, and the last ``M`` columns represent values of
            ``dP/dtheta``.

        """

        sinth = np.sin(np.deg2rad(theta))
        costh = np.cos(np.deg2rad(theta))

        # Calculate the Legendre functions
        P  = np.empty((theta.size, len(self.all_cnm_list)), dtype = np.float64)
        P[:, 0]  = np.ones_like(theta, dtype = np.float64)
        for nm in range(1, len(self.all_cnm_list)):
            n, m = self.all_cnm_list[nm]
            if n == m:
                P[:, nm]  = sinth * P[:, self.all_cnm_list.index((n - 1, m - 1))]
            else:
                if n > m:
                    P[:, nm]  = costh * P[:, self.all_cnm_list.index((n - 1, m))]

                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, self.all_cnm_list.index((n - 2, m))]

        if schmidt_normalization:
            schmidt_factors = schmidt_normalization_factors(self.all_cnm_list)
            P *= schmidt_factors

        # Calculate the derivatives of the Legendre functions
        dP = np.empty((theta.size, len(self.all_cnm_list)), dtype = np.float64)
        dP[:, 0] = np.zeros_like(theta, dtype = np.float64)
        for nm in range(1, len(self.all_cnm_list)):
            n, m = self.all_cnm_list[nm]
            if n == m:
                dP[:, nm] = sinth * dP[:, self.all_cnm_list.index((n - 1, m - 1))] + costh * P[:, self.all_cnm_list.index((n - 1, m - 1))]

            else:
                if n > m:
                    dP[:, nm] = costh * dP[:, self.all_cnm_list.index((n - 1, m))] - sinth * P[:, self.all_cnm_list.index((n - 1, m))]

                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, self.all_cnm_list.index((n - 2, m))]

        if schmidt_normalization:
            schmidt_factors = schmidt_normalization_factors(self.all_cnm_list)
            dP *= schmidt_factors

        if keys is None:
            return P, dP
        else:
            filter = [(key in keys) for key in self.all_cnm_list]
            return P[:, filter], dP[:, filter]