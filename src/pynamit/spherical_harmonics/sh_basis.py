"""Spherical harmonic basis.

This module contains the SHBasis class for representing spherical
harmonic bases.
"""

import numpy as np
from pynamit.spherical_harmonics.helpers import SHIndices, schmidt_normalization_factors


class SHBasis(object):
    """Class for representing spherical harmonic bases.

    A class to store information about a spherical harmonic basis and to
    generate matrices for evaluating the spherical harmonics at a given
    grid.

    Attributes
    ----------
    cnm : SHIndices
        Indices for cosine terms.
    snm : SHIndices
        Indices for sine terms.
    n : ndarray
        Array of degree values.
    m : ndarray
        Array of order values.
    schmidt_factors : ndarray
        Schmidt normalization factors if enabled.
    short_name : str
        Identifier for the basis ('SH').
    index_names : list
        Names of the indices ['n', 'm'].
    index_length : int
        Total number of basis functions.
    minimum_phi_sampling : int
        Minimum number of longitude points needed.
    caching : bool
        Whether caching is enabled.
    """

    def __init__(self, Nmax, Mmax, Nmin=1, schmidt_normalization=True):
        """Initialize the SHBasis instance.

        Parameters
        ----------
        Nmax : int
            Maximum degree of the spherical harmonics.
        Mmax : int
            Maximum order of the spherical harmonics.
        Nmin : int, optional
            Minimum degree of the spherical harmonics.
        schmidt_normalization : bool, optional
            Whether to use Schmidt semi-normalization.
        """
        # Make a set of all spherical harmonic index pairs up to
        # (Nmax, Mmax).
        all_index_pairs = SHIndices(Nmax, Mmax)

        # Make separate sets of spherical harmonic indices for cos and
        # sin terms, and remove n < Nmin terms and m = 0 sin terms.
        self.cnm = SHIndices(Nmax, Mmax).set_Nmin(Nmin)
        self.snm = SHIndices(Nmax, Mmax).set_Nmin(Nmin).set_Mmin(1)

        self.cnm_filter = [(index_pair in self.cnm) for index_pair in all_index_pairs]
        self.snm_filter = [(index_pair in self.snm) for index_pair in all_index_pairs]

        self.index_pairs = list(all_index_pairs)
        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))

        # Make the spherical harmonic Schmidt normalization factors.
        self.schmidt_normalization = schmidt_normalization
        if self.schmidt_normalization:
            self.schmidt_factors = schmidt_normalization_factors(self.index_pairs)

        # Set the general properties of the basis.
        self.short_name = "SH"

        self.index_names = ["n", "m"]
        self.index_length = len(self.cnm.index_pairs) + len(self.snm.index_pairs)
        self.index_arrays = [self.n, self.m]

        self.minimum_phi_sampling = 2 * Mmax + 1

        self.caching = True

    def get_G(self, grid, derivative=None, cache_in=None, cache_out=False):
        """Calculate spherical harmonic evaluation matrix.

        Calculates matrix that evaluates surface spherical harmonics at
        unit radius and the latitudes and longitudes of the given grid.

        Parameters
        ----------
        grid : Grid
            Grid object with latitudes and longitudes for evaluation.
        derivative : {None, 'phi', 'theta'}, optional
            Type of derivative to compute:
            - None: Evaluate spherical harmonics.
            - 'phi': Compute eastward derivative.
            - 'theta': Compute southward derivative.
        cache_in : ndarray, optional
            Cached Legendre functions.
        cache_out : bool, optional
            Whether to return cached Legendre functions.

        Returns
        -------
        ndarray
            Evaluation matrix of shape (N, M) where:
            - N is the size from broadcasting grid.lon and grid.lat.
            - M is the number of spherical harmonic terms.
            Cosine terms come first, followed by sine terms.
        ndarray, optional
            Cached Legendre functions if `cache_out` is ``True``.

        Raises
        ------
        Exception
            If derivative is not one of {None, 'phi', 'theta'}.
        """
        # Convert the grid coordinates to radians.
        phi = np.deg2rad(grid.phi)
        theta = np.deg2rad(grid.theta)

        # Get the Legendre functions and their derivatives.
        if cache_in is not None:
            P_unnormalized = cache_in
        else:
            P_unnormalized = self.legendre(theta)

        if derivative == "theta":
            dP_unnormalized = self.legendre_derivative(theta, P=P_unnormalized)

        if self.schmidt_normalization:
            P = P_unnormalized * self.schmidt_factors
            if derivative == "theta":
                dP = dP_unnormalized * self.schmidt_factors

        if derivative is None:
            Gc = P[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = P[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        elif derivative == "phi":
            Gc = (
                -P[:, self.cnm_filter]
                * self.cnm.m
                * np.sin(phi.reshape((-1, 1)) * self.cnm.m)
                / np.sin(theta.reshape((-1, 1)))
            )
            Gs = (
                P[:, self.snm_filter]
                * self.snm.m
                * np.cos(phi.reshape((-1, 1)) * self.snm.m)
                / np.sin(theta.reshape((-1, 1)))
            )
        elif derivative == "theta":
            Gc = dP[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = dP[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        else:
            raise Exception(
                f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.'
            )

        if cache_out:
            return np.hstack((Gc, Gs)), P_unnormalized
        else:
            return np.hstack((Gc, Gs))

    def legendre(self, theta):
        """Calculate associated Legendre functions.

        Uses algorithm from "Spacecraft Attitude Determination and
        Control" by James Richard Wertz.

        Parameters
        ----------
        theta : array-like
            Colatitude in radians.

        Returns
        -------
        ndarray
            Legendre functions evaluated at theta.
            Shape is (n_theta, n_sh) where:
            - n_theta: Number of colatitude points.
            - n_sh: Number of spherical harmonics.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Calculate the Legendre functions.
        P = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        P[:, 0] = np.ones_like(theta, dtype=np.float64)
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                P[:, nm] = sin_theta * P[:, self.index_pairs.index((n - 1, m - 1))]
            else:
                if n > m:
                    P[:, nm] = cos_theta * P[:, self.index_pairs.index((n - 1, m))]
                if n > m + 1:
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, self.index_pairs.index((n - 2, m))]

        return P

    def legendre_derivative(self, theta, P=None):
        """Calculate derivatives of associated Legendre functions.

        Computes d/dθ of the associated Legendre functions using
        algorithm from "Spacecraft Attitude Determination and Control"
        by James Richard Wertz.

        Parameters
        ----------
        theta : array-like
            Colatitude in radians.
        P : ndarray, optional
            Pre-computed Legendre functions.

        Returns
        -------
        ndarray
            Derivatives of Legendre functions evaluated at theta.
            Shape is (n_theta, n_sh) where:
            - n_theta: Number of colatitude points.
            - n_sh: Number of spherical harmonics.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if P is None:
            P = self.legendre(theta)

        # Calculate the derivatives of the Legendre functions.
        dP = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP[:, 0] = np.zeros_like(theta, dtype=np.float64)
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                dP[:, nm] = (
                    sin_theta * dP[:, self.index_pairs.index((n - 1, m - 1))]
                    + cos_theta * P[:, self.index_pairs.index((n - 1, m - 1))]
                )
            else:
                if n > m:
                    dP[:, nm] = (
                        cos_theta * dP[:, self.index_pairs.index((n - 1, m))]
                        - sin_theta * P[:, self.index_pairs.index((n - 1, m))]
                    )
                if n > m + 1:
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, self.index_pairs.index((n - 2, m))]

        return dP

    def laplacian(self, r=1.0):
        """Calculate angular Laplacian of spherical harmonics.

        Parameters
        ----------
        r : float, optional
            Radius.

        Returns
        -------
        ndarray
            Angular Laplacian values for each harmonic term.
        """
        return -self.n * (self.n + 1) / r**2

    def d_dr_V_external(self, r=1.0):
        """Calculate radial derivative of external potential.

        Calculates the vector that represents the radial derivative of
        the spherical harmonic coefficients for an external potential.

        Parameters
        ----------
        r : float, optional
            Radius.

        Returns
        -------
        array
            The radial derivative of the spherical harmonic coefficients
            for an external potential.
        """
        return self.n / r

    def d_dr_V_internal(self, r=1.0):
        """Calculate radial derivative of internal potential.

        Calculates the vector that represents the radial derivative of
        the spherical harmonic coefficients for an internal potential.

        Parameters
        ----------
        r : float, optional
            Radius.

        Returns
        -------
        array
            The radial derivative of the spherical harmonic coefficients
            for an internal potential.
        """
        return -(self.n + 1) / r

    def radial_shift_V_external(self, start, end):
        """Calculate radial shift of external potential.

        Calculates the vector that represents a shift of the reference
        radius for the spherical harmonics for an external potential,
        from `start` to `end`. Corresponds to the spherical harmonic
        functions with `end` as the reference radius divided by the
        spherical harmonic functions with `start` as the reference
        radius.

        Parameters
        ----------
        start : float
            Starting radius.
        end : float
            Ending radius.

        Returns
        -------
        array
            The vector representing the shift of the reference radius.
        """
        return (end / start) ** (self.n - 1)

    @property
    def V_external_to_delta_V(self):
        """
        Convert external coefficients to internal-external difference.

        Calculates multiplicative factors to convert spherical harmonic
        coefficients for an external potential to coefficients for
        internal-external potential difference, assuming continuous
        first-order radial derivative across surface.

        Returns
        -------
        ndarray
            Multiplicative conversion factors for each harmonic term.
        """
        if not hasattr(self, "_V_external_to_delta_V"):
            self._V_external_to_delta_V = 2 * self.n + 1

        return self._V_external_to_delta_V
