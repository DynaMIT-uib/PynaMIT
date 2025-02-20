"""Cubed sphere equations module.

This module contains the CSEquations class which provides equations
related to cubed sphere coordinates.
"""

import numpy as np


class CSEquations(object):
    """Equations related to cubed sphere coordinates.

    This class provides equations for computing quantities on cubed
    sphere grids. It encapsulates operations such as metric tensor
    calculations, coordinate transformations, and differential operators
    used in cubed sphere coordinate computations.

    Attributes
    ----------
    cs_basis : CSBasis
        Cubed sphere basis object.
    RI : float
        Radius of the sphere.
    D : array
        Differential operator matrix, where the first index is the
        direction (0 for xi, 1 for eta).
    Ps : array
        Matrix to convert from (u^east, u^north, u^up) to
        (u^1, u^2, u^3), as defined in equation A1 in Yin et al. (2017).
    Qi : array
        Matrix to convert from physical (north, east, radial) to
        (u^east, u^north, u^up), as defined in equation A1 in Yin et al.
        (2017).
    """

    def __init__(self, cs_basis, RI):
        """Initialize the CSEquations object.

        Parameters
        ----------
        cs_basis : CSBasis
            Cubed sphere basis object.
        RI : float
            Radius of the sphere.
        """
        self.cs_basis = cs_basis
        self.RI = RI

    @property
    def D(self):
        """Differential operator matrix.

        Returns
        -------
        array
            Differential operator matrix, where the first index is the
            direction (0 for xi, 1 for eta).
        """
        if not hasattr(self, "_D"):
            self._D = self.cs_basis.get_Diff(
                self.cs_basis.Ncs, coordinate="both"
            )
        return self._D

    @property
    def Ps(self):
        """Ps matrix.

        Returns
        -------
        array
            Matrix to convert from (u^east, u^north, u^up) to
            (u^1, u^2, u^3), as defined in equation A1 in Yin et al.
            (2017).
        """
        if not hasattr(self, "_Ps"):
            self._Ps = self.cs_basis.get_Ps(
                self.cs_basis.arr_xi,
                self.cs_basis.arr_eta,
                1,
                self.cs_basis.arr_block,
            )
        return self._Ps

    @property
    def Qi(self):
        """Qi matrix.

        Returns
        -------
        array
            Matrix to convert from physical (north, east, radial) to
            (u^east, u^north, u^up), as defined in equation A1 in Yin et
            al. (2017).
        """
        if not hasattr(self, "_Qi"):
            self._Qi = self.cs_basis.get_Q(
                90 - self.cs_basis.arr_theta, self.RI, inverse=True
            )
        return self._Qi

    def curlr(self, u1, u2):
        """Calculate radial curl of vector field.

        Calculates the radial curl of a vector field in cubed sphere
        components using equation B6 in Yin et al. (2017).

        Parameters
        ----------
        u1 : array-like
            First contravariant component of the vector field in cubed
            sphere components.
        u2 : array-like
            Second contravariant component of the vector field in cubed
            sphere components.

        Returns
        -------
        array
            Radial curl of the vector field.
        """
        return (
            1
            / self.cs_basis.sqrt_detg
            * (
                self.D[0].dot(
                    self.cs_basis.g[:, 0, 1] * u1
                    + self.cs_basis.g[:, 1, 1] * u2
                )
                - self.D[1].dot(
                    self.cs_basis.g[:, 0, 0] * u1
                    + self.cs_basis.g[:, 0, 1] * u2
                )
            )
        )

    def sph_to_contravariant_cs(self, Ar, Atheta, Aphi):
        """Convert from spherical to contravariant cubed sphere.

        Converts from ``(east, north, up)`` to ``(u^1, u^2, u^3)`` (ref.
        Yin, 2017).

        The input must match the CS grid.

        Parameters
        ----------
        Ar : array-like
            Radial component of the vector field. Must match the
            dimensions of the cubed sphere grid.
        Atheta : array-like
            Latitudinal component of the vector field. Must match the
            dimensions of the cubed sphere grid.
        Aphi : array-like
            Longitudinal component of the vector field. Must match the
            dimensions of the cubed sphere grid.

        Returns
        -------
        u1, u2, u3 : arrays
            Contravariant components of the vector field in cubed sphere
            coordinates.
        """
        east = Aphi
        north = -Atheta
        up = Ar

        # print('TODO: Add checks that input matches grid etc.')

        v = np.vstack((east, north, up))
        v_components = np.einsum("nij, jn -> in", self.Qi, v)
        u1, u2, u3 = np.einsum("nij, jn -> in", self.Ps, v_components)

        return u1, u2, u3
