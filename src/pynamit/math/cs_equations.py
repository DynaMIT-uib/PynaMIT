"""Cubed sphere equations module.

This module contains the CSEquations class for representing cubed sphere
related equations.
"""

import numpy as np


class CSEquations(object):
    """Class for representing cubed sphere related equations.

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
            self._D = self.cs_basis.get_Diff(self.cs_basis.Ncs, coordinate="both")
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
                self.cs_basis.arr_xi, self.cs_basis.arr_eta, 1, self.cs_basis.arr_block
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
            self._Qi = self.cs_basis.get_Q(90 - self.cs_basis.arr_theta, self.RI, inverse=True)
        return self._Qi

    def curlr(self, u1, u2):
        """Calculate radial curl of vector field.

        Calculates the radial curl of a vector field in cubed sphere
        components using equation B6 in Yin et al. (2017).

        Parameters
        ----------
        u1 : array-like
            First contravariant cubed sphere component of the vector
            field.
        u2 : array-like
            Second contravariant cubed sphere component of the vector
            field.

        Returns
        -------
        array
            Radial curl of the vector field.
        """
        return (
            1
            / self.cs_basis.sqrt_detg
            * (
                self.D[0].dot(self.cs_basis.g[:, 0, 1] * u1 + self.cs_basis.g[:, 1, 1] * u2)
                - self.D[1].dot(self.cs_basis.g[:, 0, 0] * u1 + self.cs_basis.g[:, 0, 1] * u2)
            )
        )

    def sph_to_contravariant_cs(self, Ar, Atheta, Aphi):
        """Convert from spherical to contravariant cubed sphere.

        Converts from ``(east, north, up)`` to ``(u^1, u^2, u^3)`` (ref.
        Yin, 2017).

        The input must match the dimensions of the cubed sphere grid.

        Parameters
        ----------
        Ar : array-like
            Radial component of the vector field.
        Atheta : array-like
            Latitudinal component of the vector field.
        Aphi : array-like
            Longitudinal component of the vector field.

        Returns
        -------
        u1, u2, u3 : arrays
            Contravariant cubed sphere components of the vector field.
        """
        east = Aphi
        north = -Atheta
        up = Ar

        # print('TODO: Add checks that input matches grid etc.')

        v = np.vstack((east, north, up))
        v_components = np.einsum("nij, jn -> in", self.Qi, v)
        u1, u2, u3 = np.einsum("nij, jn -> in", self.Ps, v_components)

        return u1, u2, u3

    def calculate_fd_curl_matrix(self, stencil_size=1, interpolation_points=4):
        """Calculate matrix that returns the radial curl.

        Calculate matrix that maps column vector of (theta, phi) vector
        to its radial curl, using finite differences.

        Parameters
        ----------
        stencil_size : int, optional
            Size of the finite difference stencil.
        interpolation_points : int, optional
            Number of interpolation points.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix that returns the radial curl.
        """
        Dxi, Deta = self.cs_basis.get_Diff(
            self.cs_basis.N, coordinate="both", Ns=stencil_size, Ni=interpolation_points, order=1
        )

        g11_scaled = sp.diags(self.cs_basis.g[:, 0, 0] / self.cs_basis.sqrt_detg)
        g12_scaled = sp.diags(self.cs_basis.g[:, 0, 1] / self.cs_basis.sqrt_detg)
        g22_scaled = sp.diags(self.cs_basis.g[:, 1, 1] / self.cs_basis.sqrt_detg)

        # Construct matrix that gives radial curl from (u1, u2).
        D_curlr_u1u2 = sp.hstack(
            (
                (Dxi.dot(g12_scaled) - Deta.dot(g11_scaled)),
                (Dxi.dot(g22_scaled) - Deta.dot(g12_scaled)),
            )
        )

        # Construct matrix that transforms (theta, phi) to (u1, u2).
        Ps_dense = self.cs_basis.get_Ps(
            self.cs_basis.arr_xi, self.cs_basis.arr_eta, block=self.cs_basis.arr_block
        )

        # Extract relevant elements, rearrange matrix to map from
        # (theta, phi) and not (east, north). Also include Q matrix
        # normalization factors from Yin et al. (2017).
        RI_cos_lat = self.RI * np.cos(np.deg2rad(self.state.grid.lat))
        Ps = sp.vstack(
            (
                sp.hstack(
                    (
                        sp.diags(-Ps_dense[:, 0, 1] / self.RI),
                        sp.diags(Ps_dense[:, 0, 0] / RI_cos_lat),
                    )
                ),
                sp.hstack(
                    (
                        sp.diags(-Ps_dense[:, 1, 1] / self.RI),
                        sp.diags(Ps_dense[:, 1, 0] / RI_cos_lat),
                    )
                ),
            )
        )

        return D_curlr_u1u2.dot(Ps)

    @property
    def fd_curl_matrix(self):
        """Matrix for finite difference curl calculation."""
        if not hasattr(self, "_fd_curl_matrix"):
            self._fd_curl_matrix = self.calculate_fd_curl_matrix()
        return self._fd_curl_matrix

    @property
    def sh_curl_matrix(self):
        """Matrix for spherical harmonic curl calculation.

        Matrix that gets divergence-free SH coefficients from vectors of
        (theta, phi)-components, constructed from Laplacian matrix and
        (inverse) evaluation matrices.
        """
        if not hasattr(self, "_sh_curl_matrix"):
            G_df_pinv = self.state.basis_evaluator.least_squares_helmholtz.ATWA_plus_R_pinv[
                self.state.basis.index_length :, :
            ]
            self._sh_curl_matrix = self.state.basis_evaluator.G.dot(
                self.state.basis.laplacian().reshape((-1, 1)) * G_df_pinv
            )
        return self._sh_curl_matrix
