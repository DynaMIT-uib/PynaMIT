"""Cubed sphere basis module.

This module contains the CSBasis class for representing the cubed sphere
basis.
"""

import numpy as np
from pynamit.cubed_sphere import diffutils
from pynamit.cubed_sphere import arrayutils
import os
from scipy.special import binom
from scipy.sparse import coo_matrix
from scipy.interpolate import griddata

d2r = np.pi / 180
datapath = os.path.dirname(os.path.abspath(__file__)) + "/data/"


class CSBasis:
    """Class for representing cubed sphere bases.

    This module provides an implementation of the cubed sphere grid
    system following methods from Yin et al. (2017). The cubed sphere
    grid divides a sphere into six faces of a circumscribed cube,
    providing nearly uniform grid resolution and avoiding pole
    singularities. Each face uses a local (xi, eta) coordinate system
    mapped to global spherical coordinates (theta, phi). It includes
    tools for coordinate transformations, scalar and vector field
    interpolation and manipulation, numerical differentiation, and
    visualization utilities.

    Attributes
    ----------
    N : int
        Number of grid cells per cube edge (only set if N provided in
        constructor).
    arr_xi : ndarray
        Xi coordinates of grid points, in radians.
    arr_eta : ndarray
        Eta coordinates of grid points, in radians.
    arr_theta : ndarray
        Colatitude coordinates of grid points, in degrees.
    arr_phi : ndarray
        Longitude coordinates of grid points, in degrees.
    arr_block : ndarray
        Block indices (0-5) of grid points.
    arr_area : ndarray
        Grid cell areas normalized to unit sphere.
    g : ndarray
        Metric tensor
    sqrt_detg : ndarray
        Square root of determinant of the metric tensor.
    unit_area : ndarray
        Area of each grid cell.

    Notes
    -----
    The cubed sphere grid is organized into six faces as shown below,
    which defines the block structure of the grid:

          _______
          |     |
          |  V  |
    ______|_____|____________
    |     |     |     |     |
    | IV  |  I  | II  | III |
    |_____|_____|_____|_____|
          |     |
          | VI  |
          |_____|

    Block indices:
      - 0 = I   : Equator
      - 1 = II  : Equator
      - 2 = III : Equator
      - 3 = IV  : Equator
      - 4 = V   : North Pole
      - 5 = VI  : South Pole

    References
    ----------
    [1] Liang Yin, Chao Yang, Shi-Zhuang Ma, Ji-Zu Huang, Ying Cai
        (2017) Parallel numerical simulation of the thermal convection
        in the Earth's outer core on the cubed-sphere. Geophysical
        Journal International, 209(3), 1934–1954.
        DOI: 10.1093/gji/ggx125
    """

    def __init__(self, N=None):
        """Initialize the cubed sphere basis.

        If N is provided, initializes arrays for a grid with N×N cells
        on each cube face. The total number of grid points will be 6×N×N
        after removing duplicates at block boundaries.

        Parameters
        ----------
        N : int, optional
            Number of grid cells per cube edge. Must be even if
            provided.

        Raises
        ------
        TypeError
            If N is provided but is not an integer.
        ValueError
            If N is provided but is not an even number.
        """
        if N is not None:
            if not isinstance(N, (int, np.integer)):
                raise TypeError("N must be an integer")
            if N % 2 != 0:
                raise ValueError("Cubed sphere grid dimension must be even")

            self.N = N
            k, i, j = self.get_gridpoints(N)

            # Initialize grid points, skipping duplicates at boundaries.
            self.arr_xi = self.xi(i[:, :-1, :-1] + 0.5, N).flatten()
            self.arr_eta = self.eta(j[:, :-1, :-1] + 0.5, N).flatten()
            self.arr_block = k[:, :-1, :-1].flatten()

            # Convert to spherical coordinates.
            _, self.arr_theta, self.arr_phi = self.cube2spherical(
                self.arr_xi, self.arr_eta, self.arr_block, deg=True
            )

            # Calculate grid cell areas.
            step = np.diff(self.xi(np.array([0, 1]), N))[0]
            self.g = self.get_metric_tensor(self.arr_xi, self.arr_eta)
            self.sqrt_detg = np.sqrt(arrayutils.get_3D_determinants(self.g))
            self.unit_area = step**2 * self.sqrt_detg

            self.kind = "GRID"
            self.index_names = ["theta", "phi"]
            self.index_length = self.arr_theta.size + self.arr_phi.size
            self.index_arrays = [self.arr_theta, self.arr_phi]

            self.minimum_phi_sampling = 1
            self.caching = False

    def get_gridpoints(self, N, flat=False):
        """Generate grid point indices for given resolution.

        Parameters
        ----------
        N : int
            Number of grid cells per edge (N+1 points).
        flat : bool, optional
            Whether to return flattened arrays.

        Returns
        -------
        k : ndarray
            Block indices (0-5).
        i : ndarray
            Xi direction indices (0 to N).
        j : ndarray
            Eta direction indices (0 to N).

        Notes
        -----
        Arrays have shape (6,N+1,N+1) if `flat` is ``False``, or
        (6*(N+1)*(N+1),) if `flat` is ``True``.
        """
        k, i, j = np.meshgrid(np.arange(6), np.arange(N + 1), np.arange(N + 1), indexing="ij")
        if flat:
            return k.flatten(), i.flatten(), j.flatten()
        else:
            return k, i, j

    def xi(self, i, N):
        """Calculate xi coordinate for grid index.

        Maps index i=0 to -π/4 and i=N to π/4, providing the xi
        coordinate in the cubed sphere grid system.

        Parameters
        ----------
        i : array-like
            Index values (can be non-integer).
        N : int
            Grid resolution (number of cells per edge).

        Returns
        -------
        ndarray
            Xi coordinates in radians from -π/4 to π/4.

        Raises
        ------
        TypeError
            If `N` is not an integer.
        ValueError
            If `N` is less than 1.
        """
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N < 1:
            raise ValueError("N must be at least 1")
        return -np.pi / 4 + i * np.pi / (2 * N)

    def eta(self, j, N):
        """Calculate eta coordinate for grid index.

        Maps index ``j=0`` to -π/4 and ``j=N`` to π/4, providing the eta
        coordinate in the cubed sphere grid system. This function is
        mathematically identical to xi() but is provided separately for
        code clarity.

        Parameters
        ----------
        j : array-like
            Index values (can be non-integer).
        N : int
            Grid resolution (number of cells per edge).

        Returns
        -------
        ndarray
            Eta coordinates in radians from -π/4 to π/4.

        Raises
        ------
        TypeError
            If `N` is not an integer.
        ValueError
            If `N` is less than 1.
        """
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N < 1:
            raise ValueError("N must be at least 1")
        return -np.pi / 4 + j * np.pi / (2 * N)

    def get_delta(self, xi, eta):
        """Calculate delta parameter for metric calculations.

        Computes ``δ = 1 + tan²(ξ) + tan²(η)``.

        Parameters
        ----------
        xi : array-like
            Xi coordinates in radians.
        eta : array-like
            Eta coordinates in radians.

        Returns
        -------
        ndarray
            Delta values with shape determined by broadcasting rules.
        """
        xi, eta = np.broadcast_arrays(xi, eta)

        return 1 + np.tan(xi) ** 2 + np.tan(eta) ** 2

    def get_metric_tensor(self, xi, eta, r=1, covariant=True):
        """Calculate metric tensor components.

        Calculates the metric tensor components for the cubed sphere
        grid system at given points, which relate coordinate
        differentials to distances according to the equation
        ``ds² = gᵢⱼ dxⁱdxʲ``. Implementation based on equation (12) from
        Yin et al. (2017).

        Parameters
        ----------
        xi : array-like
            Xi coordinates in radians.
        eta : array-like
            Eta coordinates in radians.
        r : array-like, optional
            Radial coordinates.
        covariant : bool, optional
            If ``True`` return covariant components, otherwise return
            contravariant components.

        Returns
        -------
        g : ndarray
            Metric tensor components with shape (N,3,3) where N is the
            number of input points. Last two dimensions are tensor
            indices.
        """
        # Broadcast and flatten.
        xi, eta, r = map(np.ravel, np.broadcast_arrays(xi, eta, r))
        delta = self.get_delta(xi, eta)

        g = np.empty((xi.size, 3, 3))
        g[:, 0, 0] = r**2 / (np.cos(xi) ** 4 * np.cos(eta) ** 2 * delta**2)
        g[:, 0, 1] = (
            -(r**2) * np.tan(xi) * np.tan(eta) / (np.cos(xi) ** 2 * np.cos(eta) ** 2 * delta**2)
        )
        g[:, 0, 2] = 0
        g[:, 1, 0] = (
            -(r**2) * np.tan(xi) * np.tan(eta) / (np.cos(xi) ** 2 * np.cos(eta) ** 2 * delta**2)
        )
        g[:, 1, 1] = r**2 / (np.cos(xi) ** 2 * np.cos(eta) ** 4 * delta**2)
        g[:, 1, 2] = 0
        g[:, 2, 0] = 0
        g[:, 2, 1] = 0
        g[:, 2, 2] = 1

        if covariant:
            # Return covariant components.
            return g
        else:
            # Return contravariant components.
            return arrayutils.invert_3D_matrices(g)

    def cube2cartesian(self, xi, eta, r=1, block=0):
        """Calculate Cartesian ECEF coordinates of given points.

        Output will have same unit as `r`.

        Calculations based on equations from Appendix A of Yin et al.
        (2017).

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates in radians.
        eta : array-like
            Array of eta coordinates in radians.
        r : array-like, optional
            Array of radii.
        block : array-like, optional
            Array of block indices.

        Returns
        -------
        x : array
            Array of Cartesian x coordinates, shape determined by input
            according to broadcasting rules.
        y : array
            Array of Cartesian y coordinates, shape determined by input
            according to broadcasting rules.
        z : array
            Array of Cartesian z coordinates, shape determined by input
            according to broadcasting rules.
        """
        xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)
        delta = self.get_delta(xi, eta)
        x, y, z = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)

        # Calculate block 0 (A2).
        iii = block == 0
        x[iii] = r[iii] / np.sqrt(delta[iii])
        y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # Calculate block 1 (A6).
        iii = block == 1
        x[iii] = -r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        y[iii] = r[iii] / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # Calculate block 2 (A10).
        iii = block == 2
        x[iii] = -r[iii] / np.sqrt(delta[iii])
        y[iii] = -r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # Calculate block 3 (A14).
        iii = block == 3
        x[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        y[iii] = -r[iii] / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # Calculate block 4 (A18).
        iii = block == 4
        x[iii] = -r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = r[iii] / np.sqrt(delta[iii])
        # Calculate block 5 (A22).
        iii = block == 5
        x[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = -r[iii] / np.sqrt(delta[iii])

        return (x, y, z)

    def cube2spherical(self, xi, eta, block, r=1, deg=False):
        """Convert from cubed sphere to spherical coordinates.

        Converts cubed sphere coordinates to spherical coordinates
        through intermediate Cartesian coordinates using equations from
        Appendix A of Yin et al. (2017).

        Parameters
        ----------
        xi : array-like
            Xi coordinates in radians.
        eta : array-like
            Eta coordinates in radians.
        block : array-like
            Block indices (0-5)
        r : float or array-like, optional
            Radial coordinates.
        deg : bool, optional
            Return angles in degrees if True, otherwise radians.

        Returns
        -------
        r : ndarray
            Radial coordinates (same units as input r).
        theta : ndarray
            Colatitude in radians or degrees.
        phi : ndarray
            Longitude in radians or degrees.
        """
        xi, eta = np.float64(xi), np.float64(eta)
        xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)

        x, y, z = self.cube2cartesian(xi, eta, r, block)
        phi = np.arctan2(y, x)
        theta = np.arccos(z / r)

        if deg:
            phi, theta = np.rad2deg(phi), np.rad2deg(theta)

        return (r, theta, phi)

    def get_Pc(self, xi, eta, r=1, block=0, inverse=False):
        """Get Pc matrix.

        Calculates elements of transformation matrix `Pc` at all input
        points.

        The `Pc` matrix transforms Cartesian components ``(ux, uy, uz)``
        to contravariant components in a cubed sphere coordinate
        system::

            |u1| = |P00 P01 P02| |ux|
            |u2| = |P10 P11 P12| |uy|
            |u3| = |P20 P21 P22| |uz|

        The output, `Pc`, will have shape ``(N, 3, 3)``.

        Calculations based on equations from Appendix A of Yin et al.
        (2017), with similar notation.

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates, in radians.
        eta : array-like
            Array of eta coordinates, in radians.
        r : array-like, optional
            Array of radii.
        block : array-like, optional
            Array of block indices.
        inverse : bool, optional
            Set to ``True`` if you want the inverse transformation
            matrix.

        Returns
        -------
        Pc : array
            Transformation matrices `Pc`, one for each point described
            by the input parameters (using broadcasting rules). For
            ``N`` such points, `Pc` will have shape ``(N, 3, 3)``, where
            the last two dimensions refer to column and row of the
            matrix.
        """
        # Broadcast and flatten.
        xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
        delta = self.get_delta(xi, et)
        Pc = np.empty((delta.size, 3, 3))

        rsec2xi = r / np.cos(xi) ** 2
        rsec2et = r / np.cos(et) ** 2

        # Calculate block 0.
        iii = block == 0
        Pc[iii, 0, 0] = -np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 1] = np.sqrt(delta[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] = 0
        Pc[iii, 1, 0] = -np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 1] = 0
        Pc[iii, 1, 2] = np.sqrt(delta[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = 1 / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = np.tan(et[iii]) / np.sqrt(delta[iii])

        # Calculate block 1.
        iii = block == 1
        Pc[iii, 0, 0] = -np.sqrt(delta[iii]) / rsec2xi[iii]
        Pc[iii, 0, 1] = -np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] = 0
        Pc[iii, 1, 0] = 0
        Pc[iii, 1, 1] = -np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 2] = np.sqrt(delta[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = -np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = 1 / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = np.tan(et[iii]) / np.sqrt(delta[iii])

        # Calculate block 2.
        iii = block == 2
        Pc[iii, 0, 0] = np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 1] = -np.sqrt(delta[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] = 0
        Pc[iii, 1, 0] = np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 1] = 0
        Pc[iii, 1, 2] = np.sqrt(delta[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = -1 / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = -np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = np.tan(et[iii]) / np.sqrt(delta[iii])

        # Calculate block 3.
        iii = block == 3
        Pc[iii, 0, 0] = np.sqrt(delta[iii]) / rsec2xi[iii]
        Pc[iii, 0, 1] = np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] = 0
        Pc[iii, 1, 0] = 0
        Pc[iii, 1, 1] = np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 2] = np.sqrt(delta[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = -1 / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = np.tan(et[iii]) / np.sqrt(delta[iii])

        # Calculate block 4.
        iii = block == 4
        Pc[iii, 0, 0] = 0
        Pc[iii, 0, 1] = np.sqrt(delta[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] = -np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 1, 0] = -np.sqrt(delta[iii]) / rsec2et[iii]
        Pc[iii, 1, 1] = 0
        Pc[iii, 1, 2] = -np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = -np.tan(et[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = 1 / np.sqrt(delta[iii])

        # Calculate block 5.
        iii = block == 5
        Pc[iii, 0, 0] = 0
        Pc[iii, 0, 1] = np.sqrt(delta[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] = np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 1, 0] = np.sqrt(delta[iii]) / rsec2et[iii]
        Pc[iii, 1, 1] = 0
        Pc[iii, 1, 2] = np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = np.tan(et[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = -1 / np.sqrt(delta[iii])

        if inverse:
            return arrayutils.invert_3D_matrices(Pc)
        else:
            return Pc

    def get_Ps(self, xi, eta, r=1, block=0, inverse=False):
        """Get Ps matrix.

        Calculates elements of transformation matrix `Ps` at all input
        points.

        The `Ps` matrix transforms vector components
        ``(u_east, u_north, u_r)`` to contravariant components in a
        cubed sphere coordinate system::

            |u1| = |P00 P01 P02| |u_east|
            |u2| = |P10 P11 P12| |u_north|
            |u3| = |P20 P21 P22| |u_r|

        The output, `Ps`, will have shape ``(N, 3, 3)``.

        Calculations based on equations from Appendix A of Yin et al.
        (2017), with similar notation, except that ``lambda`` and
        ``phi`` is replaced with ``east`` and ``north`` (here, ``phi``
        means longitude, and not latitude as in Yin et al. (2017).

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates, in radians.
        eta : array-like
            Array of eta coordinates, in radians.
        r : array-like, optional
            Array of radii.
        block : array-like, optional
            Array of block indices.
        inverse : bool, optional
            Set to ``True`` if you want the inverse transformation
            matrix.

        Returns
        -------
        Ps : array
            Transformation matrices `Ps`, one for each point described
            by the input parameters (using broadcasting rules). For
            ``N`` such points, `Ps` will have shape ``(N, 3, 3)``, where
            the last two dimensions refer to column and row of the
            matrix.
        """
        # Broadcast and flatten.
        xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
        delta = self.get_delta(xi, et)
        Ps = np.empty((delta.size, 3, 3))

        # Calculate block 0.
        iii = block == 0
        Ps[iii, 0, 0] = 1
        Ps[iii, 0, 1] = 0
        Ps[iii, 0, 2] = 0
        Ps[iii, 1, 0] = np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] = np.cos(xi[iii]) * np.sin(et[iii]) ** 2 + np.cos(et[iii]) ** 2 / np.cos(
            xi[iii]
        )
        Ps[iii, 1, 2] = 0
        Ps[iii, 2, 0] = 0
        Ps[iii, 2, 1] = 0
        Ps[iii, 2, 2] = 1

        # Calculate block 1.
        iii = block == 1
        Ps[iii, 0, 0] = 1
        Ps[iii, 0, 1] = 0
        Ps[iii, 0, 2] = 0
        Ps[iii, 1, 0] = np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] = np.cos(xi[iii]) * np.sin(et[iii]) ** 2 + np.cos(et[iii]) ** 2 / np.cos(
            xi[iii]
        )
        Ps[iii, 1, 2] = 0
        Ps[iii, 2, 0] = 0
        Ps[iii, 2, 1] = 0
        Ps[iii, 2, 2] = 1

        # Calculate block 2.
        iii = block == 2
        Ps[iii, 0, 0] = 1
        Ps[iii, 0, 1] = 0
        Ps[iii, 0, 2] = 0
        Ps[iii, 1, 0] = np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] = np.cos(xi[iii]) * np.sin(et[iii]) ** 2 + np.cos(et[iii]) ** 2 / np.cos(
            xi[iii]
        )
        Ps[iii, 1, 2] = 0
        Ps[iii, 2, 0] = 0
        Ps[iii, 2, 1] = 0
        Ps[iii, 2, 2] = 1

        # Calculate block 3.
        iii = block == 3
        Ps[iii, 0, 0] = 1
        Ps[iii, 0, 1] = 0
        Ps[iii, 0, 2] = 0
        Ps[iii, 1, 0] = np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] = np.cos(xi[iii]) * np.sin(et[iii]) ** 2 + np.cos(et[iii]) ** 2 / np.cos(
            xi[iii]
        )
        Ps[iii, 1, 2] = 0
        Ps[iii, 2, 0] = 0
        Ps[iii, 2, 1] = 0
        Ps[iii, 2, 2] = 1

        # Calculate block 4.
        iii = block == 4
        Ps[iii, 0, 0] = -(np.cos(xi[iii]) ** 2) * np.tan(et[iii])
        Ps[iii, 0, 1] = (
            -delta[iii] * np.tan(xi[iii]) * np.cos(xi[iii]) ** 2 / np.sqrt(delta[iii] - 1)
        )
        Ps[iii, 0, 2] = 0
        Ps[iii, 1, 0] = np.cos(et[iii]) ** 2 * np.tan(xi[iii])
        Ps[iii, 1, 1] = (
            -delta[iii] * np.tan(et[iii]) * np.cos(et[iii]) ** 2 / np.sqrt(delta[iii] - 1)
        )
        Ps[iii, 1, 2] = 0
        Ps[iii, 2, 0] = 0
        Ps[iii, 2, 1] = 0
        Ps[iii, 2, 2] = 1

        # Calculate block 5.
        iii = block == 5
        Ps[iii, 0, 0] = np.cos(xi[iii]) ** 2 * np.tan(et[iii])
        Ps[iii, 0, 1] = (
            delta[iii] * np.tan(xi[iii]) * np.cos(xi[iii]) ** 2 / np.sqrt(delta[iii] - 1)
        )
        Ps[iii, 0, 2] = 0
        Ps[iii, 1, 0] = -(np.cos(et[iii]) ** 2) * np.tan(xi[iii])
        Ps[iii, 1, 1] = (
            delta[iii] * np.tan(et[iii]) * np.cos(et[iii]) ** 2 / np.sqrt(delta[iii] - 1)
        )
        Ps[iii, 1, 2] = 0
        Ps[iii, 2, 0] = 0
        Ps[iii, 2, 1] = 0
        Ps[iii, 2, 2] = 1

        if inverse:
            return arrayutils.invert_3D_matrices(Ps)
        else:
            return Ps

    def get_Qij(self, xi, eta, block_i, block_j):
        """Get Qij matrix.

        Calculates matrix `Qij` that transforms contravariant vector
        components from block `block_i` to `block_j`.

        Calculations are done via transformation to spherical
        coordinates, as suggested by Yin et al. (2017) See equations
        (66) and (67) in their paper.

        It works like this, where ``(u1, u2, u3)`` refer to
        contravariant vector components in the cubed sphere coordinate
        system::

            |u1_j|      |u1_i|
            |u2_j| = Qij|u2_i|
            |u3_j|      |u3_i|

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates on block given by `block_i`, in
            radians.
        eta : array-like
            Array of eta coordinates on block given by `block_i`, in
            radians.
        block_i : array-like, optional
            Indices of block(s) from which to transform vector
            components.
        block_j : array-like, optional
            Indices of block(s) to which to transform vector components.

        Returns
        -------
        Qij : array
            Transformation matrices `Qij`, one for each point described
            by the input parameters (using broadcasting rules). For
            ``N`` such points, `Qij` will have shape ``(N, 3, 3)``,
            where the last two dimensions refer to column and row of the
            matrix.
        """
        # Broadcast and flatten.
        xi_i, eta_i, block_i, block_j = map(
            np.ravel, np.broadcast_arrays(xi, eta, block_i, block_j)
        )

        Psi_inv = self.get_Ps(xi_i, eta_i, r=1, block=block_i, inverse=True)

        # Find the xi, eta coordinates on block j.
        r, theta, phi = self.cube2spherical(xi_i, eta_i, r=1, block=block_i, deg=True)
        xi_j, eta_j, _ = self.geo2cube(phi, 90 - theta, block=block_j)

        # Calculate Ps relative to block j.
        Psj = self.get_Ps(xi_j, eta_j, r=1, block=block_j)

        # Multiply each of the N matrices to get Qij.
        Qij = np.einsum("nij, njk -> nik", Psj, Psi_inv)

        return Qij

    def get_Q(self, lat, r, inverse=False):
        """Get Q matrix.

        Calculates the matrices that convert from unnormalized spherical
        components to normalized spherical vector components::

            |u_east_normalized |    |u_east |
            |u_north_normalized| = Q|u_north|
            |u_r_normalized    |    |u_r    |

        Based on equations after (A25) in Yin et al. (2017).

        Parameters
        ----------
        lat : array
            Array of latitudes, in degrees.
        r : array
            Array of radii.
        inverse : bool, optional
            Set to ``True`` if you want the inverse transformation
            matrix.

        Returns
        -------
        Q : array
            ``(N, 3, 3)`` array, where ``N`` is the size implied by
            broadcasting the input.
        """
        lat, r = map(np.ravel, np.broadcast_arrays(lat, r))

        Q = np.zeros((lat.size, 3, 3), dtype=np.float64)
        Q[:, 0, 0] = r * np.cos(np.deg2rad(lat))
        Q[:, 1, 1] = r
        Q[:, 2, 2] = 1

        if inverse:
            return arrayutils.invert_3D_matrices(Q)
        else:
            return Q

    def get_Diff(self, N, coordinate="xi", Ns=1, Ni=4, order=1):
        """Get scalar field differentiation matrix.

        Calculate matrix that differentiates a scalar field, defined on
        a ``(6, N, N)`` grid, with respect to ``xi`` or ``eta``.

        Parameters
        ----------
        N : int
            Number of grid cells in each dimension on each block.
        coordinate : string, {'xi', 'eta', 'both'}
            Which coordinate to differentiate with respect to.
        Ns : int, optional
            Differentiation stencil size.
        Ni : int, optional
            Number of points to use for interpolation for points in the
            stencil that fall on non-integer grid points on neighboring
            blocks.
        order : int, optional
            Order of differentiation. Make sure that ``Ns >= order``.
            Currently only first order differentiation is supported.

        Returns
        -------
        D : sparse matrix
            Sparse ``(6*N*N, 6*N*N)`` matrix that calculates the
            derivative of a scalar field with respect to ``xi`` or
            ``eta`` as ``derivative = D.dot(f)``, where ``f`` is the
            scalar field.

        Raises
        ------
        ValueError
            If `coordinate` is not 'xi', 'eta', or 'both'.
            If `Ns` is less than `order.
        NotImplementedError
            If `order` is not 1.
        """
        if coordinate not in ["xi", "eta", "both"]:
            raise ValueError(
                f'coordinate must be either "xi", "eta", or "both". Not  {coordinate}.'
            )

        if Ns < order:
            raise ValueError("Ns must be >= order. You gave {} and {}".format(Ns, order))

        if order != 1:
            raise NotImplementedError("Only first order differentiation is supported.")

        shape = (6, N, N)
        size = 6 * N * N

        h = self.xi(1, N) - self.xi(0, N)  # Step size between each grid cell

        k, i, j = map(
            np.ravel, np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing="ij")
        )

        # Set up differentiation stencil for first order derivative.
        stencil_points = np.hstack((np.r_[-Ns:0], np.r_[1 : Ns + 1]))
        Nsp = len(stencil_points)
        stencil_weight = diffutils.stencil(stencil_points, order=1, h=h)

        i_diff = np.hstack([i + _ for _ in stencil_points])
        j_diff = np.hstack([j + _ for _ in stencil_points])
        k_const, i_const, j_const = (np.tile(k, Nsp), np.tile(i, Nsp), np.tile(j, Nsp))
        weights = np.repeat(stencil_weight, size)

        rows = np.tile(np.ravel_multi_index((k, i, j), shape), Nsp)
        if coordinate in ["xi", "both"]:
            Dxi = self.get_interpolation_matrix(
                k_const, i_diff, j_const, N, Ni, rows=rows, weights=weights
            )
        if coordinate in ["eta", "both"]:
            Deta = self.get_interpolation_matrix(
                k_const, i_const, j_diff, N, Ni, rows=rows, weights=weights
            )

        if coordinate == "both":
            return (Dxi, Deta)
        if coordinate == "xi":
            return Dxi
        if coordinate == "eta":
            return Deta

    def get_interpolation_matrix(self, k, i, j, N, Ni, weights=None, rows=None):
        """Get matrix for grid to cubed sphere interpolation.

        Calculates a sparse matrix D that interpolates from grid points
        in a ``(6, N, N)`` grid to the indices (`k`, `i`, `j`).

        `D` will have ``6*N**2`` columns that refer to the ``(6, N, N)``
        grid points, spanning the 6 blocks in the cubed sphere, with
        duplicate points on the boundaries.

        Parameters
        ----------
        k : array-like
            Integer indices that refer to cube block. Must be ``>= 0``
            and ``<= 5``. Will be flattened.
        i : array-like
            Integer indices that refer to the ``xi``-direction (but can
            be negative or ``>= N``). Will be flattened.
        j : array-like
            Integer indices that refer to the ``eta``-direction (but can
            be negative or ``>= N``). Will be flattened.
        N : int
            Number of grid points.
        Ni : int
            Number of interpolation points. Must be ``<= N`` (4 is often
            appropriate).
        weights : array-like, optional
            If different values of `k`, `i`, `j` are assigned to the
            same row, the corresponding element will have value 1 (or
            whatever the interpolation dictates) unless weights is
            specified. For differentiation, use weights to specify the
            stencil coefficients.
        rows : array-like, optional
            The row index of each element in `k`, `i`, `j`. Different
            elements of `k`, `i`, `j` can be put in the same row. If not
            specified, each element in `k`, `i`, `j` will be given its
            own row.

        Returns
        -------
        D : sparse matrix
            ``(rows.max() + 1 by 6*N*N)`` matrix that, when multiplied
            by a vector containing a scalar field on the ``6*N*N`` grid
            points, produces interpolated values at the given grid
            points. The grid points may be outside the cube blocks, for
            example they can be negative (actually that's the point,
            otherwise this function would not be needed).
        """
        if Ni > N:
            raise ValueError("Ni must be <= N")
        k, i, j = map(np.ravel, [k, i, j])

        shape = (6, N, N)
        size = 6 * N**2

        if rows is None:
            rows = np.arange(k.size)

        if weights is None:
            weights = np.ones(k.size)
        weights = weights / Ni

        h = self.xi(1, N) - self.xi(0, N)  # Step size between each grid cell

        cols = np.full(k.size, -1, dtype=np.int64)

        # Find new indices inside block dimensions (possibly floats).
        xi, eta = self.xi(i, N), self.eta(j, N)
        r, theta, phi = self.cube2spherical(xi, eta, k, r=1.0, deg=True)
        new_xi, new_eta, new_k = self.geo2cube(phi, 90 - theta)
        new_i, new_j = new_xi / h + (N - 1) / 2, new_eta / h + (N - 1) / 2

        # Uniform CS grids need at least one integer in each index pair.
        assert np.all(
            (np.isclose(new_i - np.rint(new_i), 0) | np.isclose(new_j - np.rint(new_j), 0))
        )

        # Fill in column indices for index pairs that are both integers.
        ii_integers = np.isclose(new_i - np.rint(new_i), 0) & np.isclose(new_j - np.rint(new_j), 0)
        cols[ii_integers] = np.ravel_multi_index(
            (
                new_k[ii_integers],
                np.rint(new_i[ii_integers]).astype(np.int64),
                np.rint(new_j[ii_integers]).astype(np.int64),
            ),
            shape,
        )

        # The rest of the index pairs need interpolation. Find these
        # indices.
        i_is_float = ~np.isclose(np.rint(new_i) - new_i, 0)
        j_is_float = ~np.isclose(np.rint(new_j) - new_j, 0)

        # No new index pair should have two floats.
        assert sum(i_is_float & j_is_float) == 0
        # All missing columns match indices where i or j are float.
        assert sum(i_is_float | j_is_float) == sum(cols == -1)

        j_floats = new_j[j_is_float].reshape((-1, 1))
        i_floats = new_i[i_is_float].reshape((-1, 1))

        # Define the (integer) points which will be used to interpolate.
        interpolation_points = np.arange(Ni).reshape((1, -1))
        j_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2 - 1, 0, N - 1, axis=1
        )
        i_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2 - 1, 0, N - 1, axis=1
        )

        # Calculate barycentric weights wj (Berrut & Trefethen, 2004).
        j_distances = j_floats - j_interpolation_points
        i_distances = i_floats - i_interpolation_points
        w = (-1) ** interpolation_points * binom(Ni - 1, interpolation_points)
        w_i = w / i_distances / np.sum(w / i_distances, axis=1).reshape((-1, 1))
        w_j = w / j_distances / np.sum(w / j_distances, axis=1).reshape((-1, 1))

        # Expand column, row, and weight arrays to allow for
        # interpolation weights (duplication where no interpolation
        # is required).
        stacked_weights = np.tile(weights, (Ni, 1)).T
        stacked_cols = np.tile(cols, (Ni, 1)).T
        stacked_rows = np.tile(rows, (Ni, 1)).T

        # Specify columns and weights where interpolation is required.
        stacked_cols[i_is_float] = np.ravel_multi_index(
            (
                np.tile(new_k[i_is_float], (Ni, 1)).T,
                i_interpolation_points,
                np.rint(np.tile(new_j[i_is_float], (Ni, 1))).astype(np.int64).T,
            ),
            shape,
        )
        stacked_cols[j_is_float] = np.ravel_multi_index(
            (
                np.tile(new_k[j_is_float], (Ni, 1)).T,
                np.rint(np.tile(new_i[j_is_float], (Ni, 1))).astype(np.int64).T,
                j_interpolation_points,
            ),
            shape,
        )
        stacked_weights[i_is_float] = stacked_weights[i_is_float] * w_i * Ni
        stacked_weights[j_is_float] = stacked_weights[j_is_float] * w_j * Ni

        D = coo_matrix(
            (stacked_weights.flatten(), (stacked_rows.flatten(), stacked_cols.flatten())),
            shape=(rows.max() + 1, size),
        )
        # Get rid of duplicates (maybe this doesn't do anything?).
        D.count_nonzero()

        return D

    def block(self, lon, lat):
        """Determine cube faces (blocks) of spherical coordinates.

        For each input point, determines which of the six cube faces is
        closest by calculating distances to face midpoints in Cartesian
        space.

        Parameters
        ----------
        lon : array-like
            Geocentric longitude(s) in degrees.
        lat : array-like
            Geocentric latitude(s) in degrees.

        Returns
        -------
        ndarray
            Indices of the block that each (lon, lat) point belongs to:
            - 0 (I)   : Equatorial face at 0° longitude
            - 1 (II)  : Equatorial face at 90° longitude
            - 2 (III) : Equatorial face at 180° longitude
            - 3 (IV)  : Equatorial face at 270° longitude
            - 4 (V)   : North polar face
            - 5 (VI)  : South polar face

        Notes
        -----
        The method uses Euclidean distances to face midpoints in
        Cartesian space to determine block membership. This ensures
        unique block assignment even for points near block boundaries.
        """
        lon, lat = np.broadcast_arrays(lon, lat)
        lat, lon = lat.flatten(), lon.flatten()

        # Convert to spherical coordinates in radians.
        th, ph = np.deg2rad(90 - lat), np.deg2rad(lon)

        # Calculate Cartesian coordinates of input points.
        xyz = np.vstack(
            (
                np.cos(ph) * np.sin(th),  # x
                np.sin(th) * np.sin(ph),  # y
                np.cos(th),  # z
            )
        )

        # Define face midpoint xyz coordinates.
        face_midpoints = np.array(
            [
                [1, 0, 0],  # I   (0°)
                [0, 1, 0],  # II  (90°)
                [-1, 0, 0],  # III (180°)
                [0, -1, 0],  # IV  (270°)
                [0, 0, 1],  # V   (North)
                [0, 0, -1],  # VI  (South)
            ]
        )

        # Calculate distances to each face midpoint.
        distances = np.empty((6, xyz.shape[1]))
        for i in range(6):
            distances[i] = np.linalg.norm(xyz - face_midpoints[i].reshape((3, 1)), axis=0)

        safety_distance = 1e-10  # To prevent ambiguous assignment at boundaries

        # Initialize blocks array.
        blocks = np.zeros(xyz.shape[1], dtype=int)

        # Assign points to blocks with smallest face midpoint distance.
        for i in range(6):
            blocks[distances[i] < np.choose(blocks, distances) - safety_distance] = i

        return blocks

    def geo2cube(self, lon, lat, block=None):
        """Convert geocentric coordinates to cube coordinates.

        Input parameters must have same shape. Output will have same
        shape.

        Parameters
        ----------
        lon : array
            Geocentric longitude(s) to convert to cube coords, in
            degrees.
        lat : array
            Geocentric latitude(s) to convert to cube coords, in
            degrees.
        block : array-like, optional
            Option to specify cube block. If ``None``, it will be
            calculated. If specified, be careful because the function
            will map points at opposite side of the sphere to specified
            block.

        Returns
        -------
        xi : array
            `xi`, as defined in Ronchi et al. (1996). Unit is radians.
        eta : array
            `eta`, as defined in Ronchi et al. (1996). Unit is radians.
        block : array
            Index of the block that `xi`, `eta` belongs to.
        """
        lon, lat = np.broadcast_arrays(lon, lat)
        shape = lon.shape
        N = lon.size

        # Find the correct block for each point.
        if block is None:
            block = self.block(lon, lat)
        else:
            block = block * np.ones_like(lat)

        block, lon, lat = block.flatten(), lon.flatten(), lat.flatten()

        # Prepare parameters.
        X, Y, xi, eta = np.empty(N), np.empty(N), np.empty(N), np.empty(N)

        # Calculate X and Y according to Ronchi et al. (1996).
        theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)
        X[block == 0] = np.tan(phi[block == 0])
        X[block == 1] = -1 / np.tan(phi[block == 1])
        X[block == 2] = np.tan(phi[block == 2])
        X[block == 3] = -1 / np.tan(phi[block == 3])
        X[block == 4] = np.tan(theta[block == 4]) * np.sin(phi[block == 4])
        X[block == 5] = -np.tan(theta[block == 5]) * np.sin(phi[block == 5])

        Y[block == 0] = 1 / (np.tan(theta[block == 0]) * np.cos(phi[block == 0]))
        Y[block == 1] = 1 / (np.tan(theta[block == 1]) * np.sin(phi[block == 1]))
        Y[block == 2] = -1 / (np.tan(theta[block == 2]) * np.cos(phi[block == 2]))
        Y[block == 3] = -1 / (np.tan(theta[block == 3]) * np.sin(phi[block == 3]))
        Y[block == 4] = -np.tan(theta[block == 4]) * np.cos(phi[block == 4])
        Y[block == 5] = -np.tan(theta[block == 5]) * np.cos(phi[block == 5])

        xi, eta = np.arctan(X), np.arctan(Y)

        return xi.reshape(shape), eta.reshape(shape), block.reshape(shape)

    def get_projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates."""
        coastlines = np.load(datapath + "coastlines_" + resolution + ".npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield self.geo2cube(lon, lat)

    def interpolate_vector_components(
        self, u_east, u_north, u_r, theta, phi, theta_target, phi_target, **kwargs
    ):
        """Interpolate vector components.

        Interpolates vector components defined on (theta, phi) to given
        spherical coordinates.

        Broadcasting rules apply for input and output separately.

        Parameters
        ----------
        u_east : array
            Array of eastward components.
        u_north : array
            Array of northward components.
        u_r : array
            Array of radial components.
        theta : array
            Array of coordinates for components.
        phi : array
            Array of coordinates for vector components.
        theta_target : array
            Array of target coordinates.
        phi_target : array
            Array of target coordinates.

        **kwargs
            Passed to scipy.interpolate.griddata which performs the
            interpolation on each block.

        Returns
        -------
        interpolated_vector : array
            3 x N vector of interpolated components (east, north, up).
        """
        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        # xi, eta, block = np.broadcast_arrays(xi, eta, block)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        u_east, u_north, u_r, theta, phi = np.broadcast_arrays(u_east, u_north, u_r, theta, phi)
        u_east, u_north, u_r, theta, phi = (
            u_east.flatten(),
            u_north.flatten(),
            u_r.flatten(),
            theta.flatten(),
            phi.flatten(),
        )

        # Define vectors that point to all the original points.
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        # Convert vector components to cubed sphere coordinates.
        u_xi, u_eta, u_block = self.geo2cube(phi, 90 - theta)
        Ps = self.get_Ps(u_xi, u_eta, r=1, block=u_block)
        Q = self.get_Q(90 - theta, r=1, inverse=True)
        Ps_normalized = np.einsum("nij, njk -> nik", Ps, Q)
        u_vec_sph = np.vstack((u_east, u_north, u_r))
        u_vec = np.einsum("nij, nj -> ni", Ps_normalized, u_vec_sph.T).T

        interpolated_u1 = np.empty_like(block, dtype=np.float64)
        interpolated_u2 = np.empty_like(block, dtype=np.float64)
        interpolated_u3 = np.empty_like(block, dtype=np.float64)

        # Loop over blocks and interpolate on each block.
        for i in range(6):
            # Express vector components with respect to block i.
            Qij = self.get_Qij(u_xi, u_eta, u_block, i)
            u_vec_i = np.einsum("nij, nj -> ni", Qij, u_vec.T).T

            # Filter points whose position vectors have component
            # anti-parallel to center of the block.
            _, th, ph = self.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).reshape(
                (-1, 1)
            )
            mask = np.sum(r0 * r, axis=0) > 0

            xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block=i)

            interpolated_u1[block == i] = griddata(
                np.vstack((xi_[mask], eta_[mask])).T,
                u_vec_i[0][mask],
                np.vstack((xi[block == i], eta[block == i])).T,
                **kwargs,
            )
            interpolated_u2[block == i] = griddata(
                np.vstack((xi_[mask], eta_[mask])).T,
                u_vec_i[1][mask],
                np.vstack((xi[block == i], eta[block == i])).T,
                **kwargs,
            )
            interpolated_u3[block == i] = griddata(
                np.vstack((xi_[mask], eta_[mask])).T,
                u_vec_i[2][mask],
                np.vstack((xi[block == i], eta[block == i])).T,
                **kwargs,
            )

        # Convert back to spherical.
        _, theta_out, _ = self.cube2spherical(xi, eta, block, deg=True)
        u = np.vstack((interpolated_u1, interpolated_u2, interpolated_u3))
        Q = self.get_Q(90 - theta_out, r=1, inverse=False)
        Ps_inv = self.get_Ps(xi, eta, r=1, block=block, inverse=True)
        Ps_normalized_inv = np.einsum("nij, njk -> nik", Q, Ps_inv)
        u_east_int, u_north_int, u_r_int = np.einsum("nij, nj -> ni", Ps_normalized_inv, u.T).T

        return u_east_int, u_north_int, u_r_int

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values.

        Interpolate scalar values defined on (`theta`, `phi`) to given
        spherical coordinates.

        Broadcasting rules apply for input and output separately.

        Parameters
        ----------
        scalar : array
            Array of scalar values.
        theta : array
            Array of coordinates for components.
        phi : array
            Array of coordinates for vector components.
        theta_target : array
            Array of target coordinates.
        phi_target : array
            Array of target coordinates.

        **kwargs
            Passed to scipy.interpolate.griddata which performs the
            interpolation on each block.

        Returns
        -------
        interpolated_scalar : array
            Array of interpolated components (east, north, up).
        """
        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        # xi, eta, block = np.broadcast_arrays(xi, eta, block)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        scalar, theta, phi = np.broadcast_arrays(scalar, theta, phi)
        scalar, theta, phi = scalar.flatten(), theta.flatten(), phi.flatten()

        # Define vectors that point to all the original points.
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        interpolated_scalar = np.empty_like(block, dtype=np.float64)

        # Loop over blocks and interpolate on each block.
        for i in range(6):
            # Filter points whose position vectors have component
            # anti-parallel to center of the block.
            _, th, ph = self.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).reshape(
                (-1, 1)
            )
            mask = np.sum(r0 * r, axis=0) > 0

            xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block=i)

            interpolated_scalar[block == i] = griddata(
                np.vstack((xi_[mask], eta_[mask])).T,
                scalar[mask],
                np.vstack((xi[block == i], eta[block == i])).T,
                **kwargs,
            )

        return interpolated_scalar
