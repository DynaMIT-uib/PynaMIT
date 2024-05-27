""" 
Python implementation of cubed sphere projection and differential
calculus, largely based on:

    Liang Yin, Chao Yang, Shi-Zhuang Ma, Ji-Zu Huang, Ying Cai, Parallel
    numerical simulation of the thermal convection in the Earth's outer
    core on the cubed-sphere, Geophysical Journal International,
    Volume 209, Issue 3, June 2017, Pages 1934–1954,
    https://doi.org/10.1093/gji/ggx125

Cube block indices:
-------------------

::

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

    0 = I  : Equator
    1 = II : Equator 
    2 = III: Equator    
    3 = IV : Equator
    4 = V  : North pole
    5 = VI : South pole

"""


import numpy as np
from pynamit.cubedsphere import diffutils
from pynamit.cubedsphere import arrayutils
import os
from scipy.special import binom
from scipy.sparse import coo_matrix
from scipy.interpolate import griddata
d2r = np.pi / 180

datapath = os.path.dirname(os.path.abspath(__file__)) + '/data/' # for coastlines


class CSProjection(object):
    def __init__(self, N = None):
        """ Set up cubed sphere projection.

        Parameters
        ----------
        N: int, optional
            Set to an integer to calculate a set of grid points that
            correspond to the centers of the cells in a ``6 x N x N``
            grid. The ``xi`` [rad], ``eta`` [rad], ``theta`` [deg] and
            ``phi`` [deg] coordinates will be calculated along with the
            cube block number and cell areas for a unit sphere. All arrays
            will be flat ``6*(N-1)**2`` arrays.

        """

        if N is not None: # Calculate grid arrays
            k, i, j = self.get_gridpoints(N)
            k, i, j = k[:, :-1, :-1], i[:, :-1, :-1], j[:, :-1, :-1] # crop, since we only want cell centers
            self.arr_xi  = self.xi( i, N).flatten()
            self.arr_eta = self.eta(j, N).flatten()
            self.arr_block = k.flatten()
            _, self.arr_theta, self.arr_phi = self.cube2spherical(self.arr_xi, self.arr_eta, self.arr_block, deg = True)

            # calcualte area
            step = np.diff(self.xi(np.array([0, 1]), N))[0] # lenght of each cell side in xi/eta coords
            self.g = self.get_metric_tensor(self.arr_xi, self.arr_eta)
            self.detg = arrayutils.get_3D_determinants(self.g)
            self.unit_area = step**2 * np.sqrt(self.detg) # Eq. (20) in Yin



    def get_gridpoints(self, N, flat = False):
        """
        Return `k`, `i`, and `j` corresponding to grid resolution `N`.

        Parameters
        ----------
        N: int
            Number of grid cell edges (``number of grid cells + 1``) in
            each direction per block.
        flat: bool, optional
            Set to ``True`` to return flat grid points.

        Returns
        -------
        k: array
            Array of block indices.
        i: array
            Array of indices referring to ``xi`` direction (from ``0`` to
            ``N-1``).
        j: array
            Array of indices referring to ``eta`` direction (from ``0`` to
            ``N-1``).

        """

        k, i, j = np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing = 'ij')
        if flat:
            return k.flatten(), i.flatten(), j.flatten()
        else:
            return k, i, j


    def xi(self, i, N):
        """
        Calculate the `xi` value for a given grid index `i` and grid
        resolution `N`.
        
        ``xi(0)`` is ``-pi/4`` and ``xi(N-1)`` is ``pi/4``.

        Parameters
        ----------
        i: array-like
            Array of indices (could be non-integer).
        N: int
            Grid resolution.

        Returns
        -------
        xi: array
            Array of `xi` values.

        """
        if not isinstance(N, (int, np.integer)):
            print('Warning: N is integer in the intended applications, did you make a mistake?')

        return(-np.pi / 4 + i * np.pi / 2 / (N-1))



    def eta(self, j, N):
        """
        Calculate the `eta` value for a given grid index `j` and grid
        resolution `N`.
        
        ``eta(0)`` is ``-pi/4`` and ``eta(N-1)`` is ``pi/4``.

        Parameters
        ----------
        j: array-like
            Array of indices (could be non-integer).
        N: int
            Grid resolution.

        Returns
        -------
        eta: array
            Array of `eta` values.

        Note
        ----
        This function is a copy of ``xi()``, it's just included because it
        makes code more readable.

        """
        if not isinstance(N, (int, np.integer)):
            print('Warning: N is integer in the intended applications, did you make a mistake?')

        return(-np.pi / 4 + j * np.pi / 2 / (N-1))



    def get_delta(self, xi, eta):
        """
        Calculate the delta parameter::

            delta = 1 + tan(xi)**2 + tan(eta)**2

        Parameters
        ----------
        xi: array-like
            Array of `xi` values.
        eta: array-like
            Array of `eta` values.

        Returns
        -------
        delta: array
            Array of `delta` values, shape determined by input according
            to broadcasting rules.

        """
        
        xi, eta = np.broadcast_arrays(xi, eta)

        return(1 + np.tan(xi)**2 + np.tan(eta)**2)


    def get_metric_tensor(self, xi, eta, r = 1, covariant = True):
        """ Calculate metric tensor `g`.

        Calculate the elements of a the metric tensor for each input
        point, using equation (12) of Yin et al. (2017).
    
        Parameters
        ----------
        xi: array-like
            Array of `xi` values.
        eta: array-like
            Array of `eta` values.
        r : array-like, optional, default = 1
            Array of radius values.
        covariant: bool, optional, default = True
            If ``True`` (default), return covariant components of the
            metric tensor. If ``False``, return contravariant components.

        Returns
        -------
        g : array
            ``(N, 3, 3)`` array of metric tensor elements. `g` will have
            shape ``(N, 3, 3)``, where the last two dimensions refer to
            column and row of the matrix, and ``N`` is the number of input
            points, using broadcasting.

        """

        xi, eta, r = map(np.ravel, np.broadcast_arrays(xi, eta, r)) # broadcast and flatten
        delta = self.get_delta(xi, eta)

        g = np.empty((xi.size, 3, 3))
        g[:, 0, 0] =  r**2 / (np.cos(xi)**4 * np.cos(eta)**2 * delta**2)
        g[:, 0, 1] = -r**2 * np.tan(xi) * np.tan(eta) / (np.cos(xi)**2 * np.cos(eta)**2 * delta **2)
        g[:, 0, 2] =  0
        g[:, 1, 0] = -r**2 * np.tan(xi) * np.tan(eta) / (np.cos(xi)**2 * np.cos(eta)**2 * delta **2)
        g[:, 1, 1] =  r**2 /    (np.cos(xi)**2 * np.cos(eta)**4 * delta**2)
        g[:, 1, 2] =  0
        g[:, 2, 0] =  0
        g[:, 2, 1] =  0
        g[:, 2, 2] =  1

        if covariant:
            return(g) # return covariant components
        else:
            return(arrayutils.invert_3D_matrices(g)) # return contravariant components



    def cube2cartesian(self, xi, eta, r = 1, block = 0):
        """
        Calculate Cartesian `x`, `y`, `z` coordinates (ECEF) of given
        points.

        Output will have same unit as `r`.

        Calculations based on equations from Appendix A of Yin et al.
        (2017).

        Parameters
        ----------
        xi: array-like
            Array of `xi` values.
        eta: array-like
            Array of `eta` values.
        r : array-like, optional, default = 1
            Array of radii.
        block: array-like, optional, default = 0
            Array of block indices.

        Returns
        -------
        x: array
            Array of `x` values, shape determined by input according to
            broadcasting rules.
        y: array
            Array of `y` values, shape determined by input according to
            broadcasting rules.
        z: array
            Array of `z` values, shape determined by input according to
            broadcasting rules.

        """

        xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)
        delta = self.get_delta(xi, eta)
        x, y, z = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)

        # block 0 (A2)
        iii = block == 0
        x[iii] =  r[iii]                    / np.sqrt(delta[iii])
        y[iii] =  r[iii] * np.tan(xi [iii]) / np.sqrt(delta[iii])
        z[iii] =  r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # block 1 (A6)
        iii = block == 1
        x[iii] = -r[iii] * np.tan(xi [iii]) / np.sqrt(delta[iii])
        y[iii] =  r[iii]                    / np.sqrt(delta[iii])
        z[iii] =  r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # block 2 (A10)
        iii = block == 2
        x[iii] = -r[iii]                    / np.sqrt(delta[iii])
        y[iii] = -r[iii] * np.tan(xi [iii]) / np.sqrt(delta[iii])
        z[iii] =  r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # block 3 (A14)
        iii = block == 3
        x[iii] =  r[iii] * np.tan(xi [iii]) / np.sqrt(delta[iii])
        y[iii] = -r[iii]                    / np.sqrt(delta[iii])
        z[iii] =  r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        # block 4 (A18)
        iii = block == 4
        x[iii] = -r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        y[iii] =  r[iii] * np.tan(xi [iii]) / np.sqrt(delta[iii])
        z[iii] =  r[iii]                    / np.sqrt(delta[iii])
        # block 5 (A22)
        iii = block == 5
        x[iii] =  r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        y[iii] =  r[iii] * np.tan(xi [iii]) / np.sqrt(delta[iii])
        z[iii] = -r[iii]                    / np.sqrt(delta[iii])

        return(x, y, z)


    def cube2spherical(self, xi, eta, block, r = 1, deg = False):
        """
        Calculate spherical (`r`, `theta`, `phi`) coordinates of given
        points.

        Parameters
        ----------
        xi: array-like
            Array of `xi` values.
        eta: array-like
            Array of `eta` values.
        block: array-like, default = 0
            Array of block indices.
        r : array-like, optional, default = 1
            Array of radii.
        deg : bool, optional, default = False
            Set to `True` if you want results in degrees, default is
            radians.

        Returns
        -------
        r: array
            Array of `r` values, shape determined by input according to
            broadcasting rules.
        theta: array
            Array of colatitude [radians] values, shape determined by
            input according to broadcasting rules.
        phi: array
            Array of longitude [radians] values, shape determined by
            input according to broadcasting rules.

        """ 
        xi, eta = np.float64(xi), np.float64(eta)
        xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)

        x, y, z = self.cube2cartesian(xi, eta, r, block)
        phi   = np.arctan2(y, x)
        theta = np.arccos(z / r)

        if deg:
            phi, theta = np.rad2deg(phi), np.rad2deg(theta)

        return(r, theta, phi)


    def get_Pc(self, xi, eta, r = 1, block = 0, inverse = False):
        """
        Calculate elements of transformation matrix `Pc` at all input
        points.

        The `Pc` matrix transforms Cartesian components ``(ux, uy, uz)``
        to contravariant components in a cubed sphere coordinate system::
        
            |u1| = |P00 P01 P02| |ux|
            |u2| = |P10 P11 P12| |uy|
            |u3| = |P20 P21 P22| |uz|

        The output, `Pc`, will have shape ``(N, 3, 3)``.

        Calculations based on equations from Appendix A of Yin et al.
        (2017), with similar notation.

        Parameters
        ----------
        xi: array-like
            Array of `xi` values.
        eta: array-like
            Array of `eta` values.
        r : array-like, optional, default = 1
            Array of radii.
        block: array-like, optional, default = 0
            Array of block indices.
        inverse: bool, optional
            Set to ``True`` if you want the inverse transformation matrix.

        Returns
        -------
        Pc: array
            Transformation matrices `Pc`, one for each point described by
            the input parameters (using broadcasting rules). For ``N``
            such points, `Pc` will have shape ``(N, 3, 3)``, where the
            last two dimensions refer to column and row of the matrix.

        """ 

        xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block)) # broadcast and flatten
        delta = self.get_delta(xi, et)
        Pc = np.empty((delta.size, 3, 3))

        rsec2xi = r / np.cos(xi)**2
        rsec2et = r / np.cos(et)**2

        # block 0
        iii = block == 0
        Pc[iii, 0, 0] = -np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 1] =  np.sqrt(delta[iii])                   / rsec2xi[iii]
        Pc[iii, 0, 2] =  0
        Pc[iii, 1, 0] = -np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 1] =  0
        Pc[iii, 1, 2] =  np.sqrt(delta[iii])                   / rsec2et[iii]
        Pc[iii, 2, 0] =  1               / np.sqrt(delta[iii])
        Pc[iii, 2, 1] =  np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] =  np.tan(et[iii]) / np.sqrt(delta[iii])

        # block 1
        iii = block == 1
        Pc[iii, 0, 0] = -np.sqrt(delta[iii])                   / rsec2xi[iii]
        Pc[iii, 0, 1] = -np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] =  0
        Pc[iii, 1, 0] =  0
        Pc[iii, 1, 1] = -np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 2] =  np.sqrt(delta[iii])                   / rsec2et[iii]
        Pc[iii, 2, 0] = -np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] =  1               / np.sqrt(delta[iii])
        Pc[iii, 2, 2] =  np.tan(et[iii]) / np.sqrt(delta[iii])

        # block 2
        iii = block == 2
        Pc[iii, 0, 0] =  np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 1] = -np.sqrt(delta[iii])                   / rsec2xi[iii]
        Pc[iii, 0, 2] =  0
        Pc[iii, 1, 0] =  np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 1] =  0
        Pc[iii, 1, 2] =  np.sqrt(delta[iii])                   / rsec2et[iii]
        Pc[iii, 2, 0] = -1               / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = -np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] =  np.tan(et[iii]) / np.sqrt(delta[iii])

        # block 3
        iii = block == 3
        Pc[iii, 0, 0] =  np.sqrt(delta[iii])                   / rsec2xi[iii]
        Pc[iii, 0, 1] =  np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 0, 2] =  0
        Pc[iii, 1, 0] =  0
        Pc[iii, 1, 1] =  np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 1, 2] =  np.sqrt(delta[iii])                   / rsec2et[iii]
        Pc[iii, 2, 0] =  np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] = -1               / np.sqrt(delta[iii])
        Pc[iii, 2, 2] =  np.tan(et[iii]) / np.sqrt(delta[iii])

        # block 4
        iii = block == 4
        Pc[iii, 0, 0] =  0
        Pc[iii, 0, 1] =  np.sqrt(delta[iii])                   / rsec2xi[iii]
        Pc[iii, 0, 2] = -np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 1, 0] = -np.sqrt(delta[iii])                   / rsec2et[iii]
        Pc[iii, 1, 1] =  0
        Pc[iii, 1, 2] = -np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] = -np.tan(et[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] =  np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] =  1               / np.sqrt(delta[iii])

        # block 5
        iii = block == 5
        Pc[iii, 0, 0] =  0
        Pc[iii, 0, 1] =  np.sqrt(delta[iii])                   / rsec2xi[iii]
        Pc[iii, 0, 2] =  np.sqrt(delta[iii]) * np.tan(xi[iii]) / rsec2xi[iii]
        Pc[iii, 1, 0] =  np.sqrt(delta[iii])                   / rsec2et[iii]
        Pc[iii, 1, 1] =  0
        Pc[iii, 1, 2] =  np.sqrt(delta[iii]) * np.tan(et[iii]) / rsec2et[iii]
        Pc[iii, 2, 0] =  np.tan(et[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 1] =  np.tan(xi[iii]) / np.sqrt(delta[iii])
        Pc[iii, 2, 2] = -1               / np.sqrt(delta[iii])

        if inverse:
            return(arrayutils.invert_3D_matrices(Pc))
        else:
            return(Pc)


    def get_Ps(self, xi, eta, r = 1, block = 0, inverse = False):
        """
        Calculate elements of transformation matrix `Ps` at all input
        points.

        The `Ps` matrix transforms vector components
        ``(u_east, u_north, u_r)`` to contravariant components in a cubed
        sphere coordinate system::
        
            |u1| = |P00 P01 P02| |u_east|
            |u2| = |P10 P11 P12| |u_north| 
            |u3| = |P20 P21 P22| |u_r|

        The output, `Ps`, will have shape ``(N, 3, 3)``.

        Calculations based on equations from Appendix A of Yin et al.
        (2017), with similar notation, except that ``lambda`` and ``phi``
        is replaced with ``east`` and ``north`` (here, ``phi`` means
        longitude, and not latitude as in the Yin paper).

        Parameters
        ----------
        xi: array-like
            Array of `xi` values.
        eta: array-like
            Array of `eta` values.
        r : array-like, optional, default = 1
            Array of radii.
        block: array-like, optional, default = 0
            Array of block indices.
        inverse: bool, optional
            Set to ``True`` if you want the inverse transformation matrix.

        Returns
        -------
        Ps: array
            Transformation matrices `Ps`, one for each point described by
            the input parameters (using broadcasting rules). For ``N``
            such points, `Ps` will have shape ``(N, 3, 3)``, where the
            last two dimensions refer to column and row of the matrix.

        """ 

        xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block)) # broadcast and flatten
        delta = self.get_delta(xi, et)
        Ps = np.empty((delta.size, 3, 3))

        # block 0
        iii = block == 0
        Ps[iii, 0, 0] =  1
        Ps[iii, 0, 1] =  0
        Ps[iii, 0, 2] =  0
        Ps[iii, 1, 0] =  np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] =  np.cos(xi[iii]) * np.sin(et[iii])**2 + np.cos(et[iii])**2 / np.cos(xi[iii])
        Ps[iii, 1, 2] =  0
        Ps[iii, 2, 0] =  0
        Ps[iii, 2, 1] =  0
        Ps[iii, 2, 2] =  1

        # block 1
        iii = block == 1
        Ps[iii, 0, 0] =  1
        Ps[iii, 0, 1] =  0
        Ps[iii, 0, 2] =  0
        Ps[iii, 1, 0] =  np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] =  np.cos(xi[iii]) * np.sin(et[iii])**2 + np.cos(et[iii])**2 / np.cos(xi[iii])
        Ps[iii, 1, 2] =  0
        Ps[iii, 2, 0] =  0
        Ps[iii, 2, 1] =  0
        Ps[iii, 2, 2] =  1

        # block 2
        iii = block == 2
        Ps[iii, 0, 0] =  1
        Ps[iii, 0, 1] =  0
        Ps[iii, 0, 2] =  0
        Ps[iii, 1, 0] =  np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] =  np.cos(xi[iii]) * np.sin(et[iii])**2 + np.cos(et[iii])**2 / np.cos(xi[iii])
        Ps[iii, 1, 2] =  0
        Ps[iii, 2, 0] =  0
        Ps[iii, 2, 1] =  0
        Ps[iii, 2, 2] =  1

        # block 3
        iii = block == 3
        Ps[iii, 0, 0] =  1
        Ps[iii, 0, 1] =  0
        Ps[iii, 0, 2] =  0
        Ps[iii, 1, 0] =  np.tan(xi[iii]) * np.sin(et[iii]) * np.cos(et[iii])
        Ps[iii, 1, 1] =  np.cos(xi[iii]) * np.sin(et[iii])**2 + np.cos(et[iii])**2 / np.cos(xi[iii])
        Ps[iii, 1, 2] =  0
        Ps[iii, 2, 0] =  0
        Ps[iii, 2, 1] =  0
        Ps[iii, 2, 2] =  1

        # block 4
        iii = block == 4
        Ps[iii, 0, 0] = -np.cos(xi[iii])**2 * np.tan(et[iii])
        Ps[iii, 0, 1] = -delta[iii] * np.tan(xi[iii]) * np.cos(xi[iii])**2 / np.sqrt(delta[iii] - 1)
        Ps[iii, 0, 2] =  0
        Ps[iii, 1, 0] =  np.cos(et[iii])**2 * np.tan(xi[iii]) 
        Ps[iii, 1, 1] = -delta[iii] * np.tan(et[iii]) * np.cos(et[iii])**2 / np.sqrt(delta[iii] - 1)
        Ps[iii, 1, 2] =  0
        Ps[iii, 2, 0] =  0
        Ps[iii, 2, 1] =  0
        Ps[iii, 2, 2] =  1

        # block 5
        iii = block == 5
        Ps[iii, 0, 0] =  np.cos(xi[iii])**2 * np.tan(et[iii])
        Ps[iii, 0, 1] =  delta[iii] * np.tan(xi[iii]) * np.cos(xi[iii])**2 / np.sqrt(delta[iii] - 1)
        Ps[iii, 0, 2] =  0
        Ps[iii, 1, 0] = -np.cos(et[iii])**2 * np.tan(xi[iii]) 
        Ps[iii, 1, 1] =  delta[iii] * np.tan(et[iii]) * np.cos(et[iii])**2 / np.sqrt(delta[iii] - 1)
        Ps[iii, 1, 2] =  0
        Ps[iii, 2, 0] =  0
        Ps[iii, 2, 1] =  0
        Ps[iii, 2, 2] =  1

        if inverse:
            return(arrayutils.invert_3D_matrices(Ps))
        else:
            return(Ps)

    def get_Qij(self, xi, eta, block_i, block_j):
        """
        Calculate matrix `Qij` that transforms contravariant vector
        components from `block_i` to `block_j`.

        Calculations are done via transformation to spherical coordinates,
        as suggested by Yin et al. See equations (66) and (67) in their
        paper.

        It works like this, where ``(u1, u2, u_3)`` refer to
        contravariant vector components::

            |u1_j|      |u1_i|
            |u2_j| = Qij|u2_i|
            |u3_j|      |u3_i|

        
        Parameters
        ----------
        xi: array-like
            Array of `xi` values on block given by `block_i`.
        eta: array-like
            Array of `eta` values on block given by `block_i`.
        block_i: array-like, optional
            Indices of block(s) from which to transform vector components.
        block_j: array-like, optional
            Indices of block(s) to which to transform vector components.

        Returns
        -------
        Qij: array
            Transformation matrices `Qij`, one for each point described by
            the input parameters (using broadcasting rules). For ``N``
            such points, `Qij` will have shape ``(N, 3, 3)``, where the
            last two dimensions refer to column and row of the matrix.

        """

        xi_i, eta_i, block_i, block_j = map(np.ravel, np.broadcast_arrays(xi, eta, block_i, block_j)) # broadcast and flatten

        Psi_inv = self.get_Ps(xi_i, eta_i, r = 1, block = block_i, inverse = True)

        # find the xi, eta coordinates on block j
        r, theta, phi = self.cube2spherical(xi_i, eta_i, r = 1, block = block_i, deg = True)
        xi_j, eta_j, _ = self.geo2cube(phi, 90 - theta, block = block_j)

        # calculate Ps relative to block j
        Psj = self.get_Ps(xi_j, eta_j, r = 1, block = block_j)

        # multiply each of the N matrices to get Qij:
        Qij = np.einsum('nij, njk -> nik', Psj, Psi_inv)

        return(Qij)


    def get_Q(self, lat, r, inverse = False):
        """
        Calculate the matrices that convert from not normalized spherical
        components to normalized spherical vector components::

            |u_east_normalized |    |u_east |
            |u_north_normalized| = Q|u_north|
            |u_r_normalized    |    |u_r    |

        Calculations based on Yin et al. 2017 (equations after A25).

        Parameters
        ----------
        lat: array
            Array of latitudes [degrees].
        r: array
            Array of radii.
        inverse: bool, optional
            Set to ``True`` if you want the inverse transformation matrix.

        Returns
        -------
        Q: array
            ``(N, 3, 3)`` array, where ``N`` is the size implied by
            broadcasting the input.

        """
        lat, r = map(np.ravel, np.broadcast_arrays(lat, r))

        Q = np.zeros((lat.size, 3, 3), dtype = np.float64)
        Q[:, 0, 0] = r * np.cos(np.deg2rad(lat))
        Q[:, 1, 1] = r
        Q[:, 2, 2] = 1

        if inverse:
            return(arrayutils.invert_3D_matrices(Q))
        else:
            return(Q)


    def get_Diff(self, N, coordinate = 'xi', Ns = 1, Ni = 4, order = 1):
        """
        Calculate matrix that differentiates a scalar field, defined on a
        ``(6, N, N)`` grid, with respect to ``xi`` or ``eta``.

        Parameters
        ----------
        N: int
            Number of grid cells in each dimension on each block.
        coordinate: string, {'xi', 'eta', 'both'}, default = 'xi'
            Which coordinate to differentiate with respect to.
        Ns: int, optional, default = 1
            Differentiation stencil size. Default gives first order
            central difference.
        Ni: int, optional
            Number of points to use for interpolation for points in the
            stencil that fall on non-integer grid points on neighboring
            block.
        order: int, optional, default = 1
            Order of differentiation. Default gives first order
            derivative. Make sure that ``Ns >= order``.

        Returns
        -------
        D: sparse matrix
            Sparse ``(6*N*N, 6*N*N)`` matrix that calculates the
            derivative of a scalar field with respect to ``xi``, ``eta``,
            or both (in which case two matrices will be returned), as
            ``derivative = D.dot(f)``, where ``f`` is the scalar field.

        """
        if coordinate not in ['xi', 'eta', 'both']:
            raise ValueError('coordinate must be either "xi", "eta", or "both". Not {}'.format(coordinate))

        if Ns < order:
            raise ValueError('Ns must be >= order. You gave {} and {}'.format(Ns, order))

        shape = (6, N, N)
        size = 6 * N * N
        h = self.xi(1, N) - self.xi(0, N) # step size between each grid cell

        k, i, j = map(np.ravel, np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing = 'ij'))

        # set up differentiation stencil
        stencil_points  = np.hstack((np.r_[-Ns:0], np.r_[1:Ns + 1]))
        Nsp = len(stencil_points)
        stencil_weight = diffutils.stencil(stencil_points, order = 1, h = h) # 1st order differentiation

        i_diff           = np.hstack([i + _ for _ in stencil_points])
        j_diff           = np.hstack([j + _ for _ in stencil_points])
        k_const, i_const, j_const = np.tile(k, Nsp), np.tile(i, Nsp), np.tile(j, Nsp)
        weights  = np.repeat(stencil_weight, size)

        rows = np.tile(np.ravel_multi_index((k, i, j), shape), Nsp)
        if coordinate in ['xi', 'both']:
            Dxi = self.get_interpolation_matrix(k_const, i_diff , j_const, N, Ni, rows = rows, weights = weights)
        if coordinate in ['eta', 'both']:
            Deta = self.get_interpolation_matrix(k_const, i_const, j_diff , N, Ni, rows = rows, weights = weights)

        if coordinate == 'both':
            return(Dxi, Deta)
        if coordinate == 'xi':
            return(Dxi)
        if coordinate == 'eta':
            return(Deta)


    def get_interpolation_matrix(self, k, i, j, N, Ni, weights = None, rows = None):
        """
        Calculate a sparse matrix D that interpolates from grid points in
        a ``(6, N, N)`` grid to the indices (`k`, `i`, `j`). 

        `D` will have ``6*N**2`` columns that refer to the ``(6, N, N)``
        grid points, spanning the 6 blocks in the cubed sphere, with
        duplicate points on the boundaries. 

        Parameters
        ----------
        k: array-like
            Integer indices that refer to cube block. Must be ``>= 0`` and
            ``<= 5``. Will be flattened.
        i: array-like
            Integer indices that refer to the ``xi``-direction (but can be
            negative or ``>= N``). Will be flattened.
        j: array-like
            Integer indices that refer to the ``eta``-direction (but can
            be negative or ``>= N``). Will be flattened.
        N: int
            Number of grid points.
        Ni: int
            Number of interpolation points. Must be ``<= N`` (usually 4 is
            ok).
        weights: array-like, optional
            If different values of `k`, `i`, `j` are assigned to the same
            row, the corresponding element will have value 1 (or whatever
            the interpolation dictates) unless weights is specified. For
            differentiation, use weights to specify the stencil
            coefficients.
        rows: array-like, optional
            The row index of each element in `k`, `i`, `j`. Different
            elements of `k`, `i`, `j` can be put in the same row. If not
            specified, each element in `k`, `i`, `j` will be given its own
            row. 

        Returns
        -------
        D: sparse matrix
            ``(rows.max() + 1 by 6*N*N)`` matrix that, when multiplied by
            a vector containing a scalar field on the ``6*N*N`` grid
            points, produces interpolated values at the given grid points.
            The grid points may be outside the cube blocks, for example
            they can be negative (actually that's the point, if not this
            function wouldn't be needed).

        """

        if Ni > N:
            raise ValueError('Ni must be <= N')
        k, i, j = map(np.ravel, [k, i, j])

        shape = (6, N, N)
        size = 6 * N**2

        if rows is None:
            rows = np.arange(k.size)

        if weights is None:
            weights = np.ones(k.size)
        weights = weights / Ni

        h = self.xi(1, N) - self.xi(0, N) # step size between each grid cell

        cols = np.full(k.size, -1, dtype = np.int64) # array that will contain column indices

        # find new indices that do not exceed block dimensions (but are possibly floats):
        xi, eta = self.xi(i, N), self.eta(j, N)
        r, theta, phi = self.cube2spherical(xi, eta, k, r = 1., deg = True)
        new_xi, new_eta, new_k = self.geo2cube(phi, 90 - theta)
        new_i, new_j = new_xi / h + (N - 1) / 2, new_eta / h + (N - 1) / 2
        
        # all index pairs should have at least one integer in a uniform cubed sphere grid:
        assert np.all((np.isclose(new_i - np.rint(new_i), 0) | np.isclose(new_j - np.rint(new_j), 0)))

        # Fill in column indices for the index pairs that are both integers:
        ii_integers = np.isclose(new_i - np.rint(new_i), 0) & np.isclose(new_j - np.rint(new_j), 0)
        cols[ii_integers] = np.ravel_multi_index((new_k[ii_integers], np.rint(new_i[ii_integers]).astype(np.int64), np.rint(new_j[ii_integers]).astype(np.int64)), shape)

        # The rest of the index pairs need interpolation. Find the indices where it is needed:
        i_is_float = ~np.isclose(np.rint(new_i) - new_i, 0)
        j_is_float = ~np.isclose(np.rint(new_j) - new_j, 0)

        assert sum(i_is_float & j_is_float) == 0 # no new index pair should have two floats
        assert sum(i_is_float | j_is_float) == sum(cols == -1) # all missing columns match indices where i or j are float

        j_floats = new_j[j_is_float].reshape((-1, 1))
        i_floats = new_i[i_is_float].reshape((-1, 1))

        # define the (integer) points which will be used to interpolate:
        interpolation_points = np.arange(Ni).reshape((1, -1))
        j_interpolation_points = arrayutils.constrain_values(interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2 - 1, 0, N - 1, axis = 1)
        i_interpolation_points = arrayutils.constrain_values(interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2 - 1, 0, N - 1, axis = 1)

        # calculate barycentric weights wj (Berrut & Trefethen, 2004):
        j_distances = j_floats - j_interpolation_points
        i_distances = i_floats - i_interpolation_points
        w = (-1)**interpolation_points * binom(Ni - 1, interpolation_points)
        w_i = w / i_distances / np.sum(w / i_distances, axis = 1).reshape( (-1, 1) )
        w_j = w / j_distances / np.sum(w / j_distances, axis = 1).reshape( (-1, 1) )

        # expand the column, row, and weight arrays to allow for interpolation weights (just duplicating where no interpolation is required)
        stacked_weights = np.tile(weights, (Ni, 1)).T 
        stacked_cols    = np.tile(cols   , (Ni, 1)).T
        stacked_rows    = np.tile(rows   , (Ni, 1)).T

        # specify the columns and weights where interpolation is required:
        stacked_cols[   i_is_float] = np.ravel_multi_index((np.tile(new_k[i_is_float], (Ni, 1)).T, i_interpolation_points                                , np.rint(np.tile(new_j[i_is_float], (Ni, 1))).astype(np.int64).T), shape)
        stacked_cols[   j_is_float] = np.ravel_multi_index((np.tile(new_k[j_is_float], (Ni, 1)).T, np.rint(np.tile(new_i[j_is_float], (Ni, 1))).astype(np.int64).T, j_interpolation_points                                ), shape)
        stacked_weights[i_is_float] = stacked_weights[i_is_float] * w_i * Ni
        stacked_weights[j_is_float] = stacked_weights[j_is_float] * w_j * Ni

        D = coo_matrix((stacked_weights.flatten(), (stacked_rows.flatten(), stacked_cols.flatten()) ), shape = (rows.max() + 1, size))
        D.count_nonzero() # get rid of duplicates (I think... maybe this doesn't do anything)

        return D


    def block(self, lon, lat):
        """ Find the block that points belong to.

        Find which block the input coordinates belong to.

        Parameters
        ----------
        lon: array
            Geocentric longitude(s) [deg] to convert to cube coords.
        lat: array:
            Geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        block: array
            Indices of the block that (`lon`, `lat`) is on
            ``(I = 0, II = 1,...)``.

        """
        lon, lat = np.broadcast_arrays(lon, lat)
        lat, lon = lat.flatten(), lon.flatten()

        th, ph = np.deg2rad(90 - lat), np.deg2rad(lon)

        xyz = np.vstack((np.cos(ph) * np.sin(th), np.sin(th) * np.sin(ph), np.cos(th)))
        xyz[np.isclose(xyz, 0)] += 1e-3 # to avoid division by zero problems

        # calculate how much xyz must be extended to intersect the various surfaces
        t = {}
        t[0] =  1 / xyz[0] # 'I'  
        t[1] =  1 / xyz[1] # 'II' 
        t[2] = -1 / xyz[0] # 'III'
        t[3] = -1 / xyz[1] # 'IV' 
        t[4] =  1 / xyz[2] # 'V'  
        t[5] = -1 / xyz[2] # 'VI' 

        norms = {}
        for key in t.keys():
            norms[key] = np.linalg.norm(xyz * t[key], axis = 0)
            norms[key][t[key] < 0] += 10 # increase norm of vectors with negative t

        return np.argmin(np.vstack([norms[i] for i in range(6)]), axis = 0)


    def geo2cube(self, lon, lat, block = None):
        """
        Convert from geocentric coordinates to cube coords (`xi`, `eta`).
        
        Input parameters must have same shape. Output will have same
        shape.

        Parameters
        ----------
        lon: array
            Geocentric longitude(s) [deg] to convert to cube coords.
        lat: array
            Geocentric latitude(s) [deg] to convert to cube coords.
        block: array-like, optional
            Option to specify cube block. If ``None``, it will be
            calculated. If specified, be careful because the function will
            map points at opposite side of the sphere to specified block.

        Returns
        -------
        xi: array
            `xi`, as defined in Ronchi et al. Unit is radians.
        eta: array
            `eta`, as defined in Ronchi et al. Unit is radians.
        block: array
            Index of the block that `xi`, `eta` belongs to.

        """

        lon, lat = np.broadcast_arrays(lon, lat)
        shape = lon.shape
        N = lon.size

        # find the correct block for each point
        if block is None:
            block = self.block(lon, lat)
        else:
            block = block * np.ones_like(lat)

        block, lon, lat = block.flatten(), lon.flatten(), lat.flatten()

        # prepare parameters
        X, Y, xi, eta = np.empty(N), np.empty(N), np.empty(N), np.empty(N)

        # calculate X and Y according to Ronchi et al., equations 1->
        theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)
        X[block == 0] =      np.tan(phi[block == 0])
        X[block == 1] = -1 / np.tan(phi[block == 1])
        X[block == 2] =      np.tan(phi[block == 2])
        X[block == 3] = -1 / np.tan(phi[block == 3])
        X[block == 4] =  np.tan(theta[block == 4]) * np.sin(phi[block == 4])
        X[block == 5] = -np.tan(theta[block == 5]) * np.sin(phi[block == 5])
 
        Y[block == 0] =  1 / (np.tan(theta[block == 0]) * np.cos(phi[block == 0]))
        Y[block == 1] =  1 / (np.tan(theta[block == 1]) * np.sin(phi[block == 1]))
        Y[block == 2] = -1 / (np.tan(theta[block == 2]) * np.cos(phi[block == 2]))
        Y[block == 3] = -1 / (np.tan(theta[block == 3]) * np.sin(phi[block == 3]))
        Y[block == 4] =     -np.tan(theta[block == 4]) * np.cos(phi[block == 4])
        Y[block == 5] =     -np.tan(theta[block == 5]) * np.cos(phi[block == 5])

        xi, eta = np.arctan(X), np.arctan(Y)

        print("block", np.linalg.norm(block))
        print("theta", np.linalg.norm(theta))
        print("phi", np.linalg.norm(phi))
        print("X", np.linalg.norm(X))
        print("Y", np.linalg.norm(Y))
        print("xi", np.linalg.norm(xi))
        print("eta", np.linalg.norm(eta))

        return xi.reshape(shape), eta.reshape(shape), block.reshape(shape)


    def get_projected_coastlines(self, resolution = '50m'):
        """ Generate coastlines in projected coordinates. """

        coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')
        for key in coastlines:
            lat, lon = coastlines[key]
            yield self.geo2cube(lon, lat)


    def interpolate_vector_components(self, u_east, u_north, u_r, theta, phi, theta_target, phi_target, **kwargs):
        """ Interpolate vector_components defined on theta, phi to given spherical coordinates
    
        Broadcasting rules apply for input and output separately


        Parameters
        ----------
        u_east: array
            array of eastward components
        u_north: array
            array northward components
        u_r: array
            array of radial components
        theta: array
            array of coordinates for components
        phi: array
            array of coordinates for vector components
        theta_target: array
            array of target coordinates. 
        phi_target: array
            array of target coordinates

        **kwargs
            passed to scipy.interpolate.griddata which performs the interpolation
            on each block

        Returns
        -------
        interpolated_vector: array
            3 x N vector of interpolated components, east, north, up
        """

        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        #xi, eta, block = np.broadcast_arrays(xi, eta, block)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        u_east, u_north, u_r, theta, phi = np.broadcast_arrays(u_east, u_north, u_r, theta, phi)
        u_east, u_north, u_r, theta, phi = u_east.flatten(), u_north.flatten(), u_r.flatten(), theta.flatten(), phi.flatten()

        # define vectors that point at all the original points:
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        # convert vector components to cubed sphere
        print("phi", np.linalg.norm(phi))
        print("90 - theta", np.linalg.norm(90 - theta))
        u_xi, u_eta, u_block = self.geo2cube(phi, 90 - theta)
        print("u_xi", np.linalg.norm(u_xi))
        print("u_eta", np.linalg.norm(u_eta))
        print("u_block", np.linalg.norm(u_block))
        Ps = self.get_Ps(u_xi, u_eta, r = 1, block = u_block)
        Q  = self.get_Q(90 - theta, r = 1, inverse = True)
        Ps_normalized = np.einsum('nij, njk -> nik', Ps, Q)
        u_vec_sph = np.vstack((u_east, u_north, u_r))
        u_vec = np.einsum('nij, nj -> ni', Ps_normalized, u_vec_sph.T).T

        print("u_vec", np.linalg.norm(u_vec))

        interpolated_u1 = np.empty_like(block, dtype = np.float64)
        interpolated_u2 = np.empty_like(block, dtype = np.float64)
        interpolated_u3 = np.empty_like(block, dtype = np.float64)


        # loop over blocks and interpolate on each block:
        for i in range(6):

            # express vector components with respect to block i:
            Qij = self.get_Qij(u_xi, u_eta, u_block, i)
            u_vec_i = np.einsum('nij, nj -> ni', Qij, u_vec.T).T

            print("Qij for block", i, np.linalg.norm(Qij))
            print("u_vec_i for block", i, np.linalg.norm(u_vec_i))

            # filter points whose position vectors have anti-parallel component to center of the block
            _, th, ph = self.cube2spherical(0, 0, i, deg = False)
            r0 = np.hstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).reshape((-1, 1))
            mask = np.sum(r0 * r, axis = 0) > 0

            print("mask for block", i, np.sum(mask))

            xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block = i)

            print ("xi_ for block", i, np.linalg.norm(xi_))
            print ("eta_ for block", i, np.linalg.norm(eta_))

            interpolated_u1[block == i] = griddata(np.vstack((xi_[mask], eta_[mask])).T, u_vec_i[0][mask], np.vstack((xi[block == i], eta[block == i])).T, **kwargs)
            interpolated_u2[block == i] = griddata(np.vstack((xi_[mask], eta_[mask])).T, u_vec_i[1][mask], np.vstack((xi[block == i], eta[block == i])).T, **kwargs)
            interpolated_u3[block == i] = griddata(np.vstack((xi_[mask], eta_[mask])).T, u_vec_i[2][mask], np.vstack((xi[block == i], eta[block == i])).T, **kwargs)

            print("interpolated_u1 for block", i, np.linalg.norm(interpolated_u1))
            print("interpolated_u2 for block", i, np.linalg.norm(interpolated_u2))
            print("interpolated_u3 for block", i, np.linalg.norm(interpolated_u3))

        # convert back to spherical:
        r_out, theta_out, phi_out = self.cube2spherical(xi, eta, block, deg = True)
        u = np.vstack((interpolated_u1, interpolated_u2, interpolated_u3))
        Q      = self.get_Q(90 - theta_out, r = 1, inverse = False)
        Ps_inv = self.get_Ps(xi, eta, r = 1, block = block, inverse = True)
        Ps_normalized_inv = np.einsum('nij, njk -> nik', Q, Ps_inv)
        u_east_int, u_north_int, u_r_int = np.einsum('nij, nj -> ni', Ps_normalized_inv, u.T).T

        return( np.vstack((u_east_int, u_north_int, u_r_int)) )



    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """ Interpolate vector_components defined on theta, phi to given spherical coordinates
    
        Broadcasting rules apply for input and output separately


        Parameters
        ----------
        scalar: array
            array of scalar values
        theta: array
            array of coordinates for components
        phi: array
            array of coordinates for vector components
        theta_target: array
            array of target coordinates. 
        phi_target: array
            array of target coordinates

        **kwargs
            passed to scipy.interpolate.griddata which performs the interpolation
            on each block

        Returns
        -------
        interpolated_scalar: array
            N-element array of interpolated components, east, north, up
        """

        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        #xi, eta, block = np.broadcast_arrays(xi, eta, block)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        scalar, theta, phi = np.broadcast_arrays(scalar, theta, phi)
        scalar, theta, phi = scalar.flatten(), theta.flatten(), phi.flatten()

        # define vectors that point at all the original points:
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        interpolated_scalar = np.empty_like(block, dtype = np.float64)


        # loop over blocks and interpolate on each block:
        for i in range(6):

            # filter points whose position vectors have anti-parallel component to center of the block
            _, th, ph = self.cube2spherical(0, 0, i, deg = False)
            r0 = np.hstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).reshape((-1, 1))
            mask = np.sum(r0 * r, axis = 0) > 0

            xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block = i)

            interpolated_scalar[block == i] = griddata(np.vstack((xi_[mask], eta_[mask])).T, scalar[mask], np.vstack((xi[block == i], eta[block == i])).T, **kwargs)

        return( interpolated_scalar )



csp = CSProjection()
