
"""Cubed sphere basis module.

This module contains the CSBasis class for representing the cubed sphere
basis.
"""

from __future__ import annotations
from typing import Any, Tuple, Optional, Callable, Dict, TYPE_CHECKING
import functools
import importlib
import numpy as np
import os
from scipy.special import binom
from scipy.sparse import coo_matrix

from pynamit.cubed_sphere import diffutils
from pynamit.math import arrayutils
from pynamit.math import cs_math
from pynamit.primitives.grid_basis import GridBasis
from pynamit.primitives.grid import Grid
from pynamit.interpolation import create_interpolator

if TYPE_CHECKING:
    from pynamit.cubed_sphere.grid import CubedSphereGrid
    from pynamit.simulation.geometry import Geometry

d2r = np.pi / 180
datapath = os.path.dirname(os.path.abspath(__file__)) + "/data/"


class CSBasis(GridBasis):
    """Class for representing cubed sphere bases.

    This module provides an implementation of the cubed sphere grid
    system following methods from Yin et al. (2017).
    """

    def __init__(self, N: int):
        """Initialize the cubed sphere basis."""

        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N % 2 != 0:
            raise ValueError("Cubed sphere grid dimension must be even")

        self.N = N
        k, i, j = self.get_gridpoints(N)

        # Initialize grid points (flattened)
        self.arr_xi: np.ndarray = self.xi(i[:, :-1, :-1] + 0.5, N).flatten()
        self.arr_eta: np.ndarray = self.eta(j[:, :-1, :-1] + 0.5, N).flatten()
        self.arr_block: np.ndarray = k[:, :-1, :-1].flatten()

        # Convert to spherical coordinates using inherited method
        _, self.arr_theta, self.arr_phi = cs_math.cube2spherical(
            self.arr_xi, self.arr_eta, self.arr_block, deg=True
        )
        
        # Initialize Grid object (Essential for GridBasis compatibility)
        self.grid = Grid(theta=self.arr_theta, phi=self.arr_phi)

        # Initialize optimized interpolator
        from pynamit.interpolation import CSInterpolator

        self._interpolator = CSInterpolator(N)

    @property
    def kind(self) -> str:
        return "CS"

    @property
    def size(self):
        """Number of grid points."""
        return self.index_length

    @property
    def theta(self):
        return self.arr_theta

    @property
    def phi(self):
        return self.arr_phi

    @functools.cached_property
    def g(self) -> np.ndarray:
        """Metric tensor."""
        return cs_math.get_metric_tensor(self.arr_xi, self.arr_eta)

    @functools.cached_property
    def sqrt_detg(self) -> np.ndarray:
        """Square root of determinant of the metric tensor."""
        return np.sqrt(arrayutils.get_3D_determinants(self.g))

    @functools.cached_property
    def unit_area(self) -> np.ndarray:
        """Area of each grid cell."""
        step = np.diff(self.xi(np.array([0, 1]), self.N))[0]
        return step**2 * self.sqrt_detg

    def get_gridpoints(self, N, flat=False):
        """Generate grid point indices for given resolution."""
        k, i, j = np.meshgrid(np.arange(6), np.arange(N + 1), np.arange(N + 1), indexing="ij")
        if flat:
            return k.flatten(), i.flatten(), j.flatten()
        else:
            return k, i, j

    def xi(self, i, N):
        """Calculate xi coordinate for grid index."""
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N < 1:
            raise ValueError("N must be at least 1")
        return -np.pi / 4 + i * np.pi / (2 * N)

    def eta(self, j, N):
        """Calculate eta coordinate for grid index."""
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N < 1:
            raise ValueError("N must be at least 1")
        return -np.pi / 4 + j * np.pi / (2 * N)

    # Methods get_delta through get_Q are inherited from GridBasis (removed here)

    def get_Diff(self, N, coordinate="xi", Ns=1, Ni=4, order=1):
        """Get scalar field differentiation matrix."""
        if coordinate not in ["xi", "eta", "both"]:
            raise ValueError(
                f'coordinate must be either "xi", "eta", or "both". Not {coordinate}.'
            )
        if Ns < order:
            raise ValueError("Ns must be >= order. You gave {} and {}".format(Ns, order))
        # if order != 1:
        #    raise NotImplementedError("Only first order differentiation is supported.")

        shape = (6, N, N)
        size = 6 * N * N
        h = self.xi(1, N) - self.xi(0, N)
        k, i, j = map(
            np.ravel, np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing="ij")
        )

        stencil_points = np.hstack((np.r_[-Ns:0], np.r_[1 : Ns + 1]))
        Nsp = len(stencil_points)
        stencil_weight = diffutils.stencil(stencil_points, order=order, h=h)

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
        """Get matrix for grid to cubed sphere interpolation."""
        if Ni > N:
            raise ValueError("Ni must be <= N")
        k, i, j = map(np.ravel, [k, i, j])
        shape, size = (6, N, N), 6 * N**2
        if rows is None:
            rows = np.arange(k.size)
        if weights is None:
            weights = np.ones(k.size)
        weights = weights / Ni
        h = self.xi(1, N) - self.xi(0, N)
        cols = np.full(k.size, -1, dtype=np.int64)

        xi, eta = self.xi(i + 0.5, N), self.eta(j + 0.5, N)
        r, theta, phi = cs_math.cube2spherical(xi, eta, k, r=1.0, deg=True)
        new_xi, new_eta, new_k = cs_math.geo2cube(phi, 90 - theta)
        new_i, new_j = new_xi / h + (N - 1) / 2, new_eta / h + (N - 1) / 2

        assert np.all(
            (np.isclose(new_i - np.rint(new_i), 0) | np.isclose(new_j - np.rint(new_j), 0))
        )
        ii_integers = np.isclose(new_i - np.rint(new_i), 0) & np.isclose(new_j - np.rint(new_j), 0)
        cols[ii_integers] = np.ravel_multi_index(
            (
                new_k[ii_integers],
                np.rint(new_i[ii_integers]).astype(np.int64),
                np.rint(new_j[ii_integers]).astype(np.int64),
            ),
            shape,
        )

        i_is_float = ~np.isclose(np.rint(new_i) - new_i, 0)
        j_is_float = ~np.isclose(np.rint(new_j) - new_j, 0)
        assert sum(i_is_float & j_is_float) == 0
        j_floats = new_j[j_is_float].reshape((-1, 1))
        i_floats = new_i[i_is_float].reshape((-1, 1))

        interpolation_points = np.arange(Ni).reshape((1, -1))
        j_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2 - 1, 0, N - 1, axis=1
        )
        i_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2 - 1, 0, N - 1, axis=1
        )

        j_distances = j_floats - j_interpolation_points
        i_distances = i_floats - i_interpolation_points
        w = (-1) ** interpolation_points * binom(Ni - 1, interpolation_points)
        w_i = w / i_distances / np.sum(w / i_distances, axis=1).reshape((-1, 1))
        w_j = w / j_distances / np.sum(w / j_distances, axis=1).reshape((-1, 1))

        stacked_weights = np.tile(weights, (Ni, 1)).T
        stacked_cols = np.tile(cols, (Ni, 1)).T
        stacked_rows = np.tile(rows, (Ni, 1)).T

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
        D.count_nonzero()
        return D

    def get_G(self, grid, derivative=None):
        """Get evaluation or differentiation matrix on the grid.

        Parameters
        ----------
        grid : object
            Grid object. Must match this basis's grid.
        derivative : str, optional
            'theta', 'phi', or None.

        Returns
        -------
        G : sparse matrix
            Evaluation or differentiation matrix.
        """
        # Relaxed check: if grid is self (BasisEvaluator(basis, basis)), it's CSBasis.
        # Or if it has 'kind' == 'CS' and same N.
        is_compatible = (
            (grid is self) or 
            (hasattr(grid, "kind") and grid.kind == "CS" and getattr(grid, "N", -1) == self.N)
        )
        if not is_compatible and hasattr(grid, "theta") and hasattr(grid, "phi"):
             # Check if coordinates match
             if grid == self.grid:
                   is_compatible = True
        
        if grid is not None and not is_compatible:
             # For now, only support evaluating on self (which is what BasisEvaluator expects for projections)
             raise NotImplementedError("CSBasis currently only supports get_G on its own grid.")

        N = self.N
        if derivative is None:
             from scipy.sparse import identity
             return identity(6 * N * N, format="csr")

        elif derivative in ["theta", "phi"]:
             # Get derivatives wrt logical coordinates
             Dxi, Deta = self.get_Diff(N, coordinate="both", Ns=1, Ni=4, order=1)
             
             # Calculate chain rule factors
             # d/dth = (dxi/dth) d/dxi + (deta/dth) d/deta
             # These are diagonal multiplication matrices
             coord_derivs = cs_math.get_coordinate_derivatives(
                 self.arr_xi, self.arr_eta, r=1.0, block=self.arr_block
             )
             dxi_dth, dxi_dph, deta_dth, deta_dph = coord_derivs

             from scipy.sparse import diags
             if derivative == "theta":
                 M_dxi_dth = diags(dxi_dth.flatten())
                 M_deta_dth = diags(deta_dth.flatten())
                 return M_dxi_dth.dot(Dxi) + M_deta_dth.dot(Deta)
             elif derivative == "phi":
                 M_dxi_dph = diags(dxi_dph.flatten())
                 M_deta_dph = diags(deta_dph.flatten())
                 return M_dxi_dph.dot(Dxi) + M_deta_dph.dot(Deta)
        
        else:
             raise ValueError(f"Unknown derivative: {derivative}")

    def laplacian(self, r=1.0):
        """Compute the Laplacian operator matrix on the sphere.

        Approximated using the strong form:
        nabla^2 V = (1/r^2 sin(theta)) * d/dtheta (sin(theta) dV/dtheta)
                  + (1/r^2 sin^2(theta)) * d^2V/dphi^2

        Returns
        -------
        L : scipy.sparse.spmatrix
            Sparse matrix representing the Laplacian operator.
        """
        import scipy.sparse
        G_th = self.get_G(self, derivative="theta")
        G_ph = self.get_G(self, derivative="phi")

        theta_rad = np.deg2rad(self.arr_theta)
        sin_th = np.sin(theta_rad)
        sin_sq_th = sin_th**2

        # Avoid division by zero at poles
        # For CS grid, valid points are usually away from poles, but handle safely
        epsilon = 1e-10
        sin_th_safe = np.where(np.abs(sin_th) < epsilon, epsilon, sin_th)
        sin_sq_th_safe = np.where(np.abs(sin_sq_th) < epsilon**2, epsilon**2, sin_sq_th)

        # Diagonal matrices for metric terms
        # 1/sin(theta)
        inv_sin_th = scipy.sparse.diags(1.0 / sin_th_safe)
        # sin(theta)
        sin_th_mat = scipy.sparse.diags(sin_th)
        # 1/sin^2(theta)
        inv_sin_sq_th = scipy.sparse.diags(1.0 / sin_sq_th_safe)

        # Term 1: (1/sin(theta)) * d/dtheta (sin(theta) d/dtheta)
        # = inv_sin_th @ G_th @ sin_th_mat @ G_th
        term1 = inv_sin_th @ G_th @ sin_th_mat @ G_th

        # Term 2: (1/sin^2(theta)) * d^2/dphi^2
        # = inv_sin_sq_th @ G_ph @ G_ph
        term2 = inv_sin_sq_th @ G_ph @ G_ph

        # Combine
        L = (term1 + term2) / (r**2)
        return L

    # Methods block, geo2cube, interpolate_scalar, interpolate_vector_components inherited

    def get_projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates."""
        coastlines = np.load(datapath + "coastlines_" + resolution + ".npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield cs_math.geo2cube(lon, lat)

    def interpolate_to_self(self, values, theta, phi, vector_type="scalar"):
        """Interpolate values to this basis's grid."""
        if vector_type == "scalar":
            return self.interpolate_scalar(values, theta, phi, self.arr_theta, self.arr_phi)
        elif vector_type == "tangential":
            u_east = values[1]
            u_north = -values[0]
            u_r = np.zeros_like(u_north)
            u_e, u_n, _ = self.interpolate_vector_components(
                u_east, u_north, u_r, theta, phi, self.arr_theta, self.arr_phi
            )
            return np.hstack((-u_n, u_e))
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def interpolate_scalar(self, val, th_src, ph_src, th_tgt, ph_tgt):
        """Spherical interpolation for scalars."""
        # Use optimized interpolator if source matches this basis
        # Wrap inputs in Grid for comparison logic (leveraging fast caching)
        src_grid = Grid(theta=th_src, phi=ph_src)
        if src_grid == self.grid:
            return self._interpolator.interpolate_scalar(val, th_tgt, ph_tgt)
            
        # Fallback to generic interpolation
        interp = create_interpolator(th_src, ph_src)
        return interp.interpolate_scalar(val, th_tgt, ph_tgt)

    def interpolate_vector_components(
        self, u_east, u_north, u_r, th_src, ph_src, th_tgt, ph_tgt
    ):
        """Interpolate vector components."""
        # Use optimized interpolator if source matches this basis
        src_grid = Grid(theta=th_src, phi=ph_src)
        if src_grid == self.grid:
            return self._interpolator.interpolate_vector(
                 u_east, u_north, u_r, th_tgt, ph_tgt
            )
            
        # Fallback to generic interpolation
        interp = create_interpolator(th_src, ph_src)
        return interp.interpolate_vector(u_east, u_north, u_r, th_tgt, ph_tgt)

    def project_to_basis(
        self,
        input_values,
        input_grid,
        vector_type,
        target_grid,
        target_basis,
        on_storage_grid,
        on_input_grid=None,
    ):
        """Project input data onto the target basis."""
        if target_grid is None:
            raise ValueError("target_grid must be provided")

        if vector_type == "scalar":
            grid_values = self.interpolate_scalar(
                input_values, input_grid.theta, input_grid.phi, target_grid.theta, target_grid.phi
            )
        elif vector_type == "tangential":
            u_east = input_values[1]
            u_north = -input_values[0]
            u_r = np.zeros_like(u_north)
            u_east_int, u_north_int, _ = self.interpolate_vector_components(
                u_east,
                u_north,
                u_r,
                input_grid.theta,
                input_grid.phi,
                target_grid.theta,
                target_grid.phi,
            )
            grid_values = np.hstack((-u_north_int, u_east_int))
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

        coeffs = target_basis.from_grid_values(grid_values, on_storage_grid(), vector_type)
        return coeffs



    def construct_projection_matrix(self, evaluator) -> Any:
        """Construct the projection matrix mapping Grid Vector -> Grid Values.
        
        For CSBasis, this is Identity (mapped to flat vector).
        """
        from scipy import sparse
        n = 2 * self.index_length
        return sparse.eye(n, format="csr")

    def get_extended_basis(self) -> "CSBasis":
        """Return a basis extended to include the monopole term.
        
        For CSBasis, the basis already includes all grid points.
        """
        return self
