"""Cubed sphere basis module.

This module contains the CSBasis class for representing the cubed sphere
basis.
"""

import numpy as np
import functools
import os
from scipy.special import binom
from scipy.sparse import coo_matrix

from pynamit.cubed_sphere import diffutils
from pynamit.math import arrayutils
from pynamit.primitives.grid_basis import GridBasis

d2r = np.pi / 180
datapath = os.path.dirname(os.path.abspath(__file__)) + "/data/"


class CSBasis(GridBasis):
    """Class for representing cubed sphere bases.

    This module provides an implementation of the cubed sphere grid
    system following methods from Yin et al. (2017).
    """

    def __init__(self, N: int = None):
        """Initialize the cubed sphere basis."""
        super().__init__(grid=None)
        
        if N is not None:
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
            _, self.arr_theta, self.arr_phi = self.cube2spherical(
                self.arr_xi, self.arr_eta, self.arr_block, deg=True
            )

            self.kind = "GRID"
            self.index_names = ["theta", "phi"]
            self.index_length = self.arr_theta.size + self.arr_phi.size
            self.index_arrays = [self.arr_theta, self.arr_phi]

            self.minimum_phi_sampling = 1
            self.caching = False

    @functools.cached_property
    def g(self) -> np.ndarray:
        """Metric tensor."""
        return self.get_metric_tensor(self.arr_xi, self.arr_eta)

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
            raise ValueError(f'coordinate must be either "xi", "eta", or "both". Not {coordinate}.')
        if Ns < order:
            raise ValueError("Ns must be >= order. You gave {} and {}".format(Ns, order))
        if order != 1:
            raise NotImplementedError("Only first order differentiation is supported.")

        shape = (6, N, N)
        size = 6 * N * N
        h = self.xi(1, N) - self.xi(0, N)
        k, i, j = map(np.ravel, np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing="ij"))

        stencil_points = np.hstack((np.r_[-Ns:0], np.r_[1 : Ns + 1]))
        Nsp = len(stencil_points)
        stencil_weight = diffutils.stencil(stencil_points, order=1, h=h)

        i_diff = np.hstack([i + _ for _ in stencil_points])
        j_diff = np.hstack([j + _ for _ in stencil_points])
        k_const, i_const, j_const = (np.tile(k, Nsp), np.tile(i, Nsp), np.tile(j, Nsp))
        weights = np.repeat(stencil_weight, size)

        rows = np.tile(np.ravel_multi_index((k, i, j), shape), Nsp)
        if coordinate in ["xi", "both"]:
            Dxi = self.get_interpolation_matrix(k_const, i_diff, j_const, N, Ni, rows=rows, weights=weights)
        if coordinate in ["eta", "both"]:
            Deta = self.get_interpolation_matrix(k_const, i_const, j_diff, N, Ni, rows=rows, weights=weights)

        if coordinate == "both":
            return (Dxi, Deta)
        if coordinate == "xi":
            return Dxi
        if coordinate == "eta":
            return Deta

    def get_interpolation_matrix(self, k, i, j, N, Ni, weights=None, rows=None):
        """Get matrix for grid to cubed sphere interpolation."""
        if Ni > N: raise ValueError("Ni must be <= N")
        k, i, j = map(np.ravel, [k, i, j])
        shape, size = (6, N, N), 6 * N**2
        if rows is None: rows = np.arange(k.size)
        if weights is None: weights = np.ones(k.size)
        weights = weights / Ni
        h = self.xi(1, N) - self.xi(0, N)
        cols = np.full(k.size, -1, dtype=np.int64)

        xi, eta = self.xi(i, N), self.eta(j, N)
        r, theta, phi = self.cube2spherical(xi, eta, k, r=1.0, deg=True)
        new_xi, new_eta, new_k = self.geo2cube(phi, 90 - theta)
        new_i, new_j = new_xi / h + (N - 1) / 2, new_eta / h + (N - 1) / 2

        assert np.all((np.isclose(new_i - np.rint(new_i), 0) | np.isclose(new_j - np.rint(new_j), 0)))
        ii_integers = np.isclose(new_i - np.rint(new_i), 0) & np.isclose(new_j - np.rint(new_j), 0)
        cols[ii_integers] = np.ravel_multi_index(
            (new_k[ii_integers], np.rint(new_i[ii_integers]).astype(np.int64), np.rint(new_j[ii_integers]).astype(np.int64)),
            shape,
        )

        i_is_float = ~np.isclose(np.rint(new_i) - new_i, 0)
        j_is_float = ~np.isclose(np.rint(new_j) - new_j, 0)
        assert sum(i_is_float & j_is_float) == 0
        j_floats = new_j[j_is_float].reshape((-1, 1))
        i_floats = new_i[i_is_float].reshape((-1, 1))

        interpolation_points = np.arange(Ni).reshape((1, -1))
        j_interpolation_points = arrayutils.constrain_values(interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2 - 1, 0, N - 1, axis=1)
        i_interpolation_points = arrayutils.constrain_values(interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2 - 1, 0, N - 1, axis=1)

        j_distances = j_floats - j_interpolation_points
        i_distances = i_floats - i_interpolation_points
        w = (-1) ** interpolation_points * binom(Ni - 1, interpolation_points)
        w_i = w / i_distances / np.sum(w / i_distances, axis=1).reshape((-1, 1))
        w_j = w / j_distances / np.sum(w / j_distances, axis=1).reshape((-1, 1))

        stacked_weights = np.tile(weights, (Ni, 1)).T
        stacked_cols = np.tile(cols, (Ni, 1)).T
        stacked_rows = np.tile(rows, (Ni, 1)).T

        stacked_cols[i_is_float] = np.ravel_multi_index((np.tile(new_k[i_is_float], (Ni, 1)).T, i_interpolation_points, np.rint(np.tile(new_j[i_is_float], (Ni, 1))).astype(np.int64).T), shape)
        stacked_cols[j_is_float] = np.ravel_multi_index((np.tile(new_k[j_is_float], (Ni, 1)).T, np.rint(np.tile(new_i[j_is_float], (Ni, 1))).astype(np.int64).T, j_interpolation_points), shape)
        stacked_weights[i_is_float] = stacked_weights[i_is_float] * w_i * Ni
        stacked_weights[j_is_float] = stacked_weights[j_is_float] * w_j * Ni

        D = coo_matrix((stacked_weights.flatten(), (stacked_rows.flatten(), stacked_cols.flatten())), shape=(rows.max() + 1, size))
        D.count_nonzero()
        return D

    # Methods block, geo2cube, interpolate_scalar, interpolate_vector_components inherited

    def get_projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates."""
        coastlines = np.load(datapath + "coastlines_" + resolution + ".npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield self.geo2cube(lon, lat)

    def interpolate_to_self(self, values, theta, phi, vector_type="scalar"):
        """Interpolate values to this basis's grid."""
        if vector_type == "scalar":
            return self.interpolate_scalar(values, theta, phi, self.arr_theta, self.arr_phi)
        elif vector_type == "tangential":
             u_east = values[1]
             u_north = -values[0]
             u_r = np.zeros_like(u_north)
             u_e, u_n, _ = self.interpolate_vector_components(u_east, u_north, u_r, theta, phi, self.arr_theta, self.arr_phi)
             return np.hstack((-u_n, u_e))
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def project_to_basis(self, input_values, input_grid, vector_type, target_grid, target_basis, on_storage_grid, on_input_grid=None):
        """Project input data onto the target basis."""
        if target_grid is None:
             raise ValueError("target_grid must be provided")

        if vector_type == "scalar":
            grid_values = self.interpolate_scalar(input_values, input_grid.theta, input_grid.phi, target_grid.theta, target_grid.phi)
        elif vector_type == "tangential":
            u_east = input_values[1]
            u_north = -input_values[0]
            u_r = np.zeros_like(u_north)
            u_east_int, u_north_int, _ = self.interpolate_vector_components(u_east, u_north, u_r, input_grid.theta, input_grid.phi, target_grid.theta, target_grid.phi)
            grid_values = np.hstack((-u_north_int, u_east_int))
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

        coeffs = target_basis.from_grid_values(grid_values, on_storage_grid(), vector_type)
        return coeffs
