"""Finite-difference operators for cubed-sphere grids."""

from __future__ import annotations

import numpy as np
from scipy.special import binom
from scipy.sparse import coo_matrix

from pynamit.sphere.cubed_sphere import arrayutils, diffutils


class CSFiniteDifferences:
    """Build finite-difference and cross-face interpolation matrices."""

    def __init__(self, basis):
        self.basis = basis

    def difference_matrix(self, N, coordinate="xi", Ns=1, Ni=4, order=1):
        """Return a scalar-field finite-difference matrix."""
        if coordinate not in ["xi", "eta", "both"]:
            raise ValueError(
                f'coordinate must be either "xi", "eta", or "both". Not  {coordinate}.'
            )

        if Ns < order:
            raise ValueError("Ns must be >= order. You gave {} and {}".format(Ns, order))

        if order != 1:
            raise NotImplementedError("Only first order differentiation is supported.")

        basis = self.basis
        shape = (6, N, N)
        size = 6 * N * N

        h = basis.xi(1, N) - basis.xi(0, N)

        k, i, j = map(
            np.ravel, np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing="ij")
        )

        stencil_points = np.hstack((np.r_[-Ns:0], np.r_[1 : Ns + 1]))
        stencil_count = len(stencil_points)
        stencil_weight = diffutils.stencil(stencil_points, order=1, h=h)

        i_diff = np.hstack([i + point for point in stencil_points])
        j_diff = np.hstack([j + point for point in stencil_points])
        k_const, i_const, j_const = (
            np.tile(k, stencil_count),
            np.tile(i, stencil_count),
            np.tile(j, stencil_count),
        )
        weights = np.repeat(stencil_weight, size)

        rows = np.tile(np.ravel_multi_index((k, i, j), shape), stencil_count)
        if coordinate in ["xi", "both"]:
            dxi = self.interpolation_matrix(
                k_const, i_diff, j_const, N, Ni, rows=rows, weights=weights
            )
        if coordinate in ["eta", "both"]:
            deta = self.interpolation_matrix(
                k_const, i_const, j_diff, N, Ni, rows=rows, weights=weights
            )

        if coordinate == "both":
            return dxi, deta
        if coordinate == "xi":
            return dxi
        return deta

    def interpolation_matrix(self, k, i, j, N, Ni, weights=None, rows=None):
        """Return a cross-face interpolation matrix."""
        if Ni > N:
            raise ValueError("Ni must be <= N")

        basis = self.basis
        k, i, j = map(np.ravel, [k, i, j])

        shape = (6, N, N)
        size = 6 * N**2

        if rows is None:
            rows = np.arange(k.size)

        if weights is None:
            weights = np.ones(k.size)
        weights = weights / Ni

        h = basis.xi(1, N) - basis.xi(0, N)
        cols = np.full(k.size, -1, dtype=np.int64)

        xi, eta = basis.xi(i + 0.5, N), basis.eta(j + 0.5, N)
        _, theta, phi = basis.cube2spherical(xi, eta, k, r=1.0, deg=True)
        new_xi, new_eta, new_k = basis.geo2cube(phi, 90 - theta)
        new_i, new_j = new_xi / h + (N - 1) / 2, new_eta / h + (N - 1) / 2

        assert np.all(
            (np.isclose(new_i - np.rint(new_i), 0) | np.isclose(new_j - np.rint(new_j), 0))
        )

        integer_pairs = np.isclose(new_i - np.rint(new_i), 0) & np.isclose(
            new_j - np.rint(new_j), 0
        )
        cols[integer_pairs] = np.ravel_multi_index(
            (
                new_k[integer_pairs],
                np.rint(new_i[integer_pairs]).astype(np.int64),
                np.rint(new_j[integer_pairs]).astype(np.int64),
            ),
            shape,
        )

        i_is_float = ~np.isclose(np.rint(new_i) - new_i, 0)
        j_is_float = ~np.isclose(np.rint(new_j) - new_j, 0)

        assert sum(i_is_float & j_is_float) == 0
        assert sum(i_is_float | j_is_float) == sum(cols == -1)

        j_floats = new_j[j_is_float].reshape((-1, 1))
        i_floats = new_i[i_is_float].reshape((-1, 1))

        interpolation_points = np.arange(Ni).reshape((1, -1))
        j_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2, 0, N - 1, axis=1
        )
        i_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2, 0, N - 1, axis=1
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

        matrix = coo_matrix(
            (stacked_weights.reshape(-1), (stacked_rows.reshape(-1), stacked_cols.reshape(-1))),
            shape=(rows.max() + 1, size),
        )
        matrix.count_nonzero()
        return matrix


__all__ = ["CSFiniteDifferences"]
