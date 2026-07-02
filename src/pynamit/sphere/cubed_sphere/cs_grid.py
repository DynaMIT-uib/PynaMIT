"""Native-grid and remapping helpers for CS surface bases."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay

from pynamit.math import as_linear_map, identity_linear_map
from pynamit.math.backend import to_numpy, use_jax
from pynamit.sphere.cubed_sphere import arrayutils


@dataclass(frozen=True)
class CSGridGeometry:
    """Native CS cell-center geometry for one grid resolution."""

    N: int
    arr_xi: np.ndarray
    arr_eta: np.ndarray
    arr_block: np.ndarray
    arr_theta: np.ndarray
    arr_phi: np.ndarray
    metric_tensor: np.ndarray
    sqrt_detg: np.ndarray
    unit_area: np.ndarray

    @property
    def index_length(self):
        """Total number of native cells."""
        return int(self.arr_theta.size)

    @classmethod
    def from_basis(cls, basis, N):
        """Build native-grid geometry from ``basis``."""
        k, i, j = basis.get_gridpoints(N)
        arr_xi = basis.xi(i[:, :-1, :-1] + 0.5, N).reshape(-1)
        arr_eta = basis.eta(j[:, :-1, :-1] + 0.5, N).reshape(-1)
        arr_block = k[:, :-1, :-1].reshape(-1)

        _, arr_theta, arr_phi = basis.cube2spherical(arr_xi, arr_eta, arr_block, deg=True)
        metric_tensor = basis.get_metric_tensor(arr_xi, arr_eta)
        sqrt_detg = np.sqrt(arrayutils.get_3D_determinants(metric_tensor))
        unit_area = cls._cell_areas(basis, N)

        return cls(
            N=int(N),
            arr_xi=arr_xi,
            arr_eta=arr_eta,
            arr_block=arr_block,
            arr_theta=arr_theta,
            arr_phi=arr_phi,
            metric_tensor=metric_tensor,
            sqrt_detg=sqrt_detg,
            unit_area=unit_area,
        )

    @staticmethod
    def spherical_triangle_area(a, b, c):
        """Return oriented unit-sphere triangle area magnitude."""
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            1.0
            + np.einsum("ij,ij->i", a, b)
            + np.einsum("ij,ij->i", b, c)
            + np.einsum("ij,ij->i", c, a)
        )
        return np.abs(2.0 * np.arctan2(numerator, denominator))

    @classmethod
    def _cell_areas(cls, basis, N):
        """Return exact spherical CS cell areas."""
        k, i, j = basis.get_gridpoints(N)
        block = k[:, :-1, :-1].reshape(-1)
        i0, i1 = i[:, :-1, :-1].reshape(-1), i[:, 1:, :-1].reshape(-1)
        j0, j1 = j[:, :-1, :-1].reshape(-1), j[:, :-1, 1:].reshape(-1)

        corners = [
            (basis.xi(i0, N), basis.eta(j0, N)),
            (basis.xi(i1, N), basis.eta(j0, N)),
            (basis.xi(i1, N), basis.eta(j1, N)),
            (basis.xi(i0, N), basis.eta(j1, N)),
        ]
        vectors = []
        for xi, eta in corners:
            x, y, z = basis.cube2cartesian(xi, eta, np.ones_like(xi), block)
            vector = np.stack([x, y, z], axis=1)
            vectors.append(vector / np.linalg.norm(vector, axis=1).reshape((-1, 1)))

        return cls.spherical_triangle_area(
            vectors[0], vectors[1], vectors[2]
        ) + cls.spherical_triangle_area(vectors[0], vectors[2], vectors[3])


class CSGridRemapper:
    """Build and cache remaps between CS-compatible grids."""

    _shared_remap_matrix_cache = OrderedDict()
    _shared_remap_matrix_cache_size = 8

    def __init__(self, basis, operator_cache=None):
        self.basis = basis
        self.operator_cache = {} if operator_cache is None else operator_cache

    @staticmethod
    def grid_theta_phi(grid):
        """Return flattened theta/phi coordinates."""
        return (
            np.asarray(to_numpy(grid.theta), dtype=float).reshape(-1),
            np.asarray(to_numpy(grid.phi), dtype=float).reshape(-1),
        )

    @staticmethod
    def grid_signature(grid):
        """Return a cache key for a grid."""
        signature = getattr(grid, "signature", None)
        if signature is None:
            raise TypeError("CS grid remapping requires Grid objects with signatures.")
        return signature

    def _cached_remap_matrix(self, key, build):
        """Return a bounded shared remap matrix cache entry."""
        cache = self.basis._shared_remap_matrix_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        matrix = build()
        cache[key] = matrix
        if len(cache) > self.basis._shared_remap_matrix_cache_size:
            cache.popitem(last=False)
        return matrix

    def remap_matrix_key(self, kind, source_grid, target_grid):
        """Return a shared remap-matrix cache key."""
        basis_type = type(self.basis)
        return (
            basis_type.__module__,
            basis_type.__qualname__,
            kind,
            self.grid_signature(source_grid),
            self.grid_signature(target_grid),
        )

    @staticmethod
    def linear_interpolation_weights(source_points, target_points):
        """Return Delaunay vertices and barycentric weights."""
        if source_points.shape[0] < 3:
            raise ValueError("At least three source points are required.")
        triangulation = Delaunay(source_points)
        simplex = triangulation.find_simplex(target_points)
        if np.any(simplex < 0):
            raise ValueError("Target points lie outside the source interpolation hull.")

        transform = triangulation.transform[simplex]
        delta = target_points - transform[:, 2]
        first_weights = np.einsum("nij,nj->ni", transform[:, :2], delta)
        weights = np.column_stack([first_weights, 1.0 - np.sum(first_weights, axis=1)])
        return triangulation.simplices[simplex], weights

    def block_interpolation_weights(self, theta, phi, theta_target, phi_target):
        """Return per-block interpolation weights."""
        basis = self.basis
        xi_target, eta_target, block_target = basis.geo2cube(phi_target, 90 - theta_target)
        xi_target = xi_target.reshape(-1)
        eta_target = eta_target.reshape(-1)
        block_target = block_target.reshape(-1)

        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        blocks = []

        for block_index in range(6):
            target_index = np.flatnonzero(block_target == block_index)
            if target_index.size == 0:
                continue

            _, th0, ph0 = basis.cube2spherical(0, 0, block_index, deg=False)
            r0 = np.array(
                [np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)]
            ).reshape((-1, 1))
            source_mask = np.sum(r0 * r, axis=0) > 0
            source_index = np.flatnonzero(source_mask)

            xi_source, eta_source, _ = basis.geo2cube(phi, 90 - theta, block=block_index)
            source_points = np.column_stack([xi_source[source_mask], eta_source[source_mask]])
            target_points = np.column_stack([xi_target[target_index], eta_target[target_index]])
            vertices, weights = self.linear_interpolation_weights(source_points, target_points)
            blocks.append((block_index, target_index, source_index[vertices], weights))

        return blocks

    def build_scalar_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse scalar grid remap."""
        theta, phi = self.grid_theta_phi(source_grid)
        theta_target, phi_target = self.grid_theta_phi(target_grid)
        blocks = self.block_interpolation_weights(theta, phi, theta_target, phi_target)

        rows = []
        cols = []
        data = []
        for _, target_index, source_vertices, weights in blocks:
            rows.append(np.repeat(target_index, 3))
            cols.append(source_vertices.reshape(-1))
            data.append(weights.reshape(-1))

        if rows:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            values = np.concatenate(data)
        else:
            row = col = np.array([], dtype=int)
            values = np.array([], dtype=float)
        return sp.coo_matrix((values, (row, col)), shape=(theta_target.size, theta.size)).tocsr()

    def build_tangential_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse tangential grid remap."""
        basis = self.basis
        theta, phi = self.grid_theta_phi(source_grid)
        theta_target, phi_target = self.grid_theta_phi(target_grid)
        blocks = self.block_interpolation_weights(theta, phi, theta_target, phi_target)

        xi_source, eta_source, block_source = basis.geo2cube(phi, 90 - theta)
        source_ps = basis.get_Ps(xi_source, eta_source, r=1, block=block_source)
        source_q = basis.get_Q(90 - theta, r=1, inverse=True)
        source_transform = np.einsum("nij,njk->nik", source_ps, source_q)

        xi_target, eta_target, block_target = basis.geo2cube(phi_target, 90 - theta_target)
        _, theta_out, _ = basis.cube2spherical(xi_target, eta_target, block_target, deg=True)
        target_q = basis.get_Q(90 - theta_out, r=1, inverse=False)
        target_ps_inv = basis.get_Ps(xi_target, eta_target, r=1, block=block_target, inverse=True)
        target_transform = np.einsum("nij,njk->nik", target_q, target_ps_inv)

        n_source = theta.size
        n_target = theta_target.size
        out_components = np.arange(2)
        rows = []
        cols = []
        data = []

        for block_index, target_index, source_vertices, weights in blocks:
            qij = basis.get_Qij(xi_source, eta_source, block_source, block_index)
            source_to_block = np.einsum("nij,njk->nik", qij, source_transform)
            source_coeff = source_to_block[source_vertices]
            source_coeff = np.stack([-source_coeff[..., 1], source_coeff[..., 0]], axis=-1)
            target_coeff = target_transform[target_index]
            target_coeff = np.stack([-target_coeff[:, 1, :], target_coeff[:, 0, :]], axis=1)

            coefficients = weights[:, :, None, None] * np.einsum(
                "tob,tvbi->tvoi", target_coeff, source_coeff
            )
            row = target_index[:, None, None, None] + (
                out_components[None, None, :, None] * n_target
            )
            col = source_vertices[:, :, None, None] + (
                out_components[None, None, None, :] * n_source
            )
            rows.append(np.broadcast_to(row, coefficients.shape).reshape(-1))
            cols.append(np.broadcast_to(col, coefficients.shape).reshape(-1))
            data.append(coefficients.reshape(-1))

        if rows:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            values = np.concatenate(data)
        else:
            row = col = np.array([], dtype=int)
            values = np.array([], dtype=float)
        return sp.coo_matrix((values, (row, col)), shape=(2 * n_target, 2 * n_source)).tocsr()

    def scalar_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached scalar grid-remap operator."""
        if source_grid.same_as(target_grid):
            return identity_linear_map((source_grid.size,))
        matrix_key = self.remap_matrix_key("scalar_grid_remap_matrix", source_grid, target_grid)
        key = ("scalar_grid_remap", matrix_key, bool(use_jax()))
        if key not in self.operator_cache:
            matrix = self._cached_remap_matrix(
                matrix_key,
                lambda: self.basis._build_scalar_grid_remap_matrix(source_grid, target_grid),
            )
            self.operator_cache[key] = as_linear_map(
                matrix, input_shape=(source_grid.size,), output_shape=(target_grid.size,)
            )
        return self.operator_cache[key]

    def tangential_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached tangential grid-remap operator."""
        if source_grid.same_as(target_grid):
            return identity_linear_map((2, source_grid.size))
        matrix_key = self.remap_matrix_key(
            "tangential_grid_remap_matrix", source_grid, target_grid
        )
        key = ("tangential_grid_remap", matrix_key, bool(use_jax()))
        if key not in self.operator_cache:
            matrix = self._cached_remap_matrix(
                matrix_key,
                lambda: self.basis._build_tangential_grid_remap_matrix(source_grid, target_grid),
            )
            self.operator_cache[key] = as_linear_map(
                matrix, input_shape=(2, source_grid.size), output_shape=(2, target_grid.size)
            )
        return self.operator_cache[key]


__all__ = ["CSGridGeometry", "CSGridRemapper"]
