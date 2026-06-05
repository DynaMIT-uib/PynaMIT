"""Cubed sphere basis module.

This module contains the CSBasis class for representing the cubed sphere
basis.
"""

from collections import OrderedDict
import numpy as np
from pynamit.sphere.cubed_sphere import diffutils
from pynamit.sphere.cubed_sphere import arrayutils
import os
from scipy.special import binom
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from pynamit.math import as_linear_map
from pynamit.math.backend import get_array_module, to_numpy, use_jax
from pynamit.sphere.core import GridBasis, SurfaceEvaluator, SurfaceOperators

d2r = np.pi / 180
datapath = os.path.dirname(os.path.abspath(__file__)) + "/data/"


class _CSSurfaceEvaluator(SurfaceEvaluator):
    """Grid-bound CS evaluator with cached matrices."""

    def __init__(self, basis, grid):
        """Bind one CS basis to one grid."""
        super().__init__(basis, grid)
        self._matrix_cache = {}
        self._operator_cache = {}

    def evaluate(self, derivative=None):
        """Evaluate and cache CS scalar or derivative matrices."""
        key = ("evaluate", derivative)
        if key not in self._matrix_cache:
            self._matrix_cache[key] = self.basis.evaluate_on_grid(
                self.grid,
                derivative=derivative,
            )
        return self._matrix_cache[key]

    def surface_gradient_matrix(self):
        """Return cached CS surface-gradient matrices."""
        key = "surface_gradient"
        if key not in self._matrix_cache:
            self._matrix_cache[key] = self.basis.get_surface_gradient_matrix(self.grid)
        return self._matrix_cache[key]

    def rhat_cross_gradient_matrix(self):
        """Return cached CS ``rhat x grad`` matrices."""
        key = "rhat_cross_gradient"
        if key not in self._matrix_cache:
            if self.basis._is_native_grid(self.grid):
                gradient = self.surface_gradient_matrix()
                xp = get_array_module(gradient)
                self._matrix_cache[key] = xp.stack([-gradient[1], gradient[0]])
            else:
                self._matrix_cache[key] = self.basis.get_rhat_cross_gradient_matrix(
                    self.grid
                )
        return self._matrix_cache[key]

    def helmholtz_synthesis_matrix(self):
        """Return cached CS Helmholtz synthesis matrices."""
        key = "helmholtz_synthesis"
        if key not in self._matrix_cache:
            gradient = self.surface_gradient_matrix()
            rhat_cross_gradient = self.rhat_cross_gradient_matrix()
            xp = get_array_module(gradient, rhat_cross_gradient)
            self._matrix_cache[key] = xp.stack(
                [-xp.asarray(gradient), xp.asarray(rhat_cross_gradient)],
                axis=2,
            )
        return self._matrix_cache[key]

    def scalar_evaluation_operator(self, derivative=None):
        """Return cached CS scalar or derivative operators."""
        key = ("scalar_evaluation", derivative)
        if key not in self._operator_cache:
            if self.basis._is_native_grid(self.grid):
                if derivative is None:
                    matrix = sp.eye(self.basis.index_length, format="csr")
                elif derivative in {"theta", "phi"}:
                    matrix = self.basis._get_derivative_bundle()[derivative]
                else:
                    raise ValueError(f'Invalid derivative "{derivative}".')
                self._operator_cache[key] = as_linear_map(
                    matrix,
                    input_shape=(self.basis.index_length,),
                    output_shape=(self.basis.index_length,),
                )
            else:
                if derivative is None:
                    self._operator_cache[key] = self.basis.scalar_grid_remap_operator(
                        self.basis.native_grid,
                        self.grid,
                    )
                else:
                    self._operator_cache[key] = super().scalar_evaluation_operator(
                        derivative=derivative
                    )
        return self._operator_cache[key]

    def surface_gradient_operator(self):
        """Return cached CS surface-gradient operator."""
        key = "surface_gradient"
        if key not in self._operator_cache:
            if self.basis._is_native_grid(self.grid):
                bundle = self.basis._get_derivative_bundle()
                matrix = sp.vstack([bundle["theta"], bundle["phi"]], format="csr")
                self._operator_cache[key] = as_linear_map(
                    matrix,
                    input_shape=(self.basis.index_length,),
                    output_shape=(2, self.basis.index_length),
                )
            else:
                bundle = self.basis._get_derivative_bundle()
                matrix = sp.vstack([bundle["theta"], bundle["phi"]], format="csr")
                native_operator = as_linear_map(
                    matrix,
                    input_shape=(self.basis.index_length,),
                    output_shape=(2, self.basis.index_length),
                )
                remap_operator = self.basis.tangential_grid_remap_operator(
                    self.basis.native_grid,
                    self.grid,
                )
                self._operator_cache[key] = remap_operator @ native_operator
        return self._operator_cache[key]

    def rhat_cross_gradient_operator(self):
        """Return cached CS ``rhat x grad`` operator."""
        key = "rhat_cross_gradient"
        if key not in self._operator_cache:
            if self.basis._is_native_grid(self.grid):
                bundle = self.basis._get_derivative_bundle()
                matrix = sp.vstack([-bundle["phi"], bundle["theta"]], format="csr")
                self._operator_cache[key] = as_linear_map(
                    matrix,
                    input_shape=(self.basis.index_length,),
                    output_shape=(2, self.basis.index_length),
                )
            else:
                bundle = self.basis._get_derivative_bundle()
                matrix = sp.vstack([-bundle["phi"], bundle["theta"]], format="csr")
                native_operator = as_linear_map(
                    matrix,
                    input_shape=(self.basis.index_length,),
                    output_shape=(2, self.basis.index_length),
                )
                remap_operator = self.basis.tangential_grid_remap_operator(
                    self.basis.native_grid,
                    self.grid,
                )
                self._operator_cache[key] = remap_operator @ native_operator
        return self._operator_cache[key]

    def helmholtz_synthesis_operator(self):
        """Return cached CS Helmholtz synthesis operator."""
        key = "helmholtz_synthesis"
        if key not in self._operator_cache:
            if self.basis._is_native_grid(self.grid):
                bundle = self.basis._get_derivative_bundle()
                theta = bundle["theta"]
                phi = bundle["phi"]
                matrix = sp.bmat(
                    [[-theta, -phi], [-phi, theta]],
                    format="csr",
                )
                self._operator_cache[key] = as_linear_map(
                    matrix,
                    input_shape=(2, self.basis.index_length),
                    output_shape=(2, self.basis.index_length),
                )
            else:
                bundle = self.basis._get_derivative_bundle()
                theta = bundle["theta"]
                phi = bundle["phi"]
                matrix = sp.bmat(
                    [[-theta, -phi], [-phi, theta]],
                    format="csr",
                )
                native_operator = as_linear_map(
                    matrix,
                    input_shape=(2, self.basis.index_length),
                    output_shape=(2, self.basis.index_length),
                )
                remap_operator = self.basis.tangential_grid_remap_operator(
                    self.basis.native_grid,
                    self.grid,
                )
                self._operator_cache[key] = remap_operator @ native_operator
        return self._operator_cache[key]


class CSBasis(GridBasis, SurfaceOperators):
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

    Native CS coefficients are stored at cell centers. Cell areas are
    computed from the surrounding mapped cell corners, while
    differential operators act on cell-centered values and return
    cell-centered derivatives.

    Attributes
    ----------
    N : int
        Number of grid cells per cube edge (only set if N provided in
        constructor).
    arr_xi : ndarray
        Xi coordinates of native cell centers, in radians.
    arr_eta : ndarray
        Eta coordinates of native cell centers, in radians.
    arr_theta : ndarray
        Colatitude coordinates of native cell centers, in degrees.
    arr_phi : ndarray
        Longitude coordinates of native cell centers, in degrees.
    arr_block : ndarray
        Block indices (0-5) of native cell centers.
    g : ndarray
        Metric tensor
    sqrt_detg : ndarray
        Square root of determinant of the metric tensor.
    unit_area : ndarray
        Spherical quadrilateral area of each unit-sphere grid cell,
        computed from mapped cell corners.

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

    _shared_remap_matrix_cache = OrderedDict()
    _shared_remap_matrix_cache_size = 8

    def __init__(self, N=None):
        """Initialize the cubed sphere basis.

        If N is provided, initializes arrays for a grid with N×N cells
        on each cube face. The native coefficients live at the 6×N×N
        cell centers.

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
        super().__init__()
        self.kind = "CS"
        self._derivative_bundle = None
        self._laplacian_cache = {}
        self._laplacian_sparse_cache = {}
        self._remap_operator_cache = {}

        if N is not None:
            if not isinstance(N, (int, np.integer)):
                raise TypeError("N must be an integer")
            if N % 2 != 0:
                raise ValueError("Cubed sphere grid dimension must be even")

            self.N = N
            k, i, j = self.get_gridpoints(N)

            # Initialize native cell centers.
            self.arr_xi = self.xi(i[:, :-1, :-1] + 0.5, N).flatten()
            self.arr_eta = self.eta(j[:, :-1, :-1] + 0.5, N).flatten()
            self.arr_block = k[:, :-1, :-1].flatten()

            # Convert to spherical coordinates.
            _, self.arr_theta, self.arr_phi = self.cube2spherical(
                self.arr_xi, self.arr_eta, self.arr_block, deg=True
            )

            # Calculate metric factors at cell centers.
            self.g = self.get_metric_tensor(self.arr_xi, self.arr_eta)
            self.sqrt_detg = np.sqrt(arrayutils.get_3D_determinants(self.g))

            # Calculate exact spherical quadrilateral cell areas.
            self.unit_area = self._cell_areas(N)

            self.index_names = ["theta", "phi"]
            self.index_length = self.arr_theta.size
            self.index_arrays = [self.arr_theta, self.arr_phi]

            self.validate_metadata()

    @property
    def coefficient_space_signature(self):
        """Return a signature for CS coefficient compatibility."""
        return ("CS", int(self.N))

    @property
    def native_grid(self):
        """Return the native CS cell centers as a ``Grid``."""
        if not hasattr(self, "_native_grid"):
            if not hasattr(self, "arr_theta") or not hasattr(self, "arr_phi"):
                raise ValueError("CSBasis native_grid requires an initialized grid.")
            from pynamit.sphere.grid import Grid

            self._native_grid = Grid(
                theta=self.arr_theta,
                phi=self.arr_phi,
                area_weights=self.unit_area,
            )
        return self._native_grid

    def evaluator_for_grid(self, grid):
        """Return a grid-bound CS evaluator."""
        return _CSSurfaceEvaluator(self, grid)

    @staticmethod
    def _spherical_triangle_area(a, b, c):
        """Return oriented unit-sphere triangle area magnitude."""
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            1.0
            + np.einsum("ij,ij->i", a, b)
            + np.einsum("ij,ij->i", b, c)
            + np.einsum("ij,ij->i", c, a)
        )
        return np.abs(2.0 * np.arctan2(numerator, denominator))

    def _cell_areas(self, N):
        """Return exact spherical quadrilateral areas for all cells."""
        k, i, j = self.get_gridpoints(N)
        block = k[:, :-1, :-1].flatten()
        i0, i1 = i[:, :-1, :-1].flatten(), i[:, 1:, :-1].flatten()
        j0, j1 = j[:, :-1, :-1].flatten(), j[:, :-1, 1:].flatten()

        corners = [
            (self.xi(i0, N), self.eta(j0, N)),
            (self.xi(i1, N), self.eta(j0, N)),
            (self.xi(i1, N), self.eta(j1, N)),
            (self.xi(i0, N), self.eta(j1, N)),
        ]
        vectors = []
        for xi, eta in corners:
            x, y, z = self.cube2cartesian(xi, eta, np.ones_like(xi), block)
            vector = np.stack([x, y, z], axis=1)
            vectors.append(vector / np.linalg.norm(vector, axis=1).reshape((-1, 1)))

        return self._spherical_triangle_area(
            vectors[0], vectors[1], vectors[2]
        ) + self._spherical_triangle_area(vectors[0], vectors[2], vectors[3])

    @property
    def scalar_mean_weights(self):
        """Return area-normalized weights for scalar surface means."""
        if not hasattr(self, "unit_area"):
            raise ValueError("CSBasis scalar mean weights require an initialized grid.")
        weights = np.asarray(self.unit_area, dtype=float)
        total_area = float(np.sum(weights))
        if total_area <= 0.0:
            raise ValueError("CSBasis unit_area must have positive total area.")
        return weights / total_area

    def scalar_mean(self, coeffs):
        """Return the area-weighted mean of scalar CS coefficients."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        if values.shape[-1] != self.index_length:
            raise ValueError(
                "CS scalar coefficients must have the basis index_length on the last axis."
            )
        return xp.tensordot(values, xp.asarray(self.scalar_mean_weights), axes=([-1], [0]))

    def project_scalar_mean_free(self, coeffs):
        """Project scalar CS coefficients to area-weighted zero mean."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        mean = self.scalar_mean(values)
        return values - xp.expand_dims(mean, axis=-1)

    def project_helmholtz_mean_free(self, coeffs):
        """Project both CS Helmholtz potentials to zero mean."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        if values.shape[-1] == self.index_length:
            return self.project_scalar_mean_free(values)
        if values.shape[-1] == 2 * self.index_length:
            original_shape = values.shape
            reshaped = values.reshape(original_shape[:-1] + (2, self.index_length))
            return self.project_scalar_mean_free(reshaped).reshape(original_shape)
        raise ValueError(
            "CS Helmholtz coefficients must end with index_length or 2*index_length."
        )

    def _is_native_grid(self, grid):
        """Return whether ``grid`` matches this basis' native points."""
        if grid is self:
            return True
        same_as = getattr(grid, "same_as", None)
        if callable(same_as):
            return bool(same_as(self.native_grid))
        if not hasattr(grid, "theta") or not hasattr(grid, "phi"):
            return False
        from pynamit.sphere.grid import Grid

        grid_hash = Grid.coordinate_hash(to_numpy(grid.theta), to_numpy(grid.phi))
        return grid_hash == self.native_grid.hash

    @staticmethod
    def _grid_theta_phi(grid):
        """Return flattened theta/phi coordinates."""
        return (
            np.asarray(to_numpy(grid.theta), dtype=float).reshape(-1),
            np.asarray(to_numpy(grid.phi), dtype=float).reshape(-1),
        )

    @staticmethod
    def _grid_signature(grid):
        """Return a cache key for a grid."""
        signature = getattr(grid, "signature", None)
        if signature is None:
            raise TypeError("CS grid remapping requires Grid objects with signatures.")
        return signature

    @classmethod
    def _cached_remap_matrix(cls, key, build):
        """Return a bounded shared remap matrix cache entry."""
        cache = cls._shared_remap_matrix_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        matrix = build()
        cache[key] = matrix
        if len(cache) > cls._shared_remap_matrix_cache_size:
            cache.popitem(last=False)
        return matrix

    def _remap_matrix_key(self, kind, source_grid, target_grid):
        """Return a shared remap-matrix cache key."""
        return (
            type(self).__module__,
            type(self).__qualname__,
            kind,
            self._grid_signature(source_grid),
            self._grid_signature(target_grid),
        )

    @staticmethod
    def _linear_interpolation_weights(source_points, target_points):
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
        weights = np.column_stack(
            [first_weights, 1.0 - np.sum(first_weights, axis=1)]
        )
        return triangulation.simplices[simplex], weights

    def _block_interpolation_weights(self, theta, phi, theta_target, phi_target):
        """Return per-block interpolation weights."""
        xi_target, eta_target, block_target = self.geo2cube(
            phi_target, 90 - theta_target
        )
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

            _, th0, ph0 = self.cube2spherical(0, 0, block_index, deg=False)
            r0 = np.array(
                [np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)]
            ).reshape((-1, 1))
            source_mask = np.sum(r0 * r, axis=0) > 0
            source_index = np.flatnonzero(source_mask)

            xi_source, eta_source, _ = self.geo2cube(
                phi, 90 - theta, block=block_index
            )
            source_points = np.column_stack(
                [xi_source[source_mask], eta_source[source_mask]]
            )
            target_points = np.column_stack(
                [xi_target[target_index], eta_target[target_index]]
            )
            vertices, weights = self._linear_interpolation_weights(
                source_points,
                target_points,
            )
            blocks.append((block_index, target_index, source_index[vertices], weights))

        return blocks

    def _build_scalar_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse scalar grid remap."""
        theta, phi = self._grid_theta_phi(source_grid)
        theta_target, phi_target = self._grid_theta_phi(target_grid)
        blocks = self._block_interpolation_weights(
            theta,
            phi,
            theta_target,
            phi_target,
        )

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
        return sp.coo_matrix(
            (values, (row, col)),
            shape=(theta_target.size, theta.size),
        ).tocsr()

    def _build_tangential_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse tangential grid remap."""
        theta, phi = self._grid_theta_phi(source_grid)
        theta_target, phi_target = self._grid_theta_phi(target_grid)
        blocks = self._block_interpolation_weights(
            theta,
            phi,
            theta_target,
            phi_target,
        )

        xi_source, eta_source, block_source = self.geo2cube(phi, 90 - theta)
        source_ps = self.get_Ps(xi_source, eta_source, r=1, block=block_source)
        source_q = self.get_Q(90 - theta, r=1, inverse=True)
        source_transform = np.einsum("nij,njk->nik", source_ps, source_q)

        xi_target, eta_target, block_target = self.geo2cube(
            phi_target,
            90 - theta_target,
        )
        _, theta_out, _ = self.cube2spherical(
            xi_target,
            eta_target,
            block_target,
            deg=True,
        )
        target_q = self.get_Q(90 - theta_out, r=1, inverse=False)
        target_ps_inv = self.get_Ps(
            xi_target,
            eta_target,
            r=1,
            block=block_target,
            inverse=True,
        )
        target_transform = np.einsum("nij,njk->nik", target_q, target_ps_inv)

        n_source = theta.size
        n_target = theta_target.size
        out_components = np.arange(2)
        rows = []
        cols = []
        data = []

        for block_index, target_index, source_vertices, weights in blocks:
            qij = self.get_Qij(xi_source, eta_source, block_source, block_index)
            source_to_block = np.einsum(
                "nij,njk->nik",
                qij,
                source_transform,
            )
            source_coeff = source_to_block[source_vertices]
            source_coeff = np.stack(
                [-source_coeff[..., 1], source_coeff[..., 0]],
                axis=-1,
            )
            target_coeff = target_transform[target_index]
            target_coeff = np.stack(
                [-target_coeff[:, 1, :], target_coeff[:, 0, :]],
                axis=1,
            )

            coefficients = weights[:, :, None, None] * np.einsum(
                "tob,tvbi->tvoi",
                target_coeff,
                source_coeff,
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
        return sp.coo_matrix(
            (values, (row, col)),
            shape=(2 * n_target, 2 * n_source),
        ).tocsr()

    def scalar_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached scalar grid-remap operator."""
        matrix_key = self._remap_matrix_key(
            "scalar_grid_remap_matrix",
            source_grid,
            target_grid,
        )
        key = (
            "scalar_grid_remap",
            matrix_key,
            bool(use_jax()),
        )
        if key not in self._remap_operator_cache:
            matrix = self._cached_remap_matrix(
                matrix_key,
                lambda: self._build_scalar_grid_remap_matrix(
                    source_grid,
                    target_grid,
                ),
            )
            self._remap_operator_cache[key] = as_linear_map(
                matrix,
                input_shape=(source_grid.size,),
                output_shape=(target_grid.size,),
            )
        return self._remap_operator_cache[key]

    def tangential_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached tangential grid-remap operator."""
        matrix_key = self._remap_matrix_key(
            "tangential_grid_remap_matrix",
            source_grid,
            target_grid,
        )
        key = (
            "tangential_grid_remap",
            matrix_key,
            bool(use_jax()),
        )
        if key not in self._remap_operator_cache:
            matrix = self._cached_remap_matrix(
                matrix_key,
                lambda: self._build_tangential_grid_remap_matrix(
                    source_grid,
                    target_grid,
                ),
            )
            self._remap_operator_cache[key] = as_linear_map(
                matrix,
                input_shape=(2, source_grid.size),
                output_shape=(2, target_grid.size),
            )
        return self._remap_operator_cache[key]

    @staticmethod
    def _safe_sin_theta(theta_deg):
        """Return sin(theta) with a pole-safe floor."""
        sin_theta = np.sin(np.deg2rad(np.asarray(theta_deg).flatten()))
        return np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)

    def _coordinate_derivatives(self):
        """Return derivatives of xi/eta with respect to theta/phi."""
        xi, eta, r, block = np.broadcast_arrays(self.arr_xi, self.arr_eta, 1.0, self.arr_block)
        xi, eta, r, block = map(np.ravel, [xi, eta, r, block])

        pc = self.get_Pc(xi, eta, r=r, block=block)
        _, theta, phi = self.cube2spherical(xi, eta, r=r, block=block)

        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        dx_dtheta = r * cos_theta * cos_phi
        dy_dtheta = r * cos_theta * sin_phi
        dz_dtheta = -r * sin_theta
        dx_dphi = -r * sin_theta * sin_phi
        dy_dphi = r * sin_theta * cos_phi
        dz_dphi = np.zeros_like(r)

        dxi_dtheta = (
            pc[:, 0, 0] * dx_dtheta + pc[:, 0, 1] * dy_dtheta + pc[:, 0, 2] * dz_dtheta
        )
        dxi_dphi = pc[:, 0, 0] * dx_dphi + pc[:, 0, 1] * dy_dphi + pc[:, 0, 2] * dz_dphi
        deta_dtheta = (
            pc[:, 1, 0] * dx_dtheta + pc[:, 1, 1] * dy_dtheta + pc[:, 1, 2] * dz_dtheta
        )
        deta_dphi = pc[:, 1, 0] * dx_dphi + pc[:, 1, 1] * dy_dphi + pc[:, 1, 2] * dz_dphi

        return dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi

    def _get_derivative_bundle(self):
        """Build native-grid angular derivative operators."""
        if self._derivative_bundle is None:
            import scipy.sparse as sp

            dxi, deta = self.get_Diff(self.N, coordinate="both", Ns=1, Ni=4, order=1)
            dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi = self._coordinate_derivatives()

            dtheta = sp.diags(dxi_dtheta) @ dxi + sp.diags(deta_dtheta) @ deta
            dphi_unscaled = sp.diags(dxi_dphi) @ dxi + sp.diags(deta_dphi) @ deta
            sin_theta = self._safe_sin_theta(self.arr_theta)

            # ``phi_unscaled`` is d/dphi. ``phi`` is the azimuthal
            # surface component sin(theta)^-1 d/dphi used by gradients.
            self._derivative_bundle = {
                "theta": dtheta.tocsr(),
                "phi_unscaled": dphi_unscaled.tocsr(),
                "phi": (sp.diags(1.0 / sin_theta) @ dphi_unscaled).tocsr(),
                "sin_theta": sp.diags(sin_theta).tocsr(),
                "inv_sin_theta": sp.diags(1.0 / sin_theta).tocsr(),
                "inv_sin2_theta": sp.diags(1.0 / (sin_theta**2)).tocsr(),
            }
        return self._derivative_bundle

    def evaluate_on_grid(self, grid, derivative=None):
        """Evaluate CS nodal basis or derivatives."""
        import scipy.sparse as sp

        xp = get_array_module(getattr(grid, "theta", None), getattr(grid, "phi", None))
        native_grid = self._is_native_grid(grid)
        if not native_grid and derivative is not None:
            raise NotImplementedError(
                "CSBasis derivative evaluation is currently implemented only "
                "on the native cubed-sphere grid."
            )
        if derivative is None:
            matrix = (
                sp.eye(self.index_length, format="csr")
                if native_grid
                else self._scalar_interpolation_matrix(grid)
            )
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
        elif derivative in {"theta", "phi"}:
            matrix = self._get_derivative_bundle()[derivative].toarray()
        else:
            raise ValueError(f'Invalid derivative "{derivative}".')

        return xp.asarray(matrix)

    def _grid_to_cs_indices(self, grid):
        """Return CS face and cell-center indices."""
        xi, eta, block = self.geo2cube(grid.phi, 90 - grid.theta)
        h = self.xi(1, self.N) - self.xi(0, self.N)
        i = xi / h + (self.N - 1) / 2
        j = eta / h + (self.N - 1) / 2
        return block.flatten(), i.flatten(), j.flatten()

    def _scalar_interpolation_matrix(self, grid):
        """Return the built-in scalar interpolation as a matrix."""
        return self.interpolate_scalar(
            np.eye(self.index_length),
            self.arr_theta,
            self.arr_phi,
            grid.theta,
            grid.phi,
        )

    def _interpolate_tangential_operator(self, tangential_operator, grid):
        """Interpolate native-grid tangential operators to ``grid``."""
        tangential_operator = np.asarray(tangential_operator)
        east, north, _ = self.interpolate_vector_components(
            tangential_operator[1],
            -tangential_operator[0],
            np.zeros_like(tangential_operator[0]),
            self.arr_theta,
            self.arr_phi,
            grid.theta,
            grid.phi,
        )
        return np.stack([-north, east], axis=0)

    def get_surface_gradient_matrix(self, grid):
        """Return the CS surface-gradient matrix on ``grid``."""
        if self._is_native_grid(grid):
            return SurfaceOperators.get_surface_gradient_matrix(self, grid)
        native_gradient = SurfaceOperators.get_surface_gradient_matrix(self, self)
        matrix = self._interpolate_tangential_operator(native_gradient, grid)
        return get_array_module(getattr(grid, "theta", None), matrix).asarray(matrix)

    def get_rhat_cross_gradient_matrix(self, grid):
        """Return the CS rhat-cross-gradient matrix on ``grid``."""
        if self._is_native_grid(grid):
            return SurfaceOperators.get_rhat_cross_gradient_matrix(self, grid)
        native_rxgrad = SurfaceOperators.get_rhat_cross_gradient_matrix(self, self)
        matrix = self._interpolate_tangential_operator(native_rxgrad, grid)
        return get_array_module(getattr(grid, "theta", None), matrix).asarray(matrix)

    def get_helmholtz_synthesis_matrix(self, grid):
        """Return the CS Helmholtz synthesis tensor on ``grid``."""
        if self._is_native_grid(grid):
            return SurfaceOperators.get_helmholtz_synthesis_matrix(self, grid)
        xp = get_array_module(getattr(grid, "theta", None), getattr(grid, "phi", None))
        native_gradient = SurfaceOperators.get_surface_gradient_matrix(self, self)
        native_rxgrad = np.stack([-native_gradient[1], native_gradient[0]], axis=0)
        return xp.stack(
            [
                -xp.asarray(self._interpolate_tangential_operator(native_gradient, grid)),
                xp.asarray(self._interpolate_tangential_operator(native_rxgrad, grid)),
            ],
            axis=2,
        )

    def _sparse_laplacian_matrix(self, r=1.0):
        """Return the cached sparse discrete scalar Laplacian."""
        key = float(r)
        if key not in self._laplacian_sparse_cache:
            bundle = self._get_derivative_bundle()
            term_theta = (
                bundle["inv_sin_theta"]
                @ bundle["theta"]
                @ bundle["sin_theta"]
                @ bundle["theta"]
            )
            term_phi = bundle["inv_sin2_theta"] @ bundle["phi_unscaled"] @ bundle["phi_unscaled"]
            self._laplacian_sparse_cache[key] = ((term_theta + term_phi) / (r**2)).tocsr()
        return self._laplacian_sparse_cache[key]

    def laplacian(self, r=1.0):
        """Return the discrete scalar Laplacian matrix."""
        key = float(r)
        if key not in self._laplacian_cache:
            self._laplacian_cache[key] = self._sparse_laplacian_matrix(r).toarray()
        return get_array_module().asarray(self._laplacian_cache[key])

    def get_surface_laplacian_operator(self, r=1.0):
        """Return the native sparse scalar Laplacian operator."""
        return as_linear_map(
            self._sparse_laplacian_matrix(r),
            input_shape=(self.index_length,),
            output_shape=(self.index_length,),
        )

    def get_gridpoints(self, N, flat=False):
        """Generate grid-line indices for a given resolution.

        Parameters
        ----------
        N : int
            Number of grid cells per edge.
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
        Native CSBasis coefficients are cell-centered at
        ``i + 0.5, j + 0.5`` for ``i, j = 0, ..., N-1``.
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
        # The native CS values are cell-centered at i+0.5, j+0.5.
        xi, eta = self.xi(i + 0.5, N), self.eta(j + 0.5, N)
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
            interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2, 0, N - 1, axis=1
        )
        i_interpolation_points = arrayutils.constrain_values(
            interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2, 0, N - 1, axis=1
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
        spherical coordinates. Extra trailing dimensions on the
        component arrays are treated as independent vector fields and
        interpolated in one call.

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
        theta_target, phi_target = np.broadcast_arrays(theta_target, phi_target)
        target_shape = theta_target.shape
        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        # xi, eta, block = np.broadcast_arrays(xi, eta, block)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        theta, phi = np.broadcast_arrays(theta, phi)
        source_shape = theta.shape
        theta, phi = theta.flatten(), phi.flatten()

        u_east = np.asarray(u_east)
        u_north = np.asarray(u_north)
        u_r = np.asarray(u_r)
        if u_east.shape[: len(source_shape)] == source_shape:
            value_shape = u_east.shape[len(source_shape) :]
            u_east_values = u_east.reshape((theta.size,) + value_shape)
            u_north_values = u_north.reshape((theta.size,) + value_shape)
            u_r_values = u_r.reshape((theta.size,) + value_shape)
        else:
            u_east_values, u_north_values, u_r_values, theta_b, phi_b = np.broadcast_arrays(
                u_east,
                u_north,
                u_r,
                theta.reshape(source_shape),
                phi.reshape(source_shape),
            )
            value_shape = ()
            u_east_values = u_east_values.flatten()
            u_north_values = u_north_values.flatten()
            u_r_values = u_r_values.flatten()
            theta = theta_b.flatten()
            phi = phi_b.flatten()

        # Define vectors that point to all the original points.
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        # Convert vector components to cubed sphere coordinates.
        u_xi, u_eta, u_block = self.geo2cube(phi, 90 - theta)
        Ps = self.get_Ps(u_xi, u_eta, r=1, block=u_block)
        Q = self.get_Q(90 - theta, r=1, inverse=True)
        Ps_normalized = np.einsum("nij, njk -> nik", Ps, Q)
        u_vec_sph = np.stack([u_east_values, u_north_values, u_r_values], axis=1)
        u_vec = np.einsum("nij,nj...->ni...", Ps_normalized, u_vec_sph)

        interpolated_u = np.empty((block.size, 3) + value_shape, dtype=np.float64)

        # Loop over blocks and interpolate on each block.
        for i in range(6):
            # Express vector components with respect to block i.
            Qij = self.get_Qij(u_xi, u_eta, u_block, i)
            u_vec_i = np.einsum("nij,nj...->ni...", Qij, u_vec)

            # Filter points whose position vectors have component
            # anti-parallel to center of the block.
            _, th, ph = self.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).reshape(
                (-1, 1)
            )
            mask = np.sum(r0 * r, axis=0) > 0

            xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block=i)

            interpolated_u[block == i] = griddata(
                np.vstack((xi_[mask], eta_[mask])).T,
                u_vec_i[mask],
                np.vstack((xi[block == i], eta[block == i])).T,
                **kwargs,
            )

        # Convert back to spherical.
        _, theta_out, _ = self.cube2spherical(xi, eta, block, deg=True)
        Q = self.get_Q(90 - theta_out, r=1, inverse=False)
        Ps_inv = self.get_Ps(xi, eta, r=1, block=block, inverse=True)
        Ps_normalized_inv = np.einsum("nij, njk -> nik", Q, Ps_inv)
        interpolated = np.einsum("nij,nj...->ni...", Ps_normalized_inv, interpolated_u)
        u_east_int = interpolated[:, 0].reshape(target_shape + value_shape)
        u_north_int = interpolated[:, 1].reshape(target_shape + value_shape)
        u_r_int = interpolated[:, 2].reshape(target_shape + value_shape)

        return u_east_int, u_north_int, u_r_int

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values.

        Interpolate scalar values defined on (`theta`, `phi`) to given
        spherical coordinates.  Extra trailing dimensions on ``scalar``
        are treated as independent scalar fields and interpolated in
        one call.

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
        theta_target, phi_target = np.broadcast_arrays(theta_target, phi_target)
        target_shape = theta_target.shape
        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        # xi, eta, block = np.broadcast_arrays(xi, eta, block)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        theta, phi = np.broadcast_arrays(theta, phi)
        source_shape = theta.shape
        theta, phi = theta.flatten(), phi.flatten()

        scalar = np.asarray(scalar)
        if scalar.shape[: len(source_shape)] == source_shape:
            value_shape = scalar.shape[len(source_shape) :]
            scalar_values = scalar.reshape((theta.size,) + value_shape)
        else:
            scalar_values, theta_broadcast, phi_broadcast = np.broadcast_arrays(
                scalar, theta.reshape(source_shape), phi.reshape(source_shape)
            )
            value_shape = ()
            scalar_values = scalar_values.flatten()
            theta = theta_broadcast.flatten()
            phi = phi_broadcast.flatten()

        # Define vectors that point to all the original points.
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        interpolated_scalar = np.empty((block.size,) + value_shape, dtype=np.float64)

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
                scalar_values[mask],
                np.vstack((xi[block == i], eta[block == i])).T,
                **kwargs,
            )

        return interpolated_scalar.reshape(target_shape + value_shape)
