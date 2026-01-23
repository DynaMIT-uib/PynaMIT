
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
from pynamit.cubed_sphere import cs_math
from pynamit.primitives.grid import Grid, GridBasis, create_interpolator
from pynamit.primitives.grid.grid_utils import get_3D_determinants, constrain_values
from pynamit.utils import asarray

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
        super().__init__()

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
        # Attach weights for compatibility with numerical integrators (like SH path)
        self.grid.weights = self.unit_area

        # Initialize optimized interpolator
        from pynamit.primitives.grid.interpolation import CSInterpolator

        self._interpolator = CSInterpolator(N)

    @property
    def kind(self) -> str:
        return "CS"

    def get_laplacian_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the Laplacian operator for CSBasis."""
        from pynamit.math.linear_map import as_linear_map
        return as_linear_map(self.laplacian(r))

    def get_gradient_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical gradient operator.
        
        Returns a LinearMap that maps scalar potential coefficients to
        vector field components: E = -grad(φ) = (-d_θ φ, -(1/sin θ) d_φ φ) / r
        """
        return self._get_grid_gradient_operator(self.grid, r=r)

    def get_curl_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical curl operator (r × grad).
        
        Returns a LinearMap that maps scalar potential coefficients to
        vector field components: E = -r × grad(ψ) = ((1/sin θ) d_φ ψ, -d_θ ψ) / r
        """
        return self._get_grid_curl_operator(self.grid, r=r)

    def get_vector_curl_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the discrete radial curl operator on the grid."""
        from pynamit.math.linear_map import as_linear_map
        target_grid = grid if grid is not None else self.grid
        return as_linear_map(self._get_grid_curl(target_grid, r=1.0))

    def get_vector_divergence_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the discrete divergence operator on the grid."""
        from pynamit.math.linear_map import as_linear_map
        target_grid = grid if grid is not None else self.grid
        return as_linear_map(self._get_grid_divergence(target_grid, r=1.0))

    def get_toroidal_potential_coeffs(self, coeffs: np.ndarray, grid: Optional[Any] = None) -> np.ndarray:
        """Extract toroidal potential coefficients using Helmholtz projection.

        Parameters
        ----------
        coeffs : np.ndarray
            Vector field coefficients with shape (2, N_coeffs, ...).
        grid : optional
            Target grid for projection. Defaults to self.grid.

        Returns
        -------
        np.ndarray
            Toroidal potential coefficients with shape (N_coeffs, ...).
        """
        target_grid = grid if grid is not None else self.grid
        # P has shape (2, N_coeffs, 2, N_grid) - P[1] is the toroidal operator
        P = self.construct_projection_matrix(target_grid)
        coeffs = asarray(coeffs)

        # P[1] has shape (N_coeffs, 2, N_grid), coeffs has shape (2, N_coeffs, ...)
        return np.einsum('ijk,jk...->i...', P[1], coeffs)

    def get_poloidal_potential_coeffs(self, coeffs: np.ndarray, grid: Optional[Any] = None) -> np.ndarray:
        """Extract poloidal potential coefficients using Helmholtz projection.

        Parameters
        ----------
        coeffs : np.ndarray
            Vector field coefficients with shape (2, N_coeffs, ...).
        grid : optional
            Target grid for projection. Defaults to self.grid.

        Returns
        -------
        np.ndarray
            Poloidal potential coefficients with shape (N_coeffs, ...).
        """
        target_grid = grid if grid is not None else self.grid
        # P has shape (2, N_coeffs, 2, N_grid) - P[0] is the poloidal operator
        P = self.construct_projection_matrix(target_grid)
        coeffs = asarray(coeffs)

        # P[0] has shape (N_coeffs, 2, N_grid), coeffs has shape (2, N_coeffs, ...)
        return np.einsum('ijk,jk...->i...', P[0], coeffs)

    def get_radial_shift_operator(
        self, start_r: float, end_r: float, kind: str = "external"
    ) -> "LinearMap":
        """Get the radial shift operator. Default to global SH-like scaling for now."""
        from pynamit.math.linear_map import diagonal_linear_map
        # Grid basis can follow same potential radial scaling as SH (physics-based)
        # We assume each point scales like a generic potential term.
        # This is strictly true only for SH, but a reasonable "grid" approximation 
        # for potential fields if we don't have a better model.
        if kind == "external":
            # For Ve, usually n-dependent. If we don't have n, we assume a representative n=1?
            # Or we use a safe default of 1.0 if not supported.
            factor = (start_r / end_r)
        else:
            factor = (start_r / end_r) ** 2
        
        return diagonal_linear_map(np.ones(self.index_length) * factor)

    def get_potential_scaling_operator(self) -> "LinearMap":
        """Get potential scaling operator. To be refined for CS geometry."""
        from pynamit.math.linear_map import diagonal_linear_map
        # SH uses (2n + 1). For grid, 1.0 is a safe identity stub if not doing induction.
        return diagonal_linear_map(np.ones(self.index_length))

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
        return np.sqrt(get_3D_determinants(self.g))

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
        j_interpolation_points = constrain_values(
            interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2 - 1, 0, N - 1, axis=1
        )
        i_interpolation_points = constrain_values(
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
        # Relaxed check: if grid is self (evaluating basis on itself), it's CSBasis.
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
             # Evaluate basis on arbitrary grid (Interpolation)
             if derivative is not None:
                  # Compute derivative on native grid first, then interpolate
                  D_native = self.get_G(self, derivative=derivative)
                  I_interp = self._get_arbitrary_interpolation_matrix(grid, Ni=4)
                  return I_interp @ D_native
             
             return self._get_arbitrary_interpolation_matrix(grid, Ni=4)

        # 1. Check final operator cache
        if hasattr(grid, "hash"):
            grid_key = grid.hash
            if grid_key in self._cache:
                if derivative in self._cache[grid_key]:
                    return self._cache[grid_key][derivative]
            else:
                self._cache[grid_key] = {}
        
        N = self.N
        if derivative is None:
             from scipy.sparse import identity
             res = identity(6 * N * N, format="csr")

        elif derivative in ["theta", "phi"]:
             # 2. Get/Cache Logical Differentiation Operators (Depends only on Resolution N)
             if "_Dxi" not in self._cache:
                 Dxi, Deta = self.get_Diff(N, coordinate="both", Ns=1, Ni=4, order=1)
                 self._cache["_Dxi"] = Dxi
                 self._cache["_Deta"] = Deta
             
             Dxi, Deta = self._cache["_Dxi"], self._cache["_Deta"]

             # 3. Get/Cache Chain Rule Factors (Depends on Grid Coordinates)
             # Note: These are specific to the grid/resolution mapping. 
             # In CSBasis, coordinates are typically native, but we follow the per-grid pattern.
             cache_entry = self._cache[grid.hash] if hasattr(grid, "hash") else {}
             if "coord_derivs" not in cache_entry:
                  cache_entry["coord_derivs"] = cs_math.get_coordinate_derivatives(
                      self.arr_xi, self.arr_eta, r=1.0, block=self.arr_block
                  )
             
             dxi_dth, dxi_dph, deta_dth, deta_dph = cache_entry["coord_derivs"]

             from scipy.sparse import diags
             if derivative == "theta":
                 M_dxi_dth = diags(dxi_dth.flatten())
                 M_deta_dth = diags(deta_dth.flatten())
                 res = M_dxi_dth.dot(Dxi) + M_deta_dth.dot(Deta)
             elif derivative == "phi":
                 # Apply 1/sin(theta) scaling to match SHBasis convention.
                 # This makes G_ph represent (1/sin θ) d/dφ, which is the
                 # physical gradient component in spherical coordinates.
                 sin_th = np.sin(np.deg2rad(grid.theta)).reshape(dxi_dph.shape)
                 epsilon = 1e-10
                 sin_th_safe = np.where(np.abs(sin_th) < epsilon, epsilon, sin_th)
                 inv_sin_th = 1.0 / sin_th_safe

                 dxi_dph_scaled = dxi_dph * inv_sin_th
                 deta_dph_scaled = deta_dph * inv_sin_th

                 M_dxi_dph = diags(dxi_dph_scaled.flatten())
                 M_deta_dph = diags(deta_dph_scaled.flatten())
                 res = M_dxi_dph.dot(Dxi) + M_deta_dph.dot(Deta)
        
        else:
             raise ValueError(f"Unknown derivative: {derivative}")

        # 4. Store final operator in cache
        if hasattr(grid, "hash"):
            self._cache[grid.hash][derivative] = res
            
        return res

    def _get_arbitrary_interpolation_matrix(self, grid, Ni=4):
        """Construct interpolation matrix for arbitrary target points (2D Tensor Product)."""
        import scipy.sparse
        
        N = self.N
        # 1. Map target grid to CS coordinates
        xi_tgt, eta_tgt, k_tgt = cs_math.geo2cube(grid.phi, 90.0 - grid.theta)
        
        # 2. Fractional indices
        h_scaling = (2 * N) / np.pi
        i_tgt = (xi_tgt + np.pi / 4) * h_scaling
        j_tgt = (eta_tgt + np.pi / 4) * h_scaling
        
        # 3. Base integer indices (floor)
        i_base = np.floor(i_tgt).astype(int)
        j_base = np.floor(j_tgt).astype(int)
        
        # 4. Prepare Stencil Iteration
        # Stencil range relative to base: e.g. -1, 0, 1, 2 for Ni=4
        start = -(Ni // 2) + 1  # -1 for Ni=4
        stencil_offsets = np.arange(start, start + Ni)
        
        rows = []
        cols = []
        data = []
        
        n_targets = i_tgt.size
        target_indices = np.arange(n_targets)
        
        # Pre-calculate distances for weight computation
        # Lagrange weights for tensor product
        # w_total(di, dj) = w(di) * w(dj)
        
        # Loop over the 2D stencil (Ni x Ni)
        for di in stencil_offsets:
            for dj in stencil_offsets:
                # A. Identify Source Candidates (on the definition face k_tgt)
                i_cand = i_base + di
                j_cand = j_base + dj
                
                # B. Map Candidates to Valid Grid Nodes (Handle Face Crossing)
                # Convert (k_tgt, i_cand, j_cand) -> (xi_cand, eta_cand)
                # Note: This xi/eta might be outside [-pi/4, pi/4]
                xi_cand = -np.pi / 4 + i_cand * np.pi / (2 * N)
                eta_cand = -np.pi / 4 + j_cand * np.pi / (2 * N)
                
                # Map through sphere to canonical CS coordinates
                r, th, ph = cs_math.cube2spherical(xi_cand, eta_cand, k_tgt, r=1.0, deg=True)
                xi_prim, eta_prim, k_prim = cs_math.geo2cube(ph, 90.0 - th)
                
                # Convert back to indices on the new face
                i_prim = np.rint((xi_prim + np.pi / 4) * h_scaling).astype(int)
                j_prim = np.rint((eta_prim + np.pi / 4) * h_scaling).astype(int)
                
                # Clamp strict limits to avoid floating point noise issues at corners? 
                # geo2cube should handle it, but indices must be 0..N
                i_prim = np.clip(i_prim, 0, N) 
                j_prim = np.clip(j_prim, 0, N)
                
                # Flattened Column Indices
                # shape (6, N, N) -> (6, N+1, N+1)? 
                # CSBasis typically N+1? 
                # self.get_gridpoints(N) uses N+1. 
                # get_gridpoints(flat=True) returns size 6*(N+1)*(N+1).
                # Wait, CSBasis index length?
                # line 49: k, i, j = self.get_gridpoints(N)
                # line 52: i[:, :-1, :-1].
                # It drops the last row/col?
                # This implies "Centered" or "Element-based"? 
                # Yin et al method usually LGL nodes or similar.
                # Lines 52-54: i[:, :-1, :-1] + 0.5.
                # This implies CELL CENTRED basis? Or nodes at 0.5?
                # "xi = -pi/4 + (i+0.5)*..."
                # If basis is defined at cell centers, N is number of cells.
                
                # My `i_tgt` calculation used `i_tgt = (xi + pi/4) / h`. 
                # If xi = -pi/4 + (i_idx + 0.5)*h
                # Then xi + pi/4 = (i_idx + 0.5)*h
                # i_tgt (from xi) = i_idx + 0.5.
                # So `i_tgt` is shifted by 0.5 relative to integer indices?
                
                # Let's check `xi` method:
                # `return -np.pi / 4 + i * np.pi / (2 * N)`
                # AND init uses `i[:, :-1, :-1] + 0.5`.
                # So the STORED grid is at half-indices. 0.5, 1.5, ...
                
                # To align `i_tgt` with integer grid storage indices `0, 1, 2...` (representing 0.5, 1.5...):
                # i_grid_idx = i_tgt - 0.5.
                # Let's adjust i_tgt in Step 2.
                
                # Re-check. 
                # Basis stored at `indices` 0..N-1?
                # `self.arr_xi` derived from `i + 0.5`.
                # If I want to interpolate from values at `i+0.5`,
                # My coordinate `u` mapping to index `idx`:
                # `u = u0 + (idx + 0.5) * h`.
                # `u - u0 = idx*h + 0.5*h`
                # `(u-u0)/h = idx + 0.5`
                # `idx = (u-u0)/h - 0.5`.
                
                # So my `i_tgt` calculation (which was `(xi-xi0)/h`) yields `idx + 0.5`.
                # So I should subtract 0.5 to get `idx`.
                # Correct.
                
                # C. Compute Weights (Lagrange Interpolation for Cell-Centered Grid)
                # Nodes are at indices like 0.5, 1.5, etc.
                # i_tgt is in these units.
                # Relative to i_base, nodes are at (offset + 0.5)
                # Target is at delta = (i_tgt - i_base)
                
                # w_i(di) = Product_{m in stencil, m!=di} (delta - (m+0.5)) / ((di+0.5) - (m+0.5))
                #         = Product_{m!=di} (delta - m - 0.5) / (di - m)
                
                delta_i = i_tgt - i_base
                delta_j = j_tgt - j_base
                
                w_i = np.ones_like(delta_i)
                w_j = np.ones_like(delta_j)
                
                for m in stencil_offsets:
                    if m != di:
                        w_i *= (delta_i - m - 0.5) / (di - m)
                    if m != dj:
                        w_j *= (delta_j - m - 0.5) / (dj - m)
                        
                weight = w_i * w_j
                
                # D. Store in Sparse Lists
                # flattened index for col: k, i, j
                # Clip to valid index range 0..N-1 just in case
                i_prim = np.clip(i_prim, 0, N - 1)
                j_prim = np.clip(j_prim, 0, N - 1)
                
                col_indices = np.ravel_multi_index(
                    (k_prim, i_prim, j_prim), (6, N, N)
                )
                
                rows.append(target_indices)
                cols.append(col_indices)
                data.append(weight)

        # 5. Build Final Matrix
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        
        from scipy.sparse import coo_matrix
        return coo_matrix((data, (rows, cols)), shape=(n_targets, 6 * N * N))

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

        # Avoid division by zero at poles
        epsilon = 1e-10
        sin_th_safe = np.where(np.abs(sin_th) < epsilon, epsilon, sin_th)

        # Diagonal matrices for metric terms
        inv_sin_th = scipy.sparse.diags(1.0 / sin_th_safe)
        sin_th_mat = scipy.sparse.diags(sin_th)

        # Term 1: (1/sin(theta)) * d/dtheta (sin(theta) d/dtheta)
        term1 = inv_sin_th @ G_th @ sin_th_mat @ G_th

        # Term 2: (1/sin^2(theta)) * d^2/dphi^2
        # G_ph already includes 1/sin(theta), so G_ph @ G_ph = (1/sin²θ) d²/dφ²
        term2 = G_ph @ G_ph

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
        **kwargs,
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

        # Extract solve parameters from kwargs for the fit
        # We override weights to None because input weights are not valid on the interpolated grid
        fit_kwargs = kwargs.copy()
        fit_kwargs["weights"] = None

        coeffs = target_basis.from_grid_values(
            grid_values,
            target_grid,
            vector_type,
            **fit_kwargs,
        )
        return coeffs



    def construct_projection_matrix(self, grid) -> Any:
        """Construct the projection matrix mapping Grid Vector -> Scalar Potentials.

        For CSBasis in cs_dominant mode, the state variables (Phi, W) are
        scalar potentials. This matrix performs Helmholtz decomposition
        on the grid to extract them.

        Uses direct pseudo-inverse of the forward Helmholtz mapping (like SHBasis)
        rather than Laplacian inversion, which avoids numerical instability from
        the Laplacian's null space (constant functions).
        """
        # 1. Check Cache
        grid_key = getattr(grid, "hash", id(grid))
        if grid_key in self._cache and "projection_matrix" in self._cache[grid_key]:
            return self._cache[grid_key]["projection_matrix"]

        import scipy.sparse
        from pynamit.utils import tensor_pinv

        # Get gradient operators
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")

        # Convert to dense if sparse
        if scipy.sparse.issparse(G_th):
            G_th = G_th.toarray()
        if scipy.sparse.issparse(G_ph):
            G_ph = G_ph.toarray()

        # Build forward Helmholtz mapping: coeffs -> grid vectors
        # G_grad: E = -grad(phi) = (-d_th phi, -(1/sin th) d_ph phi)
        # G_rxgrad: E = r x grad(psi) = ((1/sin th) d_ph psi, -d_th psi)
        # G_ph already includes 1/sin(th) factor from get_G().
        G_grad = np.array([-G_th, -G_ph])
        G_rxgrad = np.array([G_ph, -G_th])

        # G_helmholtz: (2, N_grid, 2, N_coeffs)
        # Potential types: 0=poloidal, 1=toroidal
        G_helmholtz = np.stack([G_grad, G_rxgrad], axis=2)

        # Use proper pseudo-inverse via SVD (handles rank deficiency gracefully)
        # This avoids the numerical instability of inverting the singular Laplacian
        # pinv shape: (2, N_coeffs, 2, N_grid)
        # Index 0: potential type (0=poloidal, 1=toroidal)
        # Index 1: coefficient index
        # Index 2: vector component (0=theta, 1=phi)
        # Index 3: grid point
        res = tensor_pinv(G_helmholtz, n_leading_flattened=2)
        
        # 2. Store in Cache
        if grid_key not in self._cache:
            self._cache[grid_key] = {}
        self._cache[grid_key]["projection_matrix"] = res
        
        return res

    def _get_grid_divergence(self, grid: Any, r: float = 1.0) -> Any:
        """Get the discrete divergence operator matrix on the grid."""
        import scipy.sparse
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")

        theta_rad = np.deg2rad(self.arr_theta)
        sin_th = np.sin(theta_rad)

        epsilon = 1e-10
        sin_th_safe = np.where(np.abs(sin_th) < epsilon, epsilon, sin_th)
        inv_sin_th = scipy.sparse.diags(1.0 / sin_th_safe)
        sin_th_mat = scipy.sparse.diags(sin_th)

        # Div = (1/r sin th) [ d_th (E_th sin th) + d_ph E_ph ]
        # G_ph already includes 1/sin(th), so:
        # Div_th = (1/r sin th) @ G_th @ sin_th = (1/r) @ inv_sin_th @ G_th @ sin_th
        # Div_ph = (1/r sin th) @ d_ph = (1/r) @ G_ph
        D_th = inv_sin_th @ G_th @ sin_th_mat
        D_ph = G_ph

        return scipy.sparse.hstack([D_th, D_ph]) / r

    def _get_grid_curl(self, grid: Any, r: float = 1.0) -> Any:
        """Get the discrete radial curl operator matrix on the grid."""
        import scipy.sparse
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")

        theta_rad = np.deg2rad(self.arr_theta)
        sin_th = np.sin(theta_rad)

        epsilon = 1e-10
        sin_th_safe = np.where(np.abs(sin_th) < epsilon, epsilon, sin_th)
        inv_sin_th = scipy.sparse.diags(1.0 / sin_th_safe)
        sin_th_mat = scipy.sparse.diags(sin_th)

        # Curl_r = (1/r sin th) [ d_th (E_ph sin th) - d_ph E_th ]
        # G_ph already includes 1/sin(th), so:
        # C_th (acting on E_th) = -(1/r sin th) @ d_ph = -(1/r) @ G_ph
        # C_ph (acting on E_ph) = (1/r sin th) @ G_th @ sin_th = (1/r) @ inv_sin_th @ G_th @ sin_th
        C_th = -G_ph
        C_ph = inv_sin_th @ G_th @ sin_th_mat

        return scipy.sparse.hstack([C_th, C_ph]) / r

    def _get_grid_gradient_operator(self, grid: Any, r: float = 1.0) -> "LinearMap":
        """Get gradient operator mapping spectral potential to vector grid field.
        
        Returns a LinearMap that computes E = -grad(φ) = (-d_θ φ, -(1/sin θ) d_φ φ) / r
        The returned operator maps from scalar coefficients to stacked vector components.
        """
        from pynamit.math.linear_map import as_linear_map, BlockLinearMap
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")
        
        # E = -grad(phi) = (-d_th phi, -1/sin_th * d_ph phi) / r
        # G_ph already includes 1/sin_th factor.
        op_phi_th = as_linear_map(G_th) * (-1.0 / r)
        op_phi_ph = as_linear_map(G_ph) * (-1.0 / r)
        
        return BlockLinearMap([[op_phi_th], [op_phi_ph]])

    def _get_grid_curl_operator(self, grid: Any, r: float = 1.0) -> "LinearMap":
        """Get curl operator mapping spectral potential to vector grid field.
        
        Returns a LinearMap that computes E = -r × grad(ψ) = ((1/sin θ) d_φ ψ, -d_θ ψ) / r
        The returned operator maps from scalar coefficients to stacked vector components.
        """
        from pynamit.math.linear_map import as_linear_map, BlockLinearMap
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")
        
        # E = -r x grad(psi) = (1/sin_th * d_ph psi, -d_th psi) / r
        # G_ph already includes 1/sin_th factor.
        op_psi_th = as_linear_map(G_ph) * (1.0 / r)
        op_psi_ph = as_linear_map(G_th) * (-1.0 / r)
        
        return BlockLinearMap([[op_psi_th], [op_psi_ph]])

    def get_extended_basis(self) -> "CSBasis":
        """Return a basis extended to include the monopole term.
        
        For CSBasis, the basis already includes all grid points.
        """
        return self

    def get_evaluation_matrix(self, grid: Any, derivative: str = None) -> Any:
        """Get matrix evaluating basis (or derivatives) on a grid. Alias for get_G."""
        return self.get_G(grid, derivative=derivative)

    def get_vector_basis_matrix(self, grid: Any) -> Any:
        """Get vector basis evaluation matrix.

        For Cubed Sphere, we now use the Helmholtz decomposition as the vector
        representation: [Poloidal Potential; Toroidal Potential].

        This method returns the matrix G such that E_grid = G @ [phi_coeffs; psi_coeffs].
        G = [ G_pol, G_tor ] where G_pol maps potential to -grad(phi)
        and G_tor maps potential to -r x grad(psi).
        """
        from pynamit.math.linear_map import BlockLinearMap
        # Use our grid-based Helmholtz operators (at r=1.0 for the basis definition)
        G_pol = self._get_grid_gradient_operator(grid, r=1.0)
        G_tor = self._get_grid_curl_operator(grid, r=1.0)

        # Combine into [G_pol, G_tor]
        G_vec = BlockLinearMap([[G_pol, G_tor]])

        # Return as dense matrix with shape (2, N_grid, 2*L)
        G_dense = G_vec.to_dense()
        n_grid = grid.size if hasattr(grid, "size") else self.arr_theta.size
        return G_dense.reshape(2, n_grid, 2 * self.index_length)

