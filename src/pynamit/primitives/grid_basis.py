"""Grid Basis module."""

from __future__ import annotations
from typing import Any, Tuple, Optional, TYPE_CHECKING
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from pynamit.math import arrayutils

if TYPE_CHECKING:
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.basis_evaluator import BasisEvaluator

class GridBasis:
    """Basis representing values defined on a grid, utilizing generic spherical interpolation.
    
    This class serves as the fundamental object for grid-based fields, providing
    robust spherical interpolation capabilities via projection to cubed sphere faces.
    Other specific grid bases (e.g. CSBasis) can inherit from this.
    """
    
    def __init__(self, grid: Optional[Grid] = None):
        self.grid = grid
        self.index_length = grid.size if grid else 0
        self.caching = False
        self.kind = "GRID"

    @property
    def theta(self):
        return self.grid.theta if self.grid else None

    @property
    def phi(self):
        return self.grid.phi if self.grid else None

    def to_grid_values(self, coeffs: np.ndarray, evaluator: BasisEvaluator, field_type: str = "scalar") -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        target_grid = evaluator.grid
        
        # 1. Identity check
        if self.grid and target_grid is self.grid:
            return coeffs
        if self.grid and (target_grid.size == self.grid.size and 
            np.allclose(target_grid.theta, self.grid.theta) and 
            np.allclose(target_grid.phi, self.grid.phi)):
            return coeffs
            
        # 2. Interpolate using robust spherical logic
        if not self.grid:
             # If grid is None (e.g. incomplete CSBasis subclassing), cannot interpolate FROM it.
             # However, CSBasis usually has grid points defined.
             raise ValueError("Cannot interpolate from GridBasis without a source grid.")
             
        th_src = self.grid.theta
        ph_src = self.grid.phi
        th_tgt = target_grid.theta
        ph_tgt = target_grid.phi
        
        if field_type == "scalar":
            return self.interpolate_scalar(coeffs, th_src, ph_src, th_tgt, ph_tgt)
            
        elif field_type == "tangential":
            vals = coeffs.reshape(2, -1)
            v2, v3 = vals[0], vals[1]
            # Map PynaMIT (South, East) to CS (East, North)
            u_north = -v2
            u_east = v3
            u_r = np.zeros_like(v2)
            
            u_east_int, u_north_int, _ = self.interpolate_vector_components(
                u_east, u_north, u_r, th_src, ph_src, th_tgt, ph_tgt
            )
            return np.vstack([-u_north_int, u_east_int])
            
        elif field_type == "vector":
            vals = coeffs.reshape(3, -1)
            v1, v2, v3 = vals[0], vals[1], vals[2]
            u_r = v1
            u_north = -v2
            u_east = v3
            u_east_int, u_north_int, u_r_int = self.interpolate_vector_components(
                u_east, u_north, u_r, th_src, ph_src, th_tgt, ph_tgt
            )
            return np.vstack([u_r_int, -u_north_int, u_east_int])
            
        else:
            raise ValueError(f"Unknown field_type: {field_type}")

    def regularization_term(self, coeffs, evaluator, field_type):
        return None
        
    def from_grid_values(self, values: np.ndarray, evaluator: Any, field_type: str = "scalar") -> np.ndarray:
        input_grid = evaluator.grid
        if self.grid and input_grid is self.grid:
             return values
        if not self.grid:
             return values 
             
        if field_type == "scalar":
             return self.interpolate_scalar(values.flatten(), input_grid.theta, input_grid.phi, self.grid.theta, self.grid.phi)
        raise NotImplementedError("from_grid_values (interpolation to basis) only impl for scalar")

    # -------------------------------------------------------------------------
    # Spherical Coordinate Math (Generic)
    # -------------------------------------------------------------------------

    def block(self, lon, lat):
        """Determine cube faces (blocks) of spherical coordinates."""
        lon, lat = np.broadcast_arrays(lon, lat)
        lat, lon = lat.flatten(), lon.flatten()
        th, ph = np.deg2rad(90 - lat), np.deg2rad(lon)
        xyz = np.vstack((np.cos(ph) * np.sin(th), np.sin(th) * np.sin(ph), np.cos(th)))
        face_midpoints = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        distances = np.empty((6, xyz.shape[1]))
        for i in range(6):
            distances[i] = np.linalg.norm(xyz - face_midpoints[i].reshape((3, 1)), axis=0)
        safety_distance = 1e-10
        blocks = np.zeros(xyz.shape[1], dtype=int)
        for i in range(6):
            blocks[distances[i] < np.choose(blocks, distances) - safety_distance] = i
        return blocks

    def geo2cube(self, lon, lat, block=None):
        """Convert geocentric coordinates to cube coordinates."""
        lon, lat = np.broadcast_arrays(lon, lat)
        shape = lon.shape
        N_points = lon.size
        # Copied logic from CSBasis
        if block is None:
            block = self.block(lon, lat)
        else:
            block = block * np.ones_like(lat)
        block, lon, lat = block.flatten(), lon.flatten(), lat.flatten()
        X, Y, xi, eta = np.empty(N_points), np.empty(N_points), np.empty(N_points), np.empty(N_points)
        theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)
        
        mask = block == 0
        X[mask] = np.tan(phi[mask])
        Y[mask] = 1 / (np.tan(theta[mask]) * np.cos(phi[mask]))
        mask = block == 1
        X[mask] = np.tan(phi[mask] - np.pi / 2)
        Y[mask] = 1 / (np.tan(theta[mask]) * np.sin(phi[mask]))
        mask = block == 2
        X[mask] = np.tan(phi[mask] - np.pi)
        Y[mask] = -1 / (np.tan(theta[mask]) * np.cos(phi[mask]))
        mask = block == 3
        X[mask] = np.tan(phi[mask] - 3 * np.pi / 2)
        Y[mask] = -1 / (np.tan(theta[mask]) * np.sin(phi[mask]))
        
        mask = block == 4
        # Original: X[4] = tan(theta)*sin(phi). Y[4] = -tan(theta)*cos(phi).
        X[mask] = np.tan(theta[mask]) * np.sin(phi[mask])
        Y[mask] = -np.tan(theta[mask]) * np.cos(phi[mask])
        
        mask = block == 5
        X[mask] = -np.tan(theta[mask]) * np.sin(phi[mask])
        Y[mask] = -np.tan(theta[mask]) * np.cos(phi[mask])

        xi, eta = np.arctan(X), np.arctan(Y)
        return xi.reshape(shape), eta.reshape(shape), block.reshape(shape)

    def get_delta(self, xi, eta):
        xi, eta = np.broadcast_arrays(xi, eta)
        return 1 + np.tan(xi) ** 2 + np.tan(eta) ** 2

    def cube2cartesian(self, xi, eta, r=1, block=0):
        xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)
        delta = self.get_delta(xi, eta)
        x, y, z = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)
        
        iii = block == 0
        x[iii] = r[iii] / np.sqrt(delta[iii])
        y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        iii = block == 1
        x[iii] = -r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        y[iii] = r[iii] / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        iii = block == 2
        x[iii] = -r[iii] / np.sqrt(delta[iii])
        y[iii] = -r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        iii = block == 3
        x[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        y[iii] = -r[iii] / np.sqrt(delta[iii])
        z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        
        iii = block == 4
        x[iii] = -r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = r[iii] / np.sqrt(delta[iii])
        iii = block == 5
        x[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
        y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
        z[iii] = -r[iii] / np.sqrt(delta[iii])
        return x, y, z

    def cube2spherical(self, xi, eta, block, r=1, deg=False):
        xi, eta = np.float64(xi), np.float64(eta)
        xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)
        x, y, z = self.cube2cartesian(xi, eta, r, block)
        phi = np.arctan2(y, x)
        theta = np.arccos(z / r)
        if deg:
             phi, theta = np.rad2deg(phi), np.rad2deg(theta)
        return r, theta, phi

    def get_Pc(self, xi, eta, r=1, block=0, inverse=False):
        xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
        delta = self.get_delta(xi, et)
        Pc = np.empty((delta.size, 3, 3))
        rsec2xi = r / np.cos(xi) ** 2
        rsec2et = r / np.cos(et) ** 2

        # Block 0
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
        
        # Block 1
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
        
        # Block 2
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
        
        # Block 3
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
        
        # Block 4
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

        # Block 5
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
        return Pc

    def get_Ps(self, xi, eta, r=1, block=0, inverse=False):
        xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
        delta = self.get_delta(xi, et)
        Ps = np.empty((delta.size, 3, 3))
        # Blocks 0-3 same logic
        mask_eq = block <= 3
        if np.any(mask_eq):
            idx = np.where(mask_eq)
            Ps[idx, 0, 0] = 1
            Ps[idx, 0, 1] = 0
            Ps[idx, 0, 2] = 0
            Ps[idx, 1, 0] = np.tan(xi[idx]) * np.sin(et[idx]) * np.cos(et[idx])
            Ps[idx, 1, 1] = np.cos(xi[idx]) * np.sin(et[idx]) ** 2 + np.cos(et[idx]) ** 2 / np.cos(xi[idx])
            Ps[idx, 1, 2] = 0
            Ps[idx, 2, 0] = 0
            Ps[idx, 2, 1] = 0
            Ps[idx, 2, 2] = 1
        
        iii = block == 4
        if np.any(iii):
             Ps[iii, 0, 0] = -(np.cos(xi[iii]) ** 2) * np.tan(et[iii])
             Ps[iii, 0, 1] = -delta[iii] * np.tan(xi[iii]) * np.cos(xi[iii]) ** 2 / np.sqrt(delta[iii] - 1)
             Ps[iii, 0, 2] = 0
             Ps[iii, 1, 0] = np.cos(et[iii]) ** 2 * np.tan(xi[iii])
             Ps[iii, 1, 1] = -delta[iii] * np.tan(et[iii]) * np.cos(et[iii]) ** 2 / np.sqrt(delta[iii] - 1)
             Ps[iii, 1, 2] = 0
             Ps[iii, 2, 0] = 0
             Ps[iii, 2, 1] = 0
             Ps[iii, 2, 2] = 1
        
        iii = block == 5
        if np.any(iii):
             Ps[iii, 0, 0] = np.cos(xi[iii]) ** 2 * np.tan(et[iii])
             Ps[iii, 0, 1] = delta[iii] * np.tan(xi[iii]) * np.cos(xi[iii]) ** 2 / np.sqrt(delta[iii] - 1)
             Ps[iii, 0, 2] = 0
             Ps[iii, 1, 0] = -(np.cos(et[iii]) ** 2) * np.tan(xi[iii])
             Ps[iii, 1, 1] = delta[iii] * np.tan(et[iii]) * np.cos(et[iii]) ** 2 / np.sqrt(delta[iii] - 1)
             Ps[iii, 1, 2] = 0
             Ps[iii, 2, 0] = 0
             Ps[iii, 2, 1] = 0
             Ps[iii, 2, 2] = 1
             
        if inverse:
             return arrayutils.invert_3D_matrices(Ps)
        return Ps

    def get_Qij(self, xi, eta, block_i, block_j):
        xi_i, eta_i, block_i, block_j = map(np.ravel, np.broadcast_arrays(xi, eta, block_i, block_j))
        Psi_inv = self.get_Ps(xi_i, eta_i, r=1, block=block_i, inverse=True)
        r, theta, phi = self.cube2spherical(xi_i, eta_i, r=1, block=block_i, deg=True)
        xi_j, eta_j, _ = self.geo2cube(phi, 90 - theta, block=block_j)
        Psj = self.get_Ps(xi_j, eta_j, r=1, block=block_j)
        return np.einsum("nij, njk -> nik", Psj, Psi_inv)

    def get_Q(self, lat, r, inverse=False):
        lat, r = map(np.ravel, np.broadcast_arrays(lat, r))
        Q = np.zeros((lat.size, 3, 3), dtype=np.float64)
        Q[:, 0, 0] = r * np.cos(np.deg2rad(lat))
        Q[:, 1, 1] = r
        Q[:, 2, 2] = 1
        if inverse:
             return arrayutils.invert_3D_matrices(Q)
        return Q

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values."""
        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()
        scalar, theta, phi = np.broadcast_arrays(scalar, theta, phi)
        scalar, theta, phi = scalar.flatten(), theta.flatten(), phi.flatten()
        
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        interpolated_scalar = np.zeros_like(block, dtype=np.float64)
        
        for i in range(6):
             # Simplified mask logic from CSBasis
             _, th_f, ph_f = self.cube2spherical(0, 0, i, deg=False)
             r0 = np.hstack((np.sin(th_f)*np.cos(ph_f), np.sin(th_f)*np.sin(ph_f), np.cos(th_f))).reshape((-1,1))
             mask = np.sum(r0 * r, axis=0) > 0
             
             xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block=i)
             
             # Also enforce valid projected coordinates
             mask = mask & np.isfinite(xi_) & np.isfinite(eta_)
             
             target_mask = (block == i) & np.isfinite(xi) & np.isfinite(eta)
             if np.any(target_mask):
                 try:
                     interpolated_scalar[target_mask] = griddata(
                         np.vstack((xi_[mask], eta_[mask])).T,
                         scalar[mask],
                         np.vstack((xi[target_mask], eta[target_mask])).T,
                         rescale=True,
                         **kwargs
                     )
                 except QhullError:
                     # Fallback to nearest neighbor interpolation if triangulation fails
                     # (typically due to degenerate or collinear points)
                     interpolated_scalar[target_mask] = griddata(
                         np.vstack((xi_[mask], eta_[mask])).T,
                         scalar[mask],
                         np.vstack((xi[target_mask], eta[target_mask])).T,
                         method='nearest',
                         rescale=True
                     )
        return interpolated_scalar

    def interpolate_vector_components(self, u_east, u_north, u_r, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate vector components."""
        xi, eta, block = self.geo2cube(phi_target, 90 - theta_target)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()
        u_east, u_north, u_r, theta, phi = map(np.ravel, np.broadcast_arrays(u_east, u_north, u_r, theta, phi))
        
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        
        u_xi, u_eta, u_block = self.geo2cube(phi, 90 - theta)

        # Filter out pole singularities where Ps calculation is unstable
        # Specifically on polar blocks 4/5, if xi~0 and eta~0 (the pole itself),
        # the transformation matrix is singular.
        # Relax tolerance slightly to catch near-singular points
        pole_mask = (u_block >= 4) & (np.abs(u_xi) < 1e-5) & (np.abs(u_eta) < 1e-5)
        
        # Additional algebraic check: Filter points where det(Ps) is effectively zero
        # We compute Ps for all non-pole points to check determinant
        candidates_mask = ~pole_mask
        candidates_idx = np.where(candidates_mask)[0]
        
        if len(candidates_idx) > 0:
             Ps_check = self.get_Ps(u_xi[candidates_idx], u_eta[candidates_idx], r=1, block=u_block[candidates_idx])
             det_Ps = arrayutils.get_3D_determinants(Ps_check)
             # Filter out bad determinants (too small)
             valid_candidates = ~np.isclose(det_Ps, 0, atol=1e-8)
             
             # Final valid indices are subset of candidates
             valid_idx = candidates_idx[valid_candidates]
        else:
             valid_idx = np.array([], dtype=int)

        if len(valid_idx) < len(u_xi):
             # Subset input arrays
             u_east = u_east[valid_idx]
             u_north = u_north[valid_idx]
             u_r = u_r[valid_idx]
             theta = theta[valid_idx]
             phi = phi[valid_idx]
             u_xi = u_xi[valid_idx]
             u_eta = u_eta[valid_idx]
             u_block = u_block[valid_idx]
             # Also update derived
             th = th[valid_idx]
             ph = ph[valid_idx]
             r = r[:, valid_idx]

        Ps = self.get_Ps(u_xi, u_eta, r=1, block=u_block)
        Q = self.get_Q(90 - theta, r=1, inverse=True)
        Ps_normalized = np.einsum("nij, njk -> nik", Ps, Q)
        u_vec_sph = np.vstack((u_east, u_north, u_r))
        u_vec = np.einsum("nij, nj -> ni", Ps_normalized, u_vec_sph.T).T
        
        interpolated_u1 = np.zeros_like(block, dtype=np.float64)
        interpolated_u2 = np.zeros_like(block, dtype=np.float64)
        interpolated_u3 = np.zeros_like(block, dtype=np.float64)

        for i in range(6):
            Qij = self.get_Qij(u_xi, u_eta, u_block, i)
            u_vec_i = np.einsum("nij, nj -> ni", Qij, u_vec.T).T
            
            _, th_f, ph_f = self.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack((np.sin(th_f)*np.cos(ph_f), np.sin(th_f)*np.sin(ph_f), np.cos(th_f))).reshape((-1,1))
            mask = np.sum(r0 * r, axis=0) > 0
            
            xi_, eta_, _ = self.geo2cube(phi, 90 - theta, block=i)
            
            # Also enforce valid projected coordinates
            mask = mask & np.isfinite(xi_) & np.isfinite(eta_)
            
            target_mask = (block == i) & np.isfinite(xi) & np.isfinite(eta)
            if np.any(target_mask):
                pts = np.vstack((xi_[mask], eta_[mask])).T
                tgt_pts = np.vstack((xi[target_mask], eta[target_mask])).T
                try:
                    interpolated_u1[target_mask] = griddata(pts, u_vec_i[0][mask], tgt_pts, rescale=True, **kwargs)
                    interpolated_u2[target_mask] = griddata(pts, u_vec_i[1][mask], tgt_pts, rescale=True, **kwargs)
                    interpolated_u3[target_mask] = griddata(pts, u_vec_i[2][mask], tgt_pts, rescale=True, **kwargs)
                except QhullError:
                    interpolated_u1[target_mask] = griddata(pts, u_vec_i[0][mask], tgt_pts, method='nearest', rescale=True)
                    interpolated_u2[target_mask] = griddata(pts, u_vec_i[1][mask], tgt_pts, method='nearest', rescale=True)
                    interpolated_u3[target_mask] = griddata(pts, u_vec_i[2][mask], tgt_pts, method='nearest', rescale=True)

        _, theta_out, _ = self.cube2spherical(xi, eta, block, deg=True)
        u = np.vstack((interpolated_u1, interpolated_u2, interpolated_u3))
        Q = self.get_Q(90 - theta_out, r=1, inverse=False)
        Ps_inv = self.get_Ps(xi, eta, r=1, block=block, inverse=True)
        Ps_normalized_inv = np.einsum("nij, njk -> nik", Q, Ps_inv)
        return np.einsum("nij, nj -> ni", Ps_normalized_inv, u.T).T
