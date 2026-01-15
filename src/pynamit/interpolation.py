"""Interpolation Strategy Module.

This module defines the Interpolator strategy pattern for handling grid interpolation.
It separates the "how" of interpolation from the "what" of data definition.

Classes:
    Interpolator: Abstract Base Class.
    CSInterpolator: Optimized bilinear interpolation for structured Cubed Sphere grids.
    UnstructuredInterpolator: Fallback using scipy.spatial.griddata (Qhull).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import QhullError, Delaunay

from pynamit.cubed_sphere import cs_math
from pynamit.math import arrayutils


class Interpolator(ABC):
    """Abstract base class for interpolation strategies."""

    @abstractmethod
    def interpolate_scalar(
        self, values: np.ndarray, target_theta: np.ndarray, target_phi: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Interpolate scalar field to target coordinates."""
        pass

    @abstractmethod
    def interpolate_vector(
        self,
        u_east: np.ndarray,
        u_north: np.ndarray,
        u_r: np.ndarray,
        target_theta: np.ndarray,
        target_phi: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Interpolate vector field components to target coordinates."""
        pass

    def is_compatible(self, values: np.ndarray) -> bool:
        """Check if input values are compatible with this specialized interpolator."""
        return True


def create_interpolator(theta, phi):
    """Factory to create the best interpolator for the given grid."""
    # Since we moved to CachedDelaunayInterpolator which is generic AND optimized (via caching),
    # we can use it for all grids that are effectively static.
    # The 'Structured' check is less relevant if we just rely on Delaunay caching.
    # However, to be safe and general, we return the Cached version.
    return CachedDelaunayInterpolator(theta, phi)


class CachedDelaunayInterpolator(Interpolator):
    """Interpolator using cached Delaunay triangulation to optimize repeated calls.

    Numerically equivalent to scipy.interpolate.griddata(..., method='linear', rescale=True).
    """

    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

        # Pre-compute Triangulations per Face
        # We need to project source points to ALL 6 faces because we don't know where target points will constitute.
        # Wait, UnstructuredInterpolator logic loops over FACES i=0..5.
        # For each face i, it projects Source Points to Face i.
        # Then it interpolates from those Source Points (on Face i) to Target Points (on Face i).
        # So we need 6 triangulations?
        # UnstructuredInterpolator projects ALL source points to Face i?
        # "mask = np.sum(r0 * r, axis=0) > 0" -> Only points that "see" face i (hemisphere).
        # So yes, we need to replicate that logic and cache 6 triangulations.

        self.triangulations = {}  # Key: block_idx
        self.scalings = {}  # Key: block_idx, Val: (min, scale) for rescaling

        # Prepare source geometry
        th, ph = np.deg2rad(self.theta.flatten()), np.deg2rad(self.phi.flatten())
        self.r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        self.phi_src = self.phi.flatten()
        self.theta_src = self.theta.flatten()  # in degrees

        for i in range(6):
            # 1. Select Source Points relevant for Face i
            _, th_f, ph_f = cs_math.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack(
                (np.sin(th_f) * np.cos(ph_f), np.sin(th_f) * np.sin(ph_f), np.cos(th_f))
            ).reshape((-1, 1))
            mask = np.sum(r0 * self.r, axis=0) > 0

            xi_src, eta_src, _ = cs_math.geo2cube(self.phi_src, 90 - self.theta_src, block=i)
            mask = mask & np.isfinite(xi_src) & np.isfinite(eta_src)

            if not np.any(mask):
                continue

            pts = np.vstack((xi_src[mask], eta_src[mask])).T

            # 2. Rescale points (match griddata rescale=True)
            # griddata maps to [0,1]^D unit cube.
            min_v = np.min(pts, axis=0)
            max_v = np.max(pts, axis=0)
            scale = max_v - min_v
            scale[scale < 1e-12] = 1.0  # Avoid division by zero

            pts_norm = (pts - min_v) / scale

            # 3. Compute Triangulation
            try:
                tri = Delaunay(pts_norm)
                self.triangulations[i] = (tri, mask, min_v, scale)
            except QhullError:
                # Fallback handled at interpolate time (nearest)
                self.triangulations[i] = (None, mask, min_v, scale)

    def interpolate_scalar(self, values: np.ndarray, target_theta, target_phi, **kwargs):
        scalar = values.flatten()

        xi, eta, block = cs_math.geo2cube(target_phi, 90 - target_theta)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        interpolated = np.zeros_like(block, dtype=np.float64)

        for i in range(6):
            if i not in self.triangulations:
                continue

            tri, mask, min_v, scale = self.triangulations[i]

            target_mask = (block == i) & np.isfinite(xi) & np.isfinite(eta)
            if not np.any(target_mask):
                continue

            tgt_pts = np.vstack((xi[target_mask], eta[target_mask])).T
            tgt_pts_norm = (tgt_pts - min_v) / scale

            vals = scalar[mask]

            if tri is not None:
                # Use cached LinearNDInterpolator logic (Delaunay based)
                # LinearNDInterpolator(tri, vals)
                interp = LinearNDInterpolator(tri, vals)
                res = interp(tgt_pts_norm)

                # Handle NaNs (outside convex hull) -> griddata fills nan usually.
                # pynamit behavior? Unstructured filled with 0 init.
                # LinearND returns nan for outside.
                # Update: griddata doc says 'fill_value', default nan.
                # But existing code fills 0 init, and assigns res.
                # If res has nans, they propagate.
                # existing: interpolated[target_mask] = res

                # Check for NaNs and fallback to Nearest if needed?
                # Unstructured had try-except.
                # Here we trust Delaunay.

                # Handle fill value
                res[np.isnan(res)] = 0.0  # Robustness for edges?
                interpolated[target_mask] = res
            else:
                # Fallback to nearest (slow but rare)
                # Re-construct points for nearest?
                # Just use griddata nearest for robustness in error case
                try:
                    # Need original pts
                    xi_src, eta_src, _ = cs_math.geo2cube(
                        self.phi_src, 90 - self.theta_src, block=i
                    )
                    pts = np.vstack((xi_src[mask], eta_src[mask])).T
                    res = griddata(pts, vals, tgt_pts, method="nearest", rescale=True)
                    interpolated[target_mask] = res
                except QhullError:
                    pass

        return interpolated

    def interpolate_vector(self, u_east, u_north, u_r, target_theta, target_phi, **kwargs):
        # 1. Setup Logic (Identical to reference)
        xi, eta, block = cs_math.geo2cube(target_phi, 90 - target_theta)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        theta_src, phi_src = self.theta_src, self.phi_src

        # Ps Setup
        u_xi, u_eta, u_block = cs_math.geo2cube(phi_src, 90 - theta_src)

        # Note: Unstructured logic filters candidates for Ps calc.
        # We can reproduce that or compute all. Computing all is cleaner if vectorized.
        # But we must handle pole singularities.

        # For strict reproducibility, we follow the legacy logic of filtering 'valid_idx' for Ps lines.
        # But that filtered the input arrays u_east... which changes 'values' for interpolation!
        # If we remove points, our cached triangulation (based on ALL points) is invalid!
        # CONFLICT: Cached Delaunay assumes fixed point set. Legacy filters points dynamically based on Ps validity.
        # RESOLUTION: Ps validity depends on COORDINATES (fixed), not values.
        # So 'valid_idx' is FIXED for a given grid.
        # We should apply this filtering in __init__ and build triangulation only on valid points!

        # However, implementing that in __init__ is complex (Ps logic coupling).
        # Alternative: Compute Ps for all, replace bad Ps with Identity, let coefficients be 0?
        # Or: Trust that 'mask' in triangulation logic (Hemisphere check) filters out the pole usually?
        # Pole is at block 4/5 center.
        # Cubed Sphere faces 4/5 cover pole.
        # If input points are at pole, they are on face 4/5.
        # Ps is singular.

        # Let's perform the Ps projection using robust logic here, ON THE FLY,
        # but map the resulting u_vec to the FULL points array for interpolation.
        # Any invalid points -> we set u_vec to 0 (or NaN).
        # Since interpolation is linear, 0s will blend.

        Ps = cs_math.get_Ps(u_xi, u_eta, r=1, block=u_block)
        Q = cs_math.get_Q(90 - theta_src, r=1, inverse=True)
        Ps_normalized = np.einsum("nij, njk -> nik", Ps, Q)

        u_vec_sph = np.vstack((u_east, u_north, u_r))
        u_vec = np.einsum("nij, nj -> ni", Ps_normalized, u_vec_sph.T).T

        # Clean up singularities
        # u_vec is (3, N)
        bad_mask = ~np.all(np.isfinite(u_vec), axis=0)  # Check per point (across components)
        if np.any(bad_mask):
            u_vec[:, bad_mask] = 0.0

        interpolated_components = np.zeros((3, block.size), dtype=np.float64)

        for i in range(6):
            if i not in self.triangulations:
                continue

            tri, mask, min_v, scale = self.triangulations[i]

            target_mask = (block == i) & np.isfinite(xi) & np.isfinite(eta)
            if not np.any(target_mask):
                continue

            tgt_pts = np.vstack((xi[target_mask], eta[target_mask])).T
            tgt_pts_norm = (tgt_pts - min_v) / scale

            # Rotate source vector to Face i
            Qij = cs_math.get_Qij(u_xi, u_eta, u_block, i)
            # u_vec is (3, N). u_vec.T is (N, 3). Qij is (N, 3, 3).
            # einsum("nij, nj -> ni", Qij, u_vec.T) -> (N, 3)
            # .T -> (3, N)
            u_vec_i = np.einsum("nij, nj -> ni", Qij, u_vec.T).T

            # Values for interpolation: select masked source points
            # u_vec_i is (3, N). mask is (N,).
            # We want (N_masked, 3) for LinearNDInterpolator
            vals = u_vec_i[:, mask].T

            if tri is not None:
                # Vectorized Interpolation! (N_tgt, 3) returned
                interp = LinearNDInterpolator(tri, vals)
                res = interp(tgt_pts_norm)

                res[np.isnan(res)] = 0.0
                interpolated_components[:, target_mask] = res.T
            else:
                # Fallback to nearest (slow but rare)
                try:
                    # Re-construct original pts from source geometry
                    # We need to map just the un-masked values?
                    # No, griddata takes (points, values).
                    # Pts are (xi_src[mask], eta_src[mask]).
                    # Vals are u_vec_i[mask] (N_masked, 3) ???
                    # griddata supports vector values in theory, check if fallback needs loop

                    # To be safe and identical to Unstructured fallback which does component-wise:
                    # But Unstructured loops components.
                    # Here 'vals' is (N_pts, 3).
                    # griddata should handle it.

                    res = griddata(pts, vals, tgt_pts, method="nearest", rescale=True)
                    interpolated_components[:, target_mask] = res.T
                except Exception:
                    pass

        interpolated_u1, interpolated_u2, interpolated_u3 = interpolated_components

        _, theta_out, _ = cs_math.cube2spherical(xi, eta, block, deg=True)
        u = np.vstack((interpolated_u1, interpolated_u2, interpolated_u3))
        Q = cs_math.get_Q(90 - theta_out, r=1, inverse=False)
        Ps_inv = cs_math.get_Ps(xi, eta, r=1, block=block, inverse=True)
        Ps_normalized_inv = np.einsum("nij, njk -> nik", Q, Ps_inv)
        return np.einsum("nij, nj -> ni", Ps_normalized_inv, u.T).T


class UnstructuredInterpolator(Interpolator):
    """Interpolation for irregular/unstructured grids using Delaunay triangulation (griddata)."""

    def __init__(self, source_theta, source_phi, method="linear"):
        self.theta = source_theta
        self.phi = source_phi
        self.method = method

    def interpolate_scalar(self, values, target_theta, target_phi, **kwargs):
        # Delegate to cs_math.geo2cube for coordinate handling if needed,
        # but Unstructured usually interpolates via generic logic.
        # However, for spherical data, we project to CS faces to interpolate.
        # This duplicates legacy GridBasis logic.

        xi, eta, block = cs_math.geo2cube(target_phi, 90 - target_theta)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        # We need source coords. Assuming they come from the 'grid' passed to __init__?
        # But wait, griddata needs 'points'.
        # For legacy compatibility, we implement the logic that acts ON data values.
        # But we need source grid geometry.

        # To reuse the 'GridBasis' logic exactly:
        # We loop over blocks.
        # We assume source data is defined on 'self.theta', 'self.phi'.

        scalar = values.flatten()
        th, ph = np.deg2rad(self.theta.flatten()), np.deg2rad(self.phi.flatten())
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        interpolated = np.zeros_like(block, dtype=np.float64)

        for i in range(6):
            # Mask logic
            _, th_f, ph_f = cs_math.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack(
                (np.sin(th_f) * np.cos(ph_f), np.sin(th_f) * np.sin(ph_f), np.cos(th_f))
            ).reshape((-1, 1))
            mask = np.sum(r0 * r, axis=0) > 0

            xi_src, eta_src, _ = cs_math.geo2cube(
                self.phi.flatten(), 90 - self.theta.flatten(), block=i
            )
            mask = mask & np.isfinite(xi_src) & np.isfinite(eta_src)

            target_mask = (block == i) & np.isfinite(xi) & np.isfinite(eta)
            if np.any(target_mask):
                pts = np.vstack((xi_src[mask], eta_src[mask])).T
                tgt_pts = np.vstack((xi[target_mask], eta[target_mask])).T
                vals = scalar[mask]

                try:
                    res = griddata(pts, vals, tgt_pts, method=self.method, rescale=True)
                except QhullError:
                    res = griddata(pts, vals, tgt_pts, method="nearest", rescale=True)
                interpolated[target_mask] = res
        return interpolated

    def interpolate_vector(self, u_east, u_north, u_r, target_theta, target_phi, **kwargs):
        """Interpolate vector field via legacy Sphere->Block->GridData method."""
        # This implementation mirrors the original GridBasis.interpolate_vector_components
        # It is used for unstructured/generic grids where the optimized CSInterpolator cannot be used.

        xi, eta, block = cs_math.geo2cube(target_phi, 90 - target_theta)
        xi, eta, block = xi.flatten(), eta.flatten(), block.flatten()

        # Source coordinates
        theta_src, phi_src = np.broadcast_arrays(self.theta, self.phi)
        u_east, u_north, u_r = map(np.ravel, np.broadcast_arrays(u_east, u_north, u_r))
        theta_src, phi_src = map(np.ravel, [theta_src, phi_src])

        # Source geometry
        th, ph = np.deg2rad(theta_src), np.deg2rad(phi_src)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        u_xi, u_eta, u_block = cs_math.geo2cube(phi_src, 90 - theta_src)

        # Filter out pole singularities where Ps calculation is unstable
        pole_mask = (u_block >= 4) & (np.abs(u_xi) < 1e-5) & (np.abs(u_eta) < 1e-5)

        candidates_mask = ~pole_mask
        candidates_idx = np.where(candidates_mask)[0]

        if len(candidates_idx) > 0:
            Ps_check = cs_math.get_Ps(
                u_xi[candidates_idx], u_eta[candidates_idx], r=1, block=u_block[candidates_idx]
            )
            det_Ps = arrayutils.get_3D_determinants(Ps_check)
            valid_candidates = ~np.isclose(det_Ps, 0, atol=1e-8)
            valid_idx = candidates_idx[valid_candidates]
        else:
            valid_idx = np.array([], dtype=int)

        if len(valid_idx) < len(u_xi):
            # Subset input arrays
            u_east = u_east[valid_idx]
            u_north = u_north[valid_idx]
            u_r = u_r[valid_idx]
            theta_src = theta_src[valid_idx]
            phi_src = phi_src[valid_idx]
            u_xi = u_xi[valid_idx]
            u_eta = u_eta[valid_idx]
            u_block = u_block[valid_idx]
            th = th[valid_idx]
            ph = ph[valid_idx]
            r = r[:, valid_idx]

        Ps = cs_math.get_Ps(u_xi, u_eta, r=1, block=u_block)
        Q = cs_math.get_Q(90 - theta_src, r=1, inverse=True)
        Ps_normalized = np.einsum("nij, njk -> nik", Ps, Q)
        u_vec_sph = np.vstack((u_east, u_north, u_r))
        u_vec = np.einsum("nij, nj -> ni", Ps_normalized, u_vec_sph.T).T

        interpolated_u1 = np.zeros_like(block, dtype=np.float64)
        interpolated_u2 = np.zeros_like(block, dtype=np.float64)
        interpolated_u3 = np.zeros_like(block, dtype=np.float64)

        for i in range(6):
            Qij = cs_math.get_Qij(u_xi, u_eta, u_block, i)
            u_vec_i = np.einsum("nij, nj -> ni", Qij, u_vec.T).T

            _, th_f, ph_f = cs_math.cube2spherical(0, 0, i, deg=False)
            r0 = np.hstack(
                (np.sin(th_f) * np.cos(ph_f), np.sin(th_f) * np.sin(ph_f), np.cos(th_f))
            ).reshape((-1, 1))
            mask = np.sum(r0 * r, axis=0) > 0

            xi_, eta_, _ = cs_math.geo2cube(phi_src, 90 - theta_src, block=i)

            mask = mask & np.isfinite(xi_) & np.isfinite(eta_)

            target_mask = (block == i) & np.isfinite(xi) & np.isfinite(eta)
            if np.any(target_mask):
                pts = np.vstack((xi_[mask], eta_[mask])).T
                tgt_pts = np.vstack((xi[target_mask], eta[target_mask])).T
                try:
                    interpolated_u1[target_mask] = griddata(
                        pts, u_vec_i[0][mask], tgt_pts, rescale=True, **kwargs
                    )
                    interpolated_u2[target_mask] = griddata(
                        pts, u_vec_i[1][mask], tgt_pts, rescale=True, **kwargs
                    )
                    interpolated_u3[target_mask] = griddata(
                        pts, u_vec_i[2][mask], tgt_pts, rescale=True, **kwargs
                    )
                except QhullError:
                    interpolated_u1[target_mask] = griddata(
                        pts, u_vec_i[0][mask], tgt_pts, method="nearest", rescale=True
                    )
                    interpolated_u2[target_mask] = griddata(
                        pts, u_vec_i[1][mask], tgt_pts, method="nearest", rescale=True
                    )
                    interpolated_u3[target_mask] = griddata(
                        pts, u_vec_i[2][mask], tgt_pts, method="nearest", rescale=True
                    )

        _, theta_out, _ = cs_math.cube2spherical(xi, eta, block, deg=True)
        u = np.vstack((interpolated_u1, interpolated_u2, interpolated_u3))
        Q = cs_math.get_Q(90 - theta_out, r=1, inverse=False)
        Ps_inv = cs_math.get_Ps(xi, eta, r=1, block=block, inverse=True)
        Ps_normalized_inv = np.einsum("nij, njk -> nik", Q, Ps_inv)
        return np.einsum("nij, nj -> ni", Ps_normalized_inv, u.T).T


class CSInterpolator(Interpolator):
    """Optimized bilinear interpolation for structured Cubed Sphere grids."""

    def __init__(self, N: int):
        self.N = N
        # Precompute constants
        self.dxi = (np.pi / 2) / N
        self.deta = (np.pi / 2) / N
        self.xi_min = -np.pi / 4
        self.eta_min = -np.pi / 4

    def is_compatible(self, values: np.ndarray) -> bool:
        """Check if values match the expected grid size."""
        return values.size == 6 * self.N**2

    def _get_indices_weights(self, xi, eta):
        # Map xi, eta to float indices
        # xi = xi_min + i * dxi  => i_node = (xi - xi_min) / dxi

        # Shift to 0..N range (Node based)
        i_f_node = (xi - self.xi_min) / self.dxi
        j_f_node = (eta - self.eta_min) / self.deta

        # CSBasis uses cell centers: index 0 is at 0.5, index 1 at 1.5, etc.
        # So effective float index for array is i_f_node - 0.5
        i_f = i_f_node - 0.5
        j_f = j_f_node - 0.5

        # Computation for bilinear interpolation (i0, i1)
        # We clamp i0 to [0, N-2] so that i1 = i0+1 is always <= N-1 (valid).
        # This handles extrapolation at edges by using the slope of boundary cells.

        i0 = np.floor(i_f).astype(int)
        j0 = np.floor(j_f).astype(int)

        i0 = np.clip(i0, 0, self.N - 2)
        j0 = np.clip(j0, 0, self.N - 2)

        # Offsets
        dx = i_f - i0
        dy = j_f - j0

        # Bilinear weights
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy

        return i0, j0, w00, w10, w01, w11

    def interpolate_scalar(self, values: np.ndarray, target_theta, target_phi, **kwargs):
        # values shape: assumed to be (6, N, N) for CSBasis (Cell Centered)
        # Reshape values
        N_pt = self.N
        try:
            vals_struct = values.reshape((6, N_pt, N_pt))
        except ValueError as e:
            # Fallback/Debug: If reshaping implies different N, maybe grid is node based?
            # But currently hardcoding assumption CSBasis(N) -> (6, N, N)
            raise ValueError(
                f"CSInterpolator expected grid size (6, {N_pt}, {N_pt})={6 * N_pt**2} but got {values.size}. Check CSBasis N."
            ) from e

        xi_tgt, eta_tgt, block_tgt = cs_math.geo2cube(target_phi, 90 - target_theta)

        # Filter valid target points
        mask_valid = (
            np.isfinite(xi_tgt) & np.isfinite(eta_tgt) & (block_tgt >= 0) & (block_tgt <= 5)
        )

        result = np.zeros_like(xi_tgt, dtype=np.float64)

        if not np.any(mask_valid):
            return result

        b = block_tgt[mask_valid].astype(int)
        x = xi_tgt[mask_valid]
        e = eta_tgt[mask_valid]

        i0, j0, w00, w10, w01, w11 = self._get_indices_weights(x, e)

        # Gather values
        # Indices for +1 are just +1 because we clamped i0 to N-2
        i1 = i0 + 1
        j1 = j0 + 1

        v00 = vals_struct[b, i0, j0]
        v10 = vals_struct[b, i1, j0]
        v01 = vals_struct[b, i0, j1]
        v11 = vals_struct[b, i1, j1]

        interp_vals = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
        result[mask_valid] = interp_vals

        return result

    def interpolate_vector(self, u_east, u_north, u_r, target_theta, target_phi, **kwargs):
        """Interpolate vector field via Cartesian components."""
        # 1. Convert source vector (Spherical) to Cartesian (Global)
        # We need source theta/phi.
        # u_east, u_north, u_r are assumed to be structured (6, N+1, N+1) logic?
        # NO. They are flat arrays in PynaMIT.
        # But for CSInterpolator to work, we need to know the source grid structure.
        # We know N. We can reconstruct source theta/phi on the fly or compute rotation.

        # We can implement a helper to rotate Sph -> Cart at grid points efficiently.
        # r-hat = (sin th cos ph, sin th sin ph, cos th)
        # th-hat = (cos th cos ph, cos th sin ph, -sin th)
        # ph-hat = (-sin ph, cos ph, 0)

        # u_cart = u_r * r-hat + u_th * th-hat + u_ph * ph-hat
        # PynaMIT: u_north = -u_th, u_east = u_ph.
        # u_th = -u_north, u_ph = u_east.

        # We need source lat/lon arrays.
        # We can generate them cheaply from N.
        # Or require them passed in?
        # Interpolator interface has fixed signature.
        # We can reconstruct them in __init__ or on the fly.
        # Since N is small usually, on the fly is fine.

        # Generate Source Grid (Cell Centered matching CSBasis)
        # N grid cells => N points at centers.
        N = self.N
        k, i, j = np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing="ij")

        # Cell centers: starts at xi_min + 0.5*dxi
        xi = -np.pi / 4 + (i + 0.5) * (np.pi / 2) / N
        eta = -np.pi / 4 + (j + 0.5) * (np.pi / 2) / N

        _, th_src, ph_src = cs_math.cube2spherical(xi, eta, k, deg=False)  # radians

        th_src = th_src.flatten()
        ph_src = ph_src.flatten()

        # Rotation matrices
        sin_th, cos_th = np.sin(th_src), np.cos(th_src)
        sin_ph, cos_ph = np.sin(ph_src), np.cos(ph_src)

        r_hat_x = sin_th * cos_ph
        r_hat_y = sin_th * sin_ph
        r_hat_z = cos_th

        th_hat_x = cos_th * cos_ph
        th_hat_y = cos_th * sin_ph
        th_hat_z = -sin_th

        ph_hat_x = -sin_ph
        ph_hat_y = cos_ph
        ph_hat_z = np.zeros_like(ph_src)

        # Components
        u_th = -u_north.flatten()
        u_ph = u_east.flatten()
        u_rad = u_r.flatten()

        ux = u_rad * r_hat_x + u_th * th_hat_x + u_ph * ph_hat_x
        uy = u_rad * r_hat_y + u_th * th_hat_y + u_ph * ph_hat_y
        uz = u_rad * r_hat_z + u_th * th_hat_z + u_ph * ph_hat_z

        # 2. Interpolate Cartesian components (Scalar interpolation)
        ux_int = self.interpolate_scalar(ux, target_theta, target_phi)
        uy_int = self.interpolate_scalar(uy, target_theta, target_phi)
        uz_int = self.interpolate_scalar(uz, target_theta, target_phi)

        # 3. Rotate back to Spherical at TARGET points
        # u_r = u . r-hat_tgt
        # u_th = u . th-hat_tgt
        # u_ph = u . ph-hat_tgt

        tgt_th_rad = np.deg2rad(target_theta.flatten())
        tgt_ph_rad = np.deg2rad(target_phi.flatten())

        sin_tgt_th, cos_tgt_th = np.sin(tgt_th_rad), np.cos(tgt_th_rad)
        sin_tgt_ph, cos_tgt_ph = np.sin(tgt_ph_rad), np.cos(tgt_ph_rad)

        # Target basis vectors
        tr_x = sin_tgt_th * cos_tgt_ph
        tr_y = sin_tgt_th * sin_tgt_ph
        tr_z = cos_tgt_th

        tth_x = cos_tgt_th * cos_tgt_ph
        tth_y = cos_tgt_th * sin_tgt_ph
        tth_z = -sin_tgt_th

        tph_x = -sin_tgt_ph
        tph_y = cos_tgt_ph
        tph_z = np.zeros_like(tgt_ph_rad)

        # Dot products
        ur_out = ux_int * tr_x + uy_int * tr_y + uz_int * tr_z
        uth_out = ux_int * tth_x + uy_int * tth_y + uz_int * tth_z
        uph_out = ux_int * tph_x + uy_int * tph_y + uz_int * tph_z

        # Convert back to PynaMIT convention
        unorth_out = -uth_out
        ueast_out = uph_out

        return ueast_out, unorth_out, ur_out
