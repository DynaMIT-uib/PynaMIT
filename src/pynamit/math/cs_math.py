"""Generic Cubed Sphere Coordinate Mathematics.

This module contains pure functions for performing coordinate transformations
and matrix calculations related to the Cubed Sphere grid. It is designed to be
used by GridBasis, CSInterpolator, and other modules without creating circular
dependencies.
"""

import numpy as np
from pynamit.math import arrayutils


def block(lon, lat):
    """Determine cube faces (blocks) of spherical coordinates."""
    lon, lat = np.broadcast_arrays(lon, lat)
    lat, lon = lat.flatten(), lon.flatten()
    th, ph = np.deg2rad(90 - lat), np.deg2rad(lon)
    xyz = np.vstack((np.cos(ph) * np.sin(th), np.sin(th) * np.sin(ph), np.cos(th)))
    face_midpoints = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    distances = np.empty((6, xyz.shape[1]))
    for i in range(6):
        distances[i] = np.linalg.norm(xyz - face_midpoints[i].reshape((3, 1)), axis=0)
    safety_distance = 1e-10
    blocks = np.zeros(xyz.shape[1], dtype=int)
    for i in range(6):
        blocks[distances[i] < np.choose(blocks, distances) - safety_distance] = i
    return blocks


# Alias for internal use to avoid shadowing by argument name 'block'
def compute_block(lon, lat):
    return block(lon, lat)


def geo2cube(lon, lat, block=None):
    """Convert geocentric coordinates to cube coordinates."""
    lon, lat = np.broadcast_arrays(lon, lat)
    shape = lon.shape
    N_points = lon.size

    if block is None:
        block_idx = compute_block(lon, lat)
    else:
        block_idx = block * np.ones_like(lat, dtype=int)

    block_idx, lon, lat = block_idx.flatten(), lon.flatten(), lat.flatten()
    X, Y, xi, eta = np.empty(N_points), np.empty(N_points), np.empty(N_points), np.empty(N_points)
    theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)

    mask = block_idx == 0
    X[mask] = np.tan(phi[mask])
    Y[mask] = 1 / (np.tan(theta[mask]) * np.cos(phi[mask]))
    mask = block_idx == 1
    X[mask] = np.tan(phi[mask] - np.pi / 2)
    Y[mask] = 1 / (np.tan(theta[mask]) * np.sin(phi[mask]))
    mask = block_idx == 2
    X[mask] = np.tan(phi[mask] - np.pi)
    Y[mask] = -1 / (np.tan(theta[mask]) * np.cos(phi[mask]))
    mask = block_idx == 3
    X[mask] = np.tan(phi[mask] - 3 * np.pi / 2)
    Y[mask] = -1 / (np.tan(theta[mask]) * np.sin(phi[mask]))

    mask = block_idx == 4
    X[mask] = np.tan(theta[mask]) * np.sin(phi[mask])
    Y[mask] = -np.tan(theta[mask]) * np.cos(phi[mask])

    mask = block_idx == 5
    X[mask] = -np.tan(theta[mask]) * np.sin(phi[mask])
    Y[mask] = -np.tan(theta[mask]) * np.cos(phi[mask])

    xi, eta = np.arctan(X), np.arctan(Y)
    return xi.reshape(shape), eta.reshape(shape), block_idx.reshape(shape)


def get_delta(xi, eta):
    xi, eta = np.broadcast_arrays(xi, eta)
    return 1 + np.tan(xi) ** 2 + np.tan(eta) ** 2


def cube2cartesian(xi, eta, r=1, block=0):
    xi, eta, r, block_idx = np.broadcast_arrays(xi, eta, r, block)
    delta = get_delta(xi, eta)
    x, y, z = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)

    iii = block_idx == 0
    x[iii] = r[iii] / np.sqrt(delta[iii])
    y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
    z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
    iii = block_idx == 1
    x[iii] = -r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
    y[iii] = r[iii] / np.sqrt(delta[iii])
    z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
    iii = block_idx == 2
    x[iii] = -r[iii] / np.sqrt(delta[iii])
    y[iii] = -r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
    z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
    iii = block_idx == 3
    x[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
    y[iii] = -r[iii] / np.sqrt(delta[iii])
    z[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])

    iii = block_idx == 4
    x[iii] = -r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
    y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
    z[iii] = r[iii] / np.sqrt(delta[iii])
    iii = block_idx == 5
    x[iii] = r[iii] * np.tan(eta[iii]) / np.sqrt(delta[iii])
    y[iii] = r[iii] * np.tan(xi[iii]) / np.sqrt(delta[iii])
    z[iii] = -r[iii] / np.sqrt(delta[iii])
    return x, y, z


def cube2spherical(xi, eta, block, r=1, deg=False):
    xi, eta = np.float64(xi), np.float64(eta)
    xi, eta, r, block_idx = np.broadcast_arrays(xi, eta, r, block)
    x, y, z = cube2cartesian(xi, eta, r, block_idx)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    if deg:
        phi, theta = np.rad2deg(phi), np.rad2deg(theta)
    return r, theta, phi


def get_Pc(xi, eta, r=1, block=0, inverse=False):
    xi, et, r, block_idx = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
    delta = get_delta(xi, et)
    Pc = np.empty((delta.size, 3, 3))
    rsec2xi = r / np.cos(xi) ** 2
    rsec2et = r / np.cos(et) ** 2

    # Block 0
    iii = block_idx == 0
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
    iii = block_idx == 1
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
    iii = block_idx == 2
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
    iii = block_idx == 3
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
    iii = block_idx == 4
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
    iii = block_idx == 5
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


def get_Ps(xi, eta, r=1, block=0, inverse=False):
    xi, et, r, block_idx = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
    delta = get_delta(xi, et)
    Ps = np.empty((delta.size, 3, 3))
    # Blocks 0-3 same logic
    mask_eq = block_idx <= 3
    if np.any(mask_eq):
        idx = np.where(mask_eq)
        Ps[idx, 0, 0] = 1
        Ps[idx, 0, 1] = 0
        Ps[idx, 0, 2] = 0
        Ps[idx, 1, 0] = np.tan(xi[idx]) * np.sin(et[idx]) * np.cos(et[idx])
        Ps[idx, 1, 1] = np.cos(xi[idx]) * np.sin(et[idx]) ** 2 + np.cos(et[idx]) ** 2 / np.cos(
            xi[idx]
        )
        Ps[idx, 1, 2] = 0
        Ps[idx, 2, 0] = 0
        Ps[idx, 2, 1] = 0
        Ps[idx, 2, 2] = 1

    iii = block_idx == 4
    if np.any(iii):
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

    iii = block_idx == 5
    if np.any(iii):
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
    return Ps


def get_Qij(xi, eta, block_i, block_j):
    xi_i, eta_i, block_i, block_j = map(np.ravel, np.broadcast_arrays(xi, eta, block_i, block_j))
    Psi_inv = get_Ps(xi_i, eta_i, r=1, block=block_i, inverse=True)
    r, theta, phi = cube2spherical(xi_i, eta_i, r=1, block=block_i, deg=True)
    xi_j, eta_j, _ = geo2cube(phi, 90 - theta, block=block_j)
    Psj = get_Ps(xi_j, eta_j, r=1, block=block_j)
    return np.einsum("nij, njk -> nik", Psj, Psi_inv)


def get_Q(lat, r, inverse=False):
    lat, r = map(np.ravel, np.broadcast_arrays(lat, r))
    Q = np.zeros((lat.size, 3, 3), dtype=np.float64)
    Q[:, 0, 0] = r * np.cos(np.deg2rad(lat))
    Q[:, 1, 1] = r
    Q[:, 2, 2] = 1
    if inverse:
        return arrayutils.invert_3D_matrices(Q)
    return Q


def get_metric_tensor(xi, eta, r=1, block=0, covariant=True):
    """Calculate the metric tensor of the cubed sphere system."""
    if covariant:
        # Calculate at r=1 to ensure stable inversion
        Pc = get_Pc(xi, eta, r=1.0, block=block)
        # g^ij = sum_k (dxi^i/dX^k) (dxi^j/dX^k)
        g_inv_1 = np.einsum("nik, njk -> nij", Pc, Pc)
        g_cov_1 = arrayutils.invert_3D_matrices(g_inv_1)
        
        # Scale by r^2
        r = np.array(r)
        if r.size == 1:
            return g_cov_1 * r**2
        else:
            flat_r = r.flatten() if r.size == g_cov_1.shape[0] else np.broadcast_to(r, (g_cov_1.shape[0],))
            return g_cov_1 * (flat_r.reshape(-1, 1, 1) ** 2)
    else:
        # Contravariant (Inverse Metric)
        Pc = get_Pc(xi, eta, r=r, block=block)
        g_inv = np.einsum("nik, njk -> nij", Pc, Pc)
        return g_inv
