"""
Gaunt engine for real spherical harmonics.

Calculates the triple integral of three real spherical harmonics:
C(i, j, k) = integral(Y_i * Y_j * Y_k dOmega)

Governed by selection rules:
1. m_k in {m_i + m_j, m_i - m_j, m_j - m_i, -m_i - m_j} (approx)
2. |l_i - l_j| <= l_k <= l_i + l_j
3. l_i + l_j + l_k is even
"""

import numpy as np
import scipy.special
from typing import Tuple, List, Dict

def get_real_sh_gaunt_coefficients(l1, m1, l2, m2, l3, m3):
    """
    Placeholder for analytical Gaunt coefficients for Real SH.
    In practice, numerical integration is often more robust for Real SH 
    due to the mixing of sin/cos terms.
    """
    # TODO: Implement analytical version via Wigner-3j if needed.
    pass

class GauntEngine:
    """Engine for computing and caching the triple integrals of real SH."""
    
    def __init__(self, basis: 'SHBasis', grid_resolution: int = None):
        """
        Initialize the Gaunt engine.
        Using a quadrature grid to pre-compute the basis values for fast integration.
        """
        self.basis = basis
        self.Nmax = basis.Nmax
        res = grid_resolution or int(1.5 * self.Nmax + 2)
        # Ensure res is even for consistency with GLBasis and to avoid magnetic equator
        if res % 2 != 0:
            res += 1
        
        # Gauss-Legendre quadrature in theta
        x, w = np.polynomial.legendre.leggauss(res)
        self.theta_quad = np.arccos(x)
        self.w_theta = w # integral sin(th) dth = integral dx
        
        # Uniform sampling in phi
        self.phi_quad = np.linspace(0, 2 * np.pi, 2 * res + 1, endpoint=False)
        self.w_phi = 2 * np.pi / (2 * res + 1)
        
        # Pre-evaluate basis on the quadrature grid
        th_mesh, ph_mesh = np.meshgrid(self.theta_quad, self.phi_quad, indexing='ij')
        from pynamit.primitives.grid import Grid
        self.quad_grid = Grid(theta=np.rad2deg(th_mesh.flatten()), phi=np.rad2deg(ph_mesh.flatten()))
        
        # Weights for integration: sin(theta) dtheta dphi -> weights * w_phi
        self.weights = (np.tile(self.w_theta[:, None], (1, 2 * res + 1)) * self.w_phi).flatten()
        
        # L = self.basis.index_length
        # Pre-compute Scalar and Vector Evaluation Matrices
        self.G_scalar = self.basis.get_G(self.quad_grid).reshape(self.quad_grid.size, -1).T # (L, Q)
        
        # Mass matrix D = integral Y Y^T
        # For perfectly orthogonal basis, this is diagonal.
        # But we compute the full one (or diagonal) for robustness.
        # D_scalar: (L, L)
        self.D_scalar = self.G_scalar @ (self.weights[:, None] * self.G_scalar.T)
        self.D_scalar_inv = np.linalg.inv(self.D_scalar)
        
    def get_interaction_matrix(self, coeffs_sigma: np.ndarray) -> np.ndarray:
        """
        Build the spectral interaction matrix M(Sigma) such that:
        c_coeffs = M @ b_coeffs  maps coefficients to coefficients.
        
        M = D^-1 @ integral(Y * Sigma * Y^T)
        """
        # 1. Evaluate Sigma on quadrature grid
        sigma_quad = self.G_scalar.T @ coeffs_sigma
        
        # 2. Build Galerkin matrix
        # weighted_G[i, q] = Y_i(q) * Sigma(q) * W_q
        weighted_G = self.G_scalar * (sigma_quad * self.weights)
        M_galerkin = weighted_G @ self.G_scalar.T
        
        # 3. Project to get coefficient mapping
        return self.D_scalar_inv @ M_galerkin

    def get_vector_interaction_matrix(self, tensor_sigma_quad: np.ndarray) -> np.ndarray:
        """
        Build the vector interaction matrix mapping VSH coefficients to VSH coefficients.
        J_coeffs = M_VSH @ E_coeffs
        
        tensor_sigma_quad: (2, 2, Q)
        """
        # 1. Get Vector Basis Matrix on quadrature grid: (2, Q, 2L)
        # Components: [-grad Y, rxgrad Y]
        G_vsh = self.basis.get_vector_basis_matrix(self.quad_grid) # (2, Q, 2L)
        L = self.basis.index_length
        Q = self.quad_grid.size
        
        G_vsh_flat = G_vsh.reshape(2 * Q, 2 * L)
        
        # 2. Vector Mass Matrix: D_vsh = integral G_i . G_j dOmega
        # G_vsh_weighted[comp, q, basis] = G[comp, q, basis] * W_q
        W_stacked = np.repeat(self.weights, 2) # (2Q,)? No.
        # Correct weighting for dot product sum over comp:
        # D_ij = sum_q W_q * (G_i(q) . G_j(q))
        
        # Simplified: Use the same pattern
        # D_vsh = sum_q W_q * G(q)^T @ G(q)
        # G(q) is 2x(2L). G(q)^T @ G(q) is (2L)x(2L).
        # We can implement this via sum of outer products or block-einsum.
        
        # einsum approach for D:
        # G_vsh: (comp_out, Q, comp_in, basis) e.g. (2, Q, 2, L)
        # D_ij = sum_q W_q sum_d G_d q i m * G_d q j n -> (i, m, j, n)
        D_vsh_multi = np.einsum('o q i m, q, o q j n -> i m j n', G_vsh, self.weights, G_vsh, optimize=True)
        D_vsh = D_vsh_multi.reshape(2 * L, 2 * L)
        D_inv = np.linalg.inv(D_vsh)
        
        # 3. Interaction part mapping E to J
        # J_basis: sum_d_in Sigma_{d_out d_in}(q) G_{d_in q d_match m} -> (d_out, q, d_match, m)
        # Sigma: (2, 2, Q). G: (2, Q, 2, L). 
        # J_basis_{o q i m} = sum_k Sigma_{o k}(q) G_{k q i m}
        J_basis = np.einsum('o k q, k q i m -> o q i m', tensor_sigma_quad, G_vsh, optimize=True)
        
        # Galerkin:
        # M_gal: sum_q W_q sum_d_out G_{d_out q i m} J_{d_out q j n} -> (i, m, j, n)
        M_gal_multi = np.einsum('o q i m, q, o q j n -> i m j n', G_vsh, self.weights, J_basis, optimize=True)
        M_gal = M_gal_multi.reshape(2 * L, 2 * L)
        
        return D_inv @ M_gal
        
        return D_inv @ M_gal

    def get_selection_rules_mask(self) -> np.ndarray:
        """Computes a sparse mask of potentially non-zero Gaunt coefficients."""
        # TODO: Implement l and m selection rules for sparser pre-computation.
        pass
