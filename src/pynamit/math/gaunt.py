"""
Gaunt engine for real spherical harmonics.
"""

import numpy as np
import scipy.special
from typing import Tuple, List, Dict
from pynamit.math.wigner import wigner_3j

def get_complex_gaunt_coeff(l1, m1, l2, m2, l3, m3):
    """Calculate the Gaunt coefficient for complex spherical harmonics."""
    factor = 4.0 * np.pi
    w3j_0 = wigner_3j(l1, l2, l3, 0, 0, 0)
    if w3j_0 == 0: return 0.0
    w3j_m = wigner_3j(l1, l2, l3, m1, m2, m3)
    return factor * w3j_0 * w3j_m

def get_spin_weighted_gaunt_coeff(l1, m1, s1, l2, m2, s2, l3, m3, s3):
    """Integral( s1_Y_l1m1 * s2_Y_l2m2 * s3_Y_l3m3 )"""
    from pynamit.math.wigner import wigner_3j
    return np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4.0*np.pi)) * \
           wigner_3j(l1, l2, l3, m1, m2, m3) * \
           wigner_3j(l1, l2, l3, -s1, -s2, -s3)

def get_real_decomposition(l, m):
    """Decomposition of Real SH Y_lm into Complex SH Y_l,mu."""
    if m == 0: return [(1.0, 0)]
    inv_sr2 = 1.0/np.sqrt(2)
    if m > 0:
        return [(inv_sr2, -m), ((-1)**m * inv_sr2, m)]
    else:
        k = abs(m)
        return [(1j * inv_sr2, -k), (-1j * (-1)**k * inv_sr2, k)]

class GauntEngine:
    """Engine for computing and caching the triple integrals of real SH."""
    
    _KERNEL_CACHE = {}

    def __init__(self, basis, grid_resolution=None):
        self.basis = basis
        self.Nmax = basis.Nmax
        self._elsasser_reduced_cache = {}
        # Increase resolution for better integration of complex sigma
        res = grid_resolution or int(3.0 * self.Nmax + 10)
        if res % 2 != 0: res += 1
        x, w = np.polynomial.legendre.leggauss(res)
        self.theta_quad = np.arccos(x)
        self.w_theta = w
        self.phi_quad = np.linspace(0, 2 * np.pi, 2 * res + 1, endpoint=False)
        self.w_phi = 2 * np.pi / (2 * res + 1)
        th_mesh, ph_mesh = np.meshgrid(self.theta_quad, self.phi_quad, indexing='ij')
        from pynamit.primitives.grid import Grid
        self.quad_grid = Grid(theta=np.rad2deg(th_mesh.flatten()), phi=np.rad2deg(ph_mesh.flatten()))
        self.weights = (np.tile(self.w_theta[:, None], (1, 2 * res + 1)) * self.w_phi).flatten()
        self.G_scalar = self.basis.get_G(self.quad_grid).reshape(self.quad_grid.size, -1).T
        self.D_scalar = self.G_scalar @ (self.weights[:, None] * self.G_scalar.T)
        self.D_scalar_inv = np.linalg.inv(self.D_scalar)
        
    def _compute_elsasser_gl_raw(self, li, mi, lj, mj, lk, mk):
        def get_idx(l, m):
            indices = np.where((self.basis.n == l) & (self.basis.m == abs(m)))[0]
            if len(indices) == 0: return -1
            return indices[0] if m >= 0 else indices[1]
        ii, ij, ik = get_idx(li, mi), get_idx(lj, mj), get_idx(lk, mk)
        if ii == -1 or ij == -1 or ik == -1: return 0.0
        G_th = self.basis.get_G(self.quad_grid, derivative="theta")
        G_ph = self.basis.get_G(self.quad_grid, derivative="phi")
        # Integral Y_i (dTh_j dPh_k - dPh_j dTh_k) sin theta
        # This is exactly E_ijk
        return np.sum(self.G_scalar[ii,:] * (G_th[:,ij]*G_ph[:,ik] - G_ph[:,ij]*G_th[:,ik]) * self.weights)

    def elsasser_coefficient(self, l1, m1, l2, m2, l3, m3):
        """Compute the Elsasser integral interaction E_ijk.
        
        Currently uses the exact Gauss-Legendre quadrature (valid to precision)
        as the analytic formula involves complex Wigner factors that are difficult
        to reconstruct without the original reference code.
        """
        return self._compute_elsasser_gl_raw(l1, m1, l2, m2, l3, m3)

    def get_spin_evaluation_matrix(self, s):
        from pynamit.math.wigner import wigner_small_d
        L, Q = self.basis.index_length, self.quad_grid.size
        th, ph = np.deg2rad(self.quad_grid.theta), np.deg2rad(self.quad_grid.phi)
        G = np.zeros((Q, L), dtype=complex)
        idx = 0
        l_min = getattr(self.basis, "Nmin", 1)
        for l in range(l_min, self.Nmax + 1):
            for m in range(-l, l + 1):
                if abs(s) <= l:
                    G[:, idx] = ((-1)**s * np.sqrt((2*l+1)/(4*np.pi)) * 
                               wigner_small_d(l, m, -s, th) * np.exp(1j*m*ph))
                idx += 1
        return G

    def analyze_spin_weighted(self, s, values):
        G_s = self.get_spin_evaluation_matrix(s)
        # Quadrature integration: c = sum_q W_q Y_q* v_q
        return (G_s.conj().T * self.weights) @ values

    def _compute_projection_matrix_for_indices(self, idx_list):
        L_real, L_complex = len(idx_list), (self.Nmax + 1)**2
        U = np.zeros((L_complex, L_real), dtype=np.complex128)
        sr2_inv = 1.0/np.sqrt(2)
        seen = {}
        for k, (ln, mn) in enumerate(idx_list):
            idx_base = ln**2 + ln
            if mn == 0: U[idx_base, k] = 1.0
            else:
                count = seen.get((ln, mn), 0); seen[(ln, mn)] = count + 1
                phase = (-1.0)**mn
                if count == 0: U[idx_base-mn, k], U[idx_base+mn, k] = sr2_inv, sr2_inv * phase
                else: U[idx_base-mn, k], U[idx_base+mn, k] = 1j*sr2_inv, -1j*sr2_inv * phase
        return U

    def _get_spin2_kernel_tensors(self):
        """Precompute or retrieve the Spin-2 interaction tensor kernels."""
        # Check class-level cache first
        key = (self.Nmax, self.basis.Mmax, getattr(self.basis, 'Nmin', 1))
        if key in self._KERNEL_CACHE:
            return self._KERNEL_CACHE[key]
            
        # We need Nmin=0 basis for the Sigma Coefficients
        from pynamit.spherical_harmonics.sh_basis import SHBasis # Local import
        sigma_basis = SHBasis(self.Nmax, self.basis.Mmax, Nmin=0, quasi_normalized=self.basis.is_normalized, backend="internal")
        L_sig = sigma_basis.index_length
        L_vec = self.basis.index_length
        L_phys = 2 * L_vec
        
        # Return Tensors in Physical Vector Basis (2L_vec x 2L_vec)
        T_pm = np.zeros((L_phys, L_phys, L_sig), dtype=np.complex128)
        T_mp = np.zeros((L_phys, L_phys, L_sig), dtype=np.complex128)
        
        # Pre-compute Spin matrices once
        G_p2 = self.get_spin_evaluation_matrix(2)
        G_m2 = self.get_spin_evaluation_matrix(-2)
        
        for k in range(L_sig):
            l_k = sigma_basis.n[k]
            if l_k < 2: continue
            
            # --- T_pm ---
            val_p2_k = G_p2[:, k-1]
            sigma_k = np.zeros((2, 2, self.quad_grid.size), dtype=complex)
            sigma_k[0,0,:] = 0.5 * val_p2_k
            sigma_k[1,1,:] = -0.5 * val_p2_k
            sigma_k[0,1,:] = 0.5j * val_p2_k
            sigma_k[1,0,:] = 0.5j * val_p2_k
            
            M_vec = self.get_vector_interaction_matrix(sigma_k)
            # Store directly (Physical Basis)
            T_pm[:,:,k] = M_vec
            
            # --- T_mp ---
            val_m2_k = G_m2[:, k-1]
            sigma_k[:] = 0
            sigma_k[0,0,:] = 0.5 * val_m2_k
            sigma_k[1,1,:] = -0.5 * val_m2_k
            sigma_k[0,1,:] = -0.5j * val_m2_k
            sigma_k[1,0,:] = -0.5j * val_m2_k
            
            M_vec = self.get_vector_interaction_matrix(sigma_k)
            T_mp[:,:,k] = M_vec
            
        self._KERNEL_CACHE[key] = (T_pm, T_mp)
        return T_pm, T_mp

    def _compute_term(self, l1, s1, l2, s2, s3, m1, m2, coeffs):
        val = 0.0j; m3 = m1 - m2; N_c = int(np.sqrt(len(coeffs))) - 1
        for l3 in range(abs(m3), N_c + 1):
            if abs(l1-l2) <= l3 <= l1+l2:
                idx = l3**2+l3+m3
                if 0 <= idx < len(coeffs):
                    c = coeffs[idx]
                    if abs(c) > 1e-12:
                        val += c * get_spin_weighted_gaunt_coeff(l1, -m1, -s1, l2, m2, s2, l3, m3, s3) * ((-1.0)**(m1+s1))
        return val

    def get_general_analytic_interaction_matrix(self, c_pp, c_mm, c_pm, c_mp, input_is_complex=False):
        # Always generate idx_sig if needed for non-complex input input projection
        idx_sig = []; N_sig = int(np.sqrt(len(c_pp))) - 1
        
        if not input_is_complex:
            # Legacy path for coefficients in Real Basis
            # Generate idx_sig (Real Order (n,0)..(n,n))
            for n in range(N_sig + 1):
                for m in range(n + 1): idx_sig.append((n, m))
            for n in range(1, N_sig + 1):
                for m in range(1, n + 1): idx_sig.append((n, m))
                
            U_sig = self._compute_projection_matrix_for_indices(idx_sig)
            def to_ortho(c, idx):
                c_out = c.copy()
                for k, (l, m) in enumerate(idx): c_out[k] *= np.sqrt(4.0*np.pi/(2*l+1))
                return c_out
            cp, cm = U_sig @ to_ortho(c_pp, idx_sig), U_sig @ to_ortho(c_mm, idx_sig)
            cpm, cmp = U_sig @ to_ortho(c_pm, idx_sig), U_sig @ to_ortho(c_mp, idx_sig)
        else: 
            # Input is already Complex Orthonormal
            cp, cm, cpm, cmp = c_pp, c_mm, c_pm, c_mp
        
        n_st, m_st = self.basis.index_arrays
        L_c = (self.Nmax + 1)**2
        
        # --- General Analytic Path (Spin 0 + Spin 2) ---
        # Construct Basis Projection for Output (Schmidt Real -> Complex Orthonormal)
        U_st = self._compute_projection_matrix_for_indices(list(zip(n_st, m_st)))
        
        # Precompute L, M for Scalar Complex Basis
        n_v, m_v = np.zeros(L_c, dtype=int), np.zeros(L_c, dtype=int)
        for l in range(self.Nmax+1):
            for m in range(-l, l+1): 
                idx = l**2+l+m; n_v[idx], m_v[idx] = l, m
                
        # Compute Spin Blocks
        # M_pp: 1 -> 1 (Spin 0 Field)
        # M_mm: -1 -> -1 (Spin 0 Field)
        # M_pm: -1 -> 1 (Spin 2 Field)
        # M_mp: 1 -> -1 (Spin -2 Field)
        M_pp, M_mm, M_pm, M_mp = [np.zeros((L_c, L_c), dtype=np.complex128) for _ in range(4)]
        
        for i in range(L_c):
            l1, m1 = n_v[i], m_v[i]
            for j in range(L_c):
                l2, m2 = n_v[j], m_v[j]
                # Isotropic Terms (Spin 0 coupling)
                M_pp[i,j] = self._compute_term(l1, 1, l2, 1, 0, m1, m2, cp)
                M_mm[i,j] = self._compute_term(l1, -1, l2, -1, 0, m1, m2, cm)
                
                # Anisotropic Terms (Spin 2 coupling)
                # M_pm corresponds to M_{+-} (Out +1, In -1). Requires Spin +2 field.
                M_pm[i,j] = self._compute_term(l1, 1, l2, -1, 2, m1, m2, cpm)
                # M_mp corresponds to M_{-+} (Out -1, In +1). Requires Spin -2 field.
                M_mp[i,j] = self._compute_term(l1, -1, l2, 1, -2, m1, m2, cmp)
        
        # Transform Spin Basis (+, -) to Physical Vector Basis (P, T)
        M_PP = 0.5 * (M_pp + M_mm - M_pm - M_mp)
        M_TT = 0.5 * (M_pp + M_mm + M_pm + M_mp)
        M_PT = -0.5j * (M_pp - M_mm + M_pm - M_mp)
        M_TP = 0.5j * (M_pp - M_mm - M_pm + M_mp)
        
        M_tot = np.block([[M_PP, M_PT], [M_TP, M_TT]])
        
        # Project Result to Real Schmidt Basis
        # U_st projects Vector_Schmidt -> Vector_Complex_Orthonormal (per component)
        # So P_f = diag(U, U).
        P_f = np.block([[U_st, np.zeros_like(U_st)], [np.zeros_like(U_st), U_st]])
        
        # M_phys = P^dagger @ M_spin_vec @ P
        # Wait, M_tot is in Vector_Complex_Orthonormal Basis (P_orth, T_orth).
        # We need Vector_Schmidt (P_real, T_real).
        M_real = (P_f.conj().T @ M_tot @ P_f).real
        
        # Scale for Schmidt Normalization
        tn = n_st; V = np.sqrt(tn*(tn+1)*4.0*np.pi/(2*tn+1)); V[V==0]=1.0
        S = np.concatenate([V, V])
        S_ratio = S[None,:] / S[:,None]
        # S_ratio = 1.0
        
        return M_real * S_ratio


    def get_vector_interaction_matrix(self, tensor_sigma_quad):
        G_vsh_P = -self.basis.get_gradient_matrix(self.quad_grid)
        G_vsh_T = self.basis.get_curl_matrix(self.quad_grid)
        G_vsh = np.concatenate([G_vsh_P, G_vsh_T], axis=2)
        L, Q = self.basis.index_length, self.quad_grid.size
        # Correct weighting D_ij = sum_q W_q G_i . G_j
        D = np.einsum('oqi, q, oqj -> ij', G_vsh, self.weights, G_vsh, optimize=True)
        D_inv = np.linalg.inv(D)
        J = np.einsum('okq, kqi -> oqi', tensor_sigma_quad, G_vsh, optimize=True)
        H = np.einsum('oqi, q, oqj -> ij', G_vsh, self.weights, J, optimize=True)
        return D_inv @ H
