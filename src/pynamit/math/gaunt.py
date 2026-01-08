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
from pynamit.math.wigner import wigner_3j

def get_complex_gaunt_coeff(l1, m1, l2, m2, l3, m3):
    """
    Calculate the Gaunt coefficient for determining the integral
    of the product of three complex spherical harmonics.
    """
    factor = 4.0 * np.pi
    w3j_0 = wigner_3j(l1, l2, l3, 0, 0, 0)
    if w3j_0 == 0:
        return 0.0
    w3j_m = wigner_3j(l1, l2, l3, m1, m2, m3)
    return factor * w3j_0 * w3j_m

def get_real_decomposition(l, m):
    """
    Return the decomposition of Real SH Y_lm into Complex SH Y_l,mu.
    Returns: List of (coeff, mu) pairs.
    """
    if m == 0:
        return [(1.0, 0)]
    elif m > 0:
        phase = (-1)**m
        return [(1.0/np.sqrt(2), -m), (phase/np.sqrt(2), m)]
    else:
        k = abs(m)
        phase = (-1)**k
        return [(1j/np.sqrt(2), -k), (-1j*phase/np.sqrt(2), k)]

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
        
        # Cache for Reduced Matrix Elements R(l1, l2, l3)
        self._elsasser_reduced_cache = {}

    def _compute_elsasser_gl_raw(self, li, mi, lj, mj, lk, mk) -> float:
        """
        Helper: Compute exact GL integral value for a specific triplet.
        Used for calibrating the reduced analytic element.
        """
        # Formulate grid inputs for just this triplet
        # 0. Get indices (Local robust lookup)
        try:
            # PynaMIT Basis structure: [Cosine Terms] followed by [Sine Terms]
            # m is stored as positive in both blocks.
            offset_sine = len(self.basis.cnm.index_pairs)
            
            def get_idx(l, m_signed):
                abs_m = abs(m_signed)
                # Find all matches for (l, |m|)
                matches = np.flatnonzero((self.basis.n == l) & (self.basis.m == abs_m))
                
                if len(matches) == 0:
                    raise IndexError
                
                if m_signed >= 0:
                    # Cosine term: must be in first block
                    # Filter for indices < offset_sine
                    valid = matches[matches < offset_sine]
                    if len(valid) == 0: raise IndexError
                    return valid[0]
                else:
                    # Sine term: must be in second block
                    valid = matches[matches >= offset_sine]
                    if len(valid) == 0: raise IndexError
                    return valid[0]

            idx_i = get_idx(li, mi)
            idx_j = get_idx(lj, mj)
            idx_k = get_idx(lk, mk)

        except IndexError:
            return 0.0 # Index not found in basis

        
        # 1. Retrieve VSH components on GL grid
        G_vsh_full = self.basis.get_vector_basis_matrix(self.quad_grid) # (2, Q, 2, L_basis)
        grad_Yj = G_vsh_full[:, :, 0, idx_j] # (2, Q)
        grad_Yk = G_vsh_full[:, :, 0, idx_k] # (2, Q)
        
        # 2. Compute Cross Product (Grad Y_j x Grad Y_k) . r
        cross_r = grad_Yj[0] * grad_Yk[1] - grad_Yj[1] * grad_Yk[0] # (Q,)
        
        # 3. Y_i on grid
        Y_i = self.basis.get_G(self.quad_grid)[:, idx_i] # (Q,)
        
        # 4. Integrate
        val = np.sum(self.weights * Y_i * cross_r)
        return float(val)

    def _compute_vsh_coupling_constant(self, li, lj, lk) -> complex:
        """
        Compute the Vector Spherical Harmonic coupling constant S(li, lj, lk).
        
        The Elsasser integral factorizes into:
            E(i,j,k) = S(L) * AngularSum(mi,mj,mk)
            
        Since the interaction involves parity-switching Vector Algebra (cross products),
        the scalar factor S(L) cannot be derived from simple scalar 3j/6j symbols 
        (which vanish for Odd L sum).
        
        Instead, we compute S(L) by evaluating the VSH cross-product integral 
        on the grid for a reference 'm' triplet. This effectively calculates 
        the reduced matrix element < li || X || lj lk > numericallly to machine precision.
        """
        key = (li, lj, lk)
        if key in self._elsasser_reduced_cache:
            return self._elsasser_reduced_cache[key]
            
        if (li + lj + lk) % 2 == 0: 
            return 0.0
        if not (abs(li - lj) <= lk <= li + lj): 
            return 0.0
        
        scale = 0.0j
        
        # Scan for a stable reference triplet (m_i, m_j, m_k)
        limit = min(li, 3) 
        candidates = []
        for mi in range(-limit, limit + 1):
            if len(candidates) > 2: break
            for mj in range(-min(lj, 3), min(lj, 3) + 1):
                mk = -(mi + mj)
                if abs(mk) > lk: continue
                
                # Analytic Angular Sum (Scalar Basis Projection)
                decomp_i = get_real_decomposition(li, mi)
                decomp_j = get_real_decomposition(lj, mj)
                decomp_k = get_real_decomposition(lk, mk)
                
                ang_sum = 0.0j
                for ci, mui in decomp_i:
                    for cj, muj in decomp_j:
                        for ck, muk in decomp_k:
                            if mui + muj + muk == 0:
                                ang_sum += ci * cj * ck * wigner_3j(li, lj, lk, mui, muj, muk)
                
                # If angular projection is non-zero, we can calibrate S(L)
                if abs(ang_sum) > 1e-10:
                    # Compute Exact Vector Integral
                    exact = self._compute_elsasser_gl_raw(li, mi, lj, mj, lk, mk)
                    if abs(exact) > 1e-6:
                        candidates.append((abs(exact), exact, ang_sum))
                        
        if candidates:
            # Select strongest signal to minimize numerical noise
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_exact, best_val, best_ang = candidates[0]
            scale = best_val / best_ang
            
        self._elsasser_reduced_cache[key] = scale
        return scale
        
    def elsasser_coefficient(self, li, mi, lj, mj, lk, mk) -> float:
        """
        Compute the Elsasser integral coefficient E(i,j,k).
        
        Method: Vector Basis Coupling.
        We combine the exact analytic Scalar Angular Sum (wigner_3j) with the
        pre-computed Vector Coupling Constant S(L).
        """
        scale = self._compute_vsh_coupling_constant(li, lj, lk)
        if scale == 0.0: return 0.0

            
        decomp_i = get_real_decomposition(li, mi)
        decomp_j = get_real_decomposition(lj, mj)
        decomp_k = get_real_decomposition(lk, mk)
        
        total_val = 0.0j
        for ci, mui in decomp_i:
            for cj, muj in decomp_j:
                for ck, muk in decomp_k:
                    if mui + muj + muk == 0:
                        total_val += ci * cj * ck * wigner_3j(li, lj, lk, mui, muj, muk)
        
        return float(np.real(scale * total_val))

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

    def get_analytic_interaction_matrix(self, etaP_coeffs: np.ndarray, etaH_coeffs: np.ndarray) -> np.ndarray:
        """
        Builds the full interaction matrix M using purely analytic/calibrated coefficients.
        
        The matrix is (2L, 2L) to map Vector Coeffs -> Vector Coeffs.
        Structure: [[PP, PT], [TP, TT]]
        where P=Poloidal, T=Toroidal.
        
        Blocks derived assuming Radial B approximation for analytic commutators:
        PP =   Sym(etaP) - Els(etaH)
        PT = - Els(etaP) - Sym(etaH)
        TP =   Els(etaP) + Sym(etaH)
        TT =   Sym(etaP) - Els(etaH)
        
        Ref: Poloidal/Toroidal decomposition of Surface Current J = Sigma E.
        """
        L_len = self.basis.index_length
        M_mat = np.zeros((2*L_len, 2*L_len), dtype=np.float64)
        
        # Pre-convert inputs
        if hasattr(etaP_coeffs, "values"): etaP_coeffs = etaP_coeffs.values
        if hasattr(etaH_coeffs, "values"): etaH_coeffs = etaH_coeffs.values
        etaP = np.asarray(etaP_coeffs).flatten()
        etaH = np.asarray(etaH_coeffs).flatten()
        
        ls = self.basis.n
        ms = self.basis.m
        offset_sine = len(self.basis.cnm.index_pairs)
        
        threshold = 1e-10
        
        for i in range(L_len):
            li = ls[i]
            mi_abs = ms[i]
            mi = mi_abs if i < offset_sine else -mi_abs
            
            for j in range(L_len):
                lj = ls[j]
                mj_abs = ms[j]
                mj = mj_abs if j < offset_sine else -mj_abs
                
                # Accumulators for the fundamental integrals
                sum_sym_P = 0.0
                sum_els_P = 0.0
                sum_sym_H = 0.0
                sum_els_H = 0.0
                
                # Inner loop k
                for k in range(L_len):
                    lk = ls[k]
                    mk_abs = ms[k]
                    mk = mk_abs if k < offset_sine else -mk_abs
                    
                    # Compute geometric factors once per triplet
                    # Check selection rules optimization?
                    if abs(li - lj) > lk or lk > li + lj: continue

                    # Symmetric Factor
                    sym_val = 0.0
                    # Helper decomposition for Symmetric Gaunt scalar
                    d_i = get_real_decomposition(li, mi)
                    d_j = get_real_decomposition(lj, mj)
                    d_k = get_real_decomposition(lk, mk)
                         
                    # Only compute if needed
                    cP = etaP[k]
                    cH = etaH[k]
                    has_P = abs(cP) > threshold
                    has_H = abs(cH) > threshold
                    
                    if has_P or has_H:
                        # Compute Symmetric Integral: Int(Yi Yj Yk)
                        sg = 0.0
                        for c1, m1 in d_i:
                             for c2, m2 in d_j:
                                 for c3, m3 in d_k:
                                     if m1+m2+m3 == 0:
                                         w0 = wigner_3j(li, lj, lk, 0, 0, 0)
                                         wm = wigner_3j(li, lj, lk, m1, m2, m3)
                                         sg += np.real(c1*c2*c3 * 4*np.pi * w0 * wm)
                        
                        sym_factor = 0.5 * (lk*(lk+1) - li*(li+1) - lj*(lj+1)) * sg
                        
                        if has_P: sum_sym_P += cP * sym_factor
                        if has_H: sum_sym_H += cH * sym_factor
                        
                        # Compute Elsasser Integral: E(i,j,k)
                        els_factor = self.elsasser_coefficient(li, mi, lj, mj, lk, mk)
                        
                        if has_P: sum_els_P += cP * els_factor
                        if has_H: sum_els_H += cH * els_factor

                # Fill Blocks
                # PP (0,0) and TT (1,1): Sym(P) - Els(H)
                val_diag = sum_sym_P - sum_els_H
                M_mat[i, j] = val_diag
                M_mat[i + L_len, j + L_len] = val_diag
                
                # TP (1,0): Els(P) + Sym(H)
                val_TP = sum_els_P + sum_sym_H
                M_mat[i + L_len, j] = val_TP
                
                # PT (0,1): -Els(P) - Sym(H)
                M_mat[i, j + L_len] = -val_TP
                
        return M_mat

