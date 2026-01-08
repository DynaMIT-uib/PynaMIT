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
    # Standard Gaunt: s1=s2=s3=0
    w3j_0 = wigner_3j(l1, l2, l3, 0, 0, 0)
    if w3j_0 == 0:
        return 0.0
    w3j_m = wigner_3j(l1, l2, l3, m1, m2, m3)
    return factor * w3j_0 * w3j_m

def get_spin_weighted_gaunt_coeff(l1, m1, s1, l2, m2, s2, l3, m3, s3):
    """
    Calculate the generalized Gaunt coefficient for Spin-Weighted Harmonics.
    Integral( s1_Y_l1m1 * s2_Y_l2m2 * s3_Y_l3m3 )
    
    Ref: Campbell & Leahy (2018), Eq 3.5 (standard result)
    Factor = sqrt( (2l1+1)(2l2+1)(2l3+1) / 4pi ) * W3j(m) * W3j(s)
    
    Warning: Phase conventions vary. This assumes the standard orthonormality.
    PynaMIT standard Gaunt has explicit 4pi factor because of Schmidt basis?
    Standard (Ortho) Gaunt = sqrt(...) / sqrt(4pi).
    PynaMIT Gaunt = 4pi * Standard.
    
    We will follow the numeric matching:
    Standard Integral = sqrt( (2l1+1)(2l2+1)(2l3+1) / (4*pi) ) * 3j(m) * 3j(s)
    
    If inputs are Schmidt Semi-Normalized (PynaMIT default), we need de-normalization?
    GauntEngine internally handles Schmidt by applying D_inv scaling at the end.
    So the coefficient itself should probably effectively resemble the Ortho one, 
    but we might need to be careful.
    
    Let's stick to the raw Wigner computation and let the engine handle basis scaling.
    """
    # 3j selection rules checked by wigner_3j (returns 0 if invalid)
    
    # Check spin conservation: s1 + s2 + s3 = 0? 
    # Actually the integral constraint is s1 + s2 + s3 = 0 for the W3j to be non-zero?
    
    w3j_m = wigner_3j(l1, l2, l3, m1, m2, m3)
    if w3j_m == 0:
        if l3 == 0: print(f"DEBUG_SPIN: l3=0 Zero W3j_M: l1={l1} m1={m1} m2={m2} m3={m3}")
        return 0.0
        
    w3j_s = wigner_3j(l1, l2, l3, s1, s2, s3)
    if w3j_s == 0:
        if l3 == 0: print(f"DEBUG_SPIN: l3=0 Zero W3j_S: l1={l1} s1={s1} s2={s2} s3={s3}")
        return 0.0
        
    pref = np.sqrt( (2*l1+1)*(2*l2+1)*(2*l3+1) / (4*np.pi) )
    res = pref * w3j_m * w3j_s
    if abs(res) < 1e-12 and l3==0 and s3==0:
        print(f"DEBUG_SPIN_CALC: l3=0 s3=0. pref={pref:.4e} wm={w3j_m:.4e} ws={w3j_s:.4e} res={res:.4e}")
    return res

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
        self._elsasser_reduced_cache = {}
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
        
    def _compute_elsasser_gl_raw(self, li, mi, lj, mj, lk, mk):
        """
        Numerically compute Elsasser integral E_ijk = Integral( Yi * (grad(Yj) x grad(Yk)) . r )
        Using GL quadrature.
        """
        # Indices helper
        def get_idx(l, m):
             n_arr = self.basis.n
             m_arr = self.basis.m
             # m is signed input. m_arr always positive?
             # sh_basis uses stacked cnm (m>=0) then snm (m>=1, sin)
             indices = np.where((n_arr == l) & (m_arr == abs(m)))[0]
             if len(indices) == 0: return -1
             if m == 0: return indices[0]
             if m > 0: return indices[0] # Cos (First occurrence)
             return indices[1] # Sin (Second occurrence)
             
        idx_i = get_idx(li, mi)
        idx_j = get_idx(lj, mj)
        idx_k = get_idx(lk, mk)
        
        if idx_i == -1 or idx_j == -1 or idx_k == -1:
            return 0.0
        
        # Scalar field Yi
        # G_scalar is (L, Q)
        Yi = self.G_scalar[idx_i, :]
        
        # Derivatives
        # basis.get_G returns (Q, L)
        G_th = self.basis.get_G(self.quad_grid, derivative="theta")
        G_ph = self.basis.get_G(self.quad_grid, derivative="phi")
        
        dYj_th = G_th[:, idx_j]
        dYj_ph = G_ph[:, idx_j]
        
        dYk_th = G_th[:, idx_k]
        dYk_ph = G_ph[:, idx_k]
        
        # Cross product radial component
        # (grad j x grad k) . r = (dYj_th * (1/sin dYk_phi) - (1/sin dYj_phi) * dYk_th)
        # basis.get_G('phi') returns (1/sin dY_phi).
        # So we just compute the determinant of the components.
        
        cross_r = (dYj_th * dYk_ph - dYj_ph * dYk_th)           
        
        integrand = Yi * cross_r
        return np.sum(integrand * self.weights)



    def get_spin_evaluation_matrix(self, s: int) -> np.ndarray:
        """
        Compute the evaluation matrix for Spin-Weighted Spherical Harmonics.
        
        G_s[k, alpha] = s_Y_alpha(theta_k, phi_k)
        where alpha is the flat index for (l, m).
        
        Parameters
        ----------
        s : int
            Spin weight.
            
        Returns
        -------
        G_s : (Q, L_len) Matrix
        """
        from pynamit.math.wigner import wigner_small_d
        
        L_len = self.basis.index_length
        Q = self.quad_grid.size
        theta_rad = np.deg2rad(self.quad_grid.theta)
        phi_rad = np.deg2rad(self.quad_grid.phi)
        
        G_s = np.zeros((Q, L_len), dtype=complex)
        
        idx = 0
        l_min = getattr(self.basis, "Nmin", 1)
        for l in range(l_min, self.Nmax + 1):
            for m in range(-l, l + 1):
                # Calculate prefactor
                # sYlm = (-1)^s * sqrt((2l+1)/4pi) * d^l_{m, -s}(th) * e^{im phi}
                # Check bounds: |s| <= l required?
                if abs(s) > l:
                    # sYlm is zero if |s| > l
                    G_s[:, idx] = 0.0
                else:
                    pref = (-1)**s * np.sqrt((2*l + 1) / (4 * np.pi))
                    
                    # Wigner d
                    d_val = wigner_small_d(l, m, -s, theta_rad)
                    
                    # Phase
                    phase = np.exp(1j * m * phi_rad)
                    
                    G_s[:, idx] = pref * d_val * phase
                
                idx += 1
                
        return G_s

    def analyze_spin_weighted(self, s: int, values: np.ndarray) -> np.ndarray:
        """
        Perform Spin-Weighted Spherical Harmonic Analysis (SHT) via Pseudoinverse.
        
        c = pinv(G_s) @ values
        
        Parameters
        ----------
        s : int
            Spin weight.
        values : (Q,) ndarray
            Grid values.
            
        Returns
        -------
        coeffs : (L_len,) ndarray
            Spectral coefficients.
        """
        key = f"G_s_pinv_{s}"
        if not hasattr(self, "_spin_pinv_cache"):
            self._spin_pinv_cache = {}
            
        if key not in self._spin_pinv_cache:
            G_s = self.get_spin_evaluation_matrix(s)
            # Use pseudoinverse for robustness
            # Note: G_s shape (Q, L). coeffs (L).
            # c = pinv(G) @ v
            self._spin_pinv_cache[key] = np.linalg.pinv(G_s)
            
        return self._spin_pinv_cache[key] @ values


    def get_conversion_factor(self, l: int, m: int) -> float:
        """
        Get scaling factor Lambda.
        PynaMIT (Schmidt, Unnorm Vec, No Phase) -> Standard (Ortho, Norm Vec, CS Phase).
        """
        if l == 0: return 1.0 
        
        # 1. Scalar Factor (Schmidt -> Ortho)
        schmidt = np.sqrt(4.0 * np.pi / (2 * l + 1))
        s_factor = schmidt
        
        # Geodesy/Real-to-Complex Phase
        # PynaMIT Y_real_m ~ (Y_m + (-1)^m Y_-m).
        # We work in Complex Basis coeffs.
        # But here we simply define the Magnitude Scaling.
        # The phase (-1)^m is handled by the coefficient projection step.
        # Check: sqrt(2) for m!=0 is standard in Schmidt? 
        if m != 0:
            s_factor *= np.sqrt(2.0)
            
        # 2. Vector Factor (Unnormalized -> Normalized)
        v_factor = np.sqrt(l * (l + 1))
        
        return s_factor * v_factor

    def get_general_analytic_interaction_matrix(self, coeffs_pp, coeffs_mm, coeffs_pm, coeffs_mp, input_is_complex=False):
        """
        Compute Generalized Analytic Interaction Matrix using proper Spin-to-Vector Basis transformation.
        
        Args:
            coeffs_pp: Spin Tensor Component (+2) corresponding to T_pp in geometry.
            coeffs_mm: Spin Tensor Component (-2) corresponding to T_mm.
            coeffs_pm: Spin Tensor Component (0)  corresponding to T_pm (Isotropic).
            coeffs_mp: Spin Tensor Component (0)  corresponding to T_mp (Isotropic).
            input_is_complex: If True, assumes coefficients are already in Standard Complex Orthonormal Basis (Nmin=0).
            
        Basis Projection:
            Vectors P (Poloidal) and T (Toroidal) are related to Spin Vectors (+/-):
            (+) ~ (P - iT) / sqrt(2)
            (-) ~ (P + iT) / sqrt(2)
            
            We compute Spin Matrix Blocks M++, M+-, M-+, M--.
            Then transform to P/T Basis.
        """
        # 1. Basis Projection Setup
        
        # A. State Basis Projection (for Final Matrix)
        # U_state maps Real State Basis (PynaMIT, Nmin=1 usually) -> Complex Basis.
        U_state = self._compute_basis_projection_matrix()
        
        # Dimensions
        L_complex = U_state.shape[0] # (Nmax+1)^2
        L = L_complex # Calculation Dimension
        
        M_mat = np.zeros((2*L, 2*L), dtype=np.complex128)
        
        # B. Sigma (Parameter) Basis Projection
        # Only needed if inputs are Real Schmidt coefficients.
        
        if not input_is_complex:
            print(f"DEBUG_RAW: coeffs_pp FULL:")
            for i, c in enumerate(coeffs_pp):
                print(f"  idx {i}: {c:.4e}")
            
            # Infer Sigma Nmax
            len_sigma = len(coeffs_pp)
            N_sigma = int(np.sqrt(len_sigma)) - 1
            
            # Generate Sigma Basis Indices (Split Order: g then h)
            # Assuming Nmin=0 for Sigma
            sigma_n = []
            sigma_m = []
            
            # g-terms (Cos): n=0..N, m=0..n
            for n in range(N_sigma + 1):
                for m in range(n + 1):
                    sigma_n.append(n)
                    sigma_m.append(m)
                    
            # h-terms (Sin): n=1..N, m=1..n
            for n in range(1, N_sigma + 1):
                for m in range(1, n + 1):
                    sigma_n.append(n)
                    sigma_m.append(m)
                    
            sigma_n = np.array(sigma_n)
            sigma_m = np.array(sigma_m)
            
            # Verify length
            if len(sigma_n) != len_sigma:
                # Fallback or Error? 
                # If len_sigma doesn't match complete shell, it might be truncated?
                # Proceed with care.
                pass
                
            # Construct U_sigma (Complex Size -> Real Sigma Size)
            # Complex Size for Sigma might be same as State if N_sigma = Nmax.
            L_complex_sigma = (N_sigma + 1)**2
            U_sigma = np.zeros((L_complex_sigma, len_sigma), dtype=np.complex128)
            inv_sr2 = 1.0 / np.sqrt(2.0)
            
            seen_lm = {}
            for k in range(len_sigma):
                if k >= len(sigma_n): break
                lr = sigma_n[k]
                mr = sigma_m[k]
                
                idx_base = lr**2 + lr
                idx_pos = idx_base + mr
                idx_neg = idx_base - mr
                
                if mr == 0:
                    U_sigma[idx_pos, k] = 1.0
                else:
                    count = seen_lm.get((lr, mr), 0)
                    seen_lm[(lr, mr)] = count + 1
                    if count == 0:
                        val = inv_sr2
                        U_sigma[idx_pos, k] = val
                        U_sigma[idx_neg, k] = val
                    else:
                        U_sigma[idx_pos, k] = -1j * inv_sr2
                        U_sigma[idx_neg, k] = 1j * inv_sr2

            # Scale Sigma Inputs to Orthonormal
            def scale_sigma(c_real):
                c_out = c_real.copy()
                for k in range(min(len(c_out), len(sigma_n))):
                    l = sigma_n[k]
                    m_abs = sigma_m[k]
                    f = np.sqrt(4.0 * np.pi / (2 * l + 1))
                    if m_abs != 0: f *= np.sqrt(2.0)
                    c_out[k] *= f
                return c_out

            # Project Inputs
            coeffs_pp = U_sigma @ scale_sigma(coeffs_pp)
            coeffs_mm = U_sigma @ scale_sigma(coeffs_mm)
            coeffs_pm = U_sigma @ scale_sigma(coeffs_pm)
            coeffs_mp = U_sigma @ scale_sigma(coeffs_mp)
            
            print(f"DEBUG_COEFFS: PP Norm={np.linalg.norm(coeffs_pp):.4e} Max={np.max(np.abs(coeffs_pp)):.4e}")
            print(f"DEBUG_COEFFS: PM Norm={np.linalg.norm(coeffs_pm):.4e} Max={np.max(np.abs(coeffs_pm)):.4e}")
        
        else:
            # Input is already Complex Orthonormal
            # Just pass through.
            pass

        
        # 3. Map Indices for Calculation (State Basis Matrix)
        idx_map = []
        idx_count = 0
        for l in range(0, self.Nmax + 1):
            for m in range(-l, l + 1):
                idx_map.append((l, m))
                idx_count += 1

        
        # 2. Compute Spin Blocks
        # Notation: M_s1_s2 = < s1 | Operator | s2 >
        # s1, s2 in {+1, -1}.
        # Operator Spins s3:
        # Isotropic (pm, mp) -> s3 = 0.
        # Anisotropic (pp) -> s3 = +2? (Couples -1 to +1? s1-s3-s2=0 -> s1=s2+s3 -> +1 = -1 + 2).
        # So T_pp (Spin +2) drives <+1 | T | -1>. (M_+-)
        # T_mm (Spin -2) drives <-1 | T | +1>. (M_-+)
        
        # Coeff Mapping from Geometry:
        # T_pm (Isotropic) -> Drive M_++, M_-- (Preserves Spin)
        # T_pp (Aniso) -> Drive M_+- (Flip)
        # T_mm (Aniso) -> Drive M_-+ (Flip)
        
        # Note: geometry.py names might be confusing. 
        # T_pm = S_tt + S_pp. (Scalar Like).
        # We assume T_pm couples <+|+> and <-|->.
        
        # Dense storage for blocks (since L is small)
        M_plus_plus = np.zeros((L, L), dtype=np.complex128)
        M_minus_minus = np.zeros((L, L), dtype=np.complex128)
        M_plus_minus = np.zeros((L, L), dtype=np.complex128)
        M_minus_plus = np.zeros((L, L), dtype=np.complex128)
        
        # Iterate Elements
        # (This is O(N^4) or O(N^3) depending on selection rules. N~5 is small).
        for i in range(L):
            l1, m1 = idx_map[i]
            for j in range(L):
                l2, m2 = idx_map[j]
                
                # Selection Rule: |l1-l2| <= l3 <= l1+l2.
                # m3 = m1 - m2.
                m3 = m1 - m2 # Or m2-m1?
                # Gaunt(l1, s1, l2, s2, l3, s3, m1, m2, m3) 
                # usually requires m1 = m2 + m3. So m3 = m1 - m2.
                
                # --- Isotropic (Spin 0) ---
                # s3 = 0.
                # Valid for s1=1, s2=1 OR s1=-1, s2=-1.
                
                # M_++ (<+1|0|+1>)
                # Uses Isotropic coeffs_pp
                val_pp = self._compute_term(l1, 1, l2, 1, 0, m1, m2, coeffs_pp)
                M_plus_plus[i, j] += val_pp
                
                # M_-- (<-1|0|-1>)
                # Uses Isotropic coeffs_mm
                val_mm = self._compute_term(l1, -1, l2, -1, 0, m1, m2, coeffs_mm)
                M_minus_minus[i, j] += val_mm
                
                # --- Anisotropic (Spin 2) ---
                # s3 = +2. Couples <+1 | +2 | -1>. (s1=1, s2=-1).
                # Uses Anisotropic coeffs_pm (or mp - assuming pm corresponds to Spin +2)
                val_pm = self._compute_term(l1, 1, l2, -1, 2, m1, m2, coeffs_pm)
                M_plus_minus[i, j] += val_pm
                
                # s3 = -2. Couples <-1 | -2 | +1>. (s1=-1, s2=1).
                # Uses Anisotropic coeffs_mp (or pm - assuming mp corresponds to Spin -2)
                val_mp = self._compute_term(l1, -1, l2, 1, -2, m1, m2, coeffs_mp)
                M_minus_plus[i, j] += val_mp

        # 3. Transform to P/T Basis
        # Relations:
        # <P| = 1/sqrt(2) (<+| + <-|)
        # <T| = i/sqrt(2) (<+| - <-|)
        # |P> = 1/sqrt(2) (|+> + |->)
        # |T> = -i/sqrt(2) (|+> - |->)
        
        # M_PP = <P|M|P> = 1/2 (<+| + <-|) M (|+> + |->)
        #      = 1/2 ( M++ + M+- + M-+ + M-- )
        
        # M_TT = <T|M|T> = 1/2 ( i<+| - i<-| ) M ( -i|+> + i|-> )
        #      = 1/2 ( <+| - <-| ) M ( |+> - |-> )  (i*-i = 1)
        #      = 1/2 ( M++ - M+- - M-+ + M-- )
        
        # M_PT = <P|M|T> = 1/2 (<+| + <-|) M ( -i|+> + i|-> )
        #      = -i/2 ( <+| + <-| ) M ( |+> - |-> )
        #      = -i/2 ( M++ - M+- + M-+ - M-- )
        
        # M_TP = <T|M|P> = 1/2 ( i<+| - i<-| ) M ( |+> + |-> )
        #      = i/2 ( M++ + M+- - M-+ - M-- )
        
        M_PP = 0.5 * (M_plus_plus + M_plus_minus + M_minus_plus + M_minus_minus)
        M_TT = 0.5 * (M_plus_plus - M_plus_minus - M_minus_plus + M_minus_minus)
        M_PT = -0.5j * (M_plus_plus - M_plus_minus + M_minus_plus - M_minus_minus)
        M_TP = 0.5j * (M_plus_plus + M_plus_minus - M_minus_plus - M_minus_minus)
        
        def get_scalar_lambda(l, m):
            f = np.sqrt(4.0 * np.pi / (2 * l + 1)) # Schmidt
            if m != 0: f *= np.sqrt(2.0)       # Real->Complex factor
            if m % 2 != 0: f *= -1.0           # CS Phase correction
            return f
            
        real_lambdas = np.zeros(L)
        for i in range(L):
            l, m = idx_map[i]
            lam = get_scalar_lambda(l, m)
            denom = l*(l+1)
            v_norm = np.sqrt(denom) if l > 0 else 1.0
            real_lambdas[i] = lam * v_norm
            
        scaler_col = real_lambdas[np.newaxis, :]
        scaler_row = real_lambdas[:, np.newaxis]
        scaler = scaler_row * scaler_col
        
        # 5. Assemble Final Matrix
        M_mat[0:L, 0:L] = M_PP
        M_mat[L:, L:] = M_TT
        M_mat[0:L, L:] = M_PT
        M_mat[L:, 0:L] = M_TP
        
        # 6. Real Basis Projection
        
        print(f"DEBUG_MAT: idx_map[1]={idx_map[1]}")
        print(f"DEBUG_MAT: M_PP[1,1]={M_PP[1,1]:.4e}")
        print(f"DEBUG_MAT: M_mat[1,1]={M_mat[1,1]:.4e}")
        
        # 6. Real Basis Projection
        # Map Complex P/T -> Real P/T
        # U maps Real Coeffs -> Complex Coeffs.
        # M_real = U_H @ M_complex @ U.
        # We assume M_mat is in Complex Basis (l,m).
        P_real_complex = self._compute_basis_projection_matrix()
        # Projection is Block Diagonal (P for Poloidal, P for Toroidal)
        # But M_mat mixes them.
        # We construct Full P
        zero_block = np.zeros_like(P_real_complex)
        P_full = np.block([[P_real_complex, zero_block], [zero_block, P_real_complex]])
        
        M_final = P_full.conj().T @ M_mat @ P_full
        
        print(f"DEBUG_MAT: M_final[1,1] (Real Basis)={M_final[1,1].real:.4e}")
        
        # Return Real Part (Physics is Real)
        return M_final.real

    def _compute_term(self, l1, s1, l2, s2, s3, m1, m2, coeffs):
        """Helper to compute sum over l3."""
        val = 0.0j
        m3 = m1 - m2
        l3_min = abs(l1 - l2)
        l3_max = l1 + l2
        
        # Optimization: Coeffs usually small array.
        idx = 0
        N_coeff_max = int(np.sqrt(len(coeffs))) - 1 # Nmax of sigma
        
        
        found_any = False
        
        # Loop over Sigma Coeffs
        for l3 in range(N_coeff_max + 1):
             for m_sigma in range(-l3, l3 + 1):
                 idx = l3**2 + l3 + m_sigma # Standard index
                 if idx >= len(coeffs): break
                 c = coeffs[idx]
                 if abs(c) < 1e-12: continue
                 
                 found_any = True
                 
                 # Check m conservation
                 # m1 = m2 + m3.
                 if m_sigma != m3: continue
                 
                 # Check l triangle
                 if l3 < l3_min or l3 > l3_max: continue
                 
                 # Compute Gaunt
                 # Function Def: get_spin_weighted_gaunt_coeff(l1, m1, s1, l2, m2, s2, l3, m3, s3)
                 # We need to pass Conjugated Bra (1): (-m1, -s1)
                 
                 g = get_spin_weighted_gaunt_coeff(l1, -m1, -s1, l2, m2, s2, l3, m3, s3)
                 
                 # Apply Phase (-1)^(m+s) from conjugation
                 phase = (-1.0)**(m1 + s1)
                 
                 term = c * g * phase
                 if l1 == 1 and m1 == 0 and l2 == 1 and m2 == 0:
                     print(f"DEBUG_TRACE: l3={l3} idx={idx} m={m1} s={s1} c={c.real:.4e} g={g:.4e} ph={phase} term={term.real:.4e}")
                 val += term
                 
        return val
                 
        return val
                 
        # if not found_any and len(coeffs) > 0 and abs(coeffs[0]) > 0:
        #     # print(f"DEBUG: No matching l3/m3 found for l1={l1}, l2={l2}, m3={m3}")
        #     pass
            
        return val

    def _compute_basis_projection_matrix(self):
        """
        Compute projection matrix U from Real Basis to Complex Basis.
        U_alpha_beta = < Y_complex_alpha | Y_real_beta >.
        
        M_real = U.H @ M_complex @ U.
        
        PynaMIT Real Basis Order is defined by self.basis.n and self.basis.m.
        Structure is Split: All Cosines (g) first, then Sines (h).
        We detect Cos/Sin by counting occurrences of (l,m).
        """
        L_len = self.basis.index_length
        L_complex = (self.Nmax + 1)**2
        U = np.zeros((L_complex, L_len), dtype=np.complex128)
        
        inv_sr2 = 1.0 / np.sqrt(2.0)
        
        # Track occurrences to distinguish Cos (g) and Sin (h)
        seen_lm = {} 
        
        # Flatten Basis Arrays
        n_arr = self.basis.n.flatten()
        m_arr = self.basis.m.flatten()
        
        for k in range(L_len):
            lr = n_arr[k]
            mr = m_arr[k]
            
            # Complex Index Base for this l
            # l^2 + l + m.
            idx_base = lr**2 + lr
            idx_pos = idx_base + mr
            idx_neg = idx_base - mr
            
            if mr == 0:
                # m=0 is always Cosine/Zonal. Appears once.
                # Y_real = Y_complex
                U[idx_pos, k] = 1.0
            else:
                # m > 0. Check if this is the first or second time we see this (l,m).
                count = seen_lm.get((lr, mr), 0)
                seen_lm[(lr, mr)] = count + 1
                
                if count == 0:
                    # First time: Cosine (g_lm)
                    # Y_cos = 1/sr2 (Y_m + Y_-m)
                    val = inv_sr2
                    U[idx_pos, k] = val
                    U[idx_neg, k] = val
                else:
                    # Second time: Sine (h_lm)
                    # Y_sin = -i/sr2 (Y_m - Y_-m)
                    
                    U[idx_pos, k] = -1j * inv_sr2
                    U[idx_neg, k] = 1j * inv_sr2
                    
        return U
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
                        
                        # Symmetric Factor (Corrected Sign: li+lj-lk)
                        sym_factor = 0.5 * (li*(li+1) + lj*(lj+1) - lk*(lk+1)) * sg
                        
                        if has_P: sum_sym_P += cP * sym_factor
                        if has_H: sum_sym_H += cH * sym_factor
                        
                        # Compute Elsasser Integral: E(i,j,k)
                        els_factor = self.elsasser_coefficient(li, mi, lj, mj, lk, mk)
                        
                        if has_P: sum_els_P += cP * els_factor
                        if has_H: sum_els_H += cH * els_factor

                # Fill Blocks based on Analytic Derivation
                # PP (0,0) and TT (1,1): S(P) - E(H)
                val_diag = sum_sym_P - sum_els_H
                M_mat[i, j] = val_diag
                M_mat[i + L_len, j + L_len] = val_diag
                
                # PT (0,1) (Row P, Col T): E(P) + S(H)
                val_off = sum_els_P + sum_sym_H
                M_mat[i, j + L_len] = val_off

                # TP (1,0) (Row T, Col P): -E(P) - S(H)
                M_mat[i + L_len, j] = -val_off
                
        # Apply Inverse Metric Scaling D_inv @ M
        # For Vector Harmonics (Schmidt): D_ii = l(l+1) * 4*pi / (2*l_i + 1)
        # So multiply row i by (2*l_i + 1) / (4*pi * l_i * (l_i + 1))
        
        l_vec = self.basis.n.astype(float)
        
        # Avoid division by zero for l=0 (monopole has no vector part)
        div_vec = 4 * np.pi * l_vec * (l_vec + 1)
        div_vec[div_vec == 0] = 1.0 
        
        scale_vec = (2 * l_vec + 1) / div_vec
        
        # Apply to both Poloidal and Toroidal blocks (same l_vec)
        full_scale = np.concatenate([scale_vec, scale_vec])
        
        # Row-wise multiplication
        M_mat = full_scale[:, None] * M_mat
                
        return M_mat

