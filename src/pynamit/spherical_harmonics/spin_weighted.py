
"""
Spin-Weighted Spherical Harmonics Basis.

This module provides a strictly Standard Complex Orthonormal implementation of
Spin-Weighted Spherical Harmonics, designed for use with Wigner 3j interaction integrals.

Definition:
    {}_s Y_{lm}(\theta, \phi) = (-1)^s \sqrt{\frac{2l+1}{4\pi}} d^l_{m, -s}(\theta) e^{im\phi}

Used for "Numerical Basis Projection" to bridge the gap between PynaMIT's 
Real Schmidt basis and algebraic 3j formulas.
"""

import numpy as np
from pynamit.math.wigner import wigner_small_d

class SpinWeightedBasis:
    """
    Standard Complex Orthonormal Spin-Weighted Spherical Harmonic Basis.
    """
    def __init__(self, Nmax: int):
        self.Nmax = Nmax
        # Generate generic (l, m) list for standard ordering
        # Unlike PynaMIT's SHBasis (which splits cos/sin), we use standard 
        # complex list: l=0..N, m=-l..l
        # Flattened index k corresponds to (l_k, m_k)
        
        self.indices = []
        for l in range(Nmax + 1):
            for m in range(-l, l + 1):
                self.indices.append((l, m))
        
        self.L_len = len(self.indices)
        
    def get_evaluation_matrix(self, s: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute Evaluation Matrix G_s at given grid points.
        
        Parameters
        ----------
        s : int
            Spin weight.
        theta, phi : (Q,) ndarray
            Grid coordinates in radians.
            
        Returns
        -------
        G : (Q, L_len) ndarray (complex)
            Matrix where G[q, k] = {}_s Y_{l_k, m_k} (theta[q], phi[q])
        """
        Q = len(theta)
        G = np.zeros((Q, self.L_len), dtype=complex)
        
        # Pre-compute phases? No, just loop. Optimization is secondary to correctness here.
        
        for k, (l, m) in enumerate(self.indices):
            
            # Bound check for spin
            if abs(s) > l:
                G[:, k] = 0.0
                continue
            
            # Prefactor: (-1)^s * sqrt((2l+1)/4pi)
            # Note: wigner_small_d handles the factorial parts.
            # Normalization for Orthonormal Harmonics:
            # N = sqrt( (2l+1)/4pi ) usually appearing in Ylm def.
            pref = (-1)**s * np.sqrt((2 * l + 1) / (4 * np.pi))
            
            # Wigner d: d^l_{m, -s}
            # pynamit.math.wigner.wigner_small_d(j, m1, m2, theta)
            d_val = wigner_small_d(l, m, -s, theta)
            
            # Phase: e^{im phi}
            phase = np.exp(1j * m * phi)
            
            G[:, k] = pref * d_val * phase
            
        return G
