"""basis.py - SHBasis with fully compatible internal and Scipy backends."""

import numpy as np
import math
from scipy.special import lpmv, lpmn
# The helpers file is assumed to contain SHIndices and schmidt_quasi_normalization_factors
from pynamit.spherical_harmonics.helpers import SHIndices, schmidt_quasi_normalization_factors

def _double_factorial(n):
    """
    A robust, self-contained implementation of the double factorial that
    correctly handles the n=-1 case required by the analytical scaling factor.
    """
    if n < -1:
        raise ValueError("Double factorial is not defined for n < -1 in this context.")
    if n == -1 or n == 0:
        return 1.0
    result = 1.0
    for i in range(n, 0, -2):
        result *= i
    return result

class SHBasis(object):
    """
    Class for representing spherical harmonic bases according to the Langel (1987)
    geomagnetism convention.

    This class provides two fully compatible backends for Legendre polynomial
    generation:
    - 'internal': A fast, self-contained recurrence relation for both P and dP/dθ.
    - 'scipy': Uses the trusted scipy.special.lpmn library and standard
               recurrence relations, with a precise analytical scaling factor
               applied to ensure identical output to the 'internal' backend.
    """

    def __init__(self, Nmax, Mmax, Nmin=1, quasi_normalized=True, backend='internal'):
        """
        Parameters
        ----------
        Nmax : int
            Maximum degree.
        Mmax : int
            Maximum order.
        Nmin : int, optional
            Minimum degree, by default 1.
        quasi_normalized : bool, optional
            If True, applies Schmidt quasi-normalization factors. By default True.
        backend : str, optional
            Backend for Legendre function calculation. Can be 'internal' (default)
            or 'scipy'. Both produce identical results.
        """
        if backend not in ['internal', 'scipy']:
            raise ValueError(f"Backend '{backend}' not recognized. Use 'internal' or 'scipy'.")

        self.Nmax, self.Mmax, self.backend = Nmax, Mmax, backend
        all_indices = SHIndices(Nmax, Mmax)
        self.index_pairs = list(all_indices.index_pairs)
        
        self.cnm = SHIndices(Nmax, Mmax); self.cnm.index_pairs = tuple([p for p in self.index_pairs if p[0] >= Nmin]); self.cnm.make_arrays()
        self.snm = SHIndices(Nmax, Mmax); self.snm.index_pairs = tuple([p for p in self.index_pairs if p[0] >= Nmin and p[1] >= 1]); self.snm.make_arrays()

        self.cnm_filter = [(pair in self.cnm.index_pairs) for pair in self.index_pairs]
        self.snm_filter = [(pair in self.snm.index_pairs) for pair in self.index_pairs]

        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))
        
        self.is_normalized = quasi_normalized
        if self.is_normalized:
            s_matrix = schmidt_quasi_normalization_factors(Nmax, Mmax)
            self.schmidt_factors = np.array([s_matrix[n, m] for n, m in self.index_pairs])
        else:
            self.schmidt_factors = np.ones(len(self.index_pairs))
        
        self._compute_scipy_scaling_factors()

        # --- Other properties from original code ---
        self.kind = "SH"
        self.index_names = ["n", "m"]
        self.index_length = len(self.cnm.index_pairs) + len(self.snm.index_pairs)
        self.index_arrays = [self.n, self.m]
        self.minimum_phi_sampling = 2 * Mmax + 1
        self.caching = True

    def _compute_scipy_scaling_factors(self):
        """
        Calculates the analytical scaling factor needed to convert standard (Scipy)
        un-normalized Legendre polynomials to the specific normalization used
        by the internal recurrence relations.

        The scaling factor F(n,m) such that P_internal = F * P_scipy is:
        F(n, m) = (n - m)! / (2n - 1)!!
        """
        factors = np.ones(len(self.index_pairs), dtype=np.float64)
        for i, (n, m) in enumerate(self.index_pairs):
            denominator = _double_factorial(2 * n - 1)
            numerator = math.factorial(n - m)
            factors[i] = numerator / denominator
        self.scipy_scaling_factors = factors

    def _get_legendre_scipy(self, theta, compute_derivative=False):
        """
        Computes P and optionally dP/d(theta) using the Scipy/standard convention,
        then scales them to match the internal convention. This version uses the
        highly efficient `lpmn` function.
        """
        theta = np.atleast_1d(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        P_std = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP_std = np.empty_like(P_std) if compute_derivative else None

        # lpmn is not vectorized over its third argument, so we loop over grid points.
        for i, (ct, st) in enumerate(zip(cos_theta, sin_theta)):
            # Calculate all Pnm and dPnm/dz for all n,m up to Nmax, Mmax
            p_all, dp_dz_all = lpmn(self.Mmax, self.Nmax, ct)

            for j, (n, m) in enumerate(self.index_pairs):
                # Remove Condon-Shortley phase
                cs_phase = (-1)**m
                P_std[i, j] = p_all[m, n] * cs_phase
                
                if compute_derivative:
                    # Get dP/dz from lpmn, remove C-S phase, and apply chain rule
                    dp_dz = dp_dz_all[m, n] * cs_phase
                    dP_std[i, j] = dp_dz * (-st)

        # Apply the analytical scaling factor to both P and dP.
        # Since dP/d(theta) is a linear operator, the same factor applies.
        P_scaled = P_std * self.scipy_scaling_factors
        dP_scaled = dP_std * self.scipy_scaling_factors if compute_derivative else None
        return P_scaled, dP_scaled

    def get_G(self, grid, derivative=None, cache_in=None, cache_out=False):
        phi, theta = np.deg2rad(grid.phi), np.deg2rad(grid.theta)
        
        if self.backend == 'internal':
            P_unnormalized = self.legendre(theta)
            dP_unnormalized = self.legendre_derivative(theta, P=P_unnormalized) if derivative else None
        else: # backend == 'scipy'
            P_unnormalized, dP_unnormalized = self._get_legendre_scipy(theta, compute_derivative=bool(derivative))

        P = P_unnormalized * self.schmidt_factors
        dP = dP_unnormalized * self.schmidt_factors if dP_unnormalized is not None else None
        
        # --- G-matrix assembly is unchanged ---
        if derivative is None:
            Gc = P[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = P[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        elif derivative == "theta":
            Gc = dP[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = dP[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        elif derivative == "phi":
            sin_theta = np.sin(theta).reshape(-1, 1)
            phi_col = phi.reshape(-1, 1)
            is_pole = np.abs(sin_theta) <= 1e-12
            m_c, m_s = self.cnm.m, self.snm.m
            num_Gc = -P[:, self.cnm_filter] * m_c * np.sin(m_c * phi_col)
            Gc = np.divide(num_Gc, sin_theta, out=np.zeros_like(num_Gc), where=~is_pole)
            num_Gs = P[:, self.snm_filter] * m_s * np.cos(m_s * phi_col)
            Gs = np.divide(num_Gs, sin_theta, out=np.zeros_like(num_Gs), where=~is_pole)
            idx_poles = np.where(is_pole.flatten())[0]
            if idx_poles.size:
                cnm_is_m1, snm_is_m1 = (self.cnm.m == 1).flatten(), (self.snm.m == 1).flatten()
                cnm_m1_cols, snm_m1_cols = np.where(cnm_is_m1)[0], np.where(snm_is_m1)[0]
                if cnm_m1_cols.size:
                    dP_pole = dP[idx_poles][:, self.cnm_filter][:, cnm_is_m1]
                    Gc[np.ix_(idx_poles, cnm_m1_cols)] = -dP_pole * np.sin(phi_col[idx_poles])
                if snm_m1_cols.size:
                    dP_pole = dP[idx_poles][:, self.snm_filter][:, snm_is_m1]
                    Gs[np.ix_(idx_poles, snm_m1_cols)] = dP_pole * np.cos(phi_col[idx_poles])
        else:
            raise ValueError(f'Invalid derivative "{derivative}".')
        
        if cache_out:
            return np.hstack((Gc, Gs)), P_unnormalized
        return np.hstack((Gc, Gs))
        
    def legendre(self, theta):
        """Computes un-normalized Legendre functions using the internal recurrence."""
        theta = np.asarray(theta, dtype=float)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        P = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        P[:, 0] = 1.0
        index_map = {pair: i for i, pair in enumerate(self.index_pairs)}
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                P[:, nm] = sin_theta * P[:, index_map[(n - 1, m - 1)]]
            else:
                if n > m:
                    P[:, nm] = cos_theta * P[:, index_map[(n - 1, m)]]
                if n > m + 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[:, nm] -= Knm * P[:, index_map[(n - 2, m)]]
        return P

    def legendre_derivative(self, theta, P):
        """Computes d/dθ of Legendre functions consistent with the internal recurrence."""
        theta = np.asarray(theta, dtype=float)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        dP = np.empty_like(P)
        dP[:, 0] = 0.0
        index_map = {pair: i for i, pair in enumerate(self.index_pairs)}
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                prev_idx = index_map[(n - 1, m - 1)]
                dP[:, nm] = sin_theta * dP[:, prev_idx] + cos_theta * P[:, prev_idx]
            else:
                if n > m:
                    prev_idx = index_map[(n - 1, m)]
                    dP[:, nm] = cos_theta * dP[:, prev_idx] - sin_theta * P[:, prev_idx]
                if n > m + 1:
                    prev2_idx = index_map[(n - 2, m)]
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    dP[:, nm] -= Knm * dP[:, prev2_idx]
        return dP

    # --- Other methods are unchanged ---
    def laplacian(self, r=1.0): return -self.n * (self.n + 1) / r**2
    def radial_shift_Ve(self, start, end): return (start / end) ** (1 - self.n)
    def radial_shift_Vi(self, start, end): return (start / end) ** (self.n + 2)
    @property
    def coeffs_to_delta_V(self):
        if not hasattr(self, "_coeffs_to_delta_V"): self._coeffs_to_delta_V = 2 * self.n + 1
        return self._coeffs_to_delta_V