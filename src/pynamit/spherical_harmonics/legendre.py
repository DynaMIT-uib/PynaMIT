"""Legendre Function Backends.

This module provides different backends for computing associated Legendre
polynomials and their derivatives, used by SHBasis.

Backends:
    - 'internal': Fast, self-contained recurrence relation
    - 'scipy': Uses scipy library with analytical scaling
"""

import numpy as np
import math
import warnings
from packaging import version
import scipy

# Conditional Import for SciPy Version Compatibility
_SCIPY_VERSION = version.parse(scipy.__version__)
if _SCIPY_VERSION >= version.parse("1.15.0"):
    USE_MODERN_SCIPY = True
    from scipy.special import assoc_legendre_p_all
    lpmn = None
else:
    USE_MODERN_SCIPY = False
    from scipy.special import lpmn
    assoc_legendre_p_all = None


def double_factorial(n: int) -> float:
    """Double factorial that correctly handles the n=-1 case.

    Parameters
    ----------
    n : int
        Input value. Must be >= -1.

    Returns
    -------
    float
        The double factorial n!!
    """
    if n < -1:
        raise ValueError("Double factorial is not defined for n < -1 in this context.")
    if n == -1 or n == 0:
        return 1.0
    result = 1.0
    for i in range(n, 0, -2):
        result *= i
    return result


def compute_scipy_scaling_factors(index_pairs: list) -> np.ndarray:
    """Calculate analytical scaling factors for scipy backend.

    Such that P_internal = F * P_scipy.
    F(n, m) = (n - m)! / (2n - 1)!!

    Parameters
    ----------
    index_pairs : list
        List of (n, m) pairs for the basis.

    Returns
    -------
    np.ndarray
        Scaling factors for each index pair.
    """
    factors = np.ones(len(index_pairs), dtype=np.float64)
    for i, (n, m) in enumerate(index_pairs):
        denominator = double_factorial(2 * n - 1)
        numerator = math.factorial(n - m)
        factors[i] = numerator / denominator
    return factors


class LegendreFunctions:
    """Legendre polynomial computation with multiple backends.

    Parameters
    ----------
    Nmax : int
        Maximum degree.
    Mmax : int
        Maximum order.
    index_pairs : list
        List of (n, m) index pairs.
    backend : str
        Backend to use: 'internal' or 'scipy'.
    """

    def __init__(self, Nmax: int, Mmax: int, index_pairs: list, backend: str = "internal"):
        self.Nmax = Nmax
        self.Mmax = Mmax
        self.index_pairs = index_pairs
        self.backend = backend
        self._use_modern_scipy = USE_MODERN_SCIPY

        # Pre-compute scipy scaling factors
        self._scipy_scaling_factors = None

        if backend == "scipy" and not self._use_modern_scipy:
            warnings.warn(
                f"Your SciPy version ({scipy.__version__}) is older than 1.15.0. Falling "
                "back to the deprecated 'lpmn' function. Please consider upgrading SciPy.",
                DeprecationWarning,
                stacklevel=3,
            )

    @property
    def scipy_scaling_factors(self) -> np.ndarray:
        """Lazy-loaded scipy scaling factors."""
        if self._scipy_scaling_factors is None:
            self._scipy_scaling_factors = compute_scipy_scaling_factors(self.index_pairs)
        return self._scipy_scaling_factors

    def compute(self, theta: np.ndarray, compute_derivative: bool = False):
        """Compute Legendre functions using the configured backend.

        Parameters
        ----------
        theta : np.ndarray
            Colatitude values in radians.
        compute_derivative : bool
            Whether to also compute derivatives.

        Returns
        -------
        P : np.ndarray
            Legendre function values, shape (N_points, N_coeffs).
        dP : np.ndarray or None
            Derivative values if requested, else None.
        """
        if self.backend == "internal":
            P = self._legendre_internal(theta)
            dP = self._legendre_derivative_internal(theta, P) if compute_derivative else None
            return P, dP
        else:
            return self._get_legendre_scipy(theta, compute_derivative)

    def _legendre_internal(self, theta: np.ndarray) -> np.ndarray:
        """Compute un-normalized Legendre functions using internal recurrence.

        Parameters
        ----------
        theta : np.ndarray
            Colatitude values in radians.

        Returns
        -------
        np.ndarray
            Legendre function values.
        """
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
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, index_map[(n - 2, m)]]
        return P

    def _legendre_derivative_internal(self, theta: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Compute d/dtheta of Legendre functions using internal recurrence.

        Parameters
        ----------
        theta : np.ndarray
            Colatitude values in radians.
        P : np.ndarray
            Pre-computed Legendre function values.

        Returns
        -------
        np.ndarray
            Derivative values.
        """
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
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, prev2_idx]
        return dP

    def _get_legendre_scipy(self, theta: np.ndarray, compute_derivative: bool = False):
        """Dispatcher for scipy Legendre function calculation."""
        if self._use_modern_scipy:
            return self._get_legendre_scipy_modern(theta, compute_derivative)
        else:
            return self._get_legendre_scipy_legacy(theta, compute_derivative)

    def _get_legendre_scipy_modern(self, theta: np.ndarray, compute_derivative: bool = False):
        """Legendre functions via `assoc_legendre_p_all` function (scipy >= 1.15)."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        diff_order = 1 if compute_derivative else 0
        p_and_dp_all = assoc_legendre_p_all(self.Nmax, self.Mmax, cos_theta, diff_n=diff_order)
        p_all, dp_dz_all = (
            (p_and_dp_all[0], p_and_dp_all[1]) if compute_derivative else (p_and_dp_all[0], None)
        )

        P_std = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP_std = np.empty_like(P_std) if compute_derivative else None

        for i, (n, m) in enumerate(self.index_pairs):
            if m >= 0:
                idx_m = m
            else:
                idx_m = 2 * self.Mmax + 1 + m

            p_values = p_all[n, idx_m].T
            cs_phase = (-1) ** m
            P_std[:, i] = p_values * cs_phase
            if compute_derivative:
                dp_dz_values = dp_dz_all[n, idx_m].T
                dp_dz = dp_dz_values * cs_phase
                dP_std[:, i] = dp_dz * (-sin_theta)

        P_scaled = P_std * self.scipy_scaling_factors
        dP_scaled = dP_std * self.scipy_scaling_factors if compute_derivative else None
        return P_scaled, dP_scaled

    def _get_legendre_scipy_legacy(self, theta: np.ndarray, compute_derivative: bool = False):
        """Legendre functions via `lpmn` function (scipy < 1.15)."""
        theta = np.atleast_1d(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        P_std = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP_std = np.empty_like(P_std) if compute_derivative else None

        for i, (ct, st) in enumerate(zip(cos_theta, sin_theta)):
            p_all, dp_dz_all = lpmn(self.Mmax, self.Nmax, ct)
            for j, (n, m) in enumerate(self.index_pairs):
                cs_phase = (-1) ** m
                P_std[i, j] = p_all[m, n] * cs_phase
                if compute_derivative:
                    dp_dz = dp_dz_all[m, n] * cs_phase
                    dP_std[i, j] = dp_dz * (-st)

        P_scaled = P_std * self.scipy_scaling_factors
        dP_scaled = dP_std * self.scipy_scaling_factors if compute_derivative else None
        return P_scaled, dP_scaled
