"""Spherical Harmonic Basis Class."""

import numpy as np
import math
from functools import cached_property
import warnings
from packaging import version
import scipy

from pynamit.spherical_harmonics.helpers import SHIndices, schmidt_quasi_normalization_factors

# Conditional Import for SciPy Version Compatibility
# Check the SciPy version to import the correct, available function.
_SCIPY_VERSION = version.parse(scipy.__version__)
if _SCIPY_VERSION >= version.parse("1.15.0"):
    _USE_MODERN_SCIPY = True
    from scipy.special import assoc_legendre_p_all

    # Define lpmn as None for clarity (not used in this path).
    lpmn = None
else:
    _USE_MODERN_SCIPY = False
    from scipy.special import lpmn

    # Define assoc_legendre_p_all as None so the name exists for type
    # hinting/clarity.
    assoc_legendre_p_all = None


def _double_factorial(n):
    """Double factorial that correctly handles the n=-1 case."""
    if n < -1:
        # This case is not expected, but defined for completeness.
        raise ValueError("Double factorial is not defined for n < -1 in this context.")
    if n == -1 or n == 0:
        return 1.0
    result = 1.0
    for i in range(n, 0, -2):
        result *= i
    return result


class SHBasis:
    """
    Class for representing spherical harmonic bases.

    Uses the Langel (1987) geomagnetism convention.

    This class provides two fully compatible backends for Legendre
    polynomial generation:
    - 'internal':
        A fast, self-contained recurrence relation for both P and dP/dθ.
    - 'scipy':
        Uses the trusted scipy library, with a precise analytical
        scaling factor applied to ensure identical output to the
        'internal' backend. It automatically selects the best available
        scipy function.
    """

    def __init__(
        self,
        Nmax: int,
        Mmax: int,
        Nmin: int = 1,
        quasi_normalized: bool = True,
        backend: str = "internal",
    ):
        """
        Initialize the SHBasis instance.

        Parameters
        ----------
        Nmax : int
            Maximum degree.
        Mmax : int
            Maximum order.
        Nmin : int, optional
            Minimum degree, by default 1.
        quasi_normalized : bool, optional
            If True, applies Schmidt quasi-normalization factors. By
            default True.
        backend : str, optional
            Backend for Legendre function calculation. Can be 'internal'
            (default) or 'scipy'. Both produce identical results.
        """
        if backend not in ["internal", "scipy"]:
            raise ValueError(f"Backend '{backend}' not recognized. Use 'internal' or 'scipy'.")

        self.Nmax, self.Mmax, self.backend = Nmax, Mmax, backend
        self.is_normalized = quasi_normalized
        self._use_modern_scipy = _USE_MODERN_SCIPY

        self.kind = "SH"
        self.index_names = ["n", "m"]
        self.minimum_phi_sampling = 2 * Mmax + 1
        self.caching = True

        all_indices = SHIndices(Nmax, Mmax)
        self.index_pairs = list(all_indices.index_pairs)

        self.cnm = SHIndices(Nmax, Mmax)
        self.cnm.index_pairs = tuple([p for p in self.index_pairs if p[0] >= Nmin])
        self.cnm.make_arrays()
        self.snm = SHIndices(Nmax, Mmax)
        self.snm.index_pairs = tuple([p for p in self.index_pairs if p[0] >= Nmin and p[1] >= 1])
        self.snm.make_arrays()

        self.cnm_filter = [pair in self.cnm.index_pairs for pair in self.index_pairs]
        self.snm_filter = [pair in self.snm.index_pairs for pair in self.index_pairs]

        self.n = np.hstack((self.cnm.n.flatten(), self.snm.n.flatten()))
        self.m = np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))
        self.index_arrays = [self.n, self.m]
        self.index_length = len(self.cnm.index_pairs) + len(self.snm.index_pairs)

        if self.backend == "scipy" and not self._use_modern_scipy:
            warnings.warn(
                f"Your SciPy version ({scipy.__version__}) is older than 1.15.0. Falling "
                "back to the deprecated 'lpmn' function. Please consider upgrading SciPy.",
                DeprecationWarning,
                stacklevel=2,
            )

    @cached_property
    def schmidt_factors(self) -> np.ndarray:
        """Return Schmidt quasi-normalization factors."""
        if not self.is_normalized:
            return np.ones(len(self.index_pairs))
        s_matrix = schmidt_quasi_normalization_factors(self.Nmax, self.Mmax)
        return np.array([s_matrix[n, m] for n, m in self.index_pairs])

    @cached_property
    def scipy_scaling_factors(self) -> np.ndarray:
        """
        Calculate the analytical scaling factor.

        Such that P_internal = F * P_scipy.
        F(n, m) = (n - m)! / (2n - 1)!!
        """
        factors = np.ones(len(self.index_pairs), dtype=np.float64)
        for i, (n, m) in enumerate(self.index_pairs):
            denominator = _double_factorial(2 * n - 1)
            numerator = math.factorial(n - m)
            factors[i] = numerator / denominator
        return factors

    def get_extended_basis(self) -> "SHBasis":
        """Return a basis extended to include the monopole term (Nmin=0)."""
        if self.cnm.index_pairs[0][0] == 0:
             return self
        return SHBasis(self.Nmax, self.Mmax, Nmin=0, quasi_normalized=self.is_normalized, backend=self.backend)

    def _get_legendre_scipy(self, theta: np.ndarray, compute_derivative: bool = False):
        """Dispatcher for Scipy Legendre function calculation."""
        if self._use_modern_scipy:
            return self._get_legendre_scipy_modern(theta, compute_derivative)
        else:
            return self._get_legendre_scipy_legacy(theta, compute_derivative)

    def _get_legendre_scipy_modern(self, theta, compute_derivative=False):
        """Legendre functions via `assoc_legendre_p_all` function."""
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
            # Scipy 1.15+ assoc_legendre_p_all returns orders in [0, 1, ..., M, -M, ..., -1] layout
            # NOT sorted [-M...M] as one might assume.
            if m >= 0:
                idx_m = m
            else:
                idx_m = 2 * self.Mmax + 1 + m # e.g. for M=2, m=-2 -> 5-2=3

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

    def _get_legendre_scipy_legacy(self, theta, compute_derivative=False):
        """Legendre functions via `lpmn` function (SciPy<1.15)."""
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

    def get_G(self, grid, derivative=None, cache_in=None, cache_out=False):
        """Compute basis functions G on the provided grid."""
        phi, theta = np.deg2rad(grid.phi), np.deg2rad(grid.theta)

        if self.backend == "internal":
            P_unnormalized = self.legendre(theta)
            dP_unnormalized = (
                self.legendre_derivative(theta, P=P_unnormalized) if derivative else None
            )
        else:  # backend == 'scipy'
            P_unnormalized, dP_unnormalized = self._get_legendre_scipy(
                theta, compute_derivative=bool(derivative)
            )

        P = P_unnormalized * self.schmidt_factors
        dP = dP_unnormalized * self.schmidt_factors if dP_unnormalized is not None else None

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
        """Compute un-normalized Legendre functions."""
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

    def legendre_derivative(self, theta, P):
        """Compute d/dθ of Legendre functions."""
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

    def laplacian(self, r=1.0):
        """Factor to apply the spherical harmonic Laplacian operator."""
        return -self.n * (self.n + 1) / r**2

    def radial_shift_Ve(self, start, end):
        """Factor to radially shift external potential coefficients."""
        return (start / end) ** (1 - self.n)

    def radial_shift_Vi(self, start, end):
        """Factor to radially shift internal potential coefficients."""
        return (start / end) ** (self.n + 2)

    @cached_property
    def coeffs_to_delta_V(self):
        """Factor to convert coefficients to delta V at unit radius."""
        return 2 * self.n + 1

    def to_grid_values(self, coeffs, evaluator, vector_type):
        """Convert coefficients to grid values."""
        if vector_type == "scalar":
            return evaluator.basis_to_grid(coeffs)
        elif vector_type == "tangential":
            return evaluator.basis_to_grid(coeffs, helmholtz=True)
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def regularization_term(self, coeffs, evaluator, vector_type):
        """Compute regularization penalty term.

        Parameters
        ----------
        coeffs : ndarray
            SH Coefficients.
        evaluator : BasisEvaluator
            Evaluator to use.
        vector_type : str
             "scalar" or "tangential".

        Returns
        -------
        term : float
            Regularization term.
        """
        if vector_type == "scalar":
            return evaluator.regularization_term(coeffs, helmholtz=False)
        elif vector_type == "tangential":
            return evaluator.regularization_term(coeffs, helmholtz=True)
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def from_grid_values(self, values, evaluator, vector_type):
        """Convert grid values to coefficients.

        For SHBasis, this involves fitting via the evaluator.

        Parameters
        ----------
        values : array-like
            Values on the grid.
        evaluator : BasisEvaluator
            Evaluator to use for fitting.
        vector_type : str
             "scalar" or "tangential".

        Returns
        -------
        coeffs : ndarray
            Fitted SH coefficients.
        """
        if vector_type == "scalar":
            return evaluator.grid_to_basis(values, helmholtz=False)
        elif vector_type == "tangential":
            return evaluator.grid_to_basis(values, helmholtz=True)
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def project_to_basis(
        self,
        input_values,
        input_grid,
        vector_type,
        target_grid,
        target_basis,
        on_storage_grid,
        on_input_grid,
    ):
        """Project input data onto the target basis.

        For SHBasis, we fit directly to the input grid, effectively projecting
        onto itself, ignoring the target basis.

        Parameters
        ----------
        input_values : array-like
            Raw input values.
        input_grid : Grid
            Grid object defining where input_values are located.
        vector_type : str
            "scalar" or "tangential".
        target_grid : Grid
            Unused here.
        target_basis : object
            Unused here.
        on_storage_grid : callable
            Unused.
        on_input_grid : callable
            Callback returning the evaluator for the input grid.

        Returns
        -------
        coeffs : ndarray
            The fitted SH coefficients.
        """
        coeffs = self.from_grid_values(input_values, on_input_grid(), vector_type)
        return coeffs



    def construct_projection_matrix(self, evaluator) -> np.ndarray:
        """Construct the projection matrix mapping Grid Vector -> SH Coefficients.
        
        Requires an evaluator that can compute G_helmholtz (vector basis).
        """
        from pynamit.utils import tensor_pinv
        
        # Calculate pseudo-inverse of the Helmholtz matrix: (2, N_grid, 2, N_sh)
        # Flatten input dims (2, N_grid) -> leading dim
        pinv = tensor_pinv(evaluator.G_helmholtz, n_leading_flattened=2)
        
        # Reshape to 2D matrix: (2*N_sh, 2*N_grid)
        # pinv shape: (comp_out, N_out, comp_in, N_in) = (2, N_sh, 2, N_grid)
        shape = pinv.shape
        return pinv.reshape(shape[0] * shape[1], shape[2] * shape[3])
