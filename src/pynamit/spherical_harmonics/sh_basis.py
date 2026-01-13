"""Spherical Harmonic Basis Class."""

import numpy as np
from typing import Any, Optional, Union, TYPE_CHECKING, Tuple
import math
from functools import cached_property
import warnings
from packaging import version
import scipy

from pynamit.spherical_harmonics.helpers import SHIndices, schmidt_quasi_normalization_factors
from pynamit.primitives.basis import Basis
from pynamit.utils import xp

if TYPE_CHECKING:
    from pynamit.math.linear_map import LinearMap

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


class SHBasis(Basis):
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
        self.Nmin = Nmin
        self.is_normalized = quasi_normalized
        self._use_modern_scipy = _USE_MODERN_SCIPY

        self._kind = "SH"
        self._index_names = ["n", "m"]
        self._minimum_phi_sampling = 2 * Mmax + 1
        super().__init__() # Initialize _cache, solvers

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
        self._index_arrays = [self.n, self.m]
        self._index_length = len(self.cnm.index_pairs) + len(self.snm.index_pairs)

        if self.backend == "scipy" and not self._use_modern_scipy:
            warnings.warn(
                f"Your SciPy version ({scipy.__version__}) is older than 1.15.0. Falling "
                "back to the deprecated 'lpmn' function. Please consider upgrading SciPy.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def kind(self) -> str:
        return self._kind
    
    @kind.setter
    def kind(self, value):
        self._kind = value

    @property
    def index_names(self) -> list[str]:
        return self._index_names

    @index_names.setter
    def index_names(self, value):
        self._index_names = value

    @property
    def index_length(self) -> int:
        return self._index_length
    
    @index_length.setter
    def index_length(self, value):
        self._index_length = value

    @property
    def index_arrays(self) -> list:
        return self._index_arrays
    
    @index_arrays.setter
    def index_arrays(self, value):
        self._index_arrays = value

    @property
    def minimum_phi_sampling(self) -> float:
        return self._minimum_phi_sampling
    
    @minimum_phi_sampling.setter
    def minimum_phi_sampling(self, value):
        self._minimum_phi_sampling = value

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

    @staticmethod
    def compute_exact_weights(theta_1d, L):
        """Compute exact quadrature weights for a given 1D theta grid and bandlimit.
        
        Solves the moment equations: sum_t w_t P_l(theta_t) = 2 * delta_{l0}
        for l = 0 ... L-1.
        
        Parameters
        ----------
        theta_1d : np.ndarray
             Theta coordinates (colatitude) in radians.
        L : int
             Band limit (maximum degree P_l to match).
             Usually N_theta >= L.
             
        Returns
        -------
        weights : np.ndarray
             Weights for each theta point.
        """
        N_points = len(theta_1d)
        if N_points < L:
             # Underdetermined. Can't be exact for all L moments.
             # Warn or just solve best fit.
             # We assume L <= N_points. But usually we solve for N_points weights using N_points constraints.
             pass
             
        # We solve for weights corresponding to L moments equal to N_points
        # Or if L < N_points, system is underdetermined (many solutions).
        # We typically want the solution that minimizes weight variance?
        # But for 'exact transform', we usually imply N_points = L (minimal grid).
        # If N_points > L, we can match up to degree N_points-1.
        
        N_moments = N_points 
        
        from scipy.special import eval_legendre
        P_matrix = np.zeros((N_moments, N_points))
        for l in range(N_moments):
            P_matrix[l, :] = eval_legendre(l, np.cos(theta_1d))
            
        b = np.zeros(N_moments)
        b[0] = 2.0
        
        try:
            weights = np.linalg.solve(P_matrix, b)
        except np.linalg.LinAlgError:
            weights = np.ones(N_points) * 2.0 / N_points
            
        return weights

    @staticmethod
    def get_mw_weights(L):
        """Compute quadrature weights for McEwen-Wiaux (MW) sampling.
        
        Ref: McEwen & Wiaux (2011).
        
        Parameters
        ----------
        L : int
             Band limit. N_theta = L + 1.
             
        Returns
        -------
        weights : np.ndarray
             Weights for each theta_t (shape: L+1).
        """
        N = L + 1 # N_theta
        t = np.arange(N)
        theta_t = np.pi * (2 * t + 1) / (2 * N - 1)
        
        return SHBasis.compute_exact_weights(theta_t, N)

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

    def get_G(self, grid, derivative=None):
        """Compute basis functions G on the provided grid."""
        phi, theta = np.deg2rad(grid.phi), np.deg2rad(grid.theta)
        
        # Check internal cache
        grid_key = grid.hash
        
        # Ensure sub-dictionary exists for this grid
        if grid_key not in self._cache:
            self._cache[grid_key] = {"P": None, "dP": None, "G": {}}
        
        # Handle legacy tuple structure if present (though unlikely in fresh run)
        if isinstance(self._cache[grid_key], tuple):
             # Upgrade to dict
             P_old, dP_old = self._cache[grid_key]
             self._cache[grid_key] = {"P": P_old, "dP": dP_old, "G": {}}
        
        cache_entry = self._cache[grid_key]
        
        # Check if G is already cached for this derivative
        if derivative in cache_entry["G"]:
            return cache_entry["G"][derivative]

        P_unnormalized = cache_entry["P"]
        dP_unnormalized = cache_entry["dP"]

        # If P is missing, or derivative is needed but dP is missing, compute Legendre
        need_P = P_unnormalized is None
        need_dP = derivative and dP_unnormalized is None

        if need_P or need_dP:
            if self.backend == "internal":
                if need_P:
                    P_unnormalized = self.legendre(theta)
                    cache_entry["P"] = P_unnormalized
                if need_dP:
                    dP_unnormalized = self.legendre_derivative(theta, P=P_unnormalized)
                    cache_entry["dP"] = dP_unnormalized
                
                # Retrieve from cache if we didn't just compute it
                if not need_dP and derivative: 
                    dP_unnormalized = cache_entry["dP"]

            else:  # backend == 'scipy'
                compute_dP = bool(derivative) or need_dP
                
                if need_P or need_dP:
                    P_new, dP_new = self._get_legendre_scipy(
                        theta, compute_derivative=compute_dP
                    )
                    if need_P:
                        cache_entry["P"] = P_new
                        P_unnormalized = P_new
                    if compute_dP:
                        cache_entry["dP"] = dP_new
                        dP_unnormalized = dP_new
                else:
                    P_unnormalized = cache_entry["P"]
                    dP_unnormalized = cache_entry["dP"]

        # Ensure we have what we need for the G calculation
        P_unnormalized = cache_entry["P"]
        dP_unnormalized = cache_entry["dP"]

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

        G = np.hstack((Gc, Gs))
        
        # Cache the result
        cache_entry["G"][derivative] = G
        
        return G

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

    def get_laplacian_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the Laplacian operator for this basis."""
        from pynamit.math.linear_map import diagonal_linear_map
        return diagonal_linear_map(self.laplacian(r))

    def get_radial_shift_operator(
        self, start_r: float, end_r: float, kind: str = "external"
    ) -> "LinearMap":
        """Get the radial shift operator for potential coefficients."""
        from pynamit.math.linear_map import diagonal_linear_map
        if kind == "external":
            return diagonal_linear_map(self.radial_shift_Ve(start_r, end_r))
        else:
            return diagonal_linear_map(self.radial_shift_Vi(start_r, end_r))

    def get_potential_scaling_operator(self) -> "LinearMap":
        """Get the operator for converting coefficients to surface potential."""
        from pynamit.math.linear_map import diagonal_linear_map
        return diagonal_linear_map(self.coeffs_to_delta_V)

    def get_gradient_operator(self, r: float = 1.0) -> "LinearMap":
        """
        Get the analytical gradient operator in spectral space.
        Maps scalar coefficients to Poloidal Vector coefficients.
        
        Applying this operator results in coefficients for the basis functions
        G_vsh[0] = -grad Y.
        """
        from pynamit.math.linear_map import diagonal_linear_map, BlockLinearMap
        ident = diagonal_linear_map(xp.ones(self.index_length) / r)
        zeros = diagonal_linear_map(xp.zeros(self.index_length))
        return BlockLinearMap([[ident], [zeros]]) # Shape (2L, L)

    def get_curl_operator(self, r: float = 1.0) -> "LinearMap":
        """
        Get the analytical curl operator (r x grad) in spectral space.
        Maps scalar coefficients to Toroidal Vector coefficients.
        """
        from pynamit.math.linear_map import diagonal_linear_map, BlockLinearMap
        zeros = diagonal_linear_map(xp.zeros(self.index_length))
        # Note: Torque R x Grad preserves the 1/r scaling
        ident = diagonal_linear_map(xp.ones(self.index_length) / r)
        return BlockLinearMap([[zeros], [ident]]) # Shape (2L, L)

    def get_divergence_operator(self, r: float = 1.0) -> "LinearMap":
        """
        Get the analytical divergence operator in spectral space.
        Maps [Poloidal; Toroidal] vector coefficients to scalar coefficients.
        
        div(c_pol * (-grad Y) + c_tor * (rxgrad Y)) = c_pol * (-laplacian Y)
        Note: -laplacian Y / r^2 = n(n+1)/r^2 Y
        The coefficients c_pol already include the 1/r from the gradient operator.
        So divergence of (1/r grad Y) is (1/r^2 laplacian Y).
        """
        from pynamit.math.linear_map import diagonal_linear_map, BlockLinearMap
        # Div of Poloidal: factors = n(n+1)/r 
        # (since pol basis is -grad Y, and Div(-grad Y) = -Lap Y)
        # If c_pol = V/r, then div(c_pol * -grad Y) = (V/r) * -Lap Y = V/r * n(n+1)/r^2 ... wait.
        # Actually, the divergence operator should map the vector coefficients directly.
        # div is d/dr ... wait, for sheet current on a sphere:
        # div_S (A) = 1/(r sin th) [ d/dth (A_th sin th) + d/dph A_ph ]
        # For our basis: div_S (-grad Y) = -Laplacian_S Y = n(n+1)/r^2 Y
        factors = self.n * (self.n + 1) / r
        op_pol = diagonal_linear_map(factors)
        op_tor = diagonal_linear_map(xp.zeros(self.index_length))
        return BlockLinearMap([[op_pol, op_tor]]) # Shape (L, 2L)

    def get_vector_product_operator(self, tensor_sigma_coeffs: np.ndarray) -> "LinearMap":
        """
        Get interaction operator for a 2x2 conductance tensor and VSH vectors.
        tensor_sigma_coeffs: (2, 2, L) matching SHBasis.
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        from pynamit.math.linear_map import as_linear_map
        engine = GauntEngine(self)
        
        # 1. Project tensor components to quadrature grid
        # Result: (2, 2, Q)
        Q = engine.quad_grid.size
        sigma_quad = np.empty((2, 2, Q))
        for i in range(2):
            for j in range(2):
                sigma_quad[i, j] = engine.G_scalar.T @ tensor_sigma_coeffs[i, j]
                
        # 2. Compute Vector Interaction Matrix
        M = engine.get_vector_interaction_matrix(sigma_quad)
        return as_linear_map(M)

    def get_product_operator(
        self, coeffs_a: np.ndarray, grid: Optional[Any] = None, method: str = "transform"
    ) -> "LinearMap":
        """
        Get product operator for SHBasis.

        Parameters
        ----------
        coeffs_a : np.ndarray
            Coefficients of multiplier field.
        grid : Any, optional
            Grid for transform method.
        method : str, optional
            'transform' (Grid-based evaluate/multiply/project) or 
            'spectral' (Wigner-3j / Gaunt convolution).
        """
        from pynamit.math.linear_map import as_linear_map

        if method == "spectral":
            from pynamit.spherical_harmonics.gaunt import GauntEngine
            engine = GauntEngine(self)
            M = engine.get_interaction_matrix(coeffs_a)
            return as_linear_map(M)

        if grid is None:
            raise ValueError("SHBasis.get_product_operator requires a grid for the transform method.")

        # 1. Transform multiplier coefficients to grid
        a_grid = self.evaluate(coeffs_a, grid, vector_type="scalar")
        
        # 2. Build the Transform-Product-Project operator: P @ diag(a) @ G
        G_mat = self.get_G(grid)
        P_mat = grid.get_projection_matrix(self)
        
        op_G = as_linear_map(G_mat)
        op_P = as_linear_map(P_mat)
        
        from pynamit.math.linear_map import diagonal_linear_map
        op_diag_a = diagonal_linear_map(a_grid.flatten())
        
        return op_P @ op_diag_a @ op_G

    def evaluate(self, coeffs: np.ndarray, grid: Any, vector_type: str = "scalar") -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        if vector_type == "scalar":
            return self.basis_to_grid(coeffs, grid, helmholtz=False)
        elif vector_type == "tangential":
            if coeffs.ndim == 1:
                # basis_to_grid expects (2, N_coeffs) for contraction
                coeffs = coeffs.reshape(2, -1)
            return self.basis_to_grid(coeffs, grid, helmholtz=True)
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def from_grid_values(
        self,
        values: np.ndarray,
        grid: Any,
        vector_type: str,
        **kwargs,
    ) -> np.ndarray:
        """Convert grid values to coefficients."""
        # Extract solve parameters from kwargs
        weights = kwargs.get("weights")
        reg_lambda = kwargs.get("reg_lambda")
        pinv_rtol = kwargs.get("pinv_rtol", 1e-15)
        solver_type = kwargs.get("solver_type", "svd")

        if vector_type == "scalar":
            return self.grid_to_basis(
                values, grid, helmholtz=False, weights=weights, reg_lambda=reg_lambda, 
                pinv_rtol=pinv_rtol, solver_type=solver_type
            )
        elif vector_type == "tangential":
            return self.grid_to_basis(
                values, grid, helmholtz=True, weights=weights, reg_lambda=reg_lambda, 
                pinv_rtol=pinv_rtol, solver_type=solver_type
            )
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def to_grid_values(self, coeffs, evaluator, vector_type):
        """Deprecated compatibility wrapper."""
        # This wrapper calls the new evaluate method for consistency
        return self.evaluate(coeffs, evaluator.grid, vector_type)

    def regularization_term(self, coeffs, grid, vector_type, reg_lambda=None):
        """Compute regularization penalty term."""
        return super().regularization_term(coeffs, grid, vector_type, reg_lambda=reg_lambda)

    def get_regularization_matrix(self, scalar: bool = True, reg_lambda: Optional[float] = None) -> Optional[np.ndarray]:
        """Get the regularization matrix for SHBasis."""
        if reg_lambda is None:
            return None
            
        if scalar:
            return np.diag(self.n)
        else:
            # Helmholtz/Vector regularization
            L_cf = np.stack(
                [
                    np.diag(self.n * (self.n + 1) / (2 * self.n + 1)),
                    np.zeros((self.index_length, self.index_length)),
                ],
                axis=1,
            )
            L_df = np.stack(
                [
                    np.zeros((self.index_length, self.index_length)),
                    np.diag((self.n + 1) / 2),
                ],
                axis=1,
            )
            return np.array([L_cf, L_df])


    def grid_to_basis_fast(
        self, 
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
        theta: np.ndarray,
        phi: np.ndarray = None,
        weights: np.ndarray = None,
        reg_lambda: float = None,
        vector_type: str = "scalar"
    ) -> np.ndarray:
        """
        Fast Spherical Harmonic Transform for Regular Grids via Separation of Variables.

        This method is significantly faster ($O(N^3)$) than the generic Least Squares 
        solver ($O(N^4)$) for data situated on a regular grid (separable in theta/phi).
        It performs an FFT in longitude (phi) followed by a per-m Regularized Least Squares 
        fit in colatitude (theta).

        Parameters
        ----------
        data : np.ndarray or Tuple[np.ndarray, np.ndarray]
            Input data. 
            - If scalar: array of shape (N_theta, N_phi).
            - If tangential: tuple (u_theta, u_phi), each (N_theta, N_phi).
        theta : np.ndarray
            1D array of colatitudes in radians, shape (N_theta,).
        phi : np.ndarray, optional
            1D array of longitudes in radians, shape (N_phi,).
        weights : np.ndarray, optional
            1D array of weights for the theta dimension, shape (N_theta,).
            (Phi weighting is uniform due to FFT).
        reg_lambda : float, optional
            Tikhonov regularization parameter.
        vector_type : str
            "scalar" or "tangential".

        Returns
        -------
        coeffs : np.ndarray
            Spectral coefficients vector.
```python
        """
        
        N_theta = theta.size
        # Precompute sin(theta) for vector scaling
        sin_th = np.sin(theta)
        # sin_th_safe not needed anymore as get_G(phi) handles poles internally
        
        # 0. Setup and Validation
        is_vector = (vector_type == "tangential")
        
        if is_vector:
            if not isinstance(data, (tuple, list)) or len(data) != 2:
                raise ValueError("For vector_type='tangential', data must be (u_theta, u_phi)")
            d_th_in, d_ph_in = data
            N_theta, N_phi = d_th_in.shape
        else:
            d_in = data
            N_theta, N_phi = d_in.shape

        if len(theta) != N_theta:
            raise ValueError(f"Theta shape {theta.shape} mismatch with data rows {N_theta}")

        # 1. FFT in Longitude
        # -------------------
        # Normalize by N_phi to match PynaMIT orthonormal definition
        if is_vector:
            fft_th = np.fft.fft(d_th_in, axis=1) / N_phi
            fft_ph = np.fft.fft(d_ph_in, axis=1) / N_phi
        else:
            fft_scalar = np.fft.fft(d_in, axis=1) / N_phi

        # 2. Pre-compute 1D Legendre Matrices (P_lm, dP_lm/dth, etc.)
        # -----------------------------------------------------------
        from pynamit.primitives.grid import Grid
        theta_deg = np.rad2deg(theta)
        # Dummy grid for 1D evaluation (phi=0)
        grid_1d = Grid(theta=theta_deg, phi=np.zeros(N_theta))

        # We need P(theta) and potentially dP/dth and P/sin_th (via get_G)
        # get_G(derivative=None) -> P_lm (at phi=0)
        # get_G(derivative='theta') -> dP_lm/dth (at phi=0)
        # get_G(derivative='phi') -> Im * P_lm / sin_th (at phi=0, imaginary handled by complex logic?)
        # Wait, get_G returns REAL matrices.
        # For 'phi' derivative, it returns dY/dphi.
        # Y_c = P cos(m phi), Y_s = P sin(m phi).
        # dY_c/dphi = -m P sin(m phi), dY_s/dphi = m P cos(m phi).
        # At phi=0: dY_c = 0, dY_s = m P.
        # So G_phi at phi=0 contains [0, m P]. Correct.
        # But we need 1/sin_theta * dY/dphi for the vector basis.
        # SHBasis.get_G(derivative='phi') already includes the 1/sin(theta) factor!
        
        G_0 = self.get_G(grid_1d, derivative=None)
        
        if is_vector:
            G_th = self.get_G(grid_1d, derivative='theta')
            G_ph = self.get_G(grid_1d, derivative='phi') # Includes 1/sin(theta)

        # 3. Regularization Setup
        # -----------------------
        # Penalty Matrix L_diag (diagonal of L)
        # Scalar: diag(n)
        # Vector: Pol -> diag(n(n+1)/(2n+1)), Tor -> diag((n+1)/2)
        reg_L = None
        if reg_lambda is not None and reg_lambda > 0:
            if is_vector:
                # Pol and Tor penalties
                # Note: This logic matches get_regularization_matrix
                pen_pol = self.n * (self.n + 1) / (2 * self.n + 1)
                pen_tor = (self.n + 1) / 2
                reg_L = (pen_pol, pen_tor)
            else:
                reg_L = self.n

        # Weights handling
        W_diag = None
        if weights is not None:
            # Sqrt weights for linear system: min || W (Ax - b) ||
            W_diag = weights if weights.ndim == 1 else weights.flatten()
            # If user passed sqrt_weights directly (as is common in PynaMIT), usage is direct.
            # Assuming 'weights' here means sqrt_weights (as in interpolate_and_add_entry)
        
        # 4. Indices
        offset_s = self.cnm.n.size
        coeffs = np.zeros(self.index_length * (2 if is_vector else 1), dtype=float)

        # 5. Solve per m
        # --------------
        limit_m = min(self.Mmax, N_phi // 2)

        # 6. Pre-compute Global Regularization Scaling (Strict Equivalence)
        # -----------------------------------------------------------------
        
        scale_A_global = 1.0
        scale_L_global = 1.0
        
        if reg_lambda is not None and reg_lambda > 0 and reg_L is not None:
             all_norms_A = []
             all_norms_L = []
             
             for m in range(limit_m + 1):
                 mask_c = (self.cnm.m.flatten() == m)
                 if not np.any(mask_c): continue
                 idx_c_out = np.where(mask_c)[0]
                 
                 mask_s = (self.snm.m.flatten() == m)
                 idx_s_out = (np.where(mask_s)[0] + offset_s) if np.any(mask_s) else []
                 
                 # L-Penalty Norms
                 if isinstance(reg_L, tuple):
                     pen_pol, pen_tor = reg_L
                     # Pol coeffs
                     p_vals = pen_pol[idx_c_out % len(pen_pol)]
                     all_norms_L.append(p_vals**2)
                     if m > 0:
                         t_vals = pen_tor[idx_c_out % len(pen_pol)]
                         all_norms_L.append(t_vals**2)
                     if np.any(mask_s):
                          p_vals_s = pen_pol[idx_s_out % len(pen_pol)]
                          all_norms_L.append(p_vals_s**2)
                          t_vals_s = pen_tor[idx_s_out % len(pen_pol)]
                          all_norms_L.append(t_vals_s**2)
                     
                 else:
                     l_vals = reg_L[idx_c_out]
                     all_norms_L.append(l_vals**2)
                     if np.any(mask_s):
                         all_norms_L.append(reg_L[idx_s_out]**2)
                         
                 # A-Matrix Norms using 1D approx
                 # Matching Legacy Scaling:
                 # Legacy Scale_A corresponds to ~ N_phi * Mean(Norms_2D).
                 # Fast Data is 2/N_phi scaled (Energy).
                 # With corrected physics (1/sin scaling for Gang), the Matrix Norms should be correct.
                 # Energy ratio implies phi_factor should be 1.0.
                 
                 phi_factor = N_phi / 2.0 if m > 0 else float(N_phi)
                 w_eff = W_diag if W_diag is not None else np.ones(N_theta)
                 
                 if not is_vector:
                     G_sub = G_0[:, idx_c_out]
                     Gw = G_sub * w_eff[:, None]
                     n_A = np.sum(Gw**2, axis=0) * phi_factor
                     all_norms_A.append(n_A)
                     if np.any(mask_s):
                         all_norms_A.append(n_A)
                 else:
                     Gp = G_th[:, idx_c_out]
                     Gang = np.zeros_like(Gp)
                     if m > 0:
                         # Toroidal term uses 1/sin(theta) * dY/dphi
                         # G_ph is dY/dphi (m*P) / sin(theta) ALREADY (from get_G).
                         Gang = G_ph[:, idx_c_out]
                         # 1/sin scaling is handled in get_G(derivative='phi').
                     term = (Gp**2 + Gang**2) * (w_eff[:, None]**2)
                     n_vec = np.sum(term, axis=0) * phi_factor
                     all_norms_A.append(n_vec)
                     all_norms_A.append(n_vec)
                     if np.any(mask_s):
                         all_norms_A.append(n_vec)
                         all_norms_A.append(n_vec)

             if all_norms_A:
                 flat_A = np.concatenate(all_norms_A)
                 valid_A = flat_A[flat_A > 1e-14]
                 scale_A_global = np.median(valid_A) if valid_A.size else 1.0
                 
             if all_norms_L:
                 flat_L = np.concatenate(all_norms_L)
                 valid_L = flat_L[flat_L > 1e-14]
                 scale_L_global = np.median(valid_L) if valid_L.size else 1.0

        for m in range(limit_m + 1):
            
            # Identify active L-indices for this m
            # Cosine terms (m matches)
            mask_c = (self.cnm.m.flatten() == m)
            if not np.any(mask_c): continue
            idx_c_out = np.where(mask_c)[0]
            
            # Sine terms (m matches)
            mask_s = (self.snm.m.flatten() == m)
            has_sine = np.any(mask_s)
            idx_s_out = (np.where(mask_s)[0] + offset_s) if has_sine else []

            # -------------------------------------------------------------
            # Build System Matrices for this m-block
            # -------------------------------------------------------------
            
            # SCALAR CASE
            if not is_vector:
                # Basis: P_lm(theta)
                # G_sub has shape (N_theta, n_L_active)
                # Columns correspond to P_lm for l >= m
                G_sub_c = G_0[:, idx_c_out] 
                # (For sine terms, P_lm is identical, so we reuse G_sub_c)

                # Targets
                d_c, d_s = self._get_fft_targets(fft_scalar, m)

                # Solve Cosine
                self._solve_stacked(G_sub_c, d_c, coeffs, idx_c_out, W_diag, reg_lambda, reg_L, idx_c_out, scale_A_global, scale_L_global)
                
                # Solve Sine
                if has_sine:
                    self._solve_stacked(G_sub_c, d_s, coeffs, idx_s_out, W_diag, reg_lambda, reg_L, idx_c_out, scale_A_global, scale_L_global)

            # VECTOR CASE
            else:
                # Vector Basis Block Structure (per m):
                # u_theta = -dY/dth * C_pol + 1/sin dY/dphi * C_tor
                # u_phi   = -1/sin dY/dphi * C_pol - dY/dth * C_tor
                #
                # At phi=0:
                # Y_c = P.  dY_c/dphi = 0.
                # Y_s = 0.  dY_s/dphi = m P.
                #
                # Cosine Coeffs (C_lm^c) interact with Cosine Targets?
                # This is tricky mixing.
                # Let's use the exact definition from get_vector_basis_matrix:
                # Pol = -Grad Y. Tor = Curl(rY).
                #
                # We need to construct the system for [C_pol, C_tor]^T given [d_th, d_ph]^T
                # Be careful: FFT separates e^{im\phi}.
                # For m=0: Decoupled.
                
                if m == 0:
                    # Special Case: Decoupled Logic for Zonal Flow
                    # u_theta^c = -P' C_pol^c  (u_theta^s=0, C_tor^s=0)
                    # u_phi^c   = +P' C_tor^c  (u_phi^s=0,   C_pol^s=0)
                    
                    d_th = fft_th[:, 0].real
                    d_ph = fft_ph[:, 0].real
                    
                    n_vals = self.cnm.n.flatten()[idx_c_out]
                    scale_vec = 1.0 # Reset to 1.0 based on diagnostic
                    Gp = G_th[:, idx_c_out] * scale_vec # Scaled derivative
                    
                    # Poloidal Solve: u_theta = -P' -> Target = -Gp
                    idx_pol = idx_c_out
                    self._solve_stacked(-Gp, d_th, coeffs, idx_pol, W_diag, reg_lambda, reg_L, (idx_pol, []), scale_A_global, scale_L_global)
                    
                    # Toroidal Solve: u_phi = P' -> Target = Gp
                    idx_tor = idx_c_out + self.index_length
                    self._solve_stacked(Gp, d_ph, coeffs, idx_tor, W_diag, reg_lambda, reg_L, ([], idx_tor), scale_A_global, scale_L_global)
                    
                else:
                    # Coupled Logic (m > 0)
                    # Targets
                    t_th_c, t_th_s = self._get_fft_targets(fft_th, m)
                    t_ph_c, t_ph_s = self._get_fft_targets(fft_ph, m)

                    # Block 1 (Cos-Sin Mix) [C_pol^c, C_tor^s]
                    # Target u_theta: - ( Pol_theta u_pol^c + Tor_theta u_tor^s )
                    # Target u_phi:   - ( Pol_phi u_pol^c + Tor_phi u_tor^s )? 
                    # Actually, look at Basis matrix: Pol = -Grad, Tor = Curl
                    # A_vec = [[-G_th*cos, -G_ph*sin], [-G_ph*cos, G_th*sin]]
                    
                    n_vals = self.cnm.n.flatten()[idx_c_out]
                    scale_vec = 1.0 
                    
                    Gp = G_th[:, idx_c_out] * scale_vec
                    # G_ph at phi=0 is zero for Cosine terms (dY_c/dphi ~ sin(0)=0).
                    # We need the magnitude, which is stored in Sine terms (dY_s/dphi ~ cos(0)=1).
                    G_ang = G_ph[:, idx_s_out] * scale_vec
                    
                    # Toroidal Scaling & Phase
                    # TS = -1.0 (Global Legacy Toroidal Flip verified).
                    TS = -1.0
                    
                    # Weights (applied to rows)
                    W_block = np.concatenate([W_diag, W_diag]) if W_diag is not None else None

                    # Block 1 (u_th_c, u_ph_s) -> (P_c, T_s)
                    # Pol Cos -> u_phi (Sine): -1/sin d/dphi(P cos) = +m P/sin sin = +Gang. (Positive)
                    # Tor Sin -> u_th (Cos):  -1/sin d/dphi(T sin) = -m T/sin cos = -Gang. (Matches TS=-1).
                    A_11 = -Gp      
                    A_12 = G_ang * TS    
                    A_21 = G_ang          # CORRECTED: Positive G_ang
                    A_22 = -Gp * TS      
                    
                    A_block1 = np.block([[A_11, A_12], [A_21, A_22]])
                    
                    b1 = np.concatenate([t_th_c, t_ph_s])
                    
                    idx_pol_c = idx_c_out
                    idx_tor_s = idx_s_out + self.index_length
                    
                    dest_indices_1 = np.concatenate([idx_pol_c, idx_tor_s])
                    
                    self._solve_stacked(A_block1, b1, coeffs, dest_indices_1, W_block, reg_lambda, reg_L, (idx_c_out, idx_s_out), scale_A_global, scale_L_global)
                 
                    # Block 2 (u_th_s, u_ph_c) -> (P_s, T_c)
                    if has_sine:
                        b2 = np.concatenate([t_th_s, t_ph_c])
                        
                        # Pol Sin -> u_phi (Cos): -1/sin d/dphi(P sin) = -m P/sin cos = -Gang. (Negative)
                        # Tor Cos -> u_th (Sin):  -1/sin d/dphi(T cos) = +m T/sin sin = +Gang. (Matches TS=-1 -> -(-1)=+1).
                        A_11 = -Gp      
                        A_12 = -G_ang * TS   
                        A_21 = -G_ang         # Correct: Negative G_ang
                        A_22 = -Gp * TS      
                        
                        A_block2 = np.block([[A_11, A_12], [A_21, A_22]])
                        
                        idx_pol_s = idx_s_out
                        idx_tor_c = idx_c_out + self.index_length
                        
                        dest_indices_2 = np.concatenate([idx_pol_s, idx_tor_c])
                        self._solve_stacked(A_block2, b2, coeffs, dest_indices_2, W_block, reg_lambda, reg_L, (idx_s_out, idx_c_out), scale_A_global, scale_L_global)

        return coeffs

    def _get_fft_targets(self, fft_data, m):
        """Extract Real (Cosine) and Imag (Sine) targets from FFT."""
        if m == 0:
            return fft_data[:, 0].real, None
        else:
            # 2 * Real for Cosine, -2 * Imag for Sine
            return 2 * fft_data[:, m].real, -2 * fft_data[:, m].imag

    def _solve_stacked(self, A, b, coeffs_out, dest_idxs, weights, reg_lambda, reg_L, reg_idxs_source, scale_A_forced=None, scale_L_forced=None):
        """
        Solve weighted regularized system using Stacked Matrices.
        min || W(Ax - b) ||^2 + lambda || L x ||^2
        
        Equivalent to solving:
        [ W A           ] x = [ W b ]
        [ sqrt(lam) L ]     [ 0   ]
        """
        if len(dest_idxs) == 0: return

        # Apply weights to A/b
        if weights is not None:
             # Broadcast weights if needed (e.g. for block system)
             # A is (N_rows, N_cols). weights (N_rows,)
             A_w = A * weights[:, None]
             b_w = b * weights
        else:
             A_w = A
             b_w = b
             
        # Add Regularization
        if reg_lambda is not None and reg_lambda > 0 and reg_L is not None:
            # Construct Penalty Matrix L_sub
            n_cols = A.shape[1]
            
            # Handle split penalty (Vector case: Pol/Tor distinct)
            if isinstance(reg_L, tuple):
                pen_pol, pen_tor = reg_L
                # Identify which cols are Pol vs Tor based on source indices
                # reg_idxs_source is (idx_pol, idx_tor)
                idx_p, idx_t = reg_idxs_source
                idx_p = np.asarray(idx_p, dtype=int)
                idx_t = np.asarray(idx_t, dtype=int)
                n_p = len(idx_p)
                n_t = len(idx_t) # Should sum to n_cols
                
                L_vals = np.concatenate([
                    pen_pol[idx_p % len(pen_pol)], # Use modulo just in case, but should match
                    pen_tor[idx_t % len(pen_tor)]
                ])
            else:
                # Scalar case
                idx_source = reg_idxs_source
                L_vals = reg_L[idx_source]
            
            # Auto-Scaling Logic (replicating LeastSquaresProblem)
            # ---------------------------------------------------
            # Ratio = Median(diag(A^T A)) / Median(diag(L^T L))
            if scale_A_forced is not None and scale_L_forced is not None:
                scale_A = scale_A_forced
                scale_L = scale_L_forced
            else:
                # Approx diag(A^T A) by column norms squared
                norm_A = np.sum(A_w**2, axis=0) # (n_cols,)
                norm_L = L_vals**2
                
                # Filter zeros
                valid_A = norm_A[norm_A > 1e-14]
                valid_L = norm_L[norm_L > 1e-14]
                
                scale_A = np.median(valid_A) if valid_A.size else 1.0
                scale_L = np.median(valid_L) if valid_L.size else 1.0
            
            factor = np.sqrt(scale_A / scale_L) if scale_L > 0 else 1.0
            
            effective_lam = np.sqrt(reg_lambda) * factor
            
            # Stack
            L_block = np.diag(effective_lam * L_vals)
            zeros_rhs = np.zeros(n_cols)
            
            A_final = np.vstack([A_w, L_block])
            b_final = np.concatenate([b_w, zeros_rhs])
        else:
            A_final = A_w
            b_final = b_w
            
        # Solve
        x, _, _, _ = np.linalg.lstsq(A_final, b_final, rcond=None)
        
        # Store
        coeffs_out[dest_idxs] = x

    def get_analytic_interaction_matrix(
        self, 
        c_pp: np.ndarray, 
        c_mm: np.ndarray, 
        c_pm: np.ndarray, 
        c_mp: np.ndarray,
    ) -> np.ndarray:
        """Compute the Analytic Block Interaction Matrix M.

        This matrix describes the coupling of Poloidal and Toroidal potentials
        via an Anisotropic Conductance Tensor defined by the input spin-weighted
        coefficients.

        Parameters
        ----------
        c_pp, c_mm, c_pm, c_mp : np.ndarray
            Spin-weighted coefficients of the conductivity tensor.
            (pp=Spin 0, mm=Spin 0, pm=Spin +2, mp=Spin -2).

        Returns
        -------
        M : np.ndarray
            The block interaction matrix (2L x 2L).
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        
        # 2. Instantiate Engine with self (ensure consistent basis)
        engine = GauntEngine(self)

        # 3. Compute Matrix. Force input_is_complex=True as inputs come from analyze_spin_weighted.
        return engine.get_general_analytic_interaction_matrix(
            c_pp, c_mm, c_pm, c_mp, input_is_complex=True
        )

    def get_isotropic_interaction_matrix(
        self,
        etaP_coeffs: np.ndarray,
        etaH_coeffs: np.ndarray
    ) -> np.ndarray:
        """Compute the Analytic Interaction Matrix for Isotropic conductivities.
        
        Refactored to use the Verified General Analytic Tensor Solver.
        1. Evaluates Isotropic/Hall coefficients to Grid.
        2. Constructs Canonical Tensor on Grid.
        3. Invokes General Solver.
        """
        # 1. Evaluate to Grid
        
        # Use scalar evaluation (Real SH -> Grid)
        grid = self.integration_grid
        
        # Create basis for Conductance (Nmin=0) to match input coeffs
        cond_basis = self.get_extended_basis()
        
        etaP_grid = cond_basis.evaluate(etaP_coeffs, grid, vector_type="scalar")
        etaH_grid = cond_basis.evaluate(etaH_coeffs, grid, vector_type="scalar")
        
        # 2. Construct Tensor Components
        # Isotropic: S_tt = S_pp = P
        # Hall: S_tp = H, S_pt = -H
        eta_tt = etaP_grid
        eta_pp = etaP_grid
        eta_tp = etaH_grid
        eta_pt = -etaH_grid
        
        # 3. Call General Solver
        return self.get_analytic_interaction_matrix_from_real_grid(
            eta_tt, eta_pp, eta_tp, eta_pt
        )

    def get_quadrature_interaction_matrix(self, sigma_quad: np.ndarray) -> np.ndarray:
        """Compute the Interaction Matrix via Quadrature.
        
        Wrapper for GauntEngine.get_vector_interaction_matrix.
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        engine = GauntEngine(self)
        return engine.get_vector_interaction_matrix(sigma_quad)

    def get_integration_grid(self, grid_resolution: int = None):
        """Get the quadrature grid used for integration."""
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        return GauntEngine(self, grid_resolution=grid_resolution).quad_grid

    @property
    def integration_grid(self):
        """Standard integration grid for this basis."""
        return self.get_integration_grid()

    def analyze_spin_weighted(self, spin: int, values: np.ndarray, grid_resolution: int = None) -> np.ndarray:
        """Analyze spin-weighted field `values` on the quadrature grid.

        Wrapper for GauntEngine.analyze_spin_weighted.
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        engine = GauntEngine(self, grid_resolution=grid_resolution)
        return engine.analyze_spin_weighted(spin, values)

    def get_analytic_interaction_matrix_from_real_grid(
        self,
        eta_tt: np.ndarray,
        eta_pp: np.ndarray,
        eta_tp: np.ndarray,
        eta_pt: np.ndarray,
    ) -> np.ndarray:
        """Compute Analytic Interaction Matrix from REAL Grid Components.

        This method decomposes the physical resistivity tensor components into 
        complex spin-weighted potentials (Spin-0 and Spin-2) used by the analytic solver.

        Physics Mapping (Exhaustive):
        ---------------------------
        Component      | Symmetry         | Potential    | Formula
        -------------------------------------------------------------
        Isotropic      | Symmetric Diag   | Re(Spin-0)   | 0.5 * (eta_tt + eta_pp)
        Hall           | Anti-Symmetric   | Im(Spin-0)   | 0.5 * (eta_tp - eta_pt)
        Aniso (Real)   | Trace-Free Diag  | Re(Spin-2)   | 0.5 * (eta_tt - eta_pp)
        Aniso (Imag)   | Symmetric Off-D  | Im(Spin-2)   | 0.5 * (eta_tp + eta_pt)

        Parameters
        ----------
        eta_tt : np.ndarray
            Theta-Theta resistivity component on the quadrature grid.
        eta_pp : np.ndarray
            Phi-Phi resistivity component on the quadrature grid.
        eta_tp : np.ndarray
            Theta-Phi resistivity component on the quadrature grid.
        eta_pt : np.ndarray
            Phi-Theta resistivity component on the quadrature grid.

        Returns
        -------
        M : np.ndarray
            The Real block interaction matrix.
        """
        # 1. Decompose into Isotropic/Hall (Spin-0) and Anisotropic (Spin-2)
        # -----------------------------------------------------------------
        # Resistance Tensor Structure (Cartesian-Like):
        # [ S_tt, S_tp ]
        # 1. Decompose into Isotropic/Hall (Spin-0) and Anisotropic (Spin-2)
        # -----------------------------------------------------------------
        # Resistivity Tensor Structure:
        # [ eta_tt, eta_tp ]
        # [ eta_pt, eta_pp ]
        
        # Physical Components:
        val_iso      = 0.5 * (eta_tt + eta_pp)
        val_hall_raw = 0.5 * (eta_tp - eta_pt)
        val_aniso_re = 0.5 * (eta_tt - eta_pp)
        val_aniso_im = 0.5 * (eta_tp + eta_pt)

        # Spin-0 Components (Isotropic + Hall)
        # Re(Spin-0) = Isotropic, Im(Spin-0) = -Hall (Alignment Gauge)
        val_0plus  = val_iso - 1j * val_hall_raw
        val_0minus = val_iso + 1j * val_hall_raw
        
        # Spin-2 Components (Anisotropic)
        # Re(Spin-2) = Aniso_re, Im(Spin-2) = Aniso_im
        val_p2_gaunt = val_aniso_re + 1j * val_aniso_im
        val_m2_gaunt = val_aniso_re - 1j * val_aniso_im
        
        # 2. Analyze using Spin-Weighted Harmonics (Nmin=0)
        # Note: Coupling of degree N1 and N2 requires Ls up to N1+N2 (2*Nmax).
        # We must analyze on the SAME grid as the input values (the solver grid).
        sigma_basis = SHBasis(
            2 * self.Nmax, 
            2 * self.Mmax, 
            Nmin=0, 
            quasi_normalized=self.is_normalized, 
            backend=self.backend
        )
        # Solve for the grid resolution that corresponds to the solver basis (self)
        res_solver = int(3.0 * self.Nmax + 10)
        if res_solver % 2 != 0: res_solver += 1
        
        c_pp = sigma_basis.analyze_spin_weighted(0, val_0plus.flatten(), grid_resolution=res_solver)
        c_mm = sigma_basis.analyze_spin_weighted(0, val_0minus.flatten(), grid_resolution=res_solver)
        c_pm = sigma_basis.analyze_spin_weighted(2, val_p2_gaunt.flatten(), grid_resolution=res_solver)
        c_mp = sigma_basis.analyze_spin_weighted(-2, val_m2_gaunt.flatten(), grid_resolution=res_solver)
        
        # 3. Pass Coefficients to Analytic Engine
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        engine = GauntEngine(self)
        
        return engine.get_general_analytic_interaction_matrix(
            c_pp, c_mm, c_pm, c_mp, input_is_complex=True
        )
        
    def project_to_basis(
        self,
        input_values,
        input_grid,
        vector_type,
        target_grid,
        target_basis,
        **kwargs,
    ):
        """Project input data onto the target basis.

        For SHBasis, we fit directly to the input grid, effectively projecting
        onto itself, ignoring the target basis.
        """
        coeffs = self.from_grid_values(
            input_values,
            input_grid,
            vector_type,
            **kwargs,
        )
        return coeffs

    def get_evaluation_matrix(self, grid: Any, derivative: str = None) -> np.ndarray:
        """Get matrix evaluating basis (or derivatives) on a grid. Alias for get_G."""
        return self.get_G(grid, derivative=derivative)

    # Note: get_gradient_matrix, get_curl_matrix, get_vector_basis_matrix 
    # inherited from Basis are sufficient as they use get_evaluation_matrix (via get_G).



    def construct_projection_matrix(self, grid) -> np.ndarray:
        """Construct the projection matrix mapping Grid Vector -> SH Coefficients.
        
        Requires an evaluator that can compute G_helmholtz (vector basis).
        """
        from pynamit.utils import asarray, get_array_module, use_jax, xp, tensor_pinv
        # Calculate pseudo-inverse of the Helmholtz matrix: (2, N_grid, 2, N_sh)
        # Flatten input dims (2, N_grid) -> leading dim

        # Re-calc G_helmholtz locally to avoid legacy dependencies.
        # G_helmholtz = [-G_grad, G_rxgrad]
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")
        
        # Ensure dense for tensor construction
        import scipy.sparse
        G_th = G_th.toarray() if scipy.sparse.issparse(G_th) else G_th
        G_ph = G_ph.toarray() if scipy.sparse.issparse(G_ph) else G_ph
        G_grad = np.array([G_th, G_ph])
        
        # G_rxgrad = [-G_ph, G_th]
        G_rxgrad = np.array([-G_ph, G_th])
        
        G_helmholtz = np.stack([-G_grad, G_rxgrad], axis=2)

        pinv = tensor_pinv(G_helmholtz, n_leading_flattened=2)
        
        # Reshape to 2D matrix: (2*N_sh, 2*N_grid)
        # pinv shape: (comp_out, N_out, comp_in, N_in) = (2, N_sh, 2, N_grid)
        shape = pinv.shape
        return pinv.reshape(shape[0] * shape[1], shape[2] * shape[3])
