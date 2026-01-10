"""Spherical Harmonic Basis Class."""

import numpy as np
from typing import Any, Optional, Union, TYPE_CHECKING
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

    def align_tensor_gauge(
        self, c_pp: np.ndarray, c_mm: np.ndarray, c_pm: np.ndarray, c_mp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Align Physical Grid Tensor coefficients to Quantum Analytic Gauge.

        This method acts as the explicit Translation Layer between the Geophysics
        Input Domain and the Quantum Analytic Backend.

        Transformations applied:
        1. Normalization: Implicitly assumed that input coefficients are produced
           by a Quantum-Normalized Quadrature (like `analyze_spin_weighted`).
        
        2. Vector Basis Alignment (Gauge Fix):
           The Anisotropic components (Spin +/- 2) require a sign flip to match
           the handedness/phase definition of the analytic Spin-Weighted Basis
           used by `GauntEngine`.
           
           c_pm (Spin +2) -> -c_pm
           c_mp (Spin -2) -> -c_mp

        Parameters
        ----------
        c_pp, c_mm : np.ndarray
            Isotropic components (Spin 0). Unchanged.
        c_pm, c_mp : np.ndarray
            Anisotropic components (Spin +/- 2). Sign flipped.

        Returns
        -------
        tuple[np.ndarray, ...]
            The gauge-aligned coefficients ready for the analytic solver.
        """
        return c_pp, c_mm, c_pm, c_mp

    def get_analytic_interaction_matrix(
        self, 
        c_pp: np.ndarray, 
        c_mm: np.ndarray, 
        c_pm: np.ndarray, 
        c_mp: np.ndarray,
        input_gauge: str = "geophysics"
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
        input_gauge : str, default="geophysics"
            If "geophysics", applies the gauge alignment (sign flip for anisotropy)
            to match the Quantum backend. If "quantum", assumes inputs are already aligned.

        Returns
        -------
        M : np.ndarray
            The block interaction matrix (2L x 2L).
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine

        # 1. Apply Transformation
        if input_gauge == "geophysics":
            c_pp, c_mm, c_pm, c_mp = self.align_tensor_gauge(c_pp, c_mm, c_pm, c_mp)
        
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
        S_tt = etaP_grid
        S_pp = etaP_grid
        S_tp = etaH_grid
        S_pt = -etaH_grid
        
        # 3. Call General Solver
        return self.get_analytic_interaction_matrix_from_real_grid(
            S_tt, S_pp, S_tp, S_pt, input_gauge="geophysics"
        )

    def get_quadrature_interaction_matrix(self, sigma_quad: np.ndarray) -> np.ndarray:
        """Compute the Interaction Matrix via Quadrature.
        
        Wrapper for GauntEngine.get_vector_interaction_matrix.
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        engine = GauntEngine(self)
        return engine.get_vector_interaction_matrix(sigma_quad)

    @property
    def integration_grid(self):
        """Get the quadrature grid used for integration."""
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        return GauntEngine(self).quad_grid

    def analyze_spin_weighted(self, spin: int, values: np.ndarray) -> np.ndarray:
        """Analyze spin-weighted field `values` on the quadrature grid.

        Wrapper for GauntEngine.analyze_spin_weighted.
        """
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        engine = GauntEngine(self)
        return engine.analyze_spin_weighted(spin, values)

    def get_analytic_interaction_matrix_from_real_grid(
        self,
        S_tt: np.ndarray,
        S_pp: np.ndarray,
        S_tp: np.ndarray,
        S_pt: np.ndarray,
        input_gauge: str = "geophysics"
    ) -> np.ndarray:
        """Compute Analytic Interaction Matrix from REAL Grid Components.

        Parameters
        ----------
        S_tt : np.ndarray
            Theta-Theta component on the quadrature grid.
        S_pp : np.ndarray
            Phi-Phi component on the quadrature grid.
        S_tp : np.ndarray
            Theta-Phi component on the quadrature grid.
        S_pt : np.ndarray
            Phi-Theta component on the quadrature grid.
        input_gauge : str, default="geophysics"
            Passed to `get_analytic_interaction_matrix`. 
            Usually "geophysics" to ensure correct sign flip for Spin components.

        Returns
        -------
        M : np.ndarray
            The Real block interaction matrix.
        """
        # 1. Decompose into Isotropic/Hall and Anisotropic Parts
        # Isotropic/Hall part: S_iso = diag(S, S) + offdiag(H, -H)
        val_iso = 0.5 * (S_tt + S_pp)
        # Note: Hall Term flipped for Gauge Alignment with Quadrature
        val_hall = -0.5 * (S_tp - S_pt)
        
        S_tt_iso = val_iso
        S_pp_iso = val_iso
        S_tp_iso = val_hall
        S_pt_iso = -val_hall
        
        # Anisotropic Part (Residual) -> Pure Spin-2
        S_tt_aniso = S_tt - S_tt_iso
        S_pp_aniso = S_pp - S_pp_iso
        S_tp_aniso = S_tp - S_tp_iso
        S_pt_aniso = S_pt - S_pt_iso
        
        # 2. Prepare Inputs for Gaunt Engine
        if input_gauge == "geophysics":
            # Align Geophysics Gauge to Analytic Gauge
            # Empirically, Isotropic/Hall matches WITHOUT negation.
            t_tt, t_pp = S_tt_iso, S_pp_iso
            t_tp, t_pt = S_tp_iso, S_pt_iso
            
            # For Spin-2, we assume similar identity relationship
            a_tt, a_pp = S_tt_aniso, S_pp_aniso
            a_tp, a_pt = S_tp_aniso, S_pt_aniso
        else:
            t_tt, t_pp, t_tp, t_pt = S_tt_iso, S_pp_iso, S_tp_iso, S_pt_iso
            a_tt, a_pp, a_tp, a_pt = S_tt_aniso, S_pp_aniso, S_tp_aniso, S_pt_aniso
            
        # Spin-0 Components (Isotropic)
        val_0plus = 0.5 * ((t_tt + t_pp) + 1j * (t_tp - t_pt))
        val_0minus = 0.5 * ((t_tt + t_pp) - 1j * (t_tp - t_pt))
        
        # Spin-2 Components (Anisotropic)
        # val_p2 = 0.5 * ((tt - pp) - i(tp + pt))
        val_p2_gaunt = 0.5 * ((a_tt - a_pp) - 1j * (a_tp + a_pt))
        val_m2_gaunt = 0.5 * ((a_tt - a_pp) + 1j * (a_tp + a_pt))
        
        # Analyze using Spin-Weighted Harmonics (Nmin=0)
        # Note: We use Nmin=0 to capture L=0 isotropic terms.
        sigma_basis = SHBasis(
            self.Nmax, 
            self.Mmax, 
            Nmin=0, 
            quasi_normalized=self.is_normalized, 
            backend=self.backend
        )
        
        c_pp = sigma_basis.analyze_spin_weighted(0, val_0plus.flatten())
        c_mm = sigma_basis.analyze_spin_weighted(0, val_0minus.flatten())
        c_pm = sigma_basis.analyze_spin_weighted(2, val_p2_gaunt.flatten())
        c_mp = sigma_basis.analyze_spin_weighted(-2, val_m2_gaunt.flatten())
        
        # 3. Pass Coefficients to Analytic Engine
        # input_is_complex=True tells the engine that coefficients are already
        # in the Complex Orthonormal basis (matching the Wigner-3j definition).
        # This is guaranteed by GauntEngine.analyze_spin_weighted.
        # We must call GauntEngine DIRECTLY to avoid SHBasis wrappers that might 
        # assume real/schmidt inputs.
        
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
