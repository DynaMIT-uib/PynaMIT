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
        from pynamit.utils import tensor_pinv
        
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
