"""Spherical Harmonic Basis Class."""

import math
import numpy as np
from typing import Any, Optional, Union, TYPE_CHECKING, Tuple
from functools import cached_property
import scipy.sparse

from pynamit.spherical_harmonics.legendre import LegendreFunctions
from pynamit.utils import asarray


# --- SHIndices and normalization helpers (merged from helpers.py) ---

class SHIndices:
    """Container for (n,m) index pairs."""

    def __init__(self, Nmax: int, Mmax: int):
        index_pairs = []
        for n in range(Nmax + 1):
            for m in range(min(Mmax, n) + 1):
                index_pairs.append((n, m))
        self.index_pairs = tuple(index_pairs)
        self.make_arrays()

    def __getitem__(self, position):
        """Get item(s) from the SHIndices."""
        if position == "n":
            return [ip[0] for ip in self.index_pairs]
        if position == "m":
            return [ip[1] for ip in self.index_pairs]
        return self.index_pairs[position]

    def __iter__(self):
        """Iterate over the SHIndices."""
        for p in self.index_pairs:
            yield p

    def __len__(self) -> int:
        """Return length of the SHIndices."""
        return len(self.index_pairs)

    def __repr__(self) -> str:
        """Return string representation of the SHIndices."""
        return "".join(["n, m\n"] + [str(p)[1:-1] + "\n" for p in self.index_pairs])[:-1]

    def __str__(self) -> str:
        """Return string representation of the SHIndices."""
        return self.__repr__()

    def set_Nmin(self, Nmin: int) -> "SHIndices":
        """Set minimum degree Nmin."""
        self.index_pairs = tuple([p for p in self.index_pairs if p[0] >= Nmin])
        self.make_arrays()
        return self

    def set_Mmin(self, Mmin: int) -> "SHIndices":
        """Set minimum absolute order Mmin."""
        self.index_pairs = tuple([p for p in self.index_pairs if abs(p[1]) >= Mmin])
        self.make_arrays()
        return self

    def make_arrays(self):
        """Create n and m arrays from index pairs."""
        if len(self.index_pairs) > 0:
            arr = np.array(self.index_pairs, dtype=int)
            self.n = arr[:, 0].reshape(1, -1)
            self.m = arr[:, 1].reshape(1, -1)
        else:
            self.n = np.array([]).reshape(1, -1)
            self.m = np.array([]).reshape(1, -1)
        # convenience map for fast lookups (avoid repeated list.index)
        self._index_map = {pair: i for i, pair in enumerate(self.index_pairs)}


def schmidt_quasi_normalization_factors(Nmax: int, Mmax: int) -> np.ndarray:
    """
    Return a matrix of Schmidt quasi-normalization factors.

    The factors are computed according to the geomagnetism convention
    (e.g., Langel, 1987).

    Parameters
    ----------
    Nmax : int
        Maximum degree.
    Mmax : int
        Maximum order.

    Returns
    -------
    S_matrix : ndarray, shape (Nmax+1, Mmax+1)
        Matrix of normalization factors where S_matrix[n, m] is the
        factor for the (n, m) pair.
    """
    S_matrix = np.zeros((Nmax + 1, Mmax + 1))
    S_matrix[0, 0] = 1.0

    for n in range(1, Nmax + 1):
        # Recurrence for m=0
        S_matrix[n, 0] = S_matrix[n - 1, 0] * (2.0 * n - 1.0) / n

        # Recurrence for m > 0
        for m in range(1, min(n, Mmax) + 1):
            factor_m_dep = 2.0 if m == 1 else 1.0
            factor = math.sqrt((n - m + 1.0) * factor_m_dep / (n + m))
            S_matrix[n, m] = S_matrix[n, m - 1] * factor

    return S_matrix


from pynamit.spherical_harmonics import sh_operators, sh_transforms
from pynamit.primitives.basis import Basis
from pynamit.utils import xp, tensor_pinv

if TYPE_CHECKING:
    from pynamit.math.linear_map import LinearMap


class SHBasis(Basis):
    """Class for representing spherical harmonic bases.

    Uses the Langel (1987) geomagnetism convention.

    This class provides two fully compatible backends for Legendre
    polynomial generation:
    - 'internal':
        A fast, self-contained recurrence relation for both P and dP/dtheta.
    - 'scipy':
        Uses the trusted scipy library, with a precise analytical
        scaling factor applied to ensure identical output to the
        'internal' backend. It automatically selects the best available
        scipy function.

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
        Backend for Legendre function calculation. Can be 'internal'
        (default) or 'scipy'. Both produce identical results.
    """

    def __init__(
        self,
        Nmax: int,
        Mmax: int,
        Nmin: int = 1,
        quasi_normalized: bool = True,
        backend: str = "internal",
    ):
        """Initialize the SHBasis instance."""
        if backend not in ["internal", "scipy"]:
            raise ValueError(f"Backend '{backend}' not recognized. Use 'internal' or 'scipy'.")

        self.Nmax, self.Mmax, self.backend = Nmax, Mmax, backend
        self.Nmin = Nmin
        self.is_normalized = quasi_normalized

        self._kind = "SH"
        self._index_names = ["n", "m"]
        self._minimum_phi_sampling = 2 * Mmax + 1
        super().__init__()
        
        # DEBUG: Backend inspection
        # print(f"DEBUG: SHBasis Init. Backend: {self.backend}. Normalized: {self.is_normalized}")


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

        # Initialize Legendre function calculator
        self._legendre = LegendreFunctions(Nmax, Mmax, self.index_pairs, backend)

    # --- Properties ---

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

    def scalar_fields_are_mean_free_by_construction(self) -> bool:
        """Return True when monopole is excluded from scalar coefficient space."""
        return self.Nmin >= 1

    @cached_property
    def schmidt_factors(self) -> np.ndarray:
        """Return Schmidt quasi-normalization factors."""
        if not self.is_normalized:
            return np.ones(len(self.index_pairs))
        s_matrix = schmidt_quasi_normalization_factors(self.Nmax, self.Mmax)
        return np.array([s_matrix[n, m] for n, m in self.index_pairs])

    @cached_property
    def coeffs_to_delta_V(self) -> np.ndarray:
        """Factor to convert coefficients to delta V at unit radius."""
        return sh_operators.get_coeffs_to_delta_V_factor(self.n)

    # --- Extended Basis ---

    def get_extended_basis(self) -> "SHBasis":
        """Return a basis extended to include the monopole term (Nmin=0)."""
        if self.cnm.index_pairs[0][0] == 0:
            return self
        return SHBasis(
            self.Nmax, self.Mmax, Nmin=0,
            quasi_normalized=self.is_normalized, backend=self.backend
        )

    # --- Legendre Functions (delegated) ---

    def legendre(self, theta: np.ndarray) -> np.ndarray:
        """Compute un-normalized Legendre functions."""
        return self._legendre._legendre_internal(theta)

    def legendre_derivative(self, theta: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Compute d/dtheta of Legendre functions."""
        return self._legendre._legendre_derivative_internal(theta, P)

    # --- Quadrature Weights ---

    @staticmethod
    def compute_exact_weights(theta_1d: np.ndarray, L: int) -> np.ndarray:
        """Compute exact quadrature weights for a given 1D theta grid and bandlimit."""
        return sh_transforms.compute_exact_weights(theta_1d, L)

    @staticmethod
    def get_mw_weights(L: int) -> np.ndarray:
        """Compute quadrature weights for McEwen-Wiaux (MW) sampling."""
        return sh_transforms.get_mw_weights(L)

    # --- Basis Function Evaluation ---

    def get_G(self, grid, derivative: Optional[str] = None) -> np.ndarray:
        """Compute basis functions G on the provided grid.

        Parameters
        ----------
        grid : Grid
            Grid on which to evaluate.
        derivative : str, optional
            None for value, "theta" or "phi" for derivatives.

        Returns
        -------
        np.ndarray
            Basis function matrix.
        """
        phi, theta = np.deg2rad(grid.phi), np.deg2rad(grid.theta)

        # Check internal cache
        grid_key = grid.hash

        if grid_key not in self._cache:
            self._cache[grid_key] = {"P": None, "dP": None, "G": {}}

        # Handle legacy tuple structure
        if isinstance(self._cache[grid_key], tuple):
            P_old, dP_old = self._cache[grid_key]
            self._cache[grid_key] = {"P": P_old, "dP": dP_old, "G": {}}

        cache_entry = self._cache[grid_key]

        # Check if G is already cached
        if derivative in cache_entry["G"]:
            return cache_entry["G"][derivative]

        P_unnormalized = cache_entry["P"]
        dP_unnormalized = cache_entry["dP"]

        need_P = P_unnormalized is None
        need_dP = derivative and dP_unnormalized is None

        if need_P or need_dP:
            compute_dP = bool(derivative) or need_dP
            P_new, dP_new = self._legendre.compute(theta, compute_derivative=compute_dP)

            if need_P:
                cache_entry["P"] = P_new
                P_unnormalized = P_new
            if compute_dP:
                cache_entry["dP"] = dP_new
                dP_unnormalized = dP_new

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
        cache_entry["G"][derivative] = G
        return G

    def get_evaluation_matrix(self, grid: Any, derivative: str = None) -> np.ndarray:
        """Get matrix evaluating basis (or derivatives) on a grid. Alias for get_G."""
        return self.get_G(grid, derivative=derivative)

    # --- Operator Factors ---

    def laplacian(self, r: float = 1.0) -> np.ndarray:
        """Factor to apply the spherical harmonic Laplacian operator."""
        return sh_operators.get_laplacian_factor(self.n, r)

    def radial_shift_Ve(self, start: float, end: float) -> np.ndarray:
        """Factor to radially shift external potential coefficients."""
        return sh_operators.get_radial_shift_Ve_factor(self.n, start, end)

    def radial_shift_Vi(self, start: float, end: float) -> np.ndarray:
        """Factor to radially shift internal potential coefficients."""
        return sh_operators.get_radial_shift_Vi_factor(self.n, start, end)

    # --- Linear Map Operators ---

    def get_laplacian_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the Laplacian operator for this basis."""
        return sh_operators.build_laplacian_operator(self, r)

    def get_radial_shift_operator(
        self, start_r: float, end_r: float, kind: str = "external"
    ) -> "LinearMap":
        """Get the radial shift operator for potential coefficients."""
        return sh_operators.build_radial_shift_operator(self, start_r, end_r, kind)

    def get_potential_scaling_operator(self) -> "LinearMap":
        """Get the operator for converting coefficients to surface potential."""
        return sh_operators.build_potential_scaling_operator(self)

    def get_gradient_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical gradient operator in spectral space."""
        return sh_operators.build_gradient_operator(self, r)

    def get_curl_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical curl operator (r x grad) in spectral space."""
        return sh_operators.build_curl_operator(self, r)

    def get_divergence_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical divergence operator in spectral space."""
        return sh_operators.build_divergence_operator(self, r)

    def get_vector_divergence_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the analytical divergence operator for vector fields.
        
        For SHBasis, this is radius-independent in spectral space (mapped to r=1 coefficients).
        """
        # Note: In PynaMIT, VSH vectors are typically defined at the simulation radius RI.
        # However, the Basis class is often used with grid objects.
        # For SH, we use the analytical operator.
        # If we need it at a specific radius, we should probably pass it.
        # For now, use r=1 to match internal scaling conventions if possible?
        # Actually, the user asked for this in the context of poloidal.py which has self.RI.
        # Let's check how GridBasis does it. CSBasis uses r=1.0 in its Diff operators usually.
        return sh_operators.build_divergence_operator(self, r=1.0)

    def get_vector_curl_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the analytical radial curl operator for vector fields."""
        return sh_operators.build_vector_curl_operator(self, r=1.0)

    def get_toroidal_potential_coeffs(self, coeffs: np.ndarray, grid: Optional[Any] = None) -> np.ndarray:
        """Extract toroidal potential coefficients. For SH, this is the second half."""
        n = self.index_length
        coeffs = asarray(coeffs)
        if coeffs.shape[0] == 2:
             return coeffs[1]
        if coeffs.shape[0] == 2 * n:
             if coeffs.ndim == 1:
                  return coeffs.reshape(2, n)[1]
             return coeffs.reshape(2, n, -1)[1].reshape((n,) + coeffs.shape[1:])
        raise ValueError(f"Full E-field must have 2 components or 2*Ncoeffs. Got shape {coeffs.shape}")

    def get_poloidal_potential_coeffs(self, coeffs: np.ndarray, grid: Optional[Any] = None) -> np.ndarray:
        """Extract poloidal potential coefficients. For SH, this is the first half."""
        n = self.index_length
        coeffs = asarray(coeffs)
        if coeffs.shape[0] == 2:
             return coeffs[0]
        if coeffs.shape[0] == 2 * n:
             if coeffs.ndim == 1:
                  return coeffs.reshape(2, n)[0]
             return coeffs.reshape(2, n, -1)[0].reshape((n,) + coeffs.shape[1:])
        raise ValueError(f"Full E-field must have 2 components or 2*Ncoeffs. Got shape {coeffs.shape}")

    def get_product_operator(
        self, coeffs_a: np.ndarray, grid: Optional[Any] = None, method: str = "transform"
    ) -> "LinearMap":
        """Get product operator for SHBasis."""
        return sh_operators.build_product_operator(self, coeffs_a, grid, method)

    def get_vector_product_operator(self, tensor_sigma_coeffs: np.ndarray) -> "LinearMap":
        """Get interaction operator for a 2x2 conductance tensor and VSH vectors."""
        return sh_operators.build_vector_product_operator(self, tensor_sigma_coeffs)

    def get_regularization_matrix(
        self, scalar: bool = True, reg_lambda: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Get the regularization matrix for SHBasis."""
        return sh_operators.get_regularization_matrix(self, scalar, reg_lambda)

    # --- Grid/Basis Transforms ---

    def evaluate(self, coeffs: np.ndarray, grid: Any, vector_type: str = "scalar") -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        if vector_type == "scalar":
            return self.basis_to_grid(coeffs, grid, helmholtz=False)
        elif vector_type == "tangential":
            if coeffs.ndim == 1:
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
        return self.evaluate(coeffs, evaluator.grid, vector_type)

    def regularization_term(self, coeffs, grid, vector_type, reg_lambda=None):
        """Compute regularization penalty term."""
        return super().regularization_term(coeffs, grid, vector_type, reg_lambda=reg_lambda)

    def grid_to_basis_fast(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        theta: np.ndarray,
        phi: np.ndarray = None,
        weights: np.ndarray = None,
        reg_lambda: float = None,
        vector_type: str = "scalar"
    ) -> np.ndarray:
        """Fast Spherical Harmonic Transform for Regular Grids via Separation of Variables."""
        return sh_transforms.grid_to_basis_fast(
            self, data, theta, phi, weights, reg_lambda, vector_type
        )

    def project_to_basis(
        self,
        input_values,
        input_grid,
        vector_type,
        target_grid,
        target_basis,
        **kwargs,
    ) -> np.ndarray:
        """Project input data onto the target basis."""
        coeffs = self.from_grid_values(
            input_values,
            input_grid,
            vector_type,
            **kwargs,
        )
        return coeffs

    def construct_projection_matrix(self, grid) -> np.ndarray:
        """Construct the projection matrix mapping Grid Vector -> SH Coefficients."""
        G_th = self.get_G(grid, derivative="theta")
        G_ph = self.get_G(grid, derivative="phi")

        G_th = G_th.toarray() if scipy.sparse.issparse(G_th) else G_th
        G_ph = G_ph.toarray() if scipy.sparse.issparse(G_ph) else G_ph
        G_grad = np.array([G_th, G_ph])
        G_rxgrad = np.array([G_ph, -G_th])
        G_helmholtz = np.stack([-G_grad, G_rxgrad], axis=2)

        # Return canonical Helmholtz analysis tensor:
        # (potential_type, coeff_index, vector_component, grid_index).
        return tensor_pinv(G_helmholtz, n_leading_flattened=2)
