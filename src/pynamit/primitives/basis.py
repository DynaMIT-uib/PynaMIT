"""Basis Function Utilities.

This module contains the abstract Basis class for basis representations
of fields.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pynamit.math.linear_map import LinearMap

from pynamit.math.least_squares_solver import LeastSquaresSolver


# Canonical repo surface Helmholtz convention:
#   cf: cf_sign * grad_S(phi)
#   df: df_sign * (rhat x grad_S(psi))
#
# The current repo choice is ``[-grad_S, -rhat x grad_S]``.
REPO_CF_HELMHOLTZ_SIGN = -1.0
REPO_DF_HELMHOLTZ_SIGN = -1.0


def get_repo_cf_helmholtz_sign() -> float:
    """Return the canonical repo curl-free Helmholtz sign."""
    return float(REPO_CF_HELMHOLTZ_SIGN)


def get_repo_df_helmholtz_sign() -> float:
    """Return the canonical repo divergence-free Helmholtz sign."""
    return float(REPO_DF_HELMHOLTZ_SIGN)


def apply_cf_helmholtz_sign(gradient_matrix: Any) -> Any:
    """Apply the repo curl-free sign to a tangential gradient tensor."""
    return get_repo_cf_helmholtz_sign() * gradient_matrix


def build_df_helmholtz_from_gradient_components(G_th: Any, G_ph: Any) -> np.ndarray:
    """Build the repo df tensor from angular derivative components.

    The conventional ``+rhat x grad`` operator has component form
    ``[-G_phi, G_theta]``. The repo df basis is the global sign multiple of
    that operator.
    """
    df_sign = get_repo_df_helmholtz_sign()
    return np.array([(-df_sign) * G_ph, df_sign * G_th])


def build_helmholtz_tensor_from_gradient_components(G_th: Any, G_ph: Any) -> np.ndarray:
    """Build the canonical repo Helmholtz tensor from angular derivatives."""
    G_grad = np.array([G_th, G_ph])
    G_df = build_df_helmholtz_from_gradient_components(G_th, G_ph)
    return np.stack([apply_cf_helmholtz_sign(G_grad), G_df], axis=2)


def basis_kind(basis: Any) -> Optional[str]:
    """Return the normalized basis-kind identifier, if present."""
    kind = getattr(basis, "kind", None)
    return None if kind is None else str(kind)


def is_basis_kind(basis: Any, *kinds: str) -> bool:
    """Return whether ``basis`` advertises one of the provided kind strings."""
    kind = basis_kind(basis)
    return kind is not None and kind in {str(item) for item in kinds}


def is_sh_basis(basis: Any) -> bool:
    """Return whether ``basis`` uses the spherical-harmonic kind tag."""
    return is_basis_kind(basis, "SH")


def is_cs_basis(basis: Any) -> bool:
    """Return whether ``basis`` uses the cubed-sphere kind tag."""
    return is_basis_kind(basis, "CS")


def is_cs_like_basis(basis: Any) -> bool:
    """Return whether ``basis`` behaves like the CS/grid scalar solution space."""
    return is_basis_kind(basis, "CS", "GRID")


class Basis(ABC):
    """Abstract class for basis representations of fields.

    Defines the interface for different basis representations of fields,
    including functions for evaluating basis functions and their
    derivatives on grids.

    Attributes
    ----------
    kind : str
        Short identifier for the basis.
    index_names : list[str]
        Names of the indices used in the basis representation.
    index_length : int
        Total number of basis functions.
    index_arrays : list
        Arrays containing the indices used in the basis.
    minimum_phi_sampling : float
        Minimum required sampling points in phi direction.
    caching : bool
        Whether basis evaluations can be cached.
    """

    def __init__(self):
        """Initialize the Basis object."""
        self._scalar_solvers: dict[str, LeastSquaresSolver] = {}
        self._helmholtz_solvers: dict[str, LeastSquaresSolver] = {}
        self._cache: dict[Any, Any] = {}

    @property
    def signature(self) -> tuple[Any, ...]:
        """Return a stable cache signature for this basis instance."""
        parts: list[Any] = [
            type(self).__module__,
            type(self).__qualname__,
            self.kind,
            ("repo_cf_sign", get_repo_cf_helmholtz_sign()),
            ("repo_df_sign", get_repo_df_helmholtz_sign()),
        ]
        for name in ("Nmax", "Mmax", "Nmin", "mean_free", "backend", "is_normalized", "N"):
            if hasattr(self, name):
                parts.append((name, getattr(self, name)))
        return tuple(parts)

    @property
    @abstractmethod
    def kind(self) -> str:
        """Short identifier for the basis."""
        pass

    @property
    @abstractmethod
    def index_names(self) -> list[str]:
        """Names of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def index_length(self) -> int:
        """Total number of basis functions."""
        pass

    @property
    @abstractmethod
    def index_arrays(self) -> list:
        """Arrays of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self) -> float:
        """Minimum required sampling in phi direction."""
        pass

    @abstractmethod
    def get_laplacian_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the Laplacian operator for this basis."""
        pass

    @abstractmethod
    def get_radial_shift_operator(
        self, start_r: float, end_r: float, kind: str = "external"
    ) -> "LinearMap":
        """Get the radial shift operator for potential coefficients."""
        pass

    @abstractmethod
    def get_potential_scaling_operator(self) -> "LinearMap":
        """Get the operator for converting coefficients to surface potential."""
        pass

    @abstractmethod
    def get_product_operator(
        self, coeffs_a: np.ndarray, grid: Optional[Any] = None
    ) -> "LinearMap":
        """
        Get a LinearMap that performs multiplication by a field in this basis.

        Applying the resulting operator to coeffs_b should yield the
        coefficients of the field product c = a * b.

        Parameters
        ----------
        coeffs_a : np.ndarray
            Coefficients of the multiplier field 'a'.
        grid : Any, optional
            Grid to use if the implementation requires a transform.

        Returns
        -------
        LinearMap
            Operator M such that M @ coeffs_b = coeffs(a * b).
        """
        pass

    @abstractmethod
    def get_extended_basis(self) -> "Basis":
        """Return a basis extended to include lower-order terms if applicable."""
        pass

    def scalar_fields_are_mean_free_by_construction(self) -> bool:
        """Whether scalar coefficient spaces exclude the monopole by construction.

        When this is ``True``, scalar gauge rows for mean/pinning are typically
        unnecessary for those scalar fields.
        """
        return False

    def supports_regular_grid_fast_path(self) -> bool:
        """Return whether this basis supports the regular-grid fast projection path."""
        return False

    @abstractmethod
    def get_vector_curl_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the analytical curl operator for vector fields.

        Parameters
        ----------
        grid : Any, optional
            The grid on which the operator is evaluated. If None, basis may use its default grid.

        Returns
        -------
        LinearMap
            The curl operator.
        """
        pass

    @abstractmethod
    def get_vector_divergence_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the analytical divergence operator for vector fields.

        Parameters
        ----------
        grid : Any, optional
            The grid on which the operator is evaluated. If None, basis may use its default grid.

        Returns
        -------
        LinearMap
            The divergence operator.
        """
        pass

    @abstractmethod
    def get_toroidal_potential_coeffs(
        self, coeffs: np.ndarray, grid: Optional[Any] = None
    ) -> np.ndarray:
        """Extract toroidal potential coefficients from vector coefficients."""
        pass

    @abstractmethod
    def get_poloidal_potential_coeffs(
        self, coeffs: np.ndarray, grid: Optional[Any] = None
    ) -> np.ndarray:
        """Extract poloidal potential coefficients from vector coefficients."""
        pass

    @abstractmethod
    def evaluate(self, coeffs: np.ndarray, grid: Any, vector_type: str = "scalar") -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        pass

    @abstractmethod
    def from_grid_values(
        self, values: np.ndarray, grid: Any, vector_type: str, **kwargs
    ) -> np.ndarray:
        """Convert grid values to coefficients."""
        pass

    @abstractmethod
    def get_evaluation_matrix(self, grid: Any, derivative: str = None) -> Any:
        """Get matrix evaluating basis (or derivatives) on a grid.

        Parameters
        ----------
        grid : Grid
            Target grid.
        derivative : str, optional
            'theta', 'phi', or None.

        Returns
        -------
        matrix : np.ndarray or sparse matrix
            Operator mapping coefficients to grid values.
        """
        pass

    def get_gradient_matrix(self, grid: Any) -> Any:
        """Get gradient operator matrix (components [d/dtheta, 1/sin d/dphi]).

        Returns
        -------
        matrix : array-like
            Shape (2, N_grid, N_coeffs). Stacked [G_theta, G_phi].
        """
        G_th = self.get_evaluation_matrix(grid, "theta")
        G_ph = self.get_evaluation_matrix(grid, "phi")

        # Ensure consistent array/matrix types via backend utils?
        # For now, rely on subclass implementation or simple stacking
        # This default implementation assumes subclasses return compatible types
        # But sparse stacking is backend-specific.
        # So maybe abstract is safer, or explicit check.
        # Converting to dense for safety in default impl:
        try:
            import scipy.sparse

            is_sparse = scipy.sparse.issparse(G_th)
            if is_sparse:
                G_th = G_th.toarray()
                G_ph = G_ph.toarray()
        except ImportError:
            pass

        return np.array([G_th, G_ph])

    def get_curl_matrix(self, grid: Any) -> Any:
        """Get toroidal curl operator matrix ``Curl(T r) = -r x Grad T``.

        This is also the repository-wide Helmholtz divergence-free sign
        convention. For generic tangential vector fields we therefore use the
        opposite df sign from Laundal et al. (2025) Appendix C1, which writes
        the df basis as ``+r x grad``.

        Returns
        -------
        matrix : array-like
            Shape (2, N_grid, N_coeffs). Stacked [G_phi, -G_theta].
        """
        G_grad = self.get_gradient_matrix(grid)
        G_th, G_ph = G_grad[0], G_grad[1]
        return build_df_helmholtz_from_gradient_components(G_th, G_ph)

    def get_vector_basis_matrix(self, grid: Any) -> Any:
        """Get vector basis evaluation matrix (Helmholtz decomposition).

        Maps [Poloidal_Coeffs; Toroidal_Coeffs] -> [Vector_Theta; Vector_Phi].

        Definition (Uniform Potential Convention):
        Poloidal: -Grad P
        Toroidal: Curl(T r) = -r x Grad T

        For generic Helmholtz vector fields this means the repo uses
        ``[-grad, -r x grad]``. Relative to Laundal et al. (2025) Appendix C1,
        only the divergence-free channel changes sign.

        Returns
        -------
        matrix : array-like
            Canonical Helmholtz tensor with shape ``(2, N_grid, 2, N_coeffs)``.
        """
        G_th = self.get_evaluation_matrix(grid, "theta")
        G_ph = self.get_evaluation_matrix(grid, "phi")
        return build_helmholtz_tensor_from_gradient_components(G_th, G_ph)

    def get_scaled_matrix(self, grid: Any, factor: Any) -> Any:
        """Get evaluation matrix scaled by a factor (row or column).

        Parameters
        ----------
        grid : Grid
            Target grid.
        factor : scalar or array-like
            Scaling factor. If array, detects row/col scaling by shape.

        Returns
        -------
        matrix : array-like
            Scaled matrix G.
        """
        import scipy.sparse

        G = self.get_evaluation_matrix(grid)

        if np.isscalar(factor):
            return factor * G

        factor_arr = np.asarray(factor).ravel()
        rows, cols = G.shape
        is_sparse = scipy.sparse.issparse(G)

        if factor_arr.size == rows:
            if is_sparse:
                return scipy.sparse.diags(factor_arr) @ G
            else:
                return G * factor_arr.reshape(-1, 1)
        elif factor_arr.size == cols:
            if is_sparse:
                return G @ scipy.sparse.diags(factor_arr)
            else:
                return G * factor_arr
        else:
            raise ValueError(
                f"Factor size {factor_arr.size} does not match G shape {G.shape} "
                "for either row or column scaling."
            )

    def get_regularization_matrix(
        self, scalar: bool = True, reg_lambda: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Get the regularization matrix for this basis. Default is None."""
        return None

    def construct_scalar_projection_matrix(self, grid: Any) -> Any:
        """Construct the scalar projection matrix mapping Grid Values -> Basis Coefficients.

        This method centralizes the logic for scalar projection/analysis.
        If the grid provides quadrature weights, it uses a precise weighted least-squares:
             P = (G^T W G)^{-1} G^T W
        Otherwise, it falls back to a standard pseudo-inverse:
             P = pinv(G)

        Parameters
        ----------
        grid : object
            The grid on which values are defined.

        Returns
        -------
        P : array-like
            Projection matrix of shape (n_coeffs, n_grid).
        """
        from pynamit.primitives.analysis import get_scalar_projection_matrix
        from pynamit.primitives.field_spec import FieldSpec

        spec = FieldSpec(
            basis=self, field_type="scalar", mean_free=bool(getattr(self, "mean_free", False))
        )
        return get_scalar_projection_matrix(spec, grid)

    def grid_to_basis(
        self,
        values: np.ndarray,
        grid: Any,
        helmholtz: bool = False,
        weights: Optional[np.ndarray] = None,
        reg_lambda: Optional[float] = None,
        pinv_rtol: float = 1e-15,
        solver_type: str = "svd",
    ) -> np.ndarray:
        """Project grid values to basis coefficients."""
        from pynamit.primitives.analysis import (
            get_helmholtz_least_squares_problem,
            get_scalar_least_squares_problem,
        )
        from pynamit.primitives.field_spec import FieldSpec

        spec = FieldSpec(
            basis=self,
            field_type="tangential" if helmholtz else "scalar",
            mean_free=bool(getattr(self, "mean_free", False)),
        )
        if helmholtz:
            if solver_type not in self._helmholtz_solvers:
                self._helmholtz_solvers[solver_type] = LeastSquaresSolver(
                    solver=solver_type, tolerance=pinv_rtol
                )
            solver = self._helmholtz_solvers[solver_type]
            problem = get_helmholtz_least_squares_problem(
                spec, grid, sqrt_weights=weights, reg_lambda=reg_lambda
            )
        else:
            if solver_type not in self._scalar_solvers:
                self._scalar_solvers[solver_type] = LeastSquaresSolver(
                    solver=solver_type, tolerance=pinv_rtol
                )
            solver = self._scalar_solvers[solver_type]
            problem = get_scalar_least_squares_problem(
                spec, grid, sqrt_weights=weights, reg_lambda=reg_lambda
            )

        # Basis-specific gauge constraints (e.g., CS Helmholtz constant-mode nulls).
        if helmholtz and hasattr(self, "get_helmholtz_gauge_constraint_matrix"):
            C = np.asarray(self.get_helmholtz_gauge_constraint_matrix())
            if C.ndim == 1:
                C = C.reshape(1, -1)
            if C.ndim == 2 and C.shape[0] > 0:
                return solver.solve(
                    problem=problem,
                    rhs=[values],
                    equality_operator=C,
                    equality_rhs=np.zeros(C.shape[0], dtype=C.dtype),
                    elimination_rcond=pinv_rtol,
                )

        return solver.solve(problem=problem, rhs=[values])

    def basis_to_grid(
        self,
        coeffs: np.ndarray,
        grid: Any,
        derivative: Optional[str] = None,
        helmholtz: bool = False,
    ) -> np.ndarray:
        """Interpolate coefficients to a grid."""
        if derivative:
            G = self.get_evaluation_matrix(grid, derivative=derivative)
            return G.dot(coeffs)
        elif helmholtz:
            G_h = self.get_vector_basis_matrix(grid)
            return np.tensordot(G_h, coeffs, 2)
        else:
            G = self.get_evaluation_matrix(grid)
            return G.dot(coeffs)

    def regularization_term(
        self,
        coeffs: np.ndarray,
        grid: Any,
        vector_type: str = "scalar",
        reg_lambda: Optional[float] = None,
    ) -> float:
        """Compute the regularization penalty term."""
        if reg_lambda is None or reg_lambda == 0:
            return 0.0

        is_scalar = vector_type == "scalar"
        L = self.get_regularization_matrix(scalar=is_scalar, reg_lambda=reg_lambda)
        if L is None:
            return 0.0

        if not is_scalar:
            return np.tensordot(L, coeffs, 2)
        else:
            return np.dot(coeffs, np.dot(L, coeffs))
