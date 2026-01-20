"""Basis Function Utilities.

This module contains the abstract Basis class for basis representations
of fields.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pynamit.math.least_squares_problem import LeastSquaresProblem
    from pynamit.math.linear_map import LinearMap

from pynamit.math.least_squares_solver import LeastSquaresSolver


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
    def get_product_operator(self, coeffs_a: np.ndarray, grid: Optional[Any] = None) -> "LinearMap":
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
    def get_toroidal_potential_coeffs(self, coeffs: np.ndarray, grid: Optional[Any] = None) -> np.ndarray:
        """Extract toroidal potential coefficients from vector coefficients."""
        pass

    @abstractmethod
    def get_poloidal_potential_coeffs(self, coeffs: np.ndarray, grid: Optional[Any] = None) -> np.ndarray:
        """Extract poloidal potential coefficients from vector coefficients."""
        pass



    @abstractmethod
    def evaluate(
        self, coeffs: np.ndarray, grid: Any, vector_type: str = "scalar"
    ) -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        pass

    @abstractmethod
    def from_grid_values(
        self,
        values: np.ndarray,
        grid: Any,
        vector_type: str,
        **kwargs,
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
        """Get curl (R x Grad) operator matrix.
        
        Returns
        -------
        matrix : array-like
            Shape (2, N_grid, N_coeffs). Stacked [-G_phi, G_theta].
        """
        G_grad = self.get_gradient_matrix(grid)
        G_th, G_ph = G_grad[0], G_grad[1]
        
        # Option 1: Uniform Potential Convention (-,-)
        # Poloidal: -Grad V
        # Toroidal: Curl(Tr) = -r x Grad T
        # r x Grad = [-G_phi, G_theta] -> -r x Grad = [G_phi, -G_theta]
        return np.array([G_ph, -G_th])

    def get_vector_basis_matrix(self, grid: Any) -> Any:
        """Get vector basis evaluation matrix (Helmholtz decomposition).
        
        Maps [Poloidal_Coeffs; Toroidal_Coeffs] -> [Vector_Theta; Vector_Phi].
        
        Definition (Uniform Potential Convention):
        Poloidal: -Grad P
        Toroidal: Curl(T r) = -r x Grad T
        
        Returns
        -------
        matrix : array-like
             Shape (2, N_grid, 2*N_coeffs) or similar.
        """
        # Default: Stack [-Grad, Curl(T r)]
        # get_gradient_matrix returns Grad.
        # get_curl_matrix now returns Curl(T r) = -r x Grad.
        G_grad = self.get_gradient_matrix(grid)
        G_curl_Tr = self.get_curl_matrix(grid)
        
        # Poloidal = -Grad
        G_pol = -G_grad
        
        return np.stack([G_pol, G_curl_Tr], axis=2)

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
    def get_least_squares_problem(
        self,
        grid: Any,
        sqrt_weights: Optional[np.ndarray] = None,
        reg_lambda: Optional[float] = None,
    ) -> "LeastSquaresProblem":
        """Get a least squares problem for scalar projection."""
        from pynamit.math.least_squares_problem import LeastSquaresProblem
        G = self.get_evaluation_matrix(grid)
        L = self.get_regularization_matrix(scalar=True, reg_lambda=reg_lambda)
        
        reg_matrices = [L] if L is not None else []
        reg_weights = [reg_lambda] if reg_lambda is not None else []
        
        return LeastSquaresProblem(
            A=[G],
            solution_shape=self.index_length,
            data_shapes=[grid.size],
            sqrt_weights=[sqrt_weights],
            regularization_weights=reg_weights,
            regularization_matrices=reg_matrices,
        )

    def get_least_squares_problem_helmholtz(
        self,
        grid: Any,
        sqrt_weights: Optional[np.ndarray] = None,
        reg_lambda: Optional[float] = None,
    ) -> "LeastSquaresProblem":
        """Get a least squares problem for vector projection."""
        from pynamit.math.least_squares_problem import LeastSquaresProblem
        G_h = self.get_vector_basis_matrix(grid)
        L_h = self.get_regularization_matrix(scalar=False, reg_lambda=reg_lambda)
        
        reg_matrices = [L_h] if L_h is not None else []
        reg_weights = [reg_lambda] if reg_lambda is not None else []
        
        return LeastSquaresProblem(
            A=[G_h],
            solution_shape=(2, self.index_length),
            data_shapes=[(2, grid.size)],
            sqrt_weights=[sqrt_weights],
            regularization_weights=reg_weights,
            regularization_matrices=reg_matrices,
        )

    def get_regularization_matrix(self, scalar: bool = True, reg_lambda: Optional[float] = None) -> Optional[np.ndarray]:
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
        import scipy.sparse
        from pynamit.utils import to_numpy, asarray, tensor_pinv

        # 1. Get Evaluation Matrix G
        G = self.get_evaluation_matrix(grid)
        
        # 2. Check for Weights (Quadrature support)
        if hasattr(grid, "weights") and grid.weights is not None:
            weights = grid.weights
            
            # Weighted Least Squares: P = (G^T W G)^{-1} G^T W
            # If G is sparse, ensure we handle it efficiently
            is_sparse = scipy.sparse.issparse(G)
            
            # G^T W
            if is_sparse:
                 # Sparse matrix element-wise multiplication with weights broadcasted?
                 # G.T is (N_basis, N_grid). weights is (N_grid,).
                 # G.T * weights scales columns of G.T (rows of G).
                 # Correct logic:
                 # W is diag(weights). GtW = G.T @ W.
                 # equivalent to multiplying each column j of G.T by weights[j]
                 
                 # scipy.sparse multiply: row scaling?
                 # G.T is csr/csc.
                 # Let's use robust diag multiplication
                 W_diag = scipy.sparse.diags(weights)
                 GtW = G.T @ W_diag
            else:
                 GtW = G.T * weights
            
            # Mass Matrix M = G^T W G (Should be small: N_basis x N_basis)
            M = GtW @ G
            
            if is_sparse:
                 M = M.toarray()
                 
            # Solve P = M^-1 GtW
            # Logic: P @ x_grid = c_basis
            # M @ c_basis = GtW @ x_grid
            # So P is the operator that applies M^-1 to the result of GtW @ x
            # P = solve(M, GtW)
            
            # If GtW is sparse, we might want to densify it for solve if it's not too huge,
            # or solve for each column (expensive). 
            # Usually N_basis is small enough that we can just invert M and multiply.
            # But let's stick to solve.
            if is_sparse:
                 GtW = GtW.toarray()
                 
            P = np.linalg.solve(M, GtW)
            return asarray(P)

        # 3. Fallback: Pseudo-Inverse
        # Ensure dense for tensor_pinv if currently sparse, or update tensor_pinv to handle sparse?
        # tensor_pinv handles dense arrays usually.
        if scipy.sparse.issparse(G):
             G = G.toarray()
             
        return tensor_pinv(G, n_leading_flattened=1)

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
        if helmholtz:
            if solver_type not in self._helmholtz_solvers:
                self._helmholtz_solvers[solver_type] = LeastSquaresSolver(
                    solver=solver_type, tolerance=pinv_rtol
                )
            solver = self._helmholtz_solvers[solver_type]
            problem = self.get_least_squares_problem_helmholtz(grid, weights, reg_lambda)
        else:
            if solver_type not in self._scalar_solvers:
                self._scalar_solvers[solver_type] = LeastSquaresSolver(
                    solver=solver_type, tolerance=pinv_rtol
                )
            solver = self._scalar_solvers[solver_type]
            problem = self.get_least_squares_problem(grid, weights, reg_lambda)
            
        return solver.solve(problem=problem, rhs=[values])

    def basis_to_grid(
        self, coeffs: np.ndarray, grid: Any, derivative: Optional[str] = None, helmholtz: bool = False
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
        self, coeffs: np.ndarray, grid: Any, vector_type: str = "scalar", reg_lambda: Optional[float] = None
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
