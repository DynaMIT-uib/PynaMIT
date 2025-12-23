"""Basis Function Utilities.

This module contains the abstract Basis class for basis representations
of fields.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pynamit.primitives.basis_evaluator import BasisEvaluator


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

    Notes
    -----
    Subclasses must implement all abstract methods and properties.
    """

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
        weights: Any = None,
        reg_lambda: Any = None,
        pinv_rtol: float = 1e-15,
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
        return np.array([-G_ph, G_th])

    def get_vector_basis_matrix(self, grid: Any) -> Any:
        """Get vector basis evaluation matrix (Helmholtz decomposition).
        
        Maps [Poloidal_Coeffs; Toroidal_Coeffs] -> [Vector_Theta; Vector_Phi].
        
        Returns
        -------
        matrix : array-like
             Shape (2, N_grid, 2*N_coeffs) or similar.
        """
        # Default: Stack [-Grad, Curl]
        G_grad = self.get_gradient_matrix(grid)
        G_rxgrad = self.get_curl_matrix(grid)
        return np.stack([-G_grad, G_rxgrad], axis=2)

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
