"""Basis evaluator module.

This module contains the BasisEvaluator class for evaluating basis
expansions on a grid.
"""

from __future__ import annotations
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid


class BasisEvaluator:
    """Object for evaluating basis expansions on a grid.

    This class provides methods for evaluating basis expansions on a
    grid and for constructing least-squares problems to find the basis
    expansion coefficients corresponding to given grid values.
    """

    def __init__(
        self,
        basis: SHBasis,
        grid: Grid,
        sqrt_weights: Optional[np.ndarray] = None,
        reg_lambda: Optional[float] = None,
        pinv_rtol: float = 1e-15,
    ) -> None:
        """Initialize the BasisEvaluator object."""
        self.basis = basis
        self.grid = grid
        self.sqrt_weights = sqrt_weights
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

        # Caches for configured, stateless solver instances.
        self._scalar_solvers: Dict[str, LeastSquaresSolver] = {}
        self._helmholtz_solvers: Dict[str, LeastSquaresSolver] = {}

        # Internal cache for basis generation (e.g. Legendre polynomials)
        self._cache: Optional[Any] = None

    @cached_property
    def G(self) -> np.ndarray:
        """Evaluation matrix."""
        if self.basis.caching:
            if self._cache is None:
                G, self._cache = self.basis.get_G(self.grid, cache_out=True)
                return G
            else:
                return self.basis.get_G(self.grid, cache_in=self._cache)
        else:
            return self.basis.get_G(self.grid)

    @cached_property
    def G_th(self) -> np.ndarray:
        """Matrix evaluating the theta derivative."""
        if self.basis.caching:
            if self._cache is None:
                G_th, self._cache = self.basis.get_G(self.grid, derivative="theta", cache_out=True)
                return G_th
            else:
                return self.basis.get_G(self.grid, derivative="theta", cache_in=self._cache)
        else:
            return self.basis.get_G(self.grid, derivative="theta")

    @cached_property
    def G_ph(self) -> np.ndarray:
        """Matrix evaluating the phi derivative."""
        if self.basis.caching:
            if self._cache is None:
                G_ph, self._cache = self.basis.get_G(self.grid, derivative="phi", cache_out=True)
                return G_ph
            else:
                return self.basis.get_G(self.grid, derivative="phi", cache_in=self._cache)
        else:
            return self.basis.get_G(self.grid, derivative="phi")

    @cached_property
    def G_grad(self) -> np.ndarray:
        """Matrix evaluating the horizontal gradient."""
        # Ensure dense for tensor construction
        import scipy.sparse
        G_th = self.G_th.toarray() if scipy.sparse.issparse(self.G_th) else self.G_th
        G_ph = self.G_ph.toarray() if scipy.sparse.issparse(self.G_ph) else self.G_ph
        return np.array([G_th, G_ph])

    @cached_property
    def G_rxgrad(self) -> np.ndarray:
        """Matrix evaluating r-hat x horizontal gradient."""
        # Ensure dense usage via G_grad or G_th/G_ph
        # Access G_grad to ensure consistent dense conversion
        # G_grad is [G_th, G_ph]
        G_th, G_ph = self.G_grad[0], self.G_grad[1]
        return np.array([-G_ph, G_th])

    @cached_property
    def G_rxgrad_pinv(self) -> np.ndarray:
        """Matrix evaluating r-hat x horizontal gradient pinv."""
        return np.linalg.pinv(self.G_rxgrad)

    @cached_property
    def G_helmholtz(self) -> np.ndarray:
        """Matrix evaluating horizontal vector field expansions."""
        return np.stack([-self.G_grad, self.G_rxgrad], axis=2)

    @cached_property
    def L(self) -> Optional[np.ndarray]:
        """Regularization matrix for scalar fields."""
        if self.reg_lambda is None:
            return None
        if not hasattr(self.basis, "n"):
             raise ValueError("Basis does not support regularization (missing 'n' attribute).")
        return np.diag(self.basis.n)

    @cached_property
    def L_helmholtz(self) -> Optional[np.ndarray]:
        """Regularization matrix for horizontal vector fields."""
        if self.reg_lambda is None:
            return None
        if not hasattr(self.basis, "n"):
             raise ValueError("Basis does not support regularization (missing 'n' attribute).")

        L_cf = np.stack(
            [
                np.diag(self.basis.n * (self.basis.n + 1) / (2 * self.basis.n + 1)),
                np.zeros((self.basis.index_length, self.basis.index_length)),
            ],
            axis=1,
        )
        L_df = np.stack(
            [
                np.zeros((self.basis.index_length, self.basis.index_length)),
                np.diag((self.basis.n + 1) / 2),
            ],
            axis=1,
        )
        return np.array([L_cf, L_df])

    @cached_property
    def least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        # Note: cached_property handles the single initialization
        reg_matrices = [self.L] if self.L is not None else []
        reg_weights = [self.reg_lambda] if self.reg_lambda is not None else []

        # Assuming grid.size is int, data_shapes needs a list of tuples or ints
        return LeastSquaresProblem(
            A=[self.G],
            solution_shape=self.basis.index_length,
            data_shapes=[self.grid.size],
            sqrt_weights=[self.sqrt_weights],
            regularization_weights=reg_weights,
            regularization_matrices=reg_matrices,
        )

    @cached_property
    def least_squares_problem_helmholtz(self) -> LeastSquaresProblem:
        """Least squares problem for horizontal vector fields."""
        reg_matrices = [self.L_helmholtz] if self.L_helmholtz is not None else []
        reg_weights = [self.reg_lambda] if self.reg_lambda is not None else []

        return LeastSquaresProblem(
            A=[self.G_helmholtz],
            solution_shape=(2, self.basis.index_length),
            data_shapes=[(2, self.grid.size)],
            sqrt_weights=[self.sqrt_weights],
            regularization_weights=reg_weights,
            regularization_matrices=reg_matrices,
        )

    def least_squares_solution(
        self, grid_values: np.ndarray, solver_type: str = "svd"
    ) -> np.ndarray:
        """Least squares decomposition of a scalar field."""
        if solver_type not in self._scalar_solvers:
            self._scalar_solvers[solver_type] = LeastSquaresSolver(
                solver=solver_type, tolerance=self.pinv_rtol
            )

        solver = self._scalar_solvers[solver_type]
        # RHS must be a list of inputs matching data terms in problem
        # Assuming single data term based on original code usage
        rhs = [grid_values]
        return solver.solve(problem=self.least_squares_problem, rhs=rhs)

    def least_squares_solution_helmholtz(
        self, grid_values: np.ndarray, solver_type: str = "svd"
    ) -> np.ndarray:
        """Least squares decomposition of a horizontal vector field."""
        if solver_type not in self._helmholtz_solvers:
            self._helmholtz_solvers[solver_type] = LeastSquaresSolver(
                solver=solver_type, tolerance=self.pinv_rtol
            )

        solver = self._helmholtz_solvers[solver_type]
        rhs = [grid_values]
        return solver.solve(problem=self.least_squares_problem_helmholtz, rhs=rhs)

    def basis_to_grid(
        self, coeffs: np.ndarray, derivative: Union[None, str] = None, helmholtz: bool = False
    ) -> np.ndarray:
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            return self.G_th.dot(coeffs)
        elif derivative == "phi":
            return self.G_ph.dot(coeffs)
        elif helmholtz:
            return np.tensordot(self.G_helmholtz, coeffs, 2)
        else:
            return self.G.dot(coeffs)

    def grid_to_basis(self, grid_values: np.ndarray, helmholtz: bool = False) -> np.ndarray:
        """Transform grid values to basis coefficients."""
        if helmholtz:
            return self.least_squares_solution_helmholtz(grid_values)
        else:
            return self.least_squares_solution(grid_values)

    def regularization_term(self, coeffs: np.ndarray, helmholtz: bool = False) -> np.ndarray:
        """Return the regularization term."""
        if helmholtz:
            if self.L_helmholtz is None:
                raise ValueError("Regularization not enabled (L_helmholtz is None)")
            return np.tensordot(self.L_helmholtz, coeffs, 2)
        else:
            if self.L is None:
                raise ValueError("Regularization not enabled (L is None)")
            return np.dot(coeffs, np.dot(self.L, coeffs))

    def scaled_G(self, factor: Union[float, np.ndarray]) -> np.ndarray:
        """Return the G matrix scaled by a factor.
        
        Handles:
        - Scalar scaling
        - Row scaling (factor matches grid points)
        - Column scaling (factor matches basis coefficients)
        """
        import scipy.sparse
        if np.isscalar(factor):
            return factor * self.G
        
        factor_arr = np.asarray(factor).ravel()
        rows, cols = self.G.shape
        
        is_sparse = scipy.sparse.issparse(self.G)
        
        # Check alignment
        matches_rows = factor_arr.size == rows
        matches_cols = factor_arr.size == cols
        
        if matches_rows and matches_cols and rows != cols:
             # This can't happen if rows != cols.
             pass
        elif matches_rows and matches_cols:

             # Square matrix ambiguity. PynaMIT context implies geometric row scaling.
             pass

        if matches_rows:
            # Row scaling
            if is_sparse:
                return scipy.sparse.diags(factor_arr) @ self.G
            else:
                return self.G * factor_arr.reshape(-1, 1)
        elif matches_cols:
            # Column scaling
            if is_sparse:
                 return self.G @ scipy.sparse.diags(factor_arr)
            else:
                 return self.G * factor_arr
        else:
             raise ValueError(
                 f"Factor size {factor_arr.size} does not match G shape {self.G.shape} "
                 "for either row or column scaling."
             )
