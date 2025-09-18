"""Basis evaluator module.

This module contains the BasisEvaluator class for evaluating basis
expansions on a grid.
"""

import numpy as np

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver


class BasisEvaluator(object):
    """Object for evaluating basis expansions on a grid.

    This class provides methods for evaluating basis expansions on a
    grid and for constructing least-squares problems to find the basis
    expansion coefficients corresponding to given grid values.
    """

    def __init__(self, basis, grid, sqrt_weights=None, reg_lambda=None, pinv_rtol=1e-15):
        """Initialize the BasisEvaluator object."""
        self.basis = basis
        self.grid = grid
        self.sqrt_weights = sqrt_weights
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

        self._least_squares_problem = None
        self._least_squares_problem_helmholtz = None

        self._scalar_solvers = {}
        self._helmholtz_solvers = {}

    @property
    def G(self):
        """Evaluation matrix."""
        if not hasattr(self, "_G"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._G, self._cache = self.basis.get_G(self.grid, cache_out=True)
                else:
                    self._G = self.basis.get_G(self.grid, cache_in=self._cache)
            else:
                self._G = self.basis.get_G(self.grid)
        return self._G

    @property
    def G_th(self):
        """Matrix evaluating the theta derivative."""
        if not hasattr(self, "_G_th"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._G_th, self._cache = self.basis.get_G(
                        self.grid, derivative="theta", cache_out=True
                    )
                else:
                    self._G_th = self.basis.get_G(
                        self.grid, derivative="theta", cache_in=self._cache
                    )
            else:
                self._G_th = self.basis.get_G(self.grid, derivative="theta")
        return self._G_th

    @property
    def G_ph(self):
        """Matrix evaluating the phi derivative."""
        if not hasattr(self, "_G_ph"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._G_ph, self._cache = self.basis.get_G(
                        self.grid, derivative="phi", cache_out=True
                    )
                else:
                    self._G_ph = self.basis.get_G(
                        self.grid, derivative="phi", cache_in=self._cache
                    )
            else:
                self._G_ph = self.basis.get_G(self.grid, derivative="phi")
        return self._G_ph

    @property
    def G_grad(self):
        """Matrix evaluating the horizontal gradient."""
        if not hasattr(self, "_G_grad"):
            self._G_grad = np.array([self.G_th, self.G_ph])
        return self._G_grad

    @property
    def G_rxgrad(self):
        """Matrix evaluating r-hat x horizontal gradient."""
        if not hasattr(self, "_G_rxgrad"):
            self._G_rxgrad = np.array([-self.G_ph, self.G_th])
        return self._G_rxgrad

    @property
    def G_rxgrad_pinv(self):
        """Matrix evaluating r-hat x horizontal gradient pseudoinverse."""
        if not hasattr(self, "_G_rxgrad_pinv"):
            self._G_rxgrad_pinv = np.linalg.pinv(self.G_rxgrad)
        return self._G_rxgrad_pinv

    @property
    def G_helmholtz(self):
        """Matrix evaluating horizontal vector field expansions."""
        if not hasattr(self, "_G_helmholtz"):
            self._G_helmholtz = np.stack([-self.G_grad, self.G_rxgrad], axis=2)
        return self._G_helmholtz

    @property
    def L(self):
        """Regularization matrix for scalar fields."""
        if not hasattr(self, "_L"):
            if self.reg_lambda is None:
                self._L = None
            else:
                self._L = np.diag(self.basis.n)
        return self._L

    @property
    def L_helmholtz(self):
        """Regularization matrix for horizontal vector fields."""
        if not hasattr(self, "_L_helmholtz"):
            if self.reg_lambda is None:
                self._L_helmholtz = None
            else:
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
                self._L_helmholtz = np.array([L_cf, L_df])
        return self._L_helmholtz

    @property
    def least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        if self._least_squares_problem is None:
            self._least_squares_problem = LeastSquaresProblem(
                A=self.G,
                solution_shape=self.basis.index_length,
                data_shapes=self.grid.size,
                sqrt_weights=self.sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L,
            )
        return self._least_squares_problem

    @property
    def least_squares_problem_helmholtz(self) -> LeastSquaresProblem:
        """Least squares problem for horizontal vector fields."""
        if self._least_squares_problem_helmholtz is None:
            self._least_squares_problem_helmholtz = LeastSquaresProblem(
                A=self.G_helmholtz,
                solution_shape=(2, self.basis.index_length),
                data_shapes=(2, self.grid.size),
                sqrt_weights=self.sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L_helmholtz,
            )
        return self._least_squares_problem_helmholtz

    def least_squares_solution(self, grid_values, solver_type="svd"):
        """Least squares decomposition of a scalar field."""
        if solver_type not in self._scalar_solvers:
            solver = LeastSquaresSolver(solver=solver_type, tolerance=self.pinv_rtol)
            solver.update_problem(self.least_squares_problem)
            self._scalar_solvers[solver_type] = solver

        solver = self._scalar_solvers[solver_type]
        return solver.solve(grid_values)

    def least_squares_solution_helmholtz(self, grid_values, solver_type="svd"):
        """Least squares decomposition of a horizontal vector field."""
        if solver_type not in self._helmholtz_solvers:
            solver = LeastSquaresSolver(solver=solver_type, tolerance=self.pinv_rtol)
            solver.update_problem(self.least_squares_problem_helmholtz)
            self._helmholtz_solvers[solver_type] = solver

        solver = self._helmholtz_solvers[solver_type]
        return solver.solve(grid_values)

    def basis_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            return np.dot(self.G_th, coeffs)
        elif derivative == "phi":
            return np.dot(self.G_ph, coeffs)
        elif helmholtz:
            return np.tensotensor(self.G_helmholtz, coeffs, 2)
        else:
            return np.dot(self.G, coeffs)

    def grid_to_basis(self, grid_values, helmholtz=False):
        """Transform grid values to basis coefficients."""
        if helmholtz:
            return self.least_squares_solution_helmholtz(grid_values)
        else:
            return self.least_squares_solution(grid_values)

    def regularization_term(self, coeffs, helmholtz=False):
        """Return the regularization term."""
        if helmholtz:
            return np.tensotensor(self.L_helmholtz, coeffs, 2)
        else:
            return np.dot(coeffs, np.dot(self.L, coeffs))

    def scaled_G(self, factor):
        """Return the scaled G matrix."""
        return factor * self.G
