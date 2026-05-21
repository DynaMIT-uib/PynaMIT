"""Basis evaluator module.

This module contains the BasisEvaluator class for evaluating basis
expansions on a grid.
"""

import numpy as np

from pynamit.math.backend import get_array_module
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver, get_default_least_squares_solver
from pynamit.sphere.core import SurfaceOperators


def grid_sqrt_area_weights(grid):
    """Return default sqrt area weights for a spherical grid."""
    if hasattr(grid, "area_weights"):
        xp = get_array_module(grid.area_weights)
        weights = xp.asarray(grid.area_weights, dtype=float)
    else:
        xp = get_array_module(grid.theta)
        theta = xp.asarray(grid.theta, dtype=float)
        weights = xp.sin(xp.deg2rad(theta))
    weights = xp.clip(weights, 0.0, None)
    return xp.sqrt(weights)


def resolve_sqrt_weights(grid, sqrt_weights=None, area_weighted=False, vector=False):
    """Resolve explicit or default grid sqrt weights."""
    if sqrt_weights is not None:
        return sqrt_weights
    if not area_weighted:
        return None
    weights = grid_sqrt_area_weights(grid)
    xp = get_array_module(weights)
    return xp.tile(weights, (2, 1)) if vector else weights


class BasisEvaluator(object):
    """Object for evaluating basis expansions on a grid.

    This class provides methods for evaluating basis expansions on a
    grid and for constructing least-squares problems to find the basis
    expansion coefficients corresponding to given grid values.
    """

    def __init__(
        self,
        basis,
        grid,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        area_weighted=False,
    ):
        """Initialize the BasisEvaluator object."""
        if not isinstance(basis, SurfaceOperators):
            raise TypeError("BasisEvaluator requires a basis implementing SurfaceOperators.")
        basis.validate_metadata()
        self.basis = basis
        self.grid = grid
        self.explicit_sqrt_weights = sqrt_weights is not None
        self.area_weighted = bool(area_weighted)
        self.sqrt_weights = resolve_sqrt_weights(
            grid, sqrt_weights=sqrt_weights, area_weighted=area_weighted
        )
        self.helmholtz_sqrt_weights = resolve_sqrt_weights(
            grid,
            sqrt_weights=sqrt_weights,
            area_weighted=area_weighted,
            vector=True,
        )
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

        self._least_squares_problem = None
        self._least_squares_problem_helmholtz = None

    @property
    def G(self):
        """Evaluation matrix."""
        if not hasattr(self, "_G"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._G, self._cache = self.basis.evaluate_on_grid(
                        self.grid, cache_out=True
                    )
                else:
                    self._G = self.basis.evaluate_on_grid(
                        self.grid, cache_in=self._cache
                    )
            else:
                self._G = self.basis.get_scalar_evaluation_matrix(self.grid)
        return self._G

    @property
    def G_th(self):
        """Matrix evaluating the theta derivative."""
        if not hasattr(self, "_G_th"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._G_th, self._cache = self.basis.evaluate_on_grid(
                        self.grid, derivative="theta", cache_out=True
                    )
                else:
                    self._G_th = self.basis.evaluate_on_grid(
                        self.grid, derivative="theta", cache_in=self._cache
                    )
            else:
                self._G_th = self.basis.get_surface_gradient_matrix(self.grid)[0]
        return self._G_th

    @property
    def G_ph(self):
        """Matrix evaluating the phi derivative."""
        if not hasattr(self, "_G_ph"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._G_ph, self._cache = self.basis.evaluate_on_grid(
                        self.grid, derivative="phi", cache_out=True
                    )
                else:
                    self._G_ph = self.basis.evaluate_on_grid(
                        self.grid, derivative="phi", cache_in=self._cache
                    )
            else:
                self._G_ph = self.basis.get_surface_gradient_matrix(self.grid)[1]
        return self._G_ph

    @property
    def G_grad(self):
        """Matrix evaluating the horizontal gradient."""
        if not hasattr(self, "_G_grad"):
            self._G_grad = self.basis.get_surface_gradient_matrix(self.grid)
        return self._G_grad

    @property
    def G_rxgrad(self):
        """Matrix evaluating r-hat x horizontal gradient."""
        if not hasattr(self, "_G_rxgrad"):
            self._G_rxgrad = self.basis.get_rhat_cross_gradient_matrix(self.grid)
        return self._G_rxgrad

    @property
    def G_helmholtz(self):
        """Matrix evaluating horizontal vector field expansions."""
        if not hasattr(self, "_G_helmholtz"):
            self._G_helmholtz = self.basis.get_helmholtz_synthesis_matrix(self.grid)
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
                sqrt_weights=self.helmholtz_sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L_helmholtz,
            )
        return self._least_squares_problem_helmholtz

    def least_squares_solution(self, grid_values, solver_type=None):
        """Least squares decomposition of a scalar field."""
        return self._solve_least_squares(self.least_squares_problem, grid_values, solver_type)

    def least_squares_solution_helmholtz(self, grid_values, solver_type=None):
        """Least squares decomposition of a horizontal vector field."""
        solution = self._solve_least_squares(
            self.least_squares_problem_helmholtz, grid_values, solver_type
        )
        projector = getattr(self.basis, "project_helmholtz_mean_free", None)
        return projector(solution) if callable(projector) else solution

    def _solve_least_squares(self, problem, grid_values, solver_type=None):
        """Solve one configured least-squares problem."""
        solver_type = solver_type or get_default_least_squares_solver()
        solver = LeastSquaresSolver(solver=solver_type, tolerance=self.pinv_rtol)
        return solver.solve(problem=problem, rhs=grid_values)

    def basis_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            return np.dot(self.G_th, coeffs)
        elif derivative == "phi":
            return np.dot(self.G_ph, coeffs)
        elif helmholtz:
            return np.tensordot(self.G_helmholtz, coeffs, 2)
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
            return np.tensordot(self.L_helmholtz, coeffs, 2)
        else:
            return np.dot(coeffs, np.dot(self.L, coeffs))

    def contract_G(self, operator):
        """Return G contracted with a coefficient vector or matrix."""
        operator = np.asarray(operator)
        if operator.ndim == 1:
            return self.G * operator.reshape((1, -1))
        if operator.ndim == 2:
            return self.G @ operator
        raise ValueError("operator must be a vector or matrix.")
