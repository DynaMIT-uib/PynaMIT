"""Basis evaluator module.

This module contains the BasisEvaluator class for evaluating basis
expansions on a grid.
"""

import numpy as np
from pynamit.math.least_squares import LeastSquares


class BasisEvaluator(object):
    """Object for evaluating basis expansions on a grid.

    This class provides methods for evaluating basis expansions on a
    grid and also for solving least squares problems to find the basis
    expansion coefficients corresponding to given grid values. The class
    can be used for both scalar and horizontal vector fields, where the
    latter is represented by basis expansion coefficients of the
    curl-free and divergence-free parts (Helmholtz decomposition).

    Attributes
    ----------
    basis : Basis
        Basis object representing the basis of the field.
    grid : Grid
        Grid object representing the spatial grid.
    weights : array-like, optional
        Weights for the least squares solver.
    reg_lambda : float, optional
        Regularization parameter for the least squares solver.
    pinv_rtol : float, optional
        Relative tolerance for the pseudo-inverse.
    """

    def __init__(self, basis, grid, weights=None, reg_lambda=None, pinv_rtol=1e-15):
        """Initialize the BasisEvaluator object.

        Parameters
        ----------
        basis : Basis
            Basis object representing the basis of the field.
        grid : Grid
            Grid object representing the spatial grid.
        weights : array-like, optional
            Weights for the least squares solver.
        reg_lambda : float, optional
            Regularization parameter for the least squares solver.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self.basis = basis
        self.grid = grid
        self.weights = weights
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

    @property
    def G(self):
        """Evaluation matrix.

        Returns
        -------
        array
            Matrix that evaluates a basis expansion on the grid.
        """
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
        """Matrix evaluating the theta derivative.

        Returns
        -------
        array
            Matrix that evaluates the theta derivative of a basis
            expansion on the grid.
        """
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
        """Matrix evaluating the phi derivative.

        Returns
        -------
        array
            Matrix that evaluates the phi derivative of a basis
            expansion on the grid.
        """
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
        """Matrix evaluating the horizontal gradient.

        Returns
        -------
        array
            Matrix that evaluates the horizontal gradient of a basis
            expansion on the grid.

        """
        if not hasattr(self, "_G_grad"):
            self._G_grad = np.array([self.G_th, self.G_ph])

        return self._G_grad

    @property
    def G_rxgrad(self):
        """Matrix evaluating r-hat x horizontal gradient.

        Returns
        -------
        array
            Matrix that evaluates the radial unit vector cross the
            horizontal gradient of a basis expansion on the grid.
        """
        if not hasattr(self, "_G_rxgrad"):
            self._G_rxgrad = np.array([-self.G_ph, self.G_th])

        return self._G_rxgrad

    @property
    def G_rxgrad_pinv(self):
        """Matrix evaluating r-hat x horizontal gradient pseudoinverse.

        Returns
        -------
        array
            Matrix that is the pseudoinverse of the matrix that
            evaluates the radial unit vector cross the horizontal
            gradient of a basis expansion on the grid.
        """
        if not hasattr(self, "_G_rxgrad_pinv"):
            self._G_rxgrad_pinv = np.linalg.pinv(self.G_rxgrad)
        return self._G_rxgrad_pinv

    @property
    def G_helmholtz(self):
        """Matrix evaluating horizontal vector field expansions.

        Returns
        -------
        array
            Matrix that evaluates the expansions representing the
            curl-free and divergence-free parts of a horizontal vector
            field on the grid.
        """
        if not hasattr(self, "_G_helmholtz"):
            self._G_helmholtz = np.stack([-self.G_grad, self.G_rxgrad], axis=2)
        return self._G_helmholtz

    @property
    def L(self):
        """Regularization matrix for scalar fields.

        Returns
        -------
        array
            Regularization matrix for the least squares problem of
            finding the basis expansion coefficients representing a
            scalar field.
        """
        if not hasattr(self, "_L"):
            if self.reg_lambda is None:
                self._L = None
            else:
                self._L = np.diag(self.basis.n)

        return self._L

    @property
    def L_helmholtz(self):
        """Regularization matrix for horizontal vector fields.

        Returns
        -------
        array
            Regularization matrix for the least squares problem of
            finding the basis expansion coefficients representing the
            curl-free and divergence-free parts of a horizontal vector
            field.
        """
        if not hasattr(self, "_L_helmholtz"):
            if self.reg_lambda is None:
                self._L_helmholtz = None
            else:
                L_cf = np.stack(
                    [
                        np.diag(
                            self.basis.n * (self.basis.n + 1) / (2 * self.basis.n + 1)
                        ),
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
    def least_squares(self):
        """Least squares solver for scalar fields.

        Returns
        -------
        LeastSquares
            Least squares solver for finding the basis expansion
            coefficients of a scalar field.
        """
        if not hasattr(self, "_least_squares"):
            self._least_squares = LeastSquares(
                self.G,
                1,
                weights=self.weights,
                reg_lambda=self.reg_lambda,
                reg_L=self.L,
                pinv_rtol=self.pinv_rtol,
            )

        return self._least_squares

    @property
    def least_squares_helmholtz(self):
        """Least squares solver for horizontal vector fields.

        Returns
        -------
        LeastSquares
            Least squares solver for finding the basis expansion
            coefficients of the curl-free and divergence-free parts of
            a horizontal vector field.
        """
        if not hasattr(self, "_least_squares_helmholtz"):
            self._least_squares_helmholtz = LeastSquares(
                self.G_helmholtz,
                2,
                weights=self.weights,
                reg_lambda=self.reg_lambda,
                reg_L=self.L_helmholtz,
                pinv_rtol=self.pinv_rtol,
            )

        return self._least_squares_helmholtz

    def least_squares_solution(self, grid_values):
        """Least squares decomposition of a scalar field.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.

        Returns
        -------
        ndarray
            Least squares solution for the coefficients representing a
            scalar field in the basis.
        """
        return self.least_squares.solve(grid_values)[0]

    def least_squares_solution_helmholtz(self, grid_values):
        """Least squares decomposition of a horizontal vector field.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.

        Returns
        -------
        ndarray
            Least-squares solution for the coefficients representing the
            curl-free and divergence-free components of a horizontal
            vector field in the basis.
        """
        return self.least_squares_helmholtz.solve(grid_values)[0]

    def basis_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.
        derivative : str, optional
            Derivative to evaluate ('theta' or 'phi').
        helmholtz : bool, optional
            Whether to use the Helmholtz decomposition.

        Returns
        -------
        ndarray
            Values at the grid points.
        """
        if derivative == "theta":
            return np.dot(self.G_th, coeffs)
        elif derivative == "phi":
            return np.dot(self.G_ph, coeffs)
        elif helmholtz:
            return np.tensordot(self.G_helmholtz, coeffs, 2)
        else:
            return np.dot(self.G, coeffs)

    def grid_to_basis(self, grid_values, helmholtz=False):
        """Transform grid values to basis coefficients.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.
        helmholtz : bool, optional
            Whether to use the Helmholtz decomposition.

        Returns
        -------
        ndarray
            Coefficients in the basis.
        """
        if helmholtz:
            return self.least_squares_solution_helmholtz(grid_values)

        else:
            return self.least_squares_solution(grid_values)

    def regularization_term(self, coeffs, helmholtz=False):
        """Return the regularization term.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.
        helmholtz : bool, optional
            Whether to use the Helmholtz decomposition.

        Returns
        -------
        float
            Regularization term.
        """
        if helmholtz:
            return np.tensordot(self.L_helmholtz, coeffs, 2)

        else:
            return np.dot(coeffs, np.dot(self.L, coeffs))

    def scaled_G(self, factor):
        """Return the scaled G matrix.

        The factor can be a matrix with the same shape as G, in which
        case the scaling is done element-wise.

        Parameters
        ----------
        factor : float
            The scaling factor.

        Returns
        -------
        ndarray
            The scaled G matrix.
        """
        return factor * self.G
