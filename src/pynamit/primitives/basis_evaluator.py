"""
Basis evaluator.

This module contains the BasisEvaluator class for transforming between expansions in a given basis and grid values.

Classes
-------
BasisEvaluator
    A class for transforming between expansions in a given basis and grid values.
"""

import numpy as np
from pynamit.math.least_squares import LeastSquares


class BasisEvaluator(object):
    """
    Object for transforming between expansions in a given basis and grid values. Can be used for both scalar and horizontal vector fields, where the latter is represented by basis expansion coefficients of the curl-free and divergence-free parts.

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
        """
        Initialize the BasisEvaluator object.

        Parameters
        ----------
        basis : Basis
            Basis object representing the basis of the field.
        grid : Grid
            Grid object representing the spatial grid.
        weights : array-like, optional
            Weights for the least squares solver. Default is None.
        reg_lambda : float, optional
            Regularization parameter for the least squares solver. Default is None.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse. Default is 1e-15.
        """
        self.basis = basis
        self.grid = grid
        self.weights = weights
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

    @property
    def G(self):
        """
        The matrix that corresponds to evaluating the basis expansion on the given grid.

        Returns
        -------
        array
            Matrix that evaluates the basis expansion on the grid.
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
        """
        The matrix that corresponds to evaluating the theta derivative of the basis expansion on the given grid.

        Returns
        -------
        array
            Matrix that evaluates the theta derivative of the basis expansion on the grid.
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
        """
        The matrix that corresponds to evaluating the phi derivative of the basis expansion on the given grid.

        Returns
        -------
        array
            Matrix that evaluates the phi derivative of the basis expansion on the grid.
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
        """
        The matrix that corresponds to evaluating the horizontal gradient of the basis expansion on the given grid.

        Returns
        -------
        array
            Matrix that evaluates the horizontal gradient of the basis expansion on the grid.
        """
        if not hasattr(self, "_G_grad"):
            self._G_grad = np.array([self.G_th, self.G_ph])

        return self._G_grad

    @property
    def G_rxgrad(self):
        """
        The matrix that corresponds to evaluating r cross the horizontal gradient of the basis expansion on the given grid.

        Returns
        -------
        array
            Matrix that evaluates r cross the horizontal gradient of the basis expansion on the grid.
        """
        if not hasattr(self, "_G_rxgrad"):
            self._G_rxgrad = np.array([-self.G_ph, self.G_th])

        return self._G_rxgrad

    @property
    def G_rxgrad_inv(self):
        """
        Return the inverse of the ``G_rxgrad`` matrix.

        Returns
        -------
        array
            Inverse of the ``G_rxgrad`` matrix.
        """
        if not hasattr(self, "_G_rxgrad_inv"):
            self._G_rxgrad_inv = np.linalg.pinv(self.G_rxgrad)
        return self._G_rxgrad_inv

    @property
    def G_helmholtz(self):
        """
        The matrix that corresponds to evaluating the Helmholtz decomposition of a horizontal vector field, represented by basis expansion coefficients of the curl-free and divergence-free parts, on the given grid.

        Returns
        -------
        array
            Matrix that evaluates the Helmholtz decomposition of a horizontal vector field on the grid.
        """
        if not hasattr(self, "_G_helmholtz"):
            self._G_helmholtz = np.stack([-self.G_grad, self.G_rxgrad], axis=2)
        return self._G_helmholtz

    @property
    def L(self):
        """
        The regularization matrix.

        Returns
        -------
        array
            Regularization matrix.
        """
        if not hasattr(self, "_L"):
            if self.reg_lambda is None:
                self._L = None
            else:
                self._L = np.diag(self.basis.n)

        return self._L

    @property
    def L_helmholtz(self):
        """
        The regularization matrix for the Helmholtz decomposition into basis expansion coefficients representing the curl-free and divergence-free parts of a vector field.

        Returns
        -------
        array
            Regularization matrix for the Helmholtz decomposition.
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
        """
        The least squares solver for finding the basis expansion coefficients of a scalar field.

        Returns
        -------
        LeastSquares
            Least squares solver for finding the basis expansion coefficients of a scalar field.
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
        """
        The least squares solver for finding the basis expansion coefficients of the curl-free and divergence-free parts of a vector field.

        Returns
        -------
        LeastSquares
            Least squares solver for finding the basis expansion coefficients of the curl-free and divergence-free parts of a vector field.
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
        """
        Return the least squares solution for the coefficients representing a scalar field in the given basis.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.

        Returns
        -------
        ndarray
            Coefficients in the basis.
        """
        return self.least_squares.solve(grid_values)[0]

    def least_squares_solution_helmholtz(self, grid_values):
        """
        Return the least squares solution for the two sets of coefficients representing the curl-free and divergence-free components of a horizontal vector field in the given basis.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.

        Returns
        -------
        ndarray
            Coefficients in the basis.
        """
        return self.least_squares_helmholtz.solve(grid_values)[0]

    def basis_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """
        Transform basis coefficients to grid values.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.
        derivative : str, optional
            Derivative to evaluate ('theta' or 'phi'). Default is None.
        helmholtz : bool, optional
            Whether to use the Helmholtz decomposition. Default is False.

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
        """
        Transform grid values to basis coefficients.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.
        helmholtz : bool, optional
            Whether to use the Helmholtz decomposition. Default is False.

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
        """
        Return the regularization term.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.
        helmholtz : bool, optional
            Whether to use the Helmholtz decomposition. Default is False.

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
        """
        Return the scaled G matrix. The factor can be a matrix with the same shape as G, in which case the scaling is done element-wise.

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
