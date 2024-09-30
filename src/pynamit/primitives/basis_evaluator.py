"""
Used to construct objects that transform between basis coefficients and a grid.
"""

import numpy as np
from pynamit.various.math import pinv_positive_semidefinite, tensor_pinv, tensor_pinv_positive_semidefinite, tensor_scale_left, tensor_transpose

class BasisEvaluator(object):
    """
    Class for transforming between a given basis and a grid.

    """
    
    def __init__(self, basis, grid, pinv_rtol = 1e-15, weights = None, reg_lambda = None):
        self.basis = basis
        self.grid = grid
        self.pinv_rtol = pinv_rtol
        self.weights = weights
        self.reg_lambda = reg_lambda

    @property
    def G(self):
        """
        Return matrix that transforms coefficients to grid.

        """

        if not hasattr(self, '_G'):
            if self.basis.caching:
                if not hasattr(self, '_cache'):
                    self._G, self._cache = self.basis.get_G(self.grid, cache_out = True)
                else:
                    self._G = self.basis.get_G(self.grid, cache_in = self._cache)
            else:
                self._G = self.basis.get_G(self.grid)

        return self._G

    @property
    def G_th(self):
        """
        Return matrix that differentiates coefficients with respect to
        theta and transforms to grid values.

        """

        if not hasattr(self, '_G_th'):
            if self.basis.caching:
                if not hasattr(self, '_cache'):
                    self._G_th, self._cache = self.basis.get_G(self.grid, derivative = 'theta', cache_out = True)
                else:
                    self._G_th = self.basis.get_G(self.grid, derivative = 'theta', cache_in = self._cache)
            else:
                self._G_th = self.basis.get_G(self.grid, derivative = 'theta')

        return self._G_th

    @property
    def G_ph(self):
        """
        Return matrix that differentiates coefficients with respect to phi
        and transforms to grid values.

        """

        if not hasattr(self, '_G_ph'):
            if self.basis.caching:
                if not hasattr(self, '_cache'):
                    self._G_ph, self._cache = self.basis.get_G(self.grid, derivative = 'phi', cache_out = True)
                else:
                    self._G_ph = self.basis.get_G(self.grid, derivative = 'phi', cache_in = self._cache)
            else:
                self._G_ph = self.basis.get_G(self.grid, derivative = 'phi')

        return self._G_ph

    @property
    def G_grad(self):
        """
        Return the matrix that transforms coefficients to the gradient of the
        grid values.

        """

        if not hasattr(self, '_G_grad'):
            self._G_grad = np.moveaxis(np.array([self.G_th, self.G_ph]), 0, 1)

        return self._G_grad

    @property
    def G_rxgrad(self):
        """
        Return the matrix that transforms coefficients to the cross product of
        the gradient of the grid values.

        """

        if not hasattr(self, '_G_rxgrad'):
            self._G_rxgrad = np.moveaxis(np.array([-self.G_ph, self.G_th]), 0, 1)

        return self._G_rxgrad

    @property
    def G_rxgrad_inv(self):
        """
        Return the inverse of the G matrix that transforms coefficients to the cross 
        product of the gradient of the grid values.

        """

        if not hasattr(self, '_G_rxgrad_inv'):
            self._G_rxgrad_inv = np.linalg.pinv(self.G_rxgrad)
        return self._G_rxgrad_inv

    @property
    def G_helmholtz(self):
        """
        Return the G matrix for the Helmholtz decomposition into the
        curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_G_helmholtz'):
            self._G_helmholtz = np.moveaxis(np.array([-self.G_grad, self.G_rxgrad]), [0,1,2,3], [1,0,3,2])

        return self._G_helmholtz

    @property
    def GTW(self):
        """
        Return the matrix ``G^T W``.

        """

        if not hasattr(self, '_GTW'):
            if self.weights is not None:
                self._GTW = (self.G * self.weights.reshape(-1, 1)).T
            else:
                self._GTW = self.G.T

        return self._GTW

    @property
    def GTW_helmholtz(self):
        """
        Return the matrix ``G^T W`` for the Helmholtz decomposition into the
        curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_GTW_helmholtz'):
            if self.weights is not None:
                self._GTW_helmholtz = tensor_transpose(tensor_scale_left(self.weights, self.G_helmholtz), 2)
            else:
                self._GTW_helmholtz = tensor_transpose(self.G_helmholtz, 2)

        return self._GTW_helmholtz

    @property
    def GTWG(self):
        """
        Return the matrix ``G^T W G``.

        """

        if not hasattr(self, '_GTWG'):
            self._GTWG = np.dot(self.GTW, self.G)

        return self._GTWG

    @property
    def GTWG_helmholtz(self):
        """
        Return the matrix ``G^T W G`` for the Helmholtz decomposition into the
        curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_GTWG_helmholtz'):
            self._GTWG_helmholtz = np.tensordot(self.GTW_helmholtz, self.G_helmholtz, 2)

        return self._GTWG_helmholtz

    @property
    def GTWG_inv(self):
        """
        Return the inverse of the matrix ``G^T W G``.

        """

        if not hasattr(self, '_GTWG_inv'):
            self._GTWG_inv = pinv_positive_semidefinite(self.GTWG, rtol = self.pinv_rtol)

        return self._GTWG_inv

    @property
    def GTWG_helmholtz_inv(self):
        """
        Return the inverse of the matrix ``G^T W G`` for the Helmholtz decomposition
        into the curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_GTWG_helmholtz_inv'):
            self._GTWG_helmholtz_inv = tensor_pinv_positive_semidefinite(self.GTWG_helmholtz, contracted_dims = 2, rtol = self.pinv_rtol)
        return self._GTWG_helmholtz_inv


    def basis_to_grid(self, coeffs, derivative = None, helmholtz = False):
        """
        Transform basis coefficients to grid values.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.

        Returns
        -------
        ndarray
            Values at the grid points.

        """

        if derivative == 'theta':
            return np.dot(self.G_th, coeffs)
        elif derivative == 'phi':
            return np.dot(self.G_ph, coeffs)
        elif helmholtz:
            return np.moveaxis(np.tensordot(self.G_helmholtz, np.moveaxis(coeffs, 0, 1), 2), 0, 1)
        else:
            return np.dot(self.G, coeffs)

    def grid_to_basis(self, grid_values, helmholtz = False):
        """
        Transform grid values to basis coefficients.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.

        Returns
        -------
        ndarray
            Coefficients in the basis.

        """

        if helmholtz:
            intermediate = np.tensordot(self.GTW_helmholtz, np.moveaxis(np.array(np.split(grid_values, 2)), 0, 1), 2)

            if self.reg_lambda is not None:
                regularization = self.reg_lambda * np.moveaxis(
                    np.array([[np.diag(self.basis.n * (self.basis.n + 1) / (2 * self.basis.n + 1)), np.zeros((self.basis.index_length, self.basis.index_length))],
                              [np.zeros((self.basis.index_length, self.basis.index_length)),        np.diag(self.basis.n + 1)]]
                ), [0,1,2,3], [1,3,0,2])

                GTWG_helmholtz_plus_regularization = self.GTWG_helmholtz + regularization

                first_dims = GTWG_helmholtz_plus_regularization.shape[:2]
                last_dims  = GTWG_helmholtz_plus_regularization.shape[2:]

                coeffs = np.linalg.lstsq(
                    GTWG_helmholtz_plus_regularization.reshape((np.prod(first_dims), np.prod(last_dims))),
                    intermediate.reshape((np.prod(last_dims))),
                    rcond = self.pinv_rtol
                )[0].reshape((first_dims))

            else:
                coeffs = np.tensordot(self.GTWG_helmholtz_inv, intermediate, 2)

            return np.moveaxis(coeffs, 0, 1)

        else:
            if self.reg_lambda is not None:
                regularization = self.reg_lambda * np.diag(np.ones(self.basis.index_length))
                return np.linalg.lstsq(self.GTWG + regularization, np.dot(self.GTW, grid_values), rcond = self.pinv_rtol)[0]

            else:
                return np.dot(self.GTWG_inv, np.dot(self.GTW, grid_values))

    def scaled_G(self, factor):
        """
        Return the scaled G matrix. The factor can be a matrix with the
        same shape as G, in which case the scaling is done element-wise.

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