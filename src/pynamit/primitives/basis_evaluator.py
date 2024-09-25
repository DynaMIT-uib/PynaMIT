"""
Used to construct objects that transform between basis coefficients and a grid.
"""

import numpy as np
from pynamit.various.math import pinv_positive_semidefinite, tensor_pinv, tensor_pinv_positive_semidefinite, tensor_scale_right

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
    def GTWG(self):
        """
        Return the matrix ``G^T W G``.

        """

        if not hasattr(self, '_GTWG'):
            self._GTWG = np.dot(self.GTW, self.G)

        return self._GTWG

    @property
    def GTW_helmholtz(self):
        """
        Return the matrix ``G^T W`` for the Helmholtz decomposition into the
        curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_GTW_helmholtz'):
            if self.weights is not None:
                self._GTW_helmholtz = tensor_scale_right(self.G_helmholtz.T, self.weights)
            else:
                self._GTW_helmholtz = self.G_helmholtz.T

        return self._GTW_helmholtz

    @property
    def GTWG_helmholtz(self):
        """
        Return the matrix ``G^T W G`` for the Helmholtz decomposition into the
        curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_GTWG_helmholtz'):
            self._GTWG_helmholtz = np.einsum('ijkkmn->ijmn', np.tensordot(self.GTW_helmholtz, self.G_helmholtz, 1), optimize = True)

        return self._GTWG_helmholtz
    
    @property
    def G_inv(self):
        """
        Return matrix that transforms grid to coefficients.

        """

        if not hasattr(self, '_G_inv'):
            self._G_inv = np.linalg.pinv(self.G)

        return self._G_inv

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
    def GTG_inv(self):
        """
        Return the inverse of the matrix ``G^T G``.

        """

        if not hasattr(self, '_GTG_inv'):
            self._GTG_inv = pinv_positive_semidefinite(np.dot(self.G.T, self.G), rtol = self.pinv_rtol)

        return self._GTG_inv

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
            self._GTWG_helmholtz_inv = tensor_pinv(self.GTWG_helmholtz, contracted_dims = 2, rtol = self.pinv_rtol)
        return self._GTWG_helmholtz_inv

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
            self._G_helmholtz = np.moveaxis(np.array([-self.G_grad, self.G_rxgrad]), [0,1], [1,0])

        return self._G_helmholtz

    @property
    def GTG_helmholtz_inv(self):
        """
        Return the inverse of the matrix ``G^T G`` for the Helmholtz decomposition
        into the curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_GTG_helmholtz_inv'):
            GTG_helmholtz = np.einsum('ijkl,ijmn->klmn', self.G_helmholtz, self.G_helmholtz, optimize = True)
            self._GTG_helmholtz_inv = tensor_pinv_positive_semidefinite(GTG_helmholtz, 2, rtol = self.pinv_rtol)

        return self._GTG_helmholtz_inv

    @property
    def G_helmholtz_inv(self):
        """
        Return the inverse of the G matrix for the Helmholtz decomposition
        into the curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_G_helmholtz_inv'):
            self._G_helmholtz_inv = tensor_pinv(self.G_helmholtz, contracted_dims = 2, rtol = self.pinv_rtol)

        return self._G_helmholtz_inv

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
            return np.dot(self.G_helmholtz, coeffs)
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
            if self.reg_lambda is not None:
                #reg_L = np.diag(self.basis.laplacian(), 2)
                reg_L = np.hstack((self.basis.n * (self.basis.n + 1) / (2 * self.basis.n + 1), self.basis.n + 1))
                return np.linalg.lstsq(self.GTWG_helmholtz + self.reg_lambda * np.diag(reg_L), np.dot(self.GTW_helmholtz, grid_values), rcond = self.pinv_rtol)[0]
            else:
                intermediate = np.einsum('ijkk->ij', np.tensordot(self.GTW_helmholtz, np.moveaxis(np.array(np.split(grid_values, 2)), 0, 1), 1), optimize = True)
                return np.moveaxis(np.einsum('ijkk->ij', np.tensordot(self.GTWG_helmholtz_inv, intermediate, 1), optimize = True), 0, 1)
        else:
            if self.reg_lambda is not None:
                reg_L = np.diag(np.ones(self.basis.index_length))
                return np.linalg.lstsq(self.GTWG + self.reg_lambda * np.diag(reg_L), np.dot(self.GTW, grid_values), rcond = self.pinv_rtol)[0]
            else:
                return np.dot(self.GTWG_inv, np.dot(self.GTW, grid_values))
    
    def to_other_basis(self, this_coeffs, other_coeffs):
        """
        Transform coefficients in the basis to another basis.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.

        Returns
        -------
        ndarray
            Coefficients in the new basis.

        """

        other_coeffs.values = other_coeffs.basis.grid_to_basis(self.basis_to_grid(this_coeffs))

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