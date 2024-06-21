"""
Used to construct objects that transform between basis coefficients and a grid.
"""

import numpy as np

class BasisEvaluator(object):
    """
    Class for transforming between a given basis and a grid.

    """
    
    def __init__(self, basis, grid):
        self.basis = basis
        self.grid = grid

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
    def GTG(self):
        """
        Return the matrix ``G^T G``. The first time is called, it will
        also report the condition number of the matrix.

        """

        if not hasattr(self, '_GTG'):
            # Calculate GTG
            self._GTG = self.G.T.dot(self.G)

            # Report condition number for GTG
            cond_GTG = np.linalg.cond(self._GTG)
            print('The condition number for the GTG matrix is {:.1f}'.format(cond_GTG))

        return self._GTG

    @property
    def G_grad(self):
        """
        Return the matrix that transforms coefficients to the gradient of the
        grid values.

        """

        if not hasattr(self, '_G_grad'):
            self._G_grad = np.vstack((self.G_th, self.G_ph))

        return self._G_grad

    @property
    def G_rxgrad(self):
        """
        Return the matrix that transforms coefficients to the cross product of
        the gradient of the grid values.

        """

        if not hasattr(self, '_G_rxgrad'):
            self._G_rxgrad = np.vstack((-self.G_ph, self.G_th))

        return self._G_rxgrad

    @property
    def G_helmholtz(self):
        """
        Return the G matrix for the Helmholtz decomposition into the
        curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_G_helmholtz'):
            self._G_helmholtz = np.hstack((-self.G_grad, self.G_rxgrad))
        return self._G_helmholtz

    @property
    def G_helmholtz_inv(self):
        """
        Return the inverse of the G matrix for the Helmholtz decomposition
        into the curl-free and divergence-free components of a vector.

        """

        if not hasattr(self, '_G_helmholtz_inv'):
            self._G_helmholtz_inv = np.linalg.pinv(self.G_helmholtz)
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
            return np.split(np.dot(self.G_helmholtz, np.hstack(coeffs)), 2)
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
            return np.split(np.dot(self.G_helmholtz_inv, np.hstack(grid_values)), 2)
        else:
            return np.dot(self.G_inv, grid_values)
    
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