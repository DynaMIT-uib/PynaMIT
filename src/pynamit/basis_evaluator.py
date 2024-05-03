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
        Matrix that transforms coefficients to grid.

        """

        if not hasattr(self, '_G'):
            self._G = self.basis.get_G(self.grid)

        return self._G
    
    @property
    def Ginv(self):
        """
        Matrix that transforms grid to coefficients.

        """

        if not hasattr(self, '_Ginv'):
            self._Ginv = np.linalg.pinv(self.G)

        return self._Ginv

    def to_grid(self, coeffs):
        """
        Transform coefficients to grid.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients in the basis.

        Returns
        -------
        ndarray
            Data on the grid.

        """
        return np.dot(self.G, coeffs)

    def from_grid(self, grid_values):
        """
        Transform data on the grid to coefficients in the basis.

        Parameters
        ----------
        grid_values : ndarray
            Values at the grid points.

        Returns
        -------
        ndarray
            Coefficients in the basis.

        """
        return np.dot(self.Ginv, grid_values)
    
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

        other_coeffs.values = other_coeffs.basis.from_grid(self.to_grid(this_coeffs))