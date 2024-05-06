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
            self._G = self.basis.get_G(self.grid)

        return self._G
    
    @property
    def Ginv(self):
        """
        Return matrix that transforms grid to coefficients.

        """

        if not hasattr(self, '_Ginv'):
            self._Ginv = np.linalg.pinv(self.G)

        return self._Ginv

    @property
    def G_th(self):
        """
        Return matrix that differentiates coefficients with respect to
        theta and transforms to grid values.

        """

        if not hasattr(self, '_G_th'):
            self._G_th = self.basis.get_G(self.grid, derivative = 'theta')

        return self._G_th

    @property
    def G_ph(self):
        """
        Return matrix that differentiates coefficients with respect to phi
        and transforms to grid values.

        """

        if not hasattr(self, '_G_ph'):
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
            print('The condition number for the surface SH matrix is {:.1f}'.format(cond_GTG))

        return self._GTG

    @property
    def Gcf(self):
        """
        Return the G matrix for the curl-free components of a vector.

        """

        if not hasattr(self, '_Gcf'):
            self._Gcf = np.vstack((-self.G_th, -self.G_ph))
        return self._Gcf

    @property
    def vector_to_shc_cf(self):
        """
        Return matrix for obtaining SH coefficients corresponding to the
        curl-free components of a vector.

        """

        if not hasattr(self, '_vector_to_shc_cf'):
            self.GTGcf_inv = np.linalg.pinv(self.Gcf.T.dot(self.Gcf))
            self._vector_to_shc_cf = self.GTGcf_inv.dot(self.Gcf.T)
        return self._vector_to_shc_cf

    @property
    def Gdf(self):
        """
        Return the G matrix for the divergence-free components of a
        vector.

        """

        if not hasattr(self, '_Gdf'):
            self._Gdf = np.vstack((-self.G_ph, self.G_th))
        return self._Gdf

    @property
    def vector_to_shc_df(self):
        """
        Return matrix for obtaining SH coefficients corresponding to the
        divergence-free components of a vector.

        """

        if not hasattr(self, '_vector_to_shc_df'):
            self.GTGdf_inv = np.linalg.pinv(self.Gdf.T.dot(self.Gdf))
            self._vector_to_shc_df = self.GTGdf_inv.dot(self.Gdf.T)

        return self._vector_to_shc_df

    def to_grid(self, coeffs, derivative = None):
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
        if derivative == 'theta':
            return np.dot(self.G_th, coeffs)
        elif derivative == 'phi':
            return np.dot(self.G_ph, coeffs)
        else:
            return np.dot(self.G, coeffs)

    def from_grid(self, grid_values, component = None):
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
        if component == 'cf':
            return np.dot(self.vector_to_shc_cf, grid_values)
        elif component == 'df':
            return np.dot(self.vector_to_shc_df, grid_values)
        else:
            return np.dot(self.G.T, grid_values)
    
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