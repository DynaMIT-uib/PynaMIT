import numpy as np

class grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, RI, lat, lon, sha = None):
        """ Initialize the grid for the ionosphere.

        """

        self.RI = RI
        self.lat = lat
        self.lon = lon

        self.theta = 90 - lat

        if sha is not None:
            self.sha = sha

    @property
    def G(self):
        """
        Return the ``G`` matrix, which is the matrix that transforms a
        vector of spherical harmonic coefficients into the corresponding
        grid vector.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for G to be available.')

        if not hasattr(self, '_G'):
            self._G = self.sha.get_G(self)
        return self._G

    @property
    def GTG(self):
        """
        Return the matrix ``G^T G``. The first time is called, it will
        also report the condition number of the matrix.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for GTG to be available.')
        if not hasattr(self, '_GTG'):
            # Calculate GTG
            self._GTG = self.G.T.dot(self.G)

            # Report condition number for GTG
            cond_GTG = np.linalg.cond(self._GTG)
            print('The condition number for the surface SH matrix is {:.1f}'.format(cond_GTG))

        return self._GTG

    @property
    def G_ph(self):
        """
        Return the G matrix differentiated with respect to ``phi``.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for G_ph to be available.')
        if not hasattr(self, '_G_ph'):
            self._G_ph = self.sha.get_G(self, derivative = 'phi')
        return self._G_ph

    @property
    def G_th(self):
        """
        Return the G matrix differentiated with respect to ``theta``.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for G_th to be available.')
        if not hasattr(self, '_G_th'):
            self._G_th = self.sha.get_G(self, derivative = 'theta')
        return self._G_th

    @property
    def Gcf(self):
        """
        Return the G matrix for the curl-free components of a vector.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for Gcf to be available.')
        if not hasattr(self, '_Gcf'):
            self._Gcf = np.vstack((-self.G_th, -self.G_ph))
        return self._Gcf

    @property
    def vector_to_shc_cf(self):
        """
        Return matrix for obtaining SH coefficients corresponding to the
        curl-free components of a vector.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for vector_to_shc_cf to be available.')
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

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for Gdf to be available.')
        if not hasattr(self, '_Gdf'):
            self._Gdf = np.vstack((-self.G_ph, self.G_th))
        return self._Gdf

    @property
    def vector_to_shc_df(self):
        """
        Return matrix for obtaining SH coefficients corresponding to the
        divergence-free components of a vector.

        """

        if not hasattr(self, 'sha'):
            raise AttributeError('Must pass a sha object during grid initialization for vector_to_shc_df to be available.')
        if not hasattr(self, '_vector_to_shc_df'):
            self.GTGdf_inv = np.linalg.pinv(self.Gdf.T.dot(self.Gdf))
            self._vector_to_shc_df = self.GTGdf_inv.dot(self.Gdf.T)

        return self._vector_to_shc_df
