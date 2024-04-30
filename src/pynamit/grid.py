import numpy as np

class grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, RI, lat, lon):
        """ Initialize the grid for the ionosphere.

        """

        self.RI = RI
        self.lat = lat
        self.lon = lon

        self.theta = 90 - lat

    def construct_G(self, sha):
        """ Construct the G matrix.

        """

        self.G = sha.get_G(self)

    def construct_dG(self, sha):
        """ Construct the G_ph and G_th matrices.

        """

        self.G_ph = sha.get_G(self, derivative = 'phi'  )
        self.G_th = sha.get_G(self, derivative = 'theta')

    def construct_GTG(self):
        """
        Construct the matrix ``G^T G`` and report its condition number.

        """

        # Pre-calculate GTG and its inverse
        self.GTG = self.G.T.dot(self.G)
        #self.GTG_inv = np.linalg.pinv(self.GTG)

        # Report condition number for GTG
        cond_GTG = np.linalg.cond(self.GTG)
        print('The condition number for the surface SH matrix is {:.1f}'.format(cond_GTG))


    def construct_vector_to_shc_cf_df(self):
        """
        Construct matrices for obtaining SH coefficients corresponding to
        curl- and divergence-free components of vector.

        """

        # Pre-calculate matrix to get coefficients for curl-free fields:
        self.Gcf = np.vstack((-self.G_th, -self.G_ph))
        self.GTGcf_inv = np.linalg.pinv(self.Gcf.T.dot(self.Gcf))

        self.vector_to_shc_cf = self.GTGcf_inv.dot(self.Gcf.T)

        # Pre-calculate matrix to get coefficients for divergence-free fields
        self.Gdf = np.vstack((-self.G_ph, self.G_th))
        self.GTGdf_inv = np.linalg.pinv(self.Gdf.T.dot(self.Gdf))

        self.vector_to_shc_df = self.GTGdf_inv.dot(self.Gdf.T)