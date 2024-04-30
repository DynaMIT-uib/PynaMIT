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

    def costruct_dG(self, sha):
        """ Construct the G_ph and G_th matrices.

        """

        self.G_ph = sha.get_G(self, derivative = 'phi'  )
        self.G_th = sha.get_G(self, derivative = 'theta')