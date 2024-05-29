import numpy as np

class Grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, lat, lon):
        """ Initialize the grid for the ionosphere.

        """
        lat, lon = np.broadcast_arrays(lat, lon)

        self.lat = lat.flatten()
        self.lon = lon.flatten()
        self.size = self.lon.size

        self.theta = 90 - self.lat
