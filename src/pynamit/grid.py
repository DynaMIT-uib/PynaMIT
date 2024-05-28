import numpy as np
from pynamit.b_field.b_geometry import BGeometry

class Grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, r, lat, lon, mainfield = None):
        """ Initialize the grid for the ionosphere.

        """
        r, lat, lon = np.broadcast_arrays(r, lat, lon)

        self.lat = lat.flatten()
        self.lon = lon.flatten()
        self.size = self.lon.size

        self.theta = 90 - self.lat

        if mainfield is not None:
            self.b_geometry = BGeometry(mainfield, self, r)