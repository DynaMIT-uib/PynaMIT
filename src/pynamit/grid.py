import numpy as np

class Grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, RI, lat, lon, mainfield = None):
        """ Initialize the grid for the ionosphere.

        """

        self.RI = RI
        self.lat = lat
        self.lon = lon
        self.size = self.lon.size

        self.theta = 90 - lat

        # Get magnetic field unit vectors and inclination at grid:
        if mainfield is not None:
            B = np.vstack(mainfield.get_B(self.RI, self.theta, self.lon))
            self.br, self.btheta, self.bphi = B / np.linalg.norm(B, axis = 0)
            self.sinI = -self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2) # sin(inclination)