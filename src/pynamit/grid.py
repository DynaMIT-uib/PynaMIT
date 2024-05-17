import numpy as np

class Grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, r, lat, lon, mainfield = None):
        """ Initialize the grid for the ionosphere.

        """
        r, lat, lon = np.broadcast_arrays(r, lat, lon)

        self.r = r.flatten()
        self.lat = lat.flatten()
        self.lon = lon.flatten()
        self.size = self.lon.size

        self.theta = 90 - self.lat

        # Get magnetic field unit vectors and inclination at grid:
        if mainfield is not None:
            B = np.vstack(mainfield.get_B(self.r, self.theta, self.lon))
            self.B_magnitude = np.linalg.norm(B, axis = 0)
            self.br, self.btheta, self.bphi = B / self.B_magnitude
            self.sinI = -self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2) # sin(inclination)