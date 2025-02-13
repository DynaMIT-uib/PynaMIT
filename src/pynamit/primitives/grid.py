import numpy as np

class Grid(object):
    """
    Object for storing the coordinates of a two-dimensional grid. The
    `lat`, `lon`, `theta`, and `phi` coordinates are stored as attributes
    of the object, along with the `size` of the grid (the number of grid
    points).

    """

    def __init__(self, lat = None, lon = None, theta = None, phi = None):
        """
        Initialize the object for storing the two-dimensional grid. The
        grid is initialized from `lat` or `theta` and `lon` or `phi`
        coordinates.

        """
        
        if lat is not None:
            self.lat = lat
            self.theta = 90 - self.lat
        elif theta is not None:
            self.theta = theta
            self.lat = 90 - self.theta
        else:
            raise ValueError("Latitude or theta must be provided to initialize the grid.")

        if lon is not None:
            self.lon = lon
            self.phi = lon
        elif phi is not None:
            self.phi = phi
            self.lon = phi
        else:
            raise ValueError("Longitude or phi must be provided to initialize the grid.")

        self.lat, self.lon = np.broadcast_arrays(self.lat, self.lon)
        self.theta, self.phi = np.broadcast_arrays(self.theta, self.phi)

        self.lat = self.lat.flatten()
        self.lon = self.lon.flatten()
        self.theta = self.theta.flatten()
        self.phi = self.phi.flatten()

        self.size = self.lon.size