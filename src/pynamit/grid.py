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

    def get_RI(self):
        """ Get the radius of the grid.

        """
        return self.RI

    def get_lat(self):
        """ Get the latitude of the grid.

        """
        return self.lat
    
    def get_colat(self):
        """ Get the colatitude of the grid.

        """
        return 90 - self.lat

    def get_lon(self):
        """ Get the longitude of the grid.

        """
        return self.lon