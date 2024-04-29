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