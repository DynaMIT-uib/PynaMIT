class grid(object):
    """ Grid for the ionosphere.

    """

    def __init__(self, RI, lat, lon):
        """ Initialize the grid for the ionosphere.

        """

        self.RI = RI
        self.lat = lat
        self.lon = lon
        self.size = self.lon.size

        self.theta = 90 - lat
