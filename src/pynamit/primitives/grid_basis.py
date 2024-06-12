"""
Grid basis.

This module contains the ``GridBasis`` class.
    
"""

import numpy as np

class GridBasis(object):
    """ Grid basis.

    Class to store information about a grid basis and to generate matrices
    for evaluating the basis functions at a given grid.

    """

    def __init__(self, grid):
        self.grid = grid

        # Number of grid points and coefficients
        self.num_coeffs = self.grid.size


    def get_G(self, grid, derivative = None):
        """
        Calculate matrix that evaluates the grid basis functions at the
        latitudes and longitudes of the given `grid`.
        """

        #For now, can only handle the case where the grid is the same as the basis grid

        if (not np.isclose(self.grid.lat - grid.lat, 0).all()) or (not np.isclose(self.grid.lon - grid.lon, 0).all()):
            raise ValueError('get_G: Grids must be the same')
        
        if derivative is not None:
            raise ValueError('get_G: Derivatives not implemented yet')

        return np.eye(self.num_coeffs)