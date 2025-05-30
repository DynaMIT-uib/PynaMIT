"""Test Utilities for Cubed Sphere Calculations.

This module contains helper functions for testing and validating
components of the cubed sphere implementation, including coordinate
projection adjustments for visualizations using Plate Carree
projections.
"""

import numpy as np


def Geocentric_to_PlateCarree_vector_components(east, north, latitude):
    """Convert east north vector components to Plate Carree projection.

    This function is intended to be used with Cartopy, which does not
    give reasonable results if you just call quiver directly. Maybe it
    will change in future versions...

    Parameters
    ----------
    east : array-like
        Eastward components.
    north : array-like
        Westward components.
    latitude : array-like
        Latitude of each vector.

    Returns
    -------
    east : array
        Corrected eastward components.
    north : array
        Corrected northward components.
    """
    magnitude = np.sqrt(east**2 + north**2)
    east_pc = east / np.cos(latitude * np.pi / 180)
    magnitude_pc = np.sqrt(east_pc**2 + north**2)
    east_pc = east_pc * magnitude / magnitude_pc
    north_pc = north * magnitude / magnitude_pc
    return east_pc, north_pc
