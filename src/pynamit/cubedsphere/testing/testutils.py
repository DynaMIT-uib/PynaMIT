import cupy as np

def Geocentric_to_PlateCarree_vector_components(east, north, latitude):
    """ convert east north vector components to Plate Carree projection 
        
    This function is intende to be used with Cartopy, which does not give
    reasonable results if you just call quiver directly. Maybe it will change
    in future versions...


    Parameters
    ----------
    east: array-like
        eastward components
    north: array-like
        westward components
    latitude: array-like
        latitude of each vector
    
    Returns
    -------
    east: array
        corrected eastward components
    north: array
        corrected northward components
    """

    magnitude = np.sqrt(east**2 + north**2)

    east_pc = east / np.cos(latitude * np.pi / 180)

    magnitude_pc = np.sqrt(east_pc**2 + north**2)

    east_pc  = east_pc * magnitude / magnitude_pc
    north_pc = north * magnitude / magnitude_pc

    return east_pc, north_pc

