"""Array dimension flattening utilities.

This module provides tools for converting between multidimensional and flattened
array representations while preserving the ability to reshape back to the original form.

Classes
-------
FlattenedArray
    Represents multidimensional arrays in flattened form with reshaping capability.
"""

import math

class FlattenedArray(object):
    """Represents multidimensional arrays in flattened form.
    
    Converts between multidimensional and flattened array representations while
    maintaining the ability to reshape back to original dimensions. Allows flattening
    specified numbers of dimensions from start and/or end of array.

    Parameters
    ----------
    full_array : array-like
        The multidimensional array to be flattened
    n_flattened_first : int, optional
        Number of leading dimensions to flatten into first axis
    n_flattened_last : int, optional
        Number of trailing dimensions to flatten into last axis

    Attributes
    ----------
    full_array : array-like
        Original multidimensional array
    full_shapes : tuple of tuples
        Original shapes of (flattened_first_dims, flattened_last_dims)
    shapes : tuple of int
        Shape after flattening (n_first, n_last)
    array : array-like
        Flattened array of shape `shapes`

    Notes
    -----
    Must specify at least one of n_flattened_first or n_flattened_last.
    If both are specified, they must sum to the total number of dimensions.

    Raises
    ------
    ValueError
        If neither or invalid combination of n_flattened_first/last specified
    """

    def __init__(self, full_array, n_flattened_first=None, n_flattened_last=None):
        """
        Initialize a FlattenedArray object from the multidimensional array
        `full_array` and at least one of `n_flattened_first` or
        `n_flattened_last`, specifying the number of first or last
        dimensions to flatten, respectively.

        """

        if n_flattened_first is None and n_flattened_last is None:
            raise ValueError('Either n_flattened_first or n_flattened_last must be specified.')
        elif n_flattened_first is not None and n_flattened_last is not None:
            if n_flattened_first + n_flattened_last != full_array.ndim:
                raise ValueError('n_flattened_first + n_flattened_last must be equal to the number of dimensions of the array.')
        elif n_flattened_last is not None:
            n_flattened_first = full_array.ndim - n_flattened_last
        elif n_flattened_first is not None:
            n_flattened_last = full_array.ndim - n_flattened_first
        else:
            raise ValueError('This should not happen.')

        self.full_array = full_array
        self.full_shapes = (full_array.shape[:n_flattened_first], full_array.shape[n_flattened_first:])

        self.shapes = (math.prod(self.full_shapes[0]), math.prod(self.full_shapes[1]))
        self.array = full_array.reshape(self.shapes)