import math

class FlattenedArray(object):
    """
    Class for representing a multidimensional array as a one- or
    two-dimensional array by flattening indices.

    """

    def __init__(self, full_array, n_flattened_first = None, n_flattened_last = None):
        """
        Intialize a FlattenedArray object from the multidimensional array
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