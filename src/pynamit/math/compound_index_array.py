import math

class CompoundIndexArray(object):
    """
    Class for representing a compound index array.

    """

    def __init__(self, full_array, first_n_compounded = None, last_n_compounded = None):
        if first_n_compounded is None and last_n_compounded is None:
            raise ValueError('Either first_n_compounded or last_n_compounded must be specified.')
        elif first_n_compounded is not None and last_n_compounded is not None:
            if first_n_compounded + last_n_compounded != full_array.ndim:
                raise ValueError('first_n_compounded + last_n_compounded must be equal to the number of dimensions of the array.')
        elif last_n_compounded is not None:
            first_n_compounded = full_array.ndim - last_n_compounded
        elif first_n_compounded is not None:
            last_n_compounded = full_array.ndim - first_n_compounded
        else:
            raise ValueError('This should not happen.')

        self.full_array = full_array
        self.full_shapes = (full_array.shape[:first_n_compounded], full_array.shape[first_n_compounded:])

        self.shapes = (math.prod(self.full_shapes[0]), math.prod(self.full_shapes[1]))
        self.array = full_array.reshape(self.shapes)