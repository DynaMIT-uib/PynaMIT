"""Flattened array module.

This module contains the FlattenedArray class for representing
multidimensional arrays in flattened form.
"""

import math


class FlattenedArray(object):
    """Class for representing multidimensional arrays in flattened form.

    Encapsulates a multidimensional array in flattened form while
    maintaining the ability to reference the original shape.

    Attributes
    ----------
    full_array : array-like
        Original multidimensional array.
    full_shapes : tuple of tuples
        Original shapes of (flattened_first_dims, flattened_last_dims).
    shapes : tuple of int
        Shape after flattening (n_first, n_last).
    array : array-like
        Flattened array of shape `shapes`.
    """

    def __init__(self, full_array, n_leading_flattened=None, n_trailing_flattened=None):
        """Initialize a FlattenedArray object.

        Parameters
        ----------
        full_array : array-like
            The multidimensional array to be flattened.
        n_leading_flattened : int, optional
            Number of leading dimensions to flatten into first axis.
        n_trailing_flattened : int, optional
            Number of trailing dimensions to flatten into last axis.

        Raises
        ------
        ValueError
            If neither or invalid combination of `n_leading_flattened`
            and `n_trailing_flattened` is specified.

        Notes
        -----
        If both `n_leading_flattened` and `n_trailing_flattened` are
        specified, they must sum to the total number of dimensions.
        """
        if n_leading_flattened is None and n_trailing_flattened is None:
            raise ValueError(
                "Either 'n_leading_flattened' or 'n_trailing_flattened' must be specified."
            )
        elif n_leading_flattened is not None and n_trailing_flattened is not None:
            if n_leading_flattened + n_trailing_flattened != full_array.ndim:
                raise ValueError(
                    "'n_leading_flattened' + 'n_trailing_flattened' must be equal to "
                    "the number of dimensions of the array."
                )
        elif n_trailing_flattened is not None:
            n_leading_flattened = full_array.ndim - n_trailing_flattened
        elif n_leading_flattened is not None:
            n_trailing_flattened = full_array.ndim - n_leading_flattened
        else:
            raise ValueError("This should not happen.")

        self.full_array = full_array
        self.full_shapes = (
            full_array.shape[:n_leading_flattened],
            full_array.shape[n_leading_flattened:],
        )

        self.shapes = (math.prod(self.full_shapes[0]), math.prod(self.full_shapes[1]))
        self.array = full_array.reshape(self.shapes)
