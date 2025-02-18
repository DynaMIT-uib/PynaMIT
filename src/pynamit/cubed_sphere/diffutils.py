"""Finite Difference Utilities for Cubed Sphere Calculations

This module provides functions to compute finite difference stencils and other
numerical differentiation coefficients needed for cubed sphere grid operations in PynaMIT.
"""

import numpy as np
from scipy.special import factorial
from fractions import Fraction
from itertools import combinations_with_replacement


def lcm_arr(arr):
    """Calculate least common multiplier for array of integers."""
    result = np.lcm(arr[0], arr[1])
    for i in range(2, len(arr) - 1):
        result = np.lcm(result, arr[i])

    return result


def stencil(evaluation_points, order=1, h=1, fraction=False):
    """
    Calculate stencil for finite difference calculation of derivative.

    Parameters
    ----------
    evaluation_points: array_like
        Evaluation points in regular grid. e.g. ``[-1, 0, 1]`` for central
        difference or ``[-1, 0]`` for backward difference.
    order: integer, optional, default = 1
        Order of the derivative, default gives a first order derivative.
    h: scalar, optional, default = 1
        Step size in seconds.
    fraction: bool, optional
        Set to ``True`` to return coefficients as integer numerators and a
        common denomenator. Be careful with this if you use a very large
        number of evaluation points.

    Returns
    -------
    coefficients: array
        Array of coefficients in stencil. Unless fraction is set to
        ``True`` - in which case a tuple will be returned with an array of
        numerators and an integer denominator. If fraction is ``True``,
        `h` is ignored - and you should multiply the denominator by
        ``h**order`` to get the coefficients.

    Note
    ----
    Algorithm based on Taylor series expansion. See this page for
    explanation:

        https://web.media.mit.edu/~crtaylor/calculator.html

    """

    # calculate coefficients:
    evaluation_points = np.array(evaluation_points).flatten().reshape((1, -1))
    p = np.arange(evaluation_points.size).reshape((-1, 1))
    d = np.zeros(evaluation_points.size)
    d[order] = factorial(order)

    coeffs = np.linalg.inv(evaluation_points**p).dot(d)

    if fraction:
        # format nicely:
        fracs = [Fraction(c).limit_denominator() for c in coeffs]
        denominators = [c.denominator for c in fracs]
        numerators = [c.numerator for c in fracs]
        cd = lcm_arr(denominators)
        numerators = [int(c * cd / a) for (c, a) in zip(numerators, denominators)]
        return (numerators, cd)
    else:
        return coeffs / h**order


def get_2D_stencil_coefficients(dx, dy, derivative="xx"):
    """
    Calculate stencil coefficients for numerical differentiation of
    ``f(x, y)``.

    Derivative is found by sum over ``i``::

        c[i] * f(x + dx[i], y + dy[i])

    Note
    ----
    This function is based on Taylor series expansion (see docs for rough
    summary).

    Only second order differentiation is supported at the moment but it
    would be fairly easy to expand... It should also be pretty easy to
    expand this to functions of more than one parameter.

    Also, this function should probably be tested more, in particular if
    it is used in ways that are not currently tested in the ``__main__``
    block at the bottom of the script.

    Parameters
    ----------
    dx: array-like
        Array of stencil points in x-dimension.
    dy: array-like
        Array of stencil points in y-dimension.
    derivative: string
        Defines which derivative you want. Currently these strings are
        supported:

        - 'x': ``df/dx``
        - 'y': ``df/dy``
        - 'xy': ``d^2f/dxdy``
        - 'yx': ``d^2f/dxdy``
        - 'xx': ``d^2f/dx^2``
        - 'yy': ``d^2f/dy^2``

    Returns
    -------
    c: array
        Array of finite difference coefficients.

    """
    dx, dy = np.array(dx).flatten(), np.array(dy).flatten()
    assert dx.size == dy.size

    NN = 4  # how high derivative terms to include when making the design matrix
    keys = ["_"]
    values = [np.ones_like(dx)]

    for i in range(1, NN):
        newkeys = [
            "".join(combination)
            for combination in combinations_with_replacement(["x", "y"], i)
        ]
        keys += newkeys
        permutations = [len(set(key)) for key in newkeys]
        values += [
            p * np.prod(_, axis=0) / factorial(i)
            for p, _ in zip(
                permutations,
                [np.vstack(x) for x in combinations_with_replacement([dx, dy], i)],
            )
        ]

    d = np.zeros(len(keys))
    for i, key in enumerate(keys):
        if derivative == key or derivative[::-1] == key:
            d[i] = 1

    G = np.vstack(values)
    c = np.linalg.lstsq(G, d, rcond=0)[0]

    return c


if __name__ == "__main__":
    print("Testing the 2D stencil coefficient function")
    import matplotlib.pyplot as plt
    from functools import reduce

    dx, dy = map(
        np.ravel,
        np.meshgrid(
            [-0.2, -0.1, 0, 0.1, 0.2],
            [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
            indexing="ij",
        ),
    )

    # define a test function with derivatives
    def f(x, y):
        return x**2 - y**2 + x**2 * y - x * y + y**3

    def dfdx(x, y):
        return 2 * x + 2 * x * y - y

    def dfdy(x, y):
        return -2 * y + x**2 - x + 3 * y**2

    def d2fdx2(x, y):
        return 2 + 2 * y

    def d2fdy2(x, y):
        return -2 + 6 * y

    def d2fdxdy(x, y):
        return 2 * x - 1

    x0, y0 = (np.random.random(500) - 0.5) * 20, (np.random.random(500) - 0.5) * 20

    fig, axes = plt.subplots(ncols=5, figsize=(15, 3))

    for derivative, df, ax in zip(
        ["x", "y", "xx", "yy", "xy"],
        [dfdx, dfdy, d2fdx2, d2fdy2, d2fdxdy],
        axes.flatten(),
    ):
        derivatives = df(x0, y0)
        stencil = get_2D_stencil_coefficients(dx, dy, derivative=derivative)
        derivatives_num = reduce(
            lambda x, y: x + y,
            [stencil[i] * f(x0 + dx[i], y0 + dy[i]) for i in range(len(stencil))],
        )
        ax.scatter(derivatives, derivatives_num)
        ax.set_aspect("equal")
        ax.set_title(r"$f_{" + derivative + "}$")

        print(
            "numerical and analytical values of f_"
            + derivative
            + " match: {}".format(np.allclose(derivatives - derivatives_num, 0))
        )

    plt.tight_layout()
    plt.show()
    plt.close()
