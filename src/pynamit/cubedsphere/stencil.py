import numpy as np
from itertools import combinations_with_replacement
from functools import reduce
from scipy.special import factorial


def get_2D_stencil_coefficients(dx, dy, derivative = 'xx'):
    """
    Calculate stencil coefficients for numerical differentiation of
    ``f(x, y)``

    Derivative is found by sum over ``i``::

        c[i] * f(x + dx[i], y + dy[i]) 

    Only second order differentiation is supported at the moment but it
    would be fairly easy to expand... It should also be pretty easy to
    expand this to functions of more than one parameter

    Parameters
    ----------
    dx: array-like
        array of stencil points in x-dimension
    dy: array-like
        array of stencil points in y-dimension
    derivative: string
        ``x``: df/dx, ``y``: df/dy, ``xy``: d^2f/dxdy,
        ``yx``: d^2f/dxdy, ``xx``: d^2f/dx^2, ``yy``: d^2f/dy^2

    Returns
    -------
    c: array
        array of finite difference coefficients

    """
    dx, dy = np.array(dx).flatten(), np.array(dy).flatten()
    assert dx.size == dy.size


    NN = 4 # how high derivative terms to include when making the design matrix
    keys   = ['_']
    values = [np.ones_like(dx)]

    for i in range(1, NN):
        newkeys = [''.join(combination) for combination in combinations_with_replacement(['x', 'y'], i)]
        keys   += newkeys
        permutations = [len(set(key)) for key in newkeys] 
        values += [p * np.product(_, axis=0) / factorial(i) for p, _ in zip(permutations, [np.vstack(x) for x in combinations_with_replacement([dx, dy], i)])]

    d = np.zeros(len(keys))
    for i, key in enumerate(keys):
        if derivative == key or derivative[::-1] == key:
            d[i] = 1

    G = np.vstack(values)
    c = np.linalg.lstsq(G, d, rcond = 0)[0]

    return(c)


if __name__ == '__main__':
    print('Testing the 2D stencil coefficient function')
    import matplotlib.pyplot as plt

    dx, dy = map(np.ravel, np.meshgrid([-0.2, -0.1, 0, 0.1, 0.2], [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], indexing="ij"))

    # define a test function with derivatives
    f       = lambda x, y: x**2 - y**2 + x**2 * y - x*y + y**3
    dfdx    = lambda x, y:  2* x + 2 * x * y - y
    dfdy    = lambda x, y: -2* y + x**2 - x + 3 * y**2
    d2fdx2  = lambda x, y:  2 + 2 * y
    d2fdy2  = lambda x, y: -2 + 6 * y
    d2fdxdy = lambda x, y:  2*x - 1

    x0, y0 = (np.random.random(500) - 0.5) * 20,  (np.random.random(500) - 0.5) * 20

    fig, axes = plt.subplots(ncols = 5, figsize = (15, 3))

    for derivative, df, ax in zip(['x', 'y', 'xx', 'yy', 'xy'], [dfdx, dfdy, d2fdx2, d2fdy2, d2fdxdy], axes.flatten()):
        derivatives = df(x0, y0)
        stencil = get_2D_stencil_coefficients(dx, dy, derivative = derivative)
        if derivative == 'x':
            print(stencil)
        derivatives_num = reduce(lambda x, y: x + y,  [stencil[i] * f(x0 + dx[i], y0 + dy[i]) for i in range(len(stencil))])
        ax.scatter(derivatives, derivatives_num)
        ax.set_aspect('equal')
        ax.set_title(r'$f_{' + derivative + '}$')

        print('numerical and analytical values of f_' + derivative + ' match: {}'.format(np.allclose(derivatives - derivatives_num, 0)))


    plt.tight_layout()
    plt.show()








