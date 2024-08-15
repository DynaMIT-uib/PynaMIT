import numpy as np

def inv_and_cond_hermitian(A):
    """
    Return the inverse of the matrix A and print its condition number.
    """

    u, s, vh = np.linalg.svd(A, hermitian = True)

    #print('The condition number for the matrix is {:.1f}'.format(s[0] / s[-1]))

    return np.einsum('ij, j, jk -> ik', vh.T, 1 / s, u.T)
    #return np.dot(vh.T / s, u.T) # Need to compare relative speed of einsum and broadcast + dot