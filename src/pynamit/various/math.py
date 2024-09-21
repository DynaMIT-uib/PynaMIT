import numpy as np

def pinv_positive_semidefinite(A, rtol = 1e-15, condition_number = False):
    """
    Return the pseudoinverse of the positive semidefinite matrix A and,
    if requested, print its condition number.
    """

    # For a symmetric positive semidefinite matrix, its eigenvalues are equal to its singular values.
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Filter out small eigenvalues using a relative tolerance, following the convention of numpy.linalg.pinv.
    first_nonzero = np.argmax(eigenvalues > rtol * eigenvalues[-1])
    filtered_eigenvalues = eigenvalues[first_nonzero:]

    # If there are no zero eigenvalues, the filtered eigenvectors array is a full contiguous view of the original array.
    # Otherwise, the view's memory address and shape are adjusted to avoid unnecessary data copies.
    filtered_eigenvectors = eigenvectors[:, first_nonzero:]

    if condition_number:
        print('The condition number for the matrix is: {:.1f}'.format(filtered_eigenvalues[-1] / filtered_eigenvalues[0]))

    # Compute the pseudoinverse using the filtered eigenvalues and eigenvectors.
    return np.einsum('ij, j, jk -> ik', filtered_eigenvectors, 1 / filtered_eigenvalues, filtered_eigenvectors.T)

def matrix_product(A, B, contracted_dims):
    """
    Compute the product of two matrices A and B, contracting the indices from uncontracted_first to uncontracted_last.
    """

    first_dims = A.shape[:contracted_dims]
    last_dims  = B.shape[contracted_dims:]

    AB = np.dot(
        A.reshape((np.prod(first_dims), -1)),
        B.reshape((-1, np.prod(last_dims)))
    ).reshape((first_dims + last_dims))

    return AB