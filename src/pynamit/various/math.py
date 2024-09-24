import numpy as np

def pinv_positive_semidefinite(A, rtol = 1e-15, condition_number = False):
    """
    Return the pseudoinverse of the positive semidefinite matrix A and,
    if requested, print its condition number.
    """

    # For a symmetric positive semidefinite matrix, its eigenvalues are equal to its singular values.
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Filter out small eigenvalues using a relative tolerance, following the convention of numpy.linalg.pinv.
    if rtol:
        mask = (eigenvalues > rtol * eigenvalues[-1])
    else:
        mask = False

    if np.any(mask):
        first_nonzero = np.argmax(mask)
    else:
        first_nonzero = len(eigenvalues)

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

def tensor_pinv(A, contracted_dims=2, rtol=1e-15, hermitian=False):
    """
    Compute the Moore-Penrose pseudoinverse of a tensor.

    """

    first_dims = A.shape[:contracted_dims]
    last_dims  = A.shape[contracted_dims:]

    A_inv = np.linalg.pinv(
        A.reshape((np.prod(first_dims), np.prod(last_dims))), rcond=rtol, hermitian=hermitian
    ).reshape((last_dims + first_dims))

    return A_inv

def tensor_pinv_positive_semidefinite(A, contracted_dims=2, rtol=1e-15, condition_number=False):
    """
    Compute the Moore-Penrose pseudoinverse of a positive semidefinite tensor.

    """

    first_dims = A.shape[:contracted_dims]
    last_dims  = A.shape[contracted_dims:]

    A_inv = pinv_positive_semidefinite(
        A.reshape((np.prod(first_dims), np.prod(last_dims))), rtol=rtol, condition_number=condition_number
    ).reshape((last_dims + first_dims))

    return A_inv

def tensor_scale_left(scaling_factors, A):
    """
    Scale the first indices of a tensor A by the scaling factors.

    """

    first_dims = scaling_factors.shape
    last_dims  = A.shape[len(first_dims):]

    A_scaled = scaling_factors.reshape((np.prod(first_dims), 1)) * A.reshape((np.prod(first_dims), np.prod(last_dims)))

    return A_scaled.reshape((first_dims + last_dims))

def tensor_scale_right(A, scaling_factors):
    """
    Scale the last indices of a tensor A by the scaling factors.

    """

    last_dims = scaling_factors.shape
    first_dims = A.shape[:len(last_dims)]
    print(A.shape, first_dims, last_dims)

    A_scaled = A.reshape((np.prod(first_dims), np.prod(last_dims))) * scaling_factors.reshape((1, np.prod(first_dims)))

    return A_scaled.reshape((first_dims + last_dims))

def tensor_outer(A, B, contracted_dims):

    first_A_dims = A.shape[:contracted_dims]
    first_B_dims = B.shape[:contracted_dims]

    if first_A_dims != first_B_dims:
        raise ValueError('First dimensions of outer product tensors do not match.')

    last_A_dims = A.shape[contracted_dims:]
    last_B_dims = B.shape[contracted_dims:]

    outer = np.einsum(
        'ij,ik->ijk',
        A.reshape((np.prod(first_A_dims), np.prod(last_A_dims))),
        B.reshape((np.prod(first_B_dims), np.prod(last_B_dims)))
    ).reshape((first_A_dims + last_A_dims + last_B_dims))

    return outer

def tensor_svd(A, contracted_dims=2, full_matrices=True, compute_uv=True, hermitian=False, rtol=1e-15):

    first_dims = A.shape[:contracted_dims]
    last_dims  = A.shape[contracted_dims:]

    U, S, VT = np.linalg.svd(
        A.reshape((np.prod(first_dims), np.prod(last_dims))),
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        hermitian=hermitian
    )

    if rtol:
        mask = (S <= rtol * S[0])
    else:
        mask = False

    if np.any(mask):
        first_zero = np.argmax(mask)
    else:
        first_zero = len(S)

    filtered_S = S[:first_zero]
    filtered_U = U[:, :first_zero].reshape(first_dims + (first_zero, ))
    filtered_VT = VT[:, :first_zero].reshape((first_zero, ) + last_dims)

    return filtered_U, filtered_S, filtered_VT