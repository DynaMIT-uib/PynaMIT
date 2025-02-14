import numpy as np

def tensor_product(A, B, n_contracted):
    """
    Compute the product of two matrices `A` and `B`, contracting the last
    `n_contracted` indices of the tensor `A` with the first `n_contracted`
    indices of the tensor `B`.

    """

    first_dims = A.shape[:n_contracted]
    last_dims  = B.shape[n_contracted:]

    AB = np.dot(
        A.reshape((np.prod(first_dims), -1)),
        B.reshape((-1, np.prod(last_dims)))
    ).reshape((first_dims + last_dims))

    return AB

def tensor_pinv(A, n_flattened_first=2, rtol=1e-15, hermitian=False):
    """
    Compute the Moore-Penrose pseudoinverse of the tensor `A`, by treating
    the first `n_flattened_first` indices and the remaining indices as
    flat indices.

    """

    first_dims = A.shape[:n_flattened_first]
    last_dims  = A.shape[n_flattened_first:]

    A_inv = np.linalg.pinv(
        A.reshape((np.prod(first_dims), np.prod(last_dims))), rcond=rtol, hermitian=hermitian
    ).reshape((last_dims + first_dims))

    return A_inv

def tensor_pinv_positive_semidefinite(A, n_flattened_first=2, rtol=1e-15, condition_number=False):
    """
    Compute the Moore-Penrose pseudoinverse of the positive semidefinite
    tensor `A`, by treating the first `n_flattened_first` indices and the
    remaining indices as flat indices.

    """

    first_dims = A.shape[:n_flattened_first]
    last_dims  = A.shape[n_flattened_first:]

    A_inv = pinv_positive_semidefinite(
        A.reshape((np.prod(first_dims), np.prod(last_dims))), rtol=rtol, condition_number=condition_number
    ).reshape((last_dims + first_dims))

    return A_inv

def tensor_transpose(A, n_flattened_first=2):
    """
    Transpose a tensor, by treating the first `n_flattened_first` indices
    and the remaining indices as flat indices.

    """

    first_dims = A.shape[:n_flattened_first]
    last_dims  = A.shape[n_flattened_first:]

    A_transposed = A.reshape((np.prod(first_dims), np.prod(last_dims))).T.reshape((last_dims + first_dims))

    return A_transposed

def tensor_scale_left(scaling_tensor, A):
    """
    Perform the element-wise scaling of the first indices of the tensor
    `A` by the `scaling_tensor` tensor, by treating these indices as flat
    indices.

    """

    first_dims = scaling_tensor.shape
    last_dims  = A.shape[len(first_dims):]

    A_scaled = scaling_tensor.reshape((np.prod(first_dims), 1)) * A.reshape((np.prod(first_dims), np.prod(last_dims)))

    return A_scaled.reshape((first_dims + last_dims))

def tensor_scale_right(A, scaling_tensor):
    """
    Perform the element-wise scaling of the last indices of the tensor `A`
    by the array `scaling_tensor`.
    
    """

    last_dims = scaling_tensor.shape
    first_dims = A.shape[:-len(last_dims)]

    A_scaled = A.reshape((np.prod(first_dims), np.prod(last_dims))) * scaling_tensor.reshape((1, np.prod(last_dims)))

    return A_scaled.reshape((first_dims + last_dims))

def tensor_outer(A, B, n_flattened_first):
    """
    Compute the outer product of two tensors `A` and `B` by treating the
    first `n_flattened_first` indices of `A` and `B` and the remaining
    indices of each tensor as flat indices.

    """

    first_A_dims = A.shape[:n_flattened_first]
    first_B_dims = B.shape[:n_flattened_first]

    if first_A_dims != first_B_dims:
        raise ValueError('First dimensions of outer product tensors do not match.')

    last_A_dims = A.shape[n_flattened_first:]
    last_B_dims = B.shape[n_flattened_first:]

    outer = np.einsum(
        'ij,ik->ijk',
        A.reshape((np.prod(first_A_dims), np.prod(last_A_dims))),
        B.reshape((np.prod(first_B_dims), np.prod(last_B_dims))),
        optimize = True
    ).reshape((first_A_dims + last_A_dims + last_B_dims))

    return outer

def tensor_svd(A, n_flattened_first=2, full_matrices=True, compute_uv=True, hermitian=False, rtol=1e-15):
    """
    Compute the singular value decomposition of the tensor `A`, by
    treating the first `n_flattened_first` indices and the remaining
    indices as flat indices.

    """

    first_dims = A.shape[:n_flattened_first]
    last_dims  = A.shape[n_flattened_first:]

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


def pinv_positive_semidefinite(A, rtol = 1e-15, condition_number = False):
    """
    Return the pseudoinverse of the positive semidefinite matrix `A` and,
    if requested, print its condition number.

    Note: this is very similar to what numpy.linalg.pinv does when the
    argument hermitian = True is specified, so we can just use that
    function instead.

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
    return np.einsum('ij, j, jk -> ik', filtered_eigenvectors, 1 / filtered_eigenvalues, filtered_eigenvectors.T, optimize = True)