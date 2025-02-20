"""Tensor Operations.

This module provides functions for performing various tensor operations
including tensor products, pseudoinverses, transpositions, scaling,
outer products, and singular value decompositions.
"""

import numpy as np


def tensor_product(A, B, n_contracted):
    """Product of two tensors.

    Compute the product of two tensors `A` and `B`, contracting the last
    `n_contracted` indices of the tensor `A` with the first
    `n_contracted` indices of the tensor `B`.

    Parameters
    ----------
    A : array-like
        First tensor.
    B : array-like
        Second tensor.
    n_contracted : int
        Number of indices to contract.

    Returns
    -------
    array-like
        Product of the two tensors `A` and `B`.
    """
    first_dims = A.shape[:n_contracted]
    last_dims = B.shape[n_contracted:]

    AB = np.dot(
        A.reshape((np.prod(first_dims), -1)),
        B.reshape((-1, np.prod(last_dims))),
    ).reshape((first_dims + last_dims))

    return AB


def tensor_pinv(A, n_leading_flattened=2, rtol=1e-15, hermitian=False):
    """Moore-Penrose pseudoinverse of a tensor.

    Computes the Moore-Penrose pseudoinverse of the tensor `A`, by
    treating the first `n_leading_flattened` indices and the remaining
    indices as flat indices.

    Parameters
    ----------
    A : array-like
        Input tensor.
    n_leading_flattened : int, optional
        Number of leading dimensions to flatten into first axis. Default
        is 2.
    rtol : float, optional
        Relative tolerance for small singular values. Default is 1e-15.
    hermitian : bool, optional
        Whether the matrix is Hermitian. Default is False.

    Returns
    -------
    array-like
        Pseudoinverse of the tensor.
    """
    first_dims = A.shape[:n_leading_flattened]
    last_dims = A.shape[n_leading_flattened:]

    A_inv = np.linalg.pinv(
        A.reshape((np.prod(first_dims), np.prod(last_dims))),
        rcond=rtol,
        hermitian=hermitian,
    ).reshape((last_dims + first_dims))

    return A_inv


def tensor_pinv_positive_semidefinite(
    A, n_leading_flattened=2, rtol=1e-15, condition_number=False
):
    """Moore-Penrose pseudoinverse of a positive semidefinite tensor.

    Computes the Moore-Penrose pseudoinverse of the positive
    semidefinite tensor `A`, by treating the first `n_leading_flattened`
    indices and the remaining indices as flat indices.

    Parameters
    ----------
    A : array-like
        Input tensor.
    n_leading_flattened : int, optional
        Number of leading dimensions to flatten into first axis.
        Default is 2.
    rtol : float, optional
        Relative tolerance for small singular values. Default is 1e-15.
    condition_number : bool, optional
        Whether to print the condition number. Default is False.

    Returns
    -------
    array-like
        Pseudoinverse of the tensor.
    """
    first_dims = A.shape[:n_leading_flattened]
    last_dims = A.shape[n_leading_flattened:]

    A_inv = pinv_positive_semidefinite(
        A.reshape((np.prod(first_dims), np.prod(last_dims))),
        rtol=rtol,
        condition_number=condition_number,
    ).reshape((last_dims + first_dims))

    return A_inv


def tensor_transpose(A, n_leading_flattened=2):
    """Transpose a tensor.

    Transposes a tensor by treating the first `n_leading_flattened`
    indices and the remaining indices as flat indices.

    Parameters
    ----------
    A : array-like
        Input tensor.
    n_leading_flattened : int, optional
        Number of leading dimensions to flatten into first axis. Default
        is 2.

    Returns
    -------
    array-like
        Transposed tensor.
    """
    first_dims = A.shape[:n_leading_flattened]
    last_dims = A.shape[n_leading_flattened:]

    A_transposed = A.reshape(
        (np.prod(first_dims), np.prod(last_dims))
    ).T.reshape((last_dims + first_dims))

    return A_transposed


def tensor_scale_left(scaling_tensor, A):
    """Element-wise scaling of the first indices of a tensor.

    Performs the element-wise scaling of the array corresponding to the
    first indices of the tensor `A` by the array `scaling_tensor`, by
    by treating the array indices as flat indices.

    Parameters
    ----------
    scaling_tensor : array-like
        Tensor to scale by.
    A : array-like
        Input tensor.

    Returns
    -------
    array-like
        Scaled tensor.
    """
    first_dims = scaling_tensor.shape
    last_dims = A.shape[len(first_dims):]

    A_scaled = scaling_tensor.reshape((np.prod(first_dims), 1)) * A.reshape(
        (np.prod(first_dims), np.prod(last_dims))
    )

    return A_scaled.reshape((first_dims + last_dims))


def tensor_scale_right(A, scaling_tensor):
    """Element-wise scaling of the last indices of a tensor.

    Performs the element-wise scaling of the array corresponding to the
    last indices of the tensor `A` by the array `scaling_tensor`, by
    treating the array indices as flat indices.

    Parameters
    ----------
    A : array-like
        Input tensor.
    scaling_tensor : array-like
        Tensor to scale by.

    Returns
    -------
    array-like
        Scaled tensor.
    """
    last_dims = scaling_tensor.shape
    first_dims = A.shape[: -len(last_dims)]

    A_scaled = A.reshape(
        (np.prod(first_dims), np.prod(last_dims))
    ) * scaling_tensor.reshape((1, np.prod(last_dims)))

    return A_scaled.reshape((first_dims + last_dims))


def tensor_outer(A, B, n_leading_flattened):
    """Outer product of two tensors.

    Computes the outer product of two tensors `A` and `B` by treating
    the first `n_leading_flattened` indices of `A` and `B` and the
    remaining indices of each tensor as flat indices, and computing the
    corresponding matrix outer product with numpy.einsum.

    Parameters
    ----------
    A : array-like
        First tensor.
    B : array-like
        Second tensor.
    n_leading_flattened : int
        Number of leading dimensions to flatten into first axis.

    Returns
    -------
    array-like
        Outer product of the two tensors.

    Raises
    ------
    ValueError
        If the first dimensions of the tensors do not match.
    """
    first_A_dims = A.shape[:n_leading_flattened]
    first_B_dims = B.shape[:n_leading_flattened]

    if first_A_dims != first_B_dims:
        raise ValueError(
            "First dimensions of outer product tensors do not match."
        )

    last_A_dims = A.shape[n_leading_flattened:]
    last_B_dims = B.shape[n_leading_flattened:]

    outer = np.einsum(
        "ij,ik->ijk",
        A.reshape((np.prod(first_A_dims), np.prod(last_A_dims))),
        B.reshape((np.prod(first_B_dims), np.prod(last_B_dims))),
        optimize=True,
    ).reshape((first_A_dims + last_A_dims + last_B_dims))

    return outer


def tensor_svd(
    A,
    n_leading_flattened=2,
    full_matrices=True,
    compute_uv=True,
    hermitian=False,
    rtol=1e-15,
):
    """Singular value decomposition of a tensor.

    Compute the singular value decomposition of the tensor `A` by
    treating the first `n_leading_flattened` indices and the remaining
    indices as flat indices, and calling the numpy.linalg.svd function.

    Parameters
    ----------
    A : array-like
        Input tensor.
    n_leading_flattened : int, optional
        Number of leading dimensions to flatten into first axis. Default
        is 2.
    full_matrices : bool, optional
        Whether to compute full-sized U and VT matrices. Default is
        True.
    compute_uv : bool, optional
        Whether to compute U and VT matrices. Default is True.
    hermitian : bool, optional
        Whether the matrix is Hermitian. Default is False.
    rtol : float, optional
        Relative tolerance for small singular values. Default is 1e-15.

    Returns
    -------
    tuple of array-like
        U, S, and VT matrices of the singular value decomposition.
    """
    first_dims = A.shape[:n_leading_flattened]
    last_dims = A.shape[n_leading_flattened:]

    U, S, VT = np.linalg.svd(
        A.reshape((np.prod(first_dims), np.prod(last_dims))),
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        hermitian=hermitian,
    )

    if rtol:
        mask = S <= rtol * S[0]
    else:
        mask = False

    if np.any(mask):
        first_zero = np.argmax(mask)
    else:
        first_zero = len(S)

    filtered_S = S[:first_zero]
    filtered_U = U[:, :first_zero].reshape(first_dims + (first_zero,))
    filtered_VT = VT[:, :first_zero].reshape((first_zero,) + last_dims)

    return filtered_U, filtered_S, filtered_VT


def pinv_positive_semidefinite(A, rtol=1e-15, condition_number=False):
    """Pseudoinverse of a positive semidefinite matrix.

    Parameters
    ----------
    A : array-like
        Positive semidefinite matrix.
    rtol : float, optional
        Relative tolerance for small eigenvalues. Default is 1e-15.
    condition_number : bool, optional
        Whether to print the condition number. Default is False.

    Returns
    -------
    array-like
        Pseudoinverse of the matrix `A`.

    Notes
    -----
    This is very similar to what numpy.linalg.pinv does when the
    argument ``hermitian = True`` is specified, so we can just use that
    function instead.

    If there are no zero eigenvalues, the filtered eigenvectors array is
    a full contiguous view of the original array. Otherwise, the view's
    memory address and shape are adjusted to avoid unnecessary data
    copies.
    """
    # For a symmetric positive semidefinite matrix, its eigenvalues are
    # equal to its singular values.
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Filter out small eigenvalues using a relative tolerance, following
    # numpy.linalg.pinv conventions.
    if rtol:
        mask = eigenvalues > rtol * eigenvalues[-1]
    else:
        mask = False

    if np.any(mask):
        first_nonzero = np.argmax(mask)
    else:
        first_nonzero = len(eigenvalues)

    filtered_eigenvalues = eigenvalues[first_nonzero:]
    filtered_eigenvectors = eigenvectors[:, first_nonzero:]

    if condition_number:
        print(
            "The condition number for the matrix is: {:.1f}".format(
                filtered_eigenvalues[-1] / filtered_eigenvalues[0]
            )
        )

    # Compute pseudoinverse using filtered eigenvalues and eigenvectors.
    return np.einsum(
        "ij, j, jk -> ik",
        filtered_eigenvectors,
        1 / filtered_eigenvalues,
        filtered_eigenvectors.T,
        optimize=True,
    )
