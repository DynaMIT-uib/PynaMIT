"""Tensor operations module.

This module contains functions for performing various tensor operations
including tensor products, pseudoinverses, transpositions, scaling,
outer products, and singular value decompositions. The implementations
are backend-aware and work with either NumPy or JAX NumPy arrays.
"""

from __future__ import annotations

import math

from pynamit.math.backend import get_array_module, to_numpy


def tensor_product(A, B, n_contracted):
    """Product of two tensors."""
    xp = get_array_module(A, B)
    A_arr = xp.asarray(A)
    B_arr = xp.asarray(B)
    return xp.tensordot(A_arr, B_arr, axes=n_contracted)


def tensor_pinv(A, n_leading_flattened=2, rtol=1e-15, hermitian=False):
    """Moore-Penrose pseudoinverse of a tensor."""
    xp = get_array_module(A)
    A_arr = xp.asarray(A)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    A_flat = A_arr.reshape((flat_first, flat_last))
    A_pinv = xp.linalg.pinv(A_flat, rtol=rtol, hermitian=hermitian)
    return A_pinv.reshape(last_dims + first_dims)


def weighted_tensor_pinv(A, sqrt_weights=None, n_leading_flattened=2, rtol=1e-15):
    """Weighted Moore-Penrose pseudoinverse of a tensor."""
    if sqrt_weights is None:
        return tensor_pinv(A, n_leading_flattened=n_leading_flattened, rtol=rtol)

    xp = get_array_module(A, sqrt_weights)
    A_arr = xp.asarray(A)
    weights = xp.asarray(sqrt_weights)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]
    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    weights_flat = weights.reshape(flat_first)
    A_flat = A_arr.reshape((flat_first, flat_last))
    weighted_A = weights_flat.reshape((-1, 1)) * A_flat
    weighted_pinv = xp.linalg.pinv(weighted_A, rtol=rtol)
    return (weighted_pinv * weights_flat.reshape((1, -1))).reshape(
        last_dims + first_dims
    )


def tensor_transpose(A, n_leading_flattened=2):
    """Transpose a tensor."""
    xp = get_array_module(A)
    A_arr = xp.asarray(A)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    return A_arr.reshape((flat_first, flat_last)).T.reshape(last_dims + first_dims)


def tensor_scale_left(scaling_tensor, A):
    """Element-wise scaling of the first indices of a tensor."""
    xp = get_array_module(scaling_tensor, A)
    scale_arr = xp.asarray(scaling_tensor)
    A_arr = xp.asarray(A)

    first_dims = scale_arr.shape
    last_dims = A_arr.shape[len(first_dims) :]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    scaled = scale_arr.reshape((flat_first, 1)) * A_arr.reshape((flat_first, flat_last))
    return scaled.reshape(first_dims + last_dims)


def tensor_scale_right(A, scaling_tensor):
    """Element-wise scaling of the last indices of a tensor."""
    xp = get_array_module(A, scaling_tensor)
    A_arr = xp.asarray(A)
    scale_arr = xp.asarray(scaling_tensor)

    last_dims = scale_arr.shape
    first_dims = A_arr.shape[: -len(last_dims)]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    scaled = A_arr.reshape((flat_first, flat_last)) * scale_arr.reshape((1, flat_last))
    return scaled.reshape(first_dims + last_dims)


def tensor_outer(A, B, n_leading_flattened):
    """Outer product of two tensors."""
    xp = get_array_module(A, B)
    A_arr = xp.asarray(A)
    B_arr = xp.asarray(B)

    first_A_dims = A_arr.shape[:n_leading_flattened]
    first_B_dims = B_arr.shape[:n_leading_flattened]

    if first_A_dims != first_B_dims:
        raise ValueError("First dimensions of outer product tensors do not match.")

    last_A_dims = A_arr.shape[n_leading_flattened:]
    last_B_dims = B_arr.shape[n_leading_flattened:]

    flat_first = math.prod(first_A_dims)
    flat_last_A = math.prod(last_A_dims)
    flat_last_B = math.prod(last_B_dims)

    outer = xp.einsum(
        "ij,ik->ijk",
        A_arr.reshape((flat_first, flat_last_A)),
        B_arr.reshape((flat_first, flat_last_B)),
        optimize=True,
    )
    return outer.reshape(first_A_dims + last_A_dims + last_B_dims)


def tensor_svd(
    A, n_leading_flattened=2, full_matrices=True, compute_uv=True, hermitian=False, rtol=1e-15
):
    """Singular value decomposition of a tensor."""
    xp = get_array_module(A)
    A_arr = xp.asarray(A)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    U, S, VT = xp.linalg.svd(
        A_arr.reshape((flat_first, flat_last)),
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        hermitian=hermitian,
    )

    first_zero = S.shape[0]
    if rtol:
        mask_np = to_numpy(S <= rtol * S[0])
        if mask_np.any():
            first_zero = int(mask_np.argmax())

    filtered_S = S[:first_zero]
    filtered_U = U[:, :first_zero].reshape(first_dims + (first_zero,))
    filtered_VT = VT[:, :first_zero].reshape((first_zero,) + last_dims)

    return filtered_U, filtered_S, filtered_VT
