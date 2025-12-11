"""Linear algebra utilities module."""

from __future__ import annotations

import math
from typing import Any

from .backend import asarray, xp


def tensor_pinv(
    A: Any, n_leading_flattened: int = 2, rtol: float = 1e-15, hermitian: bool = False
) -> Any:
    """Moore-Penrose pseudoinverse of a tensor."""
    A_arr = asarray(A)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    A_flat = A_arr.reshape((flat_first, flat_last))
    pinv_kwargs = {"hermitian": hermitian, "rtol": rtol}
    A_pinv = xp.linalg.pinv(A_flat, **pinv_kwargs)
    return A_pinv.reshape(last_dims + first_dims)
