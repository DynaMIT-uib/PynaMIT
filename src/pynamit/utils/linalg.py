"""Linear algebra utilities module."""

from __future__ import annotations

import math
from typing import Any
import logging

from .backend import asarray, xp

logger = logging.getLogger(__name__)


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
    try:
        A_pinv = xp.linalg.pinv(A_flat, **pinv_kwargs)
        return A_pinv.reshape(last_dims + first_dims)
    except Exception as exc:
        # NumPy LAPACK SVD can occasionally fail to converge on ill-conditioned
        # matrices; fall back to SciPy's SVD driver for robustness.
        if getattr(xp, "__name__", "") not in ("numpy", "numpy.core.multiarray"):
            raise
        logger.warning("tensor_pinv: pinv failed (%s); falling back to SciPy SVD.", exc)
        import numpy as np
        import scipy.linalg as la

        A_np = np.asarray(A_flat)
        try:
            U, s, Vh = la.svd(A_np, full_matrices=False, lapack_driver="gesvd")
        except Exception:
            U, s, Vh = la.svd(A_np, full_matrices=False)

        if s.size == 0:
            A_pinv = np.zeros((A_np.shape[1], A_np.shape[0]), dtype=A_np.dtype)
        else:
            tol = float(rtol) * float(s[0]) if rtol is not None else 0.0
            s_inv = np.where(s > tol, 1.0 / s, 0.0)
            A_pinv = (Vh.T * s_inv) @ U.T
        return asarray(A_pinv).reshape(last_dims + first_dims)
