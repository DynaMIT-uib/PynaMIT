"""Unified linear-operator wrapper used across backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, TypeAlias

import numpy as np
import scipy
import scipy.sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import aslinearoperator

from pynamit.utils import asarray, get_array_module
from pynamit.math.tensor_chain import TensorChain


Shape: TypeAlias = Tuple[int, int]
MapFunc: TypeAlias = Callable[[Any], Any]


@dataclass(frozen=True)
class LinearMap:
    """Backend-agnostic linear map on flattened vectors."""

    shape: Shape
    dtype: Any
    _matvec: MapFunc
    _rmatvec: MapFunc
    _matmat: Optional[MapFunc] = None
    _rmatmat: Optional[MapFunc] = None
    _to_dense: Optional[Callable[[], np.ndarray]] = None
    source: Any = None

    def matvec(self, x: Any) -> Any:
        return self._matvec(x)

    def rmatvec(self, y: Any) -> Any:
        return self._rmatvec(y)

    def matmat(self, x_block: Any) -> Any:
        if self._matmat is not None:
            return self._matmat(x_block)
        xp = get_array_module(x_block)
        if x_block.ndim == 1:
            return self._matvec(x_block)
        cols = x_block.shape[1]
        outputs = [self._matvec(x_block[:, i]) for i in range(cols)]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        if self._rmatmat is not None:
            return self._rmatmat(y_block)
        xp = get_array_module(y_block)
        if y_block.ndim == 1:
            return self._rmatvec(y_block)
        cols = y_block.shape[1]
        outputs = [self._rmatvec(y_block[:, i]) for i in range(cols)]
        return xp.stack(outputs, axis=1)

    def to_dense(self) -> np.ndarray:
        if self._to_dense is None:
            raise ValueError("Dense representation not available for this LinearMap.")
        return self._to_dense()

    def as_linear_operator(self) -> ScipyLinearOperator:
        if isinstance(self.source, ScipyLinearOperator) and self.source.shape == self.shape:
            return self.source

        def matvec_np(vec: np.ndarray) -> np.ndarray:
            return np.asarray(self._matvec(vec))

        def rmatvec_np(vec: np.ndarray) -> np.ndarray:
            return np.asarray(self._rmatvec(vec))

        def matmat_np(block: np.ndarray) -> np.ndarray:
            return np.asarray(self.matmat(block))

        return ScipyLinearOperator(
            self.shape, matvec=matvec_np, rmatvec=rmatvec_np, matmat=matmat_np, dtype=self.dtype
        )


def _linear_map_from_tensor_chain(chain: TensorChain) -> LinearMap:
    flat_out = math.prod(chain.output_shape)
    flat_in = math.prod(chain.input_shape)

    def matvec(vec: Any) -> Any:
        arr = asarray(vec)
        return chain.matvec(arr)

    def rmatvec(vec: Any) -> Any:
        arr = asarray(vec)
        return chain.rmatvec(arr)

    def matmat(block: Any) -> Any:
        return chain.matmat(block)

    def rmatmat(block: Any) -> Any:
        return chain.rmatmat(block)

    def to_dense() -> np.ndarray:
        return chain.to_dense()

    return LinearMap(
        shape=(flat_out, flat_in),
        dtype=chain.dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense,
        source=chain,
    )


def _linear_map_from_dense(matrix: Any) -> LinearMap:
    mat_backend = asarray(matrix)
    if mat_backend.ndim != 2:
        raise ValueError("Dense operators must be 2-D arrays.")
    shape = tuple(int(dim) for dim in mat_backend.shape)
    dtype = getattr(mat_backend, "dtype", np.asarray(mat_backend).dtype)

    def matvec(vec: Any) -> Any:
        xp = get_array_module(mat_backend, vec)
        mat_arr = xp.asarray(mat_backend)
        vec_arr = xp.asarray(vec).reshape(shape[1])
        return xp.tensordot(mat_arr, vec_arr, axes=1)

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(mat_backend, vec)
        mat_arr = xp.asarray(mat_backend)
        vec_arr = xp.asarray(vec).reshape(shape[0])
        return xp.tensordot(xp.conjugate(mat_arr), vec_arr, axes=([0], [0]))

    def matmat(block: Any) -> Any:
        xp = get_array_module(mat_backend, block)
        mat_arr = xp.asarray(mat_backend)
        block_arr = xp.asarray(block).reshape(shape[1], -1)
        return xp.matmul(mat_arr, block_arr)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(mat_backend, block)
        mat_arr = xp.asarray(mat_backend)
        block_arr = xp.asarray(block).reshape(shape[0], -1)
        adjoint = xp.swapaxes(xp.conjugate(mat_arr), -2, -1)
        return xp.matmul(adjoint, block_arr)

    def to_dense() -> np.ndarray:
        return np.asarray(mat_backend)

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense,
        source=mat_backend,
    )


def diagonal_linear_map(diag_values: Any) -> LinearMap:
    diag_backend = asarray(diag_values).reshape(-1)
    size = diag_backend.size
    dtype = getattr(diag_backend, "dtype", np.asarray(diag_backend).dtype)

    def matvec(vec: Any) -> Any:
        xp = get_array_module(diag_backend, vec)
        diag_arr = xp.asarray(diag_backend)
        vec_arr = xp.asarray(vec).reshape(size)
        return diag_arr * vec_arr

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(diag_backend, vec)
        diag_arr = xp.asarray(diag_backend)
        vec_arr = xp.asarray(vec).reshape(size)
        return xp.conjugate(diag_arr) * vec_arr

    def matmat(block: Any) -> Any:
        xp = get_array_module(diag_backend, block)
        diag_arr = xp.asarray(diag_backend).reshape(size, 1)
        block_arr = xp.asarray(block).reshape(size, -1)
        return diag_arr * block_arr

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(diag_backend, block)
        diag_arr = xp.asarray(diag_backend).reshape(size, 1)
        block_arr = xp.asarray(block).reshape(size, -1)
        return xp.conjugate(diag_arr) * block_arr

    def to_dense() -> np.ndarray:
        return np.diag(np.asarray(diag_backend))

    return LinearMap(
        shape=(size, size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense,
        source=np.asarray(diag_backend),
    )


def _linear_map_from_linear_operator(op: ScipyLinearOperator) -> LinearMap:
    shape = op.shape
    dtype = op.dtype

    def matvec(vec: Any) -> Any:
        vec_np = np.asarray(vec).reshape(shape[1])
        return op.matvec(vec_np)

    def rmatvec(vec: Any) -> Any:
        vec_np = np.asarray(vec).reshape(shape[0])
        return op.rmatvec(vec_np)

    def matmat(block: Any) -> Any:
        try:
            return op.matmat(np.asarray(block).reshape(shape[1], -1))
        except AttributeError:
            block_np = np.asarray(block).reshape(shape[1], -1)
            cols = block_np.shape[1]
            res = [op.matvec(block_np[:, i]) for i in range(cols)]
            return np.stack(res, axis=1)

    def rmatmat(block: Any) -> Any:
        try:
            return op.rmatmat(np.asarray(block).reshape(shape[0], -1))
        except AttributeError:
            block_np = np.asarray(block).reshape(shape[0], -1)
            cols = block_np.shape[1]
            res = [op.rmatvec(block_np[:, i]) for i in range(cols)]
            return np.stack(res, axis=1)

    def to_dense() -> np.ndarray:
        # Check if source is sparse matrix
        if scipy.sparse.issparse(op):
            return op.toarray()
        eye = np.eye(shape[1], dtype=dtype)
        return np.asarray(op.matmat(eye))

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense,
        source=op,
    )


def as_linear_map(
    op: Any,
    input_shape: Optional[Tuple[int, ...]] = None,
    output_shape: Optional[Tuple[int, ...]] = None,
) -> LinearMap:
    """Convert supported operator types into a LinearMap."""
    if isinstance(op, LinearMap):
        return op
    if isinstance(op, TensorChain):
        return _linear_map_from_tensor_chain(op)
    if isinstance(op, ScipyLinearOperator):
        chain = getattr(op, "_tensor_chain", None)
        if isinstance(chain, TensorChain):
            return _linear_map_from_tensor_chain(chain)
        return _linear_map_from_linear_operator(op)
    if scipy.sparse.issparse(op):
         lin_op = aslinearoperator(op)
         return _linear_map_from_linear_operator(lin_op)
    # Attempt to treat as a dense array/matrix
    try:
        arr = asarray(op)
    except Exception:
        arr = None

    if arr is not None and arr.ndim >= 2:
        if arr.ndim == 2 and input_shape is None and output_shape is None:
            return _linear_map_from_dense(arr)
        inferred_input = input_shape
        if inferred_input is None:
            inferred_input = (arr.shape[-1],)
        flat_in = math.prod(inferred_input)
        total_elements = arr.size
        if output_shape is None:
            flat_out = total_elements // flat_in
            if flat_out * flat_in != total_elements:
                raise ValueError(
                    f"Operator with shape {arr.shape} incompatible with inferred input {inferred_input}."
                )
            inferred_output = (flat_out,)
        else:
            inferred_output = output_shape
            flat_out = math.prod(inferred_output)
            if flat_out * flat_in != total_elements:
                raise ValueError(
                    f"Operator with shape {arr.shape} incompatible with provided shapes "
                    f"{inferred_output} -> {inferred_input}."
                )
        reshaped = arr.reshape(flat_out, flat_in)
        return _linear_map_from_dense(reshaped)
    raise TypeError(f"Unsupported operator type '{type(op)}' for LinearMap conversion.")
