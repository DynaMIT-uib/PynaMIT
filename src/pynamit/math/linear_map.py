"""Unified linear-operator wrapper used across backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeAlias

import numpy as np
import scipy
import scipy.sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import aslinearoperator

from pynamit.utils import asarray, get_array_module
from pynamit.math.tensor_chain import TensorChain


MatrixShape: TypeAlias = tuple[int, int]
VectorizedMapFunc: TypeAlias = Callable[[Any], Any]


@dataclass(frozen=True)
class LinearMap:
    """Backend-agnostic linear map on flattened vectors."""

    shape: MatrixShape
    dtype: Any

    @property
    def ndim(self) -> int:
        """Dimensionality of the linear map (always 2)."""
        return 2
    _matvec: VectorizedMapFunc
    _rmatvec: VectorizedMapFunc
    _matmat: Optional[VectorizedMapFunc] = None
    _rmatmat: Optional[VectorizedMapFunc] = None
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

    def __matmul__(self, other: Any) -> Any:
        """Matrix-vector product (if other is array) or Composition (if other is manager)."""
        # Allow array operands so the class behaves like a matrix in most call sites.
        if isinstance(other, (np.ndarray, list)) or (
            hasattr(other, "shape") and not hasattr(other, "matvec")
        ):
            arr = asarray(other)
            if arr.ndim == 1:
                return self.matvec(arr)
            if arr.ndim == 2:
                return self.matmat(arr)

        other_map = as_linear_map(other)
        if self.shape[1] != other_map.shape[0]:
            raise ValueError(
                f"Dimension mismatch for composition: {self.shape} @ {other_map.shape}"
            )

        flat_out = self.shape[0]
        flat_in = other_map.shape[1]
        dtype = np.promote_types(self.dtype, other_map.dtype)

        def matvec(x):
            return self.matvec(other_map.matvec(x))

        def rmatvec(y):
            return other_map.rmatvec(self.rmatvec(y))

        # Better matmat: use optimized matmats if available
        def matmat_opt(x):
            return self.matmat(other_map.matmat(x))

        def rmatmat_opt(y):
            return other_map.rmatmat(self.rmatmat(y))

        def to_dense():
            # Dense composition: A @ B. Easiest route is applying A to the dense columns of B.
            if other_map._to_dense is not None:
                return self.matmat(other_map.to_dense())
            raise ValueError("Cannot densify composition without dense right-hand side.")

        return LinearMap(
            shape=(flat_out, flat_in),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat_opt,
            _rmatmat=rmatmat_opt,
            _to_dense=to_dense,
            source=(self, other_map),
        )

    def __add__(self, other: Any) -> LinearMap:
        """Add two linear maps."""
        other_map = as_linear_map(other)
        if self.shape != other_map.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} + {other_map.shape}")

        def matvec(x):
            return self.matvec(x) + other_map.matvec(x)

        def rmatvec(y):
            return self.rmatvec(y) + other_map.rmatvec(y)

        def matmat(x):
            return self.matmat(x) + other_map.matmat(x)

        def rmatmat(y):
            return self.rmatmat(y) + other_map.rmatmat(y)

        def to_dense():
            return self.to_dense() + other_map.to_dense()

        return LinearMap(
            shape=self.shape,
            dtype=np.promote_types(self.dtype, other_map.dtype),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=to_dense if self._to_dense is not None and other_map._to_dense is not None else None,
            source=(self, other_map),
        )

    def __mul__(self, other: Any) -> LinearMap:
        """Scalar multiplication."""
        if not np.isscalar(other):
            return NotImplemented
        scalar = other

        def matvec(x):
            return self.matvec(x) * scalar

        def rmatvec(y):
            return self.rmatvec(y) * np.conj(scalar)

        def matmat(x):
            return self.matmat(x) * scalar

        def rmatmat(y):
            return self.rmatmat(y) * np.conj(scalar)

        def to_dense():
            return self.to_dense() * scalar

        return LinearMap(
            shape=self.shape,
            dtype=np.result_type(self.dtype, scalar),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=to_dense if self._to_dense else None,
            source=(self, scalar),
        )

    def __rmul__(self, other: Any) -> LinearMap:
        return self.__mul__(other)
    
    def as_linear_operator(self) -> ScipyLinearOperator:
        """Return a SciPy ``LinearOperator`` view of this map."""
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


def _linear_map_from_jax_sparse(op: Any) -> LinearMap:
    shape = op.shape
    dtype = op.dtype

    def matvec(vec: Any) -> Any:
        # BCOO @ vec (dense 1D) -> dense 1D
        # Explicit reshape to ensure compatibility
        vec_arr = asarray(vec).reshape(shape[1])
        return op @ vec_arr

    def rmatvec(vec: Any) -> Any:
        vec_arr = asarray(vec).reshape(shape[0])
        return op.T @ vec_arr

    def matmat(block: Any) -> Any:
        block_arr = asarray(block).reshape(shape[1], -1)
        return op @ block_arr

    def rmatmat(block: Any) -> Any:
        block_arr = asarray(block).reshape(shape[0], -1)
        return op.T @ block_arr

    def to_dense() -> np.ndarray:
        return op.todense()

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
        return op.to_linear_map()
    
    # Check for JAX sparse matrix (BCOO/BCSR)
    # Generic check to avoid importing jax if not present
    op_type = str(type(op))
    if "jax.experimental.sparse" in op_type or ("jax" in op_type and hasattr(op, "todense") and hasattr(op, "indices")):
        return _linear_map_from_jax_sparse(op)

    if isinstance(op, ScipyLinearOperator):
        chain = getattr(op, "_tensor_chain", None)
        if isinstance(chain, TensorChain):
            return chain.to_linear_map()
        return _linear_map_from_linear_operator(op)
    if scipy.sparse.issparse(op):
         # If using JAX, convert Scipy sparse to BCOO automatically
         from pynamit.utils import use_jax
         if use_jax():
             try:
                 from jax.experimental.sparse import BCOO
                 if hasattr(op, "tocoo"):
                     # Convert to COO first (efficient for BCOO conversion)
                     return _linear_map_from_jax_sparse(BCOO.from_scipy_sparse(op))
             except ImportError:
                 pass
         
         lin_op = aslinearoperator(op)
         return _linear_map_from_linear_operator(lin_op)
    # Attempt to treat as a dense array/matrix
    try:
        arr = asarray(op)
    except Exception:
        arr = None

    if arr is not None:
        if arr.ndim == 1:
            size = arr.size
            if input_shape is not None:
                flat_in = math.prod(input_shape)
                if flat_in != size:
                    raise ValueError(f"1D Operator size {size} mismatch with input {input_shape}")
            if output_shape is not None:
                flat_out = math.prod(output_shape)
                if flat_out != size:
                    raise ValueError(f"1D Operator size {size} mismatch with output {output_shape}")
            return diagonal_linear_map(arr)
            
        if arr.ndim >= 2:
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


def block_linear_map(blocks: list[list[Any]]) -> LinearMap:
    """
    Create a ``LinearMap`` from a block matrix of operators.
    
    blocks[i][j] is the operator mapping from component j to component i.
    """
    # Convert all blocks to LinearMaps
    lm_blocks = [[as_linear_map(b) for b in row] for row in blocks]
    
    num_rows = len(lm_blocks)
    num_cols = len(lm_blocks[0])
    
    # Check consistency of block shapes
    row_heights = [row[0].shape[0] for row in lm_blocks]
    col_widths = [lm_blocks[0][j].shape[1] for j in range(num_cols)]
    
    total_height = sum(row_heights)
    total_width = sum(col_widths)
    
    dtype = lm_blocks[0][0].dtype
    for row in lm_blocks:
        for block in row:
            dtype = np.promote_types(dtype, block.dtype)
    
    def matvec(x):
        x_mod = get_array_module(x)
        # x: (total_width)
        # Partition x into column components
        x_parts = []
        curr = 0
        for w in col_widths:
            x_parts.append(x[curr:curr+w])
            curr += w
            
        # Cumulative sum for each row
        res_parts = []
        for i in range(num_rows):
            row_sum = None
            for j in range(num_cols):
                term = lm_blocks[i][j].matvec(x_parts[j])
                row_sum = term if row_sum is None else row_sum + term
            res_parts.append(row_sum)
            
        return x_mod.concatenate(res_parts, axis=0)

    def rmatvec(y):
        y_mod = get_array_module(y)
        # y: (total_height)
        y_parts = []
        curr = 0
        for h in row_heights:
            y_parts.append(y[curr:curr+h])
            curr += h
            
        res_parts = []
        for j in range(num_cols):
            col_sum = None
            for i in range(num_rows):
                term = lm_blocks[i][j].rmatvec(y_parts[i])
                col_sum = term if col_sum is None else col_sum + term
            res_parts.append(col_sum)
            
        return y_mod.concatenate(res_parts, axis=0)

    def to_dense():
        rows = []
        for i in range(num_rows):
            row_dense = np.concatenate([np.asarray(b.to_dense()) for b in lm_blocks[i]], axis=1)
            rows.append(row_dense)
        return np.concatenate(rows, axis=0)

    return LinearMap(
        shape=(total_height, total_width),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _to_dense=to_dense if all(b._to_dense is not None for row in lm_blocks for b in row) else None,
        source=lm_blocks,
    )
