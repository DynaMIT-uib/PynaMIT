"""Backend-aware linear-operator wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeAlias

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import aslinearoperator

from pynamit.math.tensor_chain import TensorChain
from pynamit.utils import asarray, get_array_module, use_jax

MatrixShape: TypeAlias = tuple[int, int]
VectorizedMapFunc: TypeAlias = Callable[[Any], Any]


@dataclass(frozen=True)
class LinearMap:
    """Backend-agnostic linear map on flattened vectors."""

    shape: MatrixShape
    dtype: Any
    _matvec: VectorizedMapFunc
    _rmatvec: VectorizedMapFunc
    _matmat: Optional[VectorizedMapFunc] = None
    _rmatmat: Optional[VectorizedMapFunc] = None
    _to_dense: Optional[Callable[[], np.ndarray]] = None
    _normal_matrix_diag: Optional[Callable[[], np.ndarray]] = None
    source: Any = None
    domain_space: Optional[str] = None
    codomain_space: Optional[str] = None

    @property
    def ndim(self) -> int:
        """Dimensionality of the linear map."""
        return 2

    def matvec(self, x: Any) -> Any:
        """Apply this map to one flattened vector."""
        return self._matvec(x)

    def rmatvec(self, y: Any) -> Any:
        """Apply the adjoint map to one flattened vector."""
        return self._rmatvec(y)

    def matmat(self, x_block: Any) -> Any:
        """Apply this map to a block of column vectors."""
        if self._matmat is not None:
            return self._matmat(x_block)
        xp = get_array_module(x_block)
        x_arr = xp.asarray(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint map to a block of column vectors."""
        if self._rmatmat is not None:
            return self._rmatmat(y_block)
        xp = get_array_module(y_block)
        y_arr = xp.asarray(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def to_dense(self) -> np.ndarray:
        """Return a dense NumPy matrix representation if available."""
        if self._to_dense is None:
            raise ValueError("Dense representation not available for this LinearMap.")
        return np.asarray(self._to_dense())

    def with_spaces(
        self, *, domain_space: Optional[str] = None, codomain_space: Optional[str] = None
    ) -> "LinearMap":
        """Return a copy with explicit domain/codomain metadata."""
        return LinearMap(
            shape=self.shape,
            dtype=self.dtype,
            _matvec=self._matvec,
            _rmatvec=self._rmatvec,
            _matmat=self._matmat,
            _rmatmat=self._rmatmat,
            _to_dense=self._to_dense,
            _normal_matrix_diag=self._normal_matrix_diag,
            source=self.source,
            domain_space=domain_space,
            codomain_space=codomain_space,
        )

    def normal_matrix_diag(self) -> np.ndarray:
        """Compute ``diag(A* A)`` for this map."""
        if self._normal_matrix_diag is not None:
            return np.asarray(self._normal_matrix_diag())
        try:
            dense = self.to_dense()
            return np.sum(np.abs(dense) ** 2, axis=0)
        except ValueError:
            n_cols = self.shape[1]
            dtype = np.result_type(self.dtype, np.float64)
            diag = np.zeros(n_cols, dtype=dtype)
            block_size = min(32, max(1, n_cols))
            block = np.zeros((n_cols, block_size), dtype=dtype)
            for start in range(0, n_cols, block_size):
                stop = min(n_cols, start + block_size)
                cols = stop - start
                block[:, :cols] = 0
                block[start:stop, :cols] = np.eye(cols, dtype=dtype)
                res = np.asarray(self.matmat(block[:, :cols]))
                diag[start:stop] = np.sum(np.abs(res) ** 2, axis=0).real
            return diag

    def __matmul__(self, other: Any) -> Any:
        """Apply to arrays or compose with another operator."""
        if not scipy.sparse.issparse(other) and not _looks_like_operator(other):
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
        if (
            self.domain_space is not None
            and other_map.codomain_space is not None
            and self.domain_space != other_map.codomain_space
        ):
            raise ValueError(
                "Space mismatch for composition: "
                f"{self.codomain_space!r} <- {self.domain_space!r} @ "
                f"{other_map.codomain_space!r} <- {other_map.domain_space!r}"
            )

        def matvec(x: Any) -> Any:
            return self.matvec(other_map.matvec(x))

        def rmatvec(y: Any) -> Any:
            return other_map.rmatvec(self.rmatvec(y))

        def matmat(x: Any) -> Any:
            return self.matmat(other_map.matmat(x))

        def rmatmat(y: Any) -> Any:
            return other_map.rmatmat(self.rmatmat(y))

        def to_dense() -> np.ndarray:
            return self.matmat(other_map.to_dense())

        return LinearMap(
            shape=(self.shape[0], other_map.shape[1]),
            dtype=np.promote_types(self.dtype, other_map.dtype),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=to_dense if other_map._to_dense is not None else None,
            source=(self, other_map),
            domain_space=other_map.domain_space,
            codomain_space=self.codomain_space,
        )

    def __add__(self, other: Any) -> "LinearMap":
        """Add two linear maps."""
        other_map = as_linear_map(other)
        if self.shape != other_map.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} + {other_map.shape}")
        if (
            self.domain_space is not None
            and other_map.domain_space is not None
            and self.domain_space != other_map.domain_space
        ) or (
            self.codomain_space is not None
            and other_map.codomain_space is not None
            and self.codomain_space != other_map.codomain_space
        ):
            raise ValueError(
                "Space mismatch for addition: "
                f"{self.codomain_space!r} <- {self.domain_space!r} + "
                f"{other_map.codomain_space!r} <- {other_map.domain_space!r}"
            )

        def matvec(x: Any) -> Any:
            return self.matvec(x) + other_map.matvec(x)

        def rmatvec(y: Any) -> Any:
            return self.rmatvec(y) + other_map.rmatvec(y)

        def matmat(x: Any) -> Any:
            return self.matmat(x) + other_map.matmat(x)

        def rmatmat(y: Any) -> Any:
            return self.rmatmat(y) + other_map.rmatmat(y)

        def to_dense() -> np.ndarray:
            return self.to_dense() + other_map.to_dense()

        has_dense = self._to_dense is not None and other_map._to_dense is not None
        return LinearMap(
            shape=self.shape,
            dtype=np.promote_types(self.dtype, other_map.dtype),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=to_dense if has_dense else None,
            source=(self, other_map),
            domain_space=self.domain_space or other_map.domain_space,
            codomain_space=self.codomain_space or other_map.codomain_space,
        )

    def __sub__(self, other: Any) -> "LinearMap":
        """Subtract another linear map."""
        return self + (-1.0 * as_linear_map(other))

    def __mul__(self, other: Any) -> "LinearMap":
        """Scale this linear map."""
        if not np.isscalar(other):
            return NotImplemented
        scalar = other

        def matvec(x: Any) -> Any:
            return self.matvec(x) * scalar

        def rmatvec(y: Any) -> Any:
            return self.rmatvec(y) * np.conj(scalar)

        def matmat(x: Any) -> Any:
            return self.matmat(x) * scalar

        def rmatmat(y: Any) -> Any:
            return self.rmatmat(y) * np.conj(scalar)

        def to_dense() -> np.ndarray:
            return self.to_dense() * scalar

        return LinearMap(
            shape=self.shape,
            dtype=np.result_type(self.dtype, scalar),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=to_dense if self._to_dense is not None else None,
            source=(self, scalar),
            domain_space=self.domain_space,
            codomain_space=self.codomain_space,
        )

    def __rmul__(self, other: Any) -> "LinearMap":
        """Scale this linear map."""
        return self.__mul__(other)

    def __neg__(self) -> "LinearMap":
        """Negate this linear map."""
        return -1.0 * self

    def as_linear_operator(self) -> ScipyLinearOperator:
        """Return a SciPy ``LinearOperator`` view of this map."""
        if isinstance(self.source, ScipyLinearOperator) and self.source.shape == self.shape:
            return self.source

        def matvec(vec: np.ndarray) -> np.ndarray:
            return np.asarray(self.matvec(vec))

        def rmatvec(vec: np.ndarray) -> np.ndarray:
            return np.asarray(self.rmatvec(vec))

        def matmat(block: np.ndarray) -> np.ndarray:
            return np.asarray(self.matmat(block))

        def rmatmat(block: np.ndarray) -> np.ndarray:
            return np.asarray(self.rmatmat(block))

        return ScipyLinearOperator(
            self.shape,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=self.dtype,
        )


def _looks_like_operator(value: Any) -> bool:
    return isinstance(value, (LinearMap, TensorChain, ScipyLinearOperator)) or hasattr(
        value, "matvec"
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
        return xp.matmul(mat_arr, vec_arr)

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(mat_backend, vec)
        mat_arr = xp.asarray(mat_backend)
        vec_arr = xp.asarray(vec).reshape(shape[0])
        return xp.matmul(xp.swapaxes(xp.conjugate(mat_arr), -2, -1), vec_arr)

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

    def normal_matrix_diag() -> np.ndarray:
        return np.sum(np.abs(np.asarray(mat_backend)) ** 2, axis=0)

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense,
        _normal_matrix_diag=normal_matrix_diag,
        source=mat_backend,
    )


def diagonal_linear_map(diag_values: Any) -> LinearMap:
    """Return a map backed by a diagonal vector."""
    diag_backend = asarray(diag_values).reshape(-1)
    size = int(diag_backend.size)
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

    def normal_matrix_diag() -> np.ndarray:
        return np.abs(np.asarray(diag_backend).reshape(-1)) ** 2

    return LinearMap(
        shape=(size, size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense,
        _normal_matrix_diag=normal_matrix_diag,
        source=diag_backend,
    )


def _linear_map_from_linear_operator(op: ScipyLinearOperator) -> LinearMap:
    shape = tuple(int(dim) for dim in op.shape)
    dtype = op.dtype or np.float64

    def matvec(vec: Any) -> Any:
        return op.matvec(np.asarray(vec).reshape(shape[1]))

    def rmatvec(vec: Any) -> Any:
        return op.rmatvec(np.asarray(vec).reshape(shape[0]))

    def matmat(block: Any) -> Any:
        return op.matmat(np.asarray(block).reshape(shape[1], -1))

    def rmatmat(block: Any) -> Any:
        return op.rmatmat(np.asarray(block).reshape(shape[0], -1))

    def to_dense() -> np.ndarray:
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
    shape = tuple(int(dim) for dim in op.shape)
    dtype = op.dtype

    def matvec(vec: Any) -> Any:
        return op @ asarray(vec).reshape(shape[1])

    def rmatvec(vec: Any) -> Any:
        return op.T @ asarray(vec).reshape(shape[0])

    def matmat(block: Any) -> Any:
        return op @ asarray(block).reshape(shape[1], -1)

    def rmatmat(block: Any) -> Any:
        return op.T @ asarray(block).reshape(shape[0], -1)

    def to_dense() -> np.ndarray:
        return np.asarray(op.todense())

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
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    """Convert supported operator types into a ``LinearMap``."""
    if isinstance(op, LinearMap):
        return op
    if isinstance(op, TensorChain):
        return op.to_linear_map()

    op_type = str(type(op))
    is_jax_sparse = "jax.experimental.sparse" in op_type or (
        "jax" in op_type and hasattr(op, "todense") and hasattr(op, "indices")
    )
    if is_jax_sparse:
        return _linear_map_from_jax_sparse(op)

    if isinstance(op, ScipyLinearOperator):
        chain = getattr(op, "_tensor_chain", None)
        if isinstance(chain, TensorChain):
            return chain.to_linear_map()
        return _linear_map_from_linear_operator(op)

    if scipy.sparse.issparse(op):
        if use_jax():
            try:
                from jax.experimental.sparse import BCOO

                return _linear_map_from_jax_sparse(BCOO.from_scipy_sparse(op))
            except Exception:
                pass
        return _linear_map_from_linear_operator(aslinearoperator(op))

    try:
        arr = asarray(op)
    except Exception as exc:
        message = f"Unsupported operator type '{type(op)}' for LinearMap conversion."
        raise TypeError(message) from exc

    if arr.ndim == 1:
        size = int(arr.size)
        if input_shape is not None and math.prod(input_shape) != size:
            raise ValueError(f"1-D operator size {size} mismatch with input {input_shape}.")
        if output_shape is not None and math.prod(output_shape) != size:
            raise ValueError(f"1-D operator size {size} mismatch with output {output_shape}.")
        return diagonal_linear_map(arr)

    if arr.ndim < 2:
        raise ValueError("Operators must be at least 1-D.")

    if arr.ndim == 2 and input_shape is None and output_shape is None:
        return _linear_map_from_dense(arr)

    inferred_input = input_shape or (arr.shape[-1],)
    flat_in = math.prod(inferred_input)
    total_elements = int(arr.size)
    if output_shape is None:
        flat_out = total_elements // flat_in
        if flat_out * flat_in != total_elements:
            raise ValueError(
                f"Operator with shape {arr.shape} incompatible with inferred input "
                f"{inferred_input}."
            )
    else:
        flat_out = math.prod(output_shape)
        if flat_out * flat_in != total_elements:
            raise ValueError(
                f"Operator with shape {arr.shape} incompatible with provided shapes "
                f"{output_shape} -> {inferred_input}."
            )
    return _linear_map_from_dense(arr.reshape(flat_out, flat_in))


def block_linear_map(blocks: list[list[Any]]) -> LinearMap:
    """Create a ``LinearMap`` from a block matrix of operators."""
    lm_blocks = [[as_linear_map(block) for block in row] for row in blocks]
    if not lm_blocks or not lm_blocks[0]:
        raise ValueError("blocks must contain at least one row and one column.")

    num_rows = len(lm_blocks)
    num_cols = len(lm_blocks[0])
    if any(len(row) != num_cols for row in lm_blocks):
        raise ValueError("All block rows must have the same number of columns.")

    row_heights = [row[0].shape[0] for row in lm_blocks]
    col_widths = [lm_blocks[0][col].shape[1] for col in range(num_cols)]
    for row, height in zip(lm_blocks, row_heights):
        if any(block.shape[0] != height for block in row):
            raise ValueError("All blocks in a block row must have the same row count.")
    for col, width in enumerate(col_widths):
        if any(row[col].shape[1] != width for row in lm_blocks):
            raise ValueError("All blocks in a block column must have the same column count.")

    total_height = sum(row_heights)
    total_width = sum(col_widths)
    dtype = lm_blocks[0][0].dtype
    for row in lm_blocks:
        for block in row:
            dtype = np.promote_types(dtype, block.dtype)

    def _split_columns(block: Any) -> list[Any]:
        parts = []
        start = 0
        for width in col_widths:
            parts.append(block[start : start + width, :])
            start += width
        return parts

    def _split_rows(block: Any) -> list[Any]:
        parts = []
        start = 0
        for height in row_heights:
            parts.append(block[start : start + height, :])
            start += height
        return parts

    def matmat(block: Any) -> Any:
        xp = get_array_module(block)
        block_arr = xp.asarray(block).reshape(total_width, -1)
        x_parts = _split_columns(block_arr)

        row_outputs = []
        for row in lm_blocks:
            row_sum = None
            for block, x_part in zip(row, x_parts):
                term = block.matmat(x_part)
                row_sum = term if row_sum is None else row_sum + term
            row_outputs.append(row_sum)
        return xp.concatenate(row_outputs, axis=0)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block)
        block_arr = xp.asarray(block).reshape(total_height, -1)
        y_parts = _split_rows(block_arr)

        col_outputs = []
        for col in range(num_cols):
            col_sum = None
            for row in range(num_rows):
                term = lm_blocks[row][col].rmatmat(y_parts[row])
                col_sum = term if col_sum is None else col_sum + term
            col_outputs.append(col_sum)
        return xp.concatenate(col_outputs, axis=0)

    def matvec(x: Any) -> Any:
        return matmat(x).reshape(total_height)

    def rmatvec(y: Any) -> Any:
        return rmatmat(y).reshape(total_width)

    def to_dense() -> np.ndarray:
        dense_rows = []
        for row in lm_blocks:
            dense_rows.append(np.concatenate([block.to_dense() for block in row], axis=1))
        return np.concatenate(dense_rows, axis=0)

    has_dense = all(block._to_dense is not None for row in lm_blocks for block in row)
    return LinearMap(
        shape=(total_height, total_width),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _to_dense=to_dense if has_dense else None,
        source=lm_blocks,
    )
