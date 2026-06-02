"""Backend-aware linear-operator wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional, TypeAlias

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from pynamit.math.backend import (
    JAX_AVAILABLE,
    asarray,
    block_until_ready,
    get_array_module,
    to_numpy,
    use_jax,
)

MatrixShape: TypeAlias = tuple[int, int]
VectorizedMapFunc: TypeAlias = Callable[[Any], Any]
DenseBackend: TypeAlias = Literal["numpy", "jax"]


def _array_module_for_dense_backend(backend: DenseBackend | None = None) -> Any:
    """Return the array module for explicit dense materialization."""
    if backend is None:
        return None
    if not isinstance(backend, str):
        raise TypeError("backend must be None, 'numpy', or 'jax'.")

    normalized = backend.strip().lower()
    if normalized == "numpy":
        return np
    if normalized == "jax":
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed; cannot materialize on JAX.")
        import jax.numpy as jnp

        return jnp
    raise ValueError(
        f"Unknown dense backend {backend!r}. "
        "Use None, 'numpy', or 'jax'."
    )


@dataclass(frozen=True)
class LinearMap:
    """Backend-agnostic linear map with flattened matrix operations."""

    shape: MatrixShape
    dtype: Any
    _matvec: VectorizedMapFunc = field(repr=False)
    _rmatvec: VectorizedMapFunc = field(repr=False)
    _matmat: Optional[VectorizedMapFunc] = field(default=None, repr=False)
    _rmatmat: Optional[VectorizedMapFunc] = field(default=None, repr=False)
    _dense_array_func: Optional[Callable[[Any], Any]] = field(
        default=None, repr=False
    )
    _diagonal_array_func: Optional[Callable[[Any], Any]] = field(
        default=None, repr=False
    )
    _normal_matrix_diag: Optional[Callable[[], np.ndarray]] = field(
        default=None, repr=False
    )
    _backend_context: tuple[Any, ...] = field(default=(), repr=False)
    output_shape: Optional[tuple[int, ...]] = None
    input_shape: Optional[tuple[int, ...]] = None
    _dense_cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate shaped metadata and fill flat defaults."""
        output_shape, input_shape = _map_shapes(
            self.shape, self.input_shape, self.output_shape
        )
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "input_shape", input_shape)

    @property
    def ndim(self) -> int:
        """Dimensionality of the linear map."""
        return 2

    @property
    def backend_context(self) -> tuple[Any, ...]:
        """Closed-over operands used for backend selection."""
        return self._backend_context

    def array_module(self, *operands: Any) -> Any:
        """Return the array module implied by operands and this map."""
        return get_array_module(*operands, *self._backend_context)

    @staticmethod
    def _dense_cache_key(xp: Any) -> str:
        """Return a stable cache key for one array module."""
        return getattr(xp, "__name__", repr(xp))

    def _cached_dense(self, xp: Any) -> Optional[Any]:
        """Return cached dense materialization for ``xp`` if present."""
        return self._dense_cache.get(self._dense_cache_key(xp))

    def _store_dense(self, xp: Any, dense: Any) -> None:
        """Store dense materialization for ``xp``."""
        self._dense_cache[self._dense_cache_key(xp)] = dense

    def matvec(self, x: Any) -> Any:
        """Apply this map to one flattened vector."""
        xp = self.array_module(x)
        dense = self._cached_dense(xp)
        if dense is not None:
            x_arr = xp.asarray(x).reshape(self.shape[1])
            return dense @ x_arr
        return self._matvec(x)

    def rmatvec(self, y: Any) -> Any:
        """Apply the adjoint map to one flattened vector."""
        xp = self.array_module(y)
        dense = self._cached_dense(xp)
        if dense is not None:
            y_arr = xp.asarray(y).reshape(self.shape[0])
            return xp.swapaxes(xp.conjugate(dense), -2, -1) @ y_arr
        return self._rmatvec(y)

    def matmat(self, x_block: Any) -> Any:
        """Apply this map to a block of column vectors."""
        xp = self.array_module(x_block)
        dense = self._cached_dense(xp)
        if dense is not None:
            x_arr = xp.asarray(x_block)
            if x_arr.ndim == 1:
                return dense @ x_arr.reshape(self.shape[1])
            return dense @ x_arr.reshape(self.shape[1], -1)
        if self._matmat is not None:
            return self._matmat(x_block)
        x_arr = xp.asarray(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint map to a block of column vectors."""
        xp = self.array_module(y_block)
        dense = self._cached_dense(xp)
        if dense is not None:
            y_arr = xp.asarray(y_block)
            adjoint = xp.swapaxes(xp.conjugate(dense), -2, -1)
            if y_arr.ndim == 1:
                return adjoint @ y_arr.reshape(self.shape[0])
            return adjoint @ y_arr.reshape(self.shape[0], -1)
        if self._rmatmat is not None:
            return self._rmatmat(y_block)
        y_arr = xp.asarray(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def _dense_array(self, xp: Any = None) -> Any:
        """Materialize this map as a dense array on ``xp``."""
        xp = self.array_module() if xp is None else xp
        cached = self._cached_dense(xp)
        if cached is not None:
            return cached

        if self._dense_array_func is not None:
            dense = self._dense_array_func(xp)
            self._store_dense(xp, dense)
            return dense

        eye_dtype = np.result_type(self.dtype, np.float64)
        eye = xp.eye(self.shape[1], dtype=eye_dtype)
        dense = self.matmat(eye)
        dense = np.asarray(dense) if xp is np else xp.asarray(dense)
        self._store_dense(xp, dense)
        return dense

    def dense(self, *, backend: DenseBackend | None = None) -> Any:
        """Materialize this map as a dense array on one backend."""
        xp = _array_module_for_dense_backend(backend)
        return block_until_ready(self._dense_array(xp))

    def diagonal(self, *, backend: DenseBackend | None = None) -> Any:
        """Return diagonal scale values for a diagonal map."""
        xp = _array_module_for_dense_backend(backend)
        return block_until_ready(self._diagonal_array(xp))

    def _diagonal_array(self, xp: Any = None) -> Any:
        """Return exact diagonal scale values on ``xp``."""
        xp = self.array_module() if xp is None else xp
        if self._diagonal_array_func is not None:
            return self._diagonal_array_func(xp)
        if self.shape[0] != self.shape[1]:
            raise ValueError("Diagonal values require a square operator.")

        dense = np.asarray(self.dense(backend="numpy"))
        diagonal = np.diag(dense)
        if not np.allclose(dense, np.diag(diagonal), rtol=0.0, atol=0.0):
            raise ValueError("Operator is not diagonal.")
        return xp.asarray(diagonal)

    def normal_matrix_diag(self) -> np.ndarray:
        """Compute ``diag(A* A)`` for this map."""
        if self._normal_matrix_diag is not None:
            return np.asarray(self._normal_matrix_diag())
        if self._diagonal_array_func is not None:
            return np.abs(np.asarray(self.diagonal(backend="numpy"))) ** 2
        if (
            self._cached_dense(np) is not None
            or self._dense_array_func is not None
        ):
            dense = np.asarray(self.dense(backend="numpy"))
            return np.sum(np.abs(dense) ** 2, axis=0)
        return _normal_matrix_diag_from_matmat(self.shape, self.dtype, self.matmat)

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

        def matvec(x: Any) -> Any:
            return self.matvec(other_map.matvec(x))

        def rmatvec(y: Any) -> Any:
            return other_map.rmatvec(self.rmatvec(y))

        def matmat(x: Any) -> Any:
            return self.matmat(other_map.matmat(x))

        def rmatmat(y: Any) -> Any:
            return other_map.rmatmat(self.rmatmat(y))

        def dense_array(xp: Any) -> Any:
            return xp.asarray(self.matmat(other_map._dense_array(xp)))

        dtype = np.promote_types(self.dtype, other_map.dtype)

        def normal_matrix_diag() -> np.ndarray:
            return _normal_matrix_diag_from_matmat(
                (self.shape[0], other_map.shape[1]), dtype, matmat
            )

        return LinearMap(
            shape=(self.shape[0], other_map.shape[1]),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _dense_array_func=dense_array,
            _normal_matrix_diag=normal_matrix_diag,
            _backend_context=self._backend_context + other_map._backend_context,
            output_shape=self.output_shape,
            input_shape=other_map.input_shape,
        )

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

        def dense_array(xp: Any) -> Any:
            return self._dense_array(xp) * scalar

        def normal_matrix_diag() -> np.ndarray:
            return np.abs(scalar) ** 2 * self.normal_matrix_diag()

        def diagonal_array(xp: Any) -> Any:
            return self._diagonal_array(xp) * scalar

        return LinearMap(
            shape=self.shape,
            dtype=np.result_type(self.dtype, scalar),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _dense_array_func=dense_array,
            _diagonal_array_func=(
                diagonal_array if self._diagonal_array_func is not None else None
            ),
            _normal_matrix_diag=normal_matrix_diag,
            _backend_context=self._backend_context,
            output_shape=self.output_shape,
            input_shape=self.input_shape,
        )

    def __rmul__(self, other: Any) -> "LinearMap":
        """Scale this linear map."""
        return self.__mul__(other)

    def __neg__(self) -> "LinearMap":
        """Negate this linear map."""
        return -1.0 * self

    def as_linear_operator(self) -> ScipyLinearOperator:
        """Return a SciPy ``LinearOperator`` view of this map."""

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
    return isinstance(value, (LinearMap, ScipyLinearOperator)) or hasattr(value, "matvec")


def _runtime_array_module(*values: Any) -> Any:
    """Select JAX only when an operand is already a JAX array."""
    if any("jax" in type(value).__module__ for value in values):
        return get_array_module(*values)
    return np


def _normal_matrix_diag_from_matmat(
    shape: MatrixShape, dtype: Any, matmat: Callable[[Any], Any]
) -> np.ndarray:
    """Compute ``diag(A* A)`` from bounded identity blocks."""
    n_cols = shape[1]
    work_dtype = np.result_type(dtype, np.float64)
    diag = np.zeros(n_cols, dtype=work_dtype)
    block_size = min(32, max(1, n_cols))
    block = np.zeros((n_cols, block_size), dtype=work_dtype)
    for start in range(0, n_cols, block_size):
        stop = min(n_cols, start + block_size)
        cols = stop - start
        block[:, :cols] = 0
        block[start:stop, :cols] = np.eye(cols, dtype=work_dtype)
        res = np.asarray(matmat(block[:, :cols]))
        diag[start:stop] = np.sum(np.abs(res) ** 2, axis=0).real
    return diag


def _dense_array_candidate(value: Any) -> Any:
    """Return dense input without materializing backend arrays."""
    if (
        getattr(value, "shape", None) is not None
        and getattr(value, "ndim", None) is not None
        and getattr(value, "dtype", None) is not None
    ):
        return value
    return np.asarray(value)


def _map_shapes(
    shape: MatrixShape,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return shape metadata compatible with flat dimensions."""
    in_shape = (shape[1],) if input_shape is None else tuple(input_shape)
    out_shape = (shape[0],) if output_shape is None else tuple(output_shape)
    if math.prod(in_shape) != shape[1]:
        raise ValueError(f"Input shape {in_shape} incompatible with operator {shape}.")
    if math.prod(out_shape) != shape[0]:
        raise ValueError(f"Output shape {out_shape} incompatible with operator {shape}.")
    return out_shape, in_shape


def _linear_map_from_dense(
    matrix: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    mat_array = _dense_array_candidate(matrix)
    if mat_array.ndim != 2:
        raise ValueError("Dense operators must be 2-D arrays.")
    shape = tuple(int(dim) for dim in mat_array.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = mat_array.dtype

    def matvec(vec: Any) -> Any:
        xp = _runtime_array_module(mat_array, vec)
        mat_arr = xp.asarray(mat_array)
        vec_arr = xp.asarray(vec).reshape(shape[1])
        return xp.matmul(mat_arr, vec_arr)

    def rmatvec(vec: Any) -> Any:
        xp = _runtime_array_module(mat_array, vec)
        mat_arr = xp.asarray(mat_array)
        vec_arr = xp.asarray(vec).reshape(shape[0])
        return xp.matmul(xp.swapaxes(xp.conjugate(mat_arr), -2, -1), vec_arr)

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(mat_array, block)
        mat_arr = xp.asarray(mat_array)
        block_arr = xp.asarray(block).reshape(shape[1], -1)
        return xp.matmul(mat_arr, block_arr)

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(mat_array, block)
        mat_arr = xp.asarray(mat_array)
        block_arr = xp.asarray(block).reshape(shape[0], -1)
        adjoint = xp.swapaxes(xp.conjugate(mat_arr), -2, -1)
        return xp.matmul(adjoint, block_arr)

    def normal_matrix_diag() -> np.ndarray:
        mat_np = to_numpy(mat_array)
        return np.sum(np.abs(mat_np) ** 2, axis=0)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(mat_array)

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=(mat_array,),
        output_shape=out_shape,
        input_shape=in_shape,
    )


def diagonal_linear_map(
    diag_values: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    """Return a map backed by a diagonal vector."""
    diag_array = _dense_array_candidate(diag_values).reshape(-1)
    size = int(diag_array.size)
    out_shape, in_shape = _map_shapes((size, size), input_shape, output_shape)
    dtype = diag_array.dtype

    def matvec(vec: Any) -> Any:
        xp = _runtime_array_module(diag_array, vec)
        diag_arr = xp.asarray(diag_array)
        vec_arr = xp.asarray(vec).reshape(size)
        return diag_arr * vec_arr

    def rmatvec(vec: Any) -> Any:
        xp = _runtime_array_module(diag_array, vec)
        diag_arr = xp.asarray(diag_array)
        vec_arr = xp.asarray(vec).reshape(size)
        return xp.conjugate(diag_arr) * vec_arr

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(diag_array, block)
        diag_arr = xp.asarray(diag_array).reshape(size, 1)
        block_arr = xp.asarray(block).reshape(size, -1)
        return diag_arr * block_arr

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(diag_array, block)
        diag_arr = xp.asarray(diag_array).reshape(size, 1)
        block_arr = xp.asarray(block).reshape(size, -1)
        return xp.conjugate(diag_arr) * block_arr

    def normal_matrix_diag() -> np.ndarray:
        return np.abs(to_numpy(diag_array)) ** 2

    def dense_array(xp: Any) -> Any:
        return xp.diag(xp.asarray(diag_array))

    def diagonal_array(xp: Any) -> Any:
        return xp.asarray(diag_array).reshape(size)

    return LinearMap(
        shape=(size, size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _diagonal_array_func=diagonal_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=(diag_array,),
        output_shape=out_shape,
        input_shape=in_shape,
    )


def _linear_map_from_linear_operator(
    op: ScipyLinearOperator,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    shape = tuple(int(dim) for dim in op.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = op.dtype or np.float64

    def matvec(vec: Any) -> Any:
        return op.matvec(np.asarray(vec).reshape(shape[1]))

    def rmatvec(vec: Any) -> Any:
        return op.rmatvec(np.asarray(vec).reshape(shape[0]))

    def matmat(block: Any) -> Any:
        return op.matmat(np.asarray(block).reshape(shape[1], -1))

    def rmatmat(block: Any) -> Any:
        return op.rmatmat(np.asarray(block).reshape(shape[0], -1))

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        output_shape=out_shape,
        input_shape=in_shape,
    )


def _linear_map_from_scipy_sparse(
    op: scipy.sparse.spmatrix,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    sparse = op.tocsr()
    adjoint = sparse.conjugate().transpose().tocsr()
    shape = tuple(int(dim) for dim in sparse.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = sparse.dtype

    def matvec(vec: Any) -> np.ndarray:
        return sparse @ np.asarray(vec).reshape(shape[1])

    def rmatvec(vec: Any) -> np.ndarray:
        return adjoint @ np.asarray(vec).reshape(shape[0])

    def matmat(block: Any) -> np.ndarray:
        return sparse @ np.asarray(block).reshape(shape[1], -1)

    def rmatmat(block: Any) -> np.ndarray:
        return adjoint @ np.asarray(block).reshape(shape[0], -1)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(sparse.toarray())

    def normal_matrix_diag() -> np.ndarray:
        return np.asarray(sparse.multiply(sparse.conjugate()).sum(axis=0)).ravel().real

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        output_shape=out_shape,
        input_shape=in_shape,
    )


def _linear_map_from_jax_sparse(
    op: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    shape = tuple(int(dim) for dim in op.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = op.dtype
    backend_context = tuple(
        operand
        for operand in (getattr(op, "data", None), getattr(op, "indices", None))
        if operand is not None
    )

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec, *backend_context)
        return op @ xp.asarray(vec).reshape(shape[1])

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(vec, *backend_context)
        return op.T @ xp.asarray(vec).reshape(shape[0])

    def matmat(block: Any) -> Any:
        xp = get_array_module(block, *backend_context)
        return op @ xp.asarray(block).reshape(shape[1], -1)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block, *backend_context)
        return op.T @ xp.asarray(block).reshape(shape[0], -1)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(op.todense())

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _backend_context=backend_context,
        output_shape=out_shape,
        input_shape=in_shape,
    )


def as_linear_map(
    op: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> LinearMap:
    """Convert supported operator types into a ``LinearMap``."""
    if isinstance(op, LinearMap):
        if input_shape is None and output_shape is None:
            return op
        out_shape, in_shape = _map_shapes(
            op.shape,
            input_shape if input_shape is not None else op.input_shape,
            output_shape if output_shape is not None else op.output_shape,
        )
        if out_shape == op.output_shape and in_shape == op.input_shape:
            return op
        return replace(op, output_shape=out_shape, input_shape=in_shape)

    op_type = str(type(op))
    is_jax_sparse = "jax.experimental.sparse" in op_type or (
        "jax" in op_type and hasattr(op, "todense") and hasattr(op, "indices")
    )
    if is_jax_sparse:
        return _linear_map_from_jax_sparse(
            op, input_shape=input_shape, output_shape=output_shape
        )

    if isinstance(op, ScipyLinearOperator):
        return _linear_map_from_linear_operator(
            op, input_shape=input_shape, output_shape=output_shape
        )

    if scipy.sparse.issparse(op):
        if use_jax():
            try:
                from jax.experimental.sparse import BCOO

                return _linear_map_from_jax_sparse(
                    BCOO.from_scipy_sparse(op),
                    input_shape=input_shape,
                    output_shape=output_shape,
                )
            except Exception:
                pass
        return _linear_map_from_scipy_sparse(
            op, input_shape=input_shape, output_shape=output_shape
        )

    try:
        arr = _dense_array_candidate(op)
    except Exception as exc:
        message = f"Unsupported operator type '{type(op)}' for LinearMap conversion."
        raise TypeError(message) from exc

    if arr.ndim == 1:
        size = int(arr.size)
        if input_shape is not None and math.prod(input_shape) != size:
            raise ValueError(f"1-D operator size {size} mismatch with input {input_shape}.")
        if output_shape is not None and math.prod(output_shape) != size:
            raise ValueError(f"1-D operator size {size} mismatch with output {output_shape}.")
        return diagonal_linear_map(arr, input_shape=input_shape, output_shape=output_shape)

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
        input_ndim = len(inferred_input)
        if input_ndim <= arr.ndim and tuple(arr.shape[-input_ndim:]) == inferred_input:
            inferred_output = tuple(arr.shape[:-input_ndim])
        else:
            inferred_output = (flat_out,)
    else:
        flat_out = math.prod(output_shape)
        if flat_out * flat_in != total_elements:
            raise ValueError(
                f"Operator with shape {arr.shape} incompatible with provided shapes "
                f"{output_shape} -> {inferred_input}."
            )
        inferred_output = output_shape
    return _linear_map_from_dense(
        arr.reshape(flat_out, flat_in),
        input_shape=inferred_input,
        output_shape=inferred_output,
    )
