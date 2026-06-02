"""Implementation for einsum-backed ``LinearMap`` factories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from pynamit.math.backend import get_array_module, to_numpy
from pynamit.math.linear_map import LinearMap

_EINSUM_BATCH_LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _dtype_of(value: Any):
    """Return dtype metadata without materializing backend arrays."""
    dtype = getattr(value, "dtype", None)
    return np.asarray(value).dtype if dtype is None else dtype


def _batched_einsum_string(einsum_string: str, operand_index: int) -> Optional[str]:
    """Return an einsum string with one extra batch axis."""
    spec = einsum_string.replace(" ", "")
    if "..." in spec or "->" not in spec:
        return None
    lhs, rhs = spec.split("->", maxsplit=1)
    operands = lhs.split(",")
    if operand_index < 0:
        operand_index += len(operands)
    if operand_index < 0 or operand_index >= len(operands):
        return None

    used_labels = set(lhs.replace(",", "") + rhs)
    batch_label = next(
        (label for label in _EINSUM_BATCH_LABELS if label not in used_labels), None
    )
    if batch_label is None:
        return None

    operands[operand_index] = operands[operand_index] + batch_label
    return ",".join(operands) + "->" + rhs + batch_label


def _derive_einsum_strings_from_matvec(
    einsum_string_matvec: str,
    num_component_tensors: int,
    input_operand_index: int,
) -> tuple[str, str]:
    """Derive dense and adjoint einsum strings from a forward map."""
    spec = einsum_string_matvec.replace(" ", "")
    if "..." in spec or "->" not in spec:
        raise ValueError("Derived einsum maps require an explicit non-ellipsis output.")

    lhs, output_subscript = spec.split("->", maxsplit=1)
    operand_subscripts = lhs.split(",")
    if input_operand_index < 0:
        input_operand_index += len(operand_subscripts)
    if input_operand_index < 0 or input_operand_index >= len(operand_subscripts):
        raise ValueError("input_operand_index is outside the einsum operands.")
    if len(operand_subscripts) != num_component_tensors + 1:
        raise ValueError(
            "Forward einsum must contain all component tensors plus one input operand."
        )

    input_subscript = operand_subscripts[input_operand_index]
    component_subscripts = tuple(
        subscript
        for index, subscript in enumerate(operand_subscripts)
        if index != input_operand_index
    )
    if set(output_subscript).intersection(input_subscript):
        raise ValueError("Input and output subscripts must use distinct labels.")

    dense_subscript = ",".join(component_subscripts) + "->" + output_subscript + input_subscript
    rmatvec_subscript = (
        ",".join((output_subscript,) + component_subscripts) + "->" + input_subscript
    )
    return dense_subscript, rmatvec_subscript


@dataclass
class _EinsumMap:
    """Einsum implementation backing one ``LinearMap``."""

    component_tensors: tuple[Any, ...]
    einsum_string_dense: str
    einsum_string_matvec: str
    einsum_string_rmatvec: str
    output_shape: tuple[int, ...]
    input_shape: tuple[int, ...]
    _einsum_path_matvec: Optional[list] = field(default=None, repr=False)
    _einsum_path_rmatvec: Optional[list] = field(default=None, repr=False)
    _einsum_path_matmat: Optional[list] = field(default=None, repr=False)
    _einsum_path_rmatmat: Optional[list] = field(default=None, repr=False)
    _einsum_string_matmat: Optional[str] = field(default=None, repr=False)
    _einsum_string_rmatmat: Optional[str] = field(default=None, repr=False)
    _component_arrays_np: Optional[list[np.ndarray]] = field(default=None, repr=False)

    @property
    def dtype(self):
        """Data type of the operator, given by its component tensors."""
        return np.result_type(*[_dtype_of(tensor) for tensor in self.component_tensors])

    def to_linear_map(self) -> LinearMap:
        """Return this einsum contraction as a ``LinearMap``."""
        flat_out = math.prod(self.output_shape)
        flat_in = math.prod(self.input_shape)
        return LinearMap(
            shape=(flat_out, flat_in),
            dtype=self.dtype,
            _matvec=self.matvec,
            _rmatvec=self.rmatvec,
            _matmat=self.matmat,
            _rmatmat=self.rmatmat,
            _dense_array_func=self.dense_array,
            _normal_matrix_diag=self.normal_matrix_diag,
            _backend_context=self.component_tensors,
            output_shape=self.output_shape,
            input_shape=self.input_shape,
        )

    def dense_array(self, xp: Any = None) -> Any:
        """Return dense matrix on the requested backend."""
        xp = get_array_module(*self.component_tensors) if xp is None else xp
        component_arrays = [xp.asarray(tensor) for tensor in self.component_tensors]
        dense_matrix = xp.einsum(
            self.einsum_string_dense,
            *component_arrays,
            optimize=True,
        )
        return dense_matrix.reshape(
            math.prod(self.output_shape), math.prod(self.input_shape)
        )

    def normal_matrix_diag(self) -> np.ndarray:
        """Compute ``diag(A* A)`` without building the dense matrix."""
        flat_in = math.prod(self.input_shape)
        diag = np.zeros(flat_in, dtype=np.float64)
        if flat_in == 0:
            return diag

        block_size = min(32, flat_in)
        block = np.zeros((flat_in, block_size), dtype=self.dtype)
        for start in range(0, flat_in, block_size):
            stop = min(flat_in, start + block_size)
            cols = stop - start
            block[:, :cols] = 0
            block[start:stop, :cols] = np.eye(cols, dtype=self.dtype)
            res = np.asarray(self.matmat(block[:, :cols]))
            diag[start:stop] = np.sum(np.abs(res) ** 2, axis=0).real
        return diag

    def _numpy_component_arrays(self) -> list[np.ndarray]:
        """Return cached NumPy component arrays."""
        if self._component_arrays_np is None:
            self._component_arrays_np = [to_numpy(t) for t in self.component_tensors]
        return self._component_arrays_np

    def _matvec_path(self) -> list:
        """Return the cached optimized NumPy path for matvec."""
        if self._einsum_path_matvec is None:
            dummy_input = np.empty(self.input_shape, dtype=self.dtype)
            self._einsum_path_matvec = np.einsum_path(
                self.einsum_string_matvec,
                *self._numpy_component_arrays(),
                dummy_input,
                optimize="greedy",
            )[0]
        return self._einsum_path_matvec

    def _rmatvec_path(self) -> list:
        """Return the cached optimized NumPy path for rmatvec."""
        if self._einsum_path_rmatvec is None:
            dummy_grad_output = np.empty(self.output_shape, dtype=self.dtype)
            self._einsum_path_rmatvec = np.einsum_path(
                self.einsum_string_rmatvec,
                dummy_grad_output,
                *self._numpy_component_arrays(),
                optimize="greedy",
            )[0]
        return self._einsum_path_rmatvec

    def _matmat_string(self) -> Optional[str]:
        """Return a batched matvec einsum string if possible."""
        if self._einsum_string_matmat is None:
            self._einsum_string_matmat = _batched_einsum_string(
                self.einsum_string_matvec, -1
            )
        return self._einsum_string_matmat

    def _rmatmat_string(self) -> Optional[str]:
        """Return a batched adjoint einsum string if possible."""
        if self._einsum_string_rmatmat is None:
            self._einsum_string_rmatmat = _batched_einsum_string(
                self.einsum_string_rmatvec, 0
            )
        return self._einsum_string_rmatmat

    def _matmat_path(self) -> Optional[list]:
        """Return the cached optimized NumPy path for matmat."""
        einsum_string = self._matmat_string()
        if einsum_string is None:
            return None
        if self._einsum_path_matmat is None:
            dummy_input = np.empty(self.input_shape + (1,), dtype=self.dtype)
            self._einsum_path_matmat = np.einsum_path(
                einsum_string,
                *self._numpy_component_arrays(),
                dummy_input,
                optimize="greedy",
            )[0]
        return self._einsum_path_matmat

    def _rmatmat_path(self) -> Optional[list]:
        """Return the cached optimized NumPy path for rmatmat."""
        einsum_string = self._rmatmat_string()
        if einsum_string is None:
            return None
        if self._einsum_path_rmatmat is None:
            dummy_grad_output = np.empty(self.output_shape + (1,), dtype=self.dtype)
            self._einsum_path_rmatmat = np.einsum_path(
                einsum_string,
                dummy_grad_output,
                *self._numpy_component_arrays(),
                optimize="greedy",
            )[0]
        return self._einsum_path_rmatmat

    def _matvec_numpy(self, x_flat: Any) -> np.ndarray:
        """Apply using cached NumPy contraction paths."""
        x_tensor = np.asarray(x_flat).reshape(self.input_shape)
        res = np.einsum(
            self.einsum_string_matvec,
            *self._numpy_component_arrays(),
            x_tensor,
            optimize=self._matvec_path(),
        )
        return res.reshape(-1)

    def _rmatvec_numpy(self, y_flat: Any) -> np.ndarray:
        """Apply the adjoint using cached NumPy contraction paths."""
        grad_tensor = np.asarray(y_flat).reshape(self.output_shape)
        conj_tensors = [arr.conj() for arr in self._numpy_component_arrays()]
        grad_x = np.einsum(
            self.einsum_string_rmatvec,
            grad_tensor,
            *conj_tensors,
            optimize=self._rmatvec_path(),
        )
        return grad_x.reshape(-1)

    def _matmat_numpy(self, x_block: Any) -> Optional[np.ndarray]:
        """Apply multiple vectors with cached NumPy paths."""
        einsum_string = self._matmat_string()
        einsum_path = self._matmat_path()
        if einsum_string is None or einsum_path is None:
            return None
        block = np.asarray(x_block)
        x_tensor = block.reshape(self.input_shape + (block.shape[1],))
        res = np.einsum(
            einsum_string,
            *self._numpy_component_arrays(),
            x_tensor,
            optimize=einsum_path,
        )
        return res.reshape(math.prod(self.output_shape), block.shape[1])

    def _rmatmat_numpy(self, y_block: Any) -> Optional[np.ndarray]:
        """Apply adjoints using cached NumPy paths."""
        einsum_string = self._rmatmat_string()
        einsum_path = self._rmatmat_path()
        if einsum_string is None or einsum_path is None:
            return None
        block = np.asarray(y_block)
        grad_tensor = block.reshape(self.output_shape + (block.shape[1],))
        conj_tensors = [arr.conj() for arr in self._numpy_component_arrays()]
        grad_x = np.einsum(
            einsum_string, grad_tensor, *conj_tensors, optimize=einsum_path
        )
        return grad_x.reshape(math.prod(self.input_shape), block.shape[1])

    def matvec(self, x_flat: Any) -> Any:
        """Apply the tensor contraction to one flattened vector."""
        xp = get_array_module(x_flat, *self.component_tensors)
        if xp is np:
            return self._matvec_numpy(x_flat)
        component_arrays = [xp.asarray(t) for t in self.component_tensors]
        x_tensor = xp.asarray(x_flat).reshape(self.input_shape)
        res = xp.einsum(
            self.einsum_string_matvec, *component_arrays, x_tensor, optimize=True
        )
        return xp.reshape(res, (-1,))

    def rmatvec(self, y_flat: Any) -> Any:
        """Apply the adjoint contraction to one flat vector."""
        xp = get_array_module(y_flat, *self.component_tensors)
        if xp is np:
            return self._rmatvec_numpy(y_flat)
        grad_tensor = xp.asarray(y_flat).reshape(self.output_shape)
        conj_tensors = [xp.conjugate(xp.asarray(t)) for t in self.component_tensors]
        grad_x = xp.einsum(
            self.einsum_string_rmatvec,
            grad_tensor,
            *conj_tensors,
            optimize=True,
        )
        return xp.reshape(grad_x, (-1,))

    def matmat(self, x_block: Any) -> Any:
        """Apply the tensor contraction to multiple vectors."""
        xp = get_array_module(x_block, *self.component_tensors)
        x_arr = xp.asarray(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        if xp is np:
            batched = self._matmat_numpy(x_arr)
            if batched is not None:
                return batched
        einsum_string = self._matmat_string()
        if einsum_string is not None:
            component_arrays = [xp.asarray(t) for t in self.component_tensors]
            x_tensor = x_arr.reshape(self.input_shape + (x_arr.shape[1],))
            res = xp.einsum(einsum_string, *component_arrays, x_tensor, optimize=True)
            return xp.reshape(res, (-1, x_arr.shape[1]))
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint tensor contraction to multiple vectors."""
        xp = get_array_module(y_block, *self.component_tensors)
        y_arr = xp.asarray(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        if xp is np:
            batched = self._rmatmat_numpy(y_arr)
            if batched is not None:
                return batched
        einsum_string = self._rmatmat_string()
        if einsum_string is not None:
            grad_tensor = y_arr.reshape(self.output_shape + (y_arr.shape[1],))
            conj_tensors = [xp.conjugate(xp.asarray(t)) for t in self.component_tensors]
            grad_x = xp.einsum(
                einsum_string, grad_tensor, *conj_tensors, optimize=True
            )
            return xp.reshape(grad_x, (-1, y_arr.shape[1]))
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)


def einsum_linear_map(
    *,
    component_tensors: Sequence[Any],
    einsum_string_dense: str,
    einsum_string_matvec: str,
    einsum_string_rmatvec: str,
    output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> LinearMap:
    """Return a ``LinearMap`` backed by cached einsum contractions."""
    return _EinsumMap(
        component_tensors=tuple(component_tensors),
        einsum_string_dense=einsum_string_dense,
        einsum_string_matvec=einsum_string_matvec,
        einsum_string_rmatvec=einsum_string_rmatvec,
        output_shape=tuple(output_shape),
        input_shape=tuple(input_shape),
    ).to_linear_map()


def einsum_linear_map_from_matvec(
    *,
    component_tensors: Sequence[Any],
    einsum_string_matvec: str,
    output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    input_operand_index: int = -1,
) -> LinearMap:
    """Return an einsum-backed map from one forward contraction."""
    component_tensors = tuple(component_tensors)
    einsum_string_dense, einsum_string_rmatvec = _derive_einsum_strings_from_matvec(
        einsum_string_matvec,
        len(component_tensors),
        input_operand_index,
    )
    return einsum_linear_map(
        component_tensors=component_tensors,
        einsum_string_dense=einsum_string_dense,
        einsum_string_matvec=einsum_string_matvec,
        einsum_string_rmatvec=einsum_string_rmatvec,
        output_shape=output_shape,
        input_shape=input_shape,
    )
