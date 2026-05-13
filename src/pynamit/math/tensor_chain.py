"""
Tensor chain linear operator module.

A helper class to represent a linear operator defined by a chain of
tensor contractions (einsum).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional
from scipy.sparse.linalg import LinearOperator

from pynamit.utils import asarray, get_array_module, to_numpy


@dataclass
class TensorChain:
    """
    Represents a linear operator formed by an einsum contraction.

    This class can generate a dense matrix representation or a
    matrix-free scipy LinearOperator, which is efficient for iterative
    solvers. It also handles caching of optimized einsum paths for
    performance.
    """

    component_tensors: List[Any]
    einsum_string_dense: str
    einsum_string_matvec: str
    einsum_string_rmatvec: str
    output_shape: tuple
    input_shape: tuple
    scaling_factor: float = 1.0
    _einsum_path_matvec: Optional[list] = field(default=None, repr=False)
    _einsum_path_rmatvec: Optional[list] = field(default=None, repr=False)
    _component_arrays_np: Optional[List[np.ndarray]] = field(default=None, repr=False)

    @property
    def dtype(self):
        """Data type of the operator, given by its component tensors."""
        return np.result_type(*[arr.dtype for arr in self._numpy_component_arrays()])

    def with_scaling(self, factor: float) -> "TensorChain":
        """Return a scaled TensorChain instance."""
        return TensorChain(
            component_tensors=self.component_tensors,
            einsum_string_dense=self.einsum_string_dense,
            einsum_string_matvec=self.einsum_string_matvec,
            einsum_string_rmatvec=self.einsum_string_rmatvec,
            output_shape=self.output_shape,
            input_shape=self.input_shape,
            scaling_factor=self.scaling_factor * factor,
        )

    def __mul__(self, other: Any) -> "TensorChain":
        """Return a scalar-scaled tensor chain."""
        if not np.isscalar(other):
            return NotImplemented
        return self.with_scaling(other)

    def __rmul__(self, other: Any) -> "TensorChain":
        """Return a scalar-scaled tensor chain."""
        return self.__mul__(other)

    def to_linear_map(self):
        """Convert this tensor chain to a generic ``LinearMap``."""
        from pynamit.math.linear_map import LinearMap

        flat_out = math.prod(self.output_shape)
        flat_in = math.prod(self.input_shape)

        return LinearMap(
            shape=(flat_out, flat_in),
            dtype=self.dtype,
            _matvec=lambda vec: self.matvec(vec),
            _rmatvec=lambda vec: self.rmatvec(vec),
            _matmat=lambda block: self.matmat(block),
            _rmatmat=lambda block: self.rmatmat(block),
            _to_dense=self.to_dense,
            source=self,
        )

    def to_dense(self) -> np.ndarray:
        """Return dense matrix representation of the operator."""
        xp = get_array_module(*self.component_tensors)
        dense_matrix = xp.einsum(self.einsum_string_dense, *self.component_tensors, optimize=True)
        dense_matrix = to_numpy(dense_matrix)
        return (dense_matrix * self.scaling_factor).reshape(
            math.prod(self.output_shape), math.prod(self.input_shape)
        )

    def _numpy_component_arrays(self) -> List[np.ndarray]:
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

    def _matvec_numpy(self, x_flat: Any) -> np.ndarray:
        """Apply using cached NumPy contraction paths."""
        x_tensor = np.asarray(x_flat).reshape(self.input_shape)
        res = np.einsum(
            self.einsum_string_matvec,
            *self._numpy_component_arrays(),
            x_tensor,
            optimize=self._matvec_path(),
        )
        return (res * self.scaling_factor).reshape(-1)

    def _rmatvec_numpy(self, y_flat: Any) -> np.ndarray:
        """Apply the adjoint using cached NumPy contraction paths."""
        grad_tensor = np.asarray(y_flat).reshape(self.output_shape)
        conj_tensors = [arr.conj() for arr in self._numpy_component_arrays()]
        grad_x = np.einsum(
            self.einsum_string_rmatvec, grad_tensor, *conj_tensors, optimize=self._rmatvec_path()
        )
        return (grad_x.conj() * self.scaling_factor).reshape(-1)

    def matvec(self, x_flat: Any) -> Any:
        """Apply the tensor chain to one flattened vector."""
        xp = get_array_module(x_flat, *self.component_tensors)
        if xp is np:
            return self._matvec_numpy(x_flat)
        component_arrays = [xp.asarray(t) for t in self.component_tensors]
        x_tensor = xp.asarray(x_flat).reshape(self.input_shape)
        res = xp.einsum(self.einsum_string_matvec, *component_arrays, x_tensor, optimize=True)
        return xp.reshape(res * self.scaling_factor, (-1,))

    def rmatvec(self, y_flat: Any) -> Any:
        """Apply the adjoint tensor chain to one flattened vector."""
        xp = get_array_module(y_flat, *self.component_tensors)
        if xp is np:
            return self._rmatvec_numpy(y_flat)
        grad_tensor = xp.asarray(y_flat).reshape(self.output_shape)
        conj_tensors = [xp.conjugate(xp.asarray(t)) for t in self.component_tensors]
        grad_x = xp.einsum(self.einsum_string_rmatvec, grad_tensor, *conj_tensors, optimize=True)
        return xp.reshape(xp.conjugate(grad_x) * self.scaling_factor, (-1,))

    def matmat(self, x_block: Any) -> Any:
        """Apply the tensor chain to multiple flattened vectors."""
        xp = get_array_module(x_block, *self.component_tensors)
        x_arr = asarray(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint chain to multiple vectors."""
        xp = get_array_module(y_block, *self.component_tensors)
        y_arr = asarray(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def as_linear_operator(self) -> LinearOperator:
        """Return a matrix-free LinearOperator representation."""
        lin_op = self.to_linear_map().as_linear_operator()
        setattr(lin_op, "_tensor_chain", self)
        return lin_op
