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

from pynamit.utils import get_array_module, to_numpy


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

    @property
    def dtype(self):
        """Data type of the operator, given by its component tensors."""
        return np.result_type(*[to_numpy(t).dtype for t in self.component_tensors])

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

    def to_dense(self) -> np.ndarray:
        """Return dense matrix representation of the operator."""
        xp = get_array_module(*self.component_tensors)
        dense_matrix = xp.einsum(self.einsum_string_dense, *self.component_tensors, optimize=True)
        dense_matrix = to_numpy(dense_matrix)
        return (dense_matrix * self.scaling_factor).reshape(
            math.prod(self.output_shape), math.prod(self.input_shape)
        )

    def as_linear_operator(self) -> LinearOperator:
        """Return a matrix-free LinearOperator representation."""
        flat_out = math.prod(self.output_shape)
        flat_in = math.prod(self.input_shape)
        component_arrays = [to_numpy(t) for t in self.component_tensors]

        # Prepare and cache the optimized einsum path for matvec
        if self._einsum_path_matvec is None:
            # Create a dummy input array
            # to find the optimal contraction path
            dummy_input = np.empty(self.input_shape, dtype=self.dtype)
            self._einsum_path_matvec = np.einsum_path(
                self.einsum_string_matvec, *component_arrays, dummy_input, optimize="greedy"
            )[0]

        # Prepare and cache the optimized einsum path for rmatvec
        if self._einsum_path_rmatvec is None:
            # Create a dummy output gradient
            # to find the optimal contraction path
            dummy_grad_output = np.empty(self.output_shape, dtype=self.dtype)
            self._einsum_path_rmatvec = np.einsum_path(
                self.einsum_string_rmatvec, dummy_grad_output, *component_arrays, optimize="greedy"
            )[0]

        def _matvec(x_flat):
            """Define the forward matrix-vector product."""
            x_tensor = x_flat.reshape(self.input_shape)
            all_tensors = component_arrays + [x_tensor]
            res = np.einsum(
                self.einsum_string_matvec, *all_tensors, optimize=self._einsum_path_matvec
            )
            return (res * self.scaling_factor).flatten()

        def _rmatvec(y_flat):
            """Define the adjoint matrix-vector product."""
            grad_tensor = y_flat.reshape(self.output_shape)
            conj_tensors = [arr.conj() for arr in component_arrays]
            all_adjoint_inputs = [grad_tensor] + conj_tensors
            grad_x = np.einsum(
                self.einsum_string_rmatvec, *all_adjoint_inputs, optimize=self._einsum_path_rmatvec
            )
            return (grad_x.conj() * self.scaling_factor).flatten()

        return LinearOperator(
            shape=(flat_out, flat_in), matvec=_matvec, rmatvec=_rmatvec, dtype=self.dtype
        )
