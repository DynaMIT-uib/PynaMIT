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

from pynamit.utils import get_array_module, use_jax, vmap

try:  # pragma: no cover - optional optimisation dependency
    from opt_einsum import contract_expression
except Exception:  # pragma: no cover - gracefully degrade if absent
    contract_expression = None


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
    _component_arrays: Optional[List[np.ndarray]] = field(default=None, repr=False)
    _linear_operator: Optional[LinearOperator] = field(default=None, repr=False)
    _contract_matvec: Optional[Any] = field(default=None, repr=False)
    _contract_rmatvec: Optional[Any] = field(default=None, repr=False)

    @property
    def dtype(self):
        """Data type of the operator, given by its component tensors."""
        return np.result_type(*[arr.dtype for arr in self._get_component_arrays_numpy()])

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
        component_arrays = self._get_component_arrays_numpy()
        dense_matrix = np.einsum(
            self.einsum_string_dense, *component_arrays, optimize=True
        )
        return (dense_matrix * self.scaling_factor).reshape(
            math.prod(self.output_shape), math.prod(self.input_shape)
        )

    def as_linear_operator(self) -> LinearOperator:
        """Return a matrix-free LinearOperator representation."""
        if self._linear_operator is not None:
            return self._linear_operator

        flat_out = math.prod(self.output_shape)
        flat_in = math.prod(self.input_shape)
        component_arrays = self._get_component_arrays_numpy()

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

        lin_op = LinearOperator(
            shape=(flat_out, flat_in), matvec=_matvec, rmatvec=_rmatvec, dtype=self.dtype
        )
        setattr(lin_op, "_tensor_chain", self)
        self._linear_operator = lin_op
        return lin_op

    def _get_component_arrays_numpy(self) -> List[np.ndarray]:
        """Return cached numpy copies of the component tensors."""
        if self._component_arrays is None:
            self._component_arrays = [np.asarray(t) for t in self.component_tensors]
        return self._component_arrays

    def matvec(self, x_flat: Any) -> Any:
        """Apply the tensor chain to a single flattened vector."""
        xp = get_array_module(x_flat, *self.component_tensors)
        x_arr = xp.asarray(x_flat)
        x_tensor = xp.reshape(x_arr, self.input_shape)
        contract = self._get_contract_matvec()
        if contract is not None:
            res = contract(*self.component_tensors, x_tensor)
        else:
            if use_jax():
                res = xp.einsum(self.einsum_string_matvec, *self.component_tensors, x_tensor)
            else:
                res = xp.einsum(
                    self.einsum_string_matvec, *self.component_tensors, x_tensor, optimize=True
                )
        res = res * self.scaling_factor
        return xp.reshape(res, (-1,))

    def rmatvec(self, y_flat: Any) -> Any:
        """Apply the adjoint of the tensor chain to a flattened vector."""
        xp = get_array_module(y_flat, *self.component_tensors)
        y_arr = xp.asarray(y_flat)
        y_tensor = xp.reshape(y_arr, self.output_shape)
        conj_tensors = [xp.conjugate(t) for t in self.component_tensors]
        contract = self._get_contract_rmatvec()
        if contract is not None:
            grad = contract(y_tensor, *conj_tensors)
        else:
            if use_jax():
                grad = xp.einsum(self.einsum_string_rmatvec, y_tensor, *conj_tensors)
            else:
                grad = xp.einsum(
                    self.einsum_string_rmatvec, y_tensor, *conj_tensors, optimize=True
                )
        grad = xp.conjugate(grad) * self.scaling_factor
        return xp.reshape(grad, (-1,))

    def matmat(self, x_block: Any) -> Any:
        """Apply the tensor chain to multiple flattened vectors."""
        xp = get_array_module(x_block, *self.component_tensors)
        if x_block.ndim == 1:
            return self.matvec(x_block)
        if use_jax():
            x_arr = xp.asarray(x_block)
            mv = lambda col: self.matvec(col)
            res = vmap(mv)(x_arr.T).T
            return res
        cols = x_block.shape[1]
        results = [self.matvec(x_block[:, i]) for i in range(cols)]
        return xp.stack(results, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint tensor chain to multiple flattened vectors."""
        xp = get_array_module(y_block, *self.component_tensors)
        if y_block.ndim == 1:
            return self.rmatvec(y_block)
        if use_jax():
            y_arr = xp.asarray(y_block)
            rmv = lambda col: self.rmatvec(col)
            res = vmap(rmv)(y_arr.T).T
            return res
        cols = y_block.shape[1]
        results = [self.rmatvec(y_block[:, i]) for i in range(cols)]
        return xp.stack(results, axis=1)

    def _get_contract_matvec(self):
        """Return cached opt_einsum contractor for matvec if available."""
        if contract_expression is None or use_jax():
            return None
        if self._contract_matvec is None:
            shapes = [tuple(t.shape) for t in self.component_tensors] + [tuple(self.input_shape)]
            self._contract_matvec = contract_expression(
                self.einsum_string_matvec, *shapes, optimize="greedy"
            )
        return self._contract_matvec

    def _get_contract_rmatvec(self):
        """Return cached opt_einsum contractor for rmatvec if available."""
        if contract_expression is None or use_jax():
            return None
        if self._contract_rmatvec is None:
            shapes = [tuple(self.output_shape)] + [tuple(t.shape) for t in self.component_tensors]
            self._contract_rmatvec = contract_expression(
                self.einsum_string_rmatvec, *shapes, optimize="greedy"
            )
        return self._contract_rmatvec
