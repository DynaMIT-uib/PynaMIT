"""
Least-squares problem definition.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from pynamit.math.tensor_chain import TensorChain


@dataclass
class _ProcessedItem:
    """Container for an operator plus its stated input/output shapes.

    `op` is either a numpy array shaped (out, in) or a LinearOperator with
    corresponding shape. `output_shape` and `input_shape` are the "natural"
    multi-dimensional shapes for the rows and columns respectively.
    """

    op: Union[np.ndarray, LinearOperator]
    output_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]


class LeastSquaresProblem:
    """Lean representation of a least-squares problem.

    Responsibilities:
      - bookkeeping of A operators, data shapes, weights and regularizers
      - assembly of RHS blocks and (matrix-free) system operator G

    This class defines the problem statement. It does not include solver logic
    or caching of solution-related components like scaled weights or dense matrices.
    """

    def __init__(
        self,
        A: Union[Any, List[Any]],
        solution_shape: Union[int, Tuple[int, ...]],
        data_shapes: Union[Any, List[Any]],
        sqrt_weights: Optional[Union[Any, List[Any]]] = None,
        regularization_weights: Optional[Union[float, List[float]]] = None,
        regularization_matrices: Optional[Union[Any, List[Any]]] = None,
        matrix_free: bool = True,
    ):
        self._version = 0
        self.matrix_free = bool(matrix_free)

        # normalize solution_shape -> tuple and compute size
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)

        # set A and weights
        self.update_matrices(A, sqrt_weights=sqrt_weights, data_shapes=data_shapes)

        # regularization
        reg_L_list = self._prepare_input_list(
            regularization_matrices, "regularization_matrices", is_optional=True
        )
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [
            self._flatten(L, input_shape=self.solution_shape) if L is not None else None
            for L in reg_L_list
        ]
        self.regularization_weights = self._prepare_input_list(
            regularization_weights,
            "regularization_weights",
            count=self.num_reg_terms,
            default_val=0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_matrices(self, A, sqrt_weights=None, data_shapes=None) -> None:
        """Update data operators A and optional per-term sqrt_weights.

        `A` may be a single operator or a list. `data_shapes` must be provided
        on first call (or if number of A terms changes). This method increments
        the problem version, signaling to solvers that their caches may be stale.
        """
        A_list = self._prepare_input_list(A, "A")
        self.num_data_terms = len(A_list)

        if data_shapes is not None:
            self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        elif not hasattr(self, "data_shapes") or len(self.data_shapes) != self.num_data_terms:
            raise ValueError(
                "data_shapes must be provided when setting A for the first time or changing number of A operators."
            )

        # store flattened operator descriptors for each A term
        self.A = [
            self._flatten(op, output_shape=self.data_shapes[i], input_shape=self.solution_shape)
            for i, op in enumerate(A_list)
        ]

        # normalize sqrt_weights list and convert simple diagonal vectors to small column arrays
        sqrt_weights_list = self._prepare_input_list(
            sqrt_weights, "sqrt_weights", count=self.num_data_terms
        )
        self.sqrt_weights = []
        for i, w_val in enumerate(sqrt_weights_list):
            if w_val is None:
                self.sqrt_weights.append(None)
                continue

            flat_data_dim = math.prod(self.data_shapes[i])
            asarr = np.ascontiguousarray(w_val)

            is_diagonal_vector = not isinstance(w_val, LinearOperator) and (
                (asarr.ndim == 1 and asarr.size == flat_data_dim)
                or (asarr.shape == tuple(self.data_shapes[i]))
            )

            if is_diagonal_vector:
                self.sqrt_weights.append(
                    _ProcessedItem(
                        op=asarr.reshape(flat_data_dim, 1),
                        output_shape=self.data_shapes[i],
                        input_shape=(1,),
                    )
                )
            else:
                self.sqrt_weights.append(
                    self._flatten(
                        w_val, output_shape=self.data_shapes[i], input_shape=self.data_shapes[i]
                    )
                )

        self._version += 1

    def assemble_rhs_block(
        self, b: Union[Any, List[Any]], lambdas: Optional[List[float]] = None
    ) -> Tuple[Optional[np.ndarray], Tuple[int, ...], int]:
        """Assemble RHS block `d` from one or more `b` inputs.

        The size of the block depends on which regularization terms are active,
        as determined by the `lambdas` argument.

        Returns (d_block, scenario_shape, num_scenarios). If every b term is None
        the function returns (None, (), 0).
        """
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid_b = [(p[0], p[1]) for p in processed if p[0] is not None]

        if not valid_b:
            return None, (), 0

        scenario_shape = valid_b[0][1]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1

        op_rows = sum(a.op.shape[0] for a in self.A)
        active_lambdas = lambdas if lambdas is not None else self.regularization_weights
        op_rows += sum(
            L.op.shape[0]
            for L, w in zip(self.regularization_matrices, active_lambdas)
            if L and w > 0
        )

        dtype = self.A[0].op.dtype if self.A else np.float64
        d_block = np.zeros((op_rows, num_scenarios), dtype=dtype)

        row = 0
        for i, b_val in enumerate(b_list):
            num_a_rows = self.A[i].op.shape[0]
            if b_val is not None:
                b_col_block, b_scenario_shape = processed[i]
                if b_scenario_shape != scenario_shape:
                    raise ValueError("Inconsistent scenario shapes in b terms.")

                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    if w_item.input_shape == (1,):
                        w_op = self.densify_op(w_item)
                        b_col_block = w_op * b_col_block
                    else:
                        w_op = self.densify_op(w_item)
                        b_col_block = w_op @ b_col_block

                d_block[row : row + num_a_rows, :] = b_col_block
            row += num_a_rows

        return d_block, scenario_shape, num_scenarios

    def get_system_operator(
        self,
        num_scenarios: int = 1,
        lambdas: Optional[List[float]] = None,
        include_regularization: bool = True,
    ) -> Tuple[LinearOperator, Callable, Callable]:
        """Return a matrix-free LinearOperator G and direct mat/adjoint callables.

        The returned LinearOperator has shape (op_rows*num_scenarios, solution_size*num_scenarios) and
        accepts flattened 1-D input arrays. The extra returned callables are `rmatvec_block`
        and `matvec_block` which operate on blocks with shapes ((op_rows, num_scenarios),
        (solution_size, num_scenarios)).

        Args:
            num_scenarios: The number of columns (scenarios) to operate on simultaneously.
            lambdas: A list of regularization weights. If None, the raw problem
                     regularization weights are used.
            include_regularization: If False, the operator will not include regularization terms.
        """
        active_lambdas = lambdas if lambdas is not None else self.regularization_weights

        num_features = self.solution_size
        op_rows_data = sum(a.op.shape[0] for a in self.A)
        op_rows_reg = 0
        if include_regularization:
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                    op_rows_reg += L_item.op.shape[0]
        op_rows = op_rows_data + op_rows_reg

        dtype = self.A[0].op.dtype if self.A else np.float64

        def matvec_block(x_block: np.ndarray) -> np.ndarray:
            output_blocks: List[np.ndarray] = []
            for i, a_item in enumerate(self.A):
                res_block = self.apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    if w_item.input_shape == (1,):
                        w_op = self.densify_op(w_item)
                        res_block = w_op * res_block
                    else:
                        res_block = self.apply_op_to_block(w_item.op, res_block)
                output_blocks.append(res_block)

            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                        output_blocks.append(
                            active_lambdas[i] * self.apply_op_to_block(L_item.op, x_block)
                        )

            if output_blocks:
                return np.vstack(output_blocks)
            return np.zeros((0, x_block.shape[1]), dtype=dtype)

        def rmatvec_block(y_block: np.ndarray) -> np.ndarray:
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i, a_item in enumerate(self.A):
                num_a_rows = a_item.op.shape[0]
                y_part = y_block[row : row + num_a_rows, :]
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    if w_item.input_shape == (1,):
                        w_op = self.densify_op(w_item)
                        y_part = w_op.conj() * y_part
                    else:
                        y_part = self.apply_op_T_to_block(w_item.op, y_part)
                x_block += self.apply_op_T_to_block(a_item.op, y_part)
                row += num_a_rows

            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                        num_L_rows = L_item.op.shape[0]
                        y_part = y_block[row : row + num_L_rows, :]
                        x_block += active_lambdas[i] * self.apply_op_T_to_block(L_item.op, y_part)
                        row += num_L_rows

            return x_block

        shape = (op_rows * num_scenarios, num_features * num_scenarios)

        def matvec_final(x_flat: np.ndarray) -> np.ndarray:
            x_block = x_flat.reshape(num_features, num_scenarios)
            return matvec_block(x_block).ravel()

        def rmatvec_final(y_flat: np.ndarray) -> np.ndarray:
            y_block = y_flat.reshape(op_rows, num_scenarios)
            return rmatvec_block(y_block).ravel()

        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        return op, rmatvec_block, matvec_block

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def apply_op_to_block(
        self, op: Union[np.ndarray, LinearOperator], x_block: np.ndarray
    ) -> np.ndarray:
        """Apply operator `op` to a block of column vectors `x_block`."""
        if isinstance(op, LinearOperator):
            return (
                op.matmat(x_block)
                if x_block.shape[1] > 1
                else op.matvec(x_block[:, 0])[:, np.newaxis]
            )
        return op @ x_block

    def apply_op_T_to_block(
        self, op: Union[np.ndarray, LinearOperator], y_block: np.ndarray
    ) -> np.ndarray:
        """Apply adjoint (Hermitian transpose) of `op` to y_block columns."""
        if isinstance(op, LinearOperator):
            return (
                op.rmatmat(y_block)
                if y_block.shape[1] > 1
                else op.rmatvec(y_block[:, 0])[:, np.newaxis]
            )
        return op.T.conj() @ y_block

    def densify_op(self, item: Optional[_ProcessedItem]) -> Optional[np.ndarray]:
        """Convert a _ProcessedItem's operator to a dense numpy array."""
        if item is None:
            return None
        op = item.op
        if isinstance(op, LinearOperator):
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_input_list(
        item: Optional[Any],
        name: str,
        count: Optional[int] = None,
        is_optional: bool = False,
        default_val: Any = None,
    ) -> list:
        if item is None:
            if is_optional:
                return []
            if count is None:
                raise ValueError(f"Input '{name}' cannot be None if 'count' is not specified.")
            return [default_val] * count
        lst = item if isinstance(item, list) else [item]
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, but expected {count}.")
        return lst

    def _normalize_data_shapes(
        self, data_shapes: Any, expected_count: int
    ) -> List[Tuple[int, ...]]:
        if not isinstance(data_shapes, list):
            data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1:
            data_shapes = data_shapes * expected_count
        if len(data_shapes) != expected_count:
            raise ValueError("Number of data_shapes does not match number of A operators.")
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    def _flatten(
        self, op: Any, output_shape: Tuple[int, ...] = None, input_shape: Tuple[int, ...] = None
    ) -> _ProcessedItem:
        if isinstance(op, TensorChain):
            lin = op.as_linear_operator() if self.matrix_free else op.to_dense()
            return _ProcessedItem(op=lin, output_shape=op.output_shape, input_shape=op.input_shape)
        if isinstance(op, LinearOperator):
            return _ProcessedItem(op=op, output_shape=(op.shape[0],), input_shape=(op.shape[1],))
        if not isinstance(op, np.ndarray):
            raise TypeError("Input must be a numpy array, TensorChain, or LinearOperator")

        array = np.ascontiguousarray(op)
        if input_shape is None and output_shape is None:
            raise ValueError("At least one of output_shape or input_shape must be provided.")

        if input_shape is None:
            flat_in = array.size // math.prod(output_shape)
            input_shape = (flat_in,)
        elif output_shape is None:
            flat_out = array.size // math.prod(input_shape)
            output_shape = (flat_out,)

        flat_in, flat_out = math.prod(input_shape), math.prod(output_shape)
        if array.size != flat_in * flat_out:
            raise ValueError("Array size is incompatible with specified shapes.")
        return _ProcessedItem(array.reshape(flat_out, flat_in), output_shape, input_shape)

    def _process_b_vector(
        self, b_val: Any, data_shape: Tuple[int, ...]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, ...]]]:
        if b_val is None:
            return None, None

        b = np.ascontiguousarray(b_val)
        num_data_dims = len(data_shape)

        is_exact = b.shape == data_shape
        is_multi = b.ndim > num_data_dims and b.shape[:num_data_dims] == data_shape
        is_flat = b.ndim == 1 and b.size == math.prod(data_shape)

        if not (is_exact or is_multi or is_flat):
            raise ValueError("Shape of b is incompatible with its data_shape.")

        scenario_shape = b.shape[num_data_dims:] if is_multi else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = b.reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape
