"""
Least-squares problem definition.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator

from pynamit.math.tensor_chain import TensorChain

# --- Type Aliases for Clarity ---
OperatorInput: TypeAlias = Union[np.ndarray, LinearOperator, TensorChain]
OperatorInputList: TypeAlias = Union[OperatorInput, List[OperatorInput]]
NumericInputList: TypeAlias = Union[float, List[float]]


# --- A simple cached_property decorator (if not using Python 3.8+) ---
def cached_property(func: Callable):
    """A simple decorator to cache the result of a method."""
    return property(functools.lru_cache(maxsize=None)(func))


@dataclass
class ProcessedOperator:
    """Container for a processed operator, its shapes, and properties.

    Attributes:
        op: The core flattened operator (ndarray or LinearOperator).
        output_shape: The "natural" multi-dimensional output shape.
        input_shape: The "natural" multi-dimensional input shape.
        is_diagonal: True if `op` represents a diagonal operator, enabling
                     element-wise multiplication instead of matmul.
    """

    op: Union[np.ndarray, LinearOperator]
    output_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    is_diagonal: bool = False

    @property
    def num_rows(self) -> int:
        return self.op.shape[0]

    @property
    def dtype(self) -> np.dtype:
        return self.op.dtype


class LeastSquaresProblem:
    """
    Defines and manages the mathematical structure of a least-squares problem.

    This class assembles operators, weights, and regularization terms into a
    cohesive problem definition. It provides methods to construct the system
    in both matrix-free (LinearOperator) and dense (numpy.ndarray) forms,
    caching expensive computations for efficiency.

    The problem is formulated as minimizing:
        || A x - b ||_W^2 + sum(|| lambda_i L_i x ||^2)
    where:
        - x is the solution vector.
        - A are the data operators.
        - b are the data vectors.
        - W are the data weighting matrices (sqrt_weights).
        - L_i are the regularization operators.
        - lambda_i are the regularization strengths.
    """

    def __init__(
        self,
        A: OperatorInputList,
        solution_shape: Union[int, Tuple[int, ...]],
        data_shapes: Union[Any, List[Any]],
        sqrt_weights: Optional[Union[Any, List[Any]]] = None,
        regularization_weights: Optional[NumericInputList] = None,
        regularization_matrices: Optional[OperatorInputList] = None,
        matrix_free: bool = True,
    ):
        self.matrix_free = matrix_free
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)

        # REFACTORED: Delegate processing to helper methods
        self._process_data_terms(A, data_shapes, sqrt_weights)
        self._process_regularization_terms(regularization_matrices, regularization_weights)

    # ------------------------------------------------------------------
    # Initialization Helpers
    # ------------------------------------------------------------------

    def _process_data_terms(self, A_in, data_shapes_in, sqrt_weights_in):
        """Processes and validates all data-related inputs."""
        A_list = self._prepare_input_list(A_in, "A")
        self.num_data_terms = len(A_list)
        self.data_shapes = self._normalize_data_shapes(data_shapes_in, self.num_data_terms)

        self.A = [
            self._flatten_operator(
                op, output_shape=self.data_shapes[i], input_shape=self.solution_shape
            )
            for i, op in enumerate(A_list)
        ]

        sqrt_weights_list = self._prepare_input_list(
            sqrt_weights_in, "sqrt_weights", count=self.num_data_terms
        )
        self.sqrt_weights = [
            self._create_weight_operator(w, self.data_shapes[i])
            for i, w in enumerate(sqrt_weights_list)
        ]

    def _process_regularization_terms(self, reg_matrices_in, reg_weights_in):
        """Processes and validates all regularization-related inputs."""
        reg_L_list = self._prepare_input_list(
            reg_matrices_in, "regularization_matrices", is_optional=True
        )
        self.num_reg_terms = len(reg_L_list)

        self.regularization_matrices = [
            self._flatten_operator(L, input_shape=self.solution_shape) if L is not None else None
            for L in reg_L_list
        ]

        self.regularization_weights = self._prepare_input_list(
            reg_weights_in, "regularization_weights", count=self.num_reg_terms, default_val=0.0
        )

    def _create_weight_operator(
        self, w_val: Any, shape: Tuple[int, ...]
    ) -> Optional[ProcessedOperator]:
        """Creates a ProcessedOperator for a given weight input."""
        if w_val is None:
            return None

        flat_dim = math.prod(shape)
        # Handle non-LinearOperator cases (numpy arrays)
        if not isinstance(w_val, LinearOperator):
            arr = np.ascontiguousarray(w_val)
            # Check if it can be treated as a diagonal operator for efficiency
            is_diagonal = (arr.ndim == 1 and arr.size == flat_dim) or (arr.shape == shape)
            if is_diagonal:
                return ProcessedOperator(
                    op=arr.reshape(flat_dim, 1),
                    output_shape=shape,
                    input_shape=shape,
                    is_diagonal=True,
                )
        # Default case: treat as a standard matrix/operator
        return self._flatten_operator(w_val, output_shape=shape, input_shape=shape)

    # ------------------------------------------------------------------
    # Public API - Cached Properties
    # ------------------------------------------------------------------

    @cached_property
    def scaled_lambdas(self) -> List[float]:
        """
        Regularization weights scaled to be commensurate with the data term.

        This scaling improves numerical stability by ensuring the regularization
        and data terms have comparable magnitudes. The scaling factor is derived
        from the median of the diagonal elements of the normal matrices (A'A and L'L),
        which is a robust proxy for the operator's energy.
        """
        diag_A_T_A = self._compute_normal_matrix_diag(self.data_operator.op)
        active_diag_A = diag_A_T_A[diag_A_T_A > 0]
        data_term_scale = np.median(active_diag_A) if active_diag_A.size > 0 else 1.0

        scaled_lambdas = []
        for i, L_item in enumerate(self.regularization_matrices):
            raw_weight = self.regularization_weights[i]
            if raw_weight == 0 or L_item is None:
                scaled_lambdas.append(0.0)
                continue

            diag_L_T_L = self._compute_normal_matrix_diag(L_item.op)
            active_diag_L = diag_L_T_L[diag_L_T_L > 0]
            reg_term_scale = np.median(active_diag_L) if active_diag_L.size > 0 else 1.0

            scale_factor = (
                math.sqrt(data_term_scale / reg_term_scale) if reg_term_scale > 1e-14 else 0.0
            )
            scaled_lambdas.append(math.sqrt(raw_weight) * scale_factor)

        return scaled_lambdas

    @cached_property
    def data_operator(self) -> ProcessedOperator:
        """A matrix-free LinearOperator for the data part of the problem (W @ A)."""
        # The 'op' attribute will be the LinearOperator
        op = self._build_system_operator(include_regularization=False)
        shape = (op.shape[0],)  # Output shape is just a flat vector
        return ProcessedOperator(op, output_shape=shape, input_shape=self.solution_shape)

    @cached_property
    def dense_system_matrix(self) -> np.ndarray:
        """The full system matrix G, assembled as a dense numpy array."""
        all_rows = []
        # Data terms
        for i, a_item in enumerate(self.A):
            op = self.densify_op(a_item)
            w_item = self.sqrt_weights[i]
            if w_item:
                w_op = self.densify_op(w_item)
                op = w_op * op if w_item.is_diagonal else w_op @ op
            all_rows.append(op)

        # Regularization terms
        lambdas = self.scaled_lambdas
        for i, L_item in enumerate(self.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_rows.append(lambdas[i] * self.densify_op(L_item))

        dtype = self.A[0].dtype if self.A else np.float64
        if not all_rows:
            return np.zeros((0, self.solution_size), dtype=dtype)

        return np.vstack(all_rows)

    @cached_property
    def svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The SVD of the dense system matrix: (U, s, Vh)."""
        return np.linalg.svd(self.dense_system_matrix, full_matrices=False)

    # ------------------------------------------------------------------
    # Public API - Methods
    # ------------------------------------------------------------------

    def assemble_rhs_block(
        self, b: Union[Any, List[Any]]
    ) -> Tuple[Optional[np.ndarray], Tuple[int, ...], int]:
        """Assembles the right-hand-side block 'd' for the system Gx = d."""
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid_b = [p for p in processed if p[0] is not None]

        if not valid_b:
            return None, (), 0

        # Validate that all b-terms have consistent scenario shapes
        scenario_shape = valid_b[0][1]
        if not all(p[1] == scenario_shape for p in valid_b):
            raise ValueError("Inconsistent scenario shapes in b terms.")
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1

        # Pre-calculate total rows for the d_block
        op_rows = self.get_system_operator(include_regularization=True)[0].shape[0]
        dtype = self.A[0].dtype if self.A else np.float64
        d_block = np.zeros((op_rows, num_scenarios), dtype=dtype)

        row = 0
        for i, (b_col_block, _) in enumerate(processed):
            num_a_rows = self.A[i].num_rows
            if b_col_block is not None:
                w_item = self.sqrt_weights[i]
                if w_item:
                    if w_item.is_diagonal:
                        b_col_block = self.densify_op(w_item) * b_col_block
                    else:
                        b_col_block = self.apply_op_to_block(w_item.op, b_col_block)
                d_block[row : row + num_a_rows, :] = b_col_block
            row += num_a_rows

        return d_block, scenario_shape, num_scenarios

    def get_system_operator(
        self, num_scenarios: int = 1, include_regularization: bool = True
    ) -> Tuple[LinearOperator, Callable, Callable]:  # Always return a tuple
        """
        Returns a matrix-free LinearOperator and block operations for the system.
        """
        op_block = self._build_system_operator(include_regularization)

        # Helper functions for block operations are always useful
        rmatvec_block = lambda y_block: self.apply_op_T_to_block(op_block, y_block)
        matvec_block = lambda x_block: self.apply_op_to_block(op_block, x_block)

        if num_scenarios == 1:
            # For a single scenario, the base operator is the final operator
            return op_block, rmatvec_block, matvec_block

        # For multiple scenarios, create a wrapper operator for flattened vectors
        op_rows, num_features = op_block.shape
        shape = (op_rows * num_scenarios, num_features * num_scenarios)
        dtype = op_block.dtype

        def matvec_final(x_flat: np.ndarray) -> np.ndarray:
            x_block = x_flat.reshape(num_features, num_scenarios)
            return matvec_block(x_block).ravel()

        def rmatvec_final(y_flat: np.ndarray) -> np.ndarray:
            y_block = y_flat.reshape(op_rows, num_scenarios)
            return rmatvec_block(y_block).ravel()

        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        return op, rmatvec_block, matvec_block

    # ------------------------------------------------------------------
    # Core Operator Logic
    # ------------------------------------------------------------------

    def _build_system_operator(self, include_regularization: bool) -> LinearOperator:
        """
        CONSOLIDATED: Builds a LinearOperator for the system (data +/- regularization).
        This helper reduces code duplication between the old `get_system_operator`
        and `_build_data_operator`.
        """
        num_features = self.solution_size
        active_lambdas = self.scaled_lambdas if include_regularization else []

        # Determine total rows
        op_rows_data = sum(a.num_rows for a in self.A)
        op_rows_reg = 0
        if include_regularization:
            op_rows_reg = sum(
                L.num_rows
                for i, L in enumerate(self.regularization_matrices)
                if i < len(active_lambdas) and L and active_lambdas[i] > 0
            )
        op_rows = op_rows_data + op_rows_reg
        dtype = self.A[0].dtype if self.A else np.float64

        def matvec(x: np.ndarray) -> np.ndarray:
            # Assumes a single column vector for matvec
            x_block = x.reshape(-1, 1)
            output_blocks: List[np.ndarray] = []

            # Data terms
            for i, a_item in enumerate(self.A):
                res_block = self.apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item:
                    if w_item.is_diagonal:
                        res_block = self.densify_op(w_item) * res_block
                    else:
                        res_block = self.apply_op_to_block(w_item.op, res_block)
                output_blocks.append(res_block)

            # Regularization terms
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                        term = active_lambdas[i] * self.apply_op_to_block(L_item.op, x_block)
                        output_blocks.append(term)

            if not output_blocks:
                return np.zeros((op_rows,), dtype=dtype)
            return np.vstack(output_blocks).ravel()

        def rmatvec(y: np.ndarray) -> np.ndarray:
            y_block = y.reshape(-1, 1)
            x_block = np.zeros((num_features, 1), dtype=y.dtype)
            row = 0

            # Data terms
            for i, a_item in enumerate(self.A):
                y_part = y_block[row : row + a_item.num_rows, :]
                w_item = self.sqrt_weights[i]
                if w_item:
                    if w_item.is_diagonal:
                        y_part = self.densify_op(w_item).conj() * y_part
                    else:
                        y_part = self.apply_op_T_to_block(w_item.op, y_part)
                x_block += self.apply_op_T_to_block(a_item.op, y_part)
                row += a_item.num_rows

            # Regularization terms
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                        y_part = y_block[row : row + L_item.num_rows, :]
                        x_block += active_lambdas[i] * self.apply_op_T_to_block(L_item.op, y_part)
                        row += L_item.num_rows
            return x_block.ravel()

        return LinearOperator((op_rows, num_features), matvec=matvec, rmatvec=rmatvec, dtype=dtype)

    # ------------------------------------------------------------------
    # Static and Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_normal_matrix_diag(op: Union[LinearOperator, np.ndarray]) -> np.ndarray:
        """Computes the diagonal of the normal matrix op.T @ op efficiently."""
        if isinstance(op, np.ndarray):
            return np.sum(np.abs(op) ** 2, axis=0)

        n_cols = op.shape[1]
        diag = np.zeros(n_cols, dtype=op.dtype)
        e = np.zeros(n_cols, dtype=op.dtype)
        for i in range(n_cols):
            e[i] = 1.0
            col = op.matvec(e)
            diag[i] = np.dot(col.conj(), col).real
            e[i] = 0.0
        return diag

    def apply_op_to_block(
        self, op: Union[np.ndarray, LinearOperator], x_block: np.ndarray
    ) -> np.ndarray:
        """Applies an operator to a block of vectors (N, K)."""
        if isinstance(op, LinearOperator):
            # Use matmat for efficiency with multiple vectors
            return op.matmat(x_block)
        return op @ x_block

    def apply_op_T_to_block(
        self, op: Union[np.ndarray, LinearOperator], y_block: np.ndarray
    ) -> np.ndarray:
        """Applies the adjoint of an operator to a block of vectors."""
        if isinstance(op, LinearOperator):
            return op.rmatmat(y_block)
        return op.T.conj() @ y_block

    def densify_op(self, item: Optional[ProcessedOperator]) -> Optional[np.ndarray]:
        """Converts a ProcessedOperator into a dense numpy array."""
        if item is None:
            return None
        op = item.op
        if isinstance(op, LinearOperator):
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    @staticmethod
    def _prepare_input_list(
        item: Optional[Any],
        name: str,
        count: Optional[int] = None,
        is_optional: bool = False,
        default_val: Any = None,
    ) -> list:
        # (This helper was already good, no changes needed)
        if item is None:
            if is_optional:
                return []
            if count is None:
                raise ValueError(f"Input '{name}' cannot be None.")
            return [default_val] * count
        lst = item if isinstance(item, list) else [item]
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, but expected {count}.")
        return lst

    def _normalize_data_shapes(
        self, data_shapes: Any, expected_count: int
    ) -> List[Tuple[int, ...]]:
        # (This helper was already good, no changes needed)
        if not isinstance(data_shapes, list):
            data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1:
            data_shapes = data_shapes * expected_count
        if len(data_shapes) != expected_count:
            raise ValueError("Number of data_shapes does not match number of A operators.")
        return [(s,) if isinstance(s, int) else tuple(s) for s in data_shapes]

    def _flatten_operator(
        self,
        op: OperatorInput,
        output_shape: Tuple[int, ...] = None,
        input_shape: Tuple[int, ...] = None,
    ) -> ProcessedOperator:
        """
        Processes and flattens an operator input into a standard 2D form.
        This was previously named `_flatten`.
        """
        # (Logic is mostly the same, just better organized type handling)
        if isinstance(op, TensorChain):
            lin_op = op.as_linear_operator() if self.matrix_free else op.to_dense()
            return ProcessedOperator(
                op=lin_op, output_shape=op.output_shape, input_shape=op.input_shape
            )
        if isinstance(op, LinearOperator):
            return ProcessedOperator(
                op=op, output_shape=(op.shape[0],), input_shape=(op.shape[1],)
            )
        if isinstance(op, np.ndarray):
            array = np.ascontiguousarray(op)
            if input_shape is None and output_shape is None:
                raise ValueError("At least one shape must be provided for numpy arrays.")

            # Infer missing shape
            if input_shape is None:
                flat_out = math.prod(output_shape)
                if array.size % flat_out != 0:
                    raise ValueError("Array size is incompatible with provided output_shape.")
                input_shape = (array.size // flat_out,)
            elif output_shape is None:
                flat_in = math.prod(input_shape)
                if array.size % flat_in != 0:
                    raise ValueError("Array size is incompatible with provided input_shape.")
                output_shape = (array.size // flat_in,)

            flat_in_size, flat_out_size = math.prod(input_shape), math.prod(output_shape)
            if array.size != flat_in_size * flat_out_size:
                raise ValueError(
                    f"Array size ({array.size}) is incompatible with specified shapes."
                )

            return ProcessedOperator(
                array.reshape(flat_out_size, flat_in_size), output_shape, input_shape
            )

        raise TypeError(
            f"Input must be a numpy array, TensorChain, or LinearOperator, got {type(op)}"
        )

    def _process_b_vector(
        self, b_val: Any, data_shape: Tuple[int, ...]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, ...]]]:
        # (This helper was already good, no changes needed)
        if b_val is None:
            return None, None
        b = np.ascontiguousarray(b_val)
        num_data_dims = len(data_shape)
        is_exact = b.shape == data_shape
        is_multi = b.ndim > num_data_dims and b.shape[:num_data_dims] == data_shape
        is_flat = b.ndim == 1 and b.size == math.prod(data_shape)
        if not (is_exact or is_multi or is_flat):
            raise ValueError(f"Shape {b.shape} incompatible with data_shape {data_shape}.")
        scenario_shape = b.shape[num_data_dims:] if is_multi else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = b.reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape
