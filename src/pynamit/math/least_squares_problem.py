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
class ProcessedOperator:
    """Container for an operator, its shapes, and properties.

    Attributes:
        op: The core operator (ndarray or LinearOperator).
        output_shape: The "natural" multi-dimensional output shape.
        input_shape: The "natural" multi-dimensional input shape.
        is_diagonal: True if `op` represents a diagonal operator, enabling
                     element-wise multiplication instead of matmul.
    """

    op: Union[np.ndarray, LinearOperator]
    output_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    is_diagonal: bool = False


class LeastSquaresProblem:
    """Defines the mathematical structure of a least-squares problem.

    This class is responsible for managing the operators, weights, and
    regularization terms that constitute the problem. It provides methods
    to assemble the system matrix in various forms (dense or matrix-free)
    and caches expensive computations.
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
        self.matrix_free = bool(matrix_free)
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)
        self._cache: dict = {}

        A_list = self._prepare_input_list(A, "A")
        self.num_data_terms = len(A_list)
        self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        self.A = [
            self._flatten(op, output_shape=self.data_shapes[i], input_shape=self.solution_shape)
            for i, op in enumerate(A_list)
        ]
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
            is_diagonal = not isinstance(w_val, LinearOperator) and (
                (asarr.ndim == 1 and asarr.size == flat_data_dim)
                or (asarr.shape == self.data_shapes[i])
            )
            if is_diagonal:
                self.sqrt_weights.append(
                    ProcessedOperator(
                        op=asarr.reshape(flat_data_dim, 1),
                        output_shape=self.data_shapes[i],
                        input_shape=self.data_shapes[i],
                        is_diagonal=True,
                    )
                )
            else:
                self.sqrt_weights.append(
                    self._flatten(
                        w_val, output_shape=self.data_shapes[i], input_shape=self.data_shapes[i]
                    )
                )

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

    def _get_or_compute(self, key: Any, compute_func: Callable[[], Any]) -> Any:
        """Helper to cache results of expensive computations."""
        if key not in self._cache:
            self._cache[key] = compute_func()
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scaled_lambdas(self) -> List[float]:
        """
        Calculates regularization weights scaled to be commensurate with the
        data term operator. This improves numerical stability.
        """
        key = ("scaled_lambdas",)

        def compute():
            data_op = self.get_data_operator()
            diag_A_T_A = self._compute_normal_matrix_diag(data_op)

            # Use a more descriptive name: this scale is based on the operator, not the RHS data.
            data_term_scale = np.median(diag_A_T_A[diag_A_T_A > 0]) if np.any(diag_A_T_A > 0) else 1.0

            scaled_lambdas = []
            for i, L_item in enumerate(self.regularization_matrices):
                raw_weight = self.regularization_weights[i]
                if raw_weight == 0 or L_item is None:
                    scaled_lambdas.append(0.0)
                    continue

                diag_L_T_L = self._compute_normal_matrix_diag(L_item.op)
                reg_term_scale = (
                    np.median(diag_L_T_L[diag_L_T_L > 0]) if np.any(diag_L_T_L > 0) else 1.0
                )

                scale_factor = math.sqrt(data_term_scale / reg_term_scale) if reg_term_scale > 1e-14 else 0.0
                scaled_lambdas.append(math.sqrt(raw_weight) * scale_factor)

            return scaled_lambdas

        return self._get_or_compute(key, compute)

    def assemble_rhs_block(
        self, b: Union[Any, List[Any]], lambdas: Optional[List[float]] = None
    ) -> Tuple[Optional[np.ndarray], Tuple[int, ...], int]:
        """Assemble RHS block `d` from one or more `b` inputs."""
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
                    w_op = self.densify_op(w_item)
                    if w_item.is_diagonal:
                        b_col_block = w_op * b_col_block
                    else:
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
        """Return a matrix-free LinearOperator G and associated block operations."""
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
                    if w_item.is_diagonal:
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
                    if w_item.is_diagonal:
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

    def get_dense_system_matrix(self, lambdas: List[float]) -> np.ndarray:
        """Assembles and returns the cached full system matrix G."""
        key = ("dense_system_matrix", tuple(lambdas))

        def compute():
            all_rows = []
            for i, a_item in enumerate(self.A):
                op = self.densify_op(a_item)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    w_op = self.densify_op(w_item)
                    op = w_op * op if w_item.is_diagonal else w_op @ op
                all_rows.append(op)

            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                    all_rows.append(lambdas[i] * self.densify_op(L_item))

            dtype = self.A[0].op.dtype if self.A else np.float64
            if not all_rows:
                return np.zeros((0, self.solution_size), dtype=dtype)

            return np.vstack(all_rows)

        return self._get_or_compute(key, compute)

    def get_svd_decomposition(
        self, lambdas: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes and returns the cached SVD of the dense system matrix."""
        key = ("svd_decomposition", tuple(lambdas))

        def compute():
            G_dense = self.get_dense_system_matrix(lambdas)
            return np.linalg.svd(G_dense, full_matrices=False)

        return self._get_or_compute(key, compute)

    def get_data_operator(self) -> LinearOperator:
        """Returns a matrix-free LinearOperator for the data part of the problem."""
        op, _, _ = self.get_system_operator(num_scenarios=1, include_regularization=False)
        return op

    # ------------------------------------------------------------------
    # Public and Private Helpers
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
        if isinstance(op, LinearOperator):
            return (
                op.rmatmat(y_block)
                if y_block.shape[1] > 1
                else op.rmatvec(y_block[:, 0])[:, np.newaxis]
            )
        return op.T.conj() @ y_block

    def densify_op(self, item: Optional[ProcessedOperator]) -> Optional[np.ndarray]:
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
    ) -> ProcessedOperator:
        if isinstance(op, TensorChain):
            lin = op.as_linear_operator() if self.matrix_free else op.to_dense()
            return ProcessedOperator(
                op=lin, output_shape=op.output_shape, input_shape=op.input_shape
            )
        if isinstance(op, LinearOperator):
            return ProcessedOperator(
                op=op, output_shape=(op.shape[0],), input_shape=(op.shape[1],)
            )
        if not isinstance(op, np.ndarray):
            raise TypeError(
                f"Input must be a numpy array, TensorChain, or LinearOperator, but got {type(op)}"
            )
        array = np.ascontiguousarray(op)
        if input_shape is None and output_shape is None:
            raise ValueError(
                "At least one of output_shape or input_shape must be provided for numpy arrays."
            )
        if input_shape is None:
            flat_out = math.prod(output_shape)
            if array.size % flat_out != 0:
                raise ValueError("Array size is incompatible with provided output_shape.")
            flat_in = array.size // flat_out
            input_shape = (flat_in,)
        elif output_shape is None:
            flat_in = math.prod(input_shape)
            if array.size % flat_in != 0:
                raise ValueError("Array size is incompatible with provided input_shape.")
            flat_out = array.size // flat_in
            output_shape = (flat_out,)
        flat_in_size, flat_out_size = math.prod(input_shape), math.prod(output_shape)
        if array.size != flat_in_size * flat_out_size:
            raise ValueError(f"Array size ({array.size}) is incompatible with specified shapes.")
        return ProcessedOperator(
            array.reshape(flat_out_size, flat_in_size),
            output_shape,
            input_shape,
            is_diagonal=False,
        )

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
            raise ValueError(
                f"Shape {b.shape} of b is incompatible with its expected data_shape {data_shape}."
            )
        scenario_shape = b.shape[num_data_dims:] if is_multi else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = b.reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape