"""Least-squares problem definition."""

from __future__ import annotations
from functools import cached_property
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse


from pynamit.math.tensor_chain import TensorChain
from pynamit.math.linear_map import LinearMap, as_linear_map, diagonal_linear_map
from pynamit.utils import xp, asarray

OperatorInput: TypeAlias = Union[np.ndarray, LinearOperator, TensorChain]
Shape: TypeAlias = Union[int, Tuple[int, ...]]


@dataclass
class ProcessedOperator:
    """A processed operator with associated shape information."""

    linear_map: LinearMap
    output_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    is_diagonal: bool = False
    diag_data: Optional[Any] = None

    @property
    def num_rows(self) -> int:
        """Number of rows in the operator."""
        return self.linear_map.shape[0]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the operator."""
        return self.linear_map.dtype


class LeastSquaresProblem:
    """Defines the mathematical structure of a least-squares problem."""

    def __init__(
        self,
        A: List[OperatorInput],
        solution_shape: Shape,
        data_shapes: List[Shape],
        sqrt_weights: Optional[List[Any]] = None,
        regularization_weights: Optional[List[float]] = None,
        regularization_matrices: Optional[List[Optional[OperatorInput]]] = None,
        matrix_free: bool = True,
    ):
        self.matrix_free = matrix_free
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)
        self.num_data_terms = len(A)
        self.data_shapes = [(s,) if isinstance(s, int) else tuple(s) for s in data_shapes]
        if len(self.data_shapes) != self.num_data_terms:
            raise ValueError("Number of data_shapes does not match number of A operators.")

        self.A = [
            self._flatten_operator(
                op, output_shape=self.data_shapes[i], input_shape=self.solution_shape
            )
            for i, op in enumerate(A)
        ]

        # Weights
        weights_in = sqrt_weights if sqrt_weights is not None else [None] * self.num_data_terms
        if len(weights_in) != self.num_data_terms:
            raise ValueError("Number of sqrt_weights does not match number of A operators.")

        self.sqrt_weights = [
            self._create_weight_operator(w, self.data_shapes[i]) for i, w in enumerate(weights_in)
        ]
        self._sqrt_weight_dense_cache: List[Optional[np.ndarray]] = [None] * self.num_data_terms

        # Regularization
        self.regularization_matrices = []
        self.regularization_weights = []
        if regularization_matrices is not None:
            self.regularization_matrices = [
                self._flatten_operator(L, input_shape=self.solution_shape)
                if L is not None
                else None
                for L in regularization_matrices
            ]
            self.regularization_weights = (
                regularization_weights
                if regularization_weights is not None
                else [0.0] * len(regularization_matrices)
            )
        self.num_reg_terms = len(self.regularization_matrices)

        self._scenario_operator_cache: dict[Tuple[bool, int], LinearOperator] = {}
        self._system_linear_map_cache: dict[bool, LinearMap] = {}
        self._scenario_linear_map_cache: dict[Tuple[bool, int], LinearMap] = {}

    @staticmethod
    def _backend_module():
        """Return the active array module (NumPy or JAX NumPy)."""
        return xp

    def _create_weight_operator(
        self, w_val: Any, shape: Tuple[int, ...]
    ) -> Optional[ProcessedOperator]:
        if w_val is None:
            return None
        flat_dim = math.prod(shape)
        if not isinstance(w_val, LinearOperator) and not isinstance(w_val, TensorChain):
            arr = np.ascontiguousarray(w_val)
            is_diagonal = (arr.ndim == 1 and arr.size == flat_dim) or (arr.shape == shape)
            if is_diagonal:
                diag_data = arr.reshape(flat_dim)
                linear_map = diagonal_linear_map(diag_data)
                return ProcessedOperator(
                    linear_map=linear_map,
                    output_shape=shape,
                    input_shape=shape,
                    is_diagonal=True,
                    diag_data=asarray(diag_data),
                )
        flattened = self._flatten_operator(w_val, output_shape=shape, input_shape=shape)
        return flattened

    @cached_property
    def scaled_lambdas(self) -> List[float]:
        """Compute scaled regularization weights."""
        diag_A_T_A = self._compute_normal_matrix_diag(self.data_operator)
        active_diag_A = diag_A_T_A[diag_A_T_A > 0]
        data_term_scale = np.median(active_diag_A) if active_diag_A.size > 0 else 1.0
        scaled_lambdas = []
        for i, L_item in enumerate(self.regularization_matrices):
            raw_weight = self.regularization_weights[i]
            if raw_weight == 0 or L_item is None:
                scaled_lambdas.append(0.0)
                continue
            diag_L_T_L = self._compute_normal_matrix_diag(L_item)
            active_diag_L = diag_L_T_L[diag_L_T_L > 0]
            reg_term_scale = np.median(active_diag_L) if active_diag_L.size > 0 else 1.0
            scale_factor = (
                math.sqrt(data_term_scale / reg_term_scale) if reg_term_scale > 1e-14 else 0.0
            )
            scaled_lambdas.append(math.sqrt(raw_weight) * scale_factor)
        return scaled_lambdas

    @cached_property
    def data_operator(self) -> ProcessedOperator:
        """Assemble the data operator without regularization."""
        linear_map = self._build_system_linear_map(include_regularization=False)
        shape = (linear_map.shape[0],)
        return ProcessedOperator(linear_map, output_shape=shape, input_shape=self.solution_shape)

    @cached_property
    def dense_system_matrix(self) -> np.ndarray:
        """Assemble the dense system matrix including regularization."""
        all_rows = []
        for i, a_item in enumerate(self.A):
            op = self.densify_op(a_item)
            w_item = self.sqrt_weights[i]
            if w_item:
                w_op = self.densify_op(w_item)
                if w_item.is_diagonal:
                    op = w_op.reshape(-1, 1) * op
                else:
                    op = w_op @ op
            all_rows.append(op)
        lambdas = self.scaled_lambdas
        for i, L_item in enumerate(self.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_rows.append(lambdas[i] * self.densify_op(L_item))
        dtype = self.A[0].dtype if self.A else np.float64
        if not all_rows:
            return xp.zeros((0, self.solution_size), dtype=dtype)
        return xp.vstack(all_rows)

    @cached_property
    def svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the SVD of the dense system matrix."""
        return xp.linalg.svd(self.dense_system_matrix, full_matrices=False)

    def assemble_rhs_block(
        self, b: Union[Any, List[Any]]
    ) -> Tuple[Optional[Any], Tuple[int, ...], int]:
        """Assemble right-hand side block for all scenarios."""
        b_list = b if isinstance(b, list) else [b]
        if len(b_list) != self.num_data_terms:
            raise ValueError(
                f"Number of RHS terms ({len(b_list)}) does not match number of data terms ({self.num_data_terms})."
            )

        processed = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid_b = [p for p in processed if p[0] is not None]
        if not valid_b:
            return None, (), 0
        scenario_shape = valid_b[0][1]

        # Check scenario consistency
        for _, s_shape in valid_b:
            if s_shape != scenario_shape:
                raise ValueError("Inconsistent scenario shapes in b terms.")

        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1

        # Calculate expected number of rows
        op_rows_data = sum(a.num_rows for a in self.A)
        active_lambdas = self.scaled_lambdas
        op_rows_reg = sum(
            L.num_rows
            for i, L in enumerate(self.regularization_matrices)
            if i < len(active_lambdas) and L and active_lambdas[i] > 0
        )
        op_rows = op_rows_data + op_rows_reg

        dtype = self.A[0].dtype if self.A else xp.float64
        blocks = []
        filled_rows = 0
        for i, (b_col_block, _) in enumerate(processed):
            num_a_rows = self.A[i].num_rows
            if b_col_block is None:
                block = xp.zeros((num_a_rows, num_scenarios), dtype=dtype)
            else:
                block = asarray(b_col_block)
                w_item = self.sqrt_weights[i]
                if w_item:
                    block = self._apply_weight(w_item, block)
            blocks.append(block)
            filled_rows += num_a_rows

        if filled_rows < op_rows:
            blocks.append(xp.zeros((op_rows - filled_rows, num_scenarios), dtype=dtype))

        if not blocks:
            d_block = xp.zeros((0, num_scenarios), dtype=dtype)
        else:
            d_block = xp.concatenate(blocks, axis=0)
        return d_block, scenario_shape, num_scenarios

    def get_system_operator(
        self, num_scenarios: int = 1, include_regularization: bool = True
    ) -> LinearOperator:
        """Get system operator for specified number of scenarios."""
        cache_key = (include_regularization, num_scenarios)
        if cache_key in self._scenario_operator_cache:
            return self._scenario_operator_cache[cache_key]
        linear_map = self.get_system_linear_map(
            num_scenarios=num_scenarios, include_regularization=include_regularization
        )
        lin_op = linear_map.as_linear_operator()
        self._scenario_operator_cache[cache_key] = lin_op
        return lin_op

    def get_system_linear_map(
        self, num_scenarios: int = 1, include_regularization: bool = True
    ) -> LinearMap:
        if num_scenarios == 1:
            return self._get_base_system_linear_map(include_regularization)
        cache_key = (include_regularization, num_scenarios)
        if cache_key in self._scenario_linear_map_cache:
            return self._scenario_linear_map_cache[cache_key]
        base_map = self._get_base_system_linear_map(include_regularization)
        lifted = self._lift_linear_map_to_scenarios(base_map, num_scenarios)
        self._scenario_linear_map_cache[cache_key] = lifted
        return lifted

    def _build_system_linear_map(self, include_regularization: bool) -> LinearMap:
        num_features = self.solution_size
        active_lambdas = self.scaled_lambdas if include_regularization else []
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

        def matmat(block: Any) -> Any:
            x_block = asarray(block).reshape(num_features, -1)
            return self._apply_system_block(x_block, include_regularization)

        def rmatmat(block: Any) -> Any:
            y_block = asarray(block).reshape(op_rows, -1)
            return self._apply_system_T_block(y_block, include_regularization)

        def matvec(vec: Any) -> Any:
            return matmat(asarray(vec).reshape(num_features, 1)).ravel()

        def rmatvec(vec: Any) -> Any:
            return rmatmat(asarray(vec).reshape(op_rows, 1)).ravel()

        return LinearMap(
            shape=(op_rows, num_features),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=None,
            source=None,
        )

    def _get_base_system_linear_map(self, include_regularization: bool) -> LinearMap:
        if include_regularization in self._system_linear_map_cache:
            return self._system_linear_map_cache[include_regularization]
        linear_map = self._build_system_linear_map(include_regularization)
        self._system_linear_map_cache[include_regularization] = linear_map
        return linear_map

    def _lift_linear_map_to_scenarios(self, base_map: LinearMap, num_scenarios: int) -> LinearMap:
        base_in = base_map.shape[1]
        base_out = base_map.shape[0]
        dtype = base_map.dtype

        def _apply(block: np.ndarray) -> np.ndarray:
            cols = block.shape[1]
            result = np.zeros((base_out * num_scenarios, cols), dtype=block.dtype)
            for j in range(cols):
                vec = block[:, j]
                scenario_block = vec.reshape(base_in, num_scenarios)
                res = base_map.matmat(scenario_block)
                result[:, j] = np.asarray(res).reshape(-1)
            return result

        def _apply_T(block: np.ndarray) -> np.ndarray:
            cols = block.shape[1]
            result = np.zeros((base_in * num_scenarios, cols), dtype=block.dtype)
            for j in range(cols):
                vec = block[:, j]
                scenario_block = vec.reshape(base_out, num_scenarios)
                res = base_map.rmatmat(scenario_block)
                result[:, j] = np.asarray(res).reshape(-1)
            return result

        def matmat(block: Any) -> Any:
            block_arr = asarray(block).reshape(base_in * num_scenarios, -1)
            return _apply(block_arr)

        def rmatmat(block: Any) -> Any:
            block_arr = asarray(block).reshape(base_out * num_scenarios, -1)
            return _apply_T(block_arr)

        def matvec(vec: Any) -> Any:
            return matmat(asarray(vec).reshape(base_in * num_scenarios, 1)).ravel()

        def rmatvec(vec: Any) -> Any:
            return rmatmat(asarray(vec).reshape(base_out * num_scenarios, 1)).ravel()

        return LinearMap(
            shape=(base_out * num_scenarios, base_in * num_scenarios),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _to_dense=None,
            source=None,
        )

    @staticmethod
    def _compute_normal_matrix_diag(item: ProcessedOperator) -> np.ndarray:
        if item.is_diagonal and item.diag_data is not None:
            return np.abs(item.diag_data.reshape(-1)) ** 2
        map_obj = item.linear_map
        try:
            dense = map_obj.to_dense()
            return np.sum(np.abs(dense) ** 2, axis=0)
        except ValueError:
            n_cols = map_obj.shape[1]
            dtype = np.result_type(map_obj.dtype, np.float64)
            diag = np.zeros(n_cols, dtype=dtype)
            block_size = 32 if n_cols >= 32 else max(1, n_cols)
            block = np.zeros((n_cols, block_size), dtype=dtype)
            for start in range(0, n_cols, block_size):
                stop = min(n_cols, start + block_size)
                cols = stop - start
                block[:, :cols] = 0
                block[start:stop, :cols] = np.eye(cols, dtype=dtype)
                res = map_obj.matmat(block[:, :cols])
                res_arr = asarray(res)
                diag[start:stop] = xp.sum(xp.abs(res_arr) ** 2, axis=0).real
            return diag

    def apply_linear_map_to_block(self, map_obj: LinearMap, x_block: np.ndarray) -> np.ndarray:
        """Apply a LinearMap to a block of vectors and return numpy output."""
        return np.asarray(map_obj.matmat(x_block))

    def apply_linear_map_T_to_block(self, map_obj: LinearMap, y_block: np.ndarray) -> np.ndarray:
        """Apply adjoint of a LinearMap to a block of vectors."""
        return np.asarray(map_obj.rmatmat(y_block))

    def densify_op(self, item: Optional[ProcessedOperator]) -> Optional[np.ndarray]:
        """Convert operator to dense array, if not None."""
        if item is None:
            return None
        if item.is_diagonal and item.diag_data is not None:
            return item.diag_data
        try:
            return item.linear_map.to_dense()
        except ValueError:
            map_obj = item.linear_map
            dtype = np.result_type(map_obj.dtype, np.float64)
            eye = xp.eye(map_obj.shape[1], dtype=dtype)
            return asarray(map_obj.matmat(eye))

    def _get_weight_dense(self, idx: int) -> Optional[np.ndarray]:
        cached = self._sqrt_weight_dense_cache[idx]
        if cached is not None:
            return cached
        item = self.sqrt_weights[idx]
        if not item:
            return None
        dense = self.densify_op(item)
        if dense is None:
            return None
        if item.is_diagonal:
            dense = dense.reshape(-1, 1)
        self._sqrt_weight_dense_cache[idx] = dense
        return dense

    def _apply_weight(self, item: Optional[ProcessedOperator], block: Any) -> Any:
        if item is None:
            return block
        if item.is_diagonal and item.diag_data is not None:
            diag = asarray(item.diag_data).reshape(-1, 1)
            return diag * block

        # Check if we should use dense version
        dense = self.densify_op(item)
        return dense @ block

    def _apply_weight_T(self, item: Optional[ProcessedOperator], block: Any) -> Any:
        if item is None:
            return block
        if item.is_diagonal and item.diag_data is not None:
            diag = asarray(item.diag_data).reshape(-1, 1)
            return xp.conjugate(diag) * block

        dense = self.densify_op(item)
        return xp.conjugate(dense).T @ block

    def apply_linear_map_to_block(self, map_obj: LinearMap, x_block: Any) -> Any:
        """Apply a LinearMap to a block of vectors."""
        return map_obj.matmat(x_block)

    def apply_linear_map_T_to_block(self, map_obj: LinearMap, y_block: Any) -> Any:
        """Apply adjoint of a LinearMap to a block of vectors."""
        return map_obj.rmatmat(y_block)

    def _apply_system_block(self, block: Any, include_regularization: bool) -> Any:
        num_cols = block.shape[1]
        dtype = block.dtype
        op_rows_data = sum(a.num_rows for a in self.A)
        op_rows = op_rows_data
        if include_regularization:
            active_lambdas = self.scaled_lambdas
            op_rows += sum(
                L.num_rows
                for i, L in enumerate(self.regularization_matrices)
                if i < len(active_lambdas) and L and active_lambdas[i] > 0
            )
        else:
            active_lambdas = []
        if op_rows == 0:
            return xp.zeros((0, num_cols), dtype=dtype)
        output_blocks = []
        for i, a_item in enumerate(self.A):
            res_block = self.apply_linear_map_to_block(a_item.linear_map, block)
            w_item = self.sqrt_weights[i]
            if w_item:
                res_block = self._apply_weight(w_item, res_block)
            output_blocks.append(res_block)
        if include_regularization and active_lambdas:
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                    res_block = self.apply_linear_map_to_block(L_item.linear_map, block)
                    output_blocks.append(active_lambdas[i] * res_block)
        if not output_blocks:
            return xp.zeros((op_rows, num_cols), dtype=dtype)
        return xp.vstack(output_blocks)

    def _apply_system_T_block(self, block: Any, include_regularization: bool) -> Any:
        num_cols = block.shape[1]
        dtype = block.dtype
        accum = xp.zeros((self.solution_size, num_cols), dtype=dtype)
        row = 0
        for i, a_item in enumerate(self.A):
            num_rows = a_item.num_rows
            part = block[row : row + num_rows, :]
            w_item = self.sqrt_weights[i]
            if w_item:
                part = self._apply_weight_T(w_item, part)
            accum = accum + self.apply_linear_map_T_to_block(a_item.linear_map, part)
            row += num_rows

        if include_regularization:
            active_lambdas = self.scaled_lambdas
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                    num_rows = L_item.num_rows
                    part = block[row : row + num_rows, :]
                    accum = accum + active_lambdas[i] * self.apply_linear_map_T_to_block(
                        L_item.linear_map, part
                    )
                    row += num_rows
        return accum

    def _flatten_operator(
        self,
        op: OperatorInput,
        output_shape: Tuple[int, ...] = None,
        input_shape: Tuple[int, ...] = None,
    ) -> ProcessedOperator:
        if isinstance(op, TensorChain):
            if self.matrix_free:
                linear_map = as_linear_map(op)
                return ProcessedOperator(
                    linear_map=linear_map, output_shape=op.output_shape, input_shape=op.input_shape
                )
            # For dense, we trust TensorChain.to_dense matches shapes
            dense_op = op.to_dense()
            flat_out = math.prod(op.output_shape)
            flat_in = math.prod(op.input_shape)
            dense_matrix = dense_op.reshape(flat_out, flat_in)
            linear_map = as_linear_map(dense_matrix)
            return ProcessedOperator(
                linear_map=linear_map, output_shape=op.output_shape, input_shape=op.input_shape
            )

        if isinstance(op, LinearOperator):
            linear_map = as_linear_map(op)
            return ProcessedOperator(
                linear_map=linear_map, output_shape=(op.shape[0],), input_shape=(op.shape[1],)
            )

        if isinstance(op, np.ndarray):
            array = np.ascontiguousarray(op)

            # Infer missing shapes if possible
            if output_shape is None and input_shape is not None:
                flat_in = math.prod(input_shape)
                if array.size % flat_in == 0:
                    flat_out = array.size // flat_in
                    # If square scaling (scalar/diagonal), output_shape == input_shape logic
                    # But here we just need a valid shape tuple.
                    # For regularization matrices (usually square), we might assume output=input if size matches
                    if flat_out == flat_in:
                        output_shape = input_shape
                    else:
                        output_shape = (flat_out,)
            elif input_shape is None and output_shape is not None:
                flat_out = math.prod(output_shape)
                if array.size % flat_out == 0:
                    flat_in = array.size // flat_out
                    input_shape = (flat_in,)

            if output_shape is None or input_shape is None:
                raise ValueError(f"Could not infer shapes for array of size {array.size}.")

            flat_out = math.prod(output_shape)
            flat_in = math.prod(input_shape)
            if array.size != flat_out * flat_in:
                raise ValueError(
                    f"Array size {array.size} mismatch with shapes {output_shape}->{input_shape} (prod={flat_out * flat_in})"
                )

            reshaped = array.reshape(flat_out, flat_in)
            linear_map = as_linear_map(reshaped)
            return ProcessedOperator(
                linear_map=linear_map, output_shape=output_shape, input_shape=input_shape
            )
        if scipy.sparse.issparse(op):
            linear_map = as_linear_map(op)
            return ProcessedOperator(
                linear_map=linear_map, output_shape=(op.shape[0],), input_shape=(op.shape[1],)
            )

        raise TypeError(
            f"Input must be a numpy array, TensorChain, LinearOperator, or sparse matrix, got {type(op)}"
        )

    def _process_b_vector(
        self, b_val: Any, data_shape: Tuple[int, ...]
    ) -> Tuple[Optional[Any], Optional[Tuple[int, ...]]]:
        if b_val is None:
            return None, None
        b = asarray(b_val)
        flat_data_size = math.prod(data_shape)

        # Determine scenarios
        total_size = b.size
        if total_size % flat_data_size != 0:
            raise ValueError(f"b size {total_size} not divisible by data size {flat_data_size}.")

        num_scenarios = total_size // flat_data_size
        b_col_block = xp.reshape(b, (flat_data_size, num_scenarios))

        # Infer scenario shape if possible, otherwise flat
        scenario_shape = (num_scenarios,) if num_scenarios > 1 else ()
        return b_col_block, scenario_shape
