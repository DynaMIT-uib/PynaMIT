"""Least-squares problem definition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator

from pynamit.math.linear_map import LinearMap, as_linear_map, diagonal_linear_map
from pynamit.math.tensor_chain import TensorChain
from pynamit.utils import asarray, get_array_module

OperatorInput: TypeAlias = Union[np.ndarray, LinearOperator, TensorChain, LinearMap]
OperatorInputList: TypeAlias = Union[OperatorInput, List[OperatorInput]]
NumericInputList: TypeAlias = Union[float, List[float]]


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

    def apply(self, block: Any) -> Any:
        """Apply this operator to a vector block."""
        return self.linear_map.matmat(block)

    def apply_adjoint(self, block: Any) -> Any:
        """Apply this operator's adjoint to a vector block."""
        return self.linear_map.rmatmat(block)

    def to_dense(self) -> np.ndarray:
        """Return a dense representation."""
        if self.is_diagonal and self.diag_data is not None:
            return np.asarray(self.diag_data).reshape(-1)
        try:
            return self.linear_map.to_dense()
        except ValueError:
            eye = np.eye(
                self.linear_map.shape[1], dtype=np.result_type(self.linear_map.dtype, np.float64)
            )
            return np.asarray(self.linear_map.matmat(eye))

    def normal_matrix_diag(self) -> np.ndarray:
        """Compute ``diag(A* A)`` without requiring a dense matrix."""
        return self.linear_map.normal_matrix_diag()


class LeastSquaresProblem:
    """Defines the mathematical structure of a least-squares problem."""

    def __init__(
        self,
        A: OperatorInputList,
        solution_shape: Union[int, Tuple[int, ...]],
        data_shapes: Union[Any, List[Any]],
        sqrt_weights: Optional[Union[Any, List[Any]]] = None,
        regularization_weights: Optional[NumericInputList] = None,
        regularization_matrices: Optional[OperatorInputList] = None,
    ):
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)
        self._system_linear_map_cache: dict[bool, LinearMap] = {}

        self._process_data_terms(A, data_shapes, sqrt_weights)
        self._process_regularization_terms(regularization_matrices, regularization_weights)

    def _process_data_terms(self, A_in, data_shapes_in, sqrt_weights_in):
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
        if w_val is None:
            return None
        flat_dim = math.prod(shape)
        if not isinstance(w_val, (LinearMap, LinearOperator, TensorChain)) and not (
            scipy.sparse.issparse(w_val)
        ):
            arr = np.ascontiguousarray(w_val)
            is_diagonal = (arr.ndim == 1 and arr.size == flat_dim) or (arr.shape == shape)
            if is_diagonal:
                diag_data = arr.reshape(flat_dim)
                return ProcessedOperator(
                    linear_map=diagonal_linear_map(diag_data),
                    output_shape=shape,
                    input_shape=shape,
                    is_diagonal=True,
                    diag_data=diag_data,
                )
        return self._flatten_operator(w_val, output_shape=shape, input_shape=shape)

    @cached_property
    def scaled_lambdas(self) -> List[float]:
        """Compute scaled regularization weights."""
        diag_A_T_A = self.data_operator.normal_matrix_diag()
        active_diag_A = diag_A_T_A[diag_A_T_A > 0]
        data_term_scale = np.median(active_diag_A) if active_diag_A.size > 0 else 1.0
        scaled_lambdas = []
        for i, L_item in enumerate(self.regularization_matrices):
            raw_weight = self.regularization_weights[i]
            if raw_weight == 0 or L_item is None:
                scaled_lambdas.append(0.0)
                continue
            diag_L_T_L = L_item.normal_matrix_diag()
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
        linear_map = self._get_base_system_linear_map(include_regularization=False)
        shape = (linear_map.shape[0],)
        return ProcessedOperator(linear_map, output_shape=shape, input_shape=self.solution_shape)

    @cached_property
    def dense_system_matrix(self) -> np.ndarray:
        """Assemble the dense system matrix including regularization."""
        all_rows = []
        for i, a_item in enumerate(self.A):
            op = a_item.to_dense()
            w_item = self.sqrt_weights[i]
            if w_item:
                w_op = w_item.to_dense()
                op = w_op.reshape(-1, 1) * op if w_item.is_diagonal else w_op @ op
            all_rows.append(op)
        lambdas = self.scaled_lambdas
        for i, L_item in enumerate(self.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_rows.append(lambdas[i] * L_item.to_dense())
        dtype = self.A[0].dtype if self.A else np.float64
        if not all_rows:
            return np.zeros((0, self.solution_size), dtype=dtype)
        return np.vstack([np.asarray(row) for row in all_rows])

    @cached_property
    def svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the SVD of the dense system matrix."""
        return np.linalg.svd(self.dense_system_matrix, full_matrices=False)

    def assemble_rhs_block(
        self, b: Union[Any, List[Any]]
    ) -> Tuple[Optional[np.ndarray], Tuple[int, ...], int]:
        """Assemble one or more right-hand side columns."""
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid_b = [p for p in processed if p[0] is not None]
        if not valid_b:
            return None, (), 0
        rhs_shape = valid_b[0][1]
        if not all(p[1] == rhs_shape for p in valid_b):
            raise ValueError("Inconsistent RHS column shapes in b terms.")

        num_rhs = math.prod(rhs_shape) if rhs_shape else 1
        op_rows = self.get_system_linear_map(include_regularization=True).shape[0]
        dtype = self.A[0].dtype if self.A else np.float64
        d_block = np.zeros((op_rows, num_rhs), dtype=dtype)

        row = 0
        for i, (b_col_block, _) in enumerate(processed):
            num_a_rows = self.A[i].num_rows
            if b_col_block is not None:
                w_item = self.sqrt_weights[i]
                if w_item:
                    b_col_block = self._apply_weight(w_item, b_col_block)
                d_block[row : row + num_a_rows, :] = np.asarray(b_col_block)
            row += num_a_rows
        return d_block, rhs_shape, num_rhs

    def get_system_operator(self, include_regularization: bool = True) -> LinearOperator:
        """Get a SciPy operator for the base least-squares system."""
        return self.get_system_linear_map(
            include_regularization=include_regularization
        ).as_linear_operator()

    def get_system_linear_map(self, include_regularization: bool = True) -> LinearMap:
        """Get the base ``LinearMap`` system operator."""
        return self._get_base_system_linear_map(include_regularization)

    def _build_system_linear_map(self, include_regularization: bool) -> LinearMap:
        num_features = self.solution_size
        active_lambdas = self.scaled_lambdas if include_regularization else []
        op_rows_data = sum(a.num_rows for a in self.A)
        op_rows_reg = 0
        if include_regularization:
            op_rows_reg = sum(
                L.num_rows
                for i, L in enumerate(self.regularization_matrices)
                if i < len(active_lambdas) and L and active_lambdas[i] > 1e-12
            )
        op_rows = op_rows_data + op_rows_reg
        dtype = self.A[0].dtype if self.A else np.float64

        def matmat(block: Any) -> Any:
            block_arr = asarray(block).reshape(num_features, -1)
            return self._apply_system_block(block_arr, include_regularization)

        def rmatmat(block: Any) -> Any:
            block_arr = asarray(block).reshape(op_rows, -1)
            return self._apply_system_T_block(block_arr, include_regularization)

        def matvec(vec: Any) -> Any:
            return matmat(asarray(vec).reshape(num_features, 1)).ravel()

        def rmatvec(vec: Any) -> Any:
            return rmatmat(asarray(vec).reshape(op_rows, 1)).ravel()

        def normal_matrix_diag() -> np.ndarray:
            diag = np.zeros(num_features, dtype=np.result_type(dtype, np.float64))
            for i, a_item in enumerate(self.A):
                term_map = a_item.linear_map
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    term_map = w_item.linear_map @ term_map
                diag += term_map.normal_matrix_diag()

            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 1e-12:
                        diag += np.abs(active_lambdas[i]) ** 2 * L_item.normal_matrix_diag()
            return diag

        return LinearMap(
            shape=(op_rows, num_features),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _normal_matrix_diag=normal_matrix_diag,
            source=None,
        )

    def _get_base_system_linear_map(self, include_regularization: bool) -> LinearMap:
        if include_regularization not in self._system_linear_map_cache:
            self._system_linear_map_cache[include_regularization] = self._build_system_linear_map(
                include_regularization
            )
        return self._system_linear_map_cache[include_regularization]

    def _apply_weight(self, item: Optional[ProcessedOperator], block: Any) -> Any:
        return block if item is None else item.apply(block)

    def _apply_weight_T(self, item: Optional[ProcessedOperator], block: Any) -> Any:
        return block if item is None else item.apply_adjoint(block)

    def _apply_system_block(self, block: Any, include_regularization: bool) -> Any:
        xp = get_array_module(block)
        num_cols = block.shape[1]
        active_lambdas = self.scaled_lambdas if include_regularization else []
        op_rows = sum(a.num_rows for a in self.A)
        if include_regularization:
            op_rows += sum(
                L.num_rows
                for i, L in enumerate(self.regularization_matrices)
                if i < len(active_lambdas) and L and active_lambdas[i] > 1e-12
            )
        if op_rows == 0:
            return xp.zeros((0, num_cols), dtype=block.dtype)

        output_blocks = []
        for i, a_item in enumerate(self.A):
            res_block = a_item.apply(block)
            output_blocks.append(self._apply_weight(self.sqrt_weights[i], res_block))
        if include_regularization:
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 1e-12:
                    res_block = L_item.apply(block)
                    output_blocks.append(active_lambdas[i] * res_block)
        return xp.vstack(output_blocks) if output_blocks else xp.zeros((op_rows, num_cols))

    def _apply_system_T_block(self, block: Any, include_regularization: bool) -> Any:
        xp = get_array_module(block)
        num_cols = block.shape[1]
        accum = xp.zeros((self.solution_size, num_cols), dtype=block.dtype)
        row = 0
        for i, a_item in enumerate(self.A):
            part = block[row : row + a_item.num_rows, :]
            part = self._apply_weight_T(self.sqrt_weights[i], part)
            accum = accum + a_item.apply_adjoint(part)
            row += a_item.num_rows

        if include_regularization:
            active_lambdas = self.scaled_lambdas
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 1e-12:
                    part = block[row : row + L_item.num_rows, :]
                    accum = accum + active_lambdas[i] * L_item.apply_adjoint(part)
                    row += L_item.num_rows
        return accum

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
                raise ValueError(f"Input '{name}' cannot be None.")
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
        return [(s,) if isinstance(s, int) else tuple(s) for s in data_shapes]

    def _flatten_operator(
        self,
        op: OperatorInput,
        output_shape: Tuple[int, ...] = None,
        input_shape: Tuple[int, ...] = None,
    ) -> ProcessedOperator:
        if isinstance(op, TensorChain):
            linear_map = as_linear_map(op)
            return ProcessedOperator(
                linear_map=linear_map, output_shape=op.output_shape, input_shape=op.input_shape
            )

        linear_map = as_linear_map(op, input_shape=input_shape, output_shape=output_shape)
        output_shape = output_shape if output_shape is not None else (linear_map.shape[0],)
        input_shape = input_shape if input_shape is not None else (linear_map.shape[1],)
        return ProcessedOperator(linear_map, output_shape, input_shape)

    def _process_b_vector(
        self, b_val: Any, data_shape: Tuple[int, ...]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, ...]]]:
        if b_val is None:
            return None, None
        b = np.ascontiguousarray(b_val)
        flat_data_size = math.prod(data_shape)
        num_data_dims = len(data_shape)

        if b.shape == data_shape:
            return b.reshape(flat_data_size, 1), ()

        if b.ndim > num_data_dims and tuple(b.shape[:num_data_dims]) == data_shape:
            rhs_shape = b.shape[num_data_dims:]
            return b.reshape(flat_data_size, math.prod(rhs_shape)), rhs_shape

        if b.ndim > num_data_dims and tuple(b.shape[-num_data_dims:]) == data_shape:
            rhs_shape = b.shape[:-num_data_dims]
            return b.reshape(math.prod(rhs_shape), flat_data_size).T, rhs_shape

        if b.ndim == 1 and b.size == flat_data_size:
            return b.reshape(flat_data_size, 1), ()

        if b.size % flat_data_size != 0:
            raise ValueError(f"Shape {b.shape} incompatible with data_shape {data_shape}.")
        num_rhs = b.size // flat_data_size
        rhs_shape = (num_rhs,) if num_rhs > 1 else ()
        return b.reshape(flat_data_size, num_rhs), rhs_shape
