"""Least-squares problem definition."""

from __future__ import annotations
import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator

from pynamit.math.tensor_chain import TensorChain
from pynamit.utils import use_jax

OperatorInput: TypeAlias = Union[np.ndarray, LinearOperator, TensorChain]
OperatorInputList: TypeAlias = Union[OperatorInput, List[OperatorInput]]
NumericInputList: TypeAlias = Union[float, List[float]]


def cached_property(func: Callable):
    """Cache a propertu."""
    return property(functools.lru_cache(maxsize=None)(func))


@dataclass
class ProcessedOperator:
    """A processed operator with associated shape information."""

    op: Union[np.ndarray, LinearOperator]
    output_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    is_diagonal: bool = False
    tensor_chain: Optional[TensorChain] = None
    jax_dense: Optional[Any] = None

    @property
    def num_rows(self) -> int:
        """Number of rows in the operator."""
        return self.op.shape[0]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the operator."""
        if self.tensor_chain is not None:
            return self.tensor_chain.dtype
        return self.op.dtype


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
        matrix_free: bool = True,
    ):
        self.matrix_free = matrix_free
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)
        self._process_data_terms(A, data_shapes, sqrt_weights)
        self._process_regularization_terms(regularization_matrices, regularization_weights)
        self._system_operator_cache: dict[bool, LinearOperator] = {}
        self._scenario_operator_cache: dict[Tuple[bool, int], LinearOperator] = {}

    @staticmethod
    def _backend_module():
        """Return the active array module (NumPy or JAX NumPy)."""
        if use_jax():
            import jax.numpy as jnp  # Lazy import to avoid hard dependency

            return jnp
        return np

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
        self._sqrt_weight_dense_cache: List[Optional[np.ndarray]] = [None] * self.num_data_terms

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
        if not isinstance(w_val, LinearOperator):
            arr = np.ascontiguousarray(w_val)
            is_diagonal = (arr.ndim == 1 and arr.size == flat_dim) or (arr.shape == shape)
            if is_diagonal:
                return ProcessedOperator(
                    op=arr.reshape(flat_dim, 1),
                    output_shape=shape,
                    input_shape=shape,
                    is_diagonal=True,
                )
        return self._flatten_operator(w_val, output_shape=shape, input_shape=shape)

    @cached_property
    def scaled_lambdas(self) -> List[float]:
        """Compute scaled regularization weights."""
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
        """Assemble the data operator without regularization."""
        op = self._build_system_operator(include_regularization=False)
        shape = (op.shape[0],)
        return ProcessedOperator(op, output_shape=shape, input_shape=self.solution_shape)

    @cached_property
    def dense_system_matrix(self) -> np.ndarray:
        """Assemble the dense system matrix including regularization."""
        all_rows = []
        for i, a_item in enumerate(self.A):
            op = self.densify_op(a_item)
            w_item = self.sqrt_weights[i]
            if w_item:
                w_op = self.densify_op(w_item)
                op = w_op * op if w_item.is_diagonal else w_op @ op
            all_rows.append(op)
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
        """Compute the SVD of the dense system matrix."""
        return np.linalg.svd(self.dense_system_matrix, full_matrices=False)

    def assemble_rhs_block(
        self, b: Union[Any, List[Any]]
    ) -> Tuple[Optional[Any], Tuple[int, ...], int]:
        """Assemble right-hand side block for all scenarios."""
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed = [
            self._process_b_vector(b_val, self.data_shapes[i])
            for i, b_val in enumerate(b_list)
        ]
        valid_b = [p for p in processed if p[0] is not None]
        if not valid_b:
            return None, (), 0
        scenario_shape = valid_b[0][1]
        if not all(p[1] == scenario_shape for p in valid_b):
            raise ValueError("Inconsistent scenario shapes in b terms.")
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        backend = self._backend_module()
        op_rows = self._jax_operator_row_count(include_regularization=True)
        dtype = self.A[0].dtype if self.A else backend.float64
        blocks = []
        filled_rows = 0
        for i, (b_col_block, _) in enumerate(processed):
            num_a_rows = self.A[i].num_rows
            if b_col_block is None:
                block = backend.zeros((num_a_rows, num_scenarios), dtype=dtype)
            else:
                block = backend.asarray(b_col_block)
                w_item = self.sqrt_weights[i]
                if w_item:
                    if use_jax():
                        block = self._jax_apply_weight(w_item, block)
                    else:
                        block_np = np.asarray(block)
                        block_np = self._apply_weight_numpy(i, block_np)
                        block = backend.asarray(block_np)
            blocks.append(block)
            filled_rows += num_a_rows

        if filled_rows < op_rows:
            blocks.append(backend.zeros((op_rows - filled_rows, num_scenarios), dtype=dtype))

        if not blocks:
            d_block = backend.zeros((0, num_scenarios), dtype=dtype)
        else:
            d_block = backend.concatenate(blocks, axis=0)
        return d_block, scenario_shape, num_scenarios

    def get_system_operator(
        self, num_scenarios: int = 1, include_regularization: bool = True
    ) -> LinearOperator:
        """Get system operator for specified number of scenarios."""
        op_block = self._get_base_system_operator(include_regularization)
        if num_scenarios == 1:
            return op_block
        cache_key = (include_regularization, num_scenarios)
        if cache_key in self._scenario_operator_cache:
            return self._scenario_operator_cache[cache_key]
        op_rows, num_features = op_block.shape
        shape = (op_rows * num_scenarios, num_features * num_scenarios)
        dtype = op_block.dtype

        def matvec_block(x_block):
            return self.apply_op_to_block(op_block, x_block)

        def rmatvec_block(y_block):
            return self.apply_op_T_to_block(op_block, y_block)

        def matvec_final(x_flat: np.ndarray) -> np.ndarray:
            x_block = x_flat.reshape(num_features, num_scenarios)
            return matvec_block(x_block).ravel()

        def rmatvec_final(y_flat: np.ndarray) -> np.ndarray:
            y_block = y_flat.reshape(op_rows, num_scenarios)
            return rmatvec_block(y_block).ravel()

        lifted = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        self._scenario_operator_cache[cache_key] = lifted
        return lifted

    def _build_system_operator(self, include_regularization: bool) -> LinearOperator:
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

        def matvec(x: np.ndarray) -> np.ndarray:
            x_block = x.reshape(-1, 1)
            output_blocks: List[np.ndarray] = []
            for i, a_item in enumerate(self.A):
                res_block = self.apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item:
                    res_block = self._apply_weight_numpy(i, res_block)
                output_blocks.append(res_block)
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                        output_blocks.append(
                            active_lambdas[i] * self.apply_op_to_block(L_item.op, x_block)
                        )
            if not output_blocks:
                return np.zeros((op_rows,), dtype=dtype)
            return np.vstack(output_blocks).ravel()

        def rmatvec(y: np.ndarray) -> np.ndarray:
            y_block = y.reshape(-1, 1)
            x_block = np.zeros((num_features, 1), dtype=y.dtype)
            row = 0
            for i, a_item in enumerate(self.A):
                y_part = y_block[row : row + a_item.num_rows, :]
                w_item = self.sqrt_weights[i]
                if w_item:
                    y_part = self._apply_weight_T_numpy(i, y_part)
                x_block += self.apply_op_T_to_block(a_item.op, y_part)
                row += a_item.num_rows
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                        y_part = y_block[row : row + L_item.num_rows, :]
                        x_block += active_lambdas[i] * self.apply_op_T_to_block(L_item.op, y_part)
                        row += L_item.num_rows
            return x_block.ravel()

        return LinearOperator((op_rows, num_features), matvec=matvec, rmatvec=rmatvec, dtype=dtype)

    def _get_base_system_operator(self, include_regularization: bool) -> LinearOperator:
        if include_regularization in self._system_operator_cache:
            return self._system_operator_cache[include_regularization]
        op = self._build_system_operator(include_regularization)
        self._system_operator_cache[include_regularization] = op
        return op

    @staticmethod
    def _compute_normal_matrix_diag(op: Union[LinearOperator, np.ndarray]) -> np.ndarray:
        if isinstance(op, np.ndarray):
            return np.sum(np.abs(op) ** 2, axis=0)
        tensor_chain = getattr(op, "_tensor_chain", None)
        if tensor_chain is not None:
            dense = tensor_chain.to_dense()
            return np.sum(np.abs(dense) ** 2, axis=0)
        n_cols = op.shape[1]
        dtype = np.result_type(op.dtype, np.float64)
        diag = np.zeros(n_cols, dtype=dtype)
        block_size = 32 if n_cols >= 32 else max(1, n_cols)
        block = np.zeros((n_cols, block_size), dtype=op.dtype)
        for start in range(0, n_cols, block_size):
            stop = min(n_cols, start + block_size)
            cols = stop - start
            block[:, :cols] = 0
            block[start:stop, :cols] = np.eye(cols, dtype=op.dtype)
            res = op.matmat(block[:, :cols])
            diag[start:stop] = np.sum(np.abs(res) ** 2, axis=0).real
        return diag

    def apply_op_to_block(
        self, op: Union[np.ndarray, LinearOperator], x_block: np.ndarray
    ) -> np.ndarray:
        """Apply operator to a block of vectors."""
        if isinstance(op, LinearOperator):
            return op.matmat(x_block)
        return op @ x_block

    def apply_op_T_to_block(
        self, op: Union[np.ndarray, LinearOperator], y_block: np.ndarray
    ) -> np.ndarray:
        """Apply adjoint to a block of vectors."""
        if isinstance(op, LinearOperator):
            return op.rmatmat(y_block)
        return op.T.conj() @ y_block

    def densify_op(self, item: Optional[ProcessedOperator]) -> Optional[np.ndarray]:
        """Convert operator to dense numpy array, if not None."""
        if item is None:
            return None
        if item.tensor_chain is not None and not isinstance(item.op, np.ndarray):
            return item.tensor_chain.to_dense()
        op = item.op
        if isinstance(op, LinearOperator):
            eye = np.eye(op.shape[1], dtype=op.dtype)
            return op.matmat(eye)
        return op

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

    def _apply_weight_numpy(self, idx: int, block: np.ndarray) -> np.ndarray:
        dense = self._get_weight_dense(idx)
        item = self.sqrt_weights[idx]
        if dense is None or not item:
            return block
        if item.is_diagonal:
            return dense * block
        return dense @ block

    def _apply_weight_T_numpy(self, idx: int, block: np.ndarray) -> np.ndarray:
        dense = self._get_weight_dense(idx)
        item = self.sqrt_weights[idx]
        if dense is None or not item:
            return block
        if item.is_diagonal:
            return dense.conj() * block
        return dense.T.conj() @ block

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
            if self.matrix_free:
                lin_op = op.as_linear_operator()
                return ProcessedOperator(
                    op=lin_op,
                    output_shape=op.output_shape,
                    input_shape=op.input_shape,
                    tensor_chain=op,
                )
            dense_op = op.to_dense()
            return ProcessedOperator(
                op=dense_op, output_shape=op.output_shape, input_shape=op.input_shape
            )
        if isinstance(op, LinearOperator):
            return ProcessedOperator(
                op=op, output_shape=(op.shape[0],), input_shape=(op.shape[1],)
            )
        if isinstance(op, np.ndarray):
            array = np.ascontiguousarray(op)
            if input_shape is None and output_shape is None:
                raise ValueError("At least one shape must be provided for numpy arrays.")
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
    ) -> Tuple[Optional[Any], Optional[Tuple[int, ...]]]:
        if b_val is None:
            return None, None
        module = self._backend_module()
        b = module.asarray(b_val)
        num_data_dims = len(data_shape)
        is_exact = b.shape == data_shape
        is_multi = b.ndim > num_data_dims and b.shape[:num_data_dims] == data_shape
        is_flat = b.ndim == 1 and b.size == math.prod(data_shape)
        if not (is_exact or is_multi or is_flat):
            raise ValueError(f"Shape {b.shape} incompatible with data_shape {data_shape}.")
        scenario_shape = b.shape[num_data_dims:] if is_multi else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = module.reshape(b, (math.prod(data_shape), num_scenarios))
        return b_col_block, scenario_shape

    # ----- JAX Helpers -----

    def _jax_operator_row_count(self, include_regularization: bool) -> int:
        op_rows_data = sum(a.num_rows for a in self.A)
        if not include_regularization:
            return op_rows_data
        active_lambdas = self.scaled_lambdas
        op_rows_reg = sum(
            L.num_rows
            for i, L in enumerate(self.regularization_matrices)
            if i < len(active_lambdas) and L and active_lambdas[i] > 0
        )
        return op_rows_data + op_rows_reg

    def _jax_ensure_dense(self, item: ProcessedOperator):
        import jax.numpy as jnp

        if item.jax_dense is not None:
            return item.jax_dense
        if isinstance(item.op, np.ndarray):
            item.jax_dense = jnp.asarray(item.op)
            return item.jax_dense
        dense_np = self.densify_op(item)
        item.jax_dense = jnp.asarray(dense_np)
        return item.jax_dense

    def _jax_apply_processed_operator(
        self, item: ProcessedOperator, x_block: "jax.numpy.ndarray"
    ):
        import jax.numpy as jnp

        if item.tensor_chain is not None:
            return item.tensor_chain.matmat(x_block)
        dense = self._jax_ensure_dense(item)
        return dense @ x_block

    def _jax_apply_processed_operator_T(
        self, item: ProcessedOperator, y_block: "jax.numpy.ndarray"
    ):
        import jax.numpy as jnp

        if item.tensor_chain is not None:
            return item.tensor_chain.rmatmat(y_block)
        dense = self._jax_ensure_dense(item)
        return dense.T.conj() @ y_block

    def _jax_apply_weight(
        self, item: ProcessedOperator, block: "jax.numpy.ndarray"
    ) -> "jax.numpy.ndarray":
        import jax.numpy as jnp

        if item.is_diagonal:
            diag = jnp.asarray(item.op).reshape(-1, 1)
            return diag * block
        return self._jax_apply_processed_operator(item, block)

    def _jax_apply_weight_T(
        self, item: ProcessedOperator, block: "jax.numpy.ndarray"
    ) -> "jax.numpy.ndarray":
        import jax.numpy as jnp

        if item.is_diagonal:
            diag = jnp.asarray(item.op).reshape(-1, 1)
            return diag.conj() * block
        return self._jax_apply_processed_operator_T(item, block)

    def _jax_apply_system(
        self, x_block: "jax.numpy.ndarray", include_regularization: bool
    ) -> "jax.numpy.ndarray":
        import jax.numpy as jnp

        outputs = []
        for i, a_item in enumerate(self.A):
            res_block = self._jax_apply_processed_operator(a_item, x_block)
            w_item = self.sqrt_weights[i]
            if w_item:
                res_block = self._jax_apply_weight(w_item, res_block)
            outputs.append(res_block)
        if include_regularization:
            active_lambdas = self.scaled_lambdas
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                    res_block = active_lambdas[i] * self._jax_apply_processed_operator(
                        L_item, x_block
                    )
                    outputs.append(res_block)
        if not outputs:
            return jnp.zeros((0, x_block.shape[1]), dtype=x_block.dtype)
        return jnp.concatenate(outputs, axis=0)

    def _jax_apply_system_T(
        self, y_block: "jax.numpy.ndarray", include_regularization: bool
    ) -> "jax.numpy.ndarray":
        import jax.numpy as jnp

        num_features = self.solution_size
        accum = jnp.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
        row = 0
        for i, a_item in enumerate(self.A):
            num_rows = a_item.num_rows
            part = y_block[row : row + num_rows, :]
            w_item = self.sqrt_weights[i]
            if w_item:
                part = self._jax_apply_weight_T(w_item, part)
            accum = accum + self._jax_apply_processed_operator_T(a_item, part)
            row += num_rows
        if include_regularization:
            active_lambdas = self.scaled_lambdas
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(active_lambdas) and L_item and active_lambdas[i] > 0:
                    num_rows = L_item.num_rows
                    part = y_block[row : row + num_rows, :]
                    accum = accum + active_lambdas[i] * self._jax_apply_processed_operator_T(
                        L_item, part
                    )
                    row += num_rows
        return accum

    def jax_system_matvec(self, include_regularization: bool = True):
        if not use_jax():
            raise RuntimeError("JAX backend is not enabled.")
        import jax.numpy as jnp

        def matvec(x_flat):
            x_block = jnp.reshape(x_flat, (self.solution_size, -1))
            out_block = self._jax_apply_system(x_block, include_regularization)
            return jnp.reshape(out_block, (-1,))

        return matvec

    def jax_system_rmatvec(self, include_regularization: bool = True):
        if not use_jax():
            raise RuntimeError("JAX backend is not enabled.")
        import jax.numpy as jnp

        num_rows = self._jax_operator_row_count(include_regularization)

        def rmatvec(y_flat):
            y_block = jnp.reshape(y_flat, (num_rows, -1))
            out_block = self._jax_apply_system_T(y_block, include_regularization)
            return jnp.reshape(out_block, (-1,))

        return rmatvec

    def jax_normal_matvec(self, include_regularization: bool = True):
        if not use_jax():
            raise RuntimeError("JAX backend is not enabled.")
        import jax.numpy as jnp

        def normal_matvec(x_flat):
            x_block = jnp.reshape(x_flat, (self.solution_size, -1))
            forward = self._jax_apply_system(x_block, include_regularization)
            adjoint = self._jax_apply_system_T(forward, include_regularization)
            return jnp.reshape(adjoint, (-1,))

        return normal_matvec
