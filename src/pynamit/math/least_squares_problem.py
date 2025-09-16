"""
Defines a "lean" least-squares problem, responsible only for assembling
the core system operator G and right-hand-side vector d.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Union
from scipy.sparse.linalg import LinearOperator

from pynamit.math.tensor_chain import TensorChain


@dataclass
class _ProcessedItem:
    """A private helper class to store a processed operator and its shape info."""
    op: "Union[np.ndarray, LinearOperator]"
    output_shape: tuple
    input_shape: tuple


class LeastSquaresProblem:
    """
    Defines the mathematical components of a least-squares problem.

    This class is a "lean" representation, responsible for assembling the core
    system operator G and the right-hand-side vector d. It does not contain
    logic for solving or preconditioning; that is the responsibility of a
    LeastSquaresSolver. It features a versioning system to allow external
    solvers to safely cache derived components.
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
        """
        Initializes the least-squares problem definition.

        Args:
            A: The system matrix or a list of matrices. Can be np.ndarray,
               TensorChain, or LinearOperator.
            solution_shape: The shape of the solution vector 'x'.
            data_shapes: The output shape for each operator in A.
            sqrt_weights: Optional weights applied to the data terms.
            regularization_weights: Optional scalar weights for regularization terms.
            regularization_matrices: Optional regularization operators 'L'.
            matrix_free: If True, operators like TensorChain will be kept as
                         LinearOperators. If False, they will be densified.
        """
        self._version = 0
        self._op_cache = {}
        self.matrix_free = matrix_free
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)

        self.update_matrices(A, sqrt_weights=sqrt_weights, data_shapes=data_shapes)

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

    def update_matrices(self, A, sqrt_weights=None, data_shapes=None) -> None:
        """Updates the main data operators (A) and weights of the problem."""
        A_list = self._prepare_input_list(A, "A")
        self.num_data_terms = len(A_list)

        if data_shapes is not None:
            self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        elif not hasattr(self, "data_shapes") or len(self.data_shapes) != self.num_data_terms:
            raise ValueError(
                "data_shapes must be provided when setting A for the first time or changing number of A operators."
            )

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
            is_diagonal = (
                not isinstance(w_val, LinearOperator) and np.asarray(w_val).size == flat_data_dim
            )
            if is_diagonal:
                w_op = np.ascontiguousarray(w_val).reshape(flat_data_dim, 1)
                self.sqrt_weights.append(
                    _ProcessedItem(op=w_op, output_shape=self.data_shapes[i], input_shape=(1,))
                )
            else:
                self.sqrt_weights.append(
                    self._flatten(
                        w_val, output_shape=self.data_shapes[i], input_shape=self.data_shapes[i]
                    )
                )

        self.clear_cache()

    def clear_cache(self) -> None:
        """
        Clears problem-specific caches and increments the version, signaling
        to external solvers that their caches for this problem are stale.
        """
        self._op_cache.clear()
        self._version += 1

    # --- Public API for Solvers ---

    def assemble_rhs_block(self, b: Union[Any, List[Any]]) -> Tuple[Optional[np.ndarray], tuple, int]:
        """Assembles the right-hand-side (RHS) vector 'd' from input vector(s) 'b'."""
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
        lambdas = self.get_scaled_lambdas()
        op_rows += sum(L.op.shape[0] for L, w in zip(self.regularization_matrices, lambdas) if L and w > 0)
        
        dtype = self.A[0].op.dtype
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
                    w_op = self._densify_op(w_item) if w_item.input_shape != (1,) else w_item.op
                    b_col_block = (
                        w_op * b_col_block if w_item.input_shape == (1,) else w_op @ b_col_block
                    )
                d_block[row : row + num_a_rows, :] = b_col_block
            row += num_a_rows
            
        return d_block, scenario_shape, num_scenarios

    def _apply_op_to_block(self, op: Union[np.ndarray, LinearOperator], x_block: np.ndarray) -> np.ndarray:
        """Helper to apply an operator to a block of column vectors."""
        if isinstance(op, LinearOperator):
            return op.matmat(x_block) if x_block.shape[1] > 1 else op.matvec(x_block[:, 0])[:, np.newaxis]
        return op @ x_block

    def _apply_op_T_to_block(self, op: Union[np.ndarray, LinearOperator], y_block: np.ndarray) -> np.ndarray:
        """Helper to apply an adjoint operator to a block of column vectors."""
        if isinstance(op, LinearOperator):
            return op.rmatmat(y_block) if y_block.shape[1] > 1 else op.rmatvec(y_block[:, 0])[:, np.newaxis]
        return op.T.conj() @ y_block

    def get_system_operator(self, num_scenarios: int = 1, use_scaled_lambdas: bool = True, include_regularization: bool = True) -> Tuple[LinearOperator, callable, callable]:
        """Gets the full system operator G as a matrix-free LinearOperator."""
        lambdas = self.get_scaled_lambdas() if use_scaled_lambdas else self.regularization_weights
        
        num_features = self.solution_size
        op_rows_data = sum(a.op.shape[0] for a in self.A)
        op_rows_reg = 0
        if include_regularization:
            for i, L_item in enumerate(self.regularization_matrices):
                if i < len(lambdas) and L_item and lambdas[i] > 0:
                    op_rows_reg += L_item.op.shape[0]
        op_rows = op_rows_data + op_rows_reg
        dtype = self.A[0].op.dtype

        def _apply_op_to_block(op, x_block):
            if isinstance(op, LinearOperator):
                return op.matmat(x_block) if x_block.shape[1] > 1 else op.matvec(x_block[:, 0])[:, np.newaxis]
            return op @ x_block

        def _apply_op_T_to_block(op, y_block):
            if isinstance(op, LinearOperator):
                return op.rmatmat(y_block) if y_block.shape[1] > 1 else op.rmatvec(y_block[:, 0])[:, np.newaxis]
            return op.T.conj() @ y_block

        def matvec_block(x_block):
            output_blocks = []
            for i, a_item in enumerate(self.A):
                res_block = _apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    res_block = (w_item.op * res_block if w_item.input_shape == (1,) else _apply_op_to_block(w_item.op, res_block))
                output_blocks.append(res_block)
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(lambdas) and L_item and lambdas[i] > 0:
                        output_blocks.append(lambdas[i] * _apply_op_to_block(L_item.op, x_block))
            return np.vstack(output_blocks) if output_blocks else np.zeros((0, x_block.shape[1]), dtype=dtype)

        def rmatvec_block(y_block):
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i, a_item in enumerate(self.A):
                num_a_rows = a_item.op.shape[0]
                y_part = y_block[row : row + num_a_rows, :]
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    y_part = (w_item.op.conj() * y_part if w_item.input_shape == (1,) else _apply_op_T_to_block(w_item.op, y_part))
                x_block += _apply_op_T_to_block(a_item.op, y_part)
                row += num_a_rows
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(lambdas) and L_item and lambdas[i] > 0:
                        num_L_rows = L_item.op.shape[0]
                        y_part = y_block[row : row + num_L_rows, :]
                        x_block += lambdas[i] * _apply_op_T_to_block(L_item.op, y_part)
                        row += num_L_rows
            return x_block

        shape = (op_rows * num_scenarios, num_features * num_scenarios)
        def matvec_final(x_flat): return matvec_block(x_flat.reshape(num_features, num_scenarios)).flatten()
        def rmatvec_final(y_flat): return rmatvec_block(y_flat.reshape(op_rows, num_scenarios)).flatten()
        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        return op, rmatvec_block, matvec_block

    def get_dense_system_matrix(self) -> np.ndarray:
        """Gets the full system operator G as a dense numpy array."""
        if "G_dense" in self._op_cache:
            return self._op_cache["G_dense"]
        
        lambdas = self.get_scaled_lambdas()
        all_A_weighted, all_L_weighted = [], []
        for i, a_item in enumerate(self.A):
            op = self._densify_op(a_item)
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                w_op = self._densify_op(w_item)
                op = w_op * op if w_item.input_shape == (1,) else w_op @ op
            all_A_weighted.append(op)
        for i, L_item in enumerate(self.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_L_weighted.append(lambdas[i] * self._densify_op(L_item))
        
        G_dense = np.vstack(all_A_weighted + all_L_weighted) if (all_A_weighted or all_L_weighted) else np.zeros((0, self.solution_size))
        self._op_cache["G_dense"] = G_dense
        return G_dense

    def get_scaled_lambdas(self) -> List[float]:
        """Returns the regularization weights scaled by the operator norms."""
        if "scaled_lambdas" not in self._op_cache:
            self._calculate_and_cache_scaled_lambdas()
        return self._op_cache["scaled_lambdas"]

    def picard_plot(self, title=None, ax=None, **plot_kwargs):
        """Generates a Picard plot of the singular values of the system matrix G."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for the picard_plot method.")
            return
        G_dense = self.get_dense_system_matrix()
        s = np.linalg.svd(G_dense, compute_uv=False)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(np.arange(1, len(s) + 1), s, "o-", markersize=3, **plot_kwargs)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value Magnitude")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        if title:
            ax.set_title(title)
        if 'fig' in locals():
            plt.tight_layout()
            plt.show()

    # --- Internal Helper Methods ---

    def _calculate_and_cache_scaled_lambdas(self) -> None:
        """Calculates regularization weights scaled by operator norms."""
        # --- Build a temporary "data-only" operator directly ---
        # This avoids the recursive call to get_system_operator.
        data_rows = sum(a.op.shape[0] for a in self.A)
        dtype = self.A[0].op.dtype

        def data_matvec(x_flat):
            x_block = x_flat.reshape(self.solution_size, 1)
            output_blocks = []
            for i, a_item in enumerate(self.A):
                res_block = self._apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    res_block = (w_item.op * res_block if w_item.input_shape == (1,) else self._apply_op_to_block(w_item.op, res_block))
                output_blocks.append(res_block)
            return np.vstack(output_blocks).flatten() if output_blocks else np.array([], dtype=dtype)
        
        data_op = LinearOperator(shape=(data_rows, self.solution_size), matvec=data_matvec, dtype=dtype)
        # --- End of temporary operator construction ---

        diag_A_T_A = np.zeros(self.solution_size, dtype=data_op.dtype)
        for i in range(self.solution_size):
            e = np.zeros(self.solution_size); e[i] = 1.0
            col = data_op.matvec(e)
            diag_A_T_A[i] = np.dot(col.conj(), col).real
        data_scale = np.median(diag_A_T_A[diag_A_T_A > 0]) if np.any(diag_A_T_A > 0) else 1.0

        scaled_lambdas = []
        for i, L_item in enumerate(self.regularization_matrices):
            raw_weight = self.regularization_weights[i]
            if raw_weight == 0 or L_item is None:
                scaled_lambdas.append(0.0)
                continue
            
            L_op = L_item.op
            diag_L_T_L = np.zeros(self.solution_size, dtype=L_op.dtype)
            for j in range(self.solution_size):
                e_j = np.zeros(self.solution_size); e_j[j] = 1.0
                col_j = L_op.matvec(e_j) if isinstance(L_op, LinearOperator) else self._densify_op(L_item)[:, j]
                diag_L_T_L[j] = np.dot(col_j.conj(), col_j).real
                
            reg_scale = np.median(diag_L_T_L[diag_L_T_L > 0]) if np.any(diag_L_T_L > 0) else 1.0
            scale_factor = np.sqrt(data_scale / reg_scale) if reg_scale > 1e-14 else 0.0
            scaled_lambdas.append(np.sqrt(raw_weight) * scale_factor)
            
        self._op_cache["scaled_lambdas"] = scaled_lambdas

    @staticmethod
    def _prepare_input_list(item: Optional[Any], name: str, count: Optional[int] = None, is_optional: bool = False, default_val: Any = None) -> list:
        if item is None:
            if is_optional: return []
            if count is None: raise ValueError(f"Input '{name}' cannot be None if 'count' is not specified.")
            return [default_val] * count
        lst = item if isinstance(item, list) else [item]
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, but expected {count}.")
        return lst

    def _normalize_data_shapes(self, data_shapes: Any, expected_count: int) -> List[Tuple[int, ...]]:
        if not isinstance(data_shapes, list): data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1: data_shapes *= expected_count
        if len(data_shapes) != expected_count: raise ValueError("Number of data_shapes does not match number of A operators.")
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    def _flatten(self, op: Any, output_shape: tuple = None, input_shape: tuple = None) -> _ProcessedItem:
        if isinstance(op, TensorChain):
            return _ProcessedItem(
                op=op.as_linear_operator() if self.matrix_free else op.to_dense(),
                output_shape=op.output_shape, input_shape=op.input_shape,
            )
        if isinstance(op, LinearOperator):
            return _ProcessedItem(op=op, output_shape=(op.shape[0],), input_shape=(op.shape[1],))
        if not isinstance(op, np.ndarray):
            raise TypeError("Input must be a numpy array, TensorChain, or LinearOperator")

        array = np.ascontiguousarray(op)
        if input_shape is None and output_shape is None: raise ValueError("At least one of output_shape or input_shape must be provided.")
        
        if input_shape is None: flat_in = array.size // math.prod(output_shape); input_shape = (flat_in,)
        elif output_shape is None: flat_out = array.size // math.prod(input_shape); output_shape = (flat_out,)
            
        flat_in, flat_out = math.prod(input_shape), math.prod(output_shape)
        if array.size != flat_in * flat_out: raise ValueError("Array size is incompatible with specified shapes.")
        return _ProcessedItem(array.reshape(flat_out, flat_in), output_shape, input_shape)

    def _densify_op(self, item: Optional[_ProcessedItem]) -> Optional[np.ndarray]:
        if item is None: return None
        op = item.op
        if isinstance(op, LinearOperator): return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    def _process_b_vector(self, b_val, data_shape) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        if b_val is None: return None, None
        
        num_data_dims = len(data_shape)
        b = np.ascontiguousarray(b_val)
        
        is_exact = b.shape == data_shape
        is_multi = b.ndim > num_data_dims and b.shape[:num_data_dims] == data_shape
        is_flat = b.ndim == 1 and b.size == math.prod(data_shape)
        if not (is_exact or is_multi or is_flat): raise ValueError("Shape of b is incompatible with its data_shape.")
            
        scenario_shape = b.shape[num_data_dims:] if is_multi else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = b.reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape