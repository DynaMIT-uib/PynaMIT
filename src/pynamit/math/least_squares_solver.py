"""Least-squares solver module.

This module contains the LeastSquaresSolver class for solving complex,
multi-term least-squares problems, and the TensorChain class for representing
structured, matrix-free operators.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
import math
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Union

ITERATION_SAFETY_FACTOR = 10


@dataclass
class TensorChain:
    """A self-contained object representing a tensor contraction operation.

    This class holds the tensors and the einsum strings needed to execute
    the operation either as a dense matrix or as a matrix-free LinearOperator.
    It can also pre-compute and cache optimal contraction paths for performance.
    """
    component_tensors: List[np.ndarray]
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
        """The data type of the operator, determined by its components."""
        return np.result_type(*[t.dtype for t in self.component_tensors])

    def with_scaling(self, factor: float) -> "TensorChain":
        """Returns a new TensorChain with an updated scaling factor."""
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
        """Contracts the component tensors to form a single dense matrix."""
        dense_matrix = np.einsum(
            self.einsum_string_dense, *self.component_tensors, optimize=True
        )
        return (dense_matrix * self.scaling_factor).reshape(
            math.prod(self.output_shape), math.prod(self.input_shape)
        )

    def as_linear_operator(self) -> LinearOperator:
        """Returns a scipy LinearOperator for matrix-free operations."""
        flat_output_dim = math.prod(self.output_shape)
        flat_input_dim = math.prod(self.input_shape)

        if self._einsum_path_matvec is None:
            dummy_input = np.empty(self.input_shape, dtype=self.dtype)
            all_tensors = self.component_tensors + [dummy_input]
            self._einsum_path_matvec = np.einsum_path(
                self.einsum_string_matvec, *all_tensors, optimize='greedy'
            )[0]

        if self._einsum_path_rmatvec is None:
            dummy_grad = np.empty(self.output_shape, dtype=self.dtype)
            all_adjoint_tensors = [dummy_grad] + self.component_tensors
            self._einsum_path_rmatvec = np.einsum_path(
                self.einsum_string_rmatvec, *all_adjoint_tensors, optimize='greedy'
            )[0]

        def _matvec(x_flat):
            x_tensor = x_flat.reshape(self.input_shape)
            all_tensors = self.component_tensors + [x_tensor]
            res = np.einsum(self.einsum_string_matvec, *all_tensors, optimize=self._einsum_path_matvec)
            return (res * self.scaling_factor).flatten()

        def _rmatvec(y_flat):
            grad_tensor = y_flat.reshape(self.output_shape)
            conjugated_tensors = [t.conj() for t in self.component_tensors]
            all_adjoint_tensors = [grad_tensor] + conjugated_tensors
            grad_x = np.einsum(self.einsum_string_rmatvec, *all_adjoint_tensors, optimize=self._einsum_path_rmatvec)
            return (grad_x.conj() * self.scaling_factor).flatten()

        return LinearOperator(
            shape=(flat_output_dim, flat_input_dim),
            matvec=_matvec,
            rmatvec=_rmatvec,
            dtype=self.dtype
        )


@dataclass
class _ProcessedItem:
    """Holds an operator and its multi-dimensional shape information."""
    op: "np.ndarray | LinearOperator"
    output_shape: tuple
    input_shape: tuple


class LeastSquaresSolver:
    """
    Solves least-squares problems of the form G*x = d, where G and d can
    be formed from multiple weighted and regularized components.
    """
    def __init__(
        self,
        A: Union[Any, List[Any]],
        solution_shape: Union[int, Tuple[int, ...]],
        data_shapes: Union[Any, List[Any]],
        sqrt_weights: Optional[Union[Any, List[Any]]] = None,
        regularization_weights: Optional[Union[float, List[float]]] = None,
        regularization_matrices: Optional[Union[Any, List[Any]]] = None,
        solver: str = "normal",
        tolerance: float = 1e-13,
        preconditioner: Optional[str] = None,
        picard_plot: bool = False,
    ):
        """
        Args:
            A: An operator or a list of operators. Each can be a numpy array,
               a SciPy LinearOperator, or a TensorChain.
            solution_shape: The desired shape of the solution vector `x`.
            data_shapes: A shape or a list of shapes, one for each corresponding
                         data vector `b` associated with an operator in `A`.
            ...
        """
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers:
            raise ValueError(f"Solver must be one of {solvers}")

        preconditioners = [None, "jacobi", "pinv"]
        if preconditioner not in preconditioners:
            raise ValueError(f"Preconditioner must be one of {preconditioners}")

        if preconditioner is not None and solver not in ["cg", "lsmr"]:
            print("Warning: Preconditioner is set but only applies to 'cg' or 'lsmr' solvers.")

        self._op_cache = {}
        self.solver = solver
        self.tolerance = tolerance
        self.preconditioner = preconditioner
        self.solution_shape = (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        self.solution_size = math.prod(self.solution_shape)

        self.is_matrix_free = solver in ["cg", "lsmr"]
        self.update_matrices(A, sqrt_weights=sqrt_weights, data_shapes=data_shapes)

        reg_L_list = self._prepare_input_list(regularization_matrices, "regularization_matrices", is_optional=True)
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [
            self._flatten(L, input_shape=self.solution_shape) if L is not None else None
            for L in reg_L_list
        ]
        self.regularization_weights = self._prepare_input_list(
            regularization_weights, "regularization_weights", count=self.num_reg_terms, default_val=0.0
        )

        is_reg_matrix_free = any(
            L is not None and isinstance(L.op, LinearOperator) for L in self.regularization_matrices
        )
        self.is_matrix_free = self.is_matrix_free or is_reg_matrix_free

        if self.is_matrix_free:
            if self.preconditioner == "pinv":
                print("Warning: 'pinv' preconditioner with matrix-free operators requires densification.")
            if solver in ["normal", "svd"]:
                print(f"Warning: Solver '{solver}' with matrix-free operators requires densification.")

        if picard_plot:
            self.picard_plot()

    def update_matrices(self, A, sqrt_weights=None, data_shapes=None):
        """Updates the data-fitting matrices (A) and weights for the problem."""
        A_list = self._prepare_input_list(A, "A")
        self.num_data_terms = len(A_list)

        if data_shapes is not None:
            self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        elif not hasattr(self, "data_shapes") or len(self.data_shapes) != self.num_data_terms:
            raise ValueError("data_shapes must be provided when setting A for the first time or changing the number of A operators.")

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
            is_diagonal = not isinstance(w_val, LinearOperator) and w_val.size == flat_data_dim
            if is_diagonal:
                w_op = np.ascontiguousarray(w_val).reshape(flat_data_dim, 1)
                self.sqrt_weights.append(
                    _ProcessedItem(op=w_op, output_shape=self.data_shapes[i], input_shape=(1,))
                )
            else:
                self.sqrt_weights.append(
                    self._flatten(w_val, output_shape=self.data_shapes[i], input_shape=self.data_shapes[i])
                )

        is_data_matrix_free = any(isinstance(a.op, LinearOperator) for a in self.A)
        self.is_matrix_free = (self.solver in ["cg", "lsmr"]) or is_data_matrix_free
        self.clear_cache(clear_preconditioner=False)

    def update_preconditioner(self):
        """Flags the preconditioner to be re-calculated on the next `solve` call."""
        print("Preconditioner invalidated. It will be re-computed on the next call to solve().")
        self.clear_cache(clear_preconditioner=True)

    def clear_cache(self, clear_preconditioner: bool = True):
        """Clears cached internal matrices and operators."""
        problem_specific_keys = ["scaled_lambdas", "G_dense", "normal_components", "svd_components"]
        solver_keys = [k for k in self._op_cache if k.startswith(("lsmr_components", "cg_components"))]
        for key in problem_specific_keys + solver_keys:
            if key in self._op_cache:
                del self._op_cache[key]
        if clear_preconditioner:
            preconditioner_keys = ["jacobi_diag", "pinv_components"]
            for key in preconditioner_keys:
                if key in self._op_cache:
                    del self._op_cache[key]

    @staticmethod
    def _prepare_input_list(
        item: Optional[Any], name: str, count: Optional[int] = None, is_optional: bool = False, default_val: Any = None
    ) -> list:
        """Standardizes user input into a list, accepting a single item or a list."""
        if item is None:
            if is_optional: return []
            if count is None: raise ValueError(f"Input '{name}' cannot be None if 'count' is not specified.")
            return [default_val] * count
        
        # This is the key change: flexibly accept a single item or a list
        lst = item if isinstance(item, list) else [item]
        
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, but expected {count}.")
        return lst

    def _normalize_data_shapes(self, data_shapes: Any, expected_count: int) -> List[Tuple[int, ...]]:
        """Standardizes the data_shapes input into a list of tuples."""
        if not isinstance(data_shapes, list):
            data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1:
            data_shapes *= expected_count
        if len(data_shapes) != expected_count:
            raise ValueError(f"Number of data_shapes ({len(data_shapes)}) does not match number of A operators ({expected_count}).")
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    def _flatten(self, op: Any, output_shape: tuple = None, input_shape: tuple = None) -> _ProcessedItem:
        """Converts an N-D operator into a 2D matrix or LinearOperator representation."""
        if isinstance(op, TensorChain):
            if self.is_matrix_free:
                return _ProcessedItem(op.as_linear_operator(), op.output_shape, op.input_shape)
            else:
                print(f"Densifying TensorChain via einsum ('{op.einsum_string_dense}')...")
                return _ProcessedItem(op.to_dense(), op.output_shape, op.input_shape)

        if isinstance(op, LinearOperator):
            return _ProcessedItem(op, (op.shape[0],), (op.shape[1],))
        if not isinstance(op, np.ndarray):
            raise TypeError(f"Input must be a numpy array, TensorChain, or LinearOperator, got {type(op)}")
        if output_shape is None and input_shape is None:
            raise ValueError("At least one of output_shape or input_shape must be provided for an operator.")
        
        array = np.ascontiguousarray(op)
        if input_shape is None:
            flat_output_dim = math.prod(output_shape)
            if array.size % flat_output_dim != 0:
                raise ValueError(f"Array size {array.size} not divisible by product of output_shape {output_shape}")
            flat_input_dim = array.size // flat_output_dim
            input_shape = (flat_input_dim,)
        elif output_shape is None:
            flat_input_dim = math.prod(input_shape)
            if array.size % flat_input_dim != 0:
                raise ValueError(f"Array size {array.size} not divisible by product of input_shape {input_shape}")
            flat_output_dim = array.size // flat_input_dim
            output_shape = (flat_output_dim,)
        
        flat_input_dim = math.prod(input_shape)
        flat_output_dim = math.prod(output_shape)
        return _ProcessedItem(array.reshape(flat_output_dim, flat_input_dim), output_shape, input_shape)

    def _densify_op(self, item: Optional[_ProcessedItem]) -> Optional[np.ndarray]:
        """Ensures a _ProcessedItem's operator is a dense numpy array."""
        if item is None:
            return None
        op = item.op
        if isinstance(op, LinearOperator):
            print(f"Warning: Densifying a generic LinearOperator of shape {op.shape} using slow fallback.")
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    # The rest of the file is identical to the previous version
    def _process_b_vector(self, b_val, data_shape):
        """Reshape a `b` vector into a 2D column-block format."""
        if b_val is None:
            return None, None
        num_data_dims = len(data_shape)
        is_exact = b_val.shape == data_shape
        is_multi_scenario = (
            b_val.ndim > num_data_dims and b_val.shape[:num_data_dims] == data_shape
        )
        is_flat_single_scenario = b_val.ndim == 1 and b_val.size == math.prod(data_shape)
        if not (is_exact or is_multi_scenario or is_flat_single_scenario):
            raise ValueError(
                f"Shape of b term {b_val.shape} is incompatible with its data_shape {data_shape}."
            )
        scenario_shape = b_val.shape[num_data_dims:] if is_multi_scenario else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = np.ascontiguousarray(b_val).reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape

    def _calculate_and_cache_scaled_lambdas(self):
        """Auto-scales regularization weights."""
        if "scaled_lambdas" in self._op_cache:
            return
        data_op, _, _ = self._get_multi_scenario_operator(
            num_scenarios=1, use_scaled_lambdas=False, include_regularization=False
        )
        diag_A_T_A = np.zeros(self.solution_size, dtype=data_op.dtype)
        for i in range(self.solution_size):
            e_i = np.zeros(self.solution_size)
            e_i[i] = 1.0
            col_i = data_op.matvec(e_i)
            diag_A_T_A[i] = np.dot(col_i.conj(), col_i).real
        data_scale = np.median(diag_A_T_A[diag_A_T_A > 0]) if np.any(diag_A_T_A > 0) else 1.0
        scaled_lambdas = []
        for i, L_item in enumerate(self.regularization_matrices):
            raw_weight = self.regularization_weights[i]
            if raw_weight == 0 or L_item is None:
                scaled_lambdas.append(0.0)
                continue
            diag_L_T_L = np.zeros(self.solution_size, dtype=L_item.op.dtype)
            L_op = L_item.op
            for j in range(self.solution_size):
                e_j = np.zeros(self.solution_size)
                e_j[j] = 1.0
                col_j = (
                    L_op.matvec(e_j)
                    if isinstance(L_op, LinearOperator)
                    else self._densify_op(L_item)[:, j]
                )
                diag_L_T_L[j] = np.dot(col_j.conj(), col_j).real
            reg_scale = np.median(diag_L_T_L[diag_L_T_L > 0]) if np.any(diag_L_T_L > 0) else 1.0
            scaled_lambda = (
                np.sqrt(raw_weight) * np.sqrt(data_scale / reg_scale) if reg_scale > 1e-14 else 0.0
            )
            scaled_lambdas.append(scaled_lambda)
        self._op_cache["scaled_lambdas"] = scaled_lambdas

    def _get_multi_scenario_operator(self, num_scenarios, use_scaled_lambdas, include_regularization):
        """Build the full system LinearOperator."""
        lambdas = (
            self._op_cache.get("scaled_lambdas", self.regularization_weights)
            if use_scaled_lambdas
            else self.regularization_weights
        )
        num_features = self.solution_size
        op_rows_data = sum(a.op.shape[0] for a in self.A)
        op_rows_reg = (
            sum(
                lambda_.op.shape[0]
                for i, lambda_ in enumerate(self.regularization_matrices)
                if i < len(lambdas) and lambda_ and lambdas[i] > 0
            )
            if include_regularization
            else 0
        )
        op_rows = op_rows_data + op_rows_reg
        dtype = self.A[0].op.dtype

        def _apply_op_to_block(op, x_block):
            if isinstance(op, LinearOperator):
                return (
                    op.matmat(x_block)
                    if x_block.shape[1] > 1
                    else op.matvec(x_block[:, 0])[:, np.newaxis]
                )
            return op @ x_block

        def _apply_op_T_to_block(op, y_block):
            if isinstance(op, LinearOperator):
                return (
                    op.rmatmat(y_block)
                    if y_block.shape[1] > 1
                    else op.rmatvec(y_block[:, 0])[:, np.newaxis]
                )
            return op.T.conj() @ y_block

        def matvec_block(x_block):
            output_blocks = []
            for i, a_item in enumerate(self.A):
                res_block = _apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    res_block = (
                        w_item.op * res_block
                        if w_item.input_shape == (1,)
                        else _apply_op_to_block(w_item.op, res_block)
                    )
                output_blocks.append(res_block)
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(lambdas) and L_item and lambdas[i] > 0:
                        res_block = _apply_op_to_block(L_item.op, x_block)
                        output_blocks.append(lambdas[i] * res_block)
            return (
                np.vstack(output_blocks)
                if output_blocks
                else np.zeros((0, x_block.shape[1]), dtype=dtype)
            )

        def rmatvec_block(y_block):
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i, a_item in enumerate(self.A):
                num_a_rows = a_item.op.shape[0]
                y_part = y_block[row : row + num_a_rows, :]
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    y_part = (
                        w_item.op.conj() * y_part
                        if w_item.input_shape == (1,)
                        else _apply_op_T_to_block(w_item.op, y_part)
                    )
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

    def _get_full_stacked_operator(self):
        """Build a single dense matrix G for the entire system."""
        if "G_dense" in self._op_cache:
            return self._op_cache["G_dense"]
        lambdas = self._op_cache.get("scaled_lambdas", self.regularization_weights)
        all_A_weighted, all_L_weighted = [], []
        for i, a_item in enumerate(self.A):
            op, w_item = self._densify_op(a_item), self.sqrt_weights[i]
            if w_item is not None:
                op = (self._densify_op(w_item) * op if w_item.input_shape == (1,) else self._densify_op(w_item) @ op)
            all_A_weighted.append(op)
        for i, L_item in enumerate(self.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_L_weighted.append(lambdas[i] * self._densify_op(L_item))
        G_dense = np.vstack(all_A_weighted + all_L_weighted) if (all_A_weighted or all_L_weighted) else np.zeros((0, self.solution_size))
        self._op_cache["G_dense"] = G_dense
        return G_dense

    def _get_svd_components(self):
        """Lazily computes and caches the SVD components."""
        if "svd_components" not in self._op_cache:
            G_dense = self._get_full_stacked_operator()
            u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
            s_inv = np.zeros_like(s)
            stable_s = s > (self.tolerance * (s[0] if s.size > 0 else 0))
            s_inv[stable_s] = 1.0 / s[stable_s]
            self._op_cache["svd_components"] = (u, s_inv, vt)
        return self._op_cache["svd_components"]

    def _get_normal_components(self):
        """Return the matrices for the normal equations."""
        if "normal_components" not in self._op_cache:
            G_dense = self._get_full_stacked_operator()
            G_T_G = G_dense.T.conj() @ G_dense
            G_T = G_dense.T.conj()
            self._op_cache["normal_components"] = (G_T_G, G_T)
        return self._op_cache["normal_components"]

    def _setup_preconditioner_components(self):
        """Return the Jacobi preconditioner diagonal."""
        if "jacobi_diag" in self._op_cache: return
        print(f"Calculating Jacobi preconditioner for {'matrix-free' if self.is_matrix_free else 'dense'} operator...")
        base_op, _, _ = self._get_multi_scenario_operator(num_scenarios=1, use_scaled_lambdas=True, include_regularization=True)
        diag_G_T_G = np.zeros(self.solution_size, dtype=base_op.dtype)
        for i in range(self.solution_size):
            e_i = np.zeros(self.solution_size); e_i[i] = 1.0
            col_i = base_op.matvec(e_i)
            diag_G_T_G[i] = np.dot(col_i.conj(), col_i).real
        self._op_cache["jacobi_diag"] = diag_G_T_G

    def _setup_pinv_preconditioner(self):
        """Computes and caches SVD components for the 'pinv' preconditioner."""
        if "pinv_components" in self._op_cache: return
        print("Calculating SVD for 'pinv' preconditioner...")
        G_dense = self._get_full_stacked_operator()
        u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
        s_pinv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        stable_s_mask = s > cutoff
        s_pinv[stable_s_mask] = 1.0 / s[stable_s_mask]
        s_inv_sq = s_pinv**2
        self._op_cache["pinv_components"] = (u, s, vt, s_pinv, s_inv_sq)
        print("...SVD calculation complete.")

    def _get_lsmr_components(self, num_scenarios):
        """Return components for the LSMR solver."""
        cache_key = f"lsmr_components_{num_scenarios}"
        if cache_key in self._op_cache: return self._op_cache[cache_key]
        base_op, _, matvec_block = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
        op_to_solve = base_op
        def solution_transform(sol_block): return sol_block
        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache["jacobi_diag"]; sqrt_inv_diag = np.sqrt(1.0 / diag); sqrt_inv_diag[np.isinf(sqrt_inv_diag)] = 1.0
            def precond_matvec(y_flat): return base_op.matvec((y_flat.reshape(self.solution_size, num_scenarios) * sqrt_inv_diag[:, np.newaxis]).flatten())
            def precond_rmatvec(d_flat): return (base_op.rmatvec(d_flat.reshape(-1, num_scenarios).flatten()).reshape(self.solution_size, num_scenarios) * sqrt_inv_diag[:, np.newaxis]).flatten()
            op_to_solve = LinearOperator(base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype)
            def solution_transform(sol_y_block): return sol_y_block * sqrt_inv_diag[:, np.newaxis]
        elif self.preconditioner == "pinv":
            self._setup_pinv_preconditioner()
            _, _, vt, s_pinv, _ = self._op_cache["pinv_components"]
            def p_matvec(y_block): return vt.T.conj() @ (s_pinv[:, np.newaxis] * (vt @ y_block))
            def precond_matvec(y_flat): return matvec_block(p_matvec(y_flat.reshape(self.solution_size, num_scenarios))).flatten()
            def precond_rmatvec(d_flat): return p_matvec(base_op.rmatvec(d_flat.reshape(-1, num_scenarios).flatten()).reshape(self.solution_size, num_scenarios)).flatten()
            op_to_solve = LinearOperator(base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype)
            def solution_transform(sol_y_block): return p_matvec(sol_y_block)
        self._op_cache[cache_key] = (op_to_solve, solution_transform)
        return op_to_solve, solution_transform

    def _get_cg_components(self, num_scenarios=1):
        """Return components for the CG solver."""
        cache_key = f"cg_components_{num_scenarios}"
        if cache_key in self._op_cache: return self._op_cache[cache_key]
        base_op, _, _ = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
        def normal_op_matvec(x_flat): return base_op.rmatvec(base_op.matvec(x_flat))
        cg_op = LinearOperator((self.solution_size * num_scenarios, self.solution_size * num_scenarios), matvec=normal_op_matvec, rmatvec=normal_op_matvec, dtype=base_op.dtype)
        M = None
        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache.get("jacobi_diag"); diag_inv = 1.0 / diag; diag_inv[np.isinf(diag_inv)] = 1.0; full_diag_inv = np.tile(diag_inv, num_scenarios)
            def precon_matvec(x_flat): return x_flat * full_diag_inv
            M = LinearOperator((cg_op.shape), matvec=precon_matvec, rmatvec=precon_matvec, dtype=diag.dtype)
        elif self.preconditioner == "pinv":
            self._setup_pinv_preconditioner()
            _, _, vt, _, s_inv_sq = self._op_cache["pinv_components"]
            def precon_matvec_block(x_block): return vt.T.conj() @ (s_inv_sq[:, np.newaxis] * (vt @ x_block))
            def precon_matvec(x_flat): return precon_matvec_block(x_flat.reshape(self.solution_size, num_scenarios)).flatten()
            M = LinearOperator(cg_op.shape, matvec=precon_matvec, rmatvec=precon_matvec, dtype=vt.dtype)
        self._op_cache[cache_key] = (cg_op, M)
        return cg_op, M

    def solve(self, b: Union[Any, List[Any]], **kwargs) -> np.ndarray:
        """Solves the least-squares problem for right-hand-side data."""
        self._calculate_and_cache_scaled_lambdas()
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed_b = [self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)]
        valid_b = [(p[0], p[1]) for p in processed_b if p[0] is not None]
        if not valid_b: return np.zeros(self.solution_shape, dtype=self.A[0].op.dtype)
        scenario_shape = valid_b[0][1]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        base_op, rmatvec_block, _ = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
        d_block = np.zeros((base_op.shape[0] // num_scenarios, num_scenarios), dtype=base_op.dtype)
        current_row = 0
        for i, b_val in enumerate(b_list):
            num_a_rows = self.A[i].op.shape[0]
            if b_val is not None:
                b_col_block, b_scenario_shape = processed_b[i]
                if b_scenario_shape != scenario_shape: raise ValueError("Inconsistent scenario shapes in b terms.")
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    w_op = self._densify_op(w_item) if w_item.input_shape != (1,) else w_item.op
                    b_col_block = (w_op * b_col_block if w_item.input_shape == (1,) else w_op @ b_col_block)
                d_block[current_row : current_row + num_a_rows, :] = b_col_block
            current_row += num_a_rows
        sol_block = None
        if self.solver == "svd":
            u, s_inv, vt = self._get_svd_components()
            sol_block = vt.T.conj() @ (s_inv[:, np.newaxis] * (u.T.conj() @ d_block))
        elif self.solver == "normal":
            G_T_G, G_T = self._get_normal_components()
            sol_block = np.linalg.solve(G_T_G, G_T @ d_block)
        elif self.solver == "lsmr":
            op_to_solve, solution_transform = self._get_lsmr_components(num_scenarios)
            m, n = op_to_solve.shape[0] // num_scenarios, op_to_solve.shape[1] // num_scenarios
            max_iter = ITERATION_SAFETY_FACTOR * min(m, n) if min(m, n) > 0 else self.solution_size
            lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance, "maxiter": max_iter, **kwargs}
            sol_y_flat, istop, *_ = lsmr(op_to_solve, d_block.flatten(), **lsmr_kwargs)
            if istop not in [0, 1, 2]: print(f"Warning: LSMR may not have fully converged (istop={istop}).")
            sol_block = solution_transform(sol_y_flat.reshape(self.solution_size, num_scenarios))
        elif self.solver == "cg":
            cg_op, M = self._get_cg_components(num_scenarios=num_scenarios)
            rhs_flat = rmatvec_block(d_block).flatten()
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(cg_op, rhs_flat, **cg_kwargs)
            if exit_code != 0: print(f"Warning: CG solver did not converge (exit_code={exit_code}).")
            sol_block = sol_flat.reshape(self.solution_size, num_scenarios)
        return sol_block.reshape(self.solution_shape + scenario_shape)

    def solve_adjoint(self, y: np.ndarray, **kwargs) -> list:
        """Solves the adjoint of the least-squares problem."""
        self._calculate_and_cache_scaled_lambdas()
        y_ndim, sol_ndim = y.ndim, len(self.solution_shape)
        if y_ndim < sol_ndim or y.shape[:sol_ndim] != self.solution_shape:
            if y_ndim == 1 and y.size % self.solution_size == 0:
                num_scenarios = y.size // self.solution_size
                scenario_shape = (num_scenarios,) if num_scenarios > 1 else ()
            else:
                raise ValueError(f"Shape of y {y.shape} is incompatible with solution_shape {self.solution_shape}.")
        else:
            scenario_shape = y.shape[sol_ndim:]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        y_block = np.ascontiguousarray(y).reshape(self.solution_size, num_scenarios)
        z_block = None
        if self.solver == "svd":
            _, s_inv, vt = self._get_svd_components()
            z_block = vt.T.conj() @ ((s_inv**2)[:, np.newaxis] * (vt @ y_block))
        elif self.solver in ["normal", "cg", "lsmr"]: # All use normal equations for adjoint
            normal_op, M = self._get_cg_components(num_scenarios=num_scenarios)
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(normal_op, y_block.flatten(), **cg_kwargs)
            if exit_code != 0: print(f"Warning: Adjoint CG solver did not converge (exit_code={exit_code}).")
            z_block = sol_flat.reshape(self.solution_size, num_scenarios)
        _, _, matvec_block_fn = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
        grad_d_block = matvec_block_fn(z_block)
        grad_b_list = []
        current_row = 0
        for i in range(self.num_data_terms):
            num_a_rows = self.A[i].op.shape[0]
            grad_d_i = grad_d_block[current_row : current_row + num_a_rows, :]
            grad_b_i = grad_d_i
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                w_op = w_item.op
                if w_item.input_shape == (1,): grad_b_i = w_op.conj() * grad_d_i
                elif isinstance(w_op, LinearOperator): grad_b_i = w_op.rmatmat(grad_d_i) if grad_d_i.shape[1] > 1 else w_op.rmatvec(grad_d_i[:, 0])[:, np.newaxis]
                else: grad_b_i = w_op.T.conj() @ grad_d_i
            grad_b_list.append(grad_b_i.reshape(self.data_shapes[i] + scenario_shape))
            current_row += num_a_rows
        return grad_b_list

    def picard_plot(self, title=None, ax=None, **plot_kwargs):
        """Performs a Picard plot of the system's singular values."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for this method.")
            return
        print("Constructing the full system matrix G...")
        G_dense = self._get_full_stacked_operator()
        print("Computing singular values using SVD...")
        s = np.linalg.svd(G_dense, compute_uv=False)
        print("...done.")
        if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
        index = np.arange(1, len(s) + 1)
        ax.semilogy(index, s, "o-", markersize=3, **plot_kwargs)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value Magnitude")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        if title: ax.set_title(title)
        if "label" in plot_kwargs: ax.legend()
        plt.tight_layout()
        plt.show()