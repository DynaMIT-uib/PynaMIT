"""Least-squares solver (refactored).

This file is a lightly cleaned version of the original solver module. The
changes are mostly style / clarity improvements and a few bug fixes
(carried through from earlier corrections in the codebase).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Union
from scipy.sparse.linalg import LinearOperator, cg, lsmr

ITERATION_SAFETY_FACTOR = 10


@dataclass
class TensorChain:
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
        return np.result_type(*[t.dtype for t in self.component_tensors])

    def with_scaling(self, factor: float) -> "TensorChain":
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
        dense_matrix = np.einsum(self.einsum_string_dense, *self.component_tensors, optimize=True)
        return (dense_matrix * self.scaling_factor).reshape(
            math.prod(self.output_shape), math.prod(self.input_shape)
        )

    def as_linear_operator(self) -> LinearOperator:
        flat_out = math.prod(self.output_shape)
        flat_in = math.prod(self.input_shape)

        # prepare cached einsum path for matvec and rmatvec
        if self._einsum_path_matvec is None:
            dummy = np.empty(self.input_shape, dtype=self.dtype)
            self._einsum_path_matvec = np.einsum_path(
                self.einsum_string_matvec, *self.component_tensors, dummy, optimize="greedy"
            )[0]
        if self._einsum_path_rmatvec is None:
            dummy_grad = np.empty(self.output_shape, dtype=self.dtype)
            self._einsum_path_rmatvec = np.einsum_path(
                self.einsum_string_rmatvec, dummy_grad, *self.component_tensors, optimize="greedy"
            )[0]

        def _matvec(x_flat):
            x_tensor = x_flat.reshape(self.input_shape)
            all_tensors = self.component_tensors + [x_tensor]
            res = np.einsum(
                self.einsum_string_matvec, *all_tensors, optimize=self._einsum_path_matvec
            )
            return (res * self.scaling_factor).flatten()

        def _rmatvec(y_flat):
            grad_tensor = y_flat.reshape(self.output_shape)
            conj_tensors = [t.conj() for t in self.component_tensors]
            all_adjoint = [grad_tensor] + conj_tensors
            grad_x = np.einsum(
                self.einsum_string_rmatvec, *all_adjoint, optimize=self._einsum_path_rmatvec
            )
            return (grad_x.conj() * self.scaling_factor).flatten()

        return LinearOperator(
            shape=(flat_out, flat_in), matvec=_matvec, rmatvec=_rmatvec, dtype=self.dtype
        )


@dataclass
class _ProcessedItem:
    op: "Union[np.ndarray, LinearOperator]"
    output_shape: tuple
    input_shape: tuple


class LeastSquaresSolver:
    def __init__(
        self,
        A: Union[Any, List[Any]],
        solution_shape: Union[int, Tuple[int, ...]],
        data_shapes: Union[Any, List[Any]],
        sqrt_weights: Optional[Union[Any, List[Any]]] = None,
        regularization_weights: Optional[Union[float, List[float]]] = None,
        regularization_matrices: Optional[Union[Any, List[Any]]] = None,
        solver: str = "lsmr",
        tolerance: float = 1e-13,
        preconditioner: Optional[str] = None,
        picard_plot: bool = False,
    ) -> None:
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers:
            raise ValueError(f"Solver must be one of {solvers}")
        preconditioners = [None, "jacobi", "pinv"]
        if preconditioner not in preconditioners:
            raise ValueError(f"Preconditioner must be one of {preconditioners}")

        self._op_cache = {}
        self.solver = solver
        self.tolerance = tolerance
        self.preconditioner = preconditioner
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

        # determine matrix-free nature
        self.is_matrix_free = self.solver in ["cg", "lsmr"]
        if any(
            L is not None and isinstance(L.op, LinearOperator)
            for L in self.regularization_matrices
        ):
            self.is_matrix_free = True

        if self.is_matrix_free and self.preconditioner == "pinv":
            # densification will be required internally later
            pass

        if picard_plot:
            self.picard_plot()

    # ----- utility helpers -----

    def update_matrices(self, A, sqrt_weights=None, data_shapes=None) -> None:
        A_list = self._prepare_input_list(A, "A")
        self.num_data_terms = len(A_list)

        if data_shapes is not None:
            self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        elif not hasattr(self, "data_shapes") or len(self.data_shapes) != self.num_data_terms:
            raise ValueError(
                "data_shapes must be provided when setting A for the first time or changing number of A operators."
            )

        self.is_matrix_free = self.solver in ["cg", "lsmr"]
        if any(isinstance(op, (LinearOperator, TensorChain)) for op in A_list):
            self.is_matrix_free = True

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

        self.clear_cache(clear_preconditioner=False)

    def update_preconditioner(self) -> None:
        self.clear_cache(clear_preconditioner=True)

    def clear_cache(self, clear_preconditioner: bool = True) -> None:
        keys = [
            k
            for k in list(self._op_cache.keys())
            if not k.startswith(("lsmr_components", "cg_components"))
        ]
        for k in keys:
            self._op_cache.pop(k, None)
        if clear_preconditioner:
            for k in ["jacobi_diag", "pinv_components"]:
                self._op_cache.pop(k, None)

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
            data_shapes *= expected_count
        if len(data_shapes) != expected_count:
            raise ValueError("Number of data_shapes does not match number of A operators.")
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    def _flatten(
        self, op: Any, output_shape: tuple = None, input_shape: tuple = None
    ) -> _ProcessedItem:
        if isinstance(op, TensorChain):
            if self.is_matrix_free:
                return _ProcessedItem(
                    op=op.as_linear_operator(),
                    output_shape=op.output_shape,
                    input_shape=op.input_shape,
                )
            return _ProcessedItem(
                op=op.to_dense(), output_shape=op.output_shape, input_shape=op.input_shape
            )
        if isinstance(op, LinearOperator):
            return _ProcessedItem(op=op, output_shape=(op.shape[0],), input_shape=(op.shape[1],))
        if not isinstance(op, np.ndarray):
            raise TypeError("Input must be a numpy array, TensorChain, or LinearOperator")
        if output_shape is None and input_shape is None:
            raise ValueError(
                "At least one of output_shape or input_shape must be provided for an operator."
            )
        array = np.ascontiguousarray(op)
        if input_shape is None:
            flat_out = math.prod(output_shape)
            if array.size % flat_out != 0:
                raise ValueError("Array size not divisible by product of output_shape")
            flat_in = array.size // flat_out
            input_shape = (flat_in,)
        elif output_shape is None:
            flat_in = math.prod(input_shape)
            if array.size % flat_in != 0:
                raise ValueError("Array size not divisible by product of input_shape")
            flat_out = array.size // flat_in
            output_shape = (flat_out,)
        flat_in = math.prod(input_shape)
        flat_out = math.prod(output_shape)
        return _ProcessedItem(array.reshape(flat_out, flat_in), output_shape, input_shape)

    def _densify_op(self, item: Optional[_ProcessedItem]) -> Optional[np.ndarray]:
        if item is None:
            return None
        op = item.op
        if isinstance(op, LinearOperator):
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    def _process_b_vector(self, b_val, data_shape):
        if b_val is None:
            return None, None
        num_data_dims = len(data_shape)
        b = np.ascontiguousarray(b_val)
        is_exact = b.shape == data_shape
        is_multi = b.ndim > num_data_dims and b.shape[:num_data_dims] == data_shape
        is_flat = b.ndim == 1 and b.size == math.prod(data_shape)
        if not (is_exact or is_multi or is_flat):
            raise ValueError("Shape of b is incompatible with its data_shape.")
        scenario_shape = b.shape[num_data_dims:] if is_multi else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = b.reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape

    # ----- operator assembly and applications -----

    def _calculate_and_cache_scaled_lambdas(self) -> None:
        if "scaled_lambdas" in self._op_cache:
            return
        data_op, _, _ = self._get_multi_scenario_operator(
            num_scenarios=1, use_scaled_lambdas=False, include_regularization=False
        )
        diag_A_T_A = np.zeros(self.solution_size, dtype=data_op.dtype)
        for i in range(self.solution_size):
            e = np.zeros(self.solution_size)
            e[i] = 1.0
            col = data_op.matvec(e)
            diag_A_T_A[i] = np.dot(col.conj(), col).real
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

    def _get_multi_scenario_operator(
        self, num_scenarios, use_scaled_lambdas, include_regularization
    ):
        lambdas = (
            self._op_cache.get("scaled_lambdas", self.regularization_weights)
            if use_scaled_lambdas
            else self.regularization_weights
        )
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

        def matvec_final(x_flat):
            return matvec_block(x_flat.reshape(num_features, num_scenarios)).flatten()

        def rmatvec_final(y_flat):
            return rmatvec_block(y_flat.reshape(op_rows, num_scenarios)).flatten()

        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        return op, rmatvec_block, matvec_block

    def _get_full_stacked_operator(self):
        if "G_dense" in self._op_cache:
            return self._op_cache["G_dense"]
        lambdas = self._op_cache.get("scaled_lambdas", self.regularization_weights)
        all_A_weighted, all_L_weighted = [], []
        for i, a_item in enumerate(self.A):
            op = self._densify_op(a_item)
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                op = (
                    self._densify_op(w_item) * op
                    if w_item.input_shape == (1,)
                    else self._densify_op(w_item) @ op
                )
            all_A_weighted.append(op)
        for i, L_item in enumerate(self.regularization_matrices):
            if i < len(lambdas) and L_item and lambdas[i] > 1e-12:
                all_L_weighted.append(lambdas[i] * self._densify_op(L_item))
        G_dense = (
            np.vstack(all_A_weighted + all_L_weighted)
            if (all_A_weighted or all_L_weighted)
            else np.zeros((0, self.solution_size))
        )
        self._op_cache["G_dense"] = G_dense
        return G_dense

    def _get_svd_components(self):
        if "svd_components" not in self._op_cache:
            G_dense = self._get_full_stacked_operator()
            u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
            s_inv = np.zeros_like(s)
            stable_s = s > (self.tolerance * (s[0] if s.size > 0 else 0))
            s_inv[stable_s] = 1.0 / s[stable_s]
            self._op_cache["svd_components"] = (u, s_inv, vt)
        return self._op_cache["svd_components"]

    def _get_normal_components(self):
        if "normal_components" not in self._op_cache:
            G_dense = self._get_full_stacked_operator()
            G_T_G = G_dense.T.conj() @ G_dense
            G_T = G_dense.T.conj()
            self._op_cache["normal_components"] = (G_T_G, G_T)
        return self._op_cache["normal_components"]

    def _setup_preconditioner_components(self):
        if "jacobi_diag" in self._op_cache:
            return
        base_op, _, _ = self._get_multi_scenario_operator(
            num_scenarios=1, use_scaled_lambdas=True, include_regularization=True
        )
        diag = np.zeros(self.solution_size, dtype=base_op.dtype)
        for i in range(self.solution_size):
            e = np.zeros(self.solution_size)
            e[i] = 1.0
            col = base_op.matvec(e)
            diag[i] = np.dot(col.conj(), col).real
        self._op_cache["jacobi_diag"] = diag

    def _setup_pinv_preconditioner(self):
        if "pinv_components" in self._op_cache:
            return
        G_dense = self._get_full_stacked_operator()
        u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
        s_pinv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        stable = s > cutoff
        s_pinv[stable] = 1.0 / s[stable]
        s_inv_sq = s_pinv**2
        self._op_cache["pinv_components"] = (u, s, vt, s_pinv, s_inv_sq)

    def _get_lsmr_components(self, num_scenarios):
        cache_key = f"lsmr_components_{num_scenarios}"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]
        base_op, _, matvec_block = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )
        op_to_solve = base_op

        def solution_transform(sol_block):
            return sol_block

        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache["jacobi_diag"]
            sqrt_inv = np.sqrt(1.0 / diag)
            sqrt_inv[np.isinf(sqrt_inv)] = 1.0

            def precond_matvec(y_flat):
                return base_op.matvec(
                    (
                        y_flat.reshape(self.solution_size, num_scenarios) * sqrt_inv[:, None]
                    ).flatten()
                )

            def precond_rmatvec(d_flat):
                return (
                    base_op.rmatvec(d_flat.reshape(-1, num_scenarios).flatten()).reshape(
                        self.solution_size, num_scenarios
                    )
                    * sqrt_inv[:, None]
                ).flatten()

            op_to_solve = LinearOperator(
                base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype
            )

            def solution_transform(sol_y_block):
                return sol_y_block * sqrt_inv[:, None]
        elif self.preconditioner == "pinv":
            self._setup_pinv_preconditioner()
            _, _, vt, s_pinv, _ = self._op_cache["pinv_components"]

            def p_matvec(y_block):
                return vt.T.conj() @ (s_pinv[:, None] * (vt @ y_block))

            def precond_matvec(y_flat):
                return matvec_block(
                    p_matvec(y_flat.reshape(self.solution_size, num_scenarios))
                ).flatten()

            def precond_rmatvec(d_flat):
                return p_matvec(
                    base_op.rmatvec(d_flat.reshape(-1, num_scenarios).flatten()).reshape(
                        self.solution_size, num_scenarios
                    )
                ).flatten()

            op_to_solve = LinearOperator(
                base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype
            )

            def solution_transform(sol_y_block):
                return p_matvec(sol_y_block)

        self._op_cache[cache_key] = (op_to_solve, solution_transform)
        return op_to_solve, solution_transform

    def _get_cg_components(self, num_scenarios=1):
        cache_key = f"cg_components_{num_scenarios}"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]
        base_op, _, _ = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )

        def normal_matvec(x_flat):
            return base_op.rmatvec(base_op.matvec(x_flat))

        cg_op = LinearOperator(
            (self.solution_size * num_scenarios, self.solution_size * num_scenarios),
            matvec=normal_matvec,
            rmatvec=normal_matvec,
            dtype=base_op.dtype,
        )
        M = None
        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache.get("jacobi_diag")
            diag_inv = 1.0 / diag
            diag_inv[np.isinf(diag_inv)] = 1.0
            full_inv = np.tile(diag_inv, num_scenarios)

            def precon_matvec(x_flat):
                return x_flat * full_inv

            M = LinearOperator(
                cg_op.shape, matvec=precon_matvec, rmatvec=precon_matvec, dtype=diag.dtype
            )
        elif self.preconditioner == "pinv":
            self._setup_pinv_preconditioner()
            _, _, vt, _, s_inv_sq = self._op_cache["pinv_components"]

            def precon_block(x_block):
                return vt.T.conj() @ (s_inv_sq[:, None] * (vt @ x_block))

            def precon_matvec(x_flat):
                return precon_block(x_flat.reshape(self.solution_size, num_scenarios)).flatten()

            M = LinearOperator(
                cg_op.shape, matvec=precon_matvec, rmatvec=precon_matvec, dtype=vt.dtype
            )
        self._op_cache[cache_key] = (cg_op, M)
        return cg_op, M

    # ----- solvers: solve / adjoint -----

    def solve(self, b: Union[Any, List[Any]], **kwargs) -> np.ndarray:
        self._calculate_and_cache_scaled_lambdas()
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid = [(p[0], p[1]) for p in processed if p[0] is not None]
        if not valid:
            return np.zeros(self.solution_shape, dtype=self.A[0].op.dtype)
        scenario_shape = valid[0][1]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        base_op, rmatvec_block, _ = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )

        d_block = np.zeros((base_op.shape[0] // num_scenarios, num_scenarios), dtype=base_op.dtype)
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

        sol_block = None
        if self.solver == "svd":
            u, s_inv, vt = self._get_svd_components()
            sol_block = vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ d_block))
        elif self.solver == "normal":
            G_T_G, G_T = self._get_normal_components()
            sol_block = np.linalg.solve(G_T_G, G_T @ d_block)
        elif self.solver == "lsmr":
            op_to_solve, solution_transform = self._get_lsmr_components(num_scenarios)
            m, n = op_to_solve.shape[0] // num_scenarios, op_to_solve.shape[1] // num_scenarios
            max_iter = ITERATION_SAFETY_FACTOR * min(m, n) if min(m, n) > 0 else self.solution_size
            lsmr_kwargs = {
                "atol": self.tolerance,
                "btol": self.tolerance,
                "maxiter": max_iter,
                **kwargs,
            }
            sol_y_flat, istop, *_ = lsmr(op_to_solve, d_block.flatten(), **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                print(f"Warning: LSMR may not have fully converged (istop={istop}).")
            sol_block = solution_transform(sol_y_flat.reshape(self.solution_size, num_scenarios))
        elif self.solver == "cg":
            cg_op, M = self._get_cg_components(num_scenarios=num_scenarios)
            rhs_flat = rmatvec_block(d_block).flatten()
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(cg_op, rhs_flat, **cg_kwargs)
            if exit_code != 0:
                print(f"Warning: CG solver did not converge (exit_code={exit_code}).")
            sol_block = sol_flat.reshape(self.solution_size, num_scenarios)
        return sol_block.reshape(self.solution_shape + scenario_shape)

    def solve_adjoint(self, y: np.ndarray, **kwargs) -> list:
        self._calculate_and_cache_scaled_lambdas()
        y_ndim, sol_ndim = y.ndim, len(self.solution_shape)
        if y_ndim < sol_ndim or y.shape[:sol_ndim] != self.solution_shape:
            if y_ndim == 1 and y.size % self.solution_size == 0:
                num_scenarios = y.size // self.solution_size
                scenario_shape = (num_scenarios,) if num_scenarios > 1 else ()
            else:
                raise ValueError("Shape of y is incompatible with solution_shape.")
        else:
            scenario_shape = y.shape[sol_ndim:]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        y_block = np.ascontiguousarray(y).reshape(self.solution_size, num_scenarios)

        z_block = None
        if self.solver == "svd":
            _, s_inv, vt = self._get_svd_components()
            z_block = vt.T.conj() @ ((s_inv**2)[:, None] * (vt @ y_block))
        elif self.solver in ["normal", "cg", "lsmr"]:
            normal_op, M = self._get_cg_components(num_scenarios=num_scenarios)
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(normal_op, y_block.flatten(), **cg_kwargs)
            if exit_code != 0:
                print(f"Warning: Adjoint CG solver did not converge (exit_code={exit_code}).")
            z_block = sol_flat.reshape(self.solution_size, num_scenarios)

        _, _, matvec_block_fn = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )
        grad_d_block = matvec_block_fn(z_block)

        grad_b_list = []
        row = 0
        for i in range(self.num_data_terms):
            num_a_rows = self.A[i].op.shape[0]
            grad_d_i = grad_d_block[row : row + num_a_rows, :]
            grad_b_i = grad_d_i
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                w_op = w_item.op
                if w_item.input_shape == (1,):
                    grad_b_i = w_op.conj() * grad_d_i
                elif isinstance(w_op, LinearOperator):
                    grad_b_i = (
                        w_op.rmatmat(grad_d_i)
                        if grad_d_i.shape[1] > 1
                        else w_op.rmatvec(grad_d_i[:, 0])[:, None]
                    )
                else:
                    grad_b_i = w_op.T.conj() @ grad_d_i
            grad_b_list.append(grad_b_i.reshape(self.data_shapes[i] + scenario_shape))
            row += num_a_rows
        return grad_b_list

    def picard_plot(self, title=None, ax=None, **plot_kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for this method.")
            return
        G_dense = self._get_full_stacked_operator()
        s = np.linalg.svd(G_dense, compute_uv=False)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(np.arange(1, len(s) + 1), s, "o-", markersize=3, **plot_kwargs)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value Magnitude")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
