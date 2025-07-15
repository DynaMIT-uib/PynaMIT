"""Least-squares solver module.

This module contains the LeastSquaresSolver class for solving complex,
multi-term least-squares problems.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
import math
from dataclasses import dataclass


@dataclass
class _ProcessedItem:
    """Holds a processed operator and its logical multi-dimensional shape information."""

    op: "np.ndarray | LinearOperator"
    output_shape: tuple
    input_shape: tuple


class LeastSquaresSolver:
    """
    Solves complex least-squares problems with multiple data and regularization terms.

    This class provides a flexible and explicit API for defining and solving problems
    of the form:

        minimize || A_1 x - b_1 ||_W_1^2 + ... + || A_n x - b_n ||_W_n^2 +
                 || L_1 x ||_lambda_1^2 + ... + || L_m x ||_lambda_m^2

    where x is the solution, A_i are data-fitting operators, b_i are the target
    data, W_i are weighting matrices (or vectors), L_j are regularization
    operators, and lambda_j are regularization strengths.

    Parameters
    ----------
    A : np.ndarray | LinearOperator | list
        A single N-D array/LinearOperator or a list of them for the data-fitting terms.
    solution_shape : int | tuple
        The multi-dimensional shape of the solution vector `x`.
    data_shapes : tuple | list[tuple]
        A list of tuples, where `data_shapes[i]` is the multi-dimensional shape
        of the output space for operator `A[i]`.
    sqrt_weights : np.ndarray | list, optional
        A single N-D weight array or a list of them, representing the square root
        of the desired weights.
    regularization_weights : float | list[float], optional
        A single scalar weight or a list of weights `lambda_j` for each regularization term.
    regularization_matrices : np.ndarray | LinearOperator | list, optional
        A single N-D regularization operator `L_j` or a list of them.
    solver : str, default="svd"
        The solver to use. One of {"svd", "normal", "lsmr", "cg"}.
    tolerance : float, default=1e-12
        The tolerance for iterative solvers ("lsmr", "cg") or for SVD truncation.
    preconditioner : str, optional
        The preconditioner for the "cg" solver. Only "jacobi" is supported.

    Notes
    -----
    **Multi-Dimensional Inputs and Flattening**
    A core feature is handling N-D arrays (tensors) for operators and data, which
    are internally flattened into 2D matrices for computation.

    **Vector vs. Matrix Weights**
    A `sqrt_weights` entry is treated as **diagonal** if its size matches the
    product of its corresponding `data_shapes` entry. Otherwise, it is treated
    as a **dense** matrix.

    **`solve` Method `b` Argument**
    The shape of `b` must match its `data_shape`, optionally with trailing
    scenario dimensions: `data_shape + scenario_shape`. A flattened 1D array is
    also accepted if its size matches the `data_shape` product. The solution `x`
    is returned with shape `solution_shape + scenario_shape`.
    """

    def __init__(
        self,
        A,
        solution_shape,
        data_shapes,
        sqrt_weights=None,
        regularization_weights=None,
        regularization_matrices=None,
        solver="svd",
        tolerance=1e-12,
        preconditioner=None,
    ):
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers:
            raise ValueError(f"Solver must be one of {solvers}")
        if preconditioner is not None and preconditioner != "jacobi":
            raise ValueError("Only 'jacobi' preconditioner supported.")
        if preconditioner is not None and solver != "cg":
            print("Warning: Preconditioner is set but only applies to 'cg' solver.")

        self.solver = solver
        self.tolerance = tolerance
        self.preconditioner = preconditioner
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)

        A_list = self._prepare_input_list(A, "A", allow_single_item=True)
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
            is_diagonal = not isinstance(w_val, LinearOperator) and w_val.size == flat_data_dim
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

        reg_L_list = self._prepare_input_list(
            regularization_matrices,
            "regularization_matrices",
            allow_single_item=True,
            is_optional=True,
        )
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [
            self._flatten(L, input_shape=self.solution_shape) if L is not None else None
            for L in reg_L_list
        ]
        for l_item in self.regularization_matrices:
            if l_item and l_item.op.shape[-1] != self.solution_size:
                raise ValueError(
                    f"Shape mismatch in regularization term. Expected {self.solution_size} columns, got {l_item.op.shape[-1]}"
                )

        self.regularization_weights = self._prepare_input_list(
            regularization_weights,
            "regularization_weights",
            count=self.num_reg_terms,
            default_val=0.0,
        )

        self.is_matrix_free = any(isinstance(a.op, LinearOperator) for a in self.A) or any(
            L is not None and isinstance(L.op, LinearOperator)
            for L in self.regularization_matrices
        )
        if self.is_matrix_free and solver in ["normal", "svd"]:
            print(
                f"Warning: Solver '{solver}' with matrix-free operators requires densification, which may be slow or memory-intensive."
            )

        self._op_cache = {}
        self._scaled_reg_weights_cache = None

    @staticmethod
    def _prepare_input_list(
        item, name, count=None, allow_single_item=False, is_optional=False, default_val=None
    ):
        """Standardizes user input into a list of a specific length."""
        if item is None:
            if is_optional:
                return []
            return [default_val] * count if count is not None else []
        lst = item if isinstance(item, list) else [item]
        if allow_single_item and count is None:
            count = len(lst)
        if len(lst) == 1 and count is not None and count > 1:
            lst *= count
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, expected {count}.")
        return lst

    def _normalize_data_shapes(self, data_shapes, expected_count):
        """Standardizes the data_shapes input into a list of tuples."""
        if not isinstance(data_shapes, list):
            data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1:
            data_shapes *= expected_count
        if len(data_shapes) != expected_count:
            raise ValueError(
                f"Number of data_shapes ({len(data_shapes)}) does not match number of A operators ({expected_count})."
            )
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    @staticmethod
    def _flatten(array, output_shape=None, input_shape=None):
        """Converts an N-D operator into a 2D matrix representation."""
        if isinstance(array, LinearOperator):
            return _ProcessedItem(array, (array.shape[0],), (array.shape[1],))
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input must be a numpy array or LinearOperator, got {type(array)}")
        if output_shape is None and input_shape is None:
            raise ValueError(
                "At least one of output_shape or input_shape must be provided for an operator."
            )
        array = np.ascontiguousarray(array)
        if input_shape is None:
            flat_output_dim = math.prod(output_shape)
            if array.size % flat_output_dim != 0:
                raise ValueError(
                    f"Array size {array.size} not divisible by product of output_shape {output_shape}"
                )
            flat_input_dim = array.size // flat_output_dim
            input_shape = (flat_input_dim,)
        elif output_shape is None:
            flat_input_dim = math.prod(input_shape)
            if array.size % flat_input_dim != 0:
                raise ValueError(
                    f"Array size {array.size} not divisible by product of input_shape {input_shape}"
                )
            flat_output_dim = array.size // flat_input_dim
            output_shape = (flat_output_dim,)
        flat_input_dim = math.prod(input_shape)
        flat_output_dim = math.prod(output_shape)
        return _ProcessedItem(
            array.reshape(flat_output_dim, flat_input_dim), output_shape, input_shape
        )

    def _densify_op(self, item):
        """Converts a _ProcessedItem into a dense numpy array if it isn't one already."""
        if item is None:
            return None
        op = item.op
        if isinstance(op, LinearOperator):
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    def _get_scaled_regularization_weights(self):
        """Calculates scaled regularization weights to balance data and regularization terms."""
        if self._scaled_reg_weights_cache is not None:
            return self._scaled_reg_weights_cache
        if self.is_matrix_free:
            print(
                "Warning: Cannot auto-scale regularization for matrix-free operators. Using raw regularization_weights."
            )
            self._scaled_reg_weights_cache = self.regularization_weights
            return self.regularization_weights
        diag_data_terms = []
        for i, a_item in enumerate(self.A):
            A_dense, w_item = self._densify_op(a_item), self.sqrt_weights[i]
            if w_item is None:
                term_diag = np.sum(A_dense**2, axis=0)
            elif w_item.input_shape == (1,):
                w_diag_sq = w_item.op.flatten() ** 2
                term_diag = np.sum(w_diag_sq[:, np.newaxis] * (A_dense**2), axis=0)
            else:
                G = self._densify_op(w_item) @ A_dense
                term_diag = np.sum(G**2, axis=0)
            diag_data_terms.append(term_diag)
        diag_data = np.sum(diag_data_terms, axis=0)
        data_scale = np.median(diag_data[diag_data > 0]) if np.any(diag_data > 0) else 1.0
        full_normal_diag = diag_data.copy()
        scaled_weights = []
        for i, L_item in enumerate(self.regularization_matrices):
            raw_weight = self.regularization_weights[i]
            weight = 0.0
            if raw_weight > 0 and L_item is not None:
                L_dense = self._densify_op(L_item)
                diag_reg = np.sum(L_dense**2, axis=0)
                reg_scale = np.median(diag_reg[diag_reg > 0]) if np.any(diag_reg > 0) else 1.0
                if reg_scale > 1e-12:
                    # This is lambda_eff^2, following the original implementation's scaling logic
                    weight = raw_weight * data_scale / reg_scale
                    full_normal_diag += weight * diag_reg
            scaled_weights.append(weight)
        self._op_cache["jacobi_diag"] = full_normal_diag
        self._scaled_reg_weights_cache = scaled_weights
        return scaled_weights

    def _get_multi_scenario_operator(self, num_scenarios):
        """
        Builds the LinearOperator for the full system using the direct, C-ordered,
        column-vector convention. This is a faithful port of the original code's
        operator construction to ensure numerical stability for iterative solvers.
        """
        cache_key = f"op_{num_scenarios}"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]
        scaled_weights = self._get_scaled_regularization_weights()
        sqrt_scaled_lambdas = [np.sqrt(w) for w in scaled_weights]
        num_features = self.solution_size
        op_rows = sum(a.op.shape[0] for a in self.A) + sum(
            l.op.shape[0]
            for i, l in enumerate(self.regularization_matrices)
            if l and sqrt_scaled_lambdas[i] > 0
        )
        dtype = self.A[0].op.dtype

        def _apply_op_to_block(op, x_block):
            if isinstance(op, LinearOperator):
                # If the operator is itself a LinearOperator, we must loop over scenarios
                # as it does not support vectorized block operations.
                res_block = np.zeros((op.shape[0], num_scenarios), dtype=x_block.dtype)
                for k in range(num_scenarios):
                    res_block[:, k] = op.matvec(x_block[:, k])
                return res_block
            # For dense numpy arrays, the vectorized matrix-matrix product is used.
            return op @ x_block

        def _apply_op_T_to_block(op, y_block):
            if isinstance(op, LinearOperator):
                res_block = np.zeros((op.shape[1], num_scenarios), dtype=y_block.dtype)
                for k in range(num_scenarios):
                    res_block[:, k] = op.rmatvec(y_block[:, k])
                return res_block
            return op.T.conj() @ y_block

        def matvec_block(x_block):
            # The full `matvec` stacks the results of each term vertically.
            output_blocks = []
            for i, a_item in enumerate(self.A):
                res_block = _apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    res_block = (
                        w_item.op * res_block
                        if w_item.input_shape == (1,)
                        else w_item.op @ res_block
                    )
                output_blocks.append(res_block)
            for i, L_item in enumerate(self.regularization_matrices):
                if L_item and sqrt_scaled_lambdas[i] > 0:
                    res_block = _apply_op_to_block(L_item.op, x_block)
                    output_blocks.append(sqrt_scaled_lambdas[i] * res_block)
            return np.vstack(output_blocks)

        def rmatvec_block(y_block):
            # The full `rmatvec` slices the input vector and sums the contributions.
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
                        else w_item.op.T.conj() @ y_part
                    )
                x_block += _apply_op_T_to_block(a_item.op, y_part)
                row += num_a_rows
            for i, L_item in enumerate(self.regularization_matrices):
                if L_item and sqrt_scaled_lambdas[i] > 0:
                    num_L_rows = L_item.op.shape[0]
                    y_part = y_block[row : row + num_L_rows, :]
                    x_block += sqrt_scaled_lambdas[i] * _apply_op_T_to_block(L_item.op, y_part)
                    row += num_L_rows
            return x_block

        shape = (op_rows * num_scenarios, num_features * num_scenarios)

        # This is the contract with the 1D solver. We define a consistent way to
        # move between the 1D view of the solver and our 2D column-block view.
        # By using the default C-ordering for both reshape and flatten, we establish
        # a simple, consistent, and NumPy-native contract.
        def matvec_final(x_flat):
            x_block = x_flat.reshape(num_features, num_scenarios)
            return matvec_block(x_block).flatten()

        def rmatvec_final(y_flat):
            y_block = y_flat.reshape(op_rows, num_scenarios)
            return rmatvec_block(y_block).flatten()

        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        self._op_cache[cache_key] = (op, rmatvec_block)
        return op, rmatvec_block

    def solve(self, b, **kwargs):
        """Solves the least-squares problem for the given right-hand-side(s)."""
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        solver_map = {
            "normal": self._solve_normal_or_svd,
            "svd": self._solve_normal_or_svd,
            "lsmr": self._solve_lsmr,
            "cg": self._solve_cg,
        }
        return solver_map[self.solver](b_list, **kwargs)

    def _process_b_vector(self, b_val, data_shape):
        """Validates and reshapes a `b` vector into a 2D column-block format."""
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

    def _get_full_stacked_operator(self):
        """Builds a single dense matrix G for the entire system for SVD/Normal solvers."""
        cache_key = "G_dense"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]
        if self.is_matrix_free:
            raise ValueError("SVD and Normal solvers require dense numpy arrays.")

        scaled_weights = self._get_scaled_regularization_weights()
        sqrt_scaled_lambdas = [np.sqrt(w) for w in scaled_weights]
        all_A_weighted, all_L_weighted = [], []
        for i, a_item in enumerate(self.A):
            op, w_item = self._densify_op(a_item), self.sqrt_weights[i]
            if w_item is not None:
                op = (
                    w_item.op * op if w_item.input_shape == (1,) else self._densify_op(w_item) @ op
                )
            all_A_weighted.append(op)
        for i, L_item in enumerate(self.regularization_matrices):
            if L_item and sqrt_scaled_lambdas[i] > 1e-12:
                all_L_weighted.append(sqrt_scaled_lambdas[i] * self._densify_op(L_item))

        G_dense = np.vstack(all_A_weighted + all_L_weighted)
        self._op_cache[cache_key] = G_dense
        return G_dense

    def _solve_normal_or_svd(self, b_list, **kwargs):
        """Handles the dense solvers (SVD and Normal Equations)."""
        G_dense = self._get_full_stacked_operator()
        if self.solver == "svd":
            if "svd" not in self._op_cache:
                u, s, vt = np.linalg.svd(G_dense, full_matrices=False)
                s_inv = np.zeros_like(s)
                stable_s = s > (self.tolerance * (s[0] if s.size > 0 else 0))
                s_inv[stable_s] = 1.0 / s[stable_s]
                self._op_cache["svd"] = (u, s_inv, vt)
            u, s_inv, vt = self._op_cache["svd"]
        else:  # 'normal'
            if "normal_G_T_G" not in self._op_cache:
                self._op_cache["normal_G_T_G"] = G_dense.T.conj() @ G_dense
            G_T_G = self._op_cache["normal_G_T_G"]

        solutions = []
        for i, b_val in enumerate(b_list):
            b_col_block, scenario_shape = self._process_b_vector(b_val, self.data_shapes[i])
            if b_col_block is None:
                solutions.append(None)
                continue

            d_block = np.zeros((G_dense.shape[0], b_col_block.shape[1]), dtype=G_dense.dtype)
            w_item = self.sqrt_weights[i]
            weighted_b = b_col_block
            if w_item is not None:
                weighted_b = (
                    w_item.op * b_col_block
                    if w_item.input_shape == (1,)
                    else self._densify_op(w_item) @ b_col_block
                )

            start_row = sum(a.op.shape[0] for a in self.A[:i])
            d_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b

            if self.solver == "svd":
                sol_block = vt.T.conj() @ (s_inv[:, np.newaxis] * (u.T.conj() @ d_block))
            else:
                sol_block = np.linalg.solve(G_T_G, G_dense.T.conj() @ d_block)

            solutions.append(sol_block.reshape(self.solution_shape + scenario_shape))
        return solutions

    def _solve_lsmr(self, b_list, **kwargs):
        """Solves the system using LSMR, iterating through each b term."""
        solutions, lsmr_kwargs = [], {"atol": self.tolerance, "btol": self.tolerance, **kwargs}
        for i, b_val in enumerate(b_list):
            b_col_block, scenario_shape = self._process_b_vector(b_val, self.data_shapes[i])
            if b_col_block is None:
                solutions.append(None)
                continue

            num_scenarios = b_col_block.shape[1]
            op, _ = self._get_multi_scenario_operator(num_scenarios)
            op_rows = op.shape[0] // num_scenarios

            rhs_block = np.zeros((op_rows, num_scenarios), dtype=op.dtype)
            w_item = self.sqrt_weights[i]
            weighted_b = b_col_block
            if w_item is not None:
                weighted_b = (
                    w_item.op * b_col_block
                    if w_item.input_shape == (1,)
                    else w_item.op @ b_col_block
                )

            start_row = sum(a.op.shape[0] for a in self.A[:i])
            rhs_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b

            sol_flat, *_ = lsmr(op, rhs_block.flatten(), **lsmr_kwargs)
            sol_block = sol_flat.reshape(self.solution_size, num_scenarios)
            solutions.append(sol_block.reshape(self.solution_shape + scenario_shape))
        return solutions

    def _solve_cg(self, b_list, **kwargs):
        """Solves the system using Conjugate Gradient on the normal equations."""
        cg_op, M = self._get_cg_system()
        solutions, cg_kwargs = (
            [],
            {"atol": self.tolerance, "rtol": self.tolerance, "M": M, **kwargs},
        )
        for i, b_val in enumerate(b_list):
            b_col_block, scenario_shape = self._process_b_vector(b_val, self.data_shapes[i])
            if b_col_block is None:
                solutions.append(None)
                continue

            num_scenarios = b_col_block.shape[1]
            _, rmatvec_block = self._get_multi_scenario_operator(num_scenarios)
            base_op_rows = self._op_cache[f"op_{num_scenarios}"][0].shape[0] // num_scenarios

            d_block = np.zeros((base_op_rows, num_scenarios), dtype=cg_op.dtype)
            w_item = self.sqrt_weights[i]
            weighted_b = b_col_block
            if w_item is not None:
                weighted_b = (
                    w_item.op * b_col_block
                    if w_item.input_shape == (1,)
                    else w_item.op @ b_col_block
                )

            start_row = sum(a.op.shape[0] for a in self.A[:i])
            d_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b

            rhs_block = rmatvec_block(d_block)
            sol_block = np.zeros_like(rhs_block)
            for k in range(num_scenarios):
                sol_block[:, k], exit_code = cg(cg_op, rhs_block[:, k], **cg_kwargs)
                if exit_code != 0:
                    print(
                        f"Warning: CG solver did not converge for b-term {i}, scenario {k}, exit code {exit_code}"
                    )

            solutions.append(sol_block.reshape(self.solution_shape + scenario_shape))
        return solutions

    def _get_cg_system(self):
        """Builds and caches the single-vector operator system for the CG solver."""
        cache_key = "cg_system"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        # Create the normal equation operator G.H @ G, which acts on single vectors
        base_op, _ = self._get_multi_scenario_operator(num_scenarios=1)

        def normal_op_matvec(x):
            return base_op.rmatvec(base_op.matvec(x))

        cg_op = LinearOperator(
            (self.solution_size, self.solution_size),
            matvec=normal_op_matvec,
            rmatvec=normal_op_matvec,
            dtype=base_op.dtype,
        )

        # Create the preconditioner
        M = None
        if self.preconditioner == "jacobi":
            if "jacobi_diag" not in self._op_cache:
                self._get_scaled_regularization_weights()
            diag = self._op_cache.get("jacobi_diag")
            if diag is not None and np.any(diag > 1e-12):
                diag[np.abs(diag) < 1e-12] = 1.0

                def precon_matvec(x):
                    return x / diag

                M = LinearOperator(
                    (self.solution_size, self.solution_size),
                    matvec=precon_matvec,
                    rmatvec=precon_matvec,
                    dtype=diag.dtype,
                )

        system = (cg_op, M)
        self._op_cache[cache_key] = system
        return system
