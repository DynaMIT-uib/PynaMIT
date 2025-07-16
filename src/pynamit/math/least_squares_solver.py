"""Least-squares solver module.

This module contains the LeastSquaresSolver class for solving complex,
multi-term least-squares problems.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
import math
from dataclasses import dataclass

ITERATION_SAFETY_FACTOR = 10

@dataclass
class _ProcessedItem:
    """Holds a processed operator and its logical multi-dimensional shape information."""
    op: "np.ndarray | LinearOperator"
    output_shape: tuple
    input_shape: tuple

class LeastSquaresSolver:
    """
    Solves complex least-squares problems with a fully lazy-loaded, unified API.

    This class provides a flexible and explicit API for defining and solving problems
    of the form:

        minimize || A_1 x - b_1 ||_W_1^2 + ... + || A_n x - b_n ||_W_n^2 +
                 || L_1 x ||_lambda_1^2 + ... + || L_m x ||_lambda_m^2

    The solver uses a lazy initialization pattern. The constructor (`__init__`) is
    lightweight and only defines the problem. The expensive one-time setup
    (e.g., matrix decomposition or preconditioner calculation) is deferred and
    performed transparently on the first call to the `solve()` method.

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
        of the desired weights. These define the problem and modify the solution.
    regularization_weights : float | list[float], optional
        A single scalar weight or a list of weights `lambda_j` for each regularization term.
        These are auto-scaled to be commensurate with the data terms.
    regularization_matrices : np.ndarray | LinearOperator | list, optional
        A single N-D regularization operator `L_j` or a list of them.
    solver : str, default="svd"
        The solver to use. One of {"svd", "normal", "lsmr", "cg"}.
    tolerance : float, default=1e-12
        The tolerance for iterative solvers ("lsmr", "cg") or for SVD truncation.
    preconditioner : str, optional
        The preconditioner to use. If set to "jacobi", a Jacobi (diagonal)
        preconditioner is used for the "cg" and "lsmr" solvers.
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
        preconditioner="jacobi",
    ):
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers: raise ValueError(f"Solver must be one of {solvers}")
        if preconditioner is not None and preconditioner != "jacobi": raise ValueError("Only 'jacobi' preconditioner supported.")
        if preconditioner is not None and solver not in ["cg", "lsmr"]:
             print(f"Warning: Preconditioner is set but only applies to 'cg' or 'lsmr' solvers.")
        self.solver = solver
        self.tolerance = tolerance
        self.preconditioner = preconditioner
        self.solution_shape = ((solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape))
        self.solution_size = math.prod(self.solution_shape)
        A_list = self._prepare_input_list(A, "A", allow_single_item=True)
        self.num_data_terms = len(A_list)
        self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        self.A = [self._flatten(op, output_shape=self.data_shapes[i], input_shape=self.solution_shape) for i, op in enumerate(A_list)]
        sqrt_weights_list = self._prepare_input_list(sqrt_weights, "sqrt_weights", count=self.num_data_terms)
        self.sqrt_weights = []
        for i, w_val in enumerate(sqrt_weights_list):
            if w_val is None: self.sqrt_weights.append(None); continue
            flat_data_dim = math.prod(self.data_shapes[i])
            is_diagonal = not isinstance(w_val, LinearOperator) and w_val.size == flat_data_dim
            if is_diagonal:
                w_op = np.ascontiguousarray(w_val).reshape(flat_data_dim, 1)
                self.sqrt_weights.append(_ProcessedItem(op=w_op, output_shape=self.data_shapes[i], input_shape=(1,)))
            else:
                self.sqrt_weights.append(self._flatten(w_val, output_shape=self.data_shapes[i], input_shape=self.data_shapes[i]))
        reg_L_list = self._prepare_input_list(regularization_matrices, "regularization_matrices", allow_single_item=True, is_optional=True)
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [self._flatten(L, input_shape=self.solution_shape) if L is not None else None for L in reg_L_list]
        for l_item in self.regularization_matrices:
            if l_item and l_item.op.shape[-1] != self.solution_size: raise ValueError(f"Shape mismatch in regularization term. Expected {self.solution_size} columns, got {l_item.op.shape[-1]}")
        self.regularization_weights = self._prepare_input_list(regularization_weights, "regularization_weights", count=self.num_reg_terms, default_val=0.0)
        self.is_matrix_free = any(isinstance(a.op, LinearOperator) for a in self.A) or any(L is not None and isinstance(L.op, LinearOperator) for L in self.regularization_matrices)
        if self.is_matrix_free and solver in ["normal", "svd"]: print(f"Warning: Solver '{solver}' with matrix-free operators requires densification, which may be slow or memory-intensive.")
        self._op_cache = {}

    @staticmethod
    def _prepare_input_list(item, name, count=None, allow_single_item=False, is_optional=False, default_val=None):
        """Standardizes user input into a list of a specific length."""
        if item is None:
            if is_optional: return []
            return [default_val] * count if count is not None else []
        lst = item if isinstance(item, list) else [item]
        if allow_single_item and count is None: count = len(lst)
        if len(lst) == 1 and count is not None and count > 1: lst *= count
        if count is not None and len(lst) != count: raise ValueError(f"Input '{name}' has {len(lst)} items, expected {count}.")
        return lst

    def _normalize_data_shapes(self, data_shapes, expected_count):
        """Standardizes the data_shapes input into a list of tuples."""
        if not isinstance(data_shapes, list): data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1: data_shapes *= expected_count
        if len(data_shapes) != expected_count: raise ValueError(f"Number of data_shapes ({len(data_shapes)}) does not match number of A operators ({expected_count}).")
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    @staticmethod
    def _flatten(array, output_shape=None, input_shape=None):
        """Converts an N-D operator into a 2D matrix representation."""
        if isinstance(array, LinearOperator): return _ProcessedItem(array, (array.shape[0],), (array.shape[1],))
        if not isinstance(array, np.ndarray): raise TypeError(f"Input must be a numpy array or LinearOperator, got {type(array)}")
        if output_shape is None and input_shape is None: raise ValueError("At least one of output_shape or input_shape must be provided for an operator.")
        array = np.ascontiguousarray(array)
        if input_shape is None:
            flat_output_dim = math.prod(output_shape)
            if array.size % flat_output_dim != 0: raise ValueError(f"Array size {array.size} not divisible by product of output_shape {output_shape}")
            flat_input_dim = array.size // flat_output_dim
            input_shape = (flat_input_dim,)
        elif output_shape is None:
            flat_input_dim = math.prod(input_shape)
            if array.size % flat_input_dim != 0: raise ValueError(f"Array size {array.size} not divisible by product of input_shape {input_shape}")
            flat_output_dim = array.size // flat_input_dim
            output_shape = (flat_output_dim,)
        flat_input_dim = math.prod(input_shape)
        flat_output_dim = math.prod(output_shape)
        return _ProcessedItem(array.reshape(flat_output_dim, flat_input_dim), output_shape, input_shape)

    def _densify_op(self, item):
        """Converts a _ProcessedItem into a dense numpy array if it isn't one already."""
        if item is None: return None
        op = item.op
        if isinstance(op, LinearOperator): return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op
    
    def _process_b_vector(self, b_val, data_shape):
        """Validates and reshapes a `b` vector into a 2D column-block format."""
        if b_val is None: return None, None
        num_data_dims = len(data_shape)
        is_exact = b_val.shape == data_shape
        is_multi_scenario = (b_val.ndim > num_data_dims and b_val.shape[:num_data_dims] == data_shape)
        is_flat_single_scenario = b_val.ndim == 1 and b_val.size == math.prod(data_shape)
        if not (is_exact or is_multi_scenario or is_flat_single_scenario): raise ValueError(f"Shape of b term {b_val.shape} is incompatible with its data_shape {data_shape}.")
        scenario_shape = b_val.shape[num_data_dims:] if is_multi_scenario else ()
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        b_col_block = np.ascontiguousarray(b_val).reshape(math.prod(data_shape), num_scenarios)
        return b_col_block, scenario_shape

    def _calculate_and_cache_scaled_lambdas(self):
        """Auto-scales regularization weights to be commensurate with the data terms."""
        if "scaled_lambdas" in self._op_cache: return
        data_op, _, _ = self._get_multi_scenario_operator(num_scenarios=1, use_scaled_lambdas=False, include_regularization=False)
        diag_A_T_A = np.zeros(self.solution_size, dtype=data_op.dtype)
        for i in range(self.solution_size):
            e_i = np.zeros(self.solution_size); e_i[i] = 1.0
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
                e_j = np.zeros(self.solution_size); e_j[j] = 1.0
                col_j = L_op.matvec(e_j) if isinstance(L_op, LinearOperator) else self._densify_op(L_item)[:, j]
                diag_L_T_L[j] = np.dot(col_j.conj(), col_j).real
            reg_scale = np.median(diag_L_T_L[diag_L_T_L > 0]) if np.any(diag_L_T_L > 0) else 1.0
            scaled_lambda = np.sqrt(raw_weight) * np.sqrt(data_scale / reg_scale) if reg_scale > 1e-14 else 0.0
            scaled_lambdas.append(scaled_lambda)
        self._op_cache["scaled_lambdas"] = scaled_lambdas

    def _get_multi_scenario_operator(self, num_scenarios, use_scaled_lambdas, include_regularization):
        """Builds the full system LinearOperator G such that the problem is min ||Gx - d||."""
        lambdas = self._op_cache.get("scaled_lambdas", self.regularization_weights) if use_scaled_lambdas else self.regularization_weights
        num_features = self.solution_size
        op_rows_data = sum(a.op.shape[0] for a in self.A)
        op_rows_reg = sum(l.op.shape[0] for i, l in enumerate(self.regularization_matrices) if i < len(lambdas) and l and lambdas[i] > 0) if include_regularization else 0
        op_rows = op_rows_data + op_rows_reg
        dtype = self.A[0].op.dtype
        def _apply_op_to_block(op, x_block):
            if isinstance(op, LinearOperator):
                return op.matmat(x_block)
            return op @ x_block
        def _apply_op_T_to_block(op, y_block):
            if isinstance(op, LinearOperator):
                return op.rmatmat(y_block)
            return op.T.conj() @ y_block
        def matvec_block(x_block):
            output_blocks = []
            for i, a_item in enumerate(self.A):
                res_block = _apply_op_to_block(a_item.op, x_block)
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    res_block = w_item.op * res_block if w_item.input_shape == (1,) else _apply_op_to_block(w_item.op, res_block)
                output_blocks.append(res_block)
            if include_regularization:
                for i, L_item in enumerate(self.regularization_matrices):
                    if i < len(lambdas) and L_item and lambdas[i] > 0:
                        res_block = _apply_op_to_block(L_item.op, x_block)
                        output_blocks.append(lambdas[i] * res_block)
            return np.vstack(output_blocks) if output_blocks else np.zeros((0, x_block.shape[1]), dtype=dtype)
        def rmatvec_block(y_block):
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i, a_item in enumerate(self.A):
                num_a_rows = a_item.op.shape[0]
                y_part = y_block[row : row + num_a_rows, :]
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    y_part = w_item.op.conj() * y_part if w_item.input_shape == (1,) else _apply_op_T_to_block(w_item.op, y_part)
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
        """Builds a single dense matrix G for the entire system, using the most efficient path."""
        if "G_dense" in self._op_cache: return self._op_cache["G_dense"]
        if self.is_matrix_free:
            base_op, _, _ = self._get_multi_scenario_operator(num_scenarios=1, use_scaled_lambdas=True, include_regularization=True)
            G_dense = base_op.matmat(np.eye(self.solution_size))
        else:
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
            G_dense = np.vstack(all_A_weighted + all_L_weighted)
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
        """Lazily computes and caches the matrices for the normal equations."""
        if "normal_components" not in self._op_cache:
            G_dense = self._get_full_stacked_operator()
            G_T_G = G_dense.T.conj() @ G_dense
            G_T = G_dense.T.conj()
            self._op_cache["normal_components"] = (G_T_G, G_T)
        return self._op_cache["normal_components"]

    def _setup_preconditioner_components(self):
        """Calculates and caches the Jacobi preconditioner diagonal."""
        if "jacobi_diag" in self._op_cache: return
        print(f"Calculating Jacobi preconditioner for {'matrix-free' if self.is_matrix_free else 'dense'} operator...")
        base_op, _, _ = self._get_multi_scenario_operator(num_scenarios=1, use_scaled_lambdas=True, include_regularization=True)
        diag_G_T_G = np.zeros(self.solution_size, dtype=base_op.dtype)
        for i in range(self.solution_size):
            e_i = np.zeros(self.solution_size); e_i[i] = 1.0
            col_i = base_op.matvec(e_i)
            diag_G_T_G[i] = np.dot(col_i.conj(), col_i).real
        self._op_cache["jacobi_diag"] = diag_G_T_G

    def _get_lsmr_components(self, num_scenarios):
        """Gets the operator and solution transform needed for the LSMR solver."""
        if f"lsmr_components_{num_scenarios}" in self._op_cache: return self._op_cache[f"lsmr_components_{num_scenarios}"]
        base_op, _, _ = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
        op_to_solve = base_op
        solution_transform = lambda sol_block: sol_block
        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache["jacobi_diag"]
            sqrt_inv_diag = np.sqrt(1.0 / diag)
            sqrt_inv_diag[np.isinf(sqrt_inv_diag)] = 1.0
            def precond_matvec(y_flat):
                y_block = y_flat.reshape(self.solution_size, num_scenarios)
                x_block = y_block * sqrt_inv_diag[:, np.newaxis]
                return base_op.matvec(x_block.flatten())
            def precond_rmatvec(d_flat):
                d_block_in = d_flat.reshape(-1, num_scenarios)
                x_block_T_flat = base_op.rmatvec(d_block_in.flatten())
                x_block_T = x_block_T_flat.reshape(self.solution_size, num_scenarios)
                return (x_block_T * sqrt_inv_diag[:, np.newaxis]).flatten()
            op_to_solve = LinearOperator(base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype)
            solution_transform = lambda sol_y_block: sol_y_block * sqrt_inv_diag[:, np.newaxis]
        
        components = (op_to_solve, solution_transform)
        self._op_cache[f"lsmr_components_{num_scenarios}"] = components
        return components

    def _get_cg_components(self):
        """Gets the operator and preconditioner needed for the CG solver."""
        if "cg_components" in self._op_cache: return self._op_cache["cg_components"]
        base_op, _, _ = self._get_multi_scenario_operator(num_scenarios=1, use_scaled_lambdas=True, include_regularization=True)
        def normal_op_matvec(x): return base_op.rmatvec(base_op.matvec(x))
        cg_op = LinearOperator((self.solution_size, self.solution_size), matvec=normal_op_matvec, rmatvec=normal_op_matvec, dtype=base_op.dtype)
        M = None
        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache.get("jacobi_diag")
            diag_inv = 1.0 / diag
            diag_inv[np.isinf(diag_inv)] = 1.0
            def precon_matvec(x): return x * diag_inv
            M = LinearOperator((self.solution_size, self.solution_size), matvec=precon_matvec, rmatvec=precon_matvec, dtype=diag.dtype)
        
        components = (cg_op, M)
        self._op_cache["cg_components"] = components
        return components

    def solve(self, b, **kwargs):
        """Solves the least-squares problem for the given RHS data."""
        self._calculate_and_cache_scaled_lambdas()
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed_b = [self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)]
        valid_b = [(p[0], p[1]) for p in processed_b if p[0] is not None]
        dtype = self.A[0].op.dtype
        if not valid_b: return np.zeros(self.solution_shape, dtype=dtype)
        
        scenario_shape = valid_b[0][1]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1

        lambdas = self._op_cache.get("scaled_lambdas")
        op_rows_data = sum(a.op.shape[0] for a in self.A)
        op_rows_reg = sum(l.op.shape[0] for i, l in enumerate(self.regularization_matrices) if i < len(lambdas) and l and lambdas[i] > 0)
        op_rows = op_rows_data + op_rows_reg
        d_block = np.zeros((op_rows, num_scenarios), dtype=dtype)
        
        current_row = 0
        for i, b_val in enumerate(b_list):
            num_a_rows = self.A[i].op.shape[0]
            if b_val is None:
                current_row += num_a_rows
                continue
            b_col_block, b_scenario_shape = processed_b[i]
            if b_scenario_shape != scenario_shape: raise ValueError("Inconsistent scenario shapes in b terms.")
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                w_op = self._densify_op(w_item) if not isinstance(w_item.op, np.ndarray) or w_item.input_shape != (1,) else w_item.op
                b_col_block = w_op * b_col_block if w_item.input_shape == (1,) else w_op @ b_col_block
            d_block[current_row : current_row + num_a_rows, :] = b_col_block
            current_row += num_a_rows
        
        if self.solver == "svd":
            u, s_inv, vt = self._get_svd_components()
            sol_block = vt.T.conj() @ (s_inv[:, np.newaxis] * (u.T.conj() @ d_block))
        elif self.solver == "normal":
            G_T_G, G_T = self._get_normal_components()
            G_T_d = G_T @ d_block
            sol_block = np.linalg.solve(G_T_G, G_T_d)
        elif self.solver == "lsmr":
            op_to_solve, solution_transform = self._get_lsmr_components(num_scenarios)
            m, n = op_to_solve.shape; m, n = m // num_scenarios, n // num_scenarios
            max_iter = ITERATION_SAFETY_FACTOR * min(m, n) if min(m, n) > 0 else self.solution_size
            lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance, "maxiter": max_iter, **kwargs}
            sol_y_flat, istop, *_ = lsmr(op_to_solve, d_block.flatten(), **lsmr_kwargs)
            if istop not in [0, 1, 2]: print(f"Warning: LSMR may not have fully converged (istop={istop}).")
            sol_y_block = sol_y_flat.reshape(self.solution_size, num_scenarios)
            sol_block = solution_transform(sol_y_block)
        elif self.solver == "cg":
            _, rmatvec_block, _ = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
            cg_op, M = self._get_cg_components()
            rhs_block = rmatvec_block(d_block)
            sol_block = np.zeros_like(rhs_block)
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            for k in range(num_scenarios):
                sol_block[:, k], exit_code = cg(cg_op, rhs_block[:, k], **cg_kwargs)
                if exit_code != 0: print(f"Warning: CG solver did not converge for scenario {k} (exit_code={exit_code}).")
        
        final_shape = self.solution_shape + scenario_shape
        return sol_block.reshape(final_shape)

    def solve_adjoint(self, y, **kwargs):
        """
        Solves the adjoint of the least-squares problem.

        If the forward problem is x = S(b), this computes S^T @ y.
        Mathematically, this is S^T @ y = G @ (G^T @ G)^-1 @ y, where G is the
        full (weighted and regularized) system matrix and y is a vector with the
        same shape as the solution x.

        Parameters
        ----------
        y : np.ndarray
            The input vector for the adjoint operation. Must have a shape
            that is broadcastable to `self.solution_shape`. Can include
            additional trailing dimensions for multiple scenarios.
        **kwargs : dict
            Additional keyword arguments passed to the underlying iterative solvers
            (e.g., `rtol`, `atol`, `maxiter` for `cg`). Defaults are inherited
            from the solver instance.

        Returns
        -------
        list[np.ndarray]
            A list of gradients, one for each of the `b` terms in the forward
            problem. The shape of each gradient matches the shape of the
            corresponding `b` term (including any scenario dimensions).
        """
        self._calculate_and_cache_scaled_lambdas()

        # 1. Process input vector `y` into a block of column vectors
        if not isinstance(y, np.ndarray): y = np.array(y, dtype=self.A[0].op.dtype)
        y_ndim, sol_ndim = y.ndim, len(self.solution_shape)

        if y_ndim < sol_ndim or y.shape[:sol_ndim] != self.solution_shape:
            if y_ndim == 1 and y.size % self.solution_size == 0: # Flattened input
                num_scenarios = y.size // self.solution_size
                scenario_shape = (num_scenarios,) if num_scenarios > 1 else ()
            else:
                raise ValueError(f"Shape of y {y.shape} is incompatible with solution_shape {self.solution_shape}.")
        else: # Multi-dimensional input with potential scenario dimensions
            scenario_shape = y.shape[sol_ndim:]
        
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        y_block = np.ascontiguousarray(y).reshape(self.solution_size, num_scenarios)

        # 2. Solve (G^T G) z = y for z. This intermediate step is required for all solver types.
        z_block = np.zeros_like(y_block)
        
        if self.solver == "svd":
            # For SVD, the solution is direct. Tolerance was used when creating s_inv.
            _, s_inv, vt = self._get_svd_components()
            s_inv_sq = s_inv**2
            z_block = vt.T.conj() @ (s_inv_sq[:, np.newaxis] * (vt @ y_block))
        elif self.solver == "normal":
            # For normal equations, the solution is direct.
            G_T_G, _ = self._get_normal_components()
            z_block = np.linalg.solve(G_T_G, y_block)
        elif self.solver in ["cg", "lsmr"]:
            # For iterative solvers, we solve the system using CG.
            # This is the most efficient way to solve the symmetric positive definite system (G^T G)z = y.
            cg_op, M = self._get_cg_components()
            
            # --- MODIFIED SECTION ---
            # Set up CG arguments, inheriting defaults from the solver instance
            # and allowing overrides from the user's call to this function.
            # This ensures consistent tolerance behavior with the forward solve.
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {
                "rtol": self.tolerance, # Use instance's tolerance as default rtol
                "M": M,
                "maxiter": max_iter,
                **kwargs # User-provided kwargs override the defaults
            }
            # --- END MODIFIED SECTION ---

            for k in range(num_scenarios):
                sol, exit_code = cg(cg_op, y_block[:, k], **cg_kwargs)
                if exit_code != 0: print(f"Warning: Adjoint CG solver did not converge for scenario {k} (exit_code={exit_code}).")
                z_block[:, k] = sol
        
        # 3. Compute grad_d = G @ z
        _, _, matvec_block_fn = self._get_multi_scenario_operator(num_scenarios, use_scaled_lambdas=True, include_regularization=True)
        grad_d_block = matvec_block_fn(z_block)
        
        # 4. Un-stack and un-weight to get gradients w.r.t. each b term
        grad_b_list = []
        current_row = 0
        for i in range(self.num_data_terms):
            num_a_rows = self.A[i].op.shape[0]
            grad_d_i = grad_d_block[current_row : current_row + num_a_rows, :]
            
            grad_b_i = grad_d_i
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                if w_item.input_shape == (1,): # Diagonal weight
                    grad_b_i = w_item.op.conj() * grad_d_i
                else: # Matrix/LinearOperator weight
                    w_op = w_item.op
                    grad_b_i = w_op.rmatmat(grad_d_i) if isinstance(w_op, LinearOperator) else w_op.T.conj() @ grad_d_i
            
            output_shape = self.data_shapes[i] + scenario_shape
            grad_b_list.append(grad_b_i.reshape(output_shape))
            current_row += num_a_rows
            
        return grad_b_list