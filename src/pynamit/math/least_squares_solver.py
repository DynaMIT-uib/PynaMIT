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
    """Holds a operator and its multi-dimensional shape information."""

    op: "np.ndarray | LinearOperator"
    output_shape: tuple
    input_shape: tuple


class LeastSquaresSolver:
    """
    Solves least-squares problems.

    This class provides a flexible and explicit API for defining and
    solving problems of the form:

        minimize || A_1 x - b_1 ||_W_1^2
                 + ... + || A_n x - b_n ||_W_n^2
                 + || L_1 x ||_lambda_1^2
                 + ... + || L_m x ||_lambda_m^2

    The solver uses a lazy initialization pattern. The constructor
    (`__init__`) is lightweight and only defines the problem. The
    expensive one-time setup (e.g., matrix decomposition or
    preconditioner calculation) is deferred and performed transparently
    on the first call to the `solve()` or `solve_adjoint()` methods.

    Parameters
    ----------
    A : np.ndarray | LinearOperator | list
        A single N-D array/LinearOperator or a list of them for the
        data-fitting terms `A_i`.
    solution_shape : int | tuple
        The multi-dimensional shape of the solution vector `x`.
    data_shapes : tuple | list[tuple]
        A list of tuples, where `data_shapes[i]` is the
        multi-dimensional shape of the output space for operator `A[i]`.
        Must be provided for each `A_i`.
    sqrt_weights : np.ndarray | list, optional
        A single N-D weight array or a list of them, representing the
        square root of the desired weights `W_i`. These can be diagonal
        (1D array) or dense.
    regularization_weights : float | list[float], optional
        A single scalar weight or a list of weights `lambda_j` for each
        regularization term.
    regularization_matrices : np.ndarray | LinearOperator
                              | list, optional
        A single N-D regularization operator `L_j` or a list of them.
    solver : str, default="svd"
        The solver to use. One of {"svd", "normal", "lsmr", "cg"}.
    tolerance : float, default=1e-12
        The tolerance for iterative solvers ("lsmr", "cg") or for SVD
        truncation.
    preconditioner : str, optional
        The preconditioner to use for "cg" and "lsmr" solvers.
        One of {None, "jacobi", "pinv"}.
        - "jacobi": A diagonal (Jacobi) preconditioner. Fast to compute.
        - "pinv": A preconditioner based on the pseudoinverse (via SVD).
          Much more expensive to compute, but can significantly accelerate
          convergence. Requires densification of all operators.
    """

    def __init__(
        self,
        A,
        solution_shape,
        data_shapes,
        sqrt_weights=None,
        regularization_weights=None,
        regularization_matrices=None,
        solver="normal",
        tolerance=1e-12,
        preconditioner=None,
        picard_plot=False,
    ):
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
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)
        self.num_data_terms = 0  # Will be set in update_matrices
        self.is_matrix_free = False  # Will be set in update_matrices

        # Defer matrix processing to a separate method
        self.update_matrices(A, sqrt_weights=sqrt_weights, data_shapes=data_shapes)

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
                    "Shape mismatch in regularization term. "
                    f"Expected {self.solution_size} columns, got {l_item.op.shape[-1]}"
                )
        self.regularization_weights = self._prepare_input_list(
            regularization_weights,
            "regularization_weights",
            count=self.num_reg_terms,
            default_val=0.0,
        )

        is_reg_matrix_free = any(
            L is not None and isinstance(L.op, LinearOperator)
            for L in self.regularization_matrices
        )
        self.is_matrix_free = self.is_matrix_free or is_reg_matrix_free

        if self.is_matrix_free:
            if solver in ["normal", "svd"]:
                print(
                    f"Warning: Solver '{solver}' with matrix-free operators requires "
                    "densification, which may be slow or memory-intensive."
                )
            if self.preconditioner == "pinv":
                raise ValueError(
                    "The 'pinv' preconditioner requires densifying the full system "
                    "matrix and cannot be used with matrix-free operators."
                )

        if picard_plot:
            self.picard_plot()

    def update_matrices(self, A, sqrt_weights=None, data_shapes=None):
        """
        Updates the data-fitting matrices (A) and weights for the problem.

        This method is designed for time-stepping scenarios where the system
        operators change but the problem structure (and potentially the
        preconditioner) remains the same. It updates the internal `A` and
        `sqrt_weights` attributes and clears any cached operators that depend
        on them, but preserves the expensive-to-calculate preconditioner
        by default.

        Parameters
        ----------
        A : np.ndarray | LinearOperator | list
            A single N-D array/LinearOperator or a list of them for the
            data-fitting terms `A_i`.
        sqrt_weights : np.ndarray | list, optional
            New weights corresponding to the new `A` matrices. If None,
            the existing weights are cleared.
        data_shapes : tuple | list[tuple], optional
            New data shapes corresponding to the new `A` matrices. Must be
            provided if the number or shape of operators changes. If None,
            the existing shapes are reused.
        """
        A_list = self._prepare_input_list(A, "A", allow_single_item=True)
        self.num_data_terms = len(A_list)

        if data_shapes is not None:
            self.data_shapes = self._normalize_data_shapes(data_shapes, self.num_data_terms)
        elif not hasattr(self, "data_shapes") or len(self.data_shapes) != self.num_data_terms:
            raise ValueError(
                "data_shapes must be provided when setting A for the first time or changing the number of A operators."
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

        self.is_matrix_free = any(isinstance(a.op, LinearOperator) for a in self.A)
        # Clear caches that depend on A, but not the preconditioner itself.
        self.clear_cache(clear_preconditioner=False)

    def update_preconditioner(self):
        """
        Flags the preconditioner to be re-calculated on the next `solve` call.

        This method should be called when changes to the system matrices are
        significant enough to warrant re-computing the (potentially expensive)
        preconditioner.
        """
        print("Preconditioner invalidated. It will be re-computed on the next call to solve().")
        self.clear_cache(clear_preconditioner=True)

    def clear_cache(self, clear_preconditioner=True):
        """
        Clears cached internal matrices and operators.

        Parameters
        ----------
        clear_preconditioner : bool, default=True
            If True, clears all cached items, including any computed
            preconditioners ('jacobi_diag', 'pinv_components').
            If False, preserves the preconditioner cache but clears caches
            that depend on the system matrices ('G_dense', solver components, etc.).
        """
        problem_specific_keys = ["scaled_lambdas", "G_dense", "normal_components"]
        # Remove solver-specific components which depend on G
        solver_keys = [
            k for k in self._op_cache if k.startswith(("lsmr_components", "cg_components"))
        ]

        for key in problem_specific_keys + solver_keys:
            if key in self._op_cache:
                del self._op_cache[key]

        if clear_preconditioner:
            preconditioner_keys = ["jacobi_diag", "pinv_components", "svd_components"]
            for key in preconditioner_keys:
                if key in self._op_cache:
                    del self._op_cache[key]

    # --- Internal helper methods ---
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
                f"Number of data_shapes ({len(data_shapes)}) does not match number of "
                f"A operators ({expected_count})."
            )
        return [(shape,) if isinstance(shape, int) else tuple(shape) for shape in data_shapes]

    @staticmethod
    def _flatten(array, output_shape=None, input_shape=None):
        """Convert an N-D operator into a 2D matrix representation."""
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
                    f"Array size {array.size} not divisible by product of "
                    f"output_shape {output_shape}"
                )
            flat_input_dim = array.size // flat_output_dim
            input_shape = (flat_input_dim,)
        elif output_shape is None:
            flat_input_dim = math.prod(input_shape)
            if array.size % flat_input_dim != 0:
                raise ValueError(
                    f"Array size {array.size} not divisible by product of "
                    f"input_shape {input_shape}"
                )
            flat_output_dim = array.size // flat_input_dim
            output_shape = (flat_output_dim,)
        flat_input_dim = math.prod(input_shape)
        flat_output_dim = math.prod(output_shape)
        return _ProcessedItem(
            array.reshape(flat_output_dim, flat_input_dim), output_shape, input_shape
        )

    def _densify_op(self, item):
        """Ensure a _ProcessedItem is a dense numpy array."""
        if item is None:
            return None
        op = item.op
        if isinstance(op, LinearOperator):
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

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

    def _get_multi_scenario_operator(
        self, num_scenarios, use_scaled_lambdas, include_regularization
    ):
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

        def matvec_final(x_flat):
            return matvec_block(x_flat.reshape(num_features, num_scenarios)).flatten()

        def rmatvec_final(y_flat):
            return rmatvec_block(y_flat.reshape(op_rows, num_scenarios)).flatten()

        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=dtype)
        return op, rmatvec_block, matvec_block

    def _get_full_stacked_operator(self):
        """Build a single dense matrix G for the entire system."""
        if "G_dense" in self._op_cache:
            return self._op_cache["G_dense"]
        if self.is_matrix_free:
            base_op, _, _ = self._get_multi_scenario_operator(
                num_scenarios=1, use_scaled_lambdas=True, include_regularization=True
            )
            G_dense = base_op.matmat(np.eye(self.solution_size))
        else:
            lambdas = self._op_cache.get("scaled_lambdas", self.regularization_weights)
            all_A_weighted, all_L_weighted = [], []
            for i, a_item in enumerate(self.A):
                op, w_item = self._densify_op(a_item), self.sqrt_weights[i]
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
        """Return the matrices for the normal equations."""
        if "normal_components" not in self._op_cache:
            G_dense = self._get_full_stacked_operator()
            G_T_G = G_dense.T.conj() @ G_dense
            G_T = G_dense.T.conj()
            self._op_cache["normal_components"] = (G_T_G, G_T)
        return self._op_cache["normal_components"]

    def _setup_preconditioner_components(self):
        """Return the Jacobi preconditioner diagonal."""
        if "jacobi_diag" in self._op_cache:
            return
        print(
            "Calculating Jacobi preconditioner for "
            f"{'matrix-free' if self.is_matrix_free else 'dense'} operator..."
        )
        base_op, _, _ = self._get_multi_scenario_operator(
            num_scenarios=1, use_scaled_lambdas=True, include_regularization=True
        )
        diag_G_T_G = np.zeros(self.solution_size, dtype=base_op.dtype)
        for i in range(self.solution_size):
            e_i = np.zeros(self.solution_size)
            e_i[i] = 1.0
            col_i = base_op.matvec(e_i)
            diag_G_T_G[i] = np.dot(col_i.conj(), col_i).real
        self._op_cache["jacobi_diag"] = diag_G_T_G

    def _setup_pinv_preconditioner(self):
        """Computes and caches SVD components for the 'pinv' preconditioner."""
        if "pinv_components" in self._op_cache:
            return

        print("Calculating SVD for 'pinv' preconditioner...")
        if self.is_matrix_free:
            # This case is blocked in __init__, but check again for safety.
            raise RuntimeError("Cannot use 'pinv' preconditioner with matrix-free operators.")

        G_dense = self._get_full_stacked_operator()
        u, s, vt = np.linalg.svd(G_dense, full_matrices=False)

        # For preconditioning, s_pinv is 1/s for s > tol, 0 otherwise
        s_pinv = np.zeros_like(s)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        stable_s_mask = s > cutoff
        s_pinv[stable_s_mask] = 1.0 / s[stable_s_mask]

        # For CG, we need (G^T G)^-1 = V S^-2 V^T
        s_inv_sq = s_pinv**2

        self._op_cache["pinv_components"] = (u, s, vt, s_pinv, s_inv_sq)
        print("...SVD calculation complete.")

    def _get_lsmr_components(self, num_scenarios):
        """Return components for the LSMR solver."""
        if f"lsmr_components_{num_scenarios}" in self._op_cache:
            return self._op_cache[f"lsmr_components_{num_scenarios}"]
        base_op, _, matvec_block = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )
        op_to_solve = base_op

        def solution_transform(sol_block):
            return sol_block

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

            op_to_solve = LinearOperator(
                base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype
            )

            def solution_transform(sol_y_block):
                return sol_y_block * sqrt_inv_diag[:, np.newaxis]

        elif self.preconditioner == "pinv":
            self._setup_pinv_preconditioner()
            _, _, vt, s_pinv, _ = self._op_cache["pinv_components"]

            # We are solving G x = d
            # Right preconditioning: x = P y, solve (G P) y = d
            # We choose P = V S_pinv.
            # Then G P = (U S V^T) (V S_pinv) = U S S_pinv.
            # This is a near-identity mapping in the range of G.
            # The operator to solve is G' = G P.
            # x is recovered by x = P y = V S_pinv y.
            # Note: s_pinv is 1D, vt is 2D. Broadcasting is used.
            def precond_matvec(y_flat):
                y_block = y_flat.reshape(self.solution_size, num_scenarios)
                p_y_block = vt.T @ (s_pinv[:, np.newaxis] * (vt @ y_block))
                # Now apply original G to the result
                return matvec_block(p_y_block).flatten()

            def precond_rmatvec(d_flat):
                d_block_in = d_flat.reshape(-1, num_scenarios)
                # Apply G^T first
                g_t_d_block_flat = base_op.rmatvec(d_block_in.flatten())
                g_t_d_block = g_t_d_block_flat.reshape(self.solution_size, num_scenarios)
                # Then apply P^T = (V S_pinv V^T)^T = V S_pinv V^T
                return (vt.T @ (s_pinv[:, np.newaxis] * (vt @ g_t_d_block))).flatten()

            op_to_solve = LinearOperator(
                base_op.shape, matvec=precond_matvec, rmatvec=precond_rmatvec, dtype=base_op.dtype
            )

            def solution_transform(sol_y_block):
                # x = P y = V S_pinv V^T y
                return vt.T @ (s_pinv[:, np.newaxis] * (vt @ sol_y_block))

        self._op_cache[f"lsmr_components_{num_scenarios}"] = (op_to_solve, solution_transform)
        return op_to_solve, solution_transform

    def _get_cg_components(self, num_scenarios=1):
        """Return components for the CG solver."""
        cache_key = f"cg_components_{num_scenarios}"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        base_op, _, _ = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )

        def normal_op_matvec(x_flat):
            """Compute (G^T G) @ x for a flattened block of vectors."""
            return base_op.rmatvec(base_op.matvec(x_flat))

        cg_op = LinearOperator(
            (self.solution_size * num_scenarios, self.solution_size * num_scenarios),
            matvec=normal_op_matvec,
            rmatvec=normal_op_matvec,
            dtype=base_op.dtype,
        )

        M = None
        if self.preconditioner == "jacobi":
            self._setup_preconditioner_components()
            diag = self._op_cache.get("jacobi_diag")
            diag_inv = 1.0 / diag
            diag_inv[np.isinf(diag_inv)] = 1.0
            full_diag_inv = np.tile(diag_inv, num_scenarios)

            def precon_matvec(x_flat):
                return x_flat * full_diag_inv

            M = LinearOperator(
                (self.solution_size * num_scenarios, self.solution_size * num_scenarios),
                matvec=precon_matvec,
                rmatvec=precon_matvec,
                dtype=diag.dtype,
            )
        elif self.preconditioner == "pinv":
            self._setup_pinv_preconditioner()
            _, _, vt, _, s_inv_sq = self._op_cache["pinv_components"]

            # Preconditioner M approximates (G^T G)^-1 = V S^-2 V^T
            def precon_matvec_block(x_block):
                return vt.T @ (s_inv_sq[:, np.newaxis] * (vt @ x_block))

            def precon_matvec(x_flat):
                x_block = x_flat.reshape(self.solution_size, num_scenarios)
                return precon_matvec_block(x_block).flatten()

            M = LinearOperator(
                cg_op.shape, matvec=precon_matvec, rmatvec=precon_matvec, dtype=vt.dtype
            )

        self._op_cache[cache_key] = (cg_op, M)
        return cg_op, M

    def solve(self, b, **kwargs):
        """Solve least-squares problem for right-hand-side data.

        Parameters
        ----------
        b : np.ndarray | None | list[np.ndarray | None]
            The right-hand-side data vector(s) `b_i`. Can be a single
            array if there is only one data term, or a list of arrays
            corresponding to each operator in `A`. Use `None` for terms
            that are missing. Each array can have additional trailing
            "scenario" dimensions.
        **kwargs : dict
            Additional keyword arguments passed to the underlying
            iterative solver (`cg` or `lsmr`), e.g., `atol`, `maxiter`.

        Returns
        -------
        np.ndarray
            The solution vector `x`. Its shape will be
            `self.solution_shape` concatenated with any scenario
            dimensions from `b`.
        """
        self._calculate_and_cache_scaled_lambdas()
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed_b = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid_b = [(p[0], p[1]) for p in processed_b if p[0] is not None]
        if not valid_b:
            return np.zeros(self.solution_shape, dtype=self.A[0].op.dtype)
        scenario_shape = valid_b[0][1]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        base_op, rmatvec_block, _ = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )
        d_block = np.zeros((base_op.shape[0] // num_scenarios, num_scenarios), dtype=base_op.dtype)
        current_row = 0
        for i, b_val in enumerate(b_list):
            num_a_rows = self.A[i].op.shape[0]
            if b_val is not None:
                b_col_block, b_scenario_shape = processed_b[i]
                if b_scenario_shape != scenario_shape:
                    raise ValueError("Inconsistent scenario shapes in b terms.")
                w_item = self.sqrt_weights[i]
                if w_item is not None:
                    w_op = (
                        self._densify_op(w_item)
                        if not isinstance(w_item.op, np.ndarray) or w_item.input_shape != (1,)
                        else w_item.op
                    )
                    b_col_block = (
                        w_op * b_col_block if w_item.input_shape == (1,) else w_op @ b_col_block
                    )
                d_block[current_row : current_row + num_a_rows, :] = b_col_block
            current_row += num_a_rows
        sol_block = None
        if self.solver == "svd":
            u, s_inv, vt = self._get_svd_components()
            sol_block = vt.T.conj() @ (s_inv[:, np.newaxis] * (u.T.conj() @ d_block))
        elif self.solver == "normal":
            G_T_G, G_T = self._get_normal_components()
            G_T_d = G_T @ d_block
            sol_block = np.linalg.solve(G_T_G, G_T_d)
        elif self.solver == "lsmr":
            op_to_solve, solution_transform = self._get_lsmr_components(num_scenarios)
            m, n = op_to_solve.shape
            m, n = m // num_scenarios, n // num_scenarios
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
            sol_y_block = sol_y_flat.reshape(self.solution_size, num_scenarios)
            sol_block = solution_transform(sol_y_block)
        elif self.solver == "cg":
            # The forward CG method solves (G^T G) x = G^T d.
            cg_op, M = self._get_cg_components(num_scenarios=num_scenarios)
            rhs_flat = rmatvec_block(d_block).flatten()

            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}

            sol_flat, exit_code = cg(cg_op, rhs_flat, **cg_kwargs)
            if exit_code != 0:
                print(f"Warning: CG solver did not converge (exit_code={exit_code}).")
            sol_block = sol_flat.reshape(self.solution_size, num_scenarios)

        return sol_block.reshape(self.solution_shape + scenario_shape)

    def solve_adjoint(self, y, **kwargs):
        """Solves the adjoint of the least-squares problem.

        If the forward problem is `x = S(b)`, this computes `S^H @ y`.
        Mathematically, this is `G @ (G^T @ G)^-1 @ y`, where `G` is the
        full (weighted and regularized) system matrix and `y` is a
        vector with the same shape as the solution `x`. This is useful
        for sensitivity analysis and backpropagation of gradients.

        Parameters
        ----------
        y : np.ndarray
            The input vector for the adjoint operation. This typically
            represents the gradient of a scalar loss with respect to the
            solution `x`. It must have a shape that is broadcastable to
            `self.solution_shape`. It can also include additional
            trailing "scenario" dimensions.
        **kwargs : dict
            Additional keyword arguments passed to the underlying
            iterative solver (`cg` or `lsmr`), e.g., `atol`, `rtol`,
            `maxiter`.

        Returns
        -------
        list[np.ndarray]
            A list of gradients, one for each of the `b` terms in the
            forward problem. The shape of each gradient matches the
            shape of the corresponding `b` term (including any scenario
            dimensions from `y`).
        """
        self._calculate_and_cache_scaled_lambdas()
        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=self.A[0].op.dtype)
        y_ndim, sol_ndim = y.ndim, len(self.solution_shape)

        if y_ndim < sol_ndim or y.shape[:sol_ndim] != self.solution_shape:
            if y_ndim == 1 and y.size % self.solution_size == 0:
                num_scenarios = y.size // self.solution_size
                scenario_shape = (num_scenarios,) if num_scenarios > 1 else ()
            else:
                raise ValueError(
                    f"Shape of y {y.shape} is incompatible with "
                    f"solution_shape {self.solution_shape}."
                )
        else:
            scenario_shape = y.shape[sol_ndim:]
        num_scenarios = math.prod(scenario_shape) if scenario_shape else 1
        y_block = np.ascontiguousarray(y).reshape(self.solution_size, num_scenarios)

        # Step 1: Solve (G^T G) z = y for z, for all scenarios at once
        z_block = np.zeros_like(y_block)

        if self.solver == "svd":
            _, s_inv, vt = self._get_svd_components()
            s_inv_sq = s_inv**2
            z_block = vt.T.conj() @ (s_inv_sq[:, np.newaxis] * (vt @ y_block))
        elif self.solver == "normal":
            G_T_G, _ = self._get_normal_components()
            z_block = np.linalg.solve(G_T_G, y_block)
        elif self.solver == "cg":
            normal_op, M = self._get_cg_components(num_scenarios=num_scenarios)
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            cg_kwargs = {"rtol": self.tolerance, "M": M, "maxiter": max_iter, **kwargs}
            sol_flat, exit_code = cg(normal_op, y_block.flatten(), **cg_kwargs)
            if exit_code != 0:
                print(f"Warning: Adjoint CG solver did not converge (exit_code={exit_code}).")
            z_block = sol_flat.reshape(self.solution_size, num_scenarios)
        elif self.solver == "lsmr":
            # LSMR requires manual preconditioning for the adjoint
            # system (G^T G)z = y.
            # The forward pass uses P = sqrt(diag(G^T G)). We transform
            # the system:
            # Let z = P^-1 w. Then (G^T G P^-1)w = y.
            # Left-precondition: (P^-1 G^T G P^-1)w = P^-1 y.
            # We solve this symmetric system for w,
            # then compute z = P^-1 w.
            normal_op, _ = self._get_cg_components(num_scenarios=num_scenarios)
            max_iter = ITERATION_SAFETY_FACTOR * self.solution_size
            lsmr_kwargs = {
                "atol": self.tolerance,
                "btol": self.tolerance,
                "maxiter": max_iter,
                **kwargs,
            }

            if self.preconditioner == "jacobi":
                self._setup_preconditioner_components()
                diag = self._op_cache["jacobi_diag"]
                # P_inv_diag is the diagonal of P^-1
                P_inv_diag = np.sqrt(1.0 / diag)
                P_inv_diag[np.isinf(P_inv_diag)] = 1.0
                full_P_inv_diag = np.tile(P_inv_diag, num_scenarios)

                def precond_normal_op_matvec(w_flat):
                    # Computes (P^-1 G^T G P^-1) @ w
                    temp = normal_op.matvec(w_flat * full_P_inv_diag)
                    return temp * full_P_inv_diag

                op_to_solve = LinearOperator(
                    normal_op.shape, matvec=precond_normal_op_matvec, dtype=normal_op.dtype
                )
                # Precondition the RHS: y' = P^-1 y
                y_prime_flat = y_block.flatten() * full_P_inv_diag
            else:  # No preconditioning
                op_to_solve = normal_op
                y_prime_flat = y_block.flatten()

            w_flat, istop, *_ = lsmr(op_to_solve, y_prime_flat, **lsmr_kwargs)
            if istop not in [0, 1, 2]:
                print(f"Warning: Adjoint LSMR may not have fully converged (istop={istop}).")

            # Transform back to the original variable: z = P^-1 w
            if self.preconditioner == "jacobi":
                sol_flat = w_flat * full_P_inv_diag
            else:
                sol_flat = w_flat
            z_block = sol_flat.reshape(self.solution_size, num_scenarios)

        # Step 2: Compute final gradient grad_d = G @ z
        _, _, matvec_block_fn = self._get_multi_scenario_operator(
            num_scenarios, use_scaled_lambdas=True, include_regularization=True
        )
        grad_d_block = matvec_block_fn(z_block)

        # Unpack the result into gradients for each b_i term
        grad_b_list = []
        current_row = 0
        for i in range(self.num_data_terms):
            num_a_rows = self.A[i].op.shape[0]
            grad_d_i = grad_d_block[current_row : current_row + num_a_rows, :]

            # Un-apply the weights (adjoint of the weighting operation).
            grad_b_i = grad_d_i
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                if w_item.input_shape == (1,):  # Diagonal weight
                    grad_b_i = w_item.op.conj() * grad_d_i
                else:  # Matrix weight
                    w_op = w_item.op
                    if isinstance(w_op, LinearOperator):
                        grad_b_i = (
                            w_op.rmatmat(grad_d_i)
                            if grad_d_i.shape[1] > 1
                            else w_op.rmatvec(grad_d_i[:, 0])[:, np.newaxis]
                        )
                    else:
                        grad_b_i = w_op.T.conj() @ grad_d_i

            output_shape = self.data_shapes[i] + scenario_shape
            grad_b_list.append(grad_b_i.reshape(output_shape))
            current_row += num_a_rows
        return grad_b_list

    def picard_plot(self, title=None, ax=None, **plot_kwargs):
        """Performas a Picard plot.

        Compute and plot singular values of the full system matrix.
        This method is useful for diagnosing the conditioning of the
        least-squares problem and visualizing the effect of
        regularization. It requires the problem to be densifiable
        (i.e., not using matrix-free operators, or willing to accept
        the memory cost of densification).

        Parameters
        ----------
        title : str, optional
            The title for the plot. If None, a default title is
            generated.
        ax : matplotlib.axes.Axes, optional
            An existing matplotlib axes object to plot on. If None, a
            new figure and axes are created.
        **plot_kwargs : dict
            Additional keyword arguments passed to the `ax.semilogy()`
            plotting function (e.g., `label`, `color`, `linestyle`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for this method.")
            return

        # Ensure that the full system operator G is built.
        # This uses the scaled regularization weights, which is what we
        # want to inspect.
        print("Constructing the full system matrix G...")
        G_dense = self._get_full_stacked_operator()

        # Compute the singular values of G.
        print("Computing singular values using SVD...")
        s = np.linalg.svd(G_dense, compute_uv=False)
        print("...done.")

        # Create the plot.
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # Plot the singular values.
        index = np.arange(1, len(s) + 1)
        ax.semilogy(index, s, "o-", markersize=3, **plot_kwargs)

        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value Magnitude")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Add a legend if multiple plots are on the same axes.
        if "label" in plot_kwargs:
            ax.legend()

        plt.tight_layout()
        plt.show()
