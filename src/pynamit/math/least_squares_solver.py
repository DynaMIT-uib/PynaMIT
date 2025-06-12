import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
import math
from collections import namedtuple

# An internal container for processed arrays, holding the 2D matrix and shape info.
_ProcessedArray = namedtuple('_ProcessedArray', ['matrix', 'trailing_shape', 'leading_dim_count'])

class LeastSquaresSolver:
    """
    Solves complex least-squares problems with multiple data and regularization terms.

    The problem is to find x that minimizes:
    sum_i || sqrt(W_i) * (A_i @ x - b_i) ||^2 + sum_j || lambda_j * L_j @ x ||^2
    """
    
    def __init__(self, A, solution_ndim, weights=None, regularization_weights=None, 
                 regularization_matrices=None, solver='normal', tolerance=1e-10, preconditioner=None):
        solvers = ['normal', 'lsmr', 'cg']
        if solver not in solvers: raise ValueError(f"Solver must be one of {solvers}")
        if preconditioner is not None and preconditioner != 'jacobi':
            raise ValueError("Currently, the only supported preconditioner is 'jacobi'.")
        if preconditioner is not None and solver != 'cg':
            print(f"Warning: Preconditioner is set but will be ignored. It only applies to the 'cg' solver.")

        self.solver = solver
        self.tolerance = tolerance
        self.solution_ndim = solution_ndim
        self.preconditioner = preconditioner

        A_list = LeastSquaresSolver._prepare_input_list(A, 'A', allow_single_item=True)
        self.num_data_terms = len(A_list)
        self.A = [LeastSquaresSolver._flatten(arr, num_trailing_dims=self.solution_ndim) for arr in A_list]
        
        weights_list = LeastSquaresSolver._prepare_input_list(weights, 'weights', count=self.num_data_terms)
        self.weights = [LeastSquaresSolver._flatten(w, num_trailing_dims=0) if w is not None else None for w in weights_list]
        
        reg_L_list = LeastSquaresSolver._prepare_input_list(regularization_matrices, 'regularization_matrices', allow_single_item=True, is_optional=True)
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [LeastSquaresSolver._flatten(L, num_trailing_dims=self.solution_ndim) if L is not None else None for L in reg_L_list]
        
        num_features = self.A[0].matrix.shape[1]
        for i, L_item in enumerate(self.regularization_matrices):
            if L_item is not None and L_item.matrix.shape[1] != num_features:
                raise ValueError(f"Shape mismatch in regularization_matrices term {i}: It has {L_item.matrix.shape[1]} columns, but solver expects {num_features}.")

        self.regularization_weights = LeastSquaresSolver._prepare_input_list(regularization_weights, 'regularization_weights', count=self.num_reg_terms, default_val=0.0)
        
        # Caching attributes will be populated by @property calls
        self._cached_lsmr_op = None
        self._cached_cg_op = None
        self._cached_rmatvec_func = None
        self._cached_preconditioner = None
        self._cached_base_op_rows = None

    @staticmethod
    def _prepare_input_list(input_item, name, count=None, allow_single_item=False, is_optional=False, default_val=None):
        if input_item is None:
            if is_optional: return []
            return [default_val] * count
        input_list = input_item if isinstance(input_item, list) else [input_item]
        if allow_single_item and count is None: count = len(input_list)
        if len(input_list) == 1 and count > 1: input_list = input_list * count
        if len(input_list) != count:
            raise ValueError(f"Input '{name}' has {len(input_list)} elements, but {count} were expected.")
        return input_list

    @staticmethod
    def _flatten(array, num_leading_dims=None, num_trailing_dims=None):
        if array is None: raise ValueError("Input array cannot be None.")
        if num_leading_dims is None and num_trailing_dims is None:
            split_at_dim_index = array.ndim - 1 if array.ndim > 1 else array.ndim
        elif num_leading_dims is None:
            split_at_dim_index = array.ndim - num_trailing_dims
        elif num_trailing_dims is None:
            split_at_dim_index = num_leading_dims
        else:
            if num_leading_dims + num_trailing_dims != array.ndim:
                raise ValueError(f"Dimension mismatch for array with shape {array.shape}: {num_leading_dims} + {num_trailing_dims} != {array.ndim}")
            split_at_dim_index = num_leading_dims
        if not (0 <= split_at_dim_index <= array.ndim):
            raise ValueError(f"Invalid split index {split_at_dim_index} for array with ndim {array.ndim}")
        leading_shape, trailing_shape = array.shape[:split_at_dim_index], array.shape[split_at_dim_index:]
        new_shape = (math.prod(leading_shape) if leading_shape else 1, math.prod(trailing_shape) if trailing_shape else 1)
        return _ProcessedArray(array.reshape(new_shape), trailing_shape, len(leading_shape))

    @property
    def _scaled_regularization_weights(self):
        if not hasattr(self, "_scaled_reg_weights_cache"):
            num_features = self.A[0].matrix.shape[1]
            diag_data = np.zeros(num_features, dtype=self.A[0].matrix.dtype)
            for i in range(self.num_data_terms):
                w_i = self.weights[i].matrix if self.weights[i] is not None else 1.0
                diag_data += np.sum(w_i * self.A[i].matrix**2, axis=0)
            data_scale = np.median(diag_data[diag_data > 0]) if np.any(diag_data > 0) else 1.0
            
            scaled_weights = []
            for i in range(self.num_reg_terms):
                weight = 0.0
                if self.regularization_weights[i] > 0 and self.regularization_matrices[i] is not None:
                    diag_reg = np.sum(self.regularization_matrices[i].matrix**2, axis=0)
                    reg_scale = np.median(diag_reg[diag_reg > 0]) if np.any(diag_reg > 0) else 1.0
                    if reg_scale > 1e-12: weight = self.regularization_weights[i] * data_scale / reg_scale
                scaled_weights.append(weight)
            self._scaled_reg_weights_cache = scaled_weights
        return self._scaled_reg_weights_cache

    @property
    def _full_normal_matrix(self):
        if not hasattr(self, "_full_normal_matrix_cache"):
            num_features = self.A[0].matrix.shape[1]
            dtype = self.A[0].matrix.dtype
            normal_matrix = np.zeros((num_features, num_features), dtype=dtype)
            for i in range(self.num_data_terms):
                w_i = self.weights[i].matrix if self.weights[i] else 1.0
                normal_matrix += self.A[i].matrix.T @ (w_i * self.A[i].matrix)
            for i, scaled_weight in enumerate(self._scaled_regularization_weights):
                if scaled_weight > 0 and self.regularization_matrices[i] is not None:
                    L_i = self.regularization_matrices[i].matrix
                    normal_matrix += scaled_weight * (L_i.T @ L_i)
            self._full_normal_matrix_cache = normal_matrix
        return self._full_normal_matrix_cache

    def _get_linear_operator(self, num_scenarios=1):
        num_features = self.A[0].matrix.shape[1]
        op_rows = sum(a.matrix.shape[0] for a in self.A) + \
                  sum(l.matrix.shape[0] for i, l in enumerate(self.regularization_matrices) 
                      if l is not None and self._scaled_regularization_weights[i] > 0)
        
        def matvec(x_block):
            sqrt_scaled_reg_weights = [np.sqrt(w) for w in self._scaled_regularization_weights]
            out = np.zeros((op_rows, x_block.shape[1]), dtype=self.A[0].matrix.dtype)
            row = 0
            for i in range(self.num_data_terms):
                res = self.A[i].matrix @ x_block
                if self.weights[i]: res *= np.sqrt(self.weights[i].matrix)
                out[row:row+res.shape[0]] = res
                row += res.shape[0]
            for i, L in enumerate(self.regularization_matrices):
                if L and sqrt_scaled_reg_weights[i] > 0:
                    res = sqrt_scaled_reg_weights[i] * (L.matrix @ x_block)
                    out[row:row+res.shape[0]] = res
                    row += res.shape[0]
            return out

        def rmatvec(y_block):
            sqrt_scaled_reg_weights = [np.sqrt(w) for w in self._scaled_regularization_weights]
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i in range(self.num_data_terms):
                y_part = y_block[row:row+self.A[i].matrix.shape[0]]
                if self.weights[i]: y_part = y_part * np.sqrt(self.weights[i].matrix)
                x_block += self.A[i].matrix.T @ y_part
                row += self.A[i].matrix.shape[0]
            for i, L in enumerate(self.regularization_matrices):
                if L and sqrt_scaled_reg_weights[i] > 0:
                    y_part = y_block[row:row+L.matrix.shape[0]]
                    x_block += sqrt_scaled_reg_weights[i] * (L.matrix.T @ y_part)
                    row += L.matrix.shape[0]
            return x_block.squeeze()
            
        shape = (op_rows * num_scenarios, num_features * num_scenarios)
        return LinearOperator(shape, 
                              matvec=lambda x: matvec(x.reshape(num_features, num_scenarios, order='F')).flatten('F'),
                              rmatvec=lambda y: rmatvec(y.reshape(op_rows, num_scenarios, order='F')).flatten('F'),
                              dtype=self.A[0].matrix.dtype), rmatvec

    def solve(self, b, **kwargs):
        rhs_b_list = LeastSquaresSolver._prepare_input_list(b, 'b', count=self.num_data_terms)
        processed_b_list = []
        for i, b_item in enumerate(rhs_b_list):
            if b_item is None:
                processed_b_list.append(None)
                continue
            a_info = self.A[i]
            if b_item.ndim == 1 and a_info.leading_dim_count > 1:
                if b_item.shape[0] != a_info.matrix.shape[0]:
                    raise ValueError(f"Shape mismatch for 1D b term {i}: b has {b_item.shape[0]}, A expects {a_info.matrix.shape[0]}.")
                processed_b = _ProcessedArray(b_item.reshape(-1, 1), (1,), a_info.leading_dim_count)
            else:
                processed_b = LeastSquaresSolver._flatten(b_item, num_leading_dims=a_info.leading_dim_count)
            processed_b_list.append(processed_b)

        solver_map = {'normal': self._solve_normal, 'lsmr': self._solve_lsmr, 'cg': self._solve_cg}
        return solver_map[self.solver](processed_b_list, **kwargs)

    def _solve_normal(self, processed_b_list, **kwargs):
        solutions = []
        for i, rhs_b in enumerate(processed_b_list):
            if rhs_b is None:
                solutions.append(None)
                continue
            w_i = self.weights[i].matrix if self.weights[i] is not None else 1.0
            rhs = self.A[i].matrix.T @ (w_i * rhs_b.matrix)
            sol = np.linalg.solve(self._full_normal_matrix, rhs)
            solutions.append(sol.reshape(self.A[0].trailing_shape + rhs_b.trailing_shape))
        return solutions

    def _solve_cg(self, processed_b_list, **kwargs):
        if self._cached_cg_op is None:
            num_features = self.A[0].matrix.shape[1]
            base_op, rmatvec = self._get_linear_operator(num_scenarios=1)
            self._cached_base_op_rows = base_op.shape[0]
            self._cached_cg_op = LinearOperator((num_features, num_features), matvec=lambda x: base_op.rmatvec(base_op.matvec(x)), dtype=base_op.dtype)
            self._cached_rmatvec_func = rmatvec
            if self.preconditioner == 'jacobi':
                _ = self._scaled_regularization_weights
                diag = np.zeros(num_features, dtype=base_op.dtype)
                diag_data = np.zeros_like(diag)
                for i in range(self.num_data_terms):
                    w_i = self.weights[i].matrix if self.weights[i] is not None else 1.0
                    diag_data += np.sum(w_i * self.A[i].matrix**2, axis=0)
                diag += diag_data
                for i, scaled_weight in enumerate(self._scaled_regularization_weights):
                     if scaled_weight > 0 and self.regularization_matrices[i] is not None:
                        diag += scaled_weight * np.sum(self.regularization_matrices[i].matrix**2, axis=0)
                diag[diag < 1e-12] = 1.0
                self._cached_preconditioner = LinearOperator((num_features, num_features), matvec=lambda x: x / diag, dtype=base_op.dtype)
        
        solutions = []
        for i, rhs_b in enumerate(processed_b_list):
            if rhs_b is None:
                solutions.append(None)
                continue
            num_scenarios = rhs_b.matrix.shape[1]
            d_block = np.zeros((self._cached_base_op_rows, num_scenarios), dtype=self._cached_cg_op.dtype)
            start_row = sum(self.A[k].matrix.shape[0] for k in range(i))
            weighted_b = np.sqrt(self.weights[i].matrix) * rhs_b.matrix if self.weights[i] else rhs_b.matrix
            d_block[start_row:start_row+weighted_b.shape[0], :] = weighted_b
            rhs_for_cg = self._cached_rmatvec_func(d_block)
            if rhs_for_cg.ndim == 1: rhs_for_cg = rhs_for_cg.reshape(-1, 1)
            sol = np.zeros_like(rhs_for_cg)
            # --- FIX: Use atol/rtol instead of tol ---
            cg_kwargs = {'atol': self.tolerance, 'rtol': self.tolerance, 'M': self._cached_preconditioner, **kwargs}
            for k in range(num_scenarios):
                sol[:, k], _ = cg(self._cached_cg_op, rhs_for_cg[:, k], **cg_kwargs)
            solutions.append(sol.reshape(self.A[0].trailing_shape + rhs_b.trailing_shape))
        return solutions

    def _solve_lsmr(self, processed_b_list, **kwargs):
        scenario_counts = [(b.matrix.shape[1] if b else 1) for b in processed_b_list]
        total_scenarios = sum(scenario_counts)
        if total_scenarios == 0: return [None] * len(processed_b_list)
        if self._cached_lsmr_op is None or self._cached_lsmr_op.shape[1] != self.A[0].matrix.shape[1] * total_scenarios:
            self._cached_lsmr_op, _ = self._get_linear_operator(total_scenarios)
        linear_op, dtype = self._cached_lsmr_op, self._cached_lsmr_op.dtype
        op_rows = linear_op.shape[0] // total_scenarios
        rhs = np.zeros((op_rows, total_scenarios), dtype=dtype)
        offset = 0
        for i, b in enumerate(processed_b_list):
            if b is None: continue
            weighted_b = np.sqrt(self.weights[i].matrix) * b.matrix if self.weights[i] else b.matrix
            start_row = sum(self.A[k].matrix.shape[0] for k in range(i))
            rhs[start_row:start_row+weighted_b.shape[0], offset:offset+scenario_counts[i]] = weighted_b
            offset += scenario_counts[i]
        lsmr_kwargs = {'atol': self.tolerance, 'btol': self.tolerance, **kwargs}
        sol_flat, *_ = lsmr(linear_op, rhs.flatten('F'), **lsmr_kwargs)
        solutions, offset = [], 0
        sol_block = sol_flat.reshape(self.A[0].matrix.shape[1], total_scenarios, order='F')
        for i, b in enumerate(processed_b_list):
            if b:
                sol = sol_block[:, offset:offset+scenario_counts[i]]
                solutions.append(sol.reshape(self.A[0].trailing_shape + b.trailing_shape))
                offset += scenario_counts[i]
            else:
                solutions.append(None)
        return solutions