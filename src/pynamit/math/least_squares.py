import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
import math
from collections import namedtuple

# An internal container for processed arrays, holding the 2D matrix and shape info.
_ProcessedArray = namedtuple('_ProcessedArray', ['matrix', 'trailing_shape', 'leading_dim_count'])

class LeastSquares:
    """
    Solves complex least-squares problems with multiple data and regularization terms.

    The problem is to find x that minimizes:
    sum_i || sqrt(W_i) * (A_i @ x - b_i) ||^2 + sum_j || lambda_j * L_j @ x ||^2
    """
    
    def __init__(self, A, solution_ndim, weights=None, regularization_weights=None, regularization_matrices=None, solver='normal', tolerance=1e-10):
        solvers = ['normal', 'lsmr', 'cg']
        if solver not in solvers:
            raise ValueError(f"Solver must be one of {solvers}")
        self.solver = solver
        self.tolerance = tolerance
        self.solution_ndim = solution_ndim

        A_list = self._prepare_input_list(A, 'A', allow_single_item=True)
        self.num_data_terms = len(A_list)

        self.A = [self._flatten(arr, num_trailing_dims=self.solution_ndim) for arr in A_list]
        
        weights_list = self._prepare_input_list(weights, 'weights', count=self.num_data_terms)
        self.weights = [self._flatten(w, num_trailing_dims=0) if w is not None else None for w in weights_list]
        
        reg_L_list = self._prepare_input_list(regularization_matrices, 'regularization_matrices', allow_single_item=True, is_optional=True)
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [self._flatten(L, num_trailing_dims=self.solution_ndim) if L is not None else None for L in reg_L_list]
        
        self.regularization_weights = self._prepare_input_list(regularization_weights, 'regularization_weights', count=self.num_reg_terms, default_val=0.0)
        
        # Pre-calculate matrix for the 'normal' solver to avoid re-computation.
        if self.solver == 'normal':
            self._full_normal_matrix = self._create_full_normal_matrix()

    @staticmethod
    def _prepare_input_list(input_item, name, count=None, allow_single_item=False, is_optional=False, default_val=None):
        """Prepares and validates a list of inputs (e.g., A, weights)."""
        if input_item is None:
            if is_optional: return []
            return [default_val] * count
        
        input_list = input_item if isinstance(input_item, list) else [input_item]
        
        if allow_single_item and count is None:
            count = len(input_list)

        if len(input_list) == 1 and count > 1:
            input_list = input_list * count
        
        if len(input_list) != count:
            raise ValueError(f"Input '{name}' has {len(input_list)} elements, but {count} were expected.")
        return input_list

    @staticmethod
    def _flatten(array, num_leading_dims=None, num_trailing_dims=None):
        """Reshapes a multi-dimensional array into a 2D matrix."""
        if array is None: raise ValueError("Input array cannot be None.")
        
        if num_leading_dims is None and num_trailing_dims is None:
            split_at_dim_index = array.ndim - 1 if array.ndim > 1 else array.ndim
        elif num_leading_dims is None:
            split_at_dim_index = array.ndim - num_trailing_dims
        elif num_trailing_dims is None:
            split_at_dim_index = num_leading_dims
        else:
            if num_leading_dims + num_trailing_dims != array.ndim:
                raise ValueError(f"Dimension mismatch: {num_leading_dims} + {num_trailing_dims} != {array.ndim}")
            split_at_dim_index = num_leading_dims
            
        leading_shape = array.shape[:split_at_dim_index]
        trailing_shape = array.shape[split_at_dim_index:]
        
        new_shape = (math.prod(leading_shape) if leading_shape else 1,
                     math.prod(trailing_shape) if trailing_shape else 1)
        
        return _ProcessedArray(array.reshape(new_shape), trailing_shape, len(leading_shape))

    @property
    def _scaled_regularization_weights(self):
        """Computes scaled regularization weights to balance data and reg terms."""
        if not hasattr(self, "_scaled_reg_weights_cache"):
            num_solution_features = self.A[0].matrix.shape[1]
            diag_data_normal_term = np.zeros(num_solution_features)
            for i in range(self.num_data_terms):
                A_i = self.A[i].matrix
                w_i = self.weights[i].matrix if self.weights[i] is not None else 1.0
                diag_data_normal_term += np.sum(w_i * A_i**2, axis=0)
            
            data_term_scale = np.median(diag_data_normal_term) if diag_data_normal_term.size > 0 else 1.0
            
            scaled_weights = []
            for i in range(self.num_reg_terms):
                scaled_weight = 0.0
                if self.regularization_weights[i] > 0 and self.regularization_matrices[i] is not None:
                    L_i = self.regularization_matrices[i].matrix
                    diag_reg_normal_term = np.sum(L_i**2, axis=0)
                    reg_term_scale = np.median(diag_reg_normal_term) if diag_reg_normal_term.size > 0 else 1.0
                    if reg_term_scale > 1e-12:
                        scaled_weight = self.regularization_weights[i] * data_term_scale / reg_term_scale
                scaled_weights.append(scaled_weight)
            self._scaled_reg_weights_cache = scaled_weights
        return self._scaled_reg_weights_cache

    def _create_full_normal_matrix(self):
        """Pre-computes the normal matrix A.T*W*A + L.T*L."""
        num_solution_features = self.A[0].matrix.shape[1]
        dtype = self.A[0].matrix.dtype
        
        normal_matrix_data_term = np.zeros((num_solution_features, num_solution_features), dtype=dtype)
        for i in range(self.num_data_terms):
            A_i = self.A[i].matrix
            w_i = self.weights[i].matrix if self.weights[i] else 1.0
            normal_matrix_data_term += A_i.T @ (w_i * A_i)
        
        normal_matrix_reg_term = np.zeros_like(normal_matrix_data_term)
        for i, scaled_weight in enumerate(self._scaled_regularization_weights):
            if scaled_weight > 0 and self.regularization_matrices[i] is not None:
                L_i = self.regularization_matrices[i].matrix
                normal_matrix_reg_term += scaled_weight * (L_i.T @ L_i)
        
        return normal_matrix_data_term + normal_matrix_reg_term

    def _get_linear_operator(self, num_scenarios=1):
        """Creates a LinearOperator M for the system M*x = d."""
        num_solution_features = self.A[0].matrix.shape[1]
        sqrt_scaled_reg_weights = [np.sqrt(w) for w in self._scaled_regularization_weights]

        def matvec_block(x_block):
            parts = []
            for i in range(self.num_data_terms):
                res = self.A[i].matrix @ x_block
                if self.weights[i] is not None: res *= np.sqrt(self.weights[i].matrix)
                parts.append(res)
            for i in range(self.num_reg_terms):
                if sqrt_scaled_reg_weights[i] > 0 and self.regularization_matrices[i] is not None:
                    res = sqrt_scaled_reg_weights[i] * (self.regularization_matrices[i].matrix @ x_block)
                    parts.append(res)
            return np.concatenate(parts, axis=0)
        
        def rmatvec_block(y_block):
            current_scenarios = y_block.shape[1] if y_block.ndim > 1 else 1
            x_block = np.zeros((num_solution_features, current_scenarios), dtype=y_block.dtype)
            current_row = 0
            for i in range(self.num_data_terms):
                n_rows = self.A[i].matrix.shape[0]
                y_part = y_block[current_row : current_row + n_rows]
                current_row += n_rows
                if self.weights[i] is not None:
                    # THE FIX: Avoid in-place modification which corrupts solver's internal state
                    y_part = y_part * np.sqrt(self.weights[i].matrix)
                x_block += self.A[i].matrix.T @ y_part
            for i in range(self.num_reg_terms):
                if sqrt_scaled_reg_weights[i] > 0 and self.regularization_matrices[i] is not None:
                    n_rows = self.regularization_matrices[i].matrix.shape[0]
                    y_part = y_block[current_row : current_row + n_rows]
                    current_row += n_rows
                    x_block += sqrt_scaled_reg_weights[i] * (self.regularization_matrices[i].matrix.T @ y_part)
            return x_block.squeeze()

        op_rows_A = sum(a.matrix.shape[0] for a in self.A)
        op_rows_L = sum(l.matrix.shape[0] for i, l in enumerate(self.regularization_matrices) 
                        if l is not None and self._scaled_regularization_weights[i] > 0)
        operator_rows = op_rows_A + op_rows_L
        shape = (operator_rows * num_scenarios, num_solution_features * num_scenarios)
        
        def _matvec_flat(x_flat):
            x_block = x_flat.reshape(num_solution_features, num_scenarios, order='F')
            return matvec_block(x_block).flatten('F')
        def _rmatvec_flat(y_flat):
            y_block = y_flat.reshape(operator_rows, num_scenarios, order='F')
            return rmatvec_block(y_block).flatten('F')
        
        linear_op = LinearOperator(shape, matvec=_matvec_flat, rmatvec=_rmatvec_flat, dtype=self.A[0].matrix.dtype)
        return linear_op, rmatvec_block

    def solve(self, b, **kwargs):
        """Solves the least-squares problem for the given right-hand side(s) b."""
        rhs_b_list = self._prepare_input_list(b, 'b', count=self.num_data_terms)
        processed_b_list = [
            self._flatten(b, num_leading_dims=self.A[i].leading_dim_count) if b is not None else None
            for i, b in enumerate(rhs_b_list)
        ]
        
        if self.solver == 'lsmr':
            return self._solve_lsmr_vectorized(processed_b_list, **kwargs)

        # For 'normal' and 'cg', we can loop through each b term
        solutions = []
        for i, rhs_b in enumerate(processed_b_list):
            if rhs_b is None:
                solutions.append(None)
                continue
            
            num_scenarios = rhs_b.matrix.shape[1] if rhs_b.matrix.ndim > 1 else 1
            num_solution_features = self.A[0].matrix.shape[1]
            solution_matrix = None
            
            if self.solver == 'normal':
                w_i = self.weights[i].matrix if self.weights[i] else 1.0
                rhs_for_normal_eq = self.A[i].matrix.T @ (w_i * rhs_b.matrix)
                solution_matrix = np.linalg.solve(self._full_normal_matrix, rhs_for_normal_eq)
            
            elif self.solver == 'cg':
                base_linear_op, rmatvec_block_func = self._get_linear_operator(num_scenarios=1)
                normal_op_matvec = lambda x: base_linear_op.rmatvec(base_linear_op.matvec(x))
                normal_op = LinearOperator(
                    (num_solution_features, num_solution_features),
                    matvec=normal_op_matvec,
                    dtype=base_linear_op.dtype
                )
                
                operator_rows = base_linear_op.shape[0]
                rhs_d_block = np.zeros((operator_rows, num_scenarios), dtype=base_linear_op.dtype)
                start_row = sum(self.A[k].matrix.shape[0] for k in range(i))
                end_row = start_row + self.A[i].matrix.shape[0]
                weighted_b = np.sqrt(self.weights[i].matrix) * rhs_b.matrix if self.weights[i] else rhs_b.matrix
                rhs_d_block[start_row:end_row, :] = weighted_b

                rhs_for_cg = rmatvec_block_func(rhs_d_block)
                if rhs_for_cg.ndim == 1:
                    rhs_for_cg = rhs_for_cg.reshape(-1, 1)

                solution_matrix = np.zeros((num_solution_features, num_scenarios), dtype=base_linear_op.dtype)
                cg_kwargs = {'atol': self.tolerance, 'rtol': self.tolerance, **kwargs}
                for k in range(num_scenarios):
                    solution_matrix[:, k], info = cg(normal_op, rhs_for_cg[:, k], **cg_kwargs)
                    if info != 0: print(f"Warning: CG did not converge for scenario {k}. Info: {info}")
            
            output_shape = self.A[0].trailing_shape + rhs_b.trailing_shape
            solutions.append(solution_matrix.reshape(output_shape))
            
        return solutions
    
    def _solve_lsmr_vectorized(self, processed_b_list, **kwargs):
        """Solves the full system for all b-terms at once using lsmr."""
        scenario_counts = [(b.matrix.shape[1] if b.matrix.ndim > 1 else 1) if b else 0 for b in processed_b_list]
        total_scenarios = sum(scenario_counts)
        if total_scenarios == 0:
            return [None] * len(processed_b_list)

        linear_op, _ = self._get_linear_operator(total_scenarios)
        operator_rows, dtype = linear_op.shape[0] // total_scenarios, linear_op.dtype

        # Build the single, large right-hand-side vector
        rhs_d_block = np.zeros((operator_rows, total_scenarios), dtype=dtype)
        scen_offset = 0
        for i, b_item in enumerate(processed_b_list):
            if b_item is None:
                continue
            
            num_scen_i = scenario_counts[i]
            weighted_b = np.sqrt(self.weights[i].matrix) * b_item.matrix if self.weights[i] else b_item.matrix
            
            start_row = sum(self.A[k].matrix.shape[0] for k in range(i))
            end_row = start_row + self.A[i].matrix.shape[0]
            rhs_d_block[start_row:end_row, scen_offset : scen_offset + num_scen_i] = weighted_b
            scen_offset += num_scen_i
        
        # Solve the single large system
        lsmr_kwargs = {'atol': self.tolerance, 'btol': self.tolerance, **kwargs}
        sol_flat, *_ = lsmr(linear_op, rhs_d_block.flatten('F'), **lsmr_kwargs)
        
        # Deconstruct the solution vector
        solutions = []
        num_solution_features = self.A[0].matrix.shape[1]
        solution_block = sol_flat.reshape(num_solution_features, total_scenarios, order='F')
        scen_offset = 0
        for i, b_item in enumerate(processed_b_list):
            if b_item is not None:
                num_scen_i = scenario_counts[i]
                solution_i = solution_block[:, scen_offset : scen_offset + num_scen_i]
                scen_offset += num_scen_i
                
                output_shape = self.A[0].trailing_shape + b_item.trailing_shape
                solutions.append(solution_i.reshape(output_shape))
            else:
                solutions.append(None)
        return solutions