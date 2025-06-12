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
        
        self.regularization_weights = LeastSquaresSolver._prepare_input_list(regularization_weights, 'regularization_weights', count=self.num_reg_terms, default_val=0.0)
        
        self._cached_lsmr_op = None
        self._cached_cg_op = None
        self._cached_rmatvec_func = None
        self._cached_preconditioner = None
        self._cached_base_op_rows = None
        self._jacobi_precond_diag = None 

        if self.solver == 'normal':
            self._full_normal_matrix = self._create_full_normal_matrix()

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
                raise ValueError(f"Dimension mismatch for array with shape {array.shape}: "
                                 f"{num_leading_dims} + {num_trailing_dims} != {array.ndim}")
            split_at_dim_index = num_leading_dims
        
        if not (0 <= split_at_dim_index <= array.ndim):
            raise ValueError(f"Invalid split index {split_at_dim_index} for array with ndim {array.ndim}")

        leading_shape = array.shape[:split_at_dim_index]
        trailing_shape = array.shape[split_at_dim_index:]
        
        n_rows = math.prod(leading_shape) if leading_shape else 1
        n_cols = math.prod(trailing_shape) if trailing_shape else 1
        
        new_shape = (n_rows, n_cols)
        return _ProcessedArray(array.reshape(new_shape), trailing_shape, len(leading_shape))

    @property
    def _scaled_regularization_weights(self):
        if hasattr(self, "_scaled_reg_weights_cache"):
            return self._scaled_reg_weights_cache

        num_solution_features = self.A[0].matrix.shape[1]
        dtype = self.A[0].matrix.dtype
        diag_data_normal_term = np.zeros(num_solution_features, dtype=dtype)
        for i in range(self.num_data_terms):
            A_i = self.A[i].matrix
            w_i = self.weights[i].matrix if self.weights[i] is not None else 1.0
            diag_data_normal_term += np.sum(w_i * A_i**2, axis=0)
        
        full_normal_diag = diag_data_normal_term.copy()
        data_term_scale = np.median(diag_data_normal_term[diag_data_normal_term > 0]) if np.any(diag_data_normal_term > 0) else 1.0
        
        scaled_weights = []
        for i in range(self.num_reg_terms):
            scaled_weight = 0.0
            if self.regularization_weights[i] > 0 and self.regularization_matrices[i] is not None:
                L_i = self.regularization_matrices[i].matrix
                diag_reg_normal_term = np.sum(L_i**2, axis=0)
                reg_term_scale = np.median(diag_reg_normal_term[diag_reg_normal_term > 0]) if np.any(diag_reg_normal_term > 0) else 1.0
                if reg_term_scale > 1e-12:
                    scaled_weight = self.regularization_weights[i] * data_term_scale / reg_term_scale
                    full_normal_diag += scaled_weight * diag_reg_normal_term
            scaled_weights.append(scaled_weight)
        
        self._scaled_reg_weights_cache = scaled_weights
        self._jacobi_precond_diag = full_normal_diag
        return self._scaled_reg_weights_cache

    def _create_full_normal_matrix(self):
        _ = self._scaled_regularization_weights
        num_solution_features = self.A[0].matrix.shape[1]
        dtype = self.A[0].matrix.dtype
        normal_matrix = np.zeros((num_solution_features, num_solution_features), dtype=dtype)
        for i in range(self.num_data_terms):
            A_i, w_i = self.A[i].matrix, self.weights[i]
            normal_matrix += A_i.T @ ((w_i.matrix if w_i else 1.0) * A_i)
        for i, scaled_weight in enumerate(self._scaled_regularization_weights):
            if scaled_weight > 0 and self.regularization_matrices[i] is not None:
                L_i = self.regularization_matrices[i].matrix
                normal_matrix += scaled_weight * (L_i.T @ L_i)
        return normal_matrix

    def _get_linear_operator(self, num_scenarios=1):
        num_solution_features = self.A[0].matrix.shape[1]
        dtype = self.A[0].matrix.dtype
        sqrt_scaled_reg_weights = [np.sqrt(w) for w in self._scaled_regularization_weights]
        op_rows_A = sum(a.matrix.shape[0] for a in self.A)
        op_rows_L = sum(l.matrix.shape[0] for i, l in enumerate(self.regularization_matrices) 
                        if l is not None and sqrt_scaled_reg_weights[i] > 0)
        operator_rows = op_rows_A + op_rows_L
        def matvec_block(x_block):
            out = np.zeros((operator_rows, x_block.shape[1]), dtype=dtype)
            current_row = 0
            for i in range(self.num_data_terms):
                n_rows = self.A[i].matrix.shape[0]
                res = self.A[i].matrix @ x_block
                if self.weights[i] is not None: res *= np.sqrt(self.weights[i].matrix)
                out[current_row : current_row + n_rows] = res
                current_row += n_rows
            for i, L_item in enumerate(self.regularization_matrices):
                if sqrt_scaled_reg_weights[i] > 0 and L_item is not None:
                    n_rows = L_item.matrix.shape[0]
                    res = sqrt_scaled_reg_weights[i] * (L_item.matrix @ x_block)
                    out[current_row : current_row + n_rows] = res
                    current_row += n_rows
            return out
        def rmatvec_block(y_block):
            x_block = np.zeros((num_solution_features, y_block.shape[1]), dtype=y_block.dtype)
            current_row = 0
            for i in range(self.num_data_terms):
                n_rows = self.A[i].matrix.shape[0]
                y_part = y_block[current_row : current_row + n_rows]
                current_row += n_rows
                if self.weights[i] is not None: y_part = y_part * np.sqrt(self.weights[i].matrix)
                x_block += self.A[i].matrix.T @ y_part
            for i, L_item in enumerate(self.regularization_matrices):
                if sqrt_scaled_reg_weights[i] > 0 and L_item is not None:
                    n_rows = L_item.matrix.shape[0]
                    y_part = y_block[current_row : current_row + n_rows]
                    current_row += n_rows
                    x_block += sqrt_scaled_reg_weights[i] * (L_item.matrix.T @ y_part)
            return x_block.squeeze()
        shape = (operator_rows * num_scenarios, num_solution_features * num_scenarios)
        def _matvec_flat(x_flat):
            x_block = x_flat.reshape(num_solution_features, num_scenarios, order='F')
            return matvec_block(x_block).flatten('F')
        def _rmatvec_flat(y_flat):
            y_block = y_flat.reshape(operator_rows, num_scenarios, order='F')
            return rmatvec_block(y_block).flatten('F')
        return LinearOperator(shape, matvec=_matvec_flat, rmatvec=_rmatvec_flat, dtype=dtype), rmatvec_block

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
                    raise ValueError(f"Shape mismatch for 1D b term {i}: b has {b_item.shape[0]} elements, "
                                     f"but corresponding A term expects {a_info.matrix.shape[0]}.")
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
            rhs_for_normal_eq = self.A[i].matrix.T @ (w_i * rhs_b.matrix)
            solution_matrix = np.linalg.solve(self._full_normal_matrix, rhs_for_normal_eq)
            output_shape = self.A[0].trailing_shape + rhs_b.trailing_shape
            solutions.append(solution_matrix.reshape(output_shape))
        return solutions

    def _solve_cg(self, processed_b_list, **kwargs):
        if self._cached_cg_op is None:
            num_solution_features = self.A[0].matrix.shape[1]
            base_op, rmatvec_func = self._get_linear_operator(num_scenarios=1)
            self._cached_base_op_rows = base_op.shape[0]
            
            normal_op_matvec = lambda x: base_op.rmatvec(base_op.matvec(x))
            self._cached_cg_op = LinearOperator((num_solution_features, num_solution_features),
                                                matvec=normal_op_matvec, dtype=base_op.dtype)
            self._cached_rmatvec_func = rmatvec_func
            if self.preconditioner == 'jacobi':
                _ = self._scaled_regularization_weights 
                diag = self._jacobi_precond_diag
                diag[diag < 1e-12] = 1.0
                self._cached_preconditioner = LinearOperator(
                    (num_solution_features, num_solution_features),
                    matvec=lambda x: x / diag,
                    dtype=base_op.dtype
                )
        
        solutions = []
        cg_kwargs = {'atol': self.tolerance, 'rtol': self.tolerance, 'M': self._cached_preconditioner, **kwargs}
        operator_rows = self._cached_base_op_rows
        
        for i, rhs_b in enumerate(processed_b_list):
            if rhs_b is None:
                solutions.append(None)
                continue
            num_scenarios = rhs_b.matrix.shape[1] if rhs_b.matrix.ndim > 1 else 1
            rhs_d_block = np.zeros((operator_rows, num_scenarios), dtype=self._cached_cg_op.dtype)
            start_row = sum(self.A[k].matrix.shape[0] for k in range(i))
            end_row = start_row + self.A[i].matrix.shape[0]
            weighted_b = np.sqrt(self.weights[i].matrix) * rhs_b.matrix if self.weights[i] else rhs_b.matrix
            rhs_d_block[start_row:end_row, :] = weighted_b
            rhs_for_cg = self._cached_rmatvec_func(rhs_d_block)
            if rhs_for_cg.ndim == 1: rhs_for_cg = rhs_for_cg.reshape(-1, 1)
            
            solution_matrix = np.zeros_like(rhs_for_cg)
            for k in range(num_scenarios):
                sol, info = cg(self._cached_cg_op, rhs_for_cg[:, k], **cg_kwargs)
                if info != 0: print(f"Warning: CG did not converge for scenario {k}, problem {i}. Info: {info}")
                solution_matrix[:, k] = sol
            output_shape = self.A[0].trailing_shape + rhs_b.trailing_shape
            solutions.append(solution_matrix.reshape(output_shape))
        return solutions
    
    def _solve_lsmr(self, processed_b_list, **kwargs):
        scenario_counts = [(b.matrix.shape[1] if b.matrix.ndim > 1 else 1) if b else 0 for b in processed_b_list]
        total_scenarios = sum(scenario_counts)
        if total_scenarios == 0: return [None] * len(processed_b_list)

        if self._cached_lsmr_op is None or self._cached_lsmr_op.shape[1] != self.A[0].matrix.shape[1] * total_scenarios:
            self._cached_lsmr_op, _ = self._get_linear_operator(total_scenarios)
        
        linear_op = self._cached_lsmr_op
        operator_rows, dtype = linear_op.shape[0] // total_scenarios, linear_op.dtype
        rhs_d_block = np.zeros((operator_rows, total_scenarios), dtype=dtype)
        scen_offset = 0
        for i, b_item in enumerate(processed_b_list):
            if b_item is None: continue
            num_scen_i = scenario_counts[i]
            weighted_b = np.sqrt(self.weights[i].matrix) * b_item.matrix if self.weights[i] else b_item.matrix
            start_row = sum(self.A[k].matrix.shape[0] for k in range(i))
            end_row = start_row + self.A[i].matrix.shape[0]
            rhs_d_block[start_row:end_row, scen_offset : scen_offset + num_scen_i] = weighted_b
            scen_offset += num_scen_i
        
        lsmr_kwargs = {'atol': self.tolerance, 'btol': self.tolerance, **kwargs}
        sol_flat, *_ = lsmr(linear_op, rhs_d_block.flatten('F'), **lsmr_kwargs)
        
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