import numpy as np
from scipy.sparse.linalg import LinearOperator, svds, cg, lsmr
import math

class FlattenedArray(object):
    def __init__(self, full_array, n_leading_flattened=None, n_trailing_flattened=None):
        if full_array is None: raise ValueError("Input array cannot be None.")
        if n_leading_flattened is None and n_trailing_flattened is None:
            n_leading_flattened = full_array.ndim - 1 if full_array.ndim > 1 else full_array.ndim
            n_trailing_flattened = 1 if full_array.ndim > 1 else 0
        elif n_leading_flattened is None: n_leading_flattened = full_array.ndim - n_trailing_flattened
        elif n_trailing_flattened is None: n_trailing_flattened = full_array.ndim - n_leading_flattened
        if n_leading_flattened + n_trailing_flattened != full_array.ndim:
            raise ValueError(f"Dimension mismatch during flattening.")
        self.full_array = full_array
        self.full_shapes = (full_array.shape[:n_leading_flattened], full_array.shape[n_leading_flattened:])
        self.shapes = (math.prod(self.full_shapes[0]) if self.full_shapes[0] else 1, math.prod(self.full_shapes[1]) if self.full_shapes[1] else 1)
        self.array = full_array.reshape(self.shapes)

class LeastSquares: # Benchmark Class
    def __init__(self, A, solution_dims, weights=None, reg_lambda=None, reg_L=None, solver='iterative', tolerance=1e-10):
        self.solver = solver
        self.tolerance = tolerance
        A_list = A if isinstance(A, list) else [A]
        self.n_data_terms = len(A_list)
        self.solution_dims = solution_dims
        self.A = self._flatten_arrays(A_list, self.n_data_terms, n_trailing_flattened=[solution_dims] * self.n_data_terms)
        self.weights = self._flatten_arrays(weights, self.n_data_terms, n_trailing_flattened=[0] * self.n_data_terms)
        reg_L_list = reg_L if isinstance(reg_L, list) else [reg_L] if reg_L is not None else []
        self.n_reg_terms = len(reg_L_list)
        n_leading_L = [l.ndim - solution_dims if l is not None else None for l in reg_L_list]
        self.reg_L = self._flatten_arrays(reg_L_list, self.n_reg_terms, n_leading_flattened=n_leading_L)
        if reg_lambda is None: self.reg_lambda = [0.0] * self.n_reg_terms
        elif not isinstance(reg_lambda, list): self.reg_lambda = [reg_lambda] * self.n_reg_terms
        else: self.reg_lambda = reg_lambda
        op = self.as_normal_eq_operator()
        self._ATWA_plus_R_cache = op @ np.eye(op.shape[0], dtype=op.dtype)
        self._ATW_cache = [((w.array.reshape(-1, 1) * a.array).T if w is not None else a.array.T) for a, w in zip(self.A, self.weights)]
    def _flatten_arrays(self, arrays, n_terms, n_leading_flattened=None, n_trailing_flattened=None):
        if arrays is None: return [None] * n_terms
        arr_list = arrays if isinstance(arrays, list) else [arrays]
        if len(arr_list) == 1 and n_terms > 1: arr_list = arr_list * n_terms
        lead_list = n_leading_flattened if n_leading_flattened and len(n_leading_flattened) == n_terms else [None] * n_terms
        trail_list = n_trailing_flattened if n_trailing_flattened and len(n_trailing_flattened) == n_terms else [None] * n_terms
        arrays_compounded = []
        for i in range(n_terms):
            arr = arr_list[i] if i < len(arr_list) else None
            if arr is not None: arrays_compounded.append(FlattenedArray(arr, n_leading_flattened=lead_list[i], n_trailing_flattened=trail_list[i]))
            else: arrays_compounded.append(None)
        return arrays_compounded
    @property
    def _scaled_regularization_params(self):
        if not hasattr(self, "_scaled_lambdas_cache"):
            n_features = self.A[0].array.shape[1]
            diag_ATWA = np.zeros(n_features)
            for i in range(self.n_data_terms):
                A_i = self.A[i].array
                if self.weights[i] is not None: diag_ATWA += np.sum(self.weights[i].array.flatten()[:, np.newaxis] * A_i**2, axis=0)
                else: diag_ATWA += np.sum(A_i**2, axis=0)
            ATWA_scale = np.median(diag_ATWA) if len(diag_ATWA) > 0 else 1.0
            lambdas = []
            for i in range(self.n_reg_terms):
                lambda_val = 0.0
                if self.reg_lambda[i] > 0 and self.reg_L[i] is not None:
                    Li = self.reg_L[i].array
                    LTLi_scale = np.median(np.diag(Li.T @ Li)) if Li.shape[0] > 0 and Li.shape[1] > 0 else 1.0
                    if LTLi_scale > 1e-12: lambda_val = self.reg_lambda[i] / LTLi_scale * ATWA_scale
                lambdas.append(lambda_val)
            self._scaled_lambdas_cache = lambdas
        return self._scaled_lambdas_cache
    def as_normal_eq_operator(self):
        if hasattr(self, "_normal_eq_op_cache"): return self._normal_eq_op_cache
        n_features = self.A[0].array.shape[1]; shape = (n_features, n_features); dtype = self.A[0].array.dtype
        scaled_lambdas = self._scaled_regularization_params
        def _matvec(x_flat):
            x_matrix = x_flat.reshape(n_features, -1, order='F'); y_matrix = np.zeros_like(x_matrix, dtype=dtype)
            for i in range(self.n_data_terms):
                Ax = self.A[i].array @ x_matrix
                if self.weights[i] is not None: Ax *= self.weights[i].array.reshape(-1, 1)
                y_matrix += self.A[i].array.T @ Ax
            for i in range(self.n_reg_terms):
                if scaled_lambdas[i] > 0 and self.reg_L[i] is not None: y_matrix += scaled_lambdas[i] * (self.reg_L[i].array.T @ (self.reg_L[i].array @ x_matrix))
            return y_matrix.flatten('F')
        self._normal_eq_op_cache = LinearOperator(shape, matvec=_matvec, rmatvec=_matvec, dtype=dtype)
        return self._normal_eq_op_cache
    def solve_normal(self, b):
        b_list = self._flatten_arrays(b, self.n_data_terms, n_leading_flattened=[len(a.full_shapes[0]) for a in self.A])
        solutions = []
        for i in range(len(b_list)):
            if b_list[i] is None: solutions.append(None); continue
            ref_b = b_list[i]; ATWb = self._ATW_cache[i] @ ref_b.array
            sol_matrix = np.linalg.solve(self._ATWA_plus_R_cache, ATWb)
            solutions.append(sol_matrix.reshape(self.A[0].full_shapes[1] + ref_b.full_shapes[1]))
        return solutions
    def solve(self, b, **kwargs):
        """Solves the system using the method specified at initialization."""
        if self.solver == 'iterative': return self.solve_lsmr_mega(b, **kwargs)
        elif self.solver == 'normal': return self.solve_normal(b)
        #elif self.solver == 'svds': return self._solve_svds(b, **kwargs)

    def as_mega_augmented_operator(self, scenario_counts):
        n_features = self.A[0].array.shape[1]
        sqrt_scaled_lambdas = [np.sqrt(l) for l in self._scaled_regularization_params]
        A_rows = [a.array.shape[0] for a in self.A]
        L_rows = [l.array.shape[0] if l is not None and sqrt_scaled_lambdas[i] > 0 else 0 for i, l in enumerate(self.reg_L)]
        base_op_rows = sum(A_rows) + sum(L_rows)
        total_input_elements = n_features * sum(scenario_counts)
        total_output_elements = base_op_rows * sum(scenario_counts)
        shape = (total_output_elements, total_input_elements); dtype = self.A[0].array.dtype
        input_slices = np.cumsum([0] + [n_features * s for s in scenario_counts])
        output_slices = np.cumsum([0] + [base_op_rows * s for s in scenario_counts])
        def _matvec(x_mega_flat):
            y_mega_parts = []
            for i, n_scen in enumerate(scenario_counts):
                if n_scen == 0: continue
                x_flat = x_mega_flat[input_slices[i]:input_slices[i+1]]
                x_matrix = x_flat.reshape(n_features, n_scen, order='F')
                y_parts = []
                for j in range(self.n_data_terms):
                    res = self.A[j].array @ x_matrix
                    if self.weights[j] is not None: res *= np.sqrt(self.weights[j].array.flatten())[:, np.newaxis]
                    y_parts.append(res)
                for j in range(self.n_reg_terms):
                    if sqrt_scaled_lambdas[j] > 0 and self.reg_L[j] is not None:
                        y_parts.append(sqrt_scaled_lambdas[j] * (self.reg_L[j].array @ x_matrix))
                y_mega_parts.append(np.concatenate(y_parts).flatten('F'))
            return np.concatenate(y_mega_parts)
        def _rmatvec(y_mega_flat):
            x_mega_parts = []
            for i, n_scen in enumerate(scenario_counts):
                if n_scen == 0: continue
                y_flat = y_mega_flat[output_slices[i]:output_slices[i+1]]
                y_matrix_full = y_flat.reshape(base_op_rows, n_scen, order='F')
                x_matrix = np.zeros((n_features, n_scen), dtype=dtype)
                pos = 0
                for j in range(self.n_data_terms):
                    y_part = y_matrix_full[pos:pos+A_rows[j], :]
                    if self.weights[j] is not None: y_part = y_part * np.sqrt(self.weights[j].array.flatten())[:, np.newaxis]
                    x_matrix += self.A[j].array.T @ y_part
                    pos += A_rows[j]
                for j in range(self.n_reg_terms):
                    if sqrt_scaled_lambdas[j] > 0 and self.reg_L[j] is not None:
                        y_part = y_matrix_full[pos:pos+L_rows[j], :]
                        x_matrix += sqrt_scaled_lambdas[j] * (self.reg_L[j].array.T @ y_part)
                        pos += L_rows[j]
                x_mega_parts.append(x_matrix.flatten('F'))
            return np.concatenate(x_mega_parts)
        return LinearOperator(shape, matvec=_matvec, rmatvec=_rmatvec, dtype=dtype)
    
    def solve_lsmr_mega(self, b, **kwargs):
        b_list = self._flatten_arrays(b, self.n_data_terms, n_leading_flattened=[len(a.full_shapes[0]) for a in self.A])
        scenario_counts = [(b_item.array.shape[1] if b_item.array.ndim > 1 else 1) if b_item is not None else 0 for b_item in b_list]
        
        # 1. Create the mega augmented operator M
        M = self.as_mega_augmented_operator(scenario_counts)
        
        # 2. Create the b_mega vector for the augmented system
        A_rows = [a.array.shape[0] for a in self.A]
        L_rows = [l.array.shape[0] if l is not None and np.sqrt(self._scaled_regularization_params[i]) > 0 else 0 for i, l in enumerate(self.reg_L)]
        base_op_rows = sum(A_rows) + sum(L_rows)
        
        b_mega_parts = []
        for i, b_item in enumerate(b_list):
            # For each problem, create its RHS vector for the base augmented system
            b_prime = np.zeros((base_op_rows, scenario_counts[i]), dtype=M.dtype)
            if b_item is not None:
                b_weighted = b_item.array
                if self.weights[i] is not None:
                    b_weighted = np.sqrt(self.weights[i].array.flatten())[:, np.newaxis] * b_weighted
                # Position of the non-zero block is based on which b_i is active
                pos = sum(A_rows[k] for k in range(i))
                b_prime[pos : pos + A_rows[i], :] = b_weighted
            b_mega_parts.append(b_prime.flatten('F'))
        b_mega = np.concatenate(b_mega_parts)
        
        # 3. Solve Mx = d directly using LSMR
        lsmr_kwargs = {'atol': self.tolerance, 'btol': self.tolerance}; lsmr_kwargs.update(kwargs)
        sol_mega, *_ = lsmr(M, b_mega, **lsmr_kwargs)
        
        # 4. Unpack solution
        solutions = []; n_features = self.A[0].array.shape[1]
        slices = np.cumsum([0] + [n_features * s for s in scenario_counts])
        for i, b_item in enumerate(b_list):
            if b_item is not None:
                sol_i_flat = sol_mega[slices[i]:slices[i+1]]
                sol_i_matrix = sol_i_flat.reshape((n_features, scenario_counts[i]), order='F')
                solutions.append(sol_i_matrix.reshape(self.A[0].full_shapes[1] + b_item.full_shapes[1]))
            else: solutions.append(None)
        return solutions


