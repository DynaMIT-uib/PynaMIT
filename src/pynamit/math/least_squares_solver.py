"""Least-squares solver module.

This module contains the LeastSquaresSolver class for solving complex,
multi-term least-squares problems.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
import math
from collections import namedtuple

_ProcessedItem = namedtuple("_ProcessedItem", ["op", "trailing_shape", "leading_dim_count"])

class LeastSquaresSolver:
    """
    Solves complex least-squares problems with multiple data and regularization terms.
    This class handles both dense numpy arrays and scipy LinearOperators as inputs.
    """

    def __init__(self, A, solution_ndim, weights=None, regularization_weights=None, regularization_matrices=None,
                 solver="svd", tolerance=1e-12, preconditioner=None):
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers: raise ValueError(f"Solver must be one of {solvers}")
        if preconditioner is not None and preconditioner != "jacobi": raise ValueError("Only 'jacobi' preconditioner supported.")
        if preconditioner is not None and solver != "cg": print("Warning: Preconditioner is set but only applies to 'cg' solver.")

        self.solver = solver
        self.tolerance = tolerance
        self.solution_ndim = solution_ndim
        self.preconditioner = preconditioner

        A_list = self._prepare_input_list(A, "A", allow_single_item=True)
        self.num_data_terms = len(A_list)
        self.A = [self._flatten(arr, num_trailing_dims=self.solution_ndim) for arr in A_list]
        self.is_matrix_free = any(isinstance(a.op, LinearOperator) for a in self.A)

        weights_list = self._prepare_input_list(weights, "weights", count=self.num_data_terms)
        self.weights = [self._flatten(w, num_trailing_dims=0) if w is not None else None for w in weights_list]
        
        self.sqrt_weights = []
        for w_item in self.weights:
            if w_item is not None:
                sqrt_op = np.sqrt(w_item.op)
                self.sqrt_weights.append(
                    _ProcessedItem(op=sqrt_op, trailing_shape=w_item.trailing_shape, leading_dim_count=w_item.leading_dim_count)
                )
            else: self.sqrt_weights.append(None)

        reg_L_list = self._prepare_input_list(regularization_matrices, "regularization_matrices", allow_single_item=True, is_optional=True)
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [self._flatten(L, num_trailing_dims=self.solution_ndim) if L is not None else None for L in reg_L_list]
        
        if any(L is not None and isinstance(L.op, LinearOperator) for L in self.regularization_matrices): self.is_matrix_free = True
        if self.is_matrix_free and solver in ["normal", "svd"]: print(f"Warning: Solver '{solver}' with matrix-free operators will be slow.")
        
        num_features = self.A[0].op.shape[1]
        for i, L_item in enumerate(self.regularization_matrices):
            if L_item and L_item.op.shape[1] != num_features: raise ValueError(f"Shape mismatch in regularization term {i}.")

        self.regularization_weights = self._prepare_input_list(regularization_weights, "regularization_weights", count=self.num_reg_terms, default_val=0.0)
        self._op_cache, self._cg_op_cache, self._svd_cache = {}, {}, {}

    @staticmethod
    def _prepare_input_list(item, name, count=None, allow_single_item=False, is_optional=False, default_val=None):
        if item is None:
            if is_optional: return []
            return [default_val] * count
        lst = item if isinstance(item, list) else [item]
        if allow_single_item and count is None: count = len(lst)
        if len(lst) == 1 and count > 1: lst *= count
        if len(lst) != count: raise ValueError(f"Input '{name}' has {len(lst)} items, expected {count}.")
        return lst

    @staticmethod
    def _flatten(array, num_leading_dims=None, num_trailing_dims=None):
        if isinstance(array, LinearOperator):
            if num_trailing_dims != 1: raise ValueError("LinearOperator can only be used with solution_ndim=1")
            return _ProcessedItem(array, (array.shape[1],), 1)
        if array is None: raise ValueError("Input array to _flatten cannot be None.")
        if array.ndim == 1: return _ProcessedItem(array.reshape(-1, 1), (1,), 1)
        if num_leading_dims is None and num_trailing_dims is None: split_idx = array.ndim - 1 if array.ndim > 1 else array.ndim
        elif num_leading_dims is None: split_idx = array.ndim - num_trailing_dims
        elif num_trailing_dims is None: split_idx = num_leading_dims
        else:
            if num_leading_dims + num_trailing_dims != array.ndim: raise ValueError(f"Dim mismatch for shape {array.shape}")
            split_idx = num_leading_dims
        if not (0 <= split_idx <= array.ndim): raise ValueError(f"Invalid split index {split_idx} for array with ndim {array.ndim}")
        leading_shape, trailing_shape = array.shape[:split_idx], array.shape[split_idx:]
        new_shape = (math.prod(leading_shape) if leading_shape else 1, math.prod(trailing_shape) if trailing_shape else 1)
        return _ProcessedItem(array.reshape(new_shape), trailing_shape, len(leading_shape))

    def _densify_op(self, item):
        if item is None: return None
        op = item.op
        if isinstance(op, LinearOperator): return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op

    @property
    def _scaled_regularization_weights(self):
        if hasattr(self, "_scaled_reg_weights_cache"): return self._scaled_reg_weights_cache
        if self.is_matrix_free:
            print("Warning: Cannot auto-scale regularization for matrix-free operators. Using raw weights.")
            self._scaled_reg_weights_cache = self.regularization_weights
            self._jacobi_precond_diag = None
            return self.regularization_weights
        
        num_features = self.A[0].op.shape[1]
        diag_data = np.zeros(num_features, dtype=self.A[0].op.dtype)
        for i in range(self.num_data_terms):
            w_i = self._densify_op(self.weights[i]) if self.weights[i] is not None else 1.0
            A_i = self._densify_op(self.A[i])
            diag_data += np.sum(w_i * (A_i ** 2), axis=0)

        full_normal_diag = diag_data.copy()
        data_scale = np.median(diag_data[diag_data > 0]) if np.any(diag_data > 0) else 1.0
        scaled_weights = []
        for i in range(self.num_reg_terms):
            weight = 0.0
            if self.regularization_weights[i] > 0 and self.regularization_matrices[i] is not None:
                L_i = self._densify_op(self.regularization_matrices[i])
                diag_reg = np.sum(L_i ** 2, axis=0)
                reg_scale = np.median(diag_reg[diag_reg > 0]) if np.any(diag_reg > 0) else 1.0
                if reg_scale > 1e-12:
                    weight = self.regularization_weights[i] * data_scale / reg_scale
                    full_normal_diag += weight * diag_reg
            scaled_weights.append(weight)
        self._scaled_reg_weights_cache = scaled_weights
        self._jacobi_precond_diag = full_normal_diag
        return self._scaled_reg_weights_cache

    def _get_linear_operator(self, num_scenarios=1):
        """
        Creates the master LinearOperator for the least-squares problem.
        The internal matvec/rmatvec are corrected to handle LinearOperator
        components by iterating over the scenarios (columns) of the input block.
        """
        cache_key = num_scenarios
        if cache_key in self._op_cache: return self._op_cache[cache_key]

        num_features = self.A[0].op.shape[1]
        sqrt_scaled_reg_weights = [np.sqrt(w) for w in self._scaled_regularization_weights]

        op_rows = sum(a.op.shape[0] for a in self.A) + \
                  sum(l.op.shape[0] for i, l in enumerate(self.regularization_matrices) if l and sqrt_scaled_reg_weights[i] > 0)

        def _apply_op_to_block(item, x_block):
            num_scenarios_in_block = x_block.shape[1]
            if isinstance(item.op, LinearOperator):
                # CRITICAL FIX: Iterate for LinearOperator components
                res_block = np.zeros((item.op.shape[0], num_scenarios_in_block), dtype=x_block.dtype)
                for k in range(num_scenarios_in_block):
                    res_block[:, k] = item.op.matvec(x_block[:, k])
                return res_block
            else:
                # Efficient block product for dense arrays
                return item.op @ x_block
        
        def _apply_op_T_to_block(item, y_block):
            num_scenarios_in_block = y_block.shape[1]
            if isinstance(item.op, LinearOperator):
                # CRITICAL FIX: Iterate for LinearOperator components
                res_block = np.zeros((item.op.shape[1], num_scenarios_in_block), dtype=y_block.dtype)
                for k in range(num_scenarios_in_block):
                    res_block[:, k] = item.op.rmatvec(y_block[:, k])
                return res_block
            else:
                # Efficient block product for dense arrays
                return item.op.T.conj() @ y_block

        # This is the block-based matvec. It operates on 2D arrays (features, scenarios).
        def matvec_block(x_block):
            out = np.zeros((op_rows, x_block.shape[1]), dtype=self.A[0].op.dtype)
            row = 0
            for i in range(self.num_data_terms):
                res_block = _apply_op_to_block(self.A[i], x_block)
                if self.sqrt_weights[i] is not None:
                    res_block *= self._densify_op(self.sqrt_weights[i])
                out[row : row + res_block.shape[0]] = res_block
                row += res_block.shape[0]
            for i, L in enumerate(self.regularization_matrices):
                if L and sqrt_scaled_reg_weights[i] > 0:
                    weight = sqrt_scaled_reg_weights[i]
                    res_block = _apply_op_to_block(L, x_block)
                    out[row : row + res_block.shape[0]] = weight * res_block
                    row += res_block.shape[0]
            return out

        # This is the block-based rmatvec.
        def rmatvec_block(y_block):
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i in range(self.num_data_terms):
                item = self.A[i]
                y_part = y_block[row : row + item.op.shape[0]]
                if self.sqrt_weights[i] is not None:
                    y_part = y_part * self._densify_op(self.sqrt_weights[i])
                x_block += _apply_op_T_to_block(item, y_part)
                row += item.op.shape[0]
            for i, L in enumerate(self.regularization_matrices):
                if L and sqrt_scaled_reg_weights[i] > 0:
                    weight = sqrt_scaled_reg_weights[i]
                    y_part = y_block[row : row + L.op.shape[0]]
                    x_block += weight * _apply_op_T_to_block(L, y_part)
                    row += L.op.shape[0]
            return x_block.squeeze()

        # THE BRIDGE: These lambda functions connect lsmr's 1D world to our block-based 2D world.
        # They flatten/reshape using Fortran ordering ('F') which is crucial for column-major logic.
        shape = (op_rows * num_scenarios, num_features * num_scenarios)
        op = LinearOperator(
            shape,
            matvec=lambda x: matvec_block(x.reshape(num_features, -1, order="F")).flatten("F"),
            rmatvec=lambda y: rmatvec_block(y.reshape(op_rows, -1, order="F")).flatten("F"),
            dtype=self.A[0].op.dtype
        )
        self._op_cache[cache_key] = (op, rmatvec_block)
        return op, rmatvec_block

    def solve(self, b, **kwargs):
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        proc_b = [self._flatten(item, num_leading_dims=self.A[i].leading_dim_count) if item is not None else None for i, item in enumerate(b_list)]
        solver_map = {"normal": self._solve_normal, "lsmr": self._solve_lsmr, "cg": self._solve_cg, "svd": self._solve_svd}
        return solver_map[self.solver](proc_b, **kwargs)

    def _solve_lsmr(self, proc_b, **kwargs):
        scenarios = tuple((b.op.shape[1] if b is not None else 0) for b in proc_b)
        total_scenarios = sum(scenarios)
        if total_scenarios == 0: return [None] * len(proc_b)
        
        linear_op, _ = self._get_linear_operator(total_scenarios)
        op_rows, dtype = linear_op.shape[0] // total_scenarios, linear_op.dtype
        
        # Build the combined RHS vector
        rhs_block = np.zeros((op_rows, total_scenarios), dtype=dtype)
        offset = 0
        for i, b in enumerate(proc_b):
            if b is None: continue
            num_scen_i = scenarios[i]
            weighted_b = self._densify_op(self.sqrt_weights[i]) * b.op if self.sqrt_weights[i] is not None else b.op
            start_row = sum(a.op.shape[0] for a in self.A[:i])
            rhs_block[start_row : start_row + weighted_b.shape[0], offset : offset + num_scen_i] = weighted_b
            offset += num_scen_i
        
        lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance, **kwargs}
        sol_flat, *_ = lsmr(linear_op, rhs_block.flatten("F"), **lsmr_kwargs)
        
        # Unpack the solution
        solutions, offset = [], 0
        sol_block = sol_flat.reshape(self.A[0].op.shape[1], total_scenarios, order="F")
        for i, b in enumerate(proc_b):
            if b is not None:
                num_scen_i = scenarios[i]
                solutions.append(sol_block[:, offset : offset + num_scen_i].reshape(self.A[0].trailing_shape + b.trailing_shape))
                offset += num_scen_i
            else: solutions.append(None)
        return solutions

    # --- Other solver methods are largely unchanged but benefit from the robust backend ---
    @property
    def _full_normal_matrix(self):
        if not hasattr(self, "_normal_matrix_cache"):
            num_features = self.A[0].op.shape[1]
            normal_matrix = np.zeros((num_features, num_features), dtype=self.A[0].op.dtype)
            for i in range(self.num_data_terms):
                A_i = self._densify_op(self.A[i])
                w_i = self._densify_op(self.weights[i]) if self.weights[i] is not None else 1.0
                normal_matrix += A_i.T @ (w_i * A_i)
            for i, weight in enumerate(self._scaled_regularization_weights):
                if weight > 0 and self.regularization_matrices[i] is not None:
                    L_i = self._densify_op(self.regularization_matrices[i])
                    normal_matrix += weight * (L_i.T @ L_i)
            self._normal_matrix_cache = normal_matrix
        return self._normal_matrix_cache

    def _solve_normal(self, proc_b, **kwargs):
        # Normal equation solver is typically for a single scenario
        total_rhs_list = []
        for rhs in proc_b:
            if rhs is None:
                total_rhs_list.append(None)
                continue
            
            total_rhs = np.zeros((self.A[0].op.shape[1], rhs.op.shape[1]))
            for i, A_item in enumerate(self.A):
                if i < len(proc_b) and proc_b[i] is not None: # only add if b exists for this A
                    w_i = self._densify_op(self.weights[i]) if self.weights[i] is not None else 1.0
                    A_i_T = self._densify_op(A_item).T
                    total_rhs += A_i_T @ (w_i * self._densify_op(proc_b[i]))

            sol = np.linalg.solve(self._full_normal_matrix, total_rhs)
            total_rhs_list.append(sol.reshape(self.A[0].trailing_shape + rhs.trailing_shape))

        return total_rhs_list
    
    def _solve_cg(self, proc_b, **kwargs):
        if 'op' not in self._cg_op_cache:
            num_features = self.A[0].op.shape[1]
            base_op, rmatvec_func = self._get_linear_operator(num_scenarios=1)
            self._cg_op_cache['op'] = LinearOperator((num_features, num_features), matvec=lambda x: base_op.rmatvec(base_op.matvec(x)), dtype=base_op.dtype)
            self._cg_op_cache['rmatvec_block'] = rmatvec_func
            self._cg_op_cache['base_op_rows'] = base_op.shape[0] // 1 # num_scenarios=1
            if self.preconditioner == "jacobi":
                if self.is_matrix_free:
                    print("Warning: Cannot compute Jacobi preconditioner for matrix-free operators. Preconditioner disabled.")
                    self._cg_op_cache['M'] = None
                else:
                    _ = self._scaled_regularization_weights
                    diag = self._jacobi_precond_diag
                    diag[diag < 1e-12] = 1.0
                    self._cg_op_cache['M'] = LinearOperator((num_features, num_features), matvec=lambda x: x / diag, dtype=base_op.dtype)
    
        cg_op, rmatvec_block, base_op_rows, M = self._cg_op_cache['op'], self._cg_op_cache['rmatvec_block'], self._cg_op_cache['base_op_rows'], self._cg_op_cache.get('M')
        solutions = []
        for i, rhs_b in enumerate(proc_b):
            if rhs_b is None: solutions.append(None); continue
            num_scenarios = rhs_b.op.shape[1]
            d_block = np.zeros((base_op_rows, num_scenarios), dtype=cg_op.dtype)
            start_row = sum(self.A[k].op.shape[0] for k in range(i))
            weighted_b = self._densify_op(self.sqrt_weights[i]) * rhs_b.op if self.sqrt_weights[i] is not None else rhs_b.op
            d_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b
            rhs_for_cg = rmatvec_block(d_block)
            if rhs_for_cg.ndim == 1: rhs_for_cg = rhs_for_cg.reshape(-1, 1)
            sol = np.zeros_like(rhs_for_cg)
            cg_kwargs = {"atol": self.tolerance, "rtol": self.tolerance, "M": M, **kwargs}
            for k in range(num_scenarios):
                sol[:, k], _ = cg(cg_op, rhs_for_cg[:, k], **cg_kwargs)
            solutions.append(sol.reshape(self.A[0].trailing_shape + rhs_b.trailing_shape))
        return solutions

    def _solve_svd(self, proc_b, **kwargs):
        if 'U' not in self._svd_cache:
            base_op, _ = self._get_linear_operator(num_scenarios=1)
            num_features = base_op.shape[1]
            print("INFO: Computing SVD. Densifying matrix-free operators if present.")
            M_dense = base_op.matmat(np.eye(num_features))
            u, s, vt = np.linalg.svd(M_dense, full_matrices=False)
            s_inv = np.zeros_like(s); stable_s = s > (self.tolerance * (s[0] if s.size > 0 else 0)); s_inv[stable_s] = 1.0 / s[stable_s]
            self._svd_cache.update({'U': u, 's_inv': s_inv, 'Vt': vt, 'base_op_rows': base_op.shape[0] // 1})
        U, s_inv, Vt, base_op_rows = self._svd_cache['U'], self._svd_cache['s_inv'], self._svd_cache['Vt'], self._svd_cache['base_op_rows']
        solutions = []
        for i, rhs_b in enumerate(proc_b):
            if rhs_b is None: solutions.append(None); continue
            num_scenarios = rhs_b.op.shape[1]
            d_block = np.zeros((base_op_rows, num_scenarios), dtype=U.dtype)
            start_row = sum(self.A[j].op.shape[0] for j in range(i))
            weighted_b = self._densify_op(self.sqrt_weights[i]) * rhs_b.op if self.sqrt_weights[i] is not None else rhs_b.op
            d_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b
            solution_matrix = Vt.T @ (s_inv[:, np.newaxis] * (U.T @ d_block))
            solutions.append(solution_matrix.reshape(self.A[0].trailing_shape + rhs_b.trailing_shape))
        return solutions