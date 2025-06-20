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
    def __init__(self, A, solution_ndim, sqrt_weights=None, regularization_weights=None, regularization_matrices=None,
                 solver="svd", tolerance=1e-12, preconditioner=None):
        solvers = ["normal", "lsmr", "cg", "svd"]
        if solver not in solvers: raise ValueError(f"Solver must be one of {solvers}")
        if preconditioner is not None and preconditioner != "jacobi": raise ValueError("Only 'jacobi' preconditioner supported.")
        if preconditioner is not None and solver != "cg": print("Warning: Preconditioner is set but only applies to 'cg' solver.")

        self.solver, self.tolerance, self.solution_ndim, self.preconditioner = solver, tolerance, solution_ndim, preconditioner

        A_list = self._prepare_input_list(A, "A", allow_single_item=True)
        self.num_data_terms = len(A_list)
        self.A = [self._flatten(arr, num_trailing_dims=self.solution_ndim) for arr in A_list]
        self.is_matrix_free = any(isinstance(a.op, LinearOperator) for a in self.A)

        sqrt_weights_list = self._prepare_input_list(sqrt_weights, "sqrt_weights", count=self.num_data_terms)
        self.sqrt_weights = []
        for i, sw in enumerate(sqrt_weights_list):
            if sw is None:
                self.sqrt_weights.append(None)
            elif sw.ndim == 1:
                self.sqrt_weights.append(sw)
            else:
                num_leading_dims = self.A[i].leading_dim_count
                self.sqrt_weights.append(self._flatten(sw, num_leading_dims=num_leading_dims))
        
        reg_L_list = self._prepare_input_list(regularization_matrices, "regularization_matrices", allow_single_item=True, is_optional=True)
        self.num_reg_terms = len(reg_L_list)
        self.regularization_matrices = [self._flatten(L, num_trailing_dims=self.solution_ndim) if L is not None else None for L in reg_L_list]
        
        if any(L is not None and isinstance(L.op, LinearOperator) for L in self.regularization_matrices): self.is_matrix_free = True
        if self.is_matrix_free and solver in ["normal", "svd"]: print(f"Warning: Solver '{solver}' with matrix-free operators will be slow as it requires densification.")
        
        num_features = self.A[0].op.shape[1]
        for i, L_item in enumerate(self.regularization_matrices):
            if L_item and L_item.op.shape[1] != num_features: raise ValueError(f"Shape mismatch in regularization term {i}.")

        self.regularization_weights = self._prepare_input_list(regularization_weights, "regularization_weights", count=self.num_reg_terms, default_val=0.0)
        self._op_cache, self._cg_op_cache, self._svd_cache = {}, {}, {}
        self._jacobi_precond_diag = None

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
        
        if num_leading_dims is None and num_trailing_dims is None: split_idx = array.ndim - 1
        elif num_leading_dims is None: split_idx = array.ndim - num_trailing_dims
        else: split_idx = num_leading_dims
            
        leading_shape, trailing_shape = array.shape[:split_idx], array.shape[split_idx:]
        new_shape = (math.prod(leading_shape) if leading_shape else 1, math.prod(trailing_shape) if trailing_shape else 1)
        return _ProcessedItem(array.reshape(new_shape), trailing_shape, len(leading_shape))

    def _densify_op(self, item):
        if item is None: return None
        op = item.op if isinstance(item, _ProcessedItem) else item
        if isinstance(op, LinearOperator):
            return op.matmat(np.eye(op.shape[1], dtype=op.dtype))
        return op
        
    @property
    def _scaled_regularization_weights(self):
        if hasattr(self, "_scaled_reg_weights_cache"): return self._scaled_reg_weights_cache
        
        is_fully_dense = not self.is_matrix_free
        if not is_fully_dense:
            print("Warning: Cannot auto-scale regularization for matrix-free operators. Using raw sqrt_weights.")
            self._scaled_reg_weights_cache = self.regularization_weights
            return self.regularization_weights
        
        diag_data_terms = []
        for a, sqrt_w_item in zip(self.A, self.sqrt_weights):
            A_dense = self._densify_op(a)
            if sqrt_w_item is None:
                term_diag = np.sum(A_dense**2, axis=0)
            elif isinstance(sqrt_w_item, _ProcessedItem) and sqrt_w_item.op.shape[0] == sqrt_w_item.op.shape[1]:
                L_dense = self._densify_op(sqrt_w_item)
                G = L_dense @ A_dense
                term_diag = np.sum(G**2, axis=0)
            else:
                w_diag = self._densify_op(sqrt_w_item).flatten()**2
                if A_dense.shape[0] != w_diag.shape[0]:
                    if A_dense.shape[0] % w_diag.shape[0] == 0:
                        repeat_factor = A_dense.shape[0] // w_diag.shape[0]
                        w_diag = np.repeat(w_diag, repeat_factor)
                    else:
                        raise ValueError(f"Diagonal weight for term has length {w_diag.shape[0]}, "
                                         f"incompatible with operator output dim {A_dense.shape[0]}")
                term_diag = np.sum(w_diag[:, np.newaxis] * (A_dense**2), axis=0)
            diag_data_terms.append(term_diag)
        diag_data = np.sum(diag_data_terms, axis=0)

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
        if num_scenarios in self._op_cache: return self._op_cache[num_scenarios]

        num_features = self.A[0].op.shape[1]
        sqrt_scaled_reg_weights = [np.sqrt(w) for w in self._scaled_regularization_weights]

        op_rows = sum(a.op.shape[0] for a in self.A) + \
                  sum(l.op.shape[0] for i, l in enumerate(self.regularization_matrices) if l and sqrt_scaled_reg_weights[i] > 0)

        def _apply_op_to_block(item, x_block):
            if isinstance(item.op, LinearOperator):
                num_scen_in_block = x_block.shape[1]
                res_block = np.zeros((item.op.shape[0], num_scen_in_block), dtype=x_block.dtype)
                for k in range(num_scen_in_block): res_block[:, k] = item.op.matvec(x_block[:, k])
                return res_block
            else: return item.op @ x_block
        
        def _apply_op_T_to_block(item, y_block):
            if isinstance(item.op, LinearOperator):
                num_scen_in_block = y_block.shape[1]
                res_block = np.zeros((item.op.shape[1], num_scen_in_block), dtype=y_block.dtype)
                for k in range(num_scen_in_block): res_block[:, k] = item.op.rmatvec(y_block[:, k])
                return res_block
            else: return item.op.T.conj() @ y_block

        def matvec_block(x_block):
            out = np.zeros((op_rows, x_block.shape[1]), dtype=self.A[0].op.dtype)
            row = 0
            for i in range(self.num_data_terms):
                res_block = _apply_op_to_block(self.A[i], x_block)
                sqrt_w_item = self.sqrt_weights[i]
                if sqrt_w_item is not None:
                    if isinstance(sqrt_w_item, _ProcessedItem) and sqrt_w_item.op.shape[0] == sqrt_w_item.op.shape[1]:
                        res_block = sqrt_w_item.op @ res_block
                    else: 
                        sqrt_w_diag = self._densify_op(sqrt_w_item).flatten()
                        if res_block.shape[0] != sqrt_w_diag.shape[0]:
                            sqrt_w_diag = np.repeat(sqrt_w_diag, res_block.shape[0] // sqrt_w_diag.shape[0])
                        res_block *= sqrt_w_diag[:, np.newaxis]
                out[row : row + res_block.shape[0]] = res_block
                row += res_block.shape[0]
            for i, L in enumerate(self.regularization_matrices):
                if L and sqrt_scaled_reg_weights[i] > 0:
                    res_block = _apply_op_to_block(L, x_block)
                    out[row : row + res_block.shape[0]] = sqrt_scaled_reg_weights[i] * res_block
                    row += res_block.shape[0]
            return out

        def rmatvec_block(y_block):
            x_block = np.zeros((num_features, y_block.shape[1]), dtype=y_block.dtype)
            row = 0
            for i, item in enumerate(self.A):
                y_part = y_block[row : row + item.op.shape[0]]
                sqrt_w_item = self.sqrt_weights[i]
                if sqrt_w_item is not None:
                    if isinstance(sqrt_w_item, _ProcessedItem) and sqrt_w_item.op.shape[0] == sqrt_w_item.op.shape[1]:
                        y_part = sqrt_w_item.op.T.conj() @ y_part
                    else:
                        sqrt_w_diag = self._densify_op(sqrt_w_item).flatten()
                        if y_part.shape[0] != sqrt_w_diag.shape[0]:
                            sqrt_w_diag = np.repeat(sqrt_w_diag, y_part.shape[0] // sqrt_w_diag.shape[0])
                        y_part *= sqrt_w_diag[:, np.newaxis]
                x_block += _apply_op_T_to_block(item, y_part)
                row += item.op.shape[0]
            for i, L in enumerate(self.regularization_matrices):
                if L and sqrt_scaled_reg_weights[i] > 0:
                    y_part = y_block[row : row + L.op.shape[0]]
                    x_block += sqrt_scaled_reg_weights[i] * _apply_op_T_to_block(L, y_part)
                    row += L.op.shape[0]
            return x_block

        shape = (op_rows * num_scenarios, num_features * num_scenarios)
        def matvec_final(x_flat): return matvec_block(x_flat.reshape(num_features, num_scenarios, order="F")).flatten("F")
        def rmatvec_final(y_flat): return rmatvec_block(y_flat.reshape(op_rows, num_scenarios, order="F")).flatten("F")

        op = LinearOperator(shape, matvec=matvec_final, rmatvec=rmatvec_final, dtype=self.A[0].op.dtype)
        self._op_cache[num_scenarios] = (op, rmatvec_block)
        return op, rmatvec_block

    def _apply_weight_to_b(self, b_op, sqrt_w_item):
        if sqrt_w_item is None or b_op is None: return b_op
        if isinstance(sqrt_w_item, _ProcessedItem) and sqrt_w_item.op.shape[0] == sqrt_w_item.op.shape[1]:
            return sqrt_w_item.op @ b_op
        else:
            sqrt_w_diag = self._densify_op(sqrt_w_item).flatten()
            if b_op.shape[0] != sqrt_w_diag.shape[0]:
                sqrt_w_diag = np.repeat(sqrt_w_diag, b_op.shape[0] // sqrt_w_diag.shape[0])
            return sqrt_w_diag[:, np.newaxis] * b_op

    def solve(self, b, **kwargs):
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        proc_b = [self._flatten(item, num_leading_dims=self.A[i].leading_dim_count) if item is not None else None for i, item in enumerate(b_list)]
        solver_map = {"normal": self._solve_normal, "lsmr": self._solve_lsmr, "cg": self._solve_cg, "svd": self._solve_svd}
        return solver_map[self.solver](proc_b, **kwargs)

    def _solve_lsmr(self, proc_b, **kwargs):
        solutions = []
        lsmr_kwargs = {"atol": self.tolerance, "btol": self.tolerance, **kwargs}
        for i, b_item in enumerate(proc_b):
            if b_item is None:
                solutions.append(None); continue
            
            num_scenarios = b_item.op.shape[1]
            if num_scenarios == 0:
                solutions.append(None); continue

            linear_op, _ = self._get_linear_operator(num_scenarios)
            op_rows = linear_op.shape[0] // num_scenarios
            
            rhs_block = np.zeros((op_rows, num_scenarios), dtype=linear_op.dtype)
            weighted_b = self._apply_weight_to_b(b_item.op, self.sqrt_weights[i])
            start_row = sum(a.op.shape[0] for a in self.A[:i])
            rhs_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b
            
            sol_flat, *_ = lsmr(linear_op, rhs_block.flatten("F"), **lsmr_kwargs)
            
            sol_block = sol_flat.reshape(self.A[0].op.shape[1], num_scenarios, order="F")
            solutions.append(sol_block.reshape(self.A[0].trailing_shape + b_item.trailing_shape))
            
        return solutions

    def _solve_cg(self, proc_b, **kwargs):
        if 'op' not in self._cg_op_cache:
            num_features = self.A[0].op.shape[1]
            base_op, rmatvec_func = self._get_linear_operator(num_scenarios=1)
            def normal_op_matvec(x): return base_op.rmatvec(base_op.matvec(x))
            self._cg_op_cache['op'] = LinearOperator((num_features, num_features), matvec=normal_op_matvec, dtype=base_op.dtype)
            self._cg_op_cache['rmatvec_block'] = rmatvec_func
            self._cg_op_cache['base_op_rows'] = base_op.shape[0]
            self._cg_op_cache['M'] = None
            if self.preconditioner == "jacobi":
                _ = self._scaled_regularization_weights
                diag = self._jacobi_precond_diag
                if diag is not None:
                    diag[diag < 1e-12] = 1.0
                    def precon_matvec(x): return x / diag
                    self._cg_op_cache['M'] = LinearOperator((num_features, num_features), matvec=precon_matvec, dtype=base_op.dtype)
                else: print("Warning: Cannot compute Jacobi preconditioner for matrix-free operators. Preconditioner disabled.")

        cg_op, rmatvec_block, base_op_rows, M = self._cg_op_cache['op'], self._cg_op_cache['rmatvec_block'], self._cg_op_cache['base_op_rows'], self._cg_op_cache.get('M')
        solutions = []
        for i, rhs_b in enumerate(proc_b):
            if rhs_b is None:
                solutions.append(None); continue
            num_scenarios = rhs_b.op.shape[1]
            d_block = np.zeros((base_op_rows, num_scenarios), dtype=cg_op.dtype)
            start_row = sum(self.A[k].op.shape[0] for k in range(i))
            weighted_b = self._apply_weight_to_b(rhs_b.op, self.sqrt_weights[i])
            d_block[start_row : start_row + weighted_b.shape[0], :] = weighted_b
            rhs_for_cg = rmatvec_block(d_block)
            if rhs_for_cg.ndim == 1: rhs_for_cg = rhs_for_cg.reshape(-1, 1)
            sol = np.zeros_like(rhs_for_cg)
            cg_kwargs = {"atol": self.tolerance, "rtol": self.tolerance, "M": M, **kwargs}
            for k in range(num_scenarios):
                sol[:, k], exit_code = cg(cg_op, rhs_for_cg[:, k], **cg_kwargs)
                if exit_code != 0: print(f"Warning: CG solver did not converge for scenario {k}, exit code {exit_code}")
            solutions.append(sol.reshape(self.A[0].trailing_shape + rhs_b.trailing_shape))
        return solutions

    @property
    def _full_normal_matrix(self):
        if hasattr(self, "_normal_matrix_cache"): return self._normal_matrix_cache
        num_features = self.A[0].op.shape[1]
        normal_matrix = np.zeros((num_features, num_features), dtype=self.A[0].op.dtype)
        for i in range(self.num_data_terms):
            A_i = self._densify_op(self.A[i])
            sqrt_w_item = self.sqrt_weights[i]
            if sqrt_w_item is None:
                normal_matrix += A_i.T @ A_i
            elif isinstance(sqrt_w_item, _ProcessedItem) and sqrt_w_item.op.shape[0] == sqrt_w_item.op.shape[1]:
                L_i = self._densify_op(sqrt_w_item)
                G_i = L_i @ A_i
                normal_matrix += G_i.T @ G_i
            else:
                w_diag = self._densify_op(sqrt_w_item).flatten()**2
                if A_i.shape[0] != w_diag.shape[0]:
                    w_diag = np.repeat(w_diag, A_i.shape[0] // w_diag.shape[0])
                normal_matrix += A_i.T @ (w_diag[:, np.newaxis] * A_i)
        for i, weight in enumerate(self._scaled_regularization_weights):
            if weight > 0 and self.regularization_matrices[i] is not None:
                L_reg_i = self._densify_op(self.regularization_matrices[i])
                normal_matrix += weight * (L_reg_i.T @ L_reg_i)
        self._normal_matrix_cache = normal_matrix
        return self._normal_matrix_cache

    def _solve_normal(self, proc_b, **kwargs):
        solutions = []
        normal_matrix = self._full_normal_matrix
        for i, rhs_item in enumerate(proc_b):
            if rhs_item is None:
                solutions.append(None); continue
            A_i = self._densify_op(self.A[i])
            b_i = self._densify_op(rhs_item)
            sqrt_w_item = self.sqrt_weights[i]
            rhs_for_term = None
            if sqrt_w_item is None:
                rhs_for_term = A_i.T @ b_i
            elif isinstance(sqrt_w_item, _ProcessedItem) and sqrt_w_item.op.shape[0] == sqrt_w_item.op.shape[1]:
                L_i = self._densify_op(sqrt_w_item)
                rhs_for_term = A_i.T @ L_i.T @ (L_i @ b_i)
            else:
                w_diag = self._densify_op(sqrt_w_item).flatten()**2
                if b_i.shape[0] != w_diag.shape[0]:
                    w_diag = np.repeat(w_diag, b_i.shape[0] // w_diag.shape[0])
                rhs_for_term = A_i.T @ (w_diag[:, np.newaxis] * b_i)
            sol = np.linalg.solve(normal_matrix, rhs_for_term)
            solutions.append(sol.reshape(self.A[0].trailing_shape + rhs_item.trailing_shape))
        return solutions

    def _solve_svd(self, proc_b, **kwargs):
        if 'U' not in self._svd_cache:
            base_op, _ = self._get_linear_operator(num_scenarios=1)
            num_features = base_op.shape[1]
            print("INFO: Computing SVD. Densifying matrix-free operators if present.")
            M_dense = base_op.matmat(np.eye(num_features))
            u, s, vt = np.linalg.svd(M_dense, full_matrices=False)
            s_inv = np.zeros_like(s)
            stable_s = s > (self.tolerance * s[0] if s.size > 0 else 0)
            s_inv[stable_s] = 1.0 / s[stable_s]
            self._svd_cache.update({'U': u, 's_inv': s_inv, 'Vt': vt, 'base_op_rows': base_op.shape[0]})
        U, s_inv, Vt, base_op_rows = self._svd_cache['U'], self._svd_cache['s_inv'], self._svd_cache['Vt'], self._svd_cache['base_op_rows']
        solutions = []
        for i, rhs_b in enumerate(proc_b):
            if rhs_b is None:
                solutions.append(None); continue
            num_scenarios = rhs_b.op.shape[1]
            b_eff_i = np.zeros((base_op_rows, num_scenarios), dtype=U.dtype)
            start_row = sum(self.A[j].op.shape[0] for j in range(i))
            weighted_b = self._apply_weight_to_b(self._densify_op(rhs_b), self.sqrt_weights[i])
            b_eff_i[start_row : start_row + weighted_b.shape[0], :] = weighted_b
            solution_matrix = Vt.T @ (s_inv[:, np.newaxis] * (U.T @ b_eff_i))
            solutions.append(solution_matrix.reshape(self.A[0].trailing_shape + rhs_b.trailing_shape))
        return solutions