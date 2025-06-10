"""Least squares module.

This module contains the LeastSquares class for multi-constraint
multi-dimensional least squares.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, svds

from pynamit.math.flattened_array import FlattenedArray

class LeastSquares:
    """Class for multi-constraint multi-dimensional least squares.

    Solves problems of the form::

        min_x ( Σᵢ ||Wᵢ(Aᵢx - bᵢ)||² + Σⱼ λⱼ||Lⱼx||² )
    """
    def __init__(self, A, solution_dims, weights=None, reg_lambda=None, reg_L=None, solver='normal', tolerance=1e-10):
        """Initialize the least squares solver.

        Parameters
        ----------
        A : list of ndarray or ndarray
            Forward operator array(s).
        solution_dims : int
            Number of dimensions in the shared solution space `x`.
        weights : list of ndarray or ndarray, optional
            Weight array(s) for each data term.
        reg_lambda : list of float or float, optional
            Regularization parameter(s).
        reg_L : list of ndarray or ndarray, optional
            Regularization operator array(s).
        solver : {'iterative', 'normal', 'svds'}, optional
            The solver to use.
            - 'iterative': Matrix-free Conjugate Gradient. Memory-efficient, avoids forming AᵀA.
            - 'normal': `numpy.linalg.solve` on the normal equations. Fast for smaller problems.
            - 'svds': Matrix-free Truncated SVD on the normal equations operator. Numerically robust.
        tolerance : float, optional
            Tolerance for the iterative solver or SVD cutoff.
        """
        solvers = ['iterative', 'normal', 'svds']
        if solver not in solvers: raise ValueError(f"Solver must be one of {solvers}.")
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
        if self.n_reg_terms != len(self.reg_lambda): raise ValueError("Number of reg_L must match number of reg_lambda.")

    @staticmethod
    def _flatten_arrays(arrays, n_terms, n_leading_flattened=None, n_trailing_flattened=None):
        if arrays is None: return [None] * n_terms
        arr_list = arrays if isinstance(arrays, list) else [arrays]
        if len(arr_list) == 1 and n_terms > 1: arr_list = arr_list * n_terms
        arrays_compounded = []
        for i in range(n_terms):
            arr = arr_list[i] if i < len(arr_list) else None
            if arr is not None:
                lead_dims = n_leading_flattened[i] if n_leading_flattened and i < len(n_leading_flattened) else None
                trail_dims = n_trailing_flattened[i] if n_trailing_flattened and i < len(n_trailing_flattened) else None
                arrays_compounded.append(FlattenedArray(arr, n_leading_flattened=lead_dims, n_trailing_flattened=trail_dims))
            else: arrays_compounded.append(None)
        return arrays_compounded

    @property
    def ATW(self):
        if not hasattr(self, "_ATW"): self._ATW = [((w.array.reshape(-1, 1) * a.array).T if w is not None else a.array.T) for a, w in zip(self.A, self.weights)]
        return self._ATW
    @property
    def ATWA(self):
        if not hasattr(self, "_ATWA"): self._ATWA = sum([atw @ a.array for atw, a in zip(self.ATW, self.A)])
        return self._ATWA
    @property
    def ATWA_plus_R(self):
        if not hasattr(self, "_ATWA_plus_R"):
            diag_ATWA = np.diag(self.ATWA)
            ATWA_scale = np.median(diag_ATWA) if len(diag_ATWA) > 0 else 1.0
            R = np.zeros_like(self.ATWA)
            for i in range(self.n_reg_terms):
                if self.reg_lambda[i] > 0 and self.reg_L[i] is not None:
                    Li = self.reg_L[i].array; LTLi = Li.T @ Li
                    diag_LTLi = np.diag(LTLi)
                    LTLi_scale = np.median(diag_LTLi) if len(diag_LTLi) > 0 else 1.0
                    if LTLi_scale > 1e-12: R += (self.reg_lambda[i] / LTLi_scale * ATWA_scale) * LTLi
            self._ATWA_plus_R = self.ATWA + R
        return self._ATWA_plus_R

    def solve(self, b, **kwargs):
        """Solves the system using the method specified at initialization."""
        if self.solver == 'iterative': return self._solve_iterative(b, **kwargs)
        elif self.solver == 'normal': return self._solve_normal(b)
        elif self.solver == 'svds': return self._solve_svds(b, **kwargs)

    def as_normal_eq_operator_matrix_free(self, n_scenarios):
        """Creates a matrix-free LinearOperator for the normal equations matrix (AᵀWA + R)."""
        n_features = self.A[0].array.shape[1]; dtype = self.A[0].array.dtype
        shape = (n_features * n_scenarios, n_features * n_scenarios)
        def _matvec(x_flat):
            x_matrix = x_flat.reshape(n_features, n_scenarios); y_matrix = np.zeros_like(x_matrix, dtype=dtype)
            for i in range(self.n_data_terms):
                Ax = self.A[i].array @ x_matrix
                if self.weights[i] is not None: Ax *= self.weights[i].array.reshape(-1, 1)
                y_matrix += self.A[i].array.T @ Ax
            for i in range(self.n_reg_terms):
                if self. scaled_lambdas[i] > 0: y_matrix += self.scaled_lambdas[i] * (self.reg_L[i].array.T @ (self.reg_L[i].array @ x_matrix))
            return y_matrix.flatten()
        return LinearOperator(shape, matvec=_matvec, rmatvec=_matvec, dtype=dtype)
    
    @property
    def scaled_lambdas(self):
        """Calculates and caches the scaled regularization parameters."""
        if not hasattr(self, "_scaled_lambdas"):
            n_features = self.A[0].array.shape[1]; dtype = self.A[0].array.dtype
            diag_ATWA = np.zeros(n_features)
            for i in range(self.n_data_terms):
                A_i = self.A[i].array
                if self.weights[i] is not None: diag_ATWA += np.sum(self.weights[i].array.flatten()[:, np.newaxis] * (A_i**2), axis=0)
                else: diag_ATWA += np.sum(A_i**2, axis=0)
            ATWA_scale = np.median(diag_ATWA) if len(diag_ATWA)>0 else 1.0
            self._scaled_lambdas = []
            for i in range(self.n_reg_terms):
                lambda_val = 0.0
                if self.reg_lambda[i] > 0 and self.reg_L[i] is not None:
                    Li = self.reg_L[i].array; LTLi = Li.T @ Li
                    diag_LTLi = np.diag(LTLi)
                    LTLi_scale = np.median(diag_LTLi) if len(diag_LTLi) > 0 else 1.0
                    if LTLi_scale > 1e-12: lambda_val = self.reg_lambda[i] / LTLi_scale * ATWA_scale
                self._scaled_lambdas.append(lambda_val)
        return self._scaled_lambdas

    def _calculate_rhs_components(self, b):
        b_list = self._flatten_arrays(b, self.n_data_terms, n_leading_flattened=[len(a.full_shapes[0]) for a in self.A])
        return [self.ATW[i] @ b_list[i].array if b_list[i] is not None else None for i in range(self.n_data_terms)], b_list
    
    def _solve_iterative(self, b, **kwargs):
        rhs_components, b_list = self._calculate_rhs_components(b)
        solutions, n_features, sol_shape = [], self.A[0].array.shape[1], self.A[0].full_shapes[1]
        cg_kwargs = {'atol': self.tolerance, 'rtol': self.tolerance}; cg_kwargs.update(kwargs)
        for i, ATWb_i in enumerate(rhs_components):
            if ATWb_i is None: solutions.append(None); continue
            n_scen = ATWb_i.shape[1] if ATWb_i.ndim > 1 else 1
            op = self.as_normal_eq_operator_matrix_free(n_scen)
            sol_flat, info = cg(op, ATWb_i.flatten(), **cg_kwargs)
            if info != 0: print(f"Warning: CG did not converge for data term {i}. Info: {info}")
            solutions.append(sol_flat.reshape(n_features, n_scen).reshape(sol_shape + b_list[i].full_shapes[1]))
        return solutions

    def _solve_normal(self, b):
        rhs_components, b_list = self._calculate_rhs_components(b); solutions = []
        for i, ATWb_i in enumerate(rhs_components):
            if ATWb_i is None: solutions.append(None); continue
            sol_matrix = np.linalg.solve(self.ATWA_plus_R, ATWb_i)
            solutions.append(sol_matrix.reshape(self.A[0].full_shapes[1] + b_list[i].full_shapes[1]))
        return solutions

    def _solve_svds(self, b, **kwargs):
        """Solves for each component using a matrix-free truncated SVD on the normal equations."""
        rhs_components, b_list = self._calculate_rhs_components(b)
        solutions, n_features, sol_shape = [], self.A[0].array.shape[1], self.A[0].full_shapes[1]
        
        for i, ATWb_i in enumerate(rhs_components):
            if ATWb_i is None: solutions.append(None); continue
            n_scen = ATWb_i.shape[1] if ATWb_i.ndim > 1 else 1
            
            op = self.as_normal_eq_operator_matrix_free(n_scen)
            k = kwargs.get('k', min(op.shape) - 1)
            
            u, s, vh = svds(op, k=k, tol=self.tolerance)
            
            s_inv = np.where(s > self.tolerance, 1/s, 0)
            
            # Since the operator is symmetric, U is effectively the same as V (vh = u.T).
            # Solution to Mx = y is U S⁻¹ Uᵀ y
            sol_flat = u @ (s_inv * (u.T @ ATWb_i.flatten()))
            
            solutions.append(sol_flat.reshape(n_features, n_scen).reshape(sol_shape + b_list[i].full_shapes[1]))
        return solutions