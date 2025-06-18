"""State module.

This module contains the State class for managing the electrodynamic
state of the ionosphere, with support for both dense-matrix and
matrix-free computational modes.
"""

import numpy as np
import xarray as xr
from pynamit.math.constants import mu0, RE
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.tensor_operations import tensor_pinv
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.spherical_harmonics.sh_basis import SHBasis
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres
from scipy.linalg import expm

class State(object):
    """
    Class for managing the electrodynamic state of the ionosphere.
    Supports both a dense-matrix mode and a matrix-free (iterative) mode.
    """

    def __init__(self, basis, mainfield, cs_basis, settings, PFAC_matrix=None):
        self.use_matrix_free = bool(getattr(settings, 'use_matrix_free', False))
        print(f"INFO: State class in {'MATRIX-FREE (iterative)' if self.use_matrix_free else 'DENSE (direct)'} mode.")

        self.basis, self.mainfield = basis, mainfield
        self.RI, self.RM = settings.RI, (None if settings.RM == 0 else settings.RM)
        self.latitude_boundary, self.ignore_PFAC = settings.latitude_boundary, bool(settings.ignore_PFAC)
        self.connect_hemispheres, self.FAC_integration_steps = bool(settings.connect_hemispheres), settings.FAC_integration_steps
        self.ih_constraint_scaling, self.integrator = settings.ih_constraint_scaling, settings.integrator
        if PFAC_matrix is not None: self._T_to_Ve = PFAC_matrix
        self.u, self.Br, self.jr, self.etaP, self.etaH = None, None, None, None, None

        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.basis_evaluator_zero_added = BasisEvaluator(SHBasis(settings.Nmax, settings.Mmax, Nmin=0), self.grid)
        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)

        self.cp_grid, self.cp_basis_evaluator, self.cp_b_evaluator = None, None, None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(self.RI, self.grid.theta, self.grid.phi)
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)

        self.m_ind_to_Br, self.m_imp_to_jr = -(self.RI**2) * self.basis.laplacian(self.RI), self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1 / self.RI
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = 1 / self.RI * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs

        self.bP, self.bH, self.bu = None, None, None
        
        self.initialize_constraints()
        self._build_u_coeffs_to_E_coeffs()
        self._invalidate_caches()

    # --- Properties for lazy computation ---
    @property
    def G_m_imp_to_JS(self):
        if not hasattr(self, "_G_m_imp_to_JS"):
            T_to_J_cf_coeffs = self.RI / mu0
            G_T_to_JS = -1 / self.RI * self.basis_evaluator.G_grad * T_to_J_cf_coeffs
            self._G_m_imp_to_JS = G_T_to_JS + np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2],[0]))
        return self._G_m_imp_to_JS

    @property
    def G_m_ind_to_JS(self):
        if not hasattr(self, "_G_m_ind_to_JS"):
            self._G_m_ind_to_JS = self.G_Ve_to_JS

            if self.RM is not None:
                br_shift, vi_shift = self.basis.radial_shift_Ve(self.RM, self.RI), self.basis.radial_shift_Vi(self.RI, self.RM)
                den = 1 - br_shift * vi_shift
                self.G_Br_to_JS = self.G_Ve_to_JS * (-1 / den * br_shift / self.m_ind_to_Br)
                self._G_m_ind_to_JS *= 1 + (1 / den * br_shift * vi_shift)

        return self._G_m_ind_to_JS

    @property
    def bP_prop(self):
        if self.bP is None: self.bP = np.array([[self.b_evaluator.bphi**2+self.b_evaluator.br**2, -self.b_evaluator.btheta*self.b_evaluator.bphi], [-self.b_evaluator.btheta*self.b_evaluator.bphi, self.b_evaluator.btheta**2+self.b_evaluator.br**2]])
        return self.bP

    @property
    def bH_prop(self):
        if self.bH is None: self.bH = np.array([[np.zeros(self.b_evaluator.grid.size), self.b_evaluator.br], [-self.b_evaluator.br, np.zeros(self.b_evaluator.grid.size)]])
        return self.bH
        
    @property
    def bu_prop(self):
        if self.bu is None: self.bu = -np.array([[np.zeros(self.b_evaluator.grid.size), self.b_evaluator.Br], [-self.b_evaluator.Br, np.zeros(self.b_evaluator.grid.size)]])
        return self.bu

    @property
    def m_ind_to_bP_JS_prop(self):
        if not hasattr(self, "_m_ind_to_bP_JS"): self._m_ind_to_bP_JS = np.einsum("ijk,jkl->ikl", self.bP_prop, self.G_m_ind_to_JS, optimize=True)
        return self._m_ind_to_bP_JS
    
    @property
    def m_ind_to_bH_JS_prop(self):
        if not hasattr(self, "_m_ind_to_bH_JS"): self._m_ind_to_bH_JS = np.einsum("ijk,jkl->ikl", self.bH_prop, self.G_m_ind_to_JS, optimize=True)
        return self._m_ind_to_bH_JS
        
    @property
    def m_imp_to_bP_JS_prop(self):
        if not hasattr(self, "_m_imp_to_bP_JS"): self._m_imp_to_bP_JS = np.einsum("ijk,jkl->ikl", self.bP_prop, self.G_m_imp_to_JS, optimize=True)
        return self._m_imp_to_bP_JS
        
    @property
    def m_imp_to_bH_JS_prop(self):
        if not hasattr(self, "_m_imp_to_bH_JS"): self._m_imp_to_bH_JS = np.einsum("ijk,jkl->ikl", self.bH_prop, self.G_m_imp_to_JS, optimize=True)
        return self._m_imp_to_bH_JS

    @property
    def Br_to_bP_JS_prop(self):
        if not hasattr(self, "_Br_to_bP_JS"):
            if self.RM is None: self._Br_to_bP_JS = None
            else: self._Br_to_bP_JS = np.einsum("ijk,jkl->ikl", self.bP_prop, self.G_Br_to_JS, optimize=True)
        return self._Br_to_bP_JS

    @property
    def Br_to_bH_JS_prop(self):
        if not hasattr(self, "_Br_to_bH_JS"):
            if self.RM is None: self._Br_to_bH_JS = None
            else: self._Br_to_bH_JS = np.einsum("ijk,jkl->ikl", self.bH_prop, self.G_Br_to_JS, optimize=True)
        return self._Br_to_bH_JS
        
    @property
    def T_to_Ve(self):
        if not hasattr(self, "_T_to_Ve"):
            self._T_to_Ve = xr.DataArray(data=np.zeros((self.basis.index_length, self.basis.index_length)), coords={"i": np.arange(self.basis.index_length), "j": np.arange(self.basis.index_length)}, dims=["i", "j"])
            if not (self.mainfield.kind == "radial" or self.ignore_PFAC):
                rk_steps = self.FAC_integration_steps
                Delta_k, rks = np.diff(rk_steps), np.array(rk_steps[:-1] + 0.5 * np.diff(rk_steps))
                if any(rks < self.RI):
                    raise ValueError(
                        "All FAC integration steps must be outside the ionospheric boundary (RI)."
                    )
                if self.RM is not None and any(rks > self.RM):
                    raise ValueError(
                        "All FAC integration steps must be inside the magnetospheric boundary (RM)."
                    )
                JS_rk_to_Ve_rk = tensor_pinv(self.G_Ve_to_JS, n_leading_flattened=2, rtol=0)
                for i, rk in enumerate(rks):
                    print(
                        f"Calculating matrix for poloidal field of inclined FACs. Progress: {i + 1}/{rks.size}",
                        end="\r" if i < (rks.size - 1) else "\n",
                        flush=True,
                    )
                    theta_mapped, phi_mapped = self.mainfield.map_coords(
                        self.RI, rk, self.grid.theta, self.grid.phi
                    )
                    mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
                    rk_b_evaluator, mapped_b_evaluator = (
                        FieldEvaluator(self.mainfield, self.grid, rk),
                        FieldEvaluator(self.mainfield, mapped_grid, self.RI),
                    )
                    mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)
                    m_imp_to_jr = mapped_basis_evaluator.scaled_G(self.m_imp_to_jr)
                    jr_to_JS_rk = np.array(
                        [
                            rk_b_evaluator.Btheta / mapped_b_evaluator.Br,
                            rk_b_evaluator.Bphi / mapped_b_evaluator.Br,
                        ]
                    )
                    m_imp_to_JS_rk = np.einsum(
                        "ij,jk->ijk", jr_to_JS_rk, m_imp_to_jr, optimize=True
                    )
                    Ve_rk_to_Ve = self.basis.radial_shift_Ve(rk, self.RI).reshape((-1, 1, 1))
                    if self.RM is not None:
                        Ve_rk_to_Ve -= (
                            self.basis.radial_shift_Ve(self.RM, self.RI)
                            * self.basis.radial_shift_Vi(rk, self.RM)
                        ).reshape((-1, 1, 1))
                        factor = -1 / (
                            1
                            - self.basis.radial_shift_Ve(self.RM, self.RI)
                            * self.basis.radial_shift_Vi(self.RI, self.RM)
                        )
                    else:
                        factor = -1
                    JS_rk_to_Ve = JS_rk_to_Ve_rk * Ve_rk_to_Ve
                    self._T_to_Ve += (
                        Delta_k[i] * factor * np.tensordot(JS_rk_to_Ve, m_imp_to_JS_rk, 2)
                    )
        return self._T_to_Ve

    @property
    def m_ind_to_E_coeffs(self):
        if self._m_ind_to_E_coeffs is None:
            if self.etaP is None: raise RuntimeError("Conductance must be set before accessing operators.")
            etaP_grid = self.etaP.to_grid(self.basis_evaluator_zero_added)
            etaH_grid = self.etaH.to_grid(self.basis_evaluator_zero_added)
            m_ind_to_E_grid = np.einsum("i,jik->jik", etaP_grid, self.m_ind_to_bP_JS_prop) + np.einsum("i,jik->jik", etaH_grid, self.m_ind_to_bH_JS_prop)
            dense_op = self.basis_evaluator.least_squares_solution_helmholtz(m_ind_to_E_grid)
            self._m_ind_to_E_coeffs = self._create_operator_from_dense(dense_op) if self.use_matrix_free else dense_op
        return self._m_ind_to_E_coeffs
        
    @property
    def m_imp_to_E_coeffs(self):
        if self._m_imp_to_E_coeffs is None:
            if self.etaP is None: raise RuntimeError("Conductance must be set before accessing operators.")
            etaP_grid = self.etaP.to_grid(self.basis_evaluator_zero_added)
            etaH_grid = self.etaH.to_grid(self.basis_evaluator_zero_added)
            m_imp_to_E_grid = np.einsum("i,jik->jik", etaP_grid, self.m_imp_to_bP_JS_prop) + np.einsum("i,jik->jik", etaH_grid, self.m_imp_to_bH_JS_prop)
            dense_op = self.basis_evaluator.least_squares_solution_helmholtz(m_imp_to_E_grid)
            self._m_imp_to_E_coeffs = self._create_operator_from_dense(dense_op) if self.use_matrix_free else dense_op
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self):
        if self._Br_to_E_coeffs is None:
            if self.RM is None: return None
            if self.etaP is None: raise RuntimeError("Conductance must be set before accessing operators.")
            etaP_grid = self.etaP.to_grid(self.basis_evaluator_zero_added)
            etaH_grid = self.etaH.to_grid(self.basis_evaluator_zero_added)
            Br_to_E_grid = np.einsum("i,jik->jik", etaP_grid, self.Br_to_bP_JS_prop) + np.einsum("i,jik->jik", etaH_grid, self.Br_to_bH_JS_prop)
            dense_op = self.basis_evaluator.least_squares_solution_helmholtz(Br_to_E_grid)
            self._Br_to_E_coeffs = self._create_operator_from_dense(dense_op) if self.use_matrix_free else dense_op
        return self._Br_to_E_coeffs

    def _invalidate_caches(self):
        """Invalidate all cached matrices and operators that depend on conductance."""
        self._m_ind_to_E_coeffs, self._m_imp_to_E_coeffs, self._Br_to_E_coeffs = None, None, None
        self.m_ind_to_E_df = None
        if self.use_matrix_free: self._fwd_solver_cache, self._adj_solver_cache, self._E_map_constraint_operator = {}, {}, None
        else: self._coeffs_to_m_imp_cache = None

    def initialize_constraints(self):
        """Initialize constraints."""
        if self.mainfield.kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif self.mainfield.kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(self.grid.lat, self.grid.lon, (self.RI - RE) * 1e-3)
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            # Default to no low-latitude mask if field type is unknown
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)
            
        self.jr_coeffs_to_j_apex = (self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G).copy()
        self.E_coeffs_to_E_apex_ll_diff = None
        
        # --- THIS IS THE FIX ---
        # Only perform conjugate point calculations if hemispheres are connected
        if self.connect_hemispheres:
            if self.mainfield.kind == "radial": 
                raise ValueError("Hemispheres cannot be connected with a radial magnetic field.")
            
            # These objects are only guaranteed to exist if connect_hemispheres is True
            if self.cp_b_evaluator is not None and self.cp_basis_evaluator is not None:
                # J-mapping constraint
                jr_coeffs_to_j_apex_cp = (self.cp_b_evaluator.radial_to_apex.reshape((-1, 1)) * self.cp_basis_evaluator.G)
                self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]
                
                # E-mapping constraint
                E_coeffs_to_E_apex = np.einsum("ijk,jklm->iklm", self.b_evaluator.horizontal_to_apex, self.basis_evaluator.G_helmholtz, optimize=True)
                E_coeffs_to_E_apex_cp = np.einsum("ijk,jklm->iklm", self.cp_b_evaluator.horizontal_to_apex, self.cp_basis_evaluator.G_helmholtz, optimize=True)
                self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray((E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask])
        
    def update(self, input_timeseries, time, interpolation=False):
        conductance_updated = False
        for key in input_timeseries.datasets.keys():
            updated_input_entry = input_timeseries.get_entry_if_changed(key, time, interpolation=interpolation)
            if updated_input_entry is not None:
                if key == "conductance":
                    conductance_updated = True
                    # NOTE: User needs to ensure FieldExpansion is correctly initialized
                    self.etaP = FieldExpansion(input_timeseries.storage_bases["conductance"], coeffs=updated_input_entry["etaP"], field_type=input_timeseries.vars["conductance"]["etaP"])
                    self.etaH = FieldExpansion(input_timeseries.storage_bases["conductance"], coeffs=updated_input_entry["etaH"], field_type=input_timeseries.vars["conductance"]["etaH"])

                elif key == "jr":
                    self.jr = FieldExpansion(
                        input_timeseries.storage_bases["jr"],
                        coeffs=updated_input_entry["jr"],
                        field_type=input_timeseries.vars["jr"]["jr"],
                    )
                elif key == "Br":
                    if self.RM is None:
                        raise ValueError("Br input can only be set if RM is not None")
                    self.Br = FieldExpansion(
                        input_timeseries.storage_bases["Br"],
                        coeffs=updated_input_entry["Br"],
                        field_type=input_timeseries.vars["Br"]["Br"],
                    )
                elif key == "u":
                    self.u = FieldExpansion(
                        input_timeseries.storage_bases["u"],
                        coeffs=updated_input_entry["u"].reshape((2, -1)),
                        field_type=input_timeseries.vars["u"]["u"],
                    )

        if conductance_updated:
            self._invalidate_caches()

    def _build_u_coeffs_to_E_coeffs(self):
        """Builds the conductance-independent mapping from wind to E-field."""
        u_coeffs_to_uxB_grid = np.einsum('ijk,jklm->iklm', self.bu_prop, self.basis_evaluator.G_helmholtz, optimize=True)
        dense_op = self.basis_evaluator.least_squares_solution_helmholtz(u_coeffs_to_uxB_grid)
        self.u_coeffs_to_E_coeffs = self._create_operator_from_dense(dense_op) if self.use_matrix_free else dense_op

    def _create_operator_from_dense(self, dense_op):
        """Creates a LinearOperator that mimics a dense tensor's contraction rules."""
        n_c = self.basis.index_length
        is_vector_map = (dense_op.ndim == 4)

        if is_vector_map:
            shape = (2 * n_c, 2 * n_c)
            def matvec(v_in_flat): return np.tensordot(dense_op, v_in_flat.reshape(2, -1), 2).flatten()
            def rmatvec(v_out_flat):
                A_adj = np.conj(dense_op).transpose(2,3,0,1)
                return np.tensordot(A_adj, v_out_flat.reshape(2, -1), 2).flatten()
        else: # Scalar mapping
            shape = (2 * n_c, n_c)
            def matvec(v_in): return dense_op.dot(v_in)
            def rmatvec(v_out): return dense_op.T.dot(v_out)
        
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    def _contract(self, op, coeffs, output_shape):
        """Agnostic helper to apply an operator to coefficients, returning a specific shape."""
        # --- DEFINITIVE FIX ---
        if op is None or (isinstance(coeffs, int) and coeffs == 0):
            return np.zeros(output_shape)
        
        if isinstance(op, LinearOperator):
            return op.dot(coeffs.flatten()).reshape(output_shape)
        
        # Dense numpy array logic
        if coeffs.ndim == 1: # Scalar input coeffs
            return op.dot(coeffs).reshape(output_shape)
        elif coeffs.ndim == 2: # Vector input coeffs
            return np.tensordot(op, coeffs, 2)
        else:
            raise ValueError(f"Unsupported coefficient shape: {coeffs.shape}")
    
    def calculate_noind_coeffs(self):
        """Calculate no-induction coefficients using the selected mode."""
        E_direct = self._contract(self.u_coeffs_to_E_coeffs, self.u.coeffs if self.u else 0, output_shape=(2, self.basis.index_length))
        
        if self.Br is not None:
            E_direct += self._contract(self.Br_to_E_coeffs, self.Br.coeffs, output_shape=(2, self.basis.index_length))

        if self.use_matrix_free:
            jr_c = self.jr.coeffs if self.jr is not None else None
            m_imp = self._solve_for_m_imp_iteratively(jr_c, E_direct)
        else:
            coeffs_to_m_imp = self._get_or_compute_coeffs_to_m_imp()
            m_imp = np.zeros(self.basis.index_length)
            if self.jr is not None: m_imp += coeffs_to_m_imp[0].dot(self.jr.coeffs)
            if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                m_imp += np.tensordot(coeffs_to_m_imp[1], -E_direct, 2)
        
        E_imp = self._contract(self.m_imp_to_E_coeffs, m_imp, output_shape=(2, self.basis.index_length))
        return E_direct + E_imp, m_imp

    def calculate_ind_coeffs(self, m_ind):
        """Calculate induced coefficients using the selected mode."""
        E_direct_ind = self._contract(self.m_ind_to_E_coeffs, m_ind, output_shape=(2, self.basis.index_length))
        if self.use_matrix_free:
            m_imp_ind = self._solve_for_m_imp_iteratively(None, E_direct_ind)
        else:
            coeffs_to_m_imp = self._get_or_compute_coeffs_to_m_imp()
            m_imp_ind = np.zeros(self.basis.index_length)
            if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                m_imp_ind = np.tensordot(coeffs_to_m_imp[1], -E_direct_ind, 2)
        E_imp_ind = self._contract(self.m_imp_to_E_coeffs, m_imp_ind, output_shape=(2, self.basis.index_length))
        return E_direct_ind + E_imp_ind, m_imp_ind

    # --- Mode-Specific Methods and Time Evolution ---

    def _get_or_compute_coeffs_to_m_imp(self):
        if self._coeffs_to_m_imp_cache is not None: return self._coeffs_to_m_imp_cache
        constraint_A = [self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))]
        rhs_B = [self.jr_coeffs_to_j_apex]
        if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
            constraint_A.append(np.tensordot(self.E_coeffs_to_E_apex_ll_diff, self.m_imp_to_E_coeffs, 2) * self.ih_constraint_scaling)
            rhs_B.append(self.E_coeffs_to_E_apex_ll_diff * self.ih_constraint_scaling)
        solver = LeastSquaresSolver(constraint_A, 1, solver="lsmr")
        self._coeffs_to_m_imp_cache = solver.solve(rhs_B)
        return self._coeffs_to_m_imp_cache

    def _get_or_create_E_map_constraint_operator(self):
        if self._E_map_constraint_operator is not None: return self._E_map_constraint_operator
        if self.E_coeffs_to_E_apex_ll_diff is None: return None
        A, B = self.E_coeffs_to_E_apex_ll_diff, self.m_imp_to_E_coeffs
        if B is None: return None
        shape = (A.shape[0] * A.shape[1], B.shape[1])
        def matvec(m): return np.einsum('ijkl,kl->ij', A, B.dot(m).reshape(2, -1)).flatten()
        def rmatvec(y): return B.rmatvec(np.einsum('ij,ijkl->kl', y.reshape(A.shape[0], A.shape[1]), A).flatten())
        self._E_map_constraint_operator = LinearOperator(shape=shape, matvec=matvec, rmatvec=rmatvec)
        return self._E_map_constraint_operator

    def _solve_for_m_imp_iteratively(self, jr_coeffs, E_coeffs_direct):
        A_list, b_list = [], []
        if jr_coeffs is not None:
            A_list.append(self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1)))
            b_list.append(np.dot(self.jr_coeffs_to_j_apex, jr_coeffs))
        E_map_op = self._get_or_create_E_map_constraint_operator()
        if E_map_op is not None:
            A_list.append(self.ih_constraint_scaling * E_map_op)
            b_E = -np.einsum('ijkl,kl->ij', self.E_coeffs_to_E_apex_ll_diff, E_coeffs_direct)
            b_list.append(b_E.flatten() * self.ih_constraint_scaling)
        if not A_list: return np.zeros(self.basis.index_length)
        key = (jr_coeffs is not None, E_map_op is not None)
        if key not in self._fwd_solver_cache:
            self._fwd_solver_cache[key] = LeastSquaresSolver(A_list, 1, solver="lsmr")
        solutions = self._fwd_solver_cache[key].solve(b_list)
        return sum(s.flatten() for s in solutions if s is not None)

    def _solve_for_m_imp_adjoint(self, grad_m_imp):
        A_adj_list = []
        has_jr = any(item is not None for item in [self.jr])
        E_map_op = self._get_or_create_E_map_constraint_operator()
        has_E = E_map_op is not None
        if has_jr: A_adj_list.append((self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))).T)
        if has_E: A_adj_list.append(self.ih_constraint_scaling * E_map_op.T)
        if not A_adj_list: return None, None
        key = (has_jr, has_E)
        if key not in self._adj_solver_cache:
            self._adj_solver_cache[key] = LeastSquaresSolver(A_adj_list, 1, solver="lsmr")
        grad_b_list = self._adj_solver_cache[key].solve([grad_m_imp])
        grad_b_jr = grad_b_list[0] if has_jr else None
        grad_b_E = grad_b_list[-1] if has_E else None
        return grad_b_jr, grad_b_E
    
    def build_m_ind_to_E_df(self):
        """Builds the time-evolution operator L for the selected mode."""
        if self.m_ind_to_E_df is not None: return
        shape = (self.basis.index_length, self.basis.index_length)
        def matvec(m): return self.calculate_ind_coeffs(m)[0][1]
        
        if self.use_matrix_free:
            def rmatvec(grad_out):
                grad_E = np.zeros((2, shape[0])); grad_E[1] = grad_out
                grad_m_imp = self.m_imp_to_E_coeffs.rmatvec(grad_E.flatten())
                _, grad_b_E = self._solve_for_m_imp_adjoint(grad_m_imp)
                if grad_b_E is not None:
                    E_diff = self.E_coeffs_to_E_apex_ll_diff
                    grad_E_from_solve = -np.einsum('ij,ijkl->kl', grad_b_E.reshape(E_diff.shape[0], E_diff.shape[1]), E_diff)
                    grad_E += grad_E_from_solve * self.ih_constraint_scaling
                return self.m_ind_to_E_coeffs.rmatvec(grad_E.flatten())
            self.m_ind_to_E_df = LinearOperator(shape=shape, matvec=matvec, rmatvec=rmatvec)
        else:
            L_dense = np.array([matvec(v) for v in np.eye(shape[1])]).T
            self.m_ind_to_E_df = LinearOperator(shape=shape, matvec=lambda v: L_dense @ v, rmatvec=lambda v: L_dense.T @ v)

    def evolve_m_ind(self, m_ind, dt, E_coeffs_noind, steady_state_m_ind=None):
        """Evolve induced magnetic field coefficients using the selected mode."""
        if self.m_ind_to_E_df is None: self.build_m_ind_to_E_df()
        op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df
        
        if self.integrator == "euler":
            return m_ind + dt * op.dot(m_ind) + dt * self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
        elif self.integrator == "exponential":
            if steady_state_m_ind is None: steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            diff = m_ind - steady_state_m_ind
            if self.use_matrix_free:
                return expm_multiply(dt * op, diff) + steady_state_m_ind
            else:
                A_dense = op.matmat(np.eye(op.shape[1]))
                return (expm(dt * A_dense) @ diff) + steady_state_m_ind
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

    def steady_state_m_ind(self, E_coeffs_noind):
        """Calculate steady-state m_ind using the selected mode."""
        if self.m_ind_to_E_df is None: self.build_m_ind_to_E_df()
        op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df
        b = -self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
        
        if self.use_matrix_free:
            m_ind, exit_code = gmres(op, b, rtol=1e-12, atol=0)
            if exit_code != 0: print(f"Warning: GMRES failed with exit code {exit_code}")
            return m_ind
        else:
            A_dense = op.matmat(np.eye(op.shape[1]))
            return np.linalg.solve(A_dense, b)