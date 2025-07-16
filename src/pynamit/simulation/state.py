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
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres, cg
from scipy.linalg import expm

LEAST_SQUARES_SOLVER = "svd"
MATRIX_WEIGHTS = False
MATRIX_FREE_A = False

class State(object):
    """
    Class for managing the electrodynamic state of the ionosphere.
    Supports both a dense-matrix mode and a matrix-free (iterative) mode.
    """

    def __init__(self, basis, mainfield, cs_basis, settings, PFAC_matrix=None):
        self.basis, self.mainfield = basis, mainfield
        self.RI, self.RM = settings.RI, (None if settings.RM == 0 else settings.RM)
        self.latitude_boundary, self.ignore_PFAC = (settings.latitude_boundary, bool(settings.ignore_PFAC))
        self.connect_hemispheres, self.FAC_integration_steps = (bool(settings.connect_hemispheres), settings.FAC_integration_steps)
        self.ih_constraint_scaling, self.integrator = (settings.ih_constraint_scaling, settings.integrator)
        if PFAC_matrix is not None: self._T_to_Ve = PFAC_matrix
        self.u, self.Br, self.jr, self.etaP, self.etaH = None, None, None, None, None
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.G_helmholtz_pinv = tensor_pinv(self.basis_evaluator.G_helmholtz, n_leading_flattened=2)
        self.basis_evaluator_zero_added = BasisEvaluator(SHBasis(settings.Nmax, settings.Mmax, Nmin=0), self.grid)
        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)
        self.cp_grid, self.cp_basis_evaluator, self.cp_b_evaluator = None, None, None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(self.RI, self.grid.theta, self.grid.phi)
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)
        self.m_ind_to_Br, self.m_imp_to_jr = (-(self.RI**2) * self.basis.laplacian(self.RI), self.RI / mu0 * self.basis.laplacian(self.RI))
        self.E_df_to_d_m_ind_dt = 1 / self.RI
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = 1 / self.RI * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs
        self.bP, self.bH, self.bu = None, None, None

        self._m_imp_solver = None
        
        self.initialize_constraints()
        self._build_u_coeffs_to_E_coeffs()
        self._invalidate_caches()

    def _invalidate_caches(self):
        """Invalidate all cached data that depends on conductance or other state."""
        self._m_ind_to_E_coeffs = None
        self._m_imp_to_E_coeffs = None
        self._Br_to_E_coeffs = None
        self.m_ind_to_E_df = None

        self._m_imp_solver = None
 
        if hasattr(self, "_M_total_on_grid"): del self._M_total_on_grid
        if hasattr(self, "_E_map_constraint_operator"): del self._E_map_constraint_operator

    @property
    def G_m_imp_to_JS(self):
        if not hasattr(self, "_G_m_imp_to_JS"):
            T_to_J_cf_coeffs = self.RI / mu0
            G_T_to_JS = -1 / self.RI * self.basis_evaluator.G_grad * T_to_J_cf_coeffs
            self._G_m_imp_to_JS = G_T_to_JS + np.tensordot(
                self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0])
            )
        return self._G_m_imp_to_JS

    @property
    def G_m_ind_to_JS(self):
        if not hasattr(self, "_G_m_ind_to_JS"):
            self._G_m_ind_to_JS = self.G_Ve_to_JS
            if self.RM is not None:
                br_shift, vi_shift = (
                    self.basis.radial_shift_Ve(self.RM, self.RI),
                    self.basis.radial_shift_Vi(self.RI, self.RM),
                )
                den = 1 - br_shift * vi_shift
                self.G_Br_to_JS = self.G_Ve_to_JS * (-1 / den * br_shift / self.m_ind_to_Br)
                self._G_m_ind_to_JS = self._G_m_ind_to_JS * (1 + (1 / den * br_shift * vi_shift))
        return self._G_m_ind_to_JS

    @property
    def bP_prop(self):
        if self.bP is None:
            self.bP = np.array(
                [
                    [
                        self.b_evaluator.bphi**2 + self.b_evaluator.br**2,
                        -self.b_evaluator.btheta * self.b_evaluator.bphi,
                    ],
                    [
                        -self.b_evaluator.btheta * self.b_evaluator.bphi,
                        self.b_evaluator.btheta**2 + self.b_evaluator.br**2,
                    ],
                ]
            )
        return self.bP

    @property
    def bH_prop(self):
        if self.bH is None:
            self.bH = np.array(
                [
                    [np.zeros(self.b_evaluator.grid.size), self.b_evaluator.br],
                    [-self.b_evaluator.br, np.zeros(self.b_evaluator.grid.size)],
                ]
            )
        return self.bH

    @property
    def bu_prop(self):
        if self.bu is None:
            self.bu = -np.array(
                [
                    [np.zeros(self.b_evaluator.grid.size), self.b_evaluator.Br],
                    [-self.b_evaluator.Br, np.zeros(self.b_evaluator.grid.size)],
                ]
            )
        return self.bu

    @property
    def T_to_Ve(self):
        if not hasattr(self, "_T_to_Ve"):
            self._T_to_Ve = xr.DataArray(
                data=np.zeros((self.basis.index_length, self.basis.index_length)),
                coords={
                    "i": np.arange(self.basis.index_length),
                    "j": np.arange(self.basis.index_length),
                },
                dims=["i", "j"],
            )
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
    def m_imp_solver(self):
        """
        A lazy-loaded property that builds and caches the LeastSquaresSolver
        for the m_imp calculation.
        """
        # With this simplified design, we just check if the solver has been built.
        if self._m_imp_solver is None:
            A_list, data_shapes, sqrt_weights = [], [], []
            if MATRIX_WEIGHTS:
                A_list.append(np.diag(self.m_imp_to_jr))
                data_shapes.append([self.basis.index_length])
                sqrt_weights.append(self.jr_constraint_L_matrix)
                if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                    A_list.append(self.m_imp_to_E_coeffs)
                    data_shapes.append((2, self.basis.index_length))
                    sqrt_weights.append(self.E_constraint_L_matrix * self.ih_constraint_scaling)
            else:
                A_list.append(self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1)))
                data_shapes.append((self.grid.size,))
                sqrt_weights.append(None)
                if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                    A_list.append(self._get_or_create_E_map_constraint_operator() * self.ih_constraint_scaling)
                    data_shapes.append((2, np.sum(self.ll_mask)))
                    sqrt_weights.append(None)
            
            # The solver is now built based on the current global settings
            solver = LeastSquaresSolver(A_list, self.basis.index_length, data_shapes,
                                        sqrt_weights=sqrt_weights, solver=LEAST_SQUARES_SOLVER)
            self._m_imp_solver = solver
            
        return self._m_imp_solver

    def _solve_for_m_imp(self, jr_coeffs, E_direct_coeffs):
        """
        Calculates the imposed potential coefficients (m_imp) by solving a
        least-squares problem.
        """
        # 1. Get the pre-configured solver instance
        solver = self.m_imp_solver

        # 2. Construct the right-hand-side vector `b`
        rhs_B = []
        if MATRIX_WEIGHTS:
            rhs_B.append(jr_coeffs)
            if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                rhs_B.append(-E_direct_coeffs)
        else:
            if jr_coeffs is not None: 
                rhs_B.append(np.dot(self.jr_coeffs_to_j_apex, jr_coeffs))
            else: 
                rhs_B.append(None)
                
            if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                E_apex_from_direct = np.einsum("cikl,kl->ci", self.E_coeffs_to_E_apex_ll_diff, E_direct_coeffs, optimize=True)
                rhs_B.append(-E_apex_from_direct.flatten() * self.ih_constraint_scaling)
        
        # 3. Solve the system
        m_imp = solver.solve(rhs_B)
        return m_imp if m_imp is not None else np.zeros(self.basis.index_length)

    def _solve_for_m_imp_adjoint(self, grad_m_imp):
        """
        Calculates the gradient of the loss w.r.t. the inputs `b` of the m_imp problem.
        """
        # 1. Get the pre-configured solver instance. This call is now clean and
        # will create the solver if it doesn't exist, or retrieve it from cache.
        solver = self.m_imp_solver

        # 2. Solve (A.T @ A) @ y = grad_m_imp for y using CG.
        cg_op, M = solver._cg_components
        y, exit_code = cg(cg_op, grad_m_imp.flatten(), M=M, atol=1e-12, rtol=1e-12)
        if exit_code != 0:
            print(f"Warning: Adjoint CG solver did not converge (exit_code={exit_code}).")

        # 3. Propagate y back through the A operators to get grad_b = A @ y
        grad_b_list = []
        for A_op_item in solver.A:
            grad_b = self._apply_operator(A_op_item.op, y, A_op_item.output_shape)
            grad_b_list.append(grad_b)

        # 4. Unpack and handle scaling for the gradients
        grad_b_jr, grad_b_E = None, None
        
        if len(grad_b_list) > 0 and self.jr is not None:
            grad_b_jr_raw = grad_b_list[0]
            if not MATRIX_WEIGHTS:
                 grad_b_jr = np.dot(self.jr_coeffs_to_j_apex.T, grad_b_jr_raw)
            else:
                 grad_b_jr = grad_b_jr_raw
        
        if len(grad_b_list) > 1 and self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
            grad_b_E_raw = grad_b_list[1]
            if not MATRIX_WEIGHTS:
                grad_E_apex = -grad_b_E_raw.reshape(2, np.sum(self.ll_mask)) / self.ih_constraint_scaling
                grad_b_E = np.einsum("ci,cikl->kl", grad_E_apex.conj(), self.E_coeffs_to_E_apex_ll_diff.conj(), optimize=True)
            else:
                grad_b_E = -grad_b_E_raw.reshape(2, self.basis.index_length)

        return grad_b_jr, grad_b_E

    def _create_E_coeffs_operator(self, G_X_to_JS):
        """
        Creates a dense or matrix-free operator for the transformation from input coefficients
        (X_coeffs) to electric field coefficients (E_coeffs).
        """
        if G_X_to_JS is None:
            return None
        if MATRIX_FREE_A:
            return self._create_E_coeffs_linear_operator(G_X_to_JS)
        else:
            return self._create_E_coeffs_dense_operator(G_X_to_JS)

    def _create_E_coeffs_dense_operator(self, G_X_to_JS):
        einsum_str = "cmik,ijk,jk...->cm..."
        return np.einsum(
            einsum_str, self.G_helmholtz_pinv, self.M_total_on_grid, G_X_to_JS, optimize=True
        )

    def _create_E_coeffs_linear_operator(self, G_X_to_JS):
        """
        Creates a LinearOperator for the transformation from input coefficients (X)
        to electric field coefficients (E_coeffs).
        This corresponds to the operation: E_coeffs = G_helm_pinv @ M @ G_JS @ x_coeffs
        """
        shape_in = G_X_to_JS.shape[2:]
        shape_out = (2, self.basis.index_length)
        shape = (np.prod(shape_out), np.prod(shape_in))
        M_total = self.M_total_on_grid

        def matvec(x_coeffs_flat):
            """Forward: E_coeffs = G_helm_pinv @ (M @ (G_JS @ x_coeffs))"""
            x_coeffs = x_coeffs_flat.reshape(shape_in)
            js_on_grid = np.einsum("jk...,...->jk", G_X_to_JS, x_coeffs, optimize=True)
            j_div_free_on_grid = np.einsum("ijk,jk->ik", M_total, js_on_grid, optimize=True)
            E_coeffs = np.einsum("cmik,ik->cm", self.G_helmholtz_pinv, j_div_free_on_grid, optimize=True)
            return E_coeffs.flatten()

        def rmatvec(grad_E_coeffs_flat):
            """Adjoint: grad_x = G_JS^H @ M^H @ G_helm_pinv^H @ grad_E"""
            grad_E_coeffs = grad_E_coeffs_flat.reshape(shape_out)
            grad_intermediate_1 = np.einsum("cmik,cm->ik", self.G_helmholtz_pinv.conj(), grad_E_coeffs, optimize=True)
            grad_intermediate_2 = np.einsum("ijk,ik->jk", M_total.conj(), grad_intermediate_1, optimize=True)
            grad_x_coeffs = np.einsum("jk...,jk->...", G_X_to_JS.conj(), grad_intermediate_2, optimize=True)
            return grad_x_coeffs.flatten()

        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    @property
    def M_total_on_grid(self):
        if not hasattr(self, "_M_total_on_grid"):
            if (self.etaP is None) or (self.etaH is None):
                raise RuntimeError("Conductance must be set before use.")
            eta_stacked_coeffs = np.stack([self.etaP.coeffs, self.etaH.coeffs], axis=0)
            G_eta = self.basis_evaluator_zero_added.G
            b_stacked = np.stack([self.bP_prop, self.bH_prop], axis=0)
            self._M_total_on_grid = np.einsum(
                "sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked_coeffs, optimize=True
            )
        return self._M_total_on_grid

    @property
    def m_ind_to_E_coeffs(self):
        if self._m_ind_to_E_coeffs is None:
            self._m_ind_to_E_coeffs = self._create_E_coeffs_operator(
                self.G_m_ind_to_JS
            )
        return self._m_ind_to_E_coeffs

    @property
    def m_imp_to_E_coeffs(self):
        if self._m_imp_to_E_coeffs is None:
            self._m_imp_to_E_coeffs = self._create_E_coeffs_operator(
                self.G_m_imp_to_JS
            )
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self):
        if self._Br_to_E_coeffs is None:
            G_Br_to_JS = getattr(self, "G_Br_to_JS", None)
            self._Br_to_E_coeffs = self._create_E_coeffs_operator(
                G_Br_to_JS
            )
        return self._Br_to_E_coeffs

    def initialize_constraints(self):
        if self.mainfield.kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif self.mainfield.kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - RE) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)
        self.jr_coeffs_to_j_apex = (
            self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        ).copy()
        self.E_coeffs_to_E_apex_ll_diff = None
        if self.connect_hemispheres:
            if self.mainfield.kind == "radial":
                raise ValueError("Hemispheres cannot be connected with a radial magnetic field.")
            if self.cp_b_evaluator is not None and self.cp_basis_evaluator is not None:
                jr_coeffs_to_j_apex_cp = (
                    self.cp_b_evaluator.radial_to_apex.reshape((-1, 1)) * self.cp_basis_evaluator.G
                )
                self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]
                E_coeffs_to_E_apex = np.einsum(
                    "ijk,jklm->iklm",
                    self.b_evaluator.horizontal_to_apex,
                    self.basis_evaluator.G_helmholtz,
                    optimize=True,
                )
                E_coeffs_to_E_apex_cp = np.einsum(
                    "ijk,jklm->iklm",
                    self.cp_b_evaluator.horizontal_to_apex,
                    self.cp_basis_evaluator.G_helmholtz,
                    optimize=True,
                )
                self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                    (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask]
                )

    def update(self, input_timeseries, time, interpolation=False):
        conductance_updated = False
        for key in input_timeseries.datasets.keys():
            updated_input_entry = input_timeseries.get_entry_if_changed(
                key, time, interpolation=interpolation
            )
            if updated_input_entry is not None:
                if key == "conductance":
                    conductance_updated = True
                    self.etaP = FieldExpansion(
                        input_timeseries.storage_bases["conductance"],
                        coeffs=updated_input_entry["etaP"],
                        field_type=input_timeseries.vars["conductance"]["etaP"],
                    )
                    self.etaH = FieldExpansion(
                        input_timeseries.storage_bases["conductance"],
                        coeffs=updated_input_entry["etaH"],
                        field_type=input_timeseries.vars["conductance"]["etaH"],
                    )
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
        G_u_to_uxB_grid = np.einsum(
            "ijk, jklm -> iklm", self.bu_prop, self.basis_evaluator.G_helmholtz, optimize=True
        )
        dense_op = self.basis_evaluator.least_squares_solution_helmholtz(G_u_to_uxB_grid)
        if MATRIX_FREE_A:

            def matvec(u_coeffs_flat):
                u_coeffs = u_coeffs_flat.reshape(2, -1)
                return np.einsum("iklm, lm -> ik", dense_op, u_coeffs, optimize=True).flatten()

            def rmatvec(grad_E_coeffs_flat):
                grad_E_coeffs = grad_E_coeffs_flat.reshape(2, -1)
                return np.einsum(
                    "iklm, ik -> lm", dense_op.conj(), grad_E_coeffs, optimize=True
                ).flatten()

            shape_in_out = (2, self.basis.index_length)
            shape = (np.prod(shape_in_out), np.prod(shape_in_out))
            self.u_coeffs_to_E_coeffs = LinearOperator(
                shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
            )
        else:
            self.u_coeffs_to_E_coeffs = dense_op

    def _apply_operator(self, op, coeffs, output_shape):
        if op is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return np.zeros(output_shape)
        if isinstance(op, LinearOperator):
            in_vec = coeffs.flatten() if isinstance(coeffs, np.ndarray) else coeffs
            return op.matvec(in_vec).reshape(output_shape)
        if coeffs.ndim == 1:
            return np.tensordot(op, coeffs, 1)
        elif coeffs.ndim == 2:
            return np.tensordot(op, coeffs, axes=2)
        else:
            raise ValueError(f"Unsupported coefficient shape: {coeffs.shape}")

    def calculate_noind_coeffs(self):
        E_shape = (2, self.basis.index_length)
        E_direct = self._apply_operator(
            self.u_coeffs_to_E_coeffs, self.u.coeffs if self.u else 0, E_shape
        )
        if self.Br is not None:
            E_direct += self._apply_operator(self.Br_to_E_coeffs, self.Br.coeffs, E_shape)
        m_imp = self._solve_for_m_imp(self.jr.coeffs if self.jr else None, E_direct)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct + E_imp, m_imp

    def calculate_ind_coeffs(self, m_ind):
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, m_ind, E_shape)
        m_imp_ind = self._solve_for_m_imp(None, E_direct_ind)
        E_imp_ind = self._apply_operator(self.m_imp_to_E_coeffs, m_imp_ind, E_shape)
        return E_direct_ind + E_imp_ind, m_imp_ind

    @property
    def jr_constraint_L_matrix(self):
        if not hasattr(self, "_jr_constraint_L_matrix_cache"):
            H_jr = self.jr_coeffs_to_j_apex
            _, S, Vt = np.linalg.svd(H_jr, full_matrices=False)
            L_jr_2D = Vt.T @ np.diag(S) @ Vt
            self._jr_constraint_L_matrix_cache = L_jr_2D
        return self._jr_constraint_L_matrix_cache

    @property
    def E_constraint_L_matrix(self):
        if not hasattr(self, "_E_constraint_L_matrix_cache"):
            L_E = None
            if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                H_E = self.E_coeffs_to_E_apex_ll_diff
                H_E_2D = H_E.reshape((np.prod(H_E.shape[:2]), np.prod(H_E.shape[2:])))
                _, S, Vt = np.linalg.svd(H_E_2D, full_matrices=False)
                L_E_2D = Vt.T @ np.diag(S) @ Vt
                L_E = L_E_2D.reshape(H_E.shape[2:] + H_E.shape[2:])
            self._E_constraint_L_matrix_cache = L_E
        return self._E_constraint_L_matrix_cache

    def _get_or_create_E_map_constraint_operator(self):
        if hasattr(self, "_E_map_constraint_operator"):
            return self._E_map_constraint_operator

        if not MATRIX_FREE_A:
            op = np.tensordot(
                self.E_coeffs_to_E_apex_ll_diff,
                self.m_imp_to_E_coeffs,
                axes=([2, 3], [0, 1])
            )
            n_ll = np.sum(self.ll_mask)
            op_2d = op.reshape(2 * n_ll, self.basis.index_length)
            self._E_map_constraint_operator = op_2d
        else:
            n_ll = np.sum(self.ll_mask)
            n_c_imp = self.basis.index_length
            shape_out = (2, n_ll)
            shape_in = (n_c_imp,)
            shape = (np.prod(shape_out), np.prod(shape_in))

            def matvec(m_coeffs_in):
                m_coeffs = m_coeffs_in.flatten()
                E_coeffs = self._apply_operator(self.m_imp_to_E_coeffs, m_coeffs, (2, self.basis.index_length))
                E_apex_diff = np.einsum("cikl,kl->ci", self.E_coeffs_to_E_apex_ll_diff, E_coeffs, optimize=True)
                return E_apex_diff.flatten()

            def rmatvec(grad_E_apex_flat):
                grad_E_apex = grad_E_apex_flat.reshape(shape_out)
                grad_E_coeffs = np.einsum("cikl,ci->kl", self.E_coeffs_to_E_apex_ll_diff.conj(), grad_E_apex, optimize=True)
                op = self.m_imp_to_E_coeffs
                grad_m_coeffs_flat = op.rmatvec(grad_E_coeffs.flatten())
                return grad_m_coeffs_flat

            self._E_map_constraint_operator = LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

        return self._E_map_constraint_operator

    def build_m_ind_to_E_df(self):
        if self.m_ind_to_E_df is not None: return
        shape = (self.basis.index_length, self.basis.index_length)
        def matvec(m):
            E_ind, _ = self.calculate_ind_coeffs(m)
            return E_ind[1]
        if MATRIX_FREE_A:
            def rmatvec(grad_out):
                grad_E_phi = grad_out.flatten()
                grad_E = np.zeros((2, shape[0]), dtype=grad_E_phi.dtype)
                grad_E[1] = grad_E_phi
                total_grad_E_direct_ind = grad_E.copy()
                op_m_imp_E = self.m_imp_to_E_coeffs
                grad_m_imp_ind_flat = op_m_imp_E.rmatvec(grad_E.flatten())
                _, grad_E_direct_ind_from_m_imp = self._solve_for_m_imp_adjoint(grad_m_imp_ind_flat)
                if grad_E_direct_ind_from_m_imp is not None:
                    total_grad_E_direct_ind += grad_E_direct_ind_from_m_imp
                op_m_ind_E = self.m_ind_to_E_coeffs
                grad_m_ind_flat = op_m_ind_E.rmatvec(total_grad_E_direct_ind.flatten())
                return grad_m_ind_flat
            self.m_ind_to_E_df = LinearOperator(shape=shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
        else:
            self.m_ind_to_E_df = np.array([matvec(v) for v in np.eye(shape[1])]).T

    def evolve_m_ind(self, m_ind, dt, E_coeffs_noind, steady_state_m_ind=None):
        if self.m_ind_to_E_df is None: self.build_m_ind_to_E_df()
        op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df
        b = self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
        if self.integrator == "euler": return m_ind + dt * (op.dot(m_ind) + b)
        elif self.integrator == "exponential":
            if steady_state_m_ind is None: steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            diff = m_ind - steady_state_m_ind
            if MATRIX_FREE_A: return expm_multiply(dt * op, diff) + steady_state_m_ind
            else: return (expm(dt * op) @ diff) + steady_state_m_ind
        else: raise ValueError(f"Unknown integrator: {self.integrator}")

    def steady_state_m_ind(self, E_coeffs_noind):
        if self.m_ind_to_E_df is None: self.build_m_ind_to_E_df()
        op = self.m_ind_to_E_df
        b = -E_coeffs_noind[1]
        if MATRIX_FREE_A:
            m_ind, exit_code = gmres(op, b, rtol=1e-12, atol=0)
            if exit_code != 0: print(f"Warning: GMRES failed with exit code {exit_code}")
            return m_ind
        else: return np.linalg.solve(op, b)