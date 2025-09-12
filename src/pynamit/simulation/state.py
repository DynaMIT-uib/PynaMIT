"""State module.

This module contains the State class for managing the electrodynamic
state of the ionosphere.
"""

import numpy as np
import xarray as xr
from pynamit.math.constants import mu0, RE
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.tensor_operations import tensor_pinv
from pynamit.math.least_squares_solver import LeastSquaresSolver, TensorChain
from pynamit.spherical_harmonics.sh_basis import SHBasis
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres
from scipy.linalg import expm
import math


class State(object):
    """Class for managing the electrodynamic state of the ionosphere.
    (Docstring unchanged)
    """

    def __init__(self, basis, mainfield, cs_basis, settings, PFAC_matrix=None):
        """Initialize the ionospheric state."""
        # This method is correct and unchanged from the previous version.
        self.matrix_weights = getattr(settings, "matrix_weights", False)
        self.solver_type = getattr(settings, "least_squares_solver", "cg")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.basis = basis
        self.mainfield = mainfield
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        if PFAC_matrix is not None:
            self._T_to_Ve = PFAC_matrix
        self.u, self.Br, self.jr, self.etaP, self.etaH = None, None, None, None, None
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.basis_evaluator_zero_added = BasisEvaluator(
            SHBasis(settings.Nmax, settings.Mmax, Nmin=0), self.grid
        )
        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)
        self.G_helmholtz_pinv = tensor_pinv(
            self.basis_evaluator.G_helmholtz, n_leading_flattened=2
        )
        self.cp_grid, self.cp_basis_evaluator, self.cp_b_evaluator = None, None, None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)
        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1 / self.RI
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = 1 / self.RI * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs
        self.bP, self.bH, self.bu = None, None, None
        self._m_imp_solver = None
        self.initialize_constraints()
        self._build_u_coeffs_to_E_coeffs()
        self._invalidate_caches()

    def _invalidate_caches(self):
        self._m_ind_to_E_coeffs = None
        self._m_imp_to_E_coeffs = None
        self._Br_to_E_coeffs = None
        self.m_ind_to_E_df = None
        if hasattr(self, "_M_total_on_grid"):
            del self._M_total_on_grid
        if hasattr(self, "_E_map_constraint_operator"):
            del self._E_map_constraint_operator

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
                br_shift = self.basis.radial_shift_Ve(self.RM, self.RI)
                vi_shift = self.basis.radial_shift_Vi(self.RI, self.RM)
                den = 1 - br_shift * vi_shift
                self.G_Br_to_JS = self.G_Ve_to_JS * (-1 / den * br_shift / self.m_ind_to_Br)
                self._G_m_ind_to_JS = self._G_m_ind_to_JS * (1 + (1 / den * br_shift * vi_shift))
        return self._G_m_ind_to_JS

    @property
    def bP_prop(self):
        if self.bP is None:
            b_th, b_ph, b_r = self.b_evaluator.btheta, self.b_evaluator.bphi, self.b_evaluator.br
            self.bP = np.array(
                [[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]]
            )
        return self.bP

    @property
    def bH_prop(self):
        if self.bH is None:
            br = self.b_evaluator.br
            self.bH = np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])
        return self.bH

    @property
    def bu_prop(self):
        if self.bu is None:
            Br = self.b_evaluator.Br
            self.bu = -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])
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

    def _get_m_imp_solver_terms(self):
        terms = []
        if self.matrix_weights:
            terms.append(
                {
                    "A": np.diag(self.m_imp_to_jr),
                    "data_shape": (self.basis.index_length,),
                    "sqrt_W": self.jr_constraint_L_matrix,
                    "get_b": lambda jr, E: jr,
                    "get_grad_contrib": lambda grad_b: {"grad_jr": grad_b},
                }
            )
        else:
            terms.append(
                {
                    "A": self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1)),
                    "data_shape": (self.grid.size,),
                    "sqrt_W": None,
                    "get_b": lambda jr, E: np.dot(self.jr_coeffs_to_j_apex, jr)
                    if jr is not None
                    else None,
                    "get_grad_contrib": lambda grad_b: {
                        "grad_jr": np.dot(self.jr_coeffs_to_j_apex.T, grad_b)
                    },
                }
            )
        if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
            if self.matrix_weights:
                terms.append(
                    {
                        "A": self.m_imp_to_E_coeffs,
                        "data_shape": (2, self.basis.index_length),
                        "sqrt_W": self.E_constraint_L_matrix * self.ih_constraint_scaling,
                        "get_b": lambda jr, E: -E,
                        "get_grad_contrib": lambda grad_b: {
                            "grad_E": -grad_b.reshape(2, self.basis.index_length)
                        },
                    }
                )
            else:
                original_chain = self._get_or_create_E_map_constraint_operator()
                A_E = original_chain.with_scaling(self.ih_constraint_scaling)
                terms.append(
                    {
                        "A": A_E,
                        "data_shape": A_E.output_shape,
                        "sqrt_W": None,
                        "get_b": lambda jr, E: -np.einsum(
                            "cikl,kl->ci", self.E_coeffs_to_E_apex_ll_diff, E
                        ).flatten()
                        * self.ih_constraint_scaling
                        if E is not None
                        else None,
                        "get_grad_contrib": lambda grad_b: {
                            "grad_E": -np.einsum(
                                "ci,cikl->kl",
                                (grad_b.reshape(A_E.output_shape) / self.ih_constraint_scaling).conj(),
                                self.E_coeffs_to_E_apex_ll_diff.conj(),
                            ).conj()
                        },
                    }
                )
        return terms

    @property
    def m_imp_solver(self):
        if self._m_imp_solver is None:
            terms = self._get_m_imp_solver_terms()
            reg_weights, reg_matrices = [], []
            if self.m_imp_regularization_lambda > 0:
                n_coeffs = self.basis.index_length
                identity_op = LinearOperator(
                    shape=(n_coeffs, n_coeffs),
                    matvec=lambda x: x,
                    rmatvec=lambda x: x,
                    dtype=np.float64,
                )
                reg_weights.append(self.m_imp_regularization_lambda)
                reg_matrices.append(identity_op)
            self._m_imp_solver = LeastSquaresSolver(
                A=[t["A"] for t in terms],
                solution_shape=self.basis.index_length,
                data_shapes=[t["data_shape"] for t in terms],
                sqrt_weights=[t["sqrt_W"] for t in terms],
                regularization_weights=reg_weights,
                regularization_matrices=reg_matrices,
                solver=self.solver_type,
                preconditioner=self.preconditioner,
                picard_plot=False,
            )
        return self._m_imp_solver

    def _solve_for_m_imp(self, jr_coeffs, E_direct_coeffs):
        terms = self._get_m_imp_solver_terms()
        rhs_B = [t["get_b"](jr_coeffs, E_direct_coeffs) for t in terms]
        m_imp = self.m_imp_solver.solve(rhs_B)
        return m_imp if m_imp is not None else np.zeros(self.basis.index_length)

    def _solve_for_m_imp_adjoint(self, grad_m_imp):
        terms = self._get_m_imp_solver_terms()
        grad_b_list = self.m_imp_solver.solve_adjoint(grad_m_imp)
        grad_jr, grad_E = None, None
        for i, term in enumerate(terms):
            grad_contribs = term["get_grad_contrib"](grad_b_list[i])
            if "grad_jr" in grad_contribs and self.jr is not None:
                grad_jr = grad_contribs["grad_jr"]
            if "grad_E" in grad_contribs and self.connect_hemispheres:
                grad_E = grad_contribs["grad_E"]
        return grad_jr, grad_E

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

    def _create_E_coeffs_operator(self, G_X_to_JS):
        """
        Creates a TensorChain "recipe" for an E-field transformation.
        """
        if G_X_to_JS is None:
            return None
    
        M_total = self.M_total_on_grid
        G_helm_pinv = self.G_helmholtz_pinv
        component_tensors = [G_helm_pinv, M_total, G_X_to_JS]
    
        output_shape = (2, self.basis.index_length)
        input_shape = G_X_to_JS.shape[2:]
        
        # Tensor Shapes and Indices:
        # G_helm_pinv (A): (c,m,p,g)
        # M_total (B)    : (p,q,g)
        # G_X_to_JS (C)  : (q,g,l)
        
        einsum_string_dense = "cmpg,pqg,qgl->cml"
        einsum_string_matvec = "cmpg,pqg,qgl,l->cm"
        einsum_string_rmatvec = "qgl,pqg,cmpg,cm->l"

        return TensorChain(
            component_tensors=component_tensors,
            einsum_string_dense=einsum_string_dense,
            einsum_string_matvec=einsum_string_matvec,
            einsum_string_rmatvec=einsum_string_rmatvec,
            output_shape=output_shape,
            input_shape=input_shape,
            dtype=np.result_type(*[t.dtype for t in component_tensors]),
        )

    def _get_or_create_E_map_constraint_operator(self):
        if hasattr(self, "_E_map_constraint_operator"):
            return self._E_map_constraint_operator

        inner_chain = self.m_imp_to_E_coeffs
        outer_tensor = self.E_coeffs_to_E_apex_ll_diff # Indices: ticm
        new_component_tensors = [outer_tensor] + inner_chain.component_tensors

        # Tensors and indices:
        # outer_tensor (D): ticm
        # G_helm_pinv (A) : cmpg
        # M_total (B)     : pqg
        # G_X_to_JS (C)   : qgl
        
        einsum_string_dense = "ticm,cmpg,pqg,qgl->til"
        einsum_string_matvec = "ticm,cmpg,pqg,qgl,l->ti"
        einsum_string_rmatvec = "qgl,pqg,cmpg,ticm,ti->l"

        self._E_map_constraint_operator = TensorChain(
            component_tensors=new_component_tensors,
            einsum_string_dense=einsum_string_dense,
            einsum_string_matvec=einsum_string_matvec,
            einsum_string_rmatvec=einsum_string_rmatvec,
            output_shape=(2, int(np.sum(self.ll_mask))),
            input_shape=inner_chain.input_shape,
            dtype=inner_chain.dtype
        )
        return self._E_map_constraint_operator

    @property
    def m_ind_to_E_coeffs(self):
        if self._m_ind_to_E_coeffs is None:
            self._m_ind_to_E_coeffs = self._create_E_coeffs_operator(self.G_m_ind_to_JS)
        return self._m_ind_to_E_coeffs

    @property
    def m_imp_to_E_coeffs(self):
        if self._m_imp_to_E_coeffs is None:
            self._m_imp_to_E_coeffs = self._create_E_coeffs_operator(self.G_m_imp_to_JS)
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self):
        if self._Br_to_E_coeffs is None:
            self._Br_to_E_coeffs = self._create_E_coeffs_operator(
                getattr(self, "G_Br_to_JS", None)
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
            if self._m_imp_solver is not None:
                print(
                    "Conductance updated. Updating solver matrices while reusing preconditioner."
                )
                new_terms = self._get_m_imp_solver_terms()
                new_A = [t["A"] for t in new_terms]
                new_sqrt_W = [t["sqrt_W"] for t in new_terms]
                self.m_imp_solver.update_matrices(A=new_A, sqrt_weights=new_sqrt_W)

    def _build_u_coeffs_to_E_coeffs(self):
        G_u_to_uxB_grid = np.einsum(
            "ijk, jklm -> iklm", self.bu_prop, self.basis_evaluator.G_helmholtz, optimize=True
        )
        self.u_coeffs_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(G_u_to_uxB_grid)

    def _execute_tensor_chain(self, chain: TensorChain, x: np.ndarray = None):
        """Helper to execute a TensorChain recipe, for internal State use."""
        if x is None:  # Densify
            dense_op = np.einsum(
                chain.einsum_string_dense, *chain.component_tensors, optimize=True
            )
            return dense_op.reshape(chain.output_shape + chain.input_shape)
        else:  # Matvec
            x_tensor = x.reshape(chain.input_shape)
            all_tensors = chain.component_tensors + [x_tensor]
            res_tensor = np.einsum(
                chain.einsum_string_matvec, *all_tensors, optimize=True
            )
            return res_tensor

    def _apply_operator(self, op, coeffs, output_shape):
        if op is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return np.zeros(output_shape)
        if isinstance(op, TensorChain):
            return self._execute_tensor_chain(op, x=coeffs).reshape(output_shape)
        if isinstance(op, LinearOperator):
            return op.matvec(
                coeffs.flatten() if isinstance(coeffs, np.ndarray) else coeffs
            ).reshape(output_shape)
        else:
            return np.tensordot(op, coeffs, coeffs.ndim)

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
            self._jr_constraint_L_matrix_cache = Vt.T @ np.diag(S) @ Vt
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


    def build_m_ind_to_E_df(self):
        if self.m_ind_to_E_df is not None:
            return
        shape = (self.basis.index_length, self.basis.index_length)

        def matvec(m):
            E_ind, _ = self.calculate_ind_coeffs(m)
            return E_ind[1]

        # Always build the dense operator for internal use
        self.m_ind_to_E_df = np.array([matvec(v) for v in np.eye(shape[1])]).T

    def evolve_m_ind(self, m_ind, dt, E_coeffs_noind, steady_state_m_ind=None):
        if self.m_ind_to_E_df is None:
            self.build_m_ind_to_E_df()
        op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df
        b = self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
        if self.integrator == "euler":
            return m_ind + dt * (op @ m_ind + b)
        elif self.integrator == "exponential":
            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            diff = m_ind - steady_state_m_ind
            return (expm(dt * op) @ diff) + steady_state_m_ind
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

    def steady_state_m_ind(self, E_coeffs_noind):
        if self.m_ind_to_E_df is None:
            self.build_m_ind_to_E_df()
        op = self.m_ind_to_E_df
        b = -E_coeffs_noind[1]
        return np.linalg.solve(op, b)