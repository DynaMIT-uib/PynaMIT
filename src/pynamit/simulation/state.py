"""State module (refactored).

Contains a cleaner, more modular implementation of the original State
class. Changes made (high-level):
- clearer separation of initialization, cache invalidation, and operator
  construction
- reduced repeated attribute lookups and small refactors to make intent
  clearer
- replaced prints with logging
- type annotations added where helpful
- _apply_operator unified handling of TensorChain / LinearOperator / ndarray

Functionality is kept equivalent to the original while improving style and
modularity.  (See the companion refactored least-squares module.)
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import expm

from pynamit.math.constants import mu0
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.tensor_operations import tensor_pinv
from pynamit.math.least_squares_solver import LeastSquaresSolver, TensorChain
from pynamit.spherical_harmonics.sh_basis import SHBasis

logger = logging.getLogger(__name__)


class State:
    """Class for managing the ionospheric electrodynamic state.

    The refactor keeps the same public API but aims for clearer structure
    and fewer redundant operations.
    """

    def __init__(
        self,
        basis,
        mainfield,
        cs_basis,
        settings,
        PFAC_matrix: Optional[xr.DataArray] = None,
    ) -> None:
        self.basis = basis
        self.mainfield = mainfield

        # 1. settings, geometry, evaluators
        self._init_settings(settings)
        self._init_geom_and_evaluators(cs_basis)

        # 2. caches + precomputed operators
        self._invalidate_caches()
        self._init_precomputed_operators(PFAC_matrix)

        # 3. constraints and state variables
        self.initialize_constraints()
        self.u = self.Br = self.jr = self.etaP = self.etaH = None
        self._m_imp_solver: Optional[LeastSquaresSolver] = None

    # ----- initialization helpers -----

    def _init_settings(self, settings) -> None:
        self.matrix_weights = getattr(settings, "matrix_weights", False)
        self.solver_type = getattr(settings, "least_squares_solver", "normal")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.ih_constraint_scaling = settings.ih_constraint_scaling

    def _init_geom_and_evaluators(self, cs_basis) -> None:
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        # zero-added evaluator for conductance projection (Nmin=0)
        self.basis_evaluator_zero_added = BasisEvaluator(
            SHBasis(self.basis.Nmax, self.basis.Mmax, Nmin=0), self.grid
        )
        self.b_evaluator = FieldEvaluator(self.mainfield, self.grid, self.RI)

        # optional conjugate (conjugate-hemisphere) evaluators
        self.cp_grid = self.cp_basis_evaluator = self.cp_b_evaluator = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(self.mainfield, self.cp_grid, self.RI)

    def _init_precomputed_operators(self, PFAC_matrix) -> None:
        # precompute items that don't depend on conductance
        self.G_helmholtz_pinv = tensor_pinv(self.basis_evaluator.G_helmholtz, n_leading_flattened=2)
        self.m_ind_to_Br = -(self.RI ** 2) * self.basis.laplacian(self.RI)
        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI

        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = (1.0 / self.RI) * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs

        # PFAC -> Ve mapping (may be calculated lazily if None is provided)
        self._T_to_Ve = PFAC_matrix

        # make u->E operator (needs self.bu property which is cached)
        G_u_to_uxB_grid = np.einsum("ijk,jklm->iklm", self.bu, self.basis_evaluator.G_helmholtz, optimize=True)
        self.u_coeffs_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(G_u_to_uxB_grid)

    def _invalidate_caches(self) -> None:
        self._bP = self._bH = self._bu = None
        self._M_total_on_grid = None
        self._G_m_imp_to_JS = None
        self._G_m_ind_to_JS = None
        self._m_ind_to_E_coeffs = None
        self._m_imp_to_E_coeffs = None
        self._Br_to_E_coeffs = None
        self._E_map_constraint_operator = None
        self._m_ind_to_E_df_matrix = None
        # solver specific
        self._m_imp_solver = None

    # ----- magnetic-field geometric factors (cached properties) -----

    @property
    def bP(self) -> np.ndarray:
        if self._bP is None:
            b_th, b_ph, b_r = self.b_evaluator.btheta, self.b_evaluator.bphi, self.b_evaluator.br
            self._bP = np.array([[b_ph ** 2 + b_r ** 2, -b_th * b_ph], [-b_th * b_ph, b_th ** 2 + b_r ** 2]])
        return self._bP

    @property
    def bH(self) -> np.ndarray:
        if self._bH is None:
            br = self.b_evaluator.br
            self._bH = np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])
        return self._bH

    @property
    def bu(self) -> np.ndarray:
        if self._bu is None:
            Br = self.b_evaluator.Br
            self._bu = -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])
        return self._bu

    # ----- PFAC / T->Ve building (kept similar but cleaned) -----

    def _build_T_to_Ve(self) -> None:
        # preallocate as xarray DataArray for compatibility with later code
        n = self.basis.index_length
        self._T_to_Ve = xr.DataArray(np.zeros((n, n)), dims=("i", "j"))
        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return

        rk_steps = np.asarray(self.FAC_integration_steps)
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        if np.any(rks < self.RI):
            raise ValueError("All FAC integration steps must be outside the ionospheric boundary (RI).")
        if self.RM is not None and np.any(rks > self.RM):
            raise ValueError("All FAC integration steps must be inside the magnetospheric boundary (RM).")

        # compute JS_rk_to_Ve_rk pseudoinverse once
        JS_rk_to_Ve_rk = tensor_pinv(self.G_Ve_to_JS, n_leading_flattened=2, rtol=0)

        for i, rk in enumerate(rks):
            logger.debug("PFAC integration step %d/%d (rk=%s)", i + 1, rks.size, rk)
            theta_mapped, phi_mapped = self.mainfield.map_coords(self.RI, rk, self.grid.theta, self.grid.phi)
            mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
            rk_b_evaluator = FieldEvaluator(self.mainfield, self.grid, rk)
            mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
            mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)

            m_imp_to_jr = mapped_basis_evaluator.scaled_G(self.m_imp_to_jr)
            jr_to_JS_rk = np.array([rk_b_evaluator.Btheta / mapped_b_evaluator.Br, rk_b_evaluator.Bphi / mapped_b_evaluator.Br])
            m_imp_to_JS_rk = np.einsum("ij,jk->ijk", jr_to_JS_rk, m_imp_to_jr, optimize=True)

            Ve_rk_to_Ve = self.basis.radial_shift_Ve(rk, self.RI).reshape((-1, 1, 1))
            if self.RM is not None:
                Ve_rk_to_Ve -= (
                    self.basis.radial_shift_Ve(self.RM, self.RI) * self.basis.radial_shift_Vi(rk, self.RM)
                ).reshape((-1, 1, 1))
                factor = -1.0 / (1.0 - self.basis.radial_shift_Ve(self.RM, self.RI) * self.basis.radial_shift_Vi(self.RI, self.RM))
            else:
                factor = -1.0

            JS_rk_to_Ve = JS_rk_to_Ve_rk * Ve_rk_to_Ve
            # Ensure arrays are numpy arrays and shapes align before contracting.
            a = np.asarray(JS_rk_to_Ve)
            b = np.asarray(m_imp_to_JS_rk)
            # Expectation: contract last two axes of `a` with first two axes of `b`.
            if a.shape[-2:] != b.shape[:2]:
                raise ValueError(f"tensordot shape mismatch in _build_T_to_Ve: JS_rk_to_Ve.shape={a.shape}, m_imp_to_JS_rk.shape={b.shape}")
            # Use axes=2 to contract (last-2,last-1) of `a` with (0,1) of `b` (matches original behavior).
            self._T_to_Ve += Delta_k[i] * factor * np.tensordot(a, b, axes=2)

    @property
    def T_to_Ve(self) -> xr.DataArray:
        if self._T_to_Ve is None:
            self._build_T_to_Ve()
        return self._T_to_Ve

    # ----- G operators mapping to sheet current (JS) -----

    @property
    def G_m_imp_to_JS(self):
        if self._G_m_imp_to_JS is None:
            T_to_J_cf_coeffs = self.RI / mu0
            G_T_to_JS = -1.0 / self.RI * self.basis_evaluator.G_grad * T_to_J_cf_coeffs
            # contract PFAC contribution
            self._G_m_imp_to_JS = G_T_to_JS + np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0]))
        return self._G_m_imp_to_JS

    @property
    def G_m_ind_to_JS(self):
        if self._G_m_ind_to_JS is None:
            G = self.G_Ve_to_JS.copy()
            if self.RM is not None:
                br_shift = self.basis.radial_shift_Ve(self.RM, self.RI)
                vi_shift = self.basis.radial_shift_Vi(self.RI, self.RM)
                den = 1.0 - br_shift * vi_shift
                # the extra operator for Br->JS used elsewhere
                self.G_Br_to_JS = self.G_Ve_to_JS * (-1.0 / den * br_shift / self.m_ind_to_Br)
                G = G * (1.0 + (1.0 / den * br_shift * vi_shift))
            self._G_m_ind_to_JS = G
        return self._G_m_ind_to_JS

    # ----- E-field construction helpers via TensorChain -----

    def _create_E_coeffs_operator(self, G_X_to_JS: Optional[np.ndarray]) -> Optional[TensorChain]:
        if G_X_to_JS is None:
            return None
        return TensorChain(
            component_tensors=[self.G_helmholtz_pinv, self.M_total_on_grid, G_X_to_JS],
            einsum_string_dense="cmpg,pqg,qgl->cml",
            einsum_string_matvec="cmpg,pqg,qgl,l->cm",
            einsum_string_rmatvec="cm,cmpg,pqg,qgl->l",
            output_shape=(2, self.basis.index_length),
            input_shape=G_X_to_JS.shape[2:],
        )

    @property
    def m_ind_to_E_coeffs(self) -> Optional[TensorChain]:
        if self._m_ind_to_E_coeffs is None:
            self._m_ind_to_E_coeffs = self._create_E_coeffs_operator(self.G_m_ind_to_JS)
        return self._m_ind_to_E_coeffs

    @property
    def m_imp_to_E_coeffs(self) -> Optional[TensorChain]:
        if self._m_imp_to_E_coeffs is None:
            self._m_imp_to_E_coeffs = self._create_E_coeffs_operator(self.G_m_imp_to_JS)
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self) -> Optional[TensorChain]:
        if self._Br_to_E_coeffs is None:
            self._Br_to_E_coeffs = self._create_E_coeffs_operator(getattr(self, "G_Br_to_JS", None))
        return self._Br_to_E_coeffs

    @property
    def E_map_constraint_operator(self) -> Optional[TensorChain]:
        """Matrix-free operator combining imposed-potential -> E-field inner chain
        with the apex-mapping outer tensor used for the interhemispheric constraint.
        Returns None when required components are not available yet.
        """
        if self._E_map_constraint_operator is None:
            inner_chain = self.m_imp_to_E_coeffs
            outer_tensor = self.E_coeffs_to_E_apex_ll_diff
            # If either piece is missing, keep operator None (will be built later)
            if inner_chain is None or outer_tensor is None:
                self._E_map_constraint_operator = None
            else:
                self._E_map_constraint_operator = TensorChain(
                    component_tensors=[outer_tensor] + inner_chain.component_tensors,
                    einsum_string_dense="ticm,cmpg,pqg,qgl->til",
                    einsum_string_matvec="ticm,cmpg,pqg,qgl,l->ti",
                    einsum_string_rmatvec="ti,ticm,cmpg,pqg,qgl->l",
                    output_shape=(2, int(np.sum(self.ll_mask))),
                    input_shape=inner_chain.input_shape,
                )
        return self._E_map_constraint_operator

    # ----- Conductance projection -----

    @property
    def M_total_on_grid(self) -> np.ndarray:
        if self._M_total_on_grid is None:
            if self.etaP is None or self.etaH is None:
                raise RuntimeError("Conductance must be set before accessing conductance-dependent properties.")
            eta_stacked = np.stack([self.etaP.coeffs, self.etaH.coeffs], axis=0)
            G_eta = self.basis_evaluator_zero_added.G
            b_stacked = np.stack([self.bP, self.bH], axis=0)
            self._M_total_on_grid = np.einsum("sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked, optimize=True)
        return self._M_total_on_grid

    # ----- Constraint operators and L-matrices -----

    @property
    def jr_constraint_L_matrix(self):
        if not hasattr(self, "_jr_constraint_L_matrix"):
            H_jr = self.jr_coeffs_to_j_apex
            _, S, Vt = np.linalg.svd(H_jr, full_matrices=False)
            self._jr_constraint_L_matrix = Vt.T @ np.diag(S) @ Vt
        return self._jr_constraint_L_matrix

    @property
    def E_constraint_L_matrix(self):
        if not hasattr(self, "_E_constraint_L_matrix"):
            L_E = None
            if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
                H_E = self.E_coeffs_to_E_apex_ll_diff
                H_E_2D = H_E.reshape((np.prod(H_E.shape[:2]), np.prod(H_E.shape[2:])))
                _, S, Vt = np.linalg.svd(H_E_2D, full_matrices=False)
                L_E_2D = Vt.T @ np.diag(S) @ Vt
                L_E = L_E_2D.reshape(H_E.shape[2:] + H_E.shape[2:])
            self._E_constraint_L_matrix = L_E
        return self._E_constraint_L_matrix

    # ----- Solver assembly / helpers -----

    def _get_m_imp_solver_terms(self):
        terms = []
        # radial current term
        if self.matrix_weights:
            A_jr = np.diag(self.m_imp_to_jr)
            sqrt_W_jr = self.jr_constraint_L_matrix
            get_b = lambda jr, E: jr
            get_grad_contrib = lambda grad_b: {"grad_jr": grad_b}
            data_shape = A_jr.shape[:-1]
        else:
            A_jr = self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))
            sqrt_W_jr = None
            get_b = lambda jr, E: (np.dot(self.jr_coeffs_to_j_apex, jr) if jr is not None else None)
            get_grad_contrib = lambda grad_b: {"grad_jr": np.dot(self.jr_coeffs_to_j_apex.T, grad_b)}
            data_shape = A_jr.shape[:-1]

        terms.append({
            "A": A_jr,
            "data_shape": data_shape,
            "sqrt_W": sqrt_W_jr,
            "get_b": get_b,
            "get_grad_contrib": get_grad_contrib,
        })

        # interhemispheric E-field constraint
        if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
            if self.matrix_weights:
                A_E = self.m_imp_to_E_coeffs.to_dense()
                sqrt_W_E = self.E_constraint_L_matrix * self.ih_constraint_scaling
                get_b = lambda jr, E: -E
                get_grad_contrib = lambda grad_b: {"grad_E": -grad_b.reshape(2, self.basis.index_length)}
            else:
                A_E = self.E_map_constraint_operator.with_scaling(self.ih_constraint_scaling)
                sqrt_W_E = None
                get_b = lambda jr, E: (
                    -np.einsum("cikl,kl->ci", self.E_coeffs_to_E_apex_ll_diff, E).flatten() * self.ih_constraint_scaling
                    if E is not None
                    else None
                )
                get_grad_contrib = lambda grad_b: {
                    "grad_E": -np.einsum("ci,cikl->kl", (grad_b.reshape(A_E.output_shape) / self.ih_constraint_scaling).conj(), self.E_coeffs_to_E_apex_ll_diff.conj()).conj()
                }

            terms.append({
                "A": A_E,
                "data_shape": A_E.output_shape if hasattr(A_E, "output_shape") else A_E.shape[:-1],
                "sqrt_W": sqrt_W_E,
                "get_b": get_b,
                "get_grad_contrib": get_grad_contrib,
            })
        return terms

    @property
    def m_imp_solver(self) -> LeastSquaresSolver:
        if self._m_imp_solver is None:
            terms = self._get_m_imp_solver_terms()

            reg_weights = []
            reg_matrices = []
            if self.m_imp_regularization_lambda > 0:
                n = self.basis.index_length
                identity_op = LinearOperator((n, n), matvec=lambda x: x, rmatvec=lambda x: x, dtype=np.float64)
                reg_weights.append(self.m_imp_regularization_lambda)
                reg_matrices.append(identity_op)

            self._m_imp_solver = LeastSquaresSolver(
                A=[t["A"] for t in terms],
                solution_shape=self.basis.index_length,
                data_shapes=[t["data_shape"] for t in terms],
                sqrt_weights=[t.get("sqrt_W") for t in terms],
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
        grad_jr = grad_E = None
        for i, term in enumerate(terms):
            contr = term["get_grad_contrib"](grad_b_list[i])
            if "grad_jr" in contr and self.jr is not None:
                grad_jr = contr["grad_jr"]
            if "grad_E" in contr and self.connect_hemispheres:
                grad_E = contr["grad_E"]
        return grad_jr, grad_E

    # ----- top-level public methods: update / calculate / evolve -----

    def initialize_constraints(self) -> None:
        kind = self.mainfield.kind
        if kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(self.grid.lat, self.grid.lon, (self.RI - 6371e3) * 1e-3)
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)

        self.jr_coeffs_to_j_apex = (self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G).copy()
        self.E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres and self.cp_b_evaluator is not None and self.cp_basis_evaluator is not None:
            jr_coeffs_to_j_apex_cp = (self.cp_b_evaluator.radial_to_apex.reshape((-1, 1)) * self.cp_basis_evaluator.G)
            self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]

            E_coeffs_to_E_apex = np.einsum("ijk,jklm->iklm", self.b_evaluator.horizontal_to_apex, self.basis_evaluator.G_helmholtz, optimize=True)
            E_coeffs_to_E_apex_cp = np.einsum("ijk,jklm->iklm", self.cp_b_evaluator.horizontal_to_apex, self.cp_basis_evaluator.G_helmholtz, optimize=True)
            self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray((E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask])

    def update(self, input_timeseries, time, interpolation: bool = False) -> None:
        conductance_updated = False
        for key in input_timeseries.datasets.keys():
            updated_input = input_timeseries.get_entry_if_changed(key, time, interpolation=interpolation)
            if updated_input is None:
                continue
            if key == "conductance":
                conductance_updated = True
                self.etaP = FieldExpansion(input_timeseries.storage_bases["conductance"], coeffs=updated_input["etaP"])
                self.etaH = FieldExpansion(input_timeseries.storage_bases["conductance"], coeffs=updated_input["etaH"])
            elif key == "jr":
                self.jr = FieldExpansion(input_timeseries.storage_bases["jr"], coeffs=updated_input["jr"])
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None")
                self.Br = FieldExpansion(input_timeseries.storage_bases["Br"], coeffs=updated_input["Br"])
            elif key == "u":
                self.u = FieldExpansion(input_timeseries.storage_bases["u"], coeffs=updated_input["u"].reshape((2, -1)))

        if conductance_updated:
            self._invalidate_caches()
            if self._m_imp_solver is not None:
                logger.info("Conductance updated: updating solver matrices reusing preconditioner if possible")
                new_terms = self._get_m_imp_solver_terms()
                self.m_imp_solver.update_matrices(A=[t["A"] for t in new_terms], sqrt_weights=[t.get("sqrt_W") for t in new_terms])

    def _apply_operator(self, op, coeffs, output_shape: Tuple[int, ...]):
        """Uniformly apply an operator or tensor to flatten coefficients and
        reshape into the desired output shape.
        """
        if op is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return np.zeros(output_shape)
        # TensorChain -> LinearOperator, LinearOperator -> matvec, ndarray -> tensordot
        if isinstance(op, TensorChain):
            linop = op.as_linear_operator()
            return linop.matvec(np.asarray(coeffs).flatten()).reshape(output_shape)
        if isinstance(op, LinearOperator):
            return op.matvec(np.asarray(coeffs).flatten()).reshape(output_shape)
        # assume ndarray with appropriate shape
        coeffs_arr = np.asarray(coeffs)
        return np.tensordot(op, coeffs_arr, axes=coeffs_arr.ndim)

    # ----- calculations / ODE evolution -----

    def calculate_noind_coeffs(self):
        E_shape = (2, self.basis.index_length)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, self.u.coeffs if self.u else 0, E_shape)
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
    def m_ind_to_E_df_matrix(self):
        if self._m_ind_to_E_df_matrix is None:
            self._build_m_ind_to_E_df_matrix()
        return self._m_ind_to_E_df_matrix

    def _build_m_ind_to_E_df_matrix(self) -> None:
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.basis.index_length
        identity = np.eye(n)
        # Probe operator with basis; result is cached.
        self._m_ind_to_E_df_matrix = np.array([self.calculate_ind_coeffs(v)[0][1] for v in identity]).T
        logger.info("Dense induction operator built.")

    def evolve_m_ind(self, m_ind, dt: float, E_coeffs_noind, steady_state_m_ind=None):
        op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df_matrix
        b = self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
        if self.integrator == "euler":
            return m_ind + dt * (op @ m_ind + b)
        if self.integrator == "exponential":
            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            diff = m_ind - steady_state_m_ind
            return expm(dt * op) @ diff + steady_state_m_ind
        raise ValueError(f"Unknown integrator: {self.integrator}")

    def steady_state_m_ind(self, E_coeffs_noind):
        op = self.m_ind_to_E_df_matrix
        b = -E_coeffs_noind[1]
        return np.linalg.solve(op, b)