"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required for
simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any

import numpy as np
import xarray as xr
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import expm

from pynamit.math.constants import mu0
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.tensor_operations import tensor_pinv
from pynamit.math.least_squares_solver import LeastSquaresSolver, TensorChain
from pynamit.simulation.geometry import Geometry
from pynamit.spherical_harmonics.sh_basis import SHBasis

logger = logging.getLogger(__name__)


class State:
    """Manages the ionospheric electrodynamic state and associated operators.

    This class encapsulates the physical state (e.g., potentials, currents),
    handles the construction of all necessary numerical operators based on the
    provided geometry and settings, and orchestrates the time evolution of
    the system. It uses an Geometry object to manage the
    underlying grid and mappings.
    """

    def __init__(
        self,
        basis: SHBasis,
        mainfield: Any,
        cs_basis: SHBasis,
        settings: Any,
        PFAC_matrix: Optional[xr.DataArray] = None,
    ) -> None:
        """Initialize the State object."""
        self.basis = basis

        # 1. Configure state from settings
        self._init_settings(settings)

        # 2. Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(basis, cs_basis, mainfield, settings)
        # Allow pre-computed PFAC matrix to override the one built by geometry
        if PFAC_matrix is not None:
            self.geometry._T_to_Ve = PFAC_matrix

        # 3. Invalidate caches and precompute constant physical operators
        self._invalidate_caches()
        self._init_precomputed_operators()

        # 4. Initialize state variables
        self.u = self.Br = self.jr = self.etaP = self.etaH = None
        self._m_imp_solver: Optional[LeastSquaresSolver] = None

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        self.solver_type = getattr(settings, "least_squares_solver", "normal")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.connect_hemispheres = bool(settings.connect_hemispheres)

    def _init_precomputed_operators(self) -> None:
        """Precompute operators that are independent of conductance."""
        self.G_helmholtz_pinv = tensor_pinv(self.geometry.basis_evaluator.G_helmholtz, n_leading_flattened=2)
        self.m_ind_to_Br = -(self.RI ** 2) * self.basis.laplacian(self.RI)
        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI

        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = (1.0 / self.RI) * self.geometry.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs

        # Operator for mapping velocity field `u` to E-field
        G_u_to_uxB_grid = np.einsum("ijk,jklm->iklm", self.geometry.bu, self.geometry.basis_evaluator.G_helmholtz, optimize=True)
        self.u_coeffs_to_E_coeffs = self.geometry.basis_evaluator.least_squares_solution_helmholtz(G_u_to_uxB_grid)

    def _invalidate_caches(self) -> None:
        """Invalidate all cached properties that depend on conductance."""
        self._M_total_on_grid = None
        self._G_m_imp_to_JS = None
        self._G_m_ind_to_JS = None
        self._m_ind_to_E_coeffs = None
        self._m_imp_to_E_coeffs = None
        self._Br_to_E_coeffs = None
        self._E_map_constraint_operator = None
        self._m_ind_to_E_df_matrix = None
        self._m_imp_solver = None

    # ----- Cached Physical Properties (dependent on conductance) -----

    @property
    def M_total_on_grid(self) -> np.ndarray:
        """Total conductance tensor evaluated on the grid."""
        if self._M_total_on_grid is None:
            if self.etaP is None or self.etaH is None:
                raise RuntimeError("Conductance must be set before accessing conductance-dependent properties.")
            eta_stacked = np.stack([self.etaP.coeffs, self.etaH.coeffs], axis=0)
            G_eta = self.geometry.basis_evaluator_zero_added.G
            b_stacked = np.stack([self.geometry.bP, self.geometry.bH], axis=0)
            self._M_total_on_grid = np.einsum("sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked, optimize=True)
        return self._M_total_on_grid

    # ----- G operators mapping to sheet current (JS) -----

    @property
    def G_m_imp_to_JS(self) -> np.ndarray:
        """Operator mapping imposed potential coeffs to sheet current on grid."""
        if self._G_m_imp_to_JS is None:
            G_T_to_JS = -1.0 / self.RI * self.geometry.basis_evaluator.G_grad * (self.RI / mu0)
            self._G_m_imp_to_JS = G_T_to_JS + np.tensordot(self.G_Ve_to_JS, self.geometry.T_to_Ve.values, axes=([2], [0]))
        return self._G_m_imp_to_JS

    @property
    def G_m_ind_to_JS(self) -> np.ndarray:
        """Operator mapping induced potential coeffs to sheet current on grid."""
        if self._G_m_ind_to_JS is None:
            G = self.G_Ve_to_JS.copy()
            if self.RM is not None:
                br_shift = self.basis.radial_shift_Ve(self.RM, self.RI)
                vi_shift = self.basis.radial_shift_Vi(self.RI, self.RM)
                den = 1.0 - br_shift * vi_shift
                self.G_Br_to_JS = self.G_Ve_to_JS * (-br_shift / den / self.m_ind_to_Br)
                G *= (1.0 + (br_shift * vi_shift / den))
            self._G_m_ind_to_JS = G
        return self._G_m_ind_to_JS

    def _create_E_coeffs_operator(self, G_X_to_JS: Optional[np.ndarray]) -> Optional[TensorChain]:
        """Factory for creating TensorChain operators that map potential coeffs to E-field coeffs."""
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
        """Matrix-free operator for the interhemispheric E-field constraint."""
        if self._E_map_constraint_operator is None:
            inner_chain = self.m_imp_to_E_coeffs
            outer_tensor = self.geometry.E_coeffs_to_E_apex_ll_diff
            if inner_chain is not None and outer_tensor is not None:
                self._E_map_constraint_operator = TensorChain(
                    component_tensors=[outer_tensor] + inner_chain.component_tensors,
                    einsum_string_dense="ticm,cmpg,pqg,qgl->til",
                    einsum_string_matvec="ticm,cmpg,pqg,qgl,l->ti",
                    einsum_string_rmatvec="ti,ticm,cmpg,pqg,qgl->l",
                    output_shape=(2, int(np.sum(self.geometry.ll_mask))),
                    input_shape=inner_chain.input_shape,
                )
        return self._E_map_constraint_operator

    # ----- Solver Setup and Execution -----

    @property
    def m_imp_solver(self) -> LeastSquaresSolver:
        """The least-squares solver instance for the imposed potential."""
        if self._m_imp_solver is None:
            A_list = []
            data_shapes = []

            # Term 1: Radial current (jr) constraint
            A_jr = self.geometry.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))
            A_list.append(A_jr)
            data_shapes.append(A_jr.shape[:-1])

            # Term 2: Interhemispheric E-field constraint
            if self.connect_hemispheres:
                A_E = self.E_map_constraint_operator.with_scaling(self.ih_constraint_scaling)
                A_list.append(A_E)
                data_shapes.append(A_E.output_shape)

            # Setup regularization
            reg_ops = []
            if self.m_imp_regularization_lambda > 0:
                n = self.basis.index_length
                identity_op = LinearOperator((n, n), matvec=lambda x: x, rmatvec=lambda x: x, dtype=np.float64)
                reg_ops.append({"weight": self.m_imp_regularization_lambda, "matrix": identity_op})

            self._m_imp_solver = LeastSquaresSolver(
                A=A_list,
                solution_shape=self.basis.index_length,
                data_shapes=data_shapes,
                sqrt_weights=[None] * len(A_list), # No explicit weights
                regularization_weights=[r["weight"] for r in reg_ops],
                regularization_matrices=[r["matrix"] for r in reg_ops],
                solver=self.solver_type,
                preconditioner=self.preconditioner,
            )
        return self._m_imp_solver

    def _solve_for_m_imp(self, jr_coeffs: Optional[np.ndarray], E_direct_coeffs: np.ndarray) -> np.ndarray:
        """Solves for the imposed potential coefficients `m_imp`."""
        rhs_B = []

        # Term 1: RHS for jr constraint
        b_jr = np.dot(self.geometry.jr_coeffs_to_j_apex, jr_coeffs) if jr_coeffs is not None else None
        rhs_B.append(b_jr)

        # Term 2: RHS for E-field constraint
        if self.connect_hemispheres:
            b_E = -np.einsum("cikl,kl->ci", self.geometry.E_coeffs_to_E_apex_ll_diff, E_direct_coeffs).flatten() * self.ih_constraint_scaling if E_direct_coeffs is not None else None
            rhs_B.append(b_E)

        m_imp = self.m_imp_solver.solve(rhs_B)
        return m_imp if m_imp is not None else np.zeros(self.basis.index_length)

    def _solve_for_m_imp_adjoint(self, grad_m_imp: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Performs the adjoint solve for `m_imp`."""
        grad_b_list = self.m_imp_solver.solve_adjoint(grad_m_imp)

        grad_jr = grad_E = None
        term_idx = 0

        # Term 1: Gradient contribution from jr constraint
        if self.jr is not None:
            grad_jr = np.dot(self.geometry.jr_coeffs_to_j_apex.T, grad_b_list[term_idx])
        term_idx += 1

        # Term 2: Gradient contribution from E-field constraint
        if self.connect_hemispheres:
            A_E_shape = self.m_imp_solver.A[term_idx].output_shape
            grad_b_E = grad_b_list[term_idx].reshape(A_E_shape) / self.ih_constraint_scaling
            grad_E = -np.einsum("ci,cikl->kl", grad_b_E.conj(), self.geometry.E_coeffs_to_E_apex_ll_diff.conj()).conj()
            term_idx += 1

        return grad_jr, grad_E

    # ----- State Update and Evolution -----

    def update(self, input_timeseries: Any, time: float, interpolation: bool = False) -> None:
        """Updates the state variables from an input timeseries at a given time."""
        conductance_updated = False
        for key, dataset in input_timeseries.datasets.items():
            updated_input = input_timeseries.get_entry_if_changed(key, time, interpolation=interpolation)
            if updated_input is None:
                continue

            storage_base = input_timeseries.storage_bases.get(key)
            if key == "conductance":
                conductance_updated = True
                self.etaP = FieldExpansion(storage_base, coeffs=updated_input["etaP"])
                self.etaH = FieldExpansion(storage_base, coeffs=updated_input["etaH"])
            elif key == "jr":
                self.jr = FieldExpansion(storage_base, coeffs=updated_input["jr"])
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None")
                self.Br = FieldExpansion(storage_base, coeffs=updated_input["Br"])
            elif key == "u":
                self.u = FieldExpansion(storage_base, coeffs=updated_input["u"].reshape((2, -1)))

        if conductance_updated:
            self._invalidate_caches()
            if self._m_imp_solver is not None:
                logger.info("Conductance updated: invalidating caches and solver.")

    def _apply_operator(self, op: Any, coeffs: Any, output_shape: Tuple[int, ...]) -> np.ndarray:
        """Apply a numerical operator to a set of coefficients."""
        if op is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return np.zeros(output_shape)

        coeffs_arr = np.asarray(coeffs)
        flat_coeffs = coeffs_arr.flatten()

        if isinstance(op, (TensorChain, LinearOperator)):
            linop = op if isinstance(op, LinearOperator) else op.as_linear_operator()
            return linop.matvec(flat_coeffs).reshape(output_shape)

        op_arr = np.ascontiguousarray(op)
        res = np.tensordot(op_arr, coeffs_arr, axes=coeffs_arr.ndim)
        return res.reshape(output_shape)

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field and m_imp coeffs without induction."""
        E_shape = (2, self.basis.index_length)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, getattr(self.u, 'coeffs', 0), E_shape)
        if self.Br is not None:
            E_direct += self._apply_operator(self.Br_to_E_coeffs, self.Br.coeffs, E_shape)

        m_imp = self._solve_for_m_imp(getattr(self.jr, 'coeffs', None), E_direct)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct + E_imp, m_imp

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field and m_imp coeffs from induction."""
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, m_ind, E_shape)
        m_imp_ind = self._solve_for_m_imp(None, E_direct_ind)
        E_imp_ind = self._apply_operator(self.m_imp_to_E_coeffs, m_imp_ind, E_shape)
        return E_direct_ind + E_imp_ind, m_imp_ind

    @property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix operator for m_ind -> E_df (divergence-free E-field)."""
        if self._m_ind_to_E_df_matrix is None:
            self._build_m_ind_to_E_df_matrix()
        return self._m_ind_to_E_df_matrix

    def _build_m_ind_to_E_df_matrix(self) -> None:
        """Construct the dense induction operator by probing with basis vectors."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.basis.index_length
        identity = np.eye(n)
        # Each column of the matrix is the result of applying the operator to a basis vector
        E_df_columns = [self.calculate_ind_coeffs(v)[0][1] for v in identity]
        self._m_ind_to_E_df_matrix = np.array(E_df_columns).T
        logger.info("Dense induction operator built.")

    def evolve_m_ind(self, m_ind: np.ndarray, dt: float, E_coeffs_noind: np.ndarray, steady_state_m_ind: Optional[np.ndarray] = None) -> np.ndarray:
        """Evolve the induced potential `m_ind` forward by a timestep `dt`."""
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

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        op = self.m_ind_to_E_df_matrix
        b = -E_coeffs_noind[1]
        return np.linalg.solve(op, b)