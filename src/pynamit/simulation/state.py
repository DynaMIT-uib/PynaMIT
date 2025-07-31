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
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.spherical_harmonics.sh_basis import SHBasis
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres
from scipy.linalg import expm


class State(object):
    """Class for managing the electrodynamic state of the ionosphere.

    Manages the ionospheric electrodynamic state, including the model
    parameters and the relationships between the physical quantities. It
    supports both dense-matrix and matrix-free (iterative) modes for
    computation.

    Attributes
    ----------
    basis : SHBasis
        The spherical harmonic basis for the main potential fields.
    mainfield : Mainfield
        The main background magnetic field model.
    grid : Grid
        The computational grid for all grid-based calculations.
    matrix_free : bool
        If True, operators are matrix-free `LinearOperator` objects.
        If False, operators are dense `numpy.ndarray` objects.
    matrix_weights : bool
        If True, constraints are applied using matrix weights.
        If False, constraints are applied by augmenting the operator
        matrix.
    ... and other physical and computational attributes ...
    """

    def __init__(self, basis, mainfield, cs_basis, settings, PFAC_matrix=None):
        """Initialize the ionospheric state.

        Parameters
        ----------
        basis : SHBasis
            The spherical harmonic basis for potential fields.
        mainfield : Mainfield
            The main background magnetic field model.
        cs_basis : object
            A basis object providing grid coordinates (`arr_theta`,
            `arr_phi`).
        settings : object
            Configuration object containing model parameters. Expected
            attributes include `RI`, `RM`, `latitude_boundary`,
            `matrix_free`, etc.
        PFAC_matrix : array-like, optional
            A pre-computed matrix mapping toroidal to poloidal potential
            coefficients for poloidal-field-aligned currents (PFACs).
        """
        # Configuration from settings
        self.matrix_free = getattr(settings, "matrix_free", False)
        self.matrix_weights = getattr(settings, "matrix_weights", False)
        self.solver_type = getattr(settings, "least_squares_solver", "svd")
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)

        # Physical and model parameters.
        self.basis = basis
        self.mainfield = mainfield
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.eta_0 = getattr(settings, "eta_0", 0)

        if PFAC_matrix is not None:
            self._T_to_Ve = PFAC_matrix

        # Initialize physical fields to None.
        self.u, self.Br, self.jr, self.etaP, self.etaH = None, None, None, None, None

        # Grid-related objects.
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.basis_evaluator_zero_added = BasisEvaluator(
            SHBasis(settings.Nmax, settings.Mmax, Nmin=0), self.grid
        )
        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)
        self.G_helmholtz_pinv = tensor_pinv(
            self.basis_evaluator.G_helmholtz, n_leading_flattened=2
        )

        # Conjugate point objects.
        self.cp_grid, self.cp_basis_evaluator, self.cp_b_evaluator = None, None, None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)

        # Fundamental coefficient and grid operators.
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)
        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1 / self.RI
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        self.G_Ve_to_JS = 1 / self.RI * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs

        # Magnetic Field Tensors (used for resistance).
        self._b_etaP, self._b_etaH, self._b_eta0, self._b_u = None, None, None, None

        # Solver cache.
        self._m_imp_solver = None

        # Final setup.
        self.initialize_constraints()
        self._build_u_coeffs_to_E_coeffs()
        self._invalidate_caches()

    def _invalidate_caches(self):
        """Invalidate all cached data that depends on conductance."""
        self._m_ind_to_E_coeffs = None
        self._m_imp_to_E_coeffs = None
        self._Br_to_E_coeffs = None
        self.m_ind_to_E_df = None
        self._m_imp_solver = None

        if hasattr(self, "_M_total_on_grid"):
            del self._M_total_on_grid
        if hasattr(self, "_E_map_constraint_operator"):
            del self._E_map_constraint_operator

    @property
    def G_m_imp_to_JS(self):
        """Operator from m_imp to gridded sheet current."""
        if not hasattr(self, "_G_m_imp_to_JS"):
            T_to_J_cf_coeffs = self.RI / mu0
            G_T_to_JS = -1 / self.RI * self.basis_evaluator.G_grad * T_to_J_cf_coeffs
            self._G_m_imp_to_JS = G_T_to_JS + np.tensordot(
                self.G_Ve_to_JS, self.T_to_Ve.values, axes=([2], [0])
            )
        return self._G_m_imp_to_JS

    @property
    def G_m_ind_to_JS(self):
        """Operator from m_ind to gridded sheet current."""
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
    def b_etaP(self):
        """Main field's contribution to the Pedersen tensor."""
        if self._b_etaP is None:
            b_th, b_ph, b_r = self.b_evaluator.btheta, self.b_evaluator.bphi, self.b_evaluator.br
            self._b_etaP = np.array(
                [[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]]
            )
        return self._b_etaP

    @property
    def b_etaH(self):
        """Main field's contribution to the Hall tensor."""
        if self._b_etaH is None:
            br = self.b_evaluator.br
            self._b_etaH = np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])
        return self._b_etaH

    @property
    def b_eta0(self):
        """Main field's contribution from the parallel tensor."""
        if self._b_eta0 is None:
            b_th, b_ph = self.b_evaluator.btheta, self.b_evaluator.bphi
            self._b_eta0 = np.array([[b_th**2, b_th * b_ph], [b_th * b_ph, b_ph**2]])
        return self._b_eta0

    @property
    def b_u(self):
        """Main field's contribution to the u x B electric field."""
        if self._b_u is None:
            Br = self.b_evaluator.Br
            self._b_u = -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])
        return self._b_u

    @property
    def T_to_Ve(self):
        """Matrix mapping toroidal coeffs to poloidal shielding coeffs.

        The toroidal field represents the radial part of the FACs, and
        the poloidal field is the field that shields the region under
        the ionosphere from the effect of the FACs, by negating the
        Biot-Savart integral of the horizontal part of the FACs above
        the ionosphere.

        Based on Engels and Olsen (1998), in particular the method in
        equation (13).

        Returns
        -------
        array
            Matrix that maps coefficients of a toroidal field to
            coefficients of a poloidal field that shields the region
            under the ionosphere from the poloidal field of inclined
        """
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
                        "All FAC integration steps must be inside the magnetospheric"
                        " boundary (RM)."
                    )
                JS_rk_to_Ve_rk = tensor_pinv(self.G_Ve_to_JS, n_leading_flattened=2, rtol=0)
                for i, rk in enumerate(rks):
                    print(
                        "Calculating matrix for poloidal field of inclined FACs. "
                        f"Progress: {i + 1}/{rks.size}",
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
        """Return terms for the m_imp least squares problem."""
        terms = []
        # Term 1: Field-aligned current constraint (jr)
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
        # Term 2: Interhemispheric E-field constraint
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
                A_E = self._get_or_create_E_map_constraint_operator() * self.ih_constraint_scaling
                terms.append(
                    {
                        "A": A_E,
                        "data_shape": (A_E.shape[0],),
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
                                (grad_b.reshape(2, -1) / self.ih_constraint_scaling).conj(),
                                self.E_coeffs_to_E_apex_ll_diff.conj(),
                            ).conj()
                        },
                    }
                )
        return terms

    @property
    def m_imp_solver(self):
        """Least-squares solver for the m_imp."""
        if self._m_imp_solver is None:
            terms = self._get_m_imp_solver_terms()

            reg_weights = []
            reg_matrices = []

            # If lambda is set, add the Tikhonov regularization term
            # This term is to improve the convergence of the m_imp
            # solution, as the m_imp problem is often ill-posed.
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
            )
        return self._m_imp_solver

    def _solve_for_m_imp(self, jr_coeffs, E_direct_coeffs):
        """Calculate m_imp by solving the least-squares problem."""
        terms = self._get_m_imp_solver_terms()
        rhs_B = [t["get_b"](jr_coeffs, E_direct_coeffs) for t in terms]
        m_imp = self.m_imp_solver.solve(rhs_B)
        return m_imp if m_imp is not None else np.zeros(self.basis.index_length)

    def _solve_for_m_imp_adjoint(self, grad_m_imp):
        """Calculate the adjoint of the m_imp problem."""
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
        """The 2x2 resistance tensor on each grid point.

        This method constructs the full, robust H_eff tensor by
        combining the Pedersen, Hall, and parallel resistivity
        (regularization) terms.
        """
        if not hasattr(self, "_M_total_on_grid"):
            if (self.etaP is None) or (self.etaH is None):
                raise RuntimeError("Conductance must be set before use.")

            # Perpendicular resistivity tensor.
            eta_stacked_coeffs = np.stack([self.etaP.coeffs, self.etaH.coeffs], axis=0)
            G_eta = self.basis_evaluator_zero_added.G
            b_stacked = np.stack([self.b_etaP, self.b_etaH], axis=0)
            M_pynamit = np.einsum(
                "sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked_coeffs, optimize=True
            )

            # Correction term from the parallel resistivity.
            M_correction = self.eta_0 * self.b_eta0

            # Add the correction to get the final tensor.
            self._M_total_on_grid = M_pynamit + M_correction

        return self._M_total_on_grid

    def _create_E_coeffs_operator(self, G_X_to_JS):
        """Create E-field operators (dense or matrix-free)."""
        if G_X_to_JS is None:
            return None
        if self.matrix_free:
            return self._create_E_coeffs_linear_operator(G_X_to_JS)
        else:
            return self._create_E_coeffs_dense_operator(G_X_to_JS)

    def _create_E_coeffs_dense_operator(self, G_X_to_JS):
        """Create a dense matrix for the E-field transformation."""
        return np.einsum(
            "cmik,ijk,jk...->cm...",
            self.G_helmholtz_pinv,
            self.M_total_on_grid,
            G_X_to_JS,
            optimize=True,
        )

    def _create_E_coeffs_linear_operator(self, G_X_to_JS):
        """Create an LinearOperator for the E-field transformation."""
        shape_in = G_X_to_JS.shape[2:]
        shape_out = (2, self.basis.index_length)
        shape = (np.prod(shape_out), np.prod(shape_in))
        M_total, G_helm_pinv = self.M_total_on_grid, self.G_helmholtz_pinv

        def matvec(x_coeffs_flat):
            x_coeffs = x_coeffs_flat.reshape(shape_in)
            js_on_grid = np.tensordot(
                G_X_to_JS,
                x_coeffs,
                axes=(tuple(range(2, G_X_to_JS.ndim)), tuple(range(x_coeffs.ndim))),
            )
            j_div_free_on_grid = np.einsum("ijk,jk->ik", M_total, js_on_grid, optimize=True)
            E_coeffs = np.einsum("cmik,ik->cm", G_helm_pinv, j_div_free_on_grid, optimize=True)
            return E_coeffs.flatten()

        def rmatvec(grad_E_coeffs_flat):
            grad_E_coeffs = grad_E_coeffs_flat.reshape(shape_out)
            grad_j_div_free = np.einsum(
                "cmik,cm->ik", G_helm_pinv.conj(), grad_E_coeffs, optimize=True
            )
            grad_js = np.einsum("ijk,ik->jk", M_total.conj(), grad_j_div_free, optimize=True)
            grad_x_coeffs = np.tensordot(grad_js, G_X_to_JS.conj(), axes=([0, 1], [0, 1]))
            return grad_x_coeffs.flatten()

        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    @property
    def m_ind_to_E_coeffs(self):
        """Operator mapping m_ind to E-field coeffs."""
        if self._m_ind_to_E_coeffs is None:
            self._m_ind_to_E_coeffs = self._create_E_coeffs_operator(self.G_m_ind_to_JS)
        return self._m_ind_to_E_coeffs

    @property
    def m_imp_to_E_coeffs(self):
        """Operator mapping m_imp to E-field coeffs."""
        if self._m_imp_to_E_coeffs is None:
            self._m_imp_to_E_coeffs = self._create_E_coeffs_operator(self.G_m_imp_to_JS)
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self):
        """Operator mapping Br to E-field coeffs."""
        if self._Br_to_E_coeffs is None:
            self._Br_to_E_coeffs = self._create_E_coeffs_operator(
                getattr(self, "G_Br_to_JS", None)
            )
        return self._Br_to_E_coeffs

    def initialize_constraints(self):
        """Initialize geometric constraint operators."""
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
        """Update the state with input data for a given time."""
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
        """Operator mapping u to E-field coeffs."""
        G_u_to_uxB_grid = np.einsum(
            "ijk, jklm -> iklm", self.b_u, self.basis_evaluator.G_helmholtz, optimize=True
        )
        dense_op = self.basis_evaluator.least_squares_solution_helmholtz(G_u_to_uxB_grid)
        if self.matrix_free:

            def matvec(u_coeffs_flat):
                return np.einsum(
                    "iklm, lm -> ik", dense_op, u_coeffs_flat.reshape(2, -1), optimize=True
                ).flatten()

            def rmatvec(grad_E_coeffs_flat):
                return np.einsum(
                    "iklm, ik -> lm",
                    dense_op.conj(),
                    grad_E_coeffs_flat.reshape(2, -1),
                    optimize=True,
                ).flatten()

            shape_in_out = (2, self.basis.index_length)
            shape = (np.prod(shape_in_out), np.prod(shape_in_out))
            self.u_coeffs_to_E_coeffs = LinearOperator(
                shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
            )
        else:
            self.u_coeffs_to_E_coeffs = dense_op

    def _apply_operator(self, op, coeffs, output_shape):
        """Apply a dense or matrix-free operator to coefficients."""
        if op is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return np.zeros(output_shape)
        if isinstance(op, LinearOperator):
            return op.matvec(
                coeffs.flatten() if isinstance(coeffs, np.ndarray) else coeffs
            ).reshape(output_shape)
        else:
            return np.tensordot(op, coeffs, coeffs.ndim)

    def calculate_noind_coeffs(self):
        """Calculate E-field and m_imp coeffs without induction."""
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
        """Calculate E-field and m_imp coeffs from induction."""
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, m_ind, E_shape)
        m_imp_ind = self._solve_for_m_imp(None, E_direct_ind)
        E_imp_ind = self._apply_operator(self.m_imp_to_E_coeffs, m_imp_ind, E_shape)
        return E_direct_ind + E_imp_ind, m_imp_ind

    @property
    def jr_constraint_L_matrix(self):
        """The weighting matrix L for the jr constraint."""
        if not hasattr(self, "_jr_constraint_L_matrix_cache"):
            H_jr = self.jr_coeffs_to_j_apex
            _, S, Vt = np.linalg.svd(H_jr, full_matrices=False)
            self._jr_constraint_L_matrix_cache = Vt.T @ np.diag(S) @ Vt
        return self._jr_constraint_L_matrix_cache

    @property
    def E_constraint_L_matrix(self):
        """The weighting matrix L for the E-field constraint."""
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
        """Operator for the interhemispheric E-field constraint."""
        if hasattr(self, "_E_map_constraint_operator"):
            return self._E_map_constraint_operator
        if not self.matrix_free:
            op = np.tensordot(
                self.E_coeffs_to_E_apex_ll_diff, self.m_imp_to_E_coeffs, axes=([2, 3], [0, 1])
            )
            n_ll = np.sum(self.ll_mask)
            self._E_map_constraint_operator = op.reshape(2 * n_ll, self.basis.index_length)
        else:
            n_ll = np.sum(self.ll_mask)
            n_c_imp = self.basis.index_length
            shape_out = (2, n_ll)
            shape_in = (n_c_imp,)
            shape = (np.prod(shape_out), np.prod(shape_in))

            def matvec(m_coeffs_in):
                m_coeffs = m_coeffs_in.flatten()
                E_coeffs = self._apply_operator(
                    self.m_imp_to_E_coeffs, m_coeffs, (2, self.basis.index_length)
                )
                E_apex_diff = np.einsum(
                    "cikl,kl->ci", self.E_coeffs_to_E_apex_ll_diff, E_coeffs, optimize=True
                )
                return E_apex_diff.flatten()

            def rmatvec(grad_E_apex_flat):
                grad_E_apex = grad_E_apex_flat.reshape(shape_out)
                grad_E_coeffs = np.einsum(
                    "cikl,ci->kl",
                    self.E_coeffs_to_E_apex_ll_diff.conj(),
                    grad_E_apex,
                    optimize=True,
                )
                op = self.m_imp_to_E_coeffs
                grad_m_coeffs_flat = op.rmatvec(grad_E_coeffs.flatten())
                return grad_m_coeffs_flat

            self._E_map_constraint_operator = LinearOperator(
                shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
            )
        return self._E_map_constraint_operator

    def build_m_ind_to_E_df(self):
        """Operator d(E_phi)/d(m_ind) for time evolution."""
        if self.m_ind_to_E_df is not None:
            return
        shape = (self.basis.index_length, self.basis.index_length)

        def matvec(m):
            E_ind, _ = self.calculate_ind_coeffs(m)
            return E_ind[1]

        if self.matrix_free:

            def rmatvec(grad_out):
                grad_E_phi = grad_out.flatten()
                grad_E = np.zeros((2, shape[0]), dtype=grad_E_phi.dtype)
                grad_E[1] = grad_E_phi
                total_grad_E_direct_ind = grad_E.copy()
                op_m_imp_E = self.m_imp_to_E_coeffs
                grad_m_imp_ind_flat = op_m_imp_E.rmatvec(grad_E.flatten())
                _, grad_E_direct_ind_from_m_imp = self._solve_for_m_imp_adjoint(
                    grad_m_imp_ind_flat
                )
                if grad_E_direct_ind_from_m_imp is not None:
                    total_grad_E_direct_ind += grad_E_direct_ind_from_m_imp
                op_m_ind_E = self.m_ind_to_E_coeffs
                grad_m_ind_flat = op_m_ind_E.rmatvec(total_grad_E_direct_ind.flatten())
                return grad_m_ind_flat

            self.m_ind_to_E_df = LinearOperator(
                shape=shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
            )
        else:
            self.m_ind_to_E_df = np.array([matvec(v) for v in np.eye(shape[1])]).T

    def evolve_m_ind(self, m_ind, dt, E_coeffs_noind, steady_state_m_ind=None):
        """Evolve induced magnetic field coefficients one time step."""
        if self.m_ind_to_E_df is None:
            self.build_m_ind_to_E_df()
        op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df
        b = self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
        if self.integrator == "euler":
            return m_ind + dt * (op.dot(m_ind) + b)
        elif self.integrator == "exponential":
            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            diff = m_ind - steady_state_m_ind
            if self.matrix_free:
                return expm_multiply(dt * op, diff) + steady_state_m_ind
            else:
                return (expm(dt * op) @ diff) + steady_state_m_ind
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

    def steady_state_m_ind(self, E_coeffs_noind):
        """Calculate steady-state m_ind coefficients."""
        if self.m_ind_to_E_df is None:
            self.build_m_ind_to_E_df()
        op = self.m_ind_to_E_df
        b = -E_coeffs_noind[1]
        if self.matrix_free:
            m_ind, exit_code = gmres(op, b, rtol=1e-12, atol=0)
            if exit_code != 0:
                print(f"Warning: GMRES failed with exit code {exit_code}")
            return m_ind
        else:
            return np.linalg.solve(op, b)
