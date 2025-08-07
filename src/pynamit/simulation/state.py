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
    computation. Constraints are applied using projection operators.

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
        self.solver_type = getattr(settings, "least_squares_solver", "normal")
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

        # Magnetic Field Tensors (used for conductivity).
        self.bP, self.bH, self.bu = None, None, None

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
    def bP_prop(self):
        """Main field's contribution to the Pedersen tensor."""
        if self.bP is None:
            b_th, b_ph, b_r = self.b_evaluator.btheta, self.b_evaluator.bphi, self.b_evaluator.br
            self.bP = np.array(
                [[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]]
            )
        return self.bP

    @property
    def bH_prop(self):
        """Main field's contribution to the Hall tensor."""
        if self.bH is None:
            br = self.b_evaluator.br
            self.bH = np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])
        return self.bH

    @property
    def bu_prop(self):
        """Main field's contribution to the u x B electric field."""
        if self.bu is None:
            Br = self.b_evaluator.Br
            self.bu = -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])
        return self.bu

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

    @property
    def jr_coeffs_to_j_apex_pinv(self):
        """Cached pseudoinverse of jr_coeffs_to_j_apex."""
        if not hasattr(self, "_jr_coeffs_to_j_apex_pinv"):
            self._jr_coeffs_to_j_apex_pinv = tensor_pinv(
                self.jr_coeffs_to_j_apex, n_leading_flattened=1
            )
        return self._jr_coeffs_to_j_apex_pinv

    @property
    def E_coeffs_to_E_apex_ll_diff_pinv(self):
        """Cached pseudoinverse of E_coeffs_to_E_apex_ll_diff."""
        if not hasattr(self, "_E_coeffs_to_E_apex_ll_diff_pinv"):
            if self.E_coeffs_to_E_apex_ll_diff is None:
                self._E_coeffs_to_E_apex_ll_diff_pinv = None
            else:
                self._E_coeffs_to_E_apex_ll_diff_pinv = tensor_pinv(
                    self.E_coeffs_to_E_apex_ll_diff, n_leading_flattened=2
                )
        return self._E_coeffs_to_E_apex_ll_diff_pinv

    @property
    def jr_projection_matrix(self):
        """Cached projection matrix for the jr constraint."""
        if not hasattr(self, "_jr_projection_matrix"):
            self._jr_projection_matrix = np.dot(
                self.jr_coeffs_to_j_apex_pinv, self.jr_coeffs_to_j_apex
            )
        return self._jr_projection_matrix

    @property
    def E_projection_matrix(self):
        """Cached projection matrix for the E-field constraint."""
        if not hasattr(self, "_E_projection_matrix"):
            if self.E_coeffs_to_E_apex_ll_diff is None:
                self._E_projection_matrix = None
            else:
                self._E_projection_matrix = np.tensordot(
                    self.E_coeffs_to_E_apex_ll_diff_pinv,
                    self.E_coeffs_to_E_apex_ll_diff,
                    axes=([2, 3], [0, 1])
                )
        return self._E_projection_matrix

    def _get_m_imp_solver_terms(self):
        """Return terms for the m_imp least squares problem."""
        terms = []
        n_coeffs = self.basis.index_length

        # Term 1: Field-aligned current (FAC) constraint (jr).
        # The constraint is that FACs are zero at the magnetic apex for
        # low-latitude points, enforcing that they close in the conjugate
        # hemisphere. This is achieved by projecting the operator and data
        # onto the subspace where this condition holds.
        Proj_jr = self.jr_projection_matrix
        # A_jr = Proj_jr @ M_jr. Since M_jr is diagonal, this is an element-wise
        # multiplication of each row of Proj_jr with the diagonal of M_jr.
        A_jr = Proj_jr * self.m_imp_to_jr

        def get_b_jr(jr, E):
            return np.dot(Proj_jr, jr) if jr is not None else None

        def get_grad_contrib_jr(grad_b):
            # Proj_jr is Hermitian, so Proj_jr.conj().T == Proj_jr
            return {"grad_jr": np.dot(Proj_jr.conj().T, grad_b)}

        terms.append({
            "A": A_jr, "data_shape": (n_coeffs,), "sqrt_W": None,
            "get_b": get_b_jr, "get_grad_contrib": get_grad_contrib_jr,
        })

        # Term 2: Interhemispheric E-field constraint.
        # This constraint enforces that the electric field at low-latitude
        # conjugate points should be equal, also handled via projection.
        if self.connect_hemispheres and self.E_coeffs_to_E_apex_ll_diff is not None:
            A_E_shape = (2 * n_coeffs, n_coeffs)

            if self.matrix_free:
                def matvec_E(m_imp_coeffs_flat):
                    E_coeffs = self.m_imp_to_E_coeffs.matvec(m_imp_coeffs_flat).reshape(2, n_coeffs)
                    return np.tensordot(self.E_projection_matrix, E_coeffs, axes=([2, 3], [0, 1])).flatten()

                def rmatvec_E(grad_b_flat):
                    grad_E_proj = np.tensordot(self.E_projection_matrix.conj(), grad_b_flat.reshape(2, n_coeffs), axes=([0, 1], [0, 1]))
                    return self.m_imp_to_E_coeffs.rmatvec(grad_E_proj.flatten())

                A_E = LinearOperator(shape=A_E_shape, matvec=matvec_E, rmatvec=rmatvec_E)
            else:
                A_E = np.tensordot(self.E_projection_matrix, self.m_imp_to_E_coeffs, axes=([2, 3], [0, 1]))

            def get_b_E(jr, E):
                if E is None: return None
                return -np.tensordot(self.E_projection_matrix, E, axes=([2, 3], [0, 1])).flatten()

            def get_grad_contrib_E(grad_b):
                grad_b_reshaped = grad_b.reshape(2, n_coeffs) / self.ih_constraint_scaling
                # Adjoint of projection is projection itself (since it's Hermitian)
                grad_E = -np.tensordot(self.E_projection_matrix.conj(), grad_b_reshaped, axes=([0, 1], [0, 1]))
                return {"grad_E": grad_E}

            terms.append({
                "A": A_E * self.ih_constraint_scaling,
                "data_shape": (2 * n_coeffs,), "sqrt_W": None,
                "get_b": lambda jr, E: get_b_E(jr, E) * self.ih_constraint_scaling if E is not None else None,
                "get_grad_contrib": get_grad_contrib_E,
            })
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
        """The 2x2 conductivity tensor on each grid point."""
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
            "ijk, jklm -> iklm", self.bu_prop, self.basis_evaluator.G_helmholtz, optimize=True
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