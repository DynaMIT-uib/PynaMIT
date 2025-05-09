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
from pynamit.math.least_squares import LeastSquares

TRIPLE_PRODUCT = False
E_MAPPING = True
J_MAPPING = True


class State(object):
    """Class for managing the electrodynamic state of the ionosphere.

    Manages the ionospheric electrodynamic state, including the model
    parameters and the relationships between the physical quantities.

    Attributes
    ----------
    basis : Basis
        Main state variable basis.
    jr_basis : Basis
        Radial current basis.
    Br_basis : Basis
        Radial magnetic field basis.
    conductance_basis : Basis
        Conductance basis.
    u_basis : Basis
        Neutral wind basis.
    grid : Grid
        Computational grid.
    mainfield : Mainfield
        Main magnetic field model.
    m_ind : FieldExpansion
        Induced magnetic field coefficients.
    m_imp : FieldExpansion
        Imposed magnetic field coefficients.
    E : FieldExpansion
        Electric field expansion (tangential).
    ... (other attributes as defined in the implementation) ...
    """

    def __init__(self, bases, mainfield, grid, settings, PFAC_matrix=None):
        """Initialize the ionospheric state.

        Parameters
        ----------
        bases : dict
            Dictionary of bases with keys:
            - 'state': for state variables
            - 'jr': for radial current
            - 'Br': for radial magnetic field
            - 'conductance': for conductivity
            - 'u': for neutral wind
        mainfield : Mainfield
            Main magnetic field model.
        grid : Grid
            Spatial grid for computations.
        settings : object
            Configuration settings containing parameters such as RI,
            latitude_boundary, ignore_PFAC, connect_hemispheres,
            FAC_integration_steps, ih_constraint_scaling, vector_jr,
            vector_Br, vector_conductance, and vector_u.
        PFAC_matrix : array-like, optional
            Pre-computed FAC poloidal field matrix.
        """
        self.basis = bases["state"]
        self.jr_basis = bases["jr"]
        self.Br_basis = bases["Br"]
        self.conductance_basis = bases["conductance"]
        self.u_basis = bases["u"]

        self.mainfield = mainfield

        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.ih_constraint_scaling = settings.ih_constraint_scaling

        self.vector_u = settings.vector_u
        self.vector_jr = settings.vector_jr
        self.vector_Br = settings.vector_Br
        self.vector_conductance = settings.vector_conductance

        self.integrator = settings.integrator

        if PFAC_matrix is not None:
            self._T_to_Ve = PFAC_matrix

        # Initialize grid-related objects.
        self.grid = grid

        # Note that these BasisEvaluator objects cannot be used for
        # inverses, as they do not include regularization and weights.
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.jr_basis_evaluator = BasisEvaluator(self.jr_basis, self.grid)
        self.Br_basis_evaluator = BasisEvaluator(self.Br_basis, self.grid)
        self.conductance_basis_evaluator = BasisEvaluator(self.conductance_basis, self.grid)
        self.u_basis_evaluator = BasisEvaluator(self.u_basis, self.grid)

        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)

        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)

        # Prepare spherical harmonic conversion factors.
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)

        if self.RM is not None:
            self.Br_RM_to_m_S = (
                -1
                / (
                    1
                    - self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(self.RI, self.RM)
                )
                * self.basis.radial_shift_Ve(self.RM, self.RI)
                / self.m_ind_to_Br
            )

        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = 1 / self.RI
        self.m_ind_to_Jeq = -self.RI / mu0 * self.basis.coeffs_to_delta_V

        # RI comes from scaling in the Ve and T potentials
        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        T_to_J_cf_coeffs = self.RI / mu0

        # 1/RI comes from scaling in the gradient theta/phi components
        self.G_Ve_to_JS = 1 / self.RI * self.basis_evaluator.G_rxgrad * Ve_to_J_df_coeffs
        self.G_T_to_JS = -1 / self.RI * self.basis_evaluator.G_grad * T_to_J_cf_coeffs

        self.G_m_ind_to_JS = self.G_Ve_to_JS
        self.G_m_imp_to_JS = self.G_T_to_JS + np.tensordot(self.G_Ve_to_JS, self.T_to_Ve.values, 1)

        # Construct the matrix elements for electric field calculations.
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

        self.bH = np.array(
            [
                [np.zeros(self.b_evaluator.grid.size), self.b_evaluator.br],
                [-self.b_evaluator.br, np.zeros(self.b_evaluator.grid.size)],
            ]
        )

        self.bu = -np.array(
            [
                [np.zeros(self.b_evaluator.grid.size), self.b_evaluator.Br],
                [-self.b_evaluator.Br, np.zeros(self.b_evaluator.grid.size)],
            ]
        )

        self.m_ind_to_bP_JS = np.einsum("ijk,jkl->ikl", self.bP, self.G_m_ind_to_JS, optimize=True)
        self.m_ind_to_bH_JS = np.einsum("ijk,jkl->ikl", self.bH, self.G_m_ind_to_JS, optimize=True)
        self.m_imp_to_bP_JS = np.einsum("ijk,jkl->ikl", self.bP, self.G_m_imp_to_JS, optimize=True)
        self.m_imp_to_bH_JS = np.einsum("ijk,jkl->ikl", self.bH, self.G_m_imp_to_JS, optimize=True)

        # Identify the high and low latitude points.
        if self.mainfield.kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif self.mainfield.kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - RE) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            print("this should not happen")

        self.G_jr_state = self.basis_evaluator.G
        self.G_jr_state_pinv = np.linalg.pinv(self.G_jr_state)

        if self.vector_jr:
            self.jr_coeffs_to_jr_coeffs_state = self.G_jr_state_pinv.dot(self.jr_basis_evaluator.G)

        if self.vector_u:
            u_coeffs_to_uxB = np.einsum(
                "ijk,jklm->iklm", self.bu, self.u_basis_evaluator.G_helmholtz, optimize=True
            )
            self.u_coeffs_to_E_coeffs_direct = (
                self.basis_evaluator.least_squares_solution_helmholtz(u_coeffs_to_uxB)
            )
        else:
            self.u_to_E_coeffs_direct = np.einsum(
                "ijkl,kml->ijml",
                self.basis_evaluator.least_squares_helmholtz.ATWA_plus_R_pinv_ATW[0].reshape(
                    (
                        self.basis_evaluator.least_squares_helmholtz.A[0].full_shapes[1]
                        + self.basis_evaluator.least_squares_helmholtz.A[0].full_shapes[0]
                    )
                ),
                self.bu,
                optimize=True,
            )

        if TRIPLE_PRODUCT and self.vector_conductance:
            self.prepare_triple_product_tensors()

        # Conductance and neutral wind should be set after state
        # initialization.
        self.neutral_wind = False
        self.conductance = False
        self.Br_input = False

        self.initialize_constraints()

    @property
    def T_to_Ve(self):
        """Matrix that maps toroidal field to poloidal shielding field.

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
                Delta_k = np.diff(rk_steps)
                rks = np.array(rk_steps[:-1] + 0.5 * Delta_k)
                if any(rks < self.RI):
                    raise ValueError(
                        "All FAC integration steps must be outside the ionospheric boundary (RI)."
                    )
                if self.RM is not None:
                    if any(rks > self.RM):
                        raise ValueError(
                            "All FAC integration steps must be inside the "
                            "magnetospheric boundary (RM)."
                        )

                JS_rk_to_Ve_rk = tensor_pinv(self.G_Ve_to_JS, n_leading_flattened=2, rtol=0)

                for i, rk in enumerate(rks):
                    print(
                        "Calculating matrix for poloidal field of "
                        f"inclined FACs. Progress: {i + 1}/{rks.size}",
                        end="\r" if i < (rks.size - 1) else "\n",
                        flush=True,
                    )
                    # Map coordinates from rk to RI.
                    theta_mapped, phi_mapped = self.mainfield.map_coords(
                        self.RI, rk, self.grid.theta, self.grid.phi
                    )
                    mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)

                    # Construct matrix that gives jr at mapped grid from
                    # toroidal coefficients, shifts to rk, and extracts
                    # horizontal current components.
                    rk_b_evaluator = FieldEvaluator(self.mainfield, self.grid, rk)
                    mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
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

                    # Construct matrix that calculates the contribution
                    # to the poloidal coefficients from the horizontal
                    # current components at rk.
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

                    # Add integration step, negative sign is to create a
                    # poloidal field that shields the region under the
                    # ionosphere from the FAC poloidal field.

                    self._T_to_Ve += (
                        Delta_k[i] * factor * np.tensordot(JS_rk_to_Ve, m_imp_to_JS_rk, 2)
                    )

        return self._T_to_Ve

    def set_model_coeffs(self, **kwargs):
        """Set model coefficients.

        Set model coefficients based on the coefficients given as
        argument. This function accepts one (and only one) set of
        coefficients.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying the coefficients to set. Valid
            values are 'm_ind' and 'm_imp'.

        Raises
        ------
        ValueError
            If more than one keyword argument is provided or if the
            keyword is invalid.
        """
        valid_kws = ["m_ind", "m_imp"]

        if len(kwargs) != 1:
            raise ValueError(
                f"Expected one and only one keyword argument, you provided {len(kwargs)}"
            )
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise ValueError("Invalid keyword. See documentation")

        if key == "m_ind":
            self.m_ind = FieldExpansion(self.basis, kwargs["m_ind"], field_type="scalar")
        elif key == "m_imp":
            self.m_imp = FieldExpansion(self.basis, kwargs["m_imp"], field_type="scalar")
        else:
            raise Exception("This should not happen")

    def initialize_constraints(self):
        """Initialize constraints."""
        jr_coeffs_to_j_apex = (
            self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        )
        self.jr_coeffs_to_j_apex = jr_coeffs_to_j_apex.copy()

        if self.connect_hemispheres:
            if self.ignore_PFAC:
                raise ValueError("Hemispheres can not be connected when ignore_PFAC is True")
            if self.mainfield.kind == "radial":
                raise ValueError("Hemispheres can not be connected with radial magnetic field")

            if J_MAPPING:
                jr_coeffs_to_j_apex_cp = (
                    self.cp_b_evaluator.radial_to_apex.reshape((-1, 1)) * self.cp_basis_evaluator.G
                )
                self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[self.ll_mask]

            if E_MAPPING:
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

    def calculate_m_imp(self, m_ind):
        """Calculate m_imp.

        Parameters
        ----------
        m_ind : array
            Coefficients for induced part of magnetic field
            perturbation.

        Returns
        -------
        array
            Coefficients for imposed part of magnetic field
            perturbation.
        """
        if self.vector_jr:
            m_imp = self.jr_coeffs_to_m_imp.dot(self.jr.coeffs)
        else:
            m_imp = self.jr_to_m_imp.dot(self.jr_on_grid)

        if self.connect_hemispheres and E_MAPPING:
            m_imp += self.m_ind_to_m_imp.dot(m_ind)
            if self.Br_input:
                m_imp += self.m_ind_to_m_imp.dot(self.Br_RM_to_m_S * self.Br.coeffs)

            if self.neutral_wind:
                if self.vector_u:
                    m_imp += np.tensordot(self.u_coeffs_to_m_imp, self.u.coeffs, 2)
                else:
                    m_imp += np.tensordot(self.u_to_m_imp, self.u_on_grid, 2)

        return m_imp

    def update_m_imp(self):
        """Impose constraints, if any.

        Leads to a contribution to m_imp from m_ind if the hemispheres
        are connected.
        """
        m_imp = self.calculate_m_imp(self.m_ind.coeffs)
        self.set_model_coeffs(m_imp=m_imp)

    def set_jr(self, jr):
        """Set radial current distribution.

        Parameters
        ----------
        jr : array-like or FieldExpansion
            Radial current density in A/m² at grid points or as vector
            coefficients
        """
        if self.vector_jr:
            self.jr = jr
        else:
            self.jr_on_grid = jr

    def set_Br(self, Br):
        """Set radial component of the magnetic field.

        Parameters
        ----------
        Br : array-like or FieldExpansion
            Radial component of Br at grid points or as vector
            coefficients
        """
        if self.RM is None:
            raise ValueError("Br can only be set if magnetospheric radius (RM) is set.")

        self.Br_input = True

        if self.vector_Br:
            self.Br = Br
        else:
            self.Br_on_grid = Br

    def set_u(self, u):
        """Set neutral wind theta and phi components.

        Parameters
        ----------
        u : array-like or FieldExpansion
            Neutral wind components.
        """
        self.neutral_wind = True

        if self.vector_u:
            self.u = u
        else:
            self.u_on_grid = u

    def set_conductance(self, etaP, etaH):
        """Set ionospheric conductance distributions.

        Parameters
        ----------
        etaP : array-like or FieldExpansion
            Pedersen conductance in S
        etaH : array-like or FieldExpansion
            Hall conductance in S
        """
        self.conductance = True

        if self.vector_conductance:
            self.etaP = etaP
            self.etaH = etaH

        else:
            etaP_on_grid = etaP
            etaH_on_grid = etaH

        if TRIPLE_PRODUCT and self.vector_conductance:
            m_ind_to_E_coeffs_direct = self.etaP_m_ind_to_E_coeffs.dot(
                self.etaP.coeffs
            ) + self.etaH_m_ind_to_E_coeffs.dot(self.etaH.coeffs)
            m_imp_to_E_coeffs = self.etaP_m_imp_to_E_coeffs.dot(
                self.etaP.coeffs
            ) + self.etaH_m_imp_to_E_coeffs.dot(self.etaH.coeffs)

        else:
            if self.vector_conductance:
                etaP_on_grid = etaP.to_grid(self.conductance_basis_evaluator)
                etaH_on_grid = etaH.to_grid(self.conductance_basis_evaluator)

            G_m_ind_to_E_direct = np.einsum(
                "i,jik->jik", etaP_on_grid, self.m_ind_to_bP_JS, optimize=True
            ) + np.einsum("i,jik->jik", etaH_on_grid, self.m_ind_to_bH_JS, optimize=True)
            G_m_imp_to_E_direct = np.einsum(
                "i,jik->jik", etaP_on_grid, self.m_imp_to_bP_JS, optimize=True
            ) + np.einsum("i,jik->jik", etaH_on_grid, self.m_imp_to_bH_JS, optimize=True)

            m_ind_to_E_coeffs_direct = self.basis_evaluator.least_squares_solution_helmholtz(
                G_m_ind_to_E_direct
            )
            m_imp_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
                G_m_imp_to_E_direct
            )

        # Set up jr constraints.
        constraint_matrices = [self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))]
        coeffs_to_constraint_vectors = [self.jr_coeffs_to_j_apex]

        if self.connect_hemispheres and E_MAPPING:
            # Append low-latitude E constraints.
            constraint_matrices.append(
                np.tensordot(self.E_coeffs_to_E_apex_ll_diff, m_imp_to_E_coeffs, 2)
                * self.ih_constraint_scaling
            )
            coeffs_to_constraint_vectors.append(
                self.E_coeffs_to_E_apex_ll_diff * self.ih_constraint_scaling
            )

        constraints_least_squares = LeastSquares(constraint_matrices, 1)
        coeffs_to_m_imp = constraints_least_squares.solve(coeffs_to_constraint_vectors)

        # Construct jr matrices.
        if self.vector_jr:
            self.jr_coeffs_to_m_imp = coeffs_to_m_imp[0].dot(self.jr_coeffs_to_jr_coeffs_state)
            self.jr_coeffs_to_E_coeffs = m_imp_to_E_coeffs.dot(self.jr_coeffs_to_m_imp)
        else:
            self.jr_to_m_imp = coeffs_to_m_imp[0].dot(self.G_jr_state_pinv)
            self.jr_to_E_coeffs = m_imp_to_E_coeffs.dot(self.jr_to_m_imp)

        # Construct m_ind matrices. Negative sign is from moving the
        # induction terms to the right hand side of E - E^cp = 0 (in
        # apex coordinates).
        self.m_ind_to_E_coeffs = m_ind_to_E_coeffs_direct.copy()
        if self.connect_hemispheres and E_MAPPING:
            self.m_ind_to_m_imp = np.tensordot(coeffs_to_m_imp[1], -m_ind_to_E_coeffs_direct, 2)
            self.m_ind_to_E_coeffs += m_imp_to_E_coeffs.dot(self.m_ind_to_m_imp)

        # Construct u matrices. Negative sign is from moving the wind
        # terms to the right hand side of E - E^cp = 0 (in apex
        # coordinates).
        if self.vector_u:
            self.u_coeffs_to_E_coeffs = self.u_coeffs_to_E_coeffs_direct.copy()
            if self.connect_hemispheres and E_MAPPING:
                self.u_coeffs_to_m_imp = np.tensordot(
                    coeffs_to_m_imp[1], -self.u_coeffs_to_E_coeffs_direct, 2
                )
                self.u_coeffs_to_E_coeffs += np.tensordot(
                    m_imp_to_E_coeffs, self.u_coeffs_to_m_imp, 1
                )
        else:
            self.u_to_E_coeffs = self.u_to_E_coeffs_direct.copy()
            if self.connect_hemispheres and E_MAPPING:
                self.u_to_m_imp = np.tensordot(coeffs_to_m_imp[1], -self.u_to_E_coeffs_direct, 2)
                self.u_to_E_coeffs += np.tensordot(m_imp_to_E_coeffs, self.u_to_m_imp, 1)

        # Construct matrix used in steady state calculations.
        self.m_ind_to_E_cf_pinv = np.linalg.pinv(self.m_ind_to_E_coeffs[1])

    def calculate_E_coeffs(self, m_ind):
        """Calculate the coefficients for the electric field.

        Parameters
        ----------
        m_ind : array
            Coefficients for induced part of magnetic field
            perturbation.

        Returns
        -------
        array
            Coefficients for the electric field.
        """
        E_coeffs_m_ind = self.m_ind_to_E_coeffs.dot(m_ind)
        if self.Br_input:
            E_coeffs_m_ind += self.m_ind_to_E_coeffs.dot(self.Br_RM_to_m_S * self.Br.coeffs)

        if self.vector_jr:
            E_coeffs_jr = self.jr_coeffs_to_E_coeffs.dot(self.jr.coeffs)
        else:
            E_coeffs_jr = self.jr_to_E_coeffs.dot(self.jr_on_grid)

        E_coeffs = E_coeffs_m_ind + E_coeffs_jr

        if self.neutral_wind:
            if self.vector_u:
                E_coeffs += np.tensordot(self.u_coeffs_to_E_coeffs, self.u.coeffs, 2)
            else:
                E_coeffs += np.tensordot(self.u_to_E_coeffs, self.u_on_grid, 2)

        return E_coeffs

    def update_E(self):
        """Update electric field coefficients.

        The coefficients represent the electric potential and the
        electric stream function.
        """
        E_coeffs = self.calculate_E_coeffs(self.m_ind.coeffs)
        self.E = FieldExpansion(self.basis, coeffs=E_coeffs, field_type="tangential")

    def evolve_m_ind(self, dt):
        """Evolve induced magnetic field coefficients.

        Updates m_ind by time-stepping dBr/dt forward.

        Parameters
        ----------
        dt : float
            Time step size in seconds.
        """
        from scipy.linalg import expm

        if self.integrator == "euler":
            new_m_ind = self.m_ind.coeffs + self.E.coeffs[1] * self.E_df_to_d_m_ind_dt * dt

        elif self.integrator == "exponential":
            steady_state_m_ind = self.steady_state_m_ind()

            propagator = expm(dt * self.E_df_to_d_m_ind_dt * self.m_ind_to_E_coeffs[1])

            inductive_m_ind = propagator.dot(self.m_ind.coeffs - steady_state_m_ind)

            new_m_ind = inductive_m_ind + steady_state_m_ind

        self.set_model_coeffs(m_ind=new_m_ind)

    def get_Br(self, _basis_evaluator):
        """Calculate ``Br``.

        Parameters
        ----------
        _basis_evaluator : BasisEvaluator
            Basis evaluator object.

        Returns
        -------
        array
            Radial magnetic field.
        """
        return _basis_evaluator.basis_to_grid(self.m_ind.coeffs * self.m_ind_to_Br)

    def get_JS(self):
        """Calculate ionospheric sheet current.

        Returns
        -------
        tuple of arrays
            Theta and phi components of the ionospheric sheet current.

        Notes
        -----
        For now, JS is always returned on self.grid.
        """
        Js_ind, Je_ind = np.split(self.G_m_ind_to_JS.dot(self.m_ind.coeffs), 2, axis=0)
        Js_imp, Je_imp = np.split(self.G_m_imp_to_JS.dot(self.m_imp.coeffs), 2, axis=0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp

        return (Jth, Jph)

    def get_jr(self, _basis_evaluator):
        """Calculate radial current.

        Parameters
        ----------
        _basis_evaluator : BasisEvaluator
            Basis evaluator object.

        Returns
        -------
        array
            Radial current.
        """
        return _basis_evaluator.basis_to_grid(self.m_imp.coeffs * self.m_imp_to_jr)

    def get_Jeq(self, _basis_evaluator):
        """Calculate equivalent current function.

        Parameters
        ----------
        _basis_evaluator : BasisEvaluator
            Basis evaluator object.

        Returns
        -------
        array
            Equivalent current function.
        """
        return _basis_evaluator.basis_to_grid(self.m_ind.coeffs * self.m_ind_to_Jeq)

    def get_Phi(self, _basis_evaluator):
        """Calculate Phi.

        Parameters
        ----------
        _basis_evaluator : BasisEvaluator
            Basis evaluator object.

        Returns
        -------
        array
            Electric potential.
        """
        return _basis_evaluator.basis_to_grid(self.E.coeffs[:, 1])

    def get_W(self, _basis_evaluator):
        """Calculate the induction electric field scalar.

        Parameters
        ----------
        _basis_evaluator : BasisEvaluator
            Basis evaluator object.

        Returns
        -------
        array
            Induction electric field scalar.
        """
        return _basis_evaluator.basis_to_grid(self.E.coeffs[:, 1])

    def get_E(self, _basis_evaluator):
        """Calculate electric field components.

        Parameters
        ----------
        _basis_evaluator : BasisEvaluator
            Evaluator for computing field on grid.

        Returns
        -------
        ndarray
            Electric field components (Etheta, Ephi) on grid points.
        """
        return self.E.to_grid(_basis_evaluator)

    def steady_state_m_ind(self):
        """Calculate coefficients for induced field in steady state.

        Returns
        -------
        array
            Coefficients for the induced magnetic field in steady state.
        """
        if self.vector_jr:
            E_coeffs_noind = self.jr_coeffs_to_E_coeffs.dot(self.jr.coeffs)
        else:
            E_coeffs_noind = self.jr_to_E_coeffs.dot(self.jr_on_grid)

        if self.neutral_wind:
            if self.vector_u:
                E_coeffs_noind += np.tensordot(self.u_coeffs_to_E_coeffs, self.u.coeffs, 2)
            else:
                E_coeffs_noind += np.tensordot(self.u_to_E_coeffs, self.u_on_grid, 2)

        m_ind = -self.m_ind_to_E_cf_pinv.dot(E_coeffs_noind[1])

        return m_ind

    def prepare_triple_product_tensors(self, plot=True):
        """Prepare tensors for triple product calculation.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the tensors.
        """
        etaP_m_ind_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_ind_to_bP_JS, self.conductance_basis_evaluator.G, optimize=True
        )
        self.etaP_m_ind_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaP_m_ind_to_E
        )

        etaH_m_ind_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_ind_to_bH_JS, self.conductance_basis_evaluator.G, optimize=True
        )
        self.etaH_m_ind_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaH_m_ind_to_E
        )

        etaP_m_imp_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_imp_to_bP_JS, self.conductance_basis_evaluator.G, optimize=True
        )
        self.etaP_m_imp_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaP_m_imp_to_E
        )

        etaH_m_imp_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_imp_to_bH_JS, self.conductance_basis_evaluator.G, optimize=True
        )
        self.etaH_m_imp_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaH_m_imp_to_E
        )

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            _, ax = plt.subplots(5, 1, tight_layout=True, figsize=(40, 10))

            vmin = 1e-4
            vmax = 1e8

            ax[0].matshow(
                np.abs(self.etaP_m_ind_to_E_coeffs.reshape((2 * self.basis.index_length, -1))),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax[1].matshow(
                np.abs(self.etaP_m_imp_to_E_coeffs.reshape((2 * self.basis.index_length, -1))),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax[2].matshow(
                np.abs(self.etaH_m_ind_to_E_coeffs.reshape((2 * self.basis.index_length, -1))),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax[3].matshow(
                np.abs(self.etaH_m_imp_to_E_coeffs.reshape((2 * self.basis.index_length, -1))),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )

            ax[4].matshow(
                (
                    np.abs(self.etaP_m_ind_to_E_coeffs)
                    + np.abs(self.etaP_m_imp_to_E_coeffs)
                    + np.abs(self.etaH_m_ind_to_E_coeffs)
                    + np.abs(self.etaH_m_imp_to_E_coeffs)
                ).reshape((2 * self.basis.index_length, -1)),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )

            plt.show()
