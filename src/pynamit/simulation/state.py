"""Ionospheric state representation and evolution.

This module provides the State class for managing the ionospheric electrodynamic
state, including currents, fields, and conductances.

Classes
-------
State
    Manages ionospheric electrodynamic state variables and their evolution.
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
    """Manages ionospheric electrodynamic state.

    Handles the representation and evolution of ionospheric currents, electric fields,
    conductances, and magnetic field perturbations. Supports both vector and grid-based
    representations of quantities.

    Parameters
    ----------
    bases : dict
        Spherical harmonic bases for different quantities:
        - 'state': for state variables
        - 'jr': for radial current
        - 'conductance': for conductivity
        - 'u': for neutral wind
    mainfield : Mainfield
        Main magnetic field model
    grid : Grid
        Spatial grid for computations
    settings : object
        Configuration settings containing:
        - RI : float
            Ionospheric radius in meters
        - latitude_boundary : float
            Simulation boundary latitude in degrees
        - ignore_PFAC : bool
            Whether to ignore FAC poloidal fields
        - connect_hemispheres : bool
            Whether hemispheres are electrically connected
        - FAC_integration_steps : array-like
            Radii for FAC poloidal field integration
        - ih_constraint_scaling : float
            Ionospheric height constraint scaling
        - vector_jr : bool
            Use vector representation for radial current
        - vector_conductance : bool
            Use vector representation for conductances
        - vector_u : bool
            Use vector representation for neutral wind
    PFAC_matrix : array-like, optional
        Pre-computed FAC poloidal field matrix, by default None

    Attributes
    ----------
    basis : SHBasis
        Main state variable basis
    jr_basis : SHBasis
        Radial current basis
    conductance_basis : SHBasis
        Conductance basis
    u_basis : SHBasis
        Neutral wind basis
    grid : Grid
        Computational grid
    mainfield : Mainfield
        Main magnetic field model
    m_ind : FieldExpansion
        Induced magnetic field coefficients
    m_imp : FieldExpansion
        Imposed magnetic field coefficients
    E : FieldExpansion
        Electric field vector
    """

    def __init__(self, bases, mainfield, grid, settings, PFAC_matrix=None):
        """
        Initialize the state of the ionosphere.
        """
        self.basis = bases["state"]
        self.jr_basis = bases["jr"]
        self.conductance_basis = bases["conductance"]
        self.u_basis = bases["u"]

        self.mainfield = mainfield

        self.RI = settings.RI
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.ih_constraint_scaling = settings.ih_constraint_scaling

        self.vector_u = settings.vector_u
        self.vector_jr = settings.vector_jr
        self.vector_conductance = settings.vector_conductance

        if PFAC_matrix is not None:
            self._m_imp_to_B_pol = PFAC_matrix

        # Initialize grid-related objects
        self.grid = grid

        # Note that these BasisEvaluator objects cannot be used for inverses, as they do not include regularization and weights
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.jr_basis_evaluator = BasisEvaluator(self.jr_basis, self.grid)
        self.conductance_basis_evaluator = BasisEvaluator(
            self.conductance_basis, self.grid
        )
        self.u_basis_evaluator = BasisEvaluator(self.u_basis, self.grid)

        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)

        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)

        # Spherical harmonic conversion factors
        self.m_ind_to_Br = -self.RI * self.basis.d_dr_V_external(self.RI)
        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.E_df_to_d_m_ind_dt = self.basis.laplacian(
            self.RI
        ) / self.basis.d_dr_V_external(
            self.RI
        )  # The same as d_dr_internal
        self.m_ind_to_Jeq = -self.RI / mu0 * self.basis.V_external_to_delta_V

        B_pol_to_J_df_coeffs = (
            -self.RI * self.basis.V_external_to_delta_V / mu0
        )  # RI comes from the scaling in the V potential
        # RI comes from the scaling in the T potential
        B_tor_to_J_cf_coeffs = self.RI / mu0
        # 1 / RI comes from the scaling in the theta/phi components of the gradient
        self.G_B_pol_to_JS = (
            self.basis_evaluator.G_rxgrad * B_pol_to_J_df_coeffs / self.RI
        )
        # 1 / RI comes from the scaling in the theta/phi components of the gradient
        self.G_B_tor_to_JS = (
            -self.basis_evaluator.G_grad * B_tor_to_J_cf_coeffs / self.RI
        )
        self.G_m_ind_to_JS = self.G_B_pol_to_JS
        self.G_m_imp_to_JS = self.G_B_tor_to_JS + np.tensordot(
            self.G_B_pol_to_JS, self.m_imp_to_B_pol.values, 1
        )

        # Construct the matrix elements used to calculate the electric field
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

        self.m_ind_to_bP_JS = np.einsum(
            "ijk,jkl->ikl", self.bP, self.G_m_ind_to_JS, optimize=True
        )
        self.m_ind_to_bH_JS = np.einsum(
            "ijk,jkl->ikl", self.bH, self.G_m_ind_to_JS, optimize=True
        )
        self.m_imp_to_bP_JS = np.einsum(
            "ijk,jkl->ikl", self.bP, self.G_m_imp_to_JS, optimize=True
        )
        self.m_imp_to_bH_JS = np.einsum(
            "ijk,jkl->ikl", self.bH, self.G_m_imp_to_JS, optimize=True
        )

        # Identify the high and low latitude points
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
            self.jr_coeffs_to_jr_coeffs_state = self.G_jr_state_pinv.dot(
                self.jr_basis_evaluator.G
            )

        if self.vector_u:
            u_coeffs_to_uxB = np.einsum(
                "ijk,jklm->iklm",
                self.bu,
                self.u_basis_evaluator.G_helmholtz,
                optimize=True,
            )
            self.u_coeffs_to_E_coeffs_direct = (
                self.basis_evaluator.least_squares_solution_helmholtz(u_coeffs_to_uxB)
            )
        else:
            self.u_to_E_coeffs_direct = np.einsum(
                "ijkl,kml->ijml",
                self.basis_evaluator.least_squares_helmholtz.ATWA_plus_R_inv_ATW[
                    0
                ].reshape(
                    (
                        self.basis_evaluator.least_squares_helmholtz.A[0].full_shapes[1]
                        + self.basis_evaluator.least_squares_helmholtz.A[0].full_shapes[
                            0
                        ]
                    )
                ),
                self.bu,
                optimize=True,
            )

        if TRIPLE_PRODUCT and self.vector_conductance:
            self.prepare_triple_product_tensors()

        # Conductance and neutral wind should be set after state initialization
        self.neutral_wind = False
        self.conductance = False

        self.initialize_constraints()

    @property
    def m_imp_to_B_pol(self):
        """
        Return matrix that maps self.m_imp to a poloidal field corresponding to a ionospheric current sheet that shields the region under the ionosphere from the poloidal field of inclined FACs.

        Uses the method by Engels and Olsen 1998, Eq. 13 to account for the poloidal part of magnetic field for FACs.

        Returns
        -------
        array
            Matrix that maps self.m_imp to a poloidal field.
        """
        if not hasattr(self, "_m_imp_to_B_pol"):

            self._m_imp_to_B_pol = xr.DataArray(
                data=np.zeros((self.basis.index_length, self.basis.index_length)),
                coords={
                    "i": np.arange(self.basis.index_length),
                    "j": np.arange(self.basis.index_length),
                },
                dims=["i", "j"],
            )

            if not (self.mainfield.kind == "radial" or self.ignore_PFAC):
                r_k_steps = self.FAC_integration_steps
                Delta_k = np.diff(r_k_steps)
                r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

                JS_shifted_to_B_pol_shifted = tensor_pinv(
                    self.G_B_pol_to_JS, n_leading_flattened=2, rtol=0
                )

                for i in range(r_k.size):
                    print(
                        f"Calculating matrix for poloidal field of inclined FACs. Progress: {i+1}/{r_k.size}",
                        end="\r" if i < (r_k.size - 1) else "\n",
                        flush=True,
                    )
                    # Map coordinates from r_k[i] to RI:
                    theta_mapped, phi_mapped = self.mainfield.map_coords(
                        self.RI, r_k[i], self.grid.theta, self.grid.phi
                    )
                    mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)

                    # Matrix that gives jr at mapped grid from toroidal coefficients, shifts to r_k[i], and extracts horizontal current components
                    shifted_b_evaluator = FieldEvaluator(
                        self.mainfield, self.grid, r_k[i]
                    )
                    mapped_b_evaluator = FieldEvaluator(
                        self.mainfield, mapped_grid, self.RI
                    )
                    mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)
                    m_imp_to_jr = mapped_basis_evaluator.scaled_G(self.m_imp_to_jr)
                    jr_to_JS_shifted = np.array(
                        [
                            shifted_b_evaluator.Btheta / mapped_b_evaluator.Br,
                            shifted_b_evaluator.Bphi / mapped_b_evaluator.Br,
                        ]
                    )

                    m_imp_to_JS_shifted = np.einsum(
                        "ij,jk->ijk", jr_to_JS_shifted, m_imp_to_jr, optimize=True
                    )

                    # Matrix that calculates the contribution to the poloidal coefficients from the horizontal current components at r_k[i]
                    B_pol_shifted_to_B_pol = self.basis.radial_shift_V_external(
                        r_k[i], self.RI
                    ).reshape((-1, 1, 1))
                    JS_shifted_to_B_pol = (
                        JS_shifted_to_B_pol_shifted * B_pol_shifted_to_B_pol
                    )

                    # Integration step, negative sign is to create a poloidal field that shields the region under the ionosphere from the FAC poloidal field
                    self._m_imp_to_B_pol -= Delta_k[i] * np.tensordot(
                        JS_shifted_to_B_pol, m_imp_to_JS_shifted, 2
                    )

        return self._m_imp_to_B_pol

    def set_model_coeffs(self, **kwargs):
        """
        Set model coefficients.

        Set model coefficients based on the coefficients given as argument.
        This function accepts one (and only one) set of coefficients.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying the coefficients to set. Valid values are 'm_ind' and 'm_imp'.
        """
        valid_kws = ["m_ind", "m_imp"]

        if len(kwargs) != 1:
            raise Exception(
                "Expected one and only one keyword argument, you provided {}".format(
                    len(kwargs)
                )
            )
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception("Invalid keyword. See documentation")

        if key == "m_ind":
            self.m_ind = FieldExpansion(self.basis, kwargs["m_ind"], field_type="scalar")
        elif key == "m_imp":
            self.m_imp = FieldExpansion(self.basis, kwargs["m_imp"], field_type="scalar")
        else:
            raise Exception("This should not happen")

    def initialize_constraints(self):
        """
        Initialize constraints.
        """
        jr_coeffs_to_j_apex = (
            self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        )
        self.jr_coeffs_to_j_apex = jr_coeffs_to_j_apex.copy()

        if self.connect_hemispheres:
            if self.ignore_PFAC:
                raise ValueError(
                    "Hemispheres can not be connected when ignore_PFAC is True"
                )
            if self.mainfield.kind == "radial":
                raise ValueError(
                    "Hemispheres can not be connected with radial magnetic field"
                )

            if J_MAPPING:
                jr_coeffs_to_j_apex_cp = (
                    self.cp_b_evaluator.radial_to_apex.reshape((-1, 1))
                    * self.cp_basis_evaluator.G
                )
                self.jr_coeffs_to_j_apex[self.ll_mask] -= jr_coeffs_to_j_apex_cp[
                    self.ll_mask
                ]

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
        """
        Calculate m_imp.

        Parameters
        ----------
        m_ind : array
            Coefficients for induced part of magnetic field perturbation.

        Returns
        -------
        array
            Coefficients for imposed part of magnetic field perturbation.
        """
        if self.vector_jr:
            m_imp = self.jr_coeffs_to_m_imp.dot(self.jr.coeffs)
        else:
            m_imp = self.jr_to_m_imp.dot(self.jr_on_grid)

        if self.connect_hemispheres and E_MAPPING:
            m_imp += self.m_ind_to_m_imp.dot(m_ind)

            if self.neutral_wind:
                if self.vector_u:
                    m_imp += np.tensordot(self.u_coeffs_to_m_imp, self.u.coeffs, 2)
                else:
                    m_imp += np.tensordot(self.u_to_m_imp, self.u_on_grid, 2)

        return m_imp

    def update_m_imp(self):
        """
        Impose constraints, if any. Leads to a contribution to m_imp from m_ind if the hemispheres are connected.
        """

        m_imp = self.calculate_m_imp(self.m_ind.coeffs)
        self.set_model_coeffs(m_imp=m_imp)

    def set_jr(self, jr):
        """Set radial current distribution.

        Parameters
        ----------
        jr : array-like or FieldExpansion
            Radial current density in A/m² at grid points or as vector coefficients
        """
        if self.vector_jr:
            self.jr = jr
        else:
            self.jr_on_grid = jr

    def set_u(self, u):
        """
        Set neutral wind theta and phi components.

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
            ) + np.einsum(
                "i,jik->jik", etaH_on_grid, self.m_ind_to_bH_JS, optimize=True
            )
            G_m_imp_to_E_direct = np.einsum(
                "i,jik->jik", etaP_on_grid, self.m_imp_to_bP_JS, optimize=True
            ) + np.einsum(
                "i,jik->jik", etaH_on_grid, self.m_imp_to_bH_JS, optimize=True
            )

            m_ind_to_E_coeffs_direct = (
                self.basis_evaluator.least_squares_solution_helmholtz(
                    G_m_ind_to_E_direct
                )
            )
            m_imp_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
                G_m_imp_to_E_direct
            )

        # jr constraints
        constraint_matrices = [
            self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))
        ]
        coeffs_to_constraint_vectors = [self.jr_coeffs_to_j_apex]

        if self.connect_hemispheres and E_MAPPING:
            # Low-latitude E constraints
            constraint_matrices.append(
                np.tensordot(self.E_coeffs_to_E_apex_ll_diff, m_imp_to_E_coeffs, 2)
                * self.ih_constraint_scaling
            )
            coeffs_to_constraint_vectors.append(
                self.E_coeffs_to_E_apex_ll_diff * self.ih_constraint_scaling
            )

        constraints_least_squares = LeastSquares(constraint_matrices, 1)
        coeffs_to_m_imp = constraints_least_squares.solve(coeffs_to_constraint_vectors)

        # jr matrices
        if self.vector_jr:
            self.jr_coeffs_to_m_imp = coeffs_to_m_imp[0].dot(
                self.jr_coeffs_to_jr_coeffs_state
            )
            self.jr_coeffs_to_E_coeffs = m_imp_to_E_coeffs.dot(self.jr_coeffs_to_m_imp)
        else:
            self.jr_to_m_imp = coeffs_to_m_imp[0].dot(self.G_jr_state_pinv)
            self.jr_to_E_coeffs = m_imp_to_E_coeffs.dot(self.jr_to_m_imp)

        # m_ind matrices, negative sign is from moving the induction terms to the right hand side of E - E^cp = 0 (in apex coordinates)
        self.m_ind_to_E_coeffs = m_ind_to_E_coeffs_direct.copy()
        if self.connect_hemispheres and E_MAPPING:
            self.m_ind_to_m_imp = np.tensordot(
                coeffs_to_m_imp[1], -m_ind_to_E_coeffs_direct, 2
            )
            self.m_ind_to_E_coeffs += m_imp_to_E_coeffs.dot(self.m_ind_to_m_imp)

        # u matrices, negative sign is from moving the wind terms to the right hand side of E - E^cp = 0 (in apex coordinates)
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
                self.u_to_m_imp = np.tensordot(
                    coeffs_to_m_imp[1], -self.u_to_E_coeffs_direct, 2
                )
                self.u_to_E_coeffs += np.tensordot(
                    m_imp_to_E_coeffs, self.u_to_m_imp, 1
                )

        # For steady state
        self.m_ind_to_E_cf_inv = np.linalg.pinv(self.m_ind_to_E_coeffs[1])

    def calculate_E_coeffs(self, m_ind):
        """
        Calculate the coefficients for the electric field.

        Parameters
        ----------
        m_ind : array
            Coefficients for induced part of magnetic field perturbation.

        Returns
        -------
        array
            Coefficients for the electric field.
        """
        E_coeffs_m_ind = self.m_ind_to_E_coeffs.dot(m_ind)

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
        """
        Update the coefficients for the electric potential and the induction electric field.
        """
        E_coeffs = self.calculate_E_coeffs(self.m_ind.coeffs)
        self.E = FieldExpansion(self.basis, coeffs=E_coeffs, field_type="tangential")

    def evolve_m_ind(self, dt):
        """Evolve induced magnetic field coefficients.

        Updates m_ind by time-stepping dBr/dt forward.

        Parameters
        ----------
        dt : float
            Time step size in seconds
        """
        new_m_ind = self.m_ind.coeffs + self.E.coeffs[1] * self.E_df_to_d_m_ind_dt * dt

        self.set_model_coeffs(m_ind=new_m_ind)

    def get_Br(self, _basis_evaluator):
        """
        Calculate ``Br``.

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

    def get_JS(self):  # for now, JS is always returned on self.grid!
        """
        Calculate ionospheric sheet current.

        Returns
        -------
        tuple of arrays
            Theta and phi components of the ionospheric sheet current.
        """
        Js_ind, Je_ind = np.split(self.G_m_ind_to_JS.dot(self.m_ind.coeffs), 2, axis=0)
        Js_imp, Je_imp = np.split(self.G_m_imp_to_JS.dot(self.m_imp.coeffs), 2, axis=0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp

        return (Jth, Jph)

    def get_jr(self, _basis_evaluator):
        """
        Calculate radial current.

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
        """
        Calculate equivalent current function.

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
        """
        Calculate Phi.

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
        """
        Calculate the induction electric field scalar.

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
            Evaluator for computing field on grid

        Returns
        -------
        ndarray
            Electric field components (Etheta, Ephi) on grid points
        """
        return self.E.to_grid(_basis_evaluator)

    def steady_state_m_ind(self):
        """
        Calculate coefficients for induced field in steady state.

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
                E_coeffs_noind += np.tensordot(
                    self.u_coeffs_to_E_coeffs, self.u.coeffs, 2
                )
            else:
                E_coeffs_noind += np.tensordot(self.u_to_E_coeffs, self.u_on_grid, 2)

        m_ind = -self.m_ind_to_E_cf_inv.dot(E_coeffs_noind[1])

        return m_ind

    def prepare_triple_product_tensors(self, plot=True):
        """
        Prepare tensors for triple product calculation.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the tensors. Default is True.
        """
        etaP_m_ind_to_E = np.einsum(
            "ijk,jl->ijkl",
            self.m_ind_to_bP_JS,
            self.conductance_basis_evaluator.G,
            optimize=True,
        )
        self.etaP_m_ind_to_E_coeffs = (
            self.basis_evaluator.least_squares_solution_helmholtz(etaP_m_ind_to_E)
        )

        etaH_m_ind_to_E = np.einsum(
            "ijk,jl->ijkl",
            self.m_ind_to_bH_JS,
            self.conductance_basis_evaluator.G,
            optimize=True,
        )
        self.etaH_m_ind_to_E_coeffs = (
            self.basis_evaluator.least_squares_solution_helmholtz(etaH_m_ind_to_E)
        )

        etaP_m_imp_to_E = np.einsum(
            "ijk,jl->ijkl",
            self.m_imp_to_bP_JS,
            self.conductance_basis_evaluator.G,
            optimize=True,
        )
        self.etaP_m_imp_to_E_coeffs = (
            self.basis_evaluator.least_squares_solution_helmholtz(etaP_m_imp_to_E)
        )

        etaH_m_imp_to_E = np.einsum(
            "ijk,jl->ijkl",
            self.m_imp_to_bH_JS,
            self.conductance_basis_evaluator.G,
            optimize=True,
        )
        self.etaH_m_imp_to_E_coeffs = (
            self.basis_evaluator.least_squares_solution_helmholtz(etaH_m_imp_to_E)
        )

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            _, ax = plt.subplots(5, 1, tight_layout=True, figsize=(40, 10))

            vmin = 1e-4
            vmax = 1e8

            ax[0].matshow(
                np.abs(
                    self.etaP_m_ind_to_E_coeffs.reshape(
                        (2 * self.basis.index_length, -1)
                    )
                ),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax[1].matshow(
                np.abs(
                    self.etaP_m_imp_to_E_coeffs.reshape(
                        (2 * self.basis.index_length, -1)
                    )
                ),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax[2].matshow(
                np.abs(
                    self.etaH_m_ind_to_E_coeffs.reshape(
                        (2 * self.basis.index_length, -1)
                    )
                ),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax[3].matshow(
                np.abs(
                    self.etaH_m_imp_to_E_coeffs.reshape(
                        (2 * self.basis.index_length, -1)
                    )
                ),
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
