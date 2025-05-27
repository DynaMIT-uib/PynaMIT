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
from pynamit.spherical_harmonics.sh_basis import SHBasis

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

    def __init__(self, basis, mainfield, cs_basis, settings, PFAC_matrix=None):
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
            FAC_integration_steps, and ih_constraint_scaling.
        PFAC_matrix : array-like, optional
            Pre-computed FAC poloidal field matrix.
        """
        self.basis = basis
        self.mainfield = mainfield

        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.ih_constraint_scaling = settings.ih_constraint_scaling

        self.integrator = settings.integrator

        if PFAC_matrix is not None:
            self._T_to_Ve = PFAC_matrix

        # Initialize grid-related objects.
        self.grid = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)

        # Note that these BasisEvaluator objects cannot be used for
        # inverses, as they do not include regularization and weights.
        self.basis_evaluator = BasisEvaluator(self.basis, self.grid)
        self.basis_evaluator_zero_added = BasisEvaluator(
            SHBasis(settings.Nmax, settings.Mmax, Nmin=0), self.grid
        )
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

        if self.RM is not None:
            Br_RM_to_m_S = (
                -1
                / (
                    1
                    - self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(self.RI, self.RM)
                )
                * self.basis.radial_shift_Ve(self.RM, self.RI)
                / self.m_ind_to_Br
            )

            self.G_Br_to_JS = self.G_Ve_to_JS * Br_RM_to_m_S

            m_ind_to_m_S = (
                1
                / (
                    1
                    - self.basis.radial_shift_Ve(self.RM, self.RI)
                    * self.basis.radial_shift_Vi(self.RI, self.RM)
                )
                * self.basis.radial_shift_Ve(self.RM, self.RI)
                * self.basis.radial_shift_Vi(self.RI, self.RM)
            )

            self.G_m_ind_to_JS *= 1 + m_ind_to_m_S

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

        if self.RM is not None:
            self.Br_to_bP_JS = np.einsum("ijk,jkl->ikl", self.bP, self.G_Br_to_JS, optimize=True)
            self.Br_to_bH_JS = np.einsum("ijk,jkl->ikl", self.bH, self.G_Br_to_JS, optimize=True)

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

        u_coeffs_to_uxB = np.einsum(
            "ijk,jklm->iklm", self.bu, self.basis_evaluator.G_helmholtz, optimize=True
        )
        self.u_coeffs_to_E_coeffs_direct = self.basis_evaluator.least_squares_solution_helmholtz(
            u_coeffs_to_uxB
        )

        if TRIPLE_PRODUCT:
            self.prepare_triple_product_tensors(plot=False)

        # Conductance and neutral wind should be set after state
        # initialization.
        self.u = None
        self.Br = None
        self.jr = None

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

    def initialize_constraints(self):
        """Initialize constraints."""
        jr_coeffs_to_j_apex = (
            self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        )
        self.jr_coeffs_to_j_apex = jr_coeffs_to_j_apex.copy()

        if self.connect_hemispheres:
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

    def update(self, input_timeseries, time, interpolation=False):
        """Select input data corresponding to the latest time."""
        for key in input_timeseries.datasets.keys():
            updated_input_entry = input_timeseries.get_entry_if_changed(
                key, time, interpolation=interpolation
            )

            if updated_input_entry is not None:
                if key == "conductance":
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

                    self.update_matrices()

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

    def update_matrices(self):
        """Update the resistance-dependent matrices.

        This method updates the matrices used to calculate the electric
        field and imposed magnetic field from the induced magnetic field
        and input variables.

        Parameters
        ----------
        etaP : FieldExpansion
            Pedersen conductance in S
        etaH : FieldExpansion
            Hall conductance in S
        """
        if TRIPLE_PRODUCT:
            self.m_ind_to_E_coeffs_direct = self.etaP_m_ind_to_E_coeffs.dot(
                self.etaP.coeffs
            ) + self.etaH_m_ind_to_E_coeffs.dot(self.etaH.coeffs)
            self.m_imp_to_E_coeffs = self.etaP_m_imp_to_E_coeffs.dot(
                self.etaP.coeffs
            ) + self.etaH_m_imp_to_E_coeffs.dot(self.etaH.coeffs)

        else:
            etaP_on_grid = self.etaP.to_grid(self.basis_evaluator_zero_added)
            etaH_on_grid = self.etaH.to_grid(self.basis_evaluator_zero_added)

            G_m_ind_to_E_direct = np.einsum(
                "i,jik->jik", etaP_on_grid, self.m_ind_to_bP_JS, optimize=True
            ) + np.einsum("i,jik->jik", etaH_on_grid, self.m_ind_to_bH_JS, optimize=True)
            G_m_imp_to_E_direct = np.einsum(
                "i,jik->jik", etaP_on_grid, self.m_imp_to_bP_JS, optimize=True
            ) + np.einsum("i,jik->jik", etaH_on_grid, self.m_imp_to_bH_JS, optimize=True)

            self.m_ind_to_E_coeffs_direct = self.basis_evaluator.least_squares_solution_helmholtz(
                G_m_ind_to_E_direct
            )
            self.m_imp_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
                G_m_imp_to_E_direct
            )
            if self.RM is not None:
                G_Br_to_E_direct = np.einsum(
                    "i,jik->jik", etaP_on_grid, self.Br_to_bP_JS, optimize=True
                ) + np.einsum("i,jik->jik", etaH_on_grid, self.Br_to_bH_JS, optimize=True)
                self.Br_to_E_coeffs_direct = self.basis_evaluator.least_squares_solution_helmholtz(
                    G_Br_to_E_direct
                )

        # Set up jr constraints.
        constraint_matrices = [self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))]
        self.coeffs_to_constraint_vectors = [self.jr_coeffs_to_j_apex]

        if self.connect_hemispheres and E_MAPPING:
            # Append low-latitude E constraints.
            constraint_matrices.append(
                np.tensordot(self.E_coeffs_to_E_apex_ll_diff, self.m_imp_to_E_coeffs, 2)
                * self.ih_constraint_scaling
            )
            self.coeffs_to_constraint_vectors.append(
                self.E_coeffs_to_E_apex_ll_diff * self.ih_constraint_scaling
            )

        self.constraints_least_squares = LeastSquares(constraint_matrices, 1)

        # Construct m_ind matrices. Negative sign is from moving the
        # induction terms to the right hand side of E - E^cp = 0 (in
        # apex coordinates).
        self.m_ind_to_E_coeffs = self.m_ind_to_E_coeffs_direct.copy()
        if self.connect_hemispheres and E_MAPPING:
            m_ind_to_constraint_vector = np.tensordot(
                self.coeffs_to_constraint_vectors[1], -self.m_ind_to_E_coeffs_direct, 2
            )
            self.m_ind_to_m_imp = self.constraints_least_squares.solve([None, m_ind_to_constraint_vector])
            self.m_ind_to_E_coeffs += self.m_imp_to_E_coeffs.dot(self.m_ind_to_m_imp[1])

        # Construct matrix used in steady state calculations.
        self.E_noind_to_m_ind_steady = -np.linalg.pinv(self.m_ind_to_E_coeffs[1])

    def calculate_noind_coeffs(self):
        """Calculate noind coefficients.

        Calculate the coefficients for the electric field and
        imposed magnetic field, without the induced contribution.

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
        E_coeffs_direct_noind = np.zeros((2, self.basis.index_length))

        if self.u is not None:
            E_coeffs_direct_noind += np.tensordot(
                self.u_coeffs_to_E_coeffs_direct, self.u.coeffs, 2
            )

        if self.Br is not None:
            E_coeffs_direct_noind += self.Br_to_E_coeffs_direct.dot(self.Br.coeffs)

        m_imp_noind = np.zeros(self.basis.index_length)

        constraint_vector = [None, None]
        if self.jr is not None:
            constraint_vector[0] = self.coeffs_to_constraint_vectors[0].dot(self.jr.coeffs)

        if self.connect_hemispheres and E_MAPPING:
            constraint_vector[1] = np.tensordot(
                self.coeffs_to_constraint_vectors[1], -E_coeffs_direct_noind, 2
            )

        solutions = self.constraints_least_squares.solve(constraint_vector)
        m_imp_noind = np.zeros(self.basis.index_length)
        for i in range(len(solutions)):
            if solutions[i] is not None:
                m_imp_noind += solutions[i]

        E_coeffs_noind = E_coeffs_direct_noind + self.m_imp_to_E_coeffs.dot(m_imp_noind)

        return E_coeffs_noind, m_imp_noind

    def calculate_ind_coeffs(self, m_ind):
        """Calculate induced coefficients.

        Calculate the coefficients for the induced contribution to
        the electric field and imposed magnetic field.

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
        E_coeffs_direct_ind = self.m_ind_to_E_coeffs_direct.dot(m_ind)

        m_imp_ind = np.zeros(self.basis.index_length)

        constraint_vector = [None, None]
        if self.connect_hemispheres and E_MAPPING:
            constraint_vector[1] = np.tensordot(
                self.coeffs_to_constraint_vectors[1], -E_coeffs_direct_ind, 2
            )
        solutions = self.constraints_least_squares.solve(constraint_vector)
        m_imp_ind = np.zeros(self.basis.index_length)
        for i in range(len(solutions)):
            if solutions[i] is not None:
                m_imp_ind += solutions[i]

        E_coeffs = E_coeffs_direct_ind + self.m_imp_to_E_coeffs.dot(m_imp_ind)

        return E_coeffs, m_imp_ind

    def evolve_m_ind(self, m_ind, dt, E_coeffs_noind, steady_state_m_ind=None):
        """Evolve induced magnetic field coefficients.

        Updates m_ind by time-stepping dBr/dt forward.

        Parameters
        ----------
        dt : float
            Time step size in seconds.
        """
        from scipy.linalg import expm

        m_ind_to_ddt_m_ind = dt * self.E_df_to_d_m_ind_dt * self.m_ind_to_E_coeffs[1]

        if self.integrator == "euler":
            new_m_ind = (
                m_ind
                + m_ind_to_ddt_m_ind.dot(m_ind)
                + dt * self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
            )

        elif self.integrator == "exponential":
            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)

            propagator = expm(m_ind_to_ddt_m_ind)

            inductive_m_ind = propagator.dot(m_ind - steady_state_m_ind)

            new_m_ind = inductive_m_ind + steady_state_m_ind

        return new_m_ind

    def steady_state_m_ind(self, E_coeffs_noind):
        """Calculate coefficients for induced field in steady state.

        Returns
        -------
        array
            Coefficients for the induced magnetic field in steady state.
        """
        m_ind = self.E_noind_to_m_ind_steady.dot(E_coeffs_noind[1])

        return m_ind

    def prepare_triple_product_tensors(self, plot=True):
        """Prepare tensors for triple product calculation.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the tensors.
        """
        etaP_m_ind_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_ind_to_bP_JS, self.basis_evaluator_zero_added.G, optimize=True
        )
        self.etaP_m_ind_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaP_m_ind_to_E
        )

        etaH_m_ind_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_ind_to_bH_JS, self.basis_evaluator_zero_added.G, optimize=True
        )
        self.etaH_m_ind_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaH_m_ind_to_E
        )

        etaP_m_imp_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_imp_to_bP_JS, self.basis_evaluator_zero_added.G, optimize=True
        )
        self.etaP_m_imp_to_E_coeffs = self.basis_evaluator.least_squares_solution_helmholtz(
            etaP_m_imp_to_E
        )

        etaH_m_imp_to_E = np.einsum(
            "ijk,jl->ijkl", self.m_imp_to_bH_JS, self.basis_evaluator_zero_added.G, optimize=True
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
