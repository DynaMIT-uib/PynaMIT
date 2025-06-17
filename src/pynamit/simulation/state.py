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
    # ... (other attributes as defined in the implementation) ...
    """

    def __init__(self, basis, mainfield, cs_basis, settings, PFAC_matrix=None):
        """Initialize the ionospheric state.

        Parameters
        ----------
        basis : Basis
            The basis for state variables.
        mainfield : Mainfield
            Main magnetic field model.
        cs_basis : Basis
            The basis for the coordinate system.
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

        Ve_to_J_df_coeffs = -self.RI / mu0 * self.basis.coeffs_to_delta_V
        T_to_J_cf_coeffs = self.RI / mu0

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

        if self.mainfield.kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif self.mainfield.kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - RE) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary

        u_coeffs_to_uxB = np.einsum(
            "ijk,jklm->iklm", self.bu, self.basis_evaluator.G_helmholtz, optimize=True
        )
        self.u_coeffs_to_E_coeffs_direct = self.basis_evaluator.least_squares_solution_helmholtz(
            u_coeffs_to_uxB
        )

        if TRIPLE_PRODUCT:
            self.prepare_triple_product_tensors(plot=False)

        # Conductance and neutral wind should be set after state initialization.
        self.u, self.Br, self.jr = None, None, None
        self.initialize_constraints()

        # Initialize caches and operator placeholders
        self._coeffs_to_m_imp_cache = None
        self.m_ind_to_E_df = None

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
            under the ionosphere from the poloidal field of inclined FACs.
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

    def initialize_constraints(self):
        """Initialize constraints."""
        self.jr_coeffs_to_j_apex = (
            self.b_evaluator.radial_to_apex.reshape((-1, 1)) * self.basis_evaluator.G
        ).copy()
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
        and input variables. It invalidates caches for operators that
        depend on these matrices.
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

        # Invalidate caches that depend on the updated matrices.
        self._coeffs_to_m_imp_cache = None
        self.m_ind_to_E_df = None

    def _get_or_compute_coeffs_to_m_imp(self):
        """
        Calculates and returns the `coeffs_to_m_imp` tensor.

        This tensor represents the complex constrained mapping from source
        coefficients (like `jr_coeffs` or E-field coeffs) to the imposed
        magnetic field coefficients (`m_imp`). The result is cached for
        performance, as it is expensive to compute but constant within a
        single time step.
        """
        if self._coeffs_to_m_imp_cache is not None:
            return self._coeffs_to_m_imp_cache

        # This is the original, correct logic for the constrained solve
        constraint_matrices = [self.jr_coeffs_to_j_apex * self.m_imp_to_jr.reshape((1, -1))]
        coeffs_to_constraint_vectors = [self.jr_coeffs_to_j_apex]
        if self.connect_hemispheres and E_MAPPING:
            constraint_matrices.append(
                np.tensordot(self.E_coeffs_to_E_apex_ll_diff, self.m_imp_to_E_coeffs, 2)
                * self.ih_constraint_scaling
            )
            coeffs_to_constraint_vectors.append(
                self.E_coeffs_to_E_apex_ll_diff * self.ih_constraint_scaling
            )

        solver = LeastSquaresSolver(constraint_matrices, 1)
        print("INFO: Performing one-time constrained least-squares solve...")
        self._coeffs_to_m_imp_cache = solver.solve(coeffs_to_constraint_vectors)
        return self._coeffs_to_m_imp_cache

    def build_m_ind_to_E_df(self):
        """
        Builds the operator that maps induced magnetic field coefficients
        to divergence-free electric field coefficients. This is the core
        time-evolution operator `L` where `d(m_ind)/dt ~ L(m_ind)`.
        """
        # The matvec closure needs access to the correct tensor, so we fetch it here.
        coeffs_to_m_imp = self._get_or_compute_coeffs_to_m_imp()

        op_shape = (self.m_ind_to_E_coeffs_direct.shape[2], self.m_ind_to_E_coeffs_direct.shape[1])
        dim = self.m_ind_to_E_coeffs_direct.shape[1]

        def matvec(m_ind):
            m_ind = m_ind.flatten()
            E_ind = np.einsum("ijk,k->ij", self.m_ind_to_E_coeffs_direct, m_ind)
            final_E_comp = E_ind[1]
            if self.connect_hemispheres and E_MAPPING:
                m_imp = np.einsum("ijk,jk->i", coeffs_to_m_imp[1], -E_ind)
                hemisphere_term = np.einsum("jk,k->j", self.m_imp_to_E_coeffs[1], m_imp)
                final_E_comp = final_E_comp + hemisphere_term
            return final_E_comp

        def rmatvec(grad_output):
            grad_final_E_comp = grad_output.flatten()
            total_grad_E_ind = np.zeros((2, dim), dtype=np.float64)
            total_grad_E_ind[1] = grad_final_E_comp
            if self.connect_hemispheres and E_MAPPING:
                C_imp_h = self.m_imp_to_E_coeffs[1]
                T_imp = coeffs_to_m_imp[1]
                grad_m_imp = np.einsum("j,jk->k", grad_final_E_comp, C_imp_h)
                grad_E_ind_from_hemi = -np.einsum("i,ijk->jk", grad_m_imp, T_imp)
                total_grad_E_ind = total_grad_E_ind + grad_E_ind_from_hemi
            return np.einsum("ij,ijk->k", total_grad_E_ind, self.m_ind_to_E_coeffs_direct)

        self.m_ind_to_E_df = LinearOperator(
            shape=op_shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64
        )

    def calculate_noind_coeffs(self):
        """Calculate no-induction coefficients.

        Calculate the coefficients for the electric field and
        imposed magnetic field from external sources like winds and
        field-aligned currents, without the induced contribution.
        """
        # Get the cached or newly computed mapping tensor
        coeffs_to_m_imp = self._get_or_compute_coeffs_to_m_imp()

        E_coeffs_direct_noind = np.zeros((2, self.basis.index_length))
        if self.u is not None:
            E_coeffs_direct_noind += np.tensordot(
                self.u_coeffs_to_E_coeffs_direct, self.u.coeffs, 2
            )
        if self.Br is not None:
            E_coeffs_direct_noind += self.Br_to_E_coeffs_direct.dot(self.Br.coeffs)

        m_imp_noind = np.zeros(self.basis.index_length)
        if self.jr is not None:
            m_imp_noind += coeffs_to_m_imp[0].dot(self.jr.coeffs)
        if self.connect_hemispheres and E_MAPPING:
            m_imp_noind += np.tensordot(coeffs_to_m_imp[1], -E_coeffs_direct_noind, 2)

        E_coeffs_noind = E_coeffs_direct_noind + self.m_imp_to_E_coeffs.dot(m_imp_noind)
        return E_coeffs_noind, m_imp_noind

    def calculate_ind_coeffs(self, m_ind):
        """Calculate induced coefficients.

        Calculate the coefficients for the induced contribution to
        the electric field and imposed magnetic field from a given
        set of induced magnetic field coefficients.
        """
        # Get the cached or newly computed mapping tensor
        coeffs_to_m_imp = self._get_or_compute_coeffs_to_m_imp()

        E_coeffs_direct_ind = self.m_ind_to_E_coeffs_direct.dot(m_ind)
        m_imp_ind = np.zeros(self.basis.index_length)
        if self.connect_hemispheres and E_MAPPING:
            m_imp_ind = np.tensordot(coeffs_to_m_imp[1], -E_coeffs_direct_ind, 2)

        E_coeffs = E_coeffs_direct_ind + self.m_imp_to_E_coeffs.dot(m_imp_ind)
        return E_coeffs, m_imp_ind

    def evolve_m_ind(self, m_ind, dt, E_coeffs_noind, steady_state_m_ind=None):
        """Evolve induced magnetic field coefficients.

        Updates m_ind by time-stepping the induction equation forward using
        either a simple Euler step or a more advanced exponential integrator.
        """
        # Build operator just-in-time, ensuring all dependencies are met.
        if self.m_ind_to_E_df is None:
            self.build_m_ind_to_E_df()

        # The full time evolution operator, L = s * L_part
        ddt_m_ind_operator = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df

        if self.integrator == "euler":
            new_m_ind = (
                m_ind
                + dt * ddt_m_ind_operator.matvec(m_ind)
                + dt * self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]
            )
        elif self.integrator == "exponential":
            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            # expm_multiply requires an adjoint, which our operator now has.
            inductive_m_ind = expm_multiply(dt * ddt_m_ind_operator, m_ind - steady_state_m_ind)
            new_m_ind = inductive_m_ind + steady_state_m_ind

        return new_m_ind

    def steady_state_m_ind(self, E_coeffs_noind):
        """
        Calculate coefficients for induced field in steady state.

        Solves the steady-state equation `L(m_ind) = -F` using an efficient
        iterative solver (GMRES) on the matrix-free time-evolution operator.
        """
        # Build operator just-in-time
        if self.m_ind_to_E_df is None:
            self.build_m_ind_to_E_df()

        # In steady state, d(m_ind)/dt = 0 => L(m_ind) = -F
        # The operator for GMRES is L itself.
        steady_state_op = self.E_df_to_d_m_ind_dt * self.m_ind_to_E_df
        b = -self.E_df_to_d_m_ind_dt * E_coeffs_noind[1]

        # GMRES is the correct iterative solver for this square system.
        m_ind, exit_code = gmres(steady_state_op, b, rtol=1e-12, atol=0)
        if exit_code != 0:
            print(f"Warning: GMRES failed to converge with exit code: {exit_code}")

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
            import matplotlib.pyplot as plt, matplotlib.colors as colors

            _, ax = plt.subplots(5, 1, tight_layout=True, figsize=(40, 10))
            vmin, vmax = 1e-4, 1e8
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
