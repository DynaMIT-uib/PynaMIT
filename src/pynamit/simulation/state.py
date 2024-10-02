import numpy as np
import xarray as xr
from pynamit.various.constants import mu0, RE
from pynamit.primitives.grid import Grid
from pynamit.primitives.vector import Vector
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.various.math import tensor_scale_left, tensor_pinv, tensor_transpose, pinv_positive_semidefinite

TRIPLE_PRODUCT = False

class State(object):
    """ State of the ionosphere.

    """

    def __init__(self, bases, pinv_rtols, reg_lambdas, mainfield, grid, settings, PFAC_matrix = None):
        """ Initialize the state of the ionosphere.
    
        """

        self.basis             = bases['state']
        self.jr_basis          = bases['jr']
        self.conductance_basis = bases['conductance']
        self.u_basis           = bases['u']

        self.mainfield = mainfield

        self.RI                     = settings.RI
        self.latitude_boundary      = settings.latitude_boundary
        self.ignore_PFAC            = bool(settings.ignore_PFAC)
        self.connect_hemispheres    = bool(settings.connect_hemispheres)
        self.FAC_integration_steps  = settings.FAC_integration_steps
        self.ih_constraint_scaling  = settings.ih_constraint_scaling

        self.vector_u           = settings.vector_u
        self.vector_jr          = settings.vector_jr
        self.vector_conductance = settings.vector_conductance

        if PFAC_matrix is not None:
            self._m_imp_to_B_pol = PFAC_matrix

        # Initialize grid-related objects
        self.grid = grid

        self.basis_evaluator             = BasisEvaluator(self.basis,             self.grid, pinv_rtol = pinv_rtols['state'],       reg_lambda = reg_lambdas['state'])
        self.jr_basis_evaluator          = BasisEvaluator(self.jr_basis,          self.grid, pinv_rtol = pinv_rtols['jr'],          reg_lambda = reg_lambdas['jr'])
        self.conductance_basis_evaluator = BasisEvaluator(self.conductance_basis, self.grid, pinv_rtol = pinv_rtols['conductance'], reg_lambda = reg_lambdas['conductance'])
        self.u_basis_evaluator           = BasisEvaluator(self.u_basis,           self.grid, pinv_rtol = pinv_rtols['u'],           reg_lambda = reg_lambdas['u'])

        self.b_evaluator = FieldEvaluator(mainfield, self.grid, self.RI)

        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(self.RI, self.grid.theta, self.grid.phi)
            self.cp_grid = Grid(theta = cp_theta, phi = cp_phi)

            self.cp_basis_evaluator    = BasisEvaluator(self.basis, self.cp_grid,    pinv_rtol = pinv_rtols['state'], reg_lambda = reg_lambdas['state'])
            self.jr_cp_basis_evaluator = BasisEvaluator(self.jr_basis, self.cp_grid, pinv_rtol = pinv_rtols['jr'],    reg_lambda = reg_lambdas['jr'])

            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)

        # Spherical harmonic conversion factors
        self.m_ind_to_Br  = self.RI * self.basis.d_dr(self.RI)
        self.m_imp_to_jr  = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.W_to_dBr_dt  = -self.RI * self.basis.laplacian(self.RI)
        self.m_ind_to_Jeq = -self.RI / mu0 * self.basis.delta_internal_external

        self.G_B_pol_to_JS = -self.basis_evaluator.G_rxgrad * self.basis.delta_internal_external / mu0
        self.G_B_tor_to_JS = -self.basis_evaluator.G_grad / mu0
        self.G_m_ind_to_JS = self.G_B_pol_to_JS
        self.G_m_imp_to_JS = self.G_B_tor_to_JS + np.tensordot(self.G_B_pol_to_JS, self.m_imp_to_B_pol.values, 1)

        # Construct the matrix elements used to calculate the electric field
        self.bP = np.array([[self.b_evaluator.bphi**2 + self.b_evaluator.br**2, -self.b_evaluator.btheta * self.b_evaluator.bphi],
                            [-self.b_evaluator.btheta * self.b_evaluator.bphi,  self.b_evaluator.btheta**2 + self.b_evaluator.br**2]])

        self.bH = np.array([[np.zeros(self.b_evaluator.grid.size), self.b_evaluator.br],
                            [-self.b_evaluator.br,                 np.zeros(self.b_evaluator.grid.size)]])

        self.bu = -np.array([[np.zeros(self.b_evaluator.grid.size), self.b_evaluator.Br],
                             [-self.b_evaluator.Br,                 np.zeros(self.b_evaluator.grid.size)]])

        self.m_ind_to_bP_JS = np.einsum('ijk,kjl->kil', self.bP, self.G_m_ind_to_JS, optimize = True)
        self.m_ind_to_bH_JS = np.einsum('ijk,kjl->kil', self.bH, self.G_m_ind_to_JS, optimize = True)
        self.m_imp_to_bP_JS = np.einsum('ijk,kjl->kil', self.bP, self.G_m_imp_to_JS, optimize = True)
        self.m_imp_to_bH_JS = np.einsum('ijk,kjl->kil', self.bH, self.G_m_imp_to_JS, optimize = True)

        if TRIPLE_PRODUCT and self.vector_conductance:
            etaP_m_ind_to_E = np.einsum('ijk,il->ijkl', self.m_ind_to_bP_JS, self.conductance_basis_evaluator.G, optimize = True)
            self.etaP_m_ind_to_helmholtz_E = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, etaP_m_ind_to_E, 2), 2)

            etaH_m_ind_to_E = np.einsum('ijk,il->ijkl', self.m_ind_to_bH_JS, self.conductance_basis_evaluator.G, optimize = True)
            self.etaH_m_ind_to_helmholtz_E = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, etaH_m_ind_to_E, 2), 2)

            etaP_m_imp_to_E = np.einsum('ijk,il->ijkl', self.m_imp_to_bP_JS, self.conductance_basis_evaluator.G, optimize = True)
            self.etaP_m_imp_to_helmholtz_E = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, etaP_m_imp_to_E, 2), 2)

            etaH_m_imp_to_E = np.einsum('ijk,il->ijkl', self.m_imp_to_bH_JS, self.conductance_basis_evaluator.G, optimize = True)
            self.etaH_m_imp_to_helmholtz_E = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, etaH_m_imp_to_E, 2), 2)

            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            _, ax = plt.subplots(5, 1, tight_layout = True, figsize = (40, 10))

            vmin = 1e-4
            vmax = 1e8

            ax[0].matshow(np.abs(self.etaP_m_ind_to_helmholtz_E.reshape((2 * self.basis.index_length, -1))), norm=colors.LogNorm(vmin = vmin, vmax = vmax))
            ax[1].matshow(np.abs(self.etaP_m_imp_to_helmholtz_E.reshape((2 * self.basis.index_length, -1))), norm=colors.LogNorm(vmin = vmin, vmax = vmax))
            ax[2].matshow(np.abs(self.etaH_m_ind_to_helmholtz_E.reshape((2 * self.basis.index_length, -1))), norm=colors.LogNorm(vmin = vmin, vmax = vmax))
            ax[3].matshow(np.abs(self.etaH_m_imp_to_helmholtz_E.reshape((2 * self.basis.index_length, -1))), norm=colors.LogNorm(vmin = vmin, vmax = vmax))
            ax[4].matshow((np.abs(self.etaP_m_ind_to_helmholtz_E) + np.abs(self.etaP_m_imp_to_helmholtz_E) + np.abs(self.etaH_m_ind_to_helmholtz_E) + np.abs(self.etaH_m_imp_to_helmholtz_E)).reshape((2 * self.basis.index_length, -1)), norm=colors.LogNorm(vmin = vmin, vmax = vmax))

            plt.show()


        self.G_m_imp_to_jr = self.jr_basis_evaluator.scaled_G(self.m_imp_to_jr)

        if self.vector_jr:
            self.G_jr = self.jr_basis_evaluator.G

        if self.vector_u:
            u_coeffs_to_uxB = np.einsum('ijk,kjlm->kilm', self.bu, self.u_basis_evaluator.G_helmholtz, optimize = True)
            self.u_coeffs_to_helmholtz_E = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, u_coeffs_to_uxB, 2), 2)
        else:
            self.u_to_helmholtz_E = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.einsum('ijkl,lmk->ijkm', self.basis_evaluator.GTW_helmholtz, self.bu, optimize = True), 2)

        # Conductance and neutral wind should be set after state initialization
        self.neutral_wind = False
        self.conductance  = False

        self.initialize_constraints()

    @property
    def m_imp_to_B_pol(self):
        """
        Return matrix that maps self.m_imp to a poloidal field
        corresponding to a ionospheric current sheet that shields the
        region under the ionosphere from the poloidal field of inclined
        FACs. Uses the method by Engels and Olsen 1998, Eq. 13 to account
        for the poloidal part of magnetic field for FACs.

        """

        if not hasattr(self, '_m_imp_to_B_pol'):

            self._m_imp_to_B_pol = xr.DataArray(
                data = np.zeros((self.basis.index_length, self.basis.index_length)),
                coords = {
                    'i': np.arange(self.basis.index_length),
                    'j': np.arange(self.basis.index_length),
                },
                dims = ['i', 'j']
            )

            if not (self.mainfield.kind == 'radial' or self.ignore_PFAC):
                r_k_steps = self.FAC_integration_steps
                Delta_k = np.diff(r_k_steps)
                r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

                JS_shifted_to_B_pol_shifted = tensor_pinv(self.G_B_pol_to_JS, contracted_dims = 2, rtol = 0)

                for i in range(r_k.size):
                    print(f'Calculating matrix for poloidal field of inclined FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
                    # Map coordinates from r_k[i] to RI:
                    theta_mapped, phi_mapped = self.mainfield.map_coords(self.RI, r_k[i], self.grid.theta, self.grid.phi)
                    mapped_grid = Grid(theta = theta_mapped, phi = phi_mapped)

                    # Matrix that gives jr at mapped grid from toroidal coefficients, shifts to r_k[i], and extracts horizontal current components
                    shifted_b_evaluator = FieldEvaluator(self.mainfield, self.grid, r_k[i])
                    mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
                    mapped_basis_evaluator = BasisEvaluator(self.jr_basis, mapped_grid)
                    m_imp_to_jr = mapped_basis_evaluator.scaled_G(self.m_imp_to_jr)
                    jr_to_JS_shifted = np.array([shifted_b_evaluator.Btheta / mapped_b_evaluator.Br,
                                                 shifted_b_evaluator.Bphi   / mapped_b_evaluator.Br])

                    m_imp_to_JS_shifted = np.einsum('ij,jk->jik', jr_to_JS_shifted, m_imp_to_jr, optimize = True)

                    # Matrix that calculates the contribution to the poloidal coefficients from the horizontal current components at r_k[i]
                    B_pol_shifted_to_B_pol = self.basis.radial_shift(r_k[i], self.RI).reshape((-1, 1, 1))
                    JS_shifted_to_B_pol = JS_shifted_to_B_pol_shifted * B_pol_shifted_to_B_pol

                    # Integration step, negative sign is to create a poloidal field that shields the region under the ionosphere from the FAC poloidal field
                    self._m_imp_to_B_pol -= Delta_k[i] * np.tensordot(JS_shifted_to_B_pol, m_imp_to_JS_shifted, 2)

        return(self._m_imp_to_B_pol)


    def set_coeffs(self, **kwargs):
        """ Set coefficients.

        Specify a set of coefficients and update the rest so that they are
        consistent.

        This function accepts one (and only one) set of coefficients.
        Valid values for kwargs (only one):

        - 'm_ind' : Coefficients for induced part of magnetic field.
        - 'm_imp' : Coefficients for imposed part of magnetic field.
        - 'Br' : Coefficients for magnetic field ``Br`` (at ``r = RI``).
        - 'jr': Coefficients for radial current scalar.

        """

        valid_kws = ['m_ind', 'm_imp', 'Phi', 'W', 'Br', 'jr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'm_ind':
            self.m_ind = Vector(self.basis, kwargs['m_ind'], type = 'scalar')
        elif key == 'm_imp':
            self.m_imp = Vector(self.basis, kwargs['m_imp'], type = 'scalar')
        elif key == 'Phi':
            self.Phi = Vector(self.basis, kwargs['Phi'], type = 'scalar')
        elif key == 'W':
            self.W = Vector(self.basis, kwargs['W'], type = 'scalar')
        elif key == 'Br':
            self.m_ind = Vector(self.basis, kwargs['Br'] / self.m_ind_to_Br, type = 'scalar')
        elif key == 'jr':
            self.m_imp = Vector(self.basis, kwargs['jr'] / self.m_imp_to_jr, type = 'scalar')
        else:
            raise Exception('This should not happen')


    def initialize_constraints(self):
        """ Initialize constraints.

        """

        if self.connect_hemispheres:
            if self.ignore_PFAC:
                raise ValueError('Hemispheres can not be connected when ignore_PFAC is True')
            if self.mainfield.kind == 'radial':
                raise ValueError('Hemispheres can not be connected with radial magnetic field')

            # Identify the high and low latitude points
            if self.mainfield.kind == 'dipole':
                self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
            elif self.mainfield.kind == 'igrf':
                mlat, _ = self.mainfield.apx.geo2apex(self.grid.lat, self.grid.lon, (self.RI - RE)*1e-3)
                self.ll_mask = np.abs(mlat) < self.latitude_boundary
            else:
                print('this should not happen')

            # The hemispheres are connected via interhemispheric currents at low latitudes
            self.G_m_imp_to_jr[self.ll_mask] += (self.jr_cp_basis_evaluator.scaled_G(self.m_imp_to_jr) * (-self.cp_b_evaluator.br / self.b_evaluator.br).reshape((-1, 1)))[self.ll_mask]

            if self.vector_jr:
                self.G_jr[self.ll_mask] = 0.0

            helmholtz_to_apex    = np.einsum('ijk,kjlm->kilm', self.b_evaluator.surface_to_apex, self.basis_evaluator.G_helmholtz, optimize = True)
            helmholtz_to_apex_cp = np.einsum('ijk,kjlm->kilm', self.cp_b_evaluator.surface_to_apex, self.cp_basis_evaluator.G_helmholtz, optimize = True)
            self.helmholtz_to_apex_ll_diff = (helmholtz_to_apex - helmholtz_to_apex_cp)[self.ll_mask]
            self.helmholtz_to_apex_ll_diff_T = tensor_transpose(self.helmholtz_to_apex_ll_diff, 2)
            self.helmholtz_to_apex_ll_diff_gram = np.tensordot(self.helmholtz_to_apex_ll_diff_T, self.helmholtz_to_apex_ll_diff, 2)

            if self.vector_u:
                self.A_u = -np.tensordot(self.helmholtz_to_apex_ll_diff, self.u_coeffs_to_helmholtz_E, 2)

        self.G_m_imp_to_jr_gram = self.G_m_imp_to_jr.T.dot(self.G_m_imp_to_jr)

        if self.vector_jr:
            self.jr_m_imp_matrix = self.G_m_imp_to_jr.T.dot(self.G_jr)


    def impose_constraints(self):
        """ Impose constraints, if any. Leads to a contribution to m_imp from
        m_ind if the hemispheres are connected.

        """

        if self.connect_hemispheres:
            E_m_ind = self.m_ind_to_helmholtz_E.dot(self.m_ind.coeffs)
            self.c = -np.tensordot(self.helmholtz_to_apex_ll_diff, E_m_ind, 2)
            if self.neutral_wind:
                self.c += self.cu

            E_c = np.tensordot(tensor_transpose(self.helmholtz_to_apex_ll_diff, 2), self.c, 2)
            self.GT_constraint_vector = self.G_m_imp_to_jr.T.dot(self.jr_on_grid) + np.einsum('ijj->i', np.tensordot(self.m_imp_to_helmholtz_E_direct.T, E_c, 1)) * self.ih_constraint_scaling**2

            self.set_coeffs(m_imp = self.GTG_constraints_inv.dot(self.GT_constraint_vector))

        else:
            self.set_coeffs(jr = self.jr.coeffs)


    def set_jr(self, jr, vector_jr = True):
        """
        Specify radial current at ``self.grid.theta``,
        ``self.grid.phi``.

            Parameters
            ----------
            jr: array
                The radial current, in A/m^2, at
                ``self.grid.theta`` and ``self.grid.phi``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        if vector_jr:
            self.jr = jr

            self.jr_on_grid = self.G_jr.dot(self.jr.coeffs)

        else:
            self.jr_on_grid = jr
            if self.connect_hemispheres:
                self.jr_on_grid[self.ll_mask] = 0


    def set_u(self, u, vector_u = True):
        """ Set neutral wind theta and phi components.

        """
        from pynamit.cubed_sphere.cubed_sphere import csp

        self.neutral_wind = True

        if vector_u:
            self.u = u
            self.helmholtz_E_u = np.tensordot(self.u_coeffs_to_helmholtz_E, np.moveaxis(np.array(np.split(self.u.coeffs, 2)), 0, 1), 2)
        else:
            self.u_theta_on_grid, self.u_phi_on_grid = np.split(u, 2)
            self.helmholtz_E_u = np.tensordot(self.u_to_helmholtz_E, np.moveaxis(np.array(np.split(u, 2)), 0, 1), 2)

        if self.connect_hemispheres:
            if vector_u:
                self.cu = np.tensordot(self.A_u, np.moveaxis(np.array(np.split(self.u.coeffs, 2)), 0, 1), 2)
            else:
                self.cu = -np.tensordot(self.helmholtz_to_apex_ll_diff, self.helmholtz_E_u, 2)


    def set_conductance(self, etaP, etaH, vector_conductance = True):
        """
        Specify Hall and Pedersen conductance at
        ``self.grid.theta``, ``self.grid.phi``.

        """
        from pynamit.cubed_sphere.cubed_sphere import csp

        self.conductance = True

        if self.vector_conductance:
            self.etaP = etaP
            self.etaH = etaH

        else:
            self.etaP = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaP, type = 'scalar')
            self.etaH = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaH, type = 'scalar')

            etaP_on_grid = etaP
            etaH_on_grid = etaH

        if TRIPLE_PRODUCT and self.vector_conductance:
            self.m_ind_to_helmholtz_E_direct = self.etaP_m_ind_to_helmholtz_E.dot(self.etaP.coeffs) + self.etaH_m_ind_to_helmholtz_E.dot(self.etaH.coeffs)
            self.m_imp_to_helmholtz_E_direct = self.etaP_m_imp_to_helmholtz_E.dot(self.etaP.coeffs) + self.etaH_m_imp_to_helmholtz_E.dot(self.etaH.coeffs)
        
        else:
            if self.vector_conductance:
                etaP_on_grid = etaP.to_grid(self.conductance_basis_evaluator)
                etaH_on_grid = etaH.to_grid(self.conductance_basis_evaluator)

            G_m_ind_to_E_direct = tensor_scale_left(etaP_on_grid, self.m_ind_to_bP_JS) + tensor_scale_left(etaH_on_grid, self.m_ind_to_bH_JS)
            G_m_imp_to_E_direct = tensor_scale_left(etaP_on_grid, self.m_imp_to_bP_JS) + tensor_scale_left(etaH_on_grid, self.m_imp_to_bH_JS)

            self.m_ind_to_helmholtz_E_direct = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, G_m_ind_to_E_direct, 2), 2)
            self.m_imp_to_helmholtz_E_direct = np.tensordot(self.basis_evaluator.GTWG_plus_R_inv_helmholtz, np.tensordot(self.basis_evaluator.GTW_helmholtz, G_m_imp_to_E_direct, 2), 2)

        self.GTG_constraints = self.G_m_imp_to_jr_gram

        # Add low latitude E field constraints
        if self.connect_hemispheres:
            m_imp_to_helmholtz_E_direct_T = tensor_transpose(self.m_imp_to_helmholtz_E_direct)
            m_imp_apex_ll_diff_gram = np.tensordot(m_imp_to_helmholtz_E_direct_T, self.helmholtz_to_apex_ll_diff_gram, 2)
            self.GTG_constraints += np.tensordot(m_imp_apex_ll_diff_gram, self.m_imp_to_helmholtz_E_direct, 2) * self.ih_constraint_scaling**2

        self.GTG_constraints_inv = pinv_positive_semidefinite(self.GTG_constraints)
        GTG_constraints_to_helmholtz_E = self.m_imp_to_helmholtz_E_direct.dot(self.GTG_constraints_inv)

        if self.vector_jr:
            self.jr_coeffs_to_helmholtz_E = GTG_constraints_to_helmholtz_E.dot(self.jr_m_imp_matrix)
        else:
            self.jr_to_helmholtz_E = GTG_constraints_to_helmholtz_E.dot(self.G_m_imp_to_jr.T)

        self.m_ind_to_helmholtz_E = self.m_ind_to_helmholtz_E_direct

        if self.connect_hemispheres:
            intermediate = np.tensordot(GTG_constraints_to_helmholtz_E, m_imp_to_helmholtz_E_direct_T, 1) * self.ih_constraint_scaling**2
            intermediate2 = np.tensordot(intermediate, self.helmholtz_to_apex_ll_diff_gram, 2)

            m_ind_to_helmholtz_E_constraints = -np.tensordot(intermediate2, self.m_ind_to_helmholtz_E_direct, 2)
            self.m_ind_to_helmholtz_E += m_ind_to_helmholtz_E_constraints

            if self.vector_u:
                self.u_coeffs_to_helmholtz_E_constraints = -np.tensordot(intermediate2, self.u_coeffs_to_helmholtz_E, 2)
            else:
                self.cu_to_helmholtz_E = np.tensordot(intermediate, self.helmholtz_to_apex_ll_diff_T, 2)

        self.m_ind_to_helmholtz_E_cf_inv = np.linalg.pinv(self.m_ind_to_helmholtz_E[:,1])

    def update_Phi_and_W(self):
        """ Update the coefficients for the electric potential and the induction electric field.

        """

        E = self.m_ind_to_helmholtz_E.dot(self.m_ind.coeffs)

        if self.vector_jr:
            E += self.jr_coeffs_to_helmholtz_E.dot(self.jr.coeffs)
        else:
            E += self.jr_to_helmholtz_E.dot(self.jr_on_grid)

        if self.neutral_wind:
            E += self.helmholtz_E_u

            if self.connect_hemispheres:
                if self.vector_u:
                    E += np.tensordot(self.u_coeffs_to_helmholtz_E_constraints, np.moveaxis(np.array(np.split(self.u.coeffs, 2)), 0, 1), 2)
                else:
                    E += np.tensordot(self.cu_to_helmholtz_E, self.cu, 2)

        self.Phi = Vector(self.basis, coeffs = E[:,0], type = 'scalar')
        self.W = Vector(self.basis, coeffs = E[:,1], type = 'scalar')


    def evolve_Br(self, dt):
        """ Evolve ``Br`` in time.

        """

        new_Br = self.m_ind.coeffs * self.m_ind_to_Br + self.W.coeffs * self.W_to_dBr_dt * dt

        self.set_coeffs(Br = new_Br)


    def get_Br(self, _basis_evaluator):
        """ Calculate ``Br``.

        """

        return(_basis_evaluator.basis_to_grid(self.m_ind.coeffs * self.m_ind_to_Br))


    def get_JS(self): # for now, JS is always returned on self.grid!
        """ Calculate ionospheric sheet current.

        """
        Js_ind, Je_ind = np.split(self.G_m_ind_to_JS.dot(self.m_ind.coeffs), 2, axis = 0)
        Js_imp, Je_imp = np.split(self.G_m_imp_to_JS.dot(self.m_imp.coeffs), 2, axis = 0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp


        return(Jth, Jph)


    def get_jr(self, _basis_evaluator):
        """ Calculate radial current.

        """

        return _basis_evaluator.basis_to_grid(self.m_imp.coeffs * self.m_imp_to_jr)


    def get_Jeq(self, _basis_evaluator):
        """ Calculate equivalent current function.

        """

        return _basis_evaluator.basis_to_grid(self.m_ind.coeffs * self.m_ind_to_Jeq)


    def get_Phi(self, _basis_evaluator):
        """ Calculate Phi.

        """

        return _basis_evaluator.basis_to_grid(self.Phi.coeffs)


    def get_W(self, _basis_evaluator):
        """ Calculate the induction electric field scalar.

        """

        return _basis_evaluator.basis_to_grid(self.W.coeffs)


    def get_E(self): # for now, E is always returned on self.grid!
        """ Calculate electric field.

        """

        if not self.conductance:
            raise ValueError('Conductance must be set before calculating electric field')

        Jth, Jph = self.get_JS()

        E = self.Jth_to_E * np.tile(Jth, 2) + self.Jph_to_E * np.tile(Jph, 2)

        if self.neutral_wind:
            uxB_theta =  self.u_phi_on_grid   * self.b_evaluator.Br
            uxB_phi   = -self.u_theta_on_grid * self.b_evaluator.Br

            E -= np.hstack((uxB_theta, uxB_phi))

        return E


    def steady_state_m_ind(self):
        """ Calculate coefficients for induced field in steady state 
            
            Parameters:
            -----------
            m_imp: array, optional
                the coefficient vector for the imposed magnetic field. If None, the
                vector for the current state will be used

            Returns:
            --------
            m_ind_ss: array
                array of coefficients for the induced magnetic field in steady state

        """

        if self.vector_jr:
            helmholtz_E_noind = self.jr_coeffs_to_helmholtz_E.dot(self.jr.coeffs)
        else:
            helmholtz_E_noind = self.jr_to_helmholtz_E.dot(self.jr_on_grid)

        if self.neutral_wind:
            helmholtz_E_noind += self.helmholtz_E_u

            if self.connect_hemispheres:
                if self.vector_u:
                    helmholtz_E_noind += np.tensordot(self.u_coeffs_to_helmholtz_E_constraints, np.moveaxis(np.array(np.split(self.u.coeffs, 2)), 0, 1), 2)
                else:
                    helmholtz_E_noind += np.tensordot(self.cu_to_helmholtz_E, self.cu, 2)

        m_ind = -self.m_ind_to_helmholtz_E_cf_inv.dot(helmholtz_E_noind[:,1])

        return(m_ind)