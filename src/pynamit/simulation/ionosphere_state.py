import numpy as np
import xarray as xr
from pynamit.various.constants import mu0, RE
from pynamit.primitives.grid import Grid
from pynamit.primitives.vector import Vector
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator


class State(object):
    """ State of the ionosphere.

    """

    def __init__(self, basis, conductance_basis, u_basis, mainfield, num_grid, settings, PFAC_matrix = None):
        """ Initialize the state of the ionosphere.
    
        """

        self.basis = basis
        self.conductance_basis = conductance_basis
        self.u_basis = u_basis
        self.mainfield = mainfield
        self.num_grid = num_grid

        self.RI                     = settings.RI
        self.latitude_boundary      = settings.latitude_boundary
        self.ignore_PFAC            = bool(settings.ignore_PFAC)
        self.connect_hemispheres    = bool(settings.connect_hemispheres)
        self.zero_jr_at_dip_equator = bool(settings.zero_jr_at_dip_equator)
        self.FAC_integration_steps  = settings.FAC_integration_steps
        self.ih_constraint_scaling  = settings.ih_constraint_scaling

        if PFAC_matrix is not None:
            self._m_imp_to_B_pol = PFAC_matrix

        # Spherical harmonic conversion factors
        self.m_ind_to_Br  = self.RI * self.basis.d_dr(self.RI)
        self.m_imp_to_Jr  = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.EW_to_dBr_dt = -self.RI * self.basis.laplacian(self.RI)
        self.m_ind_to_Jeq = self.RI / mu0 * self.basis.surface_discontinuity

        # Initialize grid-related objects
        self.basis_evaluator             = BasisEvaluator(self.basis,             num_grid)
        self.conductance_basis_evaluator = BasisEvaluator(self.conductance_basis, num_grid)
        self.u_basis_evaluator           = BasisEvaluator(self.u_basis,           num_grid)

        self.b_evaluator = FieldEvaluator(mainfield, num_grid, self.RI)

        self.G_B_pol_to_JS = self.basis_evaluator.G_rxgrad * self.basis.surface_discontinuity / mu0
        self.G_B_tor_to_JS = -self.basis_evaluator.G_grad / mu0
        self.G_m_ind_to_JS = self.G_B_pol_to_JS
        self.G_m_imp_to_JS = self.G_B_tor_to_JS + self.G_B_pol_to_JS.dot(self.m_imp_to_B_pol.values)

        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(self.RI, num_grid.theta, num_grid.phi)
            self.cp_grid = Grid(theta = cp_theta, phi = cp_phi)

            self.cp_basis_evaluator             = BasisEvaluator(self.basis,             self.cp_grid)
            self.conductance_cp_basis_evaluator = BasisEvaluator(self.conductance_basis, self.cp_grid)
            self.u_cp_basis_evaluator           = BasisEvaluator(self.u_basis,           self.cp_grid)

            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, self.RI)

            self.G_B_pol_to_JS_cp = self.cp_basis_evaluator.G_rxgrad * self.basis.surface_discontinuity / mu0
            self.G_B_tor_to_JS_cp = -self.cp_basis_evaluator.G_grad / mu0
            self.G_m_ind_to_JS_cp = self.G_B_pol_to_JS_cp
            self.G_m_imp_to_JS_cp = self.G_B_tor_to_JS_cp + self.G_B_pol_to_JS_cp.dot(self.m_imp_to_B_pol.values)

        self.initialize_constraints()

        # Initialize the spherical harmonic coefficients
        self.set_coeffs(m_ind = np.zeros(self.basis.num_coeffs))
        self.set_coeffs(m_imp = np.zeros(self.basis.num_coeffs))

        # Neutral wind and conductance should be set after I2D initialization
        self.neutral_wind = False
        self.conductance  = False

        # Construct the matrix elements used to calculate the electric field
        self.b00 = self.b_evaluator.bphi**2 + self.b_evaluator.br**2
        self.b01 = -self.b_evaluator.btheta * self.b_evaluator.bphi
        self.b10 = -self.b_evaluator.btheta * self.b_evaluator.bphi
        self.b11 = self.b_evaluator.btheta**2 + self.b_evaluator.br**2


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
                data = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs)),
                coords = {
                    'i': np.arange(self.basis.num_coeffs),
                    'j': np.arange(self.basis.num_coeffs),
                },
                dims = ['i', 'j']
            )

            if not (self.mainfield.kind == 'radial' or self.ignore_PFAC):
                r_k_steps = self.FAC_integration_steps
                Delta_k = np.diff(r_k_steps)
                r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

                JS_shifted_to_B_pol_shifted = np.linalg.pinv(self.G_B_pol_to_JS, rcond = 0)

                for i in range(r_k.size):
                    print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
                    # Map coordinates from r_k[i] to RI:
                    theta_mapped, phi_mapped = self.mainfield.map_coords(self.RI, r_k[i], self.num_grid.theta, self.num_grid.phi)
                    mapped_grid = Grid(theta = theta_mapped, phi = phi_mapped)

                    # Matrix that gives FAC at mapped grid from toroidal coefficients, shifts to r_k[i], and extracts horizontal components
                    shifted_b_evaluator = FieldEvaluator(self.mainfield, self.num_grid, r_k[i])
                    mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
                    mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)
                    m_imp_to_Jpar = mapped_basis_evaluator.scaled_G(self.m_imp_to_Jr / mapped_b_evaluator.br.reshape((-1 ,1)))
                    Jpar_to_JS_shifted = ((shifted_b_evaluator.Btheta / mapped_b_evaluator.B_magnitude).reshape((-1, 1)),
                                          (shifted_b_evaluator.Bphi   / mapped_b_evaluator.B_magnitude).reshape((-1, 1)))
                    m_imp_to_JS_shifted = np.vstack(m_imp_to_Jpar * Jpar_to_JS_shifted)

                    # Matrix that calculates the contribution to the poloidal coefficients from the horizontal components at r_k[i]
                    B_pol_shifted_to_B_pol = self.basis.radial_shift(r_k[i], self.RI).reshape((-1, 1))
                    JS_shifted_to_B_pol = JS_shifted_to_B_pol_shifted * B_pol_shifted_to_B_pol

                    # Integration step, negative sign is to create a poloidal field that shields the region under the ionosphere from the FAC poloidal field
                    self._m_imp_to_B_pol -= Delta_k[i] * JS_shifted_to_B_pol.dot(m_imp_to_JS_shifted)

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
        - 'Jr': Coefficients for radial current scalar.

        """

        valid_kws = ['m_ind', 'm_imp', 'Br', 'Jr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'm_ind':
            self.m_ind = Vector(self.basis, kwargs['m_ind'])
        elif key == 'm_imp':
            self.m_imp = Vector(self.basis, kwargs['m_imp'])
        elif key == 'Br':
            self.m_ind = Vector(self.basis, kwargs['Br'] / self.m_ind_to_Br)
        elif key == 'Jr':
            self.m_imp = Vector(self.basis, kwargs['Jr'] / self.m_imp_to_Jr)
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
                self.ll_mask = np.abs(self.num_grid.lat) < self.latitude_boundary
            elif self.mainfield.kind == 'igrf':
                mlat, _ = self.mainfield.apx.geo2apex(self.num_grid.lat, self.num_grid.lon, (self.RI - RE)*1e-3)
                self.ll_mask = np.abs(mlat) < self.latitude_boundary
            else:
                print('this should not happen')

            # Calculate the matrices that convert m_imp to FAC on num_grid and conjugate grid
            G_Jpar    =    self.basis_evaluator.scaled_G(self.m_imp_to_Jr /    self.b_evaluator.br.reshape((-1 ,1)))
            G_Jpar_cp = self.cp_basis_evaluator.scaled_G(self.m_imp_to_Jr / self.cp_b_evaluator.br.reshape((-1 ,1)))

            # Calculate matrix that ensures high latitude FACs are unaffected by other constraints
            self.G_Jpar_hl = G_Jpar[~self.ll_mask]

            # Calculate matrix that constrains outwards FACs at low latitude points to be equal to inwards FACs at conjugate points
            self.G_Jpar_ll_diff = (G_Jpar[self.ll_mask] - G_Jpar_cp[self.ll_mask])

            # Calculate constraint matrices for low latitude points and their conjugate points:
            self.aeP_ind_ll = self.b_evaluator.aeP.dot(self.G_m_ind_to_JS)[np.tile(self.ll_mask, 2)]
            self.aeH_ind_ll = self.b_evaluator.aeH.dot(self.G_m_ind_to_JS)[np.tile(self.ll_mask, 2)]
            self.aeP_imp_ll = self.b_evaluator.aeP.dot(self.G_m_imp_to_JS)[np.tile(self.ll_mask, 2)]
            self.aeH_imp_ll = self.b_evaluator.aeH.dot(self.G_m_imp_to_JS)[np.tile(self.ll_mask, 2)]
            self.aeP_ind_cp_ll = self.cp_b_evaluator.aeP.dot(self.G_m_ind_to_JS_cp)[np.tile(self.ll_mask, 2)]
            self.aeH_ind_cp_ll = self.cp_b_evaluator.aeH.dot(self.G_m_ind_to_JS_cp)[np.tile(self.ll_mask, 2)]
            self.aeP_imp_cp_ll = self.cp_b_evaluator.aeP.dot(self.G_m_imp_to_JS_cp)[np.tile(self.ll_mask, 2)]
            self.aeH_imp_cp_ll = self.cp_b_evaluator.aeH.dot(self.G_m_imp_to_JS_cp)[np.tile(self.ll_mask, 2)]

            if self.zero_jr_at_dip_equator:
                # Calculate matrix that converts m_imp to Jr at dip equator
                n_phi = self.basis.minimum_phi_sampling()
                dip_equator_phi = np.linspace(0, 360, n_phi)
                self.dip_equator_basis_evaluator = BasisEvaluator(self.basis, Grid(theta = self.mainfield.dip_equator(dip_equator_phi), phi = dip_equator_phi))

                _equation_scaling = self.num_grid.lat[self.ll_mask].size / n_phi # scaling to match importance of other equations
                self.G_Jr_dip_equator = self.dip_equator_basis_evaluator.scaled_G(self.m_imp_to_Jr) * _equation_scaling
            else:
                # Make zero-row stand-in for the Jr matrix
                self.G_Jr_dip_equator = np.empty((0, self.basis.num_coeffs))


    def impose_constraints(self):
        """ Impose constraints, if any. Leads to a contribution to m_imp from
        m_ind if the hemispheres are connected.

        """

        if self.connect_hemispheres:
            self.c = self.A_ind.dot(self.m_ind.coeffs)
            if self.neutral_wind:
                self.c += self.cu

            self.constraint_vector = np.hstack((self.Jpar_on_grid[~self.ll_mask], np.zeros(self.G_Jpar_ll_diff.shape[0]), np.zeros(self.G_Jr_dip_equator.shape[0]), self.c * self.ih_constraint_scaling ))

            self.set_coeffs(m_imp = self.G_m_imp_constraints_inv.dot(self.constraint_vector))
        else:
            self.set_coeffs(Jr = self.Jr.coeffs)


    def set_FAC(self, Jr, vector_FAC = True):
        """
        Specify field-aligned current at ``self.num_grid.theta``,
        ``self.num_grid.phi``.

            Parameters
            ----------
            FAC: array
                The field-aligned current, in A/m^2, at
                ``self.num_grid.theta`` and ``self.num_grid.phi``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        if vector_FAC:
            self.Jr = Jr

            if self.connect_hemispheres:
                self.Jpar_on_grid = Jr.to_grid(self.basis_evaluator) / self.b_evaluator.br

        else:
            self.Jr = Vector(basis = self.basis, basis_evaluator = self.basis_evaluator, grid_values = Jr)

            if self.connect_hemispheres:
                self.Jpar_on_grid = Jr / self.b_evaluator.br


    def set_u(self, u, vector_u = True):
        """ Set neutral wind theta and phi components.

        """
        from pynamit.cubed_sphere.cubed_sphere import csp

        self.neutral_wind = True

        if vector_u:
            self.u = u
            self.u_theta_on_grid, self.u_phi_on_grid = self.u.to_grid(self.u_basis_evaluator)

        else:
            self.u = Vector(basis = self.u_basis, basis_evaluator = self.u_basis_evaluator, grid_values = u, helmholtz = True)
            self.u_theta_on_grid, self.u_phi_on_grid = u

        self.uxB_theta =  self.u_phi_on_grid   * self.b_evaluator.Br
        self.uxB_phi   = -self.u_theta_on_grid * self.b_evaluator.Br

        if self.connect_hemispheres:
            if vector_u:
                # Represent as values on cp_grid
                u_theta_on_cp_grid, u_phi_on_cp_grid = self.u.to_grid(self.u_cp_basis_evaluator)
            else:
                u_cp_int = csp.interpolate_vector_components(self.u_phi_on_grid, -self.u_theta_on_grid, np.zeros_like(self.u_phi_on_grid), self.u_basis_evaluator.grid.theta, self.u_basis_evaluator.grid.phi, self.u_cp_basis_evaluator.grid.theta, self.u_cp_basis_evaluator.grid.phi)
                u_theta_on_cp_grid, u_phi_on_cp_grid = -u_cp_int[1], u_cp_int[0]

            # Neutral wind at low latitude grid points and at their conjugate points
            u_theta_ll    = self.u_theta_on_grid[self.ll_mask]
            u_phi_ll      = self.u_phi_on_grid[self.ll_mask]
            u_theta_cp_ll = u_theta_on_cp_grid[self.ll_mask]
            u_phi_cp_ll   = u_phi_on_cp_grid[self.ll_mask]

            # Constraint vector contribution from wind
            self.cu =  (np.tile(u_theta_cp_ll, 2) * self.cp_b_evaluator.aut[np.tile(self.ll_mask, 2)] + np.tile(u_phi_cp_ll, 2) * self.cp_b_evaluator.aup[np.tile(self.ll_mask, 2)]) \
                      -(np.tile(u_theta_ll,    2) *    self.b_evaluator.aut[np.tile(self.ll_mask, 2)] + np.tile(u_phi_ll,    2) *    self.b_evaluator.aup[np.tile(self.ll_mask, 2)])


    def set_conductance(self, etaP, etaH, vector_conductance = True):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.phi``.

        """
        from pynamit.cubed_sphere.cubed_sphere import csp

        self.conductance = True

        if vector_conductance:
            self.etaP = etaP
            self.etaH = etaH

            # Represent as values on num_grid
            self.etaP_on_grid = etaP.to_grid(self.conductance_basis_evaluator)
            self.etaH_on_grid = etaH.to_grid(self.conductance_basis_evaluator)

        else:
            self.etaP = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaP)
            self.etaH = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaH)

            self.etaP_on_grid = etaP
            self.etaH_on_grid = etaH

        if self.connect_hemispheres:
            if vector_conductance:
                # Represent as values on cp_grid
                etaP_on_cp_grid = etaP.to_grid(self.conductance_cp_basis_evaluator)
                etaH_on_cp_grid = etaH.to_grid(self.conductance_cp_basis_evaluator)
            else:
                etaP_on_cp_grid = csp.interpolate_scalar(self.etaP_on_grid, self.conductance_basis_evaluator.grid.theta, self.conductance_basis_evaluator.grid.phi, self.conductance_cp_basis_evaluator.grid.theta, self.conductance_cp_basis_evaluator.grid.phi)
                etaH_on_cp_grid = csp.interpolate_scalar(self.etaH_on_grid, self.conductance_basis_evaluator.grid.theta, self.conductance_basis_evaluator.grid.phi, self.conductance_cp_basis_evaluator.grid.theta, self.conductance_cp_basis_evaluator.grid.phi)

            # Resistances at low latitude grid points and at their conjugate points
            etaP_ll    = self.etaP_on_grid[self.ll_mask]
            etaH_ll    = self.etaH_on_grid[self.ll_mask]
            etaP_cp_ll = etaP_on_cp_grid[self.ll_mask]
            etaH_cp_ll = etaH_on_cp_grid[self.ll_mask]

            # Conductance-dependent constraint matrices
            self.A_ind =  (np.tile(etaP_cp_ll, 2).reshape((-1, 1)) * self.aeP_ind_cp_ll + np.tile(etaH_cp_ll, 2).reshape((-1, 1)) * self.aeH_ind_cp_ll) \
                         -(np.tile(etaP_ll,    2).reshape((-1, 1)) * self.aeP_ind_ll    + np.tile(etaH_ll,    2).reshape((-1, 1)) * self.aeH_ind_ll)

            self.A_imp =  (np.tile(etaP_ll,    2).reshape((-1, 1)) * self.aeP_imp_ll    + np.tile(etaH_ll,    2).reshape((-1, 1)) * self.aeH_imp_ll) \
                         -(np.tile(etaP_cp_ll, 2).reshape((-1, 1)) * self.aeP_imp_cp_ll + np.tile(etaH_cp_ll, 2).reshape((-1, 1)) * self.aeH_imp_cp_ll)

            # Combine constraint matrices
            self.G_m_imp_constraints = np.vstack((self.G_Jpar_hl, self.G_Jpar_ll_diff, self.G_Jr_dip_equator, self.A_imp * self.ih_constraint_scaling))
            self.G_m_imp_constraints_inv = np.linalg.pinv(self.G_m_imp_constraints, rcond = 0)


    def update_Phi_and_EW(self):
        """ Update the coefficients for the electric potential and the induction electric field.

        """

        E_cf, E_df = self.basis_evaluator.grid_to_basis(self.get_E(), helmholtz = True)

        self.Phi = Vector(self.basis, coeffs = E_cf)
        self.EW = Vector(self.basis, coeffs = E_df)


    def evolve_Br(self, dt):
        """ Evolve ``Br`` in time.

        """

        self.update_Phi_and_EW()

        new_Br = self.m_ind.coeffs * self.m_ind_to_Br + self.EW.coeffs * self.EW_to_dBr_dt * dt

        self.set_coeffs(Br = new_Br)


    def get_Br(self, _basis_evaluator):
        """ Calculate ``Br``.

        """

        return(_basis_evaluator.basis_to_grid(self.m_ind.coeffs * self.m_ind_to_Br))


    def get_JS(self): # for now, JS is always returned on num_grid!
        """ Calculate ionospheric sheet current.

        """
        Js_ind, Je_ind = np.split(self.G_m_ind_to_JS.dot(self.m_ind.coeffs), 2, axis = 0)
        Js_imp, Je_imp = np.split(self.G_m_imp_to_JS.dot(self.m_imp.coeffs), 2, axis = 0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp


        return(Jth, Jph)


    def get_Jr(self, _basis_evaluator):
        """ Calculate radial current.

        """

        return _basis_evaluator.basis_to_grid(self.m_imp.coeffs * self.m_imp_to_Jr)


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

        return _basis_evaluator.basis_to_grid(self.EW.coeffs)


    def get_E(self): # for now, E is always returned on num_grid!
        """ Calculate electric field.

        """

        if not self.conductance:
            raise ValueError('Conductance must be set before calculating electric field')

        Jth, Jph = self.get_JS()

        Eth = self.etaP_on_grid * (self.b00 * Jth + self.b01 * Jph) + self.etaH_on_grid * ( self.b_evaluator.br * Jph)
        Eph = self.etaP_on_grid * (self.b10 * Jth + self.b11 * Jph) + self.etaH_on_grid * (-self.b_evaluator.br * Jth)

        if self.neutral_wind:
            Eth -= self.uxB_theta
            Eph -= self.uxB_phi

        return(Eth, Eph)
