import numpy as np
from pynamit.primitives.grid import Grid
from pynamit.constants import mu0, RE
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.cubedsphere.cubedsphere import csp
from pynamit.primitives.vector import Vector
from pynamit.primitives.field_evaluator import FieldEvaluator


class State(object):
    """ State of the ionosphere.

    """

    def __init__(self, sh, mainfield, num_grid, RI, ignore_PFAC, FAC_integration_steps, connect_hemispheres, latitude_boundary, zero_jr_at_dip_equator, ih_constraint_scaling = 1e-5, PFAC_matrix = None):
        """ Initialize the state of the ionosphere.
    
        """

        self.sh = sh
        self.basis = sh # SHBasis for now, but can in principle be any basis if we generalize
        self.mainfield = mainfield
        self.num_grid = num_grid
        self.FAC_integration_steps = FAC_integration_steps

        self.RI = RI
        self.ignore_PFAC = ignore_PFAC
        self.zero_jr_at_dip_equator = zero_jr_at_dip_equator
        self.ih_constraint_scaling = ih_constraint_scaling

        self.connect_hemispheres = connect_hemispheres
        self.latitude_boundary = latitude_boundary

        # Spherical harmonic identities
        d_dr            = -self.sh.n / self.RI
        laplacian       = -self.sh.n * (self.sh.n + 1) / self.RI**2
        V_discontinuity = (2 * self.sh.n + 1) / (self.sh.n + 1)

        # Spherical harmonic conversion factors
        self.VB_ind_to_Br  = self.RI * d_dr
        self.TB_imp_to_Jr  = self.RI / mu0 * laplacian
        self.EW_to_dBr_dt  = -self.RI * laplacian
        self.VB_ind_to_Jeq = self.RI / mu0 * V_discontinuity

        # Initialize grid-related objects
        self.basis_evaluator = BasisEvaluator(self.basis, num_grid)
        self.b_evaluator = FieldEvaluator(mainfield, num_grid, RI)
        self.G_VB_ind_to_JS_ind = self.basis_evaluator.G_rxgrad * V_discontinuity / mu0
        self.G_VB_imp_to_JS_imp = self.G_VB_ind_to_JS_ind
        self.G_TB_imp_to_JS_imp = -self.basis_evaluator.G_grad / mu0 + self.G_VB_imp_to_JS_imp.dot(self.TB_imp_to_VB_imp)

        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(self.RI, num_grid.theta, num_grid.lon)
            self.cp_grid = Grid(90 - cp_theta, cp_phi)

            self.cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.cp_b_evaluator = FieldEvaluator(mainfield, self.cp_grid, RI)
            self.G_VB_ind_to_JS_ind_cp = self.cp_basis_evaluator.G_rxgrad * V_discontinuity / mu0
            self.G_VB_imp_to_JS_imp_cp = self.G_VB_ind_to_JS_ind_cp
            self.G_TB_imp_to_JS_imp_cp = -self.cp_basis_evaluator.G_grad / mu0 + self.G_VB_imp_to_JS_imp_cp.dot(self.TB_imp_to_VB_imp)

        self.initialize_constraints()

        # Initialize the spherical harmonic coefficients
        self.set_coeffs(VB_ind = np.zeros(self.basis.num_coeffs))
        self.set_coeffs(TB_imp = np.zeros(self.basis.num_coeffs))

        # Neutral wind and conductance should be set after I2D initialization
        self.neutral_wind = False
        self.conductance  = False

        # Construct the matrix elements used to calculate the electric field
        self.b00 = self.b_evaluator.bphi**2 + self.b_evaluator.br**2
        self.b01 = -self.b_evaluator.btheta * self.b_evaluator.bphi
        self.b10 = -self.b_evaluator.btheta * self.b_evaluator.bphi
        self.b11 = self.b_evaluator.btheta**2 + self.b_evaluator.br**2


    @property
    def TB_imp_to_VB_imp(self):
        """
        Return matrix that maps from self.TB_imp to coefficients for
        poloidal field of FACs. Uses the method by Engels and Olsen 1998,
        Eq. 13 to account for poloidal part of magnetic field for FACs.

        """

        if not hasattr(self, '_TB_imp_to_VB_imp'):

            if self.mainfield.kind == 'radial' or self.ignore_PFAC: # no Poloidal field so get matrix of zeros
                self._TB_imp_to_VB_imp = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs))

            else:
                r_k_steps = self.FAC_integration_steps
                Delta_k = np.diff(r_k_steps)
                r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

                JS_shifted_to_VB_shifted = np.linalg.pinv(self.G_VB_imp_to_JS_imp, rcond = 0)

                self._TB_imp_to_VB_imp = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs))

                for i in range(r_k.size):
                    print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
                    # Map coordinates from r_k[i] to RI:
                    theta_mapped, phi_mapped = self.mainfield.map_coords(self.RI, r_k[i], self.num_grid.theta, self.num_grid.lon)
                    mapped_grid = Grid(90 - theta_mapped, phi_mapped)

                    # Matrix that gives FAC at mapped grid from toroidal coefficients, shifts to r_k[i], and extracts horizontal components
                    shifted_b_evaluator = FieldEvaluator(self.mainfield, self.num_grid, r_k[i])
                    mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
                    mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)
                    TB_imp_to_Jpar = mapped_basis_evaluator.scaled_G(self.TB_imp_to_Jr / mapped_b_evaluator.br.reshape((-1 ,1)))
                    Jpar_to_JS_shifted = ((shifted_b_evaluator.Btheta / mapped_b_evaluator.B_magnitude).reshape((-1, 1)),
                                          (shifted_b_evaluator.Bphi   / mapped_b_evaluator.B_magnitude).reshape((-1, 1)))
                    TB_imp_to_JS_shifted = np.vstack(TB_imp_to_Jpar * Jpar_to_JS_shifted)

                    # Matrix that calculates the contribution to the poloidal coefficients from the horizontal components at r_k[i]
                    VB_shifted_to_VB_imp = (self.RI / r_k[i])**(self.sh.n - 1).reshape((-1, 1))
                    JS_shifted_to_VB_imp = JS_shifted_to_VB_shifted * VB_shifted_to_VB_imp

                    # Integration step
                    self._TB_imp_to_VB_imp -= Delta_k[i] * JS_shifted_to_VB_imp.dot(TB_imp_to_JS_shifted) # NB: where does the negative sign come from?

        return(self._TB_imp_to_VB_imp)


    def set_coeffs(self, **kwargs):
        """ Set coefficients.

        Specify a set of coefficients and update the rest so that they are
        consistent.

        This function accepts one (and only one) set of coefficients.
        Valid values for kwargs (only one):

        - 'VB_ind' : Coefficients for magnetic field scalar ``V``.
        - 'TB_imp' : Coefficients for surface current scalar ``T``.
        - 'Br' : Coefficients for magnetic field ``Br`` (at ``r = RI``).
        - 'Jr': Coefficients for radial current scalar.

        """

        valid_kws = ['VB_ind', 'TB_imp', 'Br', 'Jr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'VB_ind':
            self.VB_ind = Vector(self.basis, kwargs['VB_ind'])
        elif key == 'TB_imp':
            self.TB_imp = Vector(self.basis, kwargs['TB_imp'])
        elif key == 'Br':
            self.VB_ind = Vector(self.basis, kwargs['Br'] / self.VB_ind_to_Br)
        elif key == 'Jr':
            self.TB_imp = Vector(self.basis, kwargs['Jr'] / self.TB_imp_to_Jr)
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

            # Calculate the matrices that convert TB_imp to FAC on num_grid and conjugate grid
            G_Jpar    =    self.basis_evaluator.scaled_G(self.TB_imp_to_Jr /    self.b_evaluator.br.reshape((-1 ,1)))
            G_Jpar_cp = self.cp_basis_evaluator.scaled_G(self.TB_imp_to_Jr / self.cp_b_evaluator.br.reshape((-1 ,1)))

            # Calculate matrix that ensures high latitude FACs are unaffected by other constraints
            self.G_Jpar_hl = G_Jpar[~self.ll_mask]

            # Calculate matrix that constrains outwards FACs at low latitude points to be equal to inwards FACs at conjugate points
            self.G_Jpar_ll_diff = (G_Jpar[self.ll_mask] - G_Jpar_cp[self.ll_mask])

            # Calculate constraint matrices for low latitude points and their conjugate points:
            self.aeP_V_ll = self.b_evaluator.aeP.dot(self.G_VB_ind_to_JS_ind)[np.tile(self.ll_mask, 2)]
            self.aeH_V_ll = self.b_evaluator.aeH.dot(self.G_VB_ind_to_JS_ind)[np.tile(self.ll_mask, 2)]
            self.aeP_T_ll = self.b_evaluator.aeP.dot(self.G_TB_imp_to_JS_imp)[np.tile(self.ll_mask, 2)]
            self.aeH_T_ll = self.b_evaluator.aeH.dot(self.G_TB_imp_to_JS_imp)[np.tile(self.ll_mask, 2)]
            self.aeP_V_cp_ll = self.cp_b_evaluator.aeP.dot(self.G_VB_ind_to_JS_ind_cp)[np.tile(self.ll_mask, 2)]
            self.aeH_V_cp_ll = self.cp_b_evaluator.aeH.dot(self.G_VB_ind_to_JS_ind_cp)[np.tile(self.ll_mask, 2)]
            self.aeP_T_cp_ll = self.cp_b_evaluator.aeP.dot(self.G_TB_imp_to_JS_imp_cp)[np.tile(self.ll_mask, 2)]
            self.aeH_T_cp_ll = self.cp_b_evaluator.aeH.dot(self.G_TB_imp_to_JS_imp_cp)[np.tile(self.ll_mask, 2)]

            if self.zero_jr_at_dip_equator:
                # Calculate matrix that converts TB_imp to Jr at dip equator
                n_phi = self.sh.Mmax*2 + 1
                dip_equator_phi = np.linspace(0, 360, n_phi)
                self.dip_equator_basis_evaluator = BasisEvaluator(self.basis, Grid(90 - self.mainfield.dip_equator(dip_equator_phi), dip_equator_phi))

                _equation_scaling = self.num_grid.lat[self.ll_mask].size / n_phi # scaling to match importance of other equations
                self.G_Jr_dip_equator = self.dip_equator_basis_evaluator.scaled_G(self.TB_imp_to_Jr) * _equation_scaling
            else:
                # Make zero-row stand-in for the Jr matrix
                self.G_Jr_dip_equator = np.empty((0, self.sh.num_coeffs))


    def impose_constraints(self):
        """ Impose constraints, if any. Leads to a contribution to TB_imp from
        VB_ind if the hemispheres are connected.

        """

        if self.connect_hemispheres:
            c = self.AV.dot(self.VB_ind.coeffs)
            if self.neutral_wind:
                c += self.cu

            constraint_vector = np.hstack((self.Jpar[~self.ll_mask], np.zeros(self.G_Jpar_ll_diff.shape[0]), np.zeros(self.G_Jr_dip_equator.shape[0]), c * self.ih_constraint_scaling ))

            self.set_coeffs(TB_imp = self.G_TB_constraints_inv.dot(constraint_vector))


    def set_FAC(self, FAC, _basis_evaluator):
        """
        Specify field-aligned current at ``self.num_grid.theta``,
        ``self.num_grid.lon``.

            Parameters
            ----------
            FAC: array
                The field-aligned current, in A/m^2, at
                ``self.num_grid.theta`` and ``self.num_grid.lon``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        if FAC.size != self.num_grid.theta.size:
            raise Exception('FAC must match phi and theta')
        
        self.Jpar = FAC

        # Extract the radial component of the FAC and set the corresponding basis coefficients
        self.set_coeffs(Jr = _basis_evaluator.grid_to_basis(self.Jpar * self.b_evaluator.br))
        self.impose_constraints()


    def set_u(self, u_theta, u_phi):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        if (u_theta.size != self.num_grid.theta.size) or (u_phi.size != self.num_grid.lon.size):
            raise Exception('Wind must match dimensions of num_grid')

        self.neutral_wind = True

        self.u_theta = u_theta
        self.u_phi   = u_phi

        self.uxB_theta =  self.u_phi   * self.b_evaluator.Br
        self.uxB_phi   = -self.u_theta * self.b_evaluator.Br

        if self.connect_hemispheres:
            u_theta_ll = u_theta[self.ll_mask]
            u_phi_ll   = u_phi[self.ll_mask]
            # Wind field at conjugate grid points
            u_cp_ll = csp.interpolate_vector_components(u_phi, -u_theta, np.ones_like(u_phi), self.num_grid.theta, self.num_grid.lon, self.cp_grid.theta[self.ll_mask], self.cp_grid.lon[self.ll_mask])
            u_theta_cp_ll, u_phi_cp_ll = -u_cp_ll[1], u_cp_ll[0]

            # Constraint vector contribution from wind
            self.cu =  (np.tile(u_theta_cp_ll, 2) * self.cp_b_evaluator.aut[np.tile(self.ll_mask, 2)] + np.tile(u_phi_cp_ll, 2) * self.cp_b_evaluator.aup[np.tile(self.ll_mask, 2)]) \
                      -(np.tile(u_theta_ll,    2) *    self.b_evaluator.aut[np.tile(self.ll_mask, 2)] + np.tile(u_phi_ll,    2) *    self.b_evaluator.aup[np.tile(self.ll_mask, 2)])


    def set_conductance(self, Hall, Pedersen, _basis_evaluator):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.lon``.

        """

        if Hall.size != Pedersen.size != self.num_grid.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.conductance = True

        self.etaP = Pedersen / (Hall**2 + Pedersen**2)
        self.etaH = Hall     / (Hall**2 + Pedersen**2)

        if self.connect_hemispheres:
            self.etaP_ll = self.etaP[self.ll_mask]
            self.etaH_ll = self.etaH[self.ll_mask]
            # Resistances at conjugate grid points
            self.etaP_cp_ll = csp.interpolate_scalar(self.etaP, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.cp_grid.theta[self.ll_mask], self.cp_grid.lon[self.ll_mask])
            self.etaH_cp_ll = csp.interpolate_scalar(self.etaH, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.cp_grid.theta[self.ll_mask], self.cp_grid.lon[self.ll_mask])

            # Conductance-dependent constraint matrices
            self.AV =  (np.tile(self.etaP_cp_ll, 2).reshape((-1, 1)) * self.aeP_V_cp_ll + np.tile(self.etaH_cp_ll, 2).reshape((-1, 1)) * self.aeH_V_cp_ll) \
                      -(np.tile(self.etaP_ll,    2).reshape((-1, 1)) * self.aeP_V_ll    + np.tile(self.etaH_ll,    2).reshape((-1, 1)) * self.aeH_V_ll)

            self.AT =  (np.tile(self.etaP_ll,    2).reshape((-1, 1)) * self.aeP_T_ll    + np.tile(self.etaH_ll,    2).reshape((-1, 1)) * self.aeH_T_ll) \
                      -(np.tile(self.etaP_cp_ll, 2).reshape((-1, 1)) * self.aeP_T_cp_ll + np.tile(self.etaH_cp_ll, 2).reshape((-1, 1)) * self.aeH_T_cp_ll)

            # Combine constraint matrices
            self.G_TB_constraints = np.vstack((self.G_Jpar_hl, self.G_Jpar_ll_diff, self.G_Jr_dip_equator, self.AT * self.ih_constraint_scaling))
            self.G_TB_constraints_inv = np.linalg.pinv(self.G_TB_constraints, rcond = 0)


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

        new_Br = self.VB_ind.coeffs * self.VB_ind_to_Br + self.EW.coeffs * self.EW_to_dBr_dt * dt

        self.set_coeffs(Br = new_Br)
        self.impose_constraints()


    def get_Br(self, _basis_evaluator):
        """ Calculate ``Br``.

        """

        return(_basis_evaluator.basis_to_grid(self.VB_ind.coeffs * self.VB_ind_to_Br))


    def get_JS(self): # for now, JS is always returned on num_grid!
        """ Calculate ionospheric sheet current.

        """
        Js_ind, Je_ind = np.split(self.G_VB_ind_to_JS_ind.dot(self.VB_ind.coeffs), 2, axis = 0)
        Js_imp, Je_imp = np.split(self.G_TB_imp_to_JS_imp.dot(self.TB_imp.coeffs), 2, axis = 0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp


        return(Jth, Jph)


    def get_Jr(self, _basis_evaluator):
        """ Calculate radial current.

        """

        return _basis_evaluator.basis_to_grid(self.TB_imp.coeffs * self.TB_imp_to_Jr)


    def get_Jeq(self, _basis_evaluator):
        """ Calculate equivalent current function.

        """

        return _basis_evaluator.basis_to_grid(self.VB_ind.coeffs * self.VB_ind_to_Jeq)


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

        Eth, Eph = np.zeros(self.num_grid.size), np.zeros(self.num_grid.size)

        if self.conductance:
            Jth, Jph = self.get_JS()

            Eth += self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.b_evaluator.br * Jph)
            Eph += self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.b_evaluator.br * Jth)

        if self.neutral_wind:
            Eth -= self.uxB_theta
            Eph -= self.uxB_phi

        return(Eth, Eph)
