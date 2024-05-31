import numpy as np
from pynamit.grid import Grid
from pynamit.constants import mu0, RE
from pynamit.basis_evaluator import BasisEvaluator
from pynamit.cubedsphere.cubedsphere import csp
from pynamit.vector import Vector
from pynamit.b_field.b_geometry import BGeometry


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

        # Conversion factors
        self.VB_to_Br     = -self.sh.n
        self.laplacian    = -self.sh.n * (self.sh.n + 1) / RI**2
        self.TB_to_Jr     = self.laplacian * self.RI / mu0
        self.EW_to_dBr_dt = -self.laplacian * self.RI
        self.VB_to_Jeq    = self.RI / mu0 * (2 * self.sh.n + 1) / (self.sh.n + 1)

        # initialize the basis evaluator
        self.basis_evaluator = BasisEvaluator(self.basis, num_grid)
        self.b_geometry = BGeometry(mainfield, num_grid, RI)

        # initialize neutral wind
        self.u_theta = None
        self.u_phi = None 

        # construct the elements in the matrix in the electric field equation
        self.b00 = self.b_geometry.bphi**2 + self.b_geometry.br**2
        self.b01 = -self.b_geometry.btheta * self.b_geometry.bphi
        self.b10 = -self.b_geometry.btheta * self.b_geometry.bphi
        self.b11 = self.b_geometry.btheta**2 + self.b_geometry.br**2

        # Pre-calculate the matrix that maps from TB to the boundary magnetic field (Bh+)
        if self.mainfield.kind == 'radial' or self.ignore_PFAC: # no Poloidal field so get matrix of zeros
            self.TB_to_VB_PFAC = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs))
        else: # Use the method by Engels and Olsen 1998, Eq. 13 to account for poloidal part of magnetic field for FACs
            if PFAC_matrix is None:
                self.TB_to_VB_PFAC = self._get_PFAC_matrix()
            else:
                self.TB_to_VB_PFAC = PFAC_matrix

        self.G_TB_to_JS = self.get_G_TB_to_JS(self.basis_evaluator) # matrices that map TB to r x deltaB
        self.G_VB_to_JS = self.get_G_VB_to_JS(self.basis_evaluator) # matrices that map VB to r x deltaB

        if connect_hemispheres:
            if ignore_PFAC:
                raise ValueError('Hemispheres can not be connected when ignore_PFAC is True')
            if self.mainfield.kind == 'radial':
                raise ValueError('Hemispheres can not be connected with radial magnetic field')

            # identify the low latitude points
            if self.mainfield.kind == 'dipole':
                ll_mask = np.abs(self.num_grid.lat) < self.latitude_boundary
            elif self.mainfield.kind == 'igrf':
                mlat, mlon = self.mainfield.apx.geo2apex(self.num_grid.lat, self.num_grid.lon, (self.RI - RE)*1e-3)
                ll_mask = np.abs(mlat) < self.latitude_boundary
            else:
                print('this should not happen')

            # mask the jr so that it only applies poleward of self.latitude_boundary
            self.Gjr_hl = self.basis_evaluator.scaled_G(self.TB_to_Jr) * (1 - ll_mask).reshape((-1, 1))

            # calculate constraint matrices for low latitude points
            self.G_TB_to_JS_ll = self.G_TB_to_JS * np.tile(ll_mask, 2).reshape((-1, 1))
            self.G_VB_to_JS_ll = self.G_VB_to_JS * np.tile(ll_mask, 2).reshape((-1, 1))
            self.aeP_V_ll, self.aeH_V_ll = self.b_geometry.aeP.dot(self.G_VB_to_JS_ll), self.b_geometry.aeH.dot(self.G_VB_to_JS_ll)
            self.aeP_T_ll, self.aeH_T_ll = self.b_geometry.aeP.dot(self.G_TB_to_JS_ll), self.b_geometry.aeH.dot(self.G_TB_to_JS_ll)

            # ... and for their conjugate points:
            self.cp_theta, self.cp_phi = self.mainfield.conjugate_coordinates(self.RI, self.num_grid.theta, self.num_grid.lon)
            self.cp_grid = Grid(90 - self.cp_theta, self.cp_phi)
            cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.G_TB_to_JS_cp = self.get_G_TB_to_JS(cp_basis_evaluator) * np.tile(ll_mask, 2).reshape((-1, 1))
            self.G_VB_to_JS_cp = self.get_G_VB_to_JS(cp_basis_evaluator) * np.tile(ll_mask, 2).reshape((-1, 1))
            self.cp_b_geometry = BGeometry(self.mainfield, self.cp_grid, RI)
            self.aeP_V_cp, self.aeH_V_cp = self.cp_b_geometry.aeP.dot(self.G_VB_to_JS_cp), self.cp_b_geometry.aeH.dot(self.G_VB_to_JS_cp)
            self.aeP_T_cp, self.aeH_T_cp = self.cp_b_geometry.aeP.dot(self.G_TB_to_JS_cp), self.cp_b_geometry.aeH.dot(self.G_TB_to_JS_cp)

            # constraint matrix: FAC out of one hemisphere = FAC into the other
            self.G_par_ll = self.basis_evaluator.scaled_G(self.TB_to_Jr / self.b_geometry.br.reshape((-1 ,1)))    * ll_mask.reshape((-1, 1))
            self.G_par_cp = cp_basis_evaluator.scaled_G(  self.TB_to_Jr / self.cp_b_geometry.br.reshape((-1 ,1))) * ll_mask.reshape((-1, 1))
            self.constraint_Gpar = (self.G_par_ll - self.G_par_cp) 

            if self.zero_jr_at_dip_equator: # calculate matrix to compute jr at dip equator
                dip_equator_phi = np.linspace(0, 360, self.sh.Mmax*2 + 1)
                dip_equator_theta = self.mainfield.dip_equator(dip_equator_phi)
                self.dip_equator_grid = Grid(90 - dip_equator_theta, dip_equator_phi)
                self.dip_equator_basis_evaluator = BasisEvaluator(self.basis, self.dip_equator_grid)

                _equation_scaling = self.num_grid.lat[ll_mask].size / (self.sh.Mmax*2 + 1) # scaling to match importance of other equations
                self.G_jr_dip_equator = self.dip_equator_basis_evaluator.scaled_G(self.TB_to_Jr) * _equation_scaling
            else: # make zero-row stand-in for the jr matrix:
                self.G_jr_dip_equator = np.empty((0, self.sh.n.size))

            self._zeros = np.zeros(self.constraint_Gpar.shape[0] + self.G_jr_dip_equator.shape[0])

        # Initialize neutral wind and conductances
        self.set_u(np.zeros(self.num_grid.size), np.zeros(self.num_grid.size))
        self.set_conductance(np.zeros(self.num_grid.size), np.zeros(self.num_grid.size), self.basis_evaluator)

        # Initialize the spherical harmonic coefficients
        self.set_coeffs(VB = np.zeros(self.basis.num_coeffs))
        self.set_coeffs(TB = np.zeros(self.basis.num_coeffs))

    def _get_PFAC_matrix(self):
        """ """
        # initialize matrix that will map from self.TB to coefficients for poloidal field:

        r_k_steps = self.FAC_integration_steps
        Delta_k = np.diff(r_k_steps)
        r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

        JS_shifted_to_VB_shifted = np.linalg.pinv(self.get_G_VB_to_JS(self.basis_evaluator))

        TB_to_VB_PFAC = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs))
        for i in range(r_k.size): 
            print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
            # Map coordinates from r_k[i] to RI:
            theta_mapped, phi_mapped = self.mainfield.map_coords(self.RI, r_k[i], self.num_grid.theta, self.num_grid.lon)
            mapped_grid = Grid(90 - theta_mapped, phi_mapped)

            # Matrix that gives FAC at mapped grid from toroidal coefficients, shifts to r_k[i], and extracts horizontal components
            shifted_b_geometry = BGeometry(self.mainfield, self.num_grid, r_k[i])
            mapped_b_geometry = BGeometry(self.mainfield, mapped_grid, self.RI)
            mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)
            Jr_to_JS_shifted = ((shifted_b_geometry.Btheta / mapped_b_geometry.Br).reshape((-1, 1)),
                                (shifted_b_geometry.Bphi   / mapped_b_geometry.Br).reshape((-1, 1)))
            TB_to_JS_shifted = np.vstack(mapped_basis_evaluator.scaled_G(self.TB_to_Jr) * Jr_to_JS_shifted)

            # Matrix that calculates the contribution to the poloidal coefficients from the horizontal components at r_k[i]
            VB_shifted_to_VB = (self.RI / r_k[i])**(self.sh.n - 1).reshape((-1, 1))
            JS_shifted_to_VB = JS_shifted_to_VB_shifted * VB_shifted_to_VB

            # Integration step
            TB_to_VB_PFAC -= Delta_k[i] * JS_shifted_to_VB.dot(TB_to_JS_shifted)

        return(TB_to_VB_PFAC)


    def update_constraints(self):
        """ update the constraint arrays c and A - should be called when changing u and eta """

        if self.connect_hemispheres:
            self.cu =  (np.tile(self.u_theta_cp, 2) * self.cp_b_geometry.aut + np.tile(self.u_phi_cp, 2) * self.cp_b_geometry.aup) \
                      -(np.tile(self.u_theta,    2) * self.b_geometry.aut    + np.tile(self.u_phi,    2) * self.b_geometry.aup)

            self.AV =  (np.tile(self.etaP_cp, 2).reshape((-1, 1)) * self.aeP_V_cp + np.tile(self.etaH_cp, 2).reshape((-1, 1)) * self.aeH_V_cp) \
                      -(np.tile(self.etaP,    2).reshape((-1, 1)) * self.aeP_V_ll + np.tile(self.etaH,    2).reshape((-1, 1)) * self.aeH_V_ll)

            self.AT =  (np.tile(self.etaP,    2).reshape((-1, 1)) * self.aeP_T_ll + np.tile(self.etaH,    2).reshape((-1, 1)) * self.aeH_T_ll) \
                      -(np.tile(self.etaP_cp, 2).reshape((-1, 1)) * self.aeP_T_cp + np.tile(self.etaH_cp, 2).reshape((-1, 1)) * self.aeH_T_cp)

            # Combine constraint matrices:
            self.G_TB_constraints = np.vstack((self.Gjr_hl, self.constraint_Gpar, self.G_jr_dip_equator, self.AT * self.ih_constraint_scaling))
            self.G_TB_constraints_inv = np.linalg.pinv(self.G_TB_constraints, rcond = 0)



    def get_G_TB_to_JS(self, _basis_evaluator):
        """ Calculate matrix that maps the coefficients TB to delta B across ionosphere """

        GrxgradT_theta = -_basis_evaluator.G_th
        GrxgradT_phi   = -_basis_evaluator.G_ph
        G_TB0_to_JS = np.vstack((GrxgradT_theta, GrxgradT_phi)) / mu0

        G_VB_to_JS = self.get_G_VB_to_JS(_basis_evaluator)

        G_TB_to_JS = G_TB0_to_JS + G_VB_to_JS.dot(self.TB_to_VB_PFAC)

        return(G_TB_to_JS)


    def get_G_VB_to_JS(self, _basis_evaluator):
        """ Calculate matrix that maps the coefficients VB to delta B across ionosphere """

        GVrxdB_theta = -_basis_evaluator.G_ph * (2 * self.sh.n + 1) / (self.sh.n + 1)
        GVrxdB_phi   =  _basis_evaluator.G_th * (2 * self.sh.n + 1) / (self.sh.n + 1)


        G_VB_to_JS = np.vstack((GVrxdB_theta, GVrxdB_phi)) / mu0

        return(G_VB_to_JS)


    def set_coeffs(self, **kwargs):
        """ Set coefficients.

        Specify a set of coefficients and update the rest so that they are
        consistent.

        This function accepts one (and only one) set of coefficients.
        Valid values for kwargs (only one):

        - 'VB' : Coefficients for magnetic field scalar ``V``.
        - 'TB' : Coefficients for surface current scalar ``T``.
        - 'Br' : Coefficients for magnetic field ``Br`` (at ``r = RI``).
        - 'Jr': Coefficients for radial current scalar.

        """

        valid_kws = ['VB', 'TB', 'Br', 'Jr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'VB':
            self.VB = Vector(self.basis, kwargs['VB'])
        elif key == 'TB':
            self.TB = Vector(self.basis, kwargs['TB'])
        elif key == 'Br':
            self.VB = Vector(self.basis, kwargs['Br'] / self.VB_to_Br)
        elif key == 'Jr':
            self.TB = Vector(self.basis, kwargs['Jr'] / self.TB_to_Jr)
        else:
            raise Exception('This should not happen')


    def set_initial_condition(self):
        """ Set initial conditions.

        If this is not called, initial conditions should be zero.

        """

        print('not implemented. inital conditions will be zero')


    def impose_constraints(self):
        """ Impose constraints, if any. May lead to a contribution to TB
        from VB.

        """

        if self.connect_hemispheres:
            c = self.cu + self.AV.dot(self.VB.coeffs)
            d = np.hstack((self.jr, self._zeros, c * self.ih_constraint_scaling ))
            self.set_coeffs(TB = self.G_TB_constraints_inv.dot(d))


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

        # Extract the radial component of the FAC:
        self.jr = FAC * self.b_geometry.br

        # Get the corresponding basis coefficients and propagate to the other coefficients (TB, VB):
        self.set_coeffs(Jr = _basis_evaluator.grid_to_basis(self.jr))

        self.update_constraints()
        self.impose_constraints()


    def set_u(self, u_theta, u_phi):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        if (u_theta.size != self.num_grid.theta.size) or (u_phi.size != self.num_grid.lon.size):
            raise Exception('Wind must match dimensions of num_grid')

        self.u_theta = u_theta
        self.u_phi   = u_phi

        self.uxB_theta =  self.u_phi   * self.b_geometry.Br
        self.uxB_phi   = -self.u_theta * self.b_geometry.Br

        if self.connect_hemispheres:
            # find wind field at conjugate grid points
            u_cp = csp.interpolate_vector_components(u_phi, -u_theta, np.ones_like(u_phi), self.num_grid.theta, self.num_grid.lon, self.cp_grid.theta, self.cp_grid.lon)
            self.u_theta_cp, self.u_phi_cp = -u_cp[1], u_cp[0]


    def set_conductance(self, Hall, Pedersen, _basis_evaluator):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.lon``.

        """

        if Hall.size != Pedersen.size != self.num_grid.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.etaP = Pedersen / (Hall**2 + Pedersen**2)
        self.etaH = Hall     / (Hall**2 + Pedersen**2)

        if self.connect_hemispheres:
            # find resistances at conjugate grid points
            self.etaP_cp = csp.interpolate_scalar(self.etaP, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.cp_grid.theta, self.cp_grid.lon)
            self.etaH_cp = csp.interpolate_scalar(self.etaH, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.cp_grid.theta, self.cp_grid.lon)



    def update_Phi_and_EW(self):
        """ Update the coefficients for the electric potential and the induction electric field.

        """

        E_cf, E_df = self.basis_evaluator.grid_to_basis(self.get_E(self.basis_evaluator), helmholtz = True)

        self.Phi = Vector(self.basis, coeffs = E_cf)
        self.EW = Vector(self.basis, coeffs = E_df)


    def evolve_Br(self, dt):
        """ Evolve ``Br`` in time.

        """

        #Eth, Eph = self.get_E(self.num_grid)
        #u1, u2, u3 = self.equations.sph_to_contravariant_cs(np.zeros_like(Eph), Eth, Eph)
        #curlEr = self.equations.curlr(u1, u2) 
        #Br = -self.GBr.dot(self.VB) - dt * curlEr

        #self.set_coeffs(Br = self.basis_evaluator.grid_to_basis(-Br))

        #GTE = self.Gcf.T.dot(np.hstack( self.get_E(self.num_grid)) )
        #self.EW = self.GTGcf_inv.dot(GTE) # find coefficients for divergence-free / inductive E

        self.impose_constraints()

        self.update_Phi_and_EW()

        new_Br = self.VB.coeffs * self.VB_to_Br + self.EW.coeffs * self.EW_to_dBr_dt * dt
        self.set_coeffs(Br = new_Br)


    def get_Br(self, _basis_evaluator, deg = False):
        """ Calculate ``Br``.

        """

        return(_basis_evaluator.basis_to_grid(self.VB.coeffs * self.VB_to_Br))


    def get_JS(self, _basis_evaluator, deg = False):
        """ Calculate ionospheric sheet current.

        """
        Js_V, Je_V = np.split(self.G_VB_to_JS.dot(self.VB.coeffs), 2, axis = 0)
        Js_T, Je_T = np.split(self.G_TB_to_JS.dot(self.TB.coeffs), 2, axis = 0)

        Jth, Jph = Js_V + Js_T, Je_V + Je_T


        return(Jth, Jph)


    def get_Jr(self, _basis_evaluator, deg = False):
        """ Calculate radial current.

        """

        return _basis_evaluator.basis_to_grid(self.TB.coeffs * self.TB_to_Jr)


    def get_Jeq(self, _basis_evaluator, deg = False):
        """ Calculate equivalent current function.

        """

        return _basis_evaluator.basis_to_grid(self.VB.coeffs * self.VB_to_Jeq)


    def get_Phi(self, _basis_evaluator, deg = False):
        """ Calculate Phi.

        """

        return _basis_evaluator.basis_to_grid(self.Phi.coeffs)


    def get_W(self, _basis_evaluator, deg = False):
        """ Calculate the induction electric field scalar.

        """

        return _basis_evaluator.basis_to_grid(self.EW.coeffs)


    def get_E(self, _basis_evaluator, deg = False):
        """ Calculate electric field.

        """
        if self.u_theta is not None:
            Eth = -self.uxB_theta
            Eph = -self.uxB_phi
        else:
            Eth = 0
            Eph = 0

        Jth, Jph = self.get_JS(_basis_evaluator, deg = deg)

        Eth = Eth + self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.b_geometry.br * Jph)
        Eph = Eph + self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.b_geometry.br * Jth)

        return(Eth, Eph)

