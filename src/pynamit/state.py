import numpy as np
from pynamit.grid import Grid
from pynamit.constants import mu0, RE
from pynamit.basis_evaluator import BasisEvaluator
from pynamit.cubedsphere.cubedsphere import csp
from pynamit.vector import Vector

DEBUG_constraint_scale = mu0#1/ 3e8 # 1/(mu0 * c)
DEBUG_jpar_scale = 1#1e15
DEBUG_jr_dip_scale = 1

class State(object):
    """ State of the ionosphere.

    """

    def __init__(self, sh, mainfield, num_grid, RI, ignore_PFAC, FAC_integration_parameters, connect_hemispheres, latitude_boundary, zero_jr_at_dip_equator):
        """ Initialize the state of the ionosphere.
    
        """

        self.sh = sh
        self.basis = sh # SHBasis for now, but can in principle be any basis if we generalize
        self.mainfield = mainfield
        self.num_grid = num_grid
        self.FAC_integration_parameters = FAC_integration_parameters

        self.RI = RI
        self.ignore_PFAC = ignore_PFAC
        self.zero_jr_at_dip_equator = zero_jr_at_dip_equator

        self.connect_hemispheres = connect_hemispheres
        self.latitude_boundary = latitude_boundary

        # Conversion factors
        self.VB_to_Br = self.sh.n # Equation for Br in paper has negative sign...?
        self.TB_to_Jr = 1 / self.RI / mu0 * self.sh.n * (self.sh.n + 1) # Equation for Jr in paper has RI in the numerator instead...?

        self.VB_to_VJ = self.RI / mu0 * (2 * self.sh.n + 1) / (self.sh.n + 1)
        self.TB_to_TJ = -self.RI / mu0

        # initialize the basis evaluator
        self.basis_evaluator = BasisEvaluator(self.basis, num_grid)

        # initialize neutral wind
        self.u_theta = None
        self.u_phi = None 

        # get magnetic field unit vectors at CS grid:
        self.B = np.vstack(self.mainfield.get_B(self.RI, self.num_grid.theta, self.num_grid.lon))
        self.br, self.btheta, self.bphi = self.B / np.linalg.norm(self.B, axis = 0)
        self.sinI = -self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2) # sin(inclination)
        # construct the elements in the matrix in the electric field equation
        self.b00 = self.bphi**2 + self.br**2
        self.b01 = -self.btheta * self.bphi
        self.b10 = -self.btheta * self.bphi
        self.b11 = self.btheta**2 + self.br**2

        # Pre-calculate the matrix that maps from TB to the boundary magnetic field (Bh+)
        if self.mainfield.kind == 'radial' or self.ignore_PFAC: # no Poloidal field so get matrix of zeros
            self.TB_to_PFAC = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs))
        else: # Use the method by Engels and Olsen 1998, Eq. 13 to account for poloidal part of magnetic field for FACs
            self.TB_to_PFAC = self._get_PFAC_matrix(num_grid, self.basis_evaluator)

        self.GTBrxdB = self.get_GTrxdB(self.basis_evaluator) # matrices that map sch_TB to r x deltaB
        self.GVBrxdB = self.get_GVrxdB(self.basis_evaluator) # matrices that map sch_VB to r x deltaB

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

            # calculate constraint matrices for low latitude points
            self.ll_grid = Grid(RI, 90 - self.num_grid.theta[ll_mask], self.num_grid.lon[ll_mask])
            ll_basis_evaluator = BasisEvaluator(self.basis, self.ll_grid)
            self.aeP_ll, self.aeH_ll, self.aut_ll, self.aup_ll = self._get_alpha(self.ll_grid)
            self.GTBrxdB_ll = self.get_GTrxdB(ll_basis_evaluator) / mu0
            self.GVBrxdB_ll = self.get_GVrxdB(ll_basis_evaluator) / mu0
            self.aeP_V_ll, self.aeH_V_ll = self.aeP_ll.dot(self.GVBrxdB_ll), self.aeH_ll.dot(self.GVBrxdB_ll)
            self.aeP_T_ll, self.aeH_T_ll = self.aeP_ll.dot(self.GTBrxdB_ll), self.aeH_ll.dot(self.GTBrxdB_ll)


            # ... and for their conjugate points:
            self.cp_theta, self.cp_phi = self.mainfield.conjugate_coordinates(self.RI, self.ll_grid.theta, self.ll_grid.lon)
            self.cp_grid = Grid(RI, 90 - self.cp_theta, self.cp_phi)
            cp_basis_evaluator = BasisEvaluator(self.basis, self.cp_grid)
            self.aeP_cp, self.aeH_cp, self.aut_cp, self.aup_cp = self._get_alpha(self.cp_grid)
            self.GTBrxdB_cp = self.get_GTrxdB(cp_basis_evaluator) / mu0
            self.GVBrxdB_cp = self.get_GVrxdB(cp_basis_evaluator) / mu0
            self.aeP_V_cp, self.aeH_V_cp = self.aeP_cp.dot(self.GVBrxdB_cp), self.aeH_cp.dot(self.GVBrxdB_cp)
            self.aeP_T_cp, self.aeH_T_cp = self.aeP_cp.dot(self.GTBrxdB_cp), self.aeH_cp.dot(self.GTBrxdB_cp)

            # calculate sin(inclination)
            self.ll_sinI = self.mainfield.get_sinI(self.RI, self.ll_grid.theta, self.ll_grid.lon).reshape((-1 ,1))
            self.cp_sinI = self.mainfield.get_sinI(self.RI, self.cp_grid.theta, self.cp_grid.lon).reshape((-1 ,1))

            # constraint matrix: FAC out of one hemisphere = FAC into the other
            self.G_par_ll     = ll_basis_evaluator.scaled_G(1 /self.RI / mu0 * self.sh.n * (self.sh.n + 1) / self.ll_sinI)
            self.G_par_cp     = cp_basis_evaluator.scaled_G(1 /self.RI / mu0 * self.sh.n * (self.sh.n + 1) / self.cp_sinI)
            self.constraint_Gpar = (self.G_par_ll - self.G_par_cp) * DEBUG_jpar_scale

            if self.zero_jr_at_dip_equator: # calculate matrix to compute jr at dip equator
                dip_equator_phi = np.linspace(0, 360, self.sh.Mmax*2 + 1)
                dip_equator_theta = self.mainfield.dip_equator(dip_equator_phi)
                self.dip_equator_grid = Grid(RI, 90 - dip_equator_theta, dip_equator_phi)
                self.dip_equator_basis_evaluator = BasisEvaluator(self.basis, self.dip_equator_grid)
                self.G_jr_dip_equator = self.dip_equator_basis_evaluator.scaled_G(1 /self.RI / mu0 * self.sh.n * (self.sh.n + 1))
            else: # make zero-row stand-in for the jr matrix:
                self.G_jr_dip_equator = np.empty((0, self.sh.n.size))
 

        # Initialize neutral wind and conductances
        self.set_u(np.zeros(self.num_grid.size), np.zeros(self.num_grid.size), update = False)
        self.set_conductance(np.zeros(self.num_grid.size), np.zeros(self.num_grid.size), self.basis_evaluator, update = True)

        # Initialize the spherical harmonic coefficients
        self.set_coeffs(VB = np.zeros(self.basis.num_coeffs))
        self.set_coeffs(TB = np.zeros(self.basis.num_coeffs))

    def _get_PFAC_matrix(self, _grid, _basis_evaluator):
        """ """
        # initialize matrix that will map from self.TB to coefficients for poloidal field:

        r_k_steps = self.FAC_integration_parameters['steps']
        Delta_k = np.diff(r_k_steps)
        r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

        jh_grid_to_basis = -_basis_evaluator.Gdf_inv * self.RI * mu0 # matrix to do SHA in Eq (7) in Engels and Olsen (inc. scaling)

        TB_to_PFAC = np.zeros((self.basis.num_coeffs, self.basis.num_coeffs))
        for i in range(r_k.size): 
            print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
            # map coordinates from r_k[i] to RI:
            theta_mapped, phi_mapped = self.mainfield.map_coords(self.RI, r_k[i], _grid.theta, _grid.lon)
            mapped_grid = Grid(self.RI, 90 - theta_mapped, phi_mapped)
            mapped_basis_evaluator = BasisEvaluator(self.basis, mapped_grid)

            # Calculate magnetic field at grid points at r_k[i]:
            B_rk  = np.vstack(self.mainfield.get_B(r_k[i], _grid.theta, _grid.lon))
            B0_rk = np.linalg.norm(B_rk, axis = 0) # magnetic field magnitude
            b_rk = B_rk / B0_rk # unit vectors

            # Calculate magnetic field at the points in the ionosphere to which the grid maps:
            B_RI  = np.vstack(self.mainfield.get_B(self.RI, mapped_grid.theta, mapped_grid.lon))
            B0_RI = np.linalg.norm(B_RI, axis = 0) # magnetic field magnitude
            sinI_RI = -B_RI[0] / B0_RI

            # Calculate matrix that gives FAC from toroidal coefficients
            G_k = mapped_basis_evaluator.scaled_G(-self.sh.n * (self.sh.n + 1) / self.RI / mu0 / sinI_RI.reshape((-1, 1))) # TODO: Handle singularity at equator (may be fine)

            # matrix that scales the FAC at RI to r_k and extracts the horizontal components:
            ratio = (B0_rk / B0_RI).reshape((1, -1))
            S_k = np.vstack((np.diag(b_rk[1]), np.diag(b_rk[2]))) * ratio

            # matrix that scales the terms by (R/r_k)**(n-1):
            A_k = np.diag((self.RI / r_k[i])**(self.sh.n - 1))

            # put it all together (crazy)
            TB_to_PFAC += Delta_k[i] * A_k.dot(jh_grid_to_basis.dot(S_k.dot(G_k)))

        # return the matrix scaled by the term in front of the integral
        return(np.diag((self.sh.n + 1) / (2 * self.sh.n + 1)).dot(TB_to_PFAC) / self.RI)


    def _get_alpha(self, _grid):
        """ Calculate alpha
            

        """

        R, theta, phi = self.RI, _grid.theta, _grid.lon
        B = np.vstack(self.mainfield.get_B(R, theta, phi))
        B0 = np.linalg.norm(B, axis = 0)
        br, bt, bp = B / B0
        Br = B[0]
        _, _, _, e1, e2, _ = self.mainfield.basevectors(R, theta, phi)
        e1r, e1t, e1p = e1
        e2r, e2t, e2p = e2

        # The following lines of code are copied from output from sympy_matrix_algebra.py

        # resistance terms:
        alpha11_eP = bp**2*e1t - bp*bt*e1p + br**2*e1t - br*bt*e1r
        alpha12_eP = -bp*br*e1r - bp*bt*e1t + br**2*e1p + bt**2*e1p
        alpha21_eP = bp**2*e2t - bp*bt*e2p + br**2*e2t - br*bt*e2r
        alpha22_eP = -bp*br*e2r - bp*bt*e2t + br**2*e2p + bt**2*e2p
        alpha11_eH = bp*e1r - br*e1p
        alpha12_eH = br*e1t - bt*e1r
        alpha21_eH = bp*e2r - br*e2p
        alpha22_eH = br*e2t - bt*e2r

        # wind terms:
        alpha13_ut = -Br*bp*e1r/br + Br*e1p
        alpha23_ut = -Br*bp*e2r/br + Br*e2p
        alpha13_up = -Br*e1t + Br*bt*e1r/br
        alpha23_up = -Br*e2t + Br*bt*e2r/br

        # construct matrices
        aeP = np.vstack((np.hstack((np.diag(alpha11_eP), np.diag(alpha12_eP))), 
                         np.hstack((np.diag(alpha21_eP), np.diag(alpha22_eP)))))
        aeH = np.vstack((np.hstack((np.diag(alpha11_eH), np.diag(alpha12_eH))), 
                         np.hstack((np.diag(alpha21_eH), np.diag(alpha22_eH)))))
        aut = np.hstack((alpha13_ut, alpha23_ut))
        aup = np.hstack((alpha13_up, alpha23_up))

        return(aeP, aeH, aut, aup)


    def update_constraints(self):
        """ update the constraint arrays c and A - should be called when changing u and eta """

        self.cu =  (self.u_theta_cp * self.aut_cp + self.u_phi_cp * self.aup_cp) \
                  -(self.u_theta_ll * self.aut_ll + self.u_phi_ll * self.aup_ll)
        self.AV =  (self.etaP_cp * self.aeP_V_cp + self.etaH_cp * self.aeH_V_cp) \
                  -(self.etaP_ll * self.aeP_V_ll + self.etaH_ll * self.aeH_V_ll)
        self.AT =  (self.etaP_ll * self.aeP_T_ll + self.etaH_ll * self.aeH_T_ll)\
                  -(self.etaP_cp * self.aeP_T_cp + self.etaH_cp * self.aeH_T_cp)



    def get_GTrxdB(self, _basis_evaluator):
        """ Calculate matrix that maps the coefficients TB to delta B across ionosphere """
        print('should write a test for these functions')
        GrxgradT = -_basis_evaluator.Gdf * self.RI # matrix that gets -r x grad(T)
        GPFAC    = -_basis_evaluator.Gcf                      # matrix that calculates potential magnetic field of external source
        Gshield  = -(_basis_evaluator.Gcf / (self.sh.n + 1)) # matrix that calculates potential magnetic field of shielding current

        GTB = GrxgradT + (GPFAC + Gshield).dot(self.TB_to_PFAC)
        GTBth, GTBph = np.split(GTB, 2, axis = 0)
        GTrxdB = np.vstack((-GTBph, GTBth))
        return(GTrxdB)


    def get_GVrxdB(self, _basis_evaluator):
        """ Calculate matrix that maps the coefficients VB to delta B across ionosphere """
        GVdB = _basis_evaluator.Gcf * (self.sh.n / (self.sh.n + 1) + 1) * self.RI
        GVBth, GVBph = np.split(GVdB, 2, axis = 0)
        GVrxdB = np.vstack((-GVBph, GVBth))

        return(GVrxdB)
    

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
        self.jr = -FAC * self.sinI
        # Get the corresponding basis coefficients and propagate to the other coefficients (TB, VB):
        self.set_coeffs(Jr = _basis_evaluator.grid_to_basis(self.jr))

        if self.connect_hemispheres:

            # mask the jr so that it only applies poleward of self.latitude_boundary
            hl_mask = np.abs(_basis_evaluator.grid.lat) > self.latitude_boundary
            self.hl_grid = Grid(self.RI, 90 - _basis_evaluator.grid.theta[hl_mask], _basis_evaluator.grid.lon[hl_mask])
            hl_basis_evaluator = BasisEvaluator(self.sh, self.hl_grid)

            #B = np.vstack(self.mainfield.get_B(self.RI, self.hl_grid.theta, self.hl_grid.lon))
            #br, btheta, bphi = B / np.linalg.norm(B, axis = 0)
            #hl_sinI = -br / np.sqrt(btheta**2 + bphi**2 + br**2) # sin(inclination)

            self.Gjr = hl_basis_evaluator.scaled_G(1  / self.RI / mu0 * self.sh.n * (self.sh.n + 1))
            self.jr = self.jr[hl_mask].flatten()

            # combine matrices:
            self.G_TB = np.vstack((self.Gjr, self.constraint_Gpar, self.G_jr_dip_equator * DEBUG_jr_dip_scale, self.AT * DEBUG_constraint_scale ))
            #print('inverting')

            self._zeros = np.zeros(self.constraint_Gpar.shape[0] + self.G_jr_dip_equator.shape[0])

            self.Gpinv = np.linalg.pinv(self.G_TB, rcond = 0)
            print('rcond!')

            c = self.cu + self.AV.dot(self.VB.coeffs)
            
            self.ccc = c
            d = np.hstack((self.jr, self._zeros, c * DEBUG_constraint_scale ))
            self.set_coeffs(TB = self.Gpinv.dot(d))
            self.ggg = self.G_TB


            #print('connect_hemispheres is not fully implemented')

        print('Note: Check if rcond is really needed. It should not be necessary if the FAC is given sufficiently densely')
        print('Note to self: Remember to write a function that compares the AMPS SH coefficient to the ones derived here')


    def set_u(self, u_theta, u_phi, update = True):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """
        self.u_theta = u_theta
        self.u_phi   = u_phi

        self.uxB_theta =  self.u_phi   * self.B[0] 
        self.uxB_phi   = -self.u_theta * self.B[0] 

        if self.connect_hemispheres:
            # find wind field at low lat grid points
            u_ll = csp.interpolate_vector_components(u_phi, -u_theta, np.ones_like(u_phi), self.num_grid.theta, self.num_grid.lon, self.ll_grid.theta, self.ll_grid.lon)
            u_ll = np.tile(u_ll, (1, 2)) # duplicate 
            self.u_theta_ll, self.u_phi_ll = -u_ll[1], u_ll[0]

            # find wind field at conjugate grid points
            u_cp = csp.interpolate_vector_components(u_phi, -u_theta, np.ones_like(u_phi), self.num_grid.theta, self.num_grid.lon, self.cp_grid.theta, self.cp_grid.lon)
            u_cp = np.tile(u_cp, (1, 2)) # duplicate 
            self.u_theta_cp, self.u_phi_cp = -u_cp[1], u_cp[0]

            if update:
                self.update_constraints()


    def set_conductance(self, Hall, Pedersen, _basis_evaluator, update = True):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.lon``.

        """

        if Hall.size != Pedersen.size != self.num_grid.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.etaP = Pedersen / (Hall**2 + Pedersen**2)
        self.etaH = Hall     / (Hall**2 + Pedersen**2)

        if self.connect_hemispheres:
            # TODO: 1) csp structure is strange. 2) This is inefficient when eta is updated often
            # find resistances at low lat grid points
            self.etaP_ll      = np.tile(csp.interpolate_scalar(self.etaP, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.ll_grid.theta, self.ll_grid.lon), 2).reshape((-1, 1))
            self.etaH_ll      = np.tile(csp.interpolate_scalar(self.etaH, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.ll_grid.theta, self.ll_grid.lon), 2).reshape((-1, 1))

            # find resistances at conjugate grid points
            self.etaP_cp = np.tile(csp.interpolate_scalar(self.etaP, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.cp_grid.theta, self.cp_grid.lon), 2).reshape((-1, 1))
            self.etaH_cp = np.tile(csp.interpolate_scalar(self.etaH, _basis_evaluator.grid.theta, _basis_evaluator.grid.lon, self.cp_grid.theta, self.cp_grid.lon), 2).reshape((-1, 1))

            if update:
                self.update_constraints()            


    def update_EW(self):
        """ Update the coefficients for the induction electric field.

        """

        self.EW = Vector(self.basis, basis_evaluator = self.basis_evaluator, grid_values = np.hstack(self.get_E(self.basis_evaluator)), component='df')


    def update_Phi(self):
        """ Update the coefficients for the electric potential.

        """

        self.Phi = Vector(self.basis, basis_evaluator = self.basis_evaluator, grid_values = np.hstack(self.get_E(self.basis_evaluator)), component='cf')


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

        if self.connect_hemispheres:
            c = self.cu + self.AV.dot(self.VB.coeffs)
            d = np.hstack((self.jr, self._zeros, c * DEBUG_constraint_scale ))
            self.set_coeffs(TB = self.Gpinv.dot(d))

        self.update_EW()
        new_Br = self.VB.coeffs * self.VB_to_Br + self.sh.n * (self.sh.n + 1) * self.EW.coeffs * dt / self.RI**2
        self.set_coeffs(Br = new_Br)


    def get_Br(self, _basis_evaluator, deg = False):
        """ Calculate ``Br``.

        """

        return(_basis_evaluator.basis_to_grid(self.VB.coeffs * self.VB_to_Br))


    def get_JS(self, _basis_evaluator, deg = False):
        """ Calculate ionospheric sheet current.

        """
        Js_V, Je_V = np.split(self.GVBrxdB.dot(self.VB.coeffs) / mu0, 2, axis = 0)
        Js_T, Je_T = np.split(self.GTBrxdB.dot(self.TB.coeffs) / mu0, 2, axis = 0)

        Jth, Jph = Js_V + Js_T, Je_V + Je_T


        return(Jth, Jph)


    def get_Jr(self, _basis_evaluator, deg = False):
        """ Calculate radial current.

        """

        return _basis_evaluator.basis_to_grid(self.TB.coeffs * self.TB_to_Jr)


    def get_Je(self, _basis_evaluator, deg = False):
        """ Calculate eastward current.

        """

        return _basis_evaluator.basis_to_grid(-(self.TB.coeffs * self.TB_to_TJ), derivative = 'phi')


    def get_Jn(self, _basis_evaluator, deg = False):
        """ Calculate northward current.

        """

        return _basis_evaluator.basis_to_grid(self.TB.coeffs * self.TB_to_TJ, derivative = 'theta')


    def get_equivalent_current_function(self, grid, deg = False):
        """ Calculate equivalent current function.

        """
        print('not implemented')


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

        Eth = Eth + self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.br * Jph)
        Eph = Eph + self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.br * Jth)

        return(Eth, Eph)

