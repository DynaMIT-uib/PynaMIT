import numpy as np
import scipy.sparse as spr
from pynamit.grid import Grid
from pynamit.constants import mu0, RE
from pynamit.basis_evaluator import BasisEvaluator
from pynamit.cubedsphere.cubedsphere import csp
from pynamit.vector import Vector

DEBUG_constraint_scale = 1e-10 # to be deleted
DEBUG_jpar_scale = 1#1e15

class State(object):
    """ State of the ionosphere.

    """

    def __init__(self, sh, mainfield, num_grid, RI, ignore_PFAC, FAC_integration_parameters, connect_hemispheres, latitude_boundary):
        """ Initialize the state of the ionosphere.
    
        """

        self.sh = sh
        self.mainfield = mainfield
        self.num_grid = num_grid
        self.FAC_integration_parameters = FAC_integration_parameters

        self.RI = RI
        self.ignore_PFAC = ignore_PFAC

        self.connect_hemispheres = connect_hemispheres
        self.latitude_boundary = latitude_boundary

        # initialize the basis evaluator
        self.sh_evaluator = BasisEvaluator(sh, num_grid)

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

        # Pre-calculate the matrix that maps from shc_TB to the boundary magnetic field (Bh+)
        if self.mainfield.kind == 'radial' or self.ignore_PFAC: # no Poloidal field so get matrix of zeros
            self.shc_TB_to_shc_PFAC = np.zeros((self.sh.Nshc, self.sh.Nshc))
        else: # Use the method by Engels and Olsen 1998, Eq. 13 to account for poloidal part of magnetic field for FACs
            self.shc_TB_to_shc_PFAC = self._get_PFAC_matrix(num_grid, self.sh_evaluator)

        self.GTBrxdB = self.get_GTrxdB(num_grid, self.sh_evaluator) # matrices that map sch_TB to r x deltaB
        self.GVBrxdB = self.get_GVrxdB(num_grid, self.sh_evaluator) # matrices that map sch_VB to r x deltaB

        if connect_hemispheres:
            if ignore_PFAC:
                raise ValueError('Hemispheres can not be connected when ignore_PFAC is True')
            if self.mainfield.kind == 'radial':
                raise ValueError('Hemispheres can not be connected with radial magnetic field')

            # identify the low latitude points
            if self.mainfield.kind == 'dipole':
                ll_mask = np.abs(self.num_grid.lat) < self.latitude_boundary
            elif self.mainfield.kind == 'igrf':
                mlat, mlon = self.mainfield.apx.geo2apex(self.num_grid.lat, self.num_grid.lon, (self.num_grid.RI - RE)*1e-3)
                ll_mask = np.abs(mlat) < self.latitude_boundary
            else:
                print('this should not happen')

            # calculate constraint matrices for low latitude points
            self.ll_grid = Grid(RI, 90 - self.num_grid.theta[ll_mask], self.num_grid.lon[ll_mask])
            ll_sh_evaluator = BasisEvaluator(sh, self.ll_grid)
            self.c_u_theta, self.c_u_phi, self.A5_eP, self.A5_eH = self._get_A5_and_c(self.ll_grid)
            self.GTBrxdB_ll = self.get_GTrxdB(self.ll_grid, ll_sh_evaluator)
            self.GVBrxdB_ll = self.get_GVrxdB(self.ll_grid, ll_sh_evaluator)
            self.A_eP_T, self.A_eH_T = self.A5_eP.dot(self.GTBrxdB_ll) / mu0, self.A5_eH.dot(self.GTBrxdB_ll) / mu0
            self.A_eP_V, self.A_eH_V = self.A5_eP.dot(self.GVBrxdB_ll) / mu0, self.A5_eH.dot(self.GVBrxdB_ll) / mu0

            # ... and for their conjugate points:
            self.ll_theta_conj, self.ll_phi_conj = self.mainfield.conjugate_coordinates(self.ll_grid.RI, self.ll_grid.theta, self.ll_grid.lon)
            self.ll_grid_conj = Grid(RI, 90 - self.ll_theta_conj, self.ll_phi_conj)
            ll_conj_sh_evaluator = BasisEvaluator(sh, self.ll_grid_conj)
            self.c_u_theta_conj, self.c_u_phi_conj, self.A5_eP_conj, self.A5_eH_conj = self._get_A5_and_c(self.ll_grid_conj)
            self.GTBrxdB_ll_conj = self.get_GTrxdB(self.ll_grid_conj, ll_conj_sh_evaluator)
            self.GVBrxdB_ll_conj = self.get_GVrxdB(self.ll_grid_conj, ll_conj_sh_evaluator)
            self.A_eP_conj_T, self.A_eH_conj_T = self.A5_eP_conj.dot(self.GTBrxdB_ll_conj) / mu0, self.A5_eH_conj.dot(self.GTBrxdB_ll_conj) / mu0
            self.A_eP_conj_V, self.A_eH_conj_V = self.A5_eP_conj.dot(self.GVBrxdB_ll_conj) / mu0, self.A5_eH_conj.dot(self.GVBrxdB_ll_conj) / mu0


            # calculate sin(inclination)
            self.ll_sinI      = self.mainfield.get_sinI(self.ll_grid.RI     , self.ll_grid.theta     , self.ll_grid.lon     ).reshape((-1 ,1))
            self.ll_sinI_conj = self.mainfield.get_sinI(self.ll_grid_conj.RI, self.ll_grid_conj.theta, self.ll_grid_conj.lon).reshape((-1 ,1))

            # constraint matrix: FAC out of one hemisphere = FAC into the other
            self.G_par_ll        = ll_sh_evaluator.scaled_G(1 / self.RI / mu0 * self.sh.n * (self.sh.n + 1) / self.ll_sinI)
            self.G_par_ll_conj   = ll_conj_sh_evaluator.scaled_G(1 /self.RI / mu0 * self.sh.n * (self.sh.n + 1) / self.ll_sinI_conj)
            self.constraint_Gpar = (self.G_par_ll - self.G_par_ll_conj) * DEBUG_jpar_scale

 
            # TODO: Should check for singularities - where field is horizontal - and probably just eliminate such points
            #       from the constraint calculation
            # TODO: We need a matrix for interpolation from the cubed sphere grid to the conjugate points. 
            #       This is needed to get values for conductance (and neutral wind u once that is included)
            #       The interpolation matrix should be calculated here on initiation since the grid is fixed

        # Initialize neutral wind and conductances
        self.set_u(np.zeros(self.num_grid.size), np.zeros((self.num_grid.size)), update = False)
        self.set_conductance(np.zeros(self.num_grid.size), np.zeros((self.num_grid.size)), update = True)

        # Initialize the spherical harmonic coefficients
        self.set_shc(VB = np.zeros(sh.Nshc))
        self.set_shc(TB = np.zeros(sh.Nshc))

    def _get_PFAC_matrix(self, _grid, _sh_evaluator):
        """ """
        # initialize matrix that will map from self.TJr to coefficients for poloidal field:

        r_k_steps = self.FAC_integration_parameters['steps']
        Delta_k = np.diff(r_k_steps)
        r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

        jh_to_shc = -_sh_evaluator.vector_to_shc_df * self.RI * mu0 # matrix to do SHA in Eq (7) in Engels and Olsen (inc. scaling)

        shc_TB_to_shc_PFAC = np.zeros((self.sh.Nshc, self.sh.Nshc))
        for i in range(r_k.size): # TODO: it would be useful to use Dask for this loop to speed things up a little
            print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
            # map coordinates from r_k[i] to RI:
            theta_mapped, phi_mapped = self.mainfield.map_coords(_grid.RI, r_k[i], _grid.theta, _grid.lon)
            mapped_grid = Grid(self.RI, 90 - theta_mapped, phi_mapped)
            mapped_sh_evaluator = BasisEvaluator(self.sh, mapped_grid)

            # Calculate magnetic field at grid points at r_k[i]:
            B_rk  = np.vstack(self.mainfield.get_B(r_k[i], _grid.theta, _grid.lon))
            B0_rk = np.linalg.norm(B_rk, axis = 0) # magnetic field magnitude
            b_rk = B_rk / B0_rk # unit vectors

            # Calculate magnetic field at the points in the ionosphere to which the grid maps:
            B_RI  = np.vstack(self.mainfield.get_B(mapped_grid.RI, mapped_grid.theta, mapped_grid.lon))
            B0_RI = np.linalg.norm(B_RI, axis = 0) # magnetic field magnitude
            sinI_RI = -B_RI[0] / B0_RI

            # Calculate matrix that gives FAC from toroidal coefficients
            G_k = mapped_sh_evaluator.scaled_G(-self.sh.n * (self.sh.n + 1) / self.RI / mu0 / sinI_RI.reshape((-1, 1))) # TODO: Handle singularity at equator (may be fine)

            # matrix that scales the FAC at RI to r_k and extracts the horizontal components:
            ratio = (B0_rk / B0_RI).reshape((1, -1))
            S_k = np.vstack((np.diag(b_rk[1]), np.diag(b_rk[2]))) * ratio

            # matrix that scales the terms by (R/r_k)**(n-1):
            A_k = np.diag((self.RI / r_k[i])**(self.sh.n - 1))

            # put it all together (crazy)
            shc_TB_to_shc_PFAC += Delta_k[i] * A_k.dot(jh_to_shc.dot(S_k.dot(G_k)))

        # return the matrix scaled by the term in front of the integral
        return(np.diag((self.sh.n + 1) / (2 * self.sh.n + 1)).dot(shc_TB_to_shc_PFAC) / self.RI)


    def _get_A5_and_c(self, _grid):
        """ Calculte A5 and c 
            

        """

        R, theta, phi = _grid.RI, _grid.theta, _grid.lon
        B = np.vstack(self.mainfield.get_B(R, theta, phi))
        B0 = np.linalg.norm(B, axis = 0)
        br, bt, bp = B / B0
        Br = B[0]
        d1, d2, _, _, _, _ = self.mainfield.basevectors(R, theta, phi)
        d1p, d1n, d1r = d1 # e, n, u components
        d2p, d2n, d2r = d2
        d1t, d2t = -d1n, -d2n # north -> theta component

        # The following lines of code are copied from output from sympy_matrix_algebra.py

        # constant that is to be multiplied by u in theta direction:
        c_ut_theta = Br*(bp*(bp*d1t - bt*d1p) + br*(br*d1t - bt*d1r))/(B0*br)
        c_ut_phi   = Br*(bp*(bp*d2t - bt*d2p) + br*(br*d2t - bt*d2r))/(B0*br)
        c_ut = np.hstack((c_ut_theta, c_ut_phi)).reshape((-1, 1)) # stack and convert to column vector

        # constant that is to be multiplied by u in phi direction:
        c_up_theta = -Br*(br*(bp*d1r - br*d1p) + bt*(bp*d1t - bt*d1p))/(B0*br)
        c_up_phi   = -Br*(br*(bp*d2r - br*d2p) + bt*(bp*d2t - bt*d2p))/(B0*br)
        c_up = np.hstack((c_up_theta, c_up_phi)).reshape((-1, 1)) # stack and convert to column vector


        # A5eP matrix
        a5eP00 = spr.diags((bp**3*d1r - bp**2*br*d1p + bp*br**2*d1r + bp*bt**2*d1r - br**3*d1p - br*bt**2*d1p)/B0)
        a5eP01 = spr.diags((bp**2*br*d1t - bp**2*bt*d1r + br**3*d1t - br**2*bt*d1r + br*bt**2*d1t - bt**3*d1r)/B0)
        a5eP10 = spr.diags((bp**3*d2r - bp**2*br*d2p + bp*br**2*d2r + bp*bt**2*d2r - br**3*d2p - br*bt**2*d2p)/B0)
        a5eP11 = spr.diags((bp**2*br*d2t - bp**2*bt*d2r + br**3*d2t - br**2*bt*d2r + br*bt**2*d2t - bt**3*d2r)/B0)
        a5eP = spr.vstack((spr.hstack((a5eP00, a5eP01)), spr.hstack((a5eP10, a5eP11)))).tocsr()

        # A5eH matrix
        a5eH00 = spr.diags((-bp**2*d1t + bp*bt*d1p - br**2*d1t + br*bt*d1r)/B0)
        a5eH01 = spr.diags((bp*br*d1r + bp*bt*d1t - br**2*d1p - bt**2*d1p)/B0)
        a5eH10 = spr.diags((-bp**2*d2t + bp*bt*d2p - br**2*d2t + br*bt*d2r)/B0)
        a5eH11 = spr.diags((bp*br*d2r + bp*bt*d2t - br**2*d2p - bt**2*d2p)/B0)
        a5eH = spr.vstack((spr.hstack((a5eH00, a5eH01)), spr.hstack((a5eH10, a5eH11)))).tocsr()

        return(c_ut, c_up, a5eP, a5eH)


    def update_constraints(self):
        """ update the constraint arrays c and A - should be called when changing u and eta """
        self.cu =  (self.u_phi_ll_conj * self.c_u_phi_conj + self.u_theta_ll_conj * self.c_u_theta_conj) \
                  -(self.u_phi_ll * self.c_u_phi + self.u_theta_ll * self.c_u_theta)
        self.cu = self.cu.flatten()
        self.AT  =  (self.etaP_ll * self.A_eP_T + self.etaH_ll * self.A_eH_T) \
                   -(self.etaP_ll_conj * self.A_eP_conj_T + self.etaH_ll_conj * self.A_eH_conj_T)
        self.AV  =  (self.etaP_ll * self.A_eP_V + self.etaH_ll * self.A_eH_V) \
                   -(self.etaP_ll_conj * self.A_eP_conj_V + self.etaH_ll_conj * self.A_eH_conj_V)
        self.AV = -self.AV




    def get_GTrxdB(self, _grid, _sh_evaluator):
        """ Calculate matrix that maps the coefficients shc_TB to delta B across ionosphere """
        GrxgradT = -_sh_evaluator.Gdf * _grid.RI # matrix that gets -r x grad(T)
        GPFAC    = -_sh_evaluator.Gcf                      # matrix that calculates potential magnetic field of external source
        Gshield  = -(_sh_evaluator.Gcf / (self.sh.n + 1)) # matrix that calculates potential magnetic field of shielding current

        GTB = GrxgradT + (GPFAC + Gshield).dot(self.shc_TB_to_shc_PFAC)
        GTBth, GTBph = np.split(GTB, 2, axis = 0)
        GTrxdB = np.vstack((-GTBph, GTBth))
        return(GTrxdB)


    def get_GVrxdB(self, _grid, _sh_evaluator):
        """ Calculate matrix that maps the coefficients shc_VB to delta B across ionosphere """
        n = _sh_evaluator.basis.n
        GVdB = _sh_evaluator.Gcf * (n / (n + 1) + 1) * _grid.RI
        GVBth, GVBph = np.split(GVdB, 2, axis = 0)
        GVrxdB = np.vstack((-GVBph, GVBth))

        return(GVrxdB)
    

    def set_shc(self, **kwargs):
        """ Set spherical harmonic coefficients.

        Specify a set of spherical harmonic coefficients and update the
        rest so that they are consistent. 

        This function accepts one (and only one) set of spherical harmonic
        coefficients. Valid values for kwargs (only one):

        - 'VB' : Coefficients for magnetic field scalar ``V``.
        - 'TB' : Coefficients for surface current scalar ``T``.
        - 'VJ' : Coefficients for magnetic field scalar ``V``.
        - 'TJ' : Coefficients for surface current scalar ``T``.
        - 'Br' : Coefficients for magnetic field ``Br`` (at ``r = RI``).
        - 'TJr': Coefficients for radial current scalar.

        """

        valid_kws = ['VB', 'TB', 'VJ', 'TJ', 'Br', 'TJr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'VB':
            self.shc_VB = Vector(self.sh, kwargs['VB'])
            self.shc_VJ = Vector(self.sh, self.RI / mu0 * (2 * self.sh.n + 1) / (self.sh.n + 1) * self.shc_VB.coeffs)
            self.shc_Br = Vector(self.sh, 1 / self.sh.n * self.shc_VB.coeffs)
        elif key == 'TB':
            self.shc_TB = Vector(self.sh, kwargs['TB'])
            self.shc_TJ = Vector(self.sh, -self.RI / mu0 * self.shc_TB.coeffs)
            self.shc_TJr = Vector(self.sh, -self.sh.n * (self.sh.n + 1) / self.RI**2 * self.shc_TJ.coeffs)
            self.shc_PFAC = Vector(self.sh, self.shc_TB_to_shc_PFAC.dot(self.shc_TB.coeffs))
        elif key == 'VJ':
            self.shc_VJ = Vector(self.sh, kwargs['VJ'])
            self.shc_VB = Vector(self.sh, mu0 / self.RI * (self.sh.n + 1) / (2 * self.sh.n + 1) * self.shc_VJ.coeffs)
            self.shc_Br = Vector(self.sh, 1 / self.sh.n * self.shc_VB.coeffs)
        elif key == 'TJ':
            self.shc_TJ = Vector(self.sh, kwargs['TJ'])
            self.shc_TB = Vector(self.sh, -mu0 / self.RI * self.shc_TJ.coeffs)
            self.shc_TJr = Vector(self.sh, -self.sh.n * (self.sh.n + 1) / self.RI**2 * self.shc_TJ.coeffs)
            self.shc_PFAC = Vector(self.sh, self.shc_TB_to_shc_PFAC.dot(self.shc_TB.coeffs))
        elif key == 'Br':
            self.shc_Br = Vector(self.sh, kwargs['Br'])
            self.shc_VB = Vector(self.sh, self.shc_Br.coeffs / self.sh.n)
            self.shc_VJ = Vector(self.sh, -self.RI / mu0 * (2 * self.sh.n + 1) / (self.sh.n + 1) * self.shc_VB.coeffs)
        elif key == 'TJr':
            self.shc_TJr = kwargs['TJr']
            self.shc_TJ = Vector(self.sh, -1 /(self.sh.n * (self.sh.n + 1)) * self.shc_TJr * self.RI**2)
            self.shc_TB = Vector(self.sh, -mu0 / self.RI * self.shc_TJ.coeffs)
            self.shc_PFAC = Vector(self.sh, self.shc_TB_to_shc_PFAC.dot(self.shc_TB.coeffs))
            print('check the factor RI**2!')
        else:
            raise Exception('This should not happen')


    def set_initial_condition(self):
        """ Set initial conditions.

        If this is not called, initial conditions should be zero.

        """

        print('not implemented. inital conditions will be zero')


    def set_FAC(self, FAC):
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

        # Extract the radial component of the FAC:
        self.jr = -FAC * self.sinI 
        # Get the corresponding spherical harmonic coefficients
        TJr = np.linalg.lstsq(self.sh_evaluator.GTG, self.sh_evaluator.from_grid(self.jr), rcond = 1e-3)[0]
        # Propagate to the other coefficients (TB, TJ, PFAC):
        self.set_shc(TJr = TJr)

        if self.connect_hemispheres:

            # mask the jr so that it only applies poleward of self.latitude_boundary
            hl_mask = np.abs(self.num_grid.lat) > self.latitude_boundary
            self.hl_grid = Grid(self.RI, 90 - self.num_grid.theta[hl_mask], self.num_grid.lon[hl_mask])
            hl_sh_evaluator = BasisEvaluator(self.sh, self.hl_grid)

            B = np.vstack(self.mainfield.get_B(self.RI, self.hl_grid.theta, self.hl_grid.lon))
            br, btheta, bphi = B / np.linalg.norm(B, axis = 0)
            #hl_sinI = -br / np.sqrt(btheta**2 + bphi**2 + br**2) # sin(inclination)

            self.Gjr = hl_sh_evaluator.scaled_G(1 / mu0 * self.sh.n * (self.sh.n + 1) / self.RI)
            self.jr = self.jr[hl_mask]

            # combine matrices:
            self.G_shc_TB = np.vstack((self.Gjr, self.constraint_Gpar, self.AT * DEBUG_constraint_scale ))
            #print('inverting')

            self.Gpinv = np.linalg.pinv(self.G_shc_TB, rcond = 1e-3)
            print('rcond!')

            c = (self.cu.flatten() + self.AV.dot(self.shc_VB.coeffs))
            
            self.ccc = c
            d = np.hstack((self.jr.flatten(), np.zeros(self.constraint_Gpar.shape[0]), c.flatten() * DEBUG_constraint_scale ))
            self.set_shc(TB = self.Gpinv.dot(d))
            self.ggg = self.G_shc_TB

            #print('connect_hemispheres is not fully implemented')

        print('Note: Check if rcond is really needed. It should not be necessary if the FAC is given sufficiently densely')
        print('Note to self: Remember to write a function that compares the AMPS SH coefficient to the ones derived here')


    def set_u(self, u_theta, u_phi, update = True):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """
        self.u_theta = u_theta
        self.u_phi = u_phi

        self.uxB_theta =  self.u_phi   * self.B[0] 
        self.uxB_phi   = -self.u_theta * self.B[0] 

        if self.connect_hemispheres:
            # find wind field at low lat grid points
            u_ll = csp.interpolate_vector_components(u_phi, -u_theta, np.ones_like(u_phi), self.num_grid.theta, self.num_grid.lon, self.ll_grid.theta, self.ll_grid.lon)
            u_ll = np.tile(u_ll, (1, 2)) # duplicate 
            self.u_theta_ll, self.u_phi_ll = -u_ll[1].reshape((-1, 1)), u_ll[0].reshape((-1, 1))

            # find wind field at conjugate grid points
            u_ll_conj = csp.interpolate_vector_components(u_phi, -u_theta, np.ones_like(u_phi), self.num_grid.theta, self.num_grid.lon, self.ll_grid_conj.theta, self.ll_grid_conj.lon)
            u_ll_conj = np.tile(u_ll_conj, (1, 2)) # duplicate 
            self.u_theta_ll_conj, self.u_phi_ll_conj = -u_ll_conj[1].reshape((-1, 1)), u_ll_conj[0].reshape((-1, 1))

            if update:
                self.update_constraints()


    def set_conductance(self, Hall, Pedersen, update = True):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.lon``.

        """

        if Hall.size != Pedersen.size != self.num_grid.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.SH = Hall
        self.SP = Pedersen
        self.etaP = Pedersen / (Hall**2 + Pedersen**2)
        self.etaH = Hall     / (Hall**2 + Pedersen**2)

        if self.connect_hemispheres:
            # TODO: 1) csp structure is strange. 2) This is inefficient when eta is updated often
            # find resistances at low lat grid points
            self.etaP_ll      = np.tile(csp.interpolate_scalar(self.etaP, self.num_grid.theta, self.num_grid.lon, self.ll_grid.theta, self.ll_grid.lon), 2).reshape((-1, 1))
            self.etaH_ll      = np.tile(csp.interpolate_scalar(self.etaH, self.num_grid.theta, self.num_grid.lon, self.ll_grid.theta, self.ll_grid.lon), 2).reshape((-1, 1))

            # find resistances at conjugate grid points
            self.etaP_ll_conj = np.tile(csp.interpolate_scalar(self.etaP, self.num_grid.theta, self.num_grid.lon, self.ll_grid_conj.theta, self.ll_grid_conj.lon), 2).reshape((-1, 1))
            self.etaH_ll_conj = np.tile(csp.interpolate_scalar(self.etaH, self.num_grid.theta, self.num_grid.lon, self.ll_grid_conj.theta, self.ll_grid_conj.lon), 2).reshape((-1, 1))

            if update:
                self.update_constraints()            


    def update_shc_EW(self):
        """ Update the coefficients for the induction electric field.

        """

        self.shc_EW = Vector(self.sh, basis_evaluator = self.sh_evaluator, grid_values = np.hstack(self.get_E(self.sh_evaluator)), component='df')


    def update_shc_Phi(self):
        """ Update the coefficients for the electric potential.

        """

        self.shc_Phi = Vector(self.sh, basis_evaluator = self.sh_evaluator, grid_values = np.hstack(self.get_E(self.sh_evaluator)), component='cf')


    def evolve_Br(self, dt):
        """ Evolve ``Br`` in time.

        """

        #Eth, Eph = self.get_E(self.num_grid)
        #u1, u2, u3 = self.equations.sph_to_contravariant_cs(np.zeros_like(Eph), Eth, Eph)
        #curlEr = self.equations.curlr(u1, u2) 
        #Br = -self.GBr.dot(self.shc_VB) - dt * curlEr

        #self.set_shc(Br = self.GTG_inv.dot(self.sh_evaluator.from_grid(-Br)))

        #GTE = self.Gcf.T.dot(np.hstack( self.get_E(self.num_grid)) )
        #self.shc_EW = self.GTGcf_inv.dot(GTE) # find coefficients for divergence-free / inductive E

        if self.connect_hemispheres:
            c = (self.cu + self.AV.dot(self.shc_VB.coeffs))
            d = np.hstack((self.jr.flatten(), np.zeros(self.constraint_Gpar.shape[0]), c.flatten() * DEBUG_constraint_scale))
            self.set_shc(TB = self.Gpinv.dot(d))

        self.update_shc_EW()
        new_shc_Br = self.shc_Br.coeffs + self.sh.n * (self.sh.n + 1) * self.shc_EW.coeffs * dt / self.RI**2
        self.set_shc(Br = new_shc_Br)


    def get_Br(self, _sh_evaluator, deg = False):
        """ Calculate ``Br``.

        """

        return(_sh_evaluator.to_grid(self.shc_Br.coeffs))


    def get_JS(self, _sh_evaluator, deg = False):
        """ Calculate ionospheric sheet current.

        """
        Js_V, Je_V = np.split(self.GVBrxdB.dot(self.shc_VB.coeffs) / mu0, 2, axis = 0)
        Js_T, Je_T = np.split(self.GTBrxdB.dot(self.shc_TB.coeffs) / mu0, 2, axis = 0)

        Jth, Jph = Js_V + Js_T, Je_V + Je_T


        return(Jth, Jph)


    def get_Jr(self, _sh_evaluator, deg = False):
        """ Calculate radial current.

        """

        return _sh_evaluator.to_grid(self.shc_TJr.coeffs)


    def get_Je(self, _basis_evaluator, deg = False):
        """ Calculate eastward current.

        """

        return _basis_evaluator.to_grid(-self.shc_TJ.coeffs, derivative = 'phi')


    def get_Jn(self, _basis_evaluator, deg = False):
        """ Calculate northward current.

        """

        return _basis_evaluator.to_grid(self.shc_TJ.coeffs, derivative = 'theta')


    def get_equivalent_current_function(self, grid, deg = False):
        """ Calculate equivalent current function.

        """
        print('not implemented')


    def get_Phi(self, _sh_evaluator, deg = False):
        """ Calculate Phi.

        """

        return _sh_evaluator.to_grid(self.shc_Phi.coeffs)


    def get_W(self, _sh_evaluator, deg = False):
        """ Calculate the induction electric field scalar.

        """

        return _sh_evaluator.to_grid(self.shc_EW.coeffs)


    def get_E(self, _sh_evaluator, deg = False):
        """ Calculate electric field.

        """
        if self.u_theta is not None:
            Eth = -self.uxB_theta
            Eph = -self.uxB_phi
        else:
            Eth = 0
            Eph = 0

        Jth, Jph = self.get_JS(_sh_evaluator, deg = deg)

        Eth = Eth + self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.br * Jph)
        Eph = Eph + self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.br * Jth)

        return(Eth, Eph)

