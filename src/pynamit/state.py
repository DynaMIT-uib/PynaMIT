import numpy as np
import scipy.sparse as spr
from pynamit.grid import grid
from pynamit.constants import mu0, RE

class state(object):
    """ State of the ionosphere.

    """

    def __init__(self, sha, mainfield, num_grid, RI, ignore_PFAC, FAC_integration_parameters, connect_hemispheres, latitude_boundary):
        """ Initialize the state of the ionosphere.
    
        """

        self.sha = sha
        self.mainfield = mainfield
        self.num_grid = num_grid
        self.FAC_integration_parameters = FAC_integration_parameters

        self.RI = RI
        self.ignore_PFAC = ignore_PFAC

        self.connect_hemispheres = connect_hemispheres
        self.latitude_boundary = latitude_boundary

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
            self.shc_TB_to_shc_PFAC = np.zeros((self.sha.Nshc, self.sha.Nshc))
        else: # Use the method by Engels and Olsen 1998, Eq. 13 to account for poloidal part of magnetic field for FACs
            self.shc_TB_to_shc_PFAC = self._get_PFAC_matrix(self.num_grid)

        self.GTB = self.get_GTB(self.num_grid)





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
            self.ll_grid = grid(RI, 90 - self.num_grid.theta[ll_mask], self.num_grid.lon[ll_mask])
            self.ll_grid.construct_G (self.sha)
            self.ll_grid.construct_dG(self.sha)
            self.c_u_theta, self.c_u_phi, self.A5_eP_V, self.A5_eH_V, self.A5_eP_T, self.A5_eH_T = self._get_A5_and_c(self.ll_grid)
            # ... and for their conjugate points:
            self.ll_theta_conj, self.ll_phi_conj = self.mainfield.conjugate_coordinates(self.ll_grid.RI, self.ll_grid.theta, self.ll_grid.lon)
            self.ll_grid_conj = grid(RI, 90 - self.ll_theta_conj, self.ll_phi_conj)
            self.ll_grid_conj.construct_G (self.sha)
            self.ll_grid_conj.construct_dG(self.sha)
            self.c_u_theta_conj, self.c_u_phi_conj, self.A5_eP_conj_V, self.A5_eH_conj_V, self.A5_eP_conj_T, self.A5_eH_conj_T = self._get_A5_and_c(self.ll_grid_conj)

            # calculate sin(inclination)
            self.ll_sinI = self.mainfield.get_sinI(self.ll_grid.RI, self.ll_grid.theta, self.ll_grid.lon).reshape((-1 ,1))
            self.ll_sinI_conj = self.mainfield.get_sinI(self.ll_grid_conj.RI, self.ll_grid_conj.theta, self.ll_grid_conj.lon).reshape((-1 ,1))

            # constraint matrix: FAC out of one hemisphere = FAC into the other
            self.G_par_ll_grid = self.ll_grid.G / mu0 * self.sha.n * (self.sha.n + 1) / self.ll_sinI
            self.G_par_ll_grid_conj = self.ll_grid_conj.G / mu0 * self.sha.n * (self.sha.n + 1) / self.ll_sinI_conj
            self.constraint_Gpar = self.G_par_ll_grid - self.G_par_ll_grid_conj


            # TODO: Should check for singularities - where field is horizontal - and probably just eliminate such points
            #       from the constraint calculation
            # TODO: We need a matrix for interpolation from the cubed sphere grid to the conjugate points. 
            #       This is needed to get values for conductance (and neutral wind u once that is included)
            #       The interpolation matrix should be calculated here on initiation since the grid is fixed




        # Initialize the spherical harmonic coefficients
        self.set_shc(VB = np.zeros(sha.Nshc))
        self.set_shc(TB = np.zeros(sha.Nshc))


    def _get_PFAC_matrix(self, _grid):
        """ """
        # initialize matrix that will map from self.TJr to coefficients for poloidal field:

        r_k_steps = self.FAC_integration_parameters['steps']
        Delta_k = np.diff(r_k_steps)
        r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

        _grid.construct_vector_to_shc_cf_df()

        jh_to_shc = -_grid.vector_to_shc_df * self.RI * mu0 # matrix to do SHA in Eq (7) in Engels and Olsen (inc. scaling)

        shc_TB_to_shc_PFAC = np.zeros((self.sha.Nshc, self.sha.Nshc))
        for i in range(r_k.size): # TODO: it would be useful to use Dask for this loop to speed things up a little
            print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
            # map coordinates from r_k[i] to RI:
            theta_mapped, phi_mapped = self.mainfield.map_coords(_grid.RI, r_k[i], _grid.theta, _grid.lon)
            mapped_grid = grid(self.RI, 90 - theta_mapped, phi_mapped)

            # Calculate magnetic field at grid points at r_k[i]:
            B_rk  = np.vstack(self.mainfield.get_B(r_k[i], _grid.theta, _grid.lon))
            B0_rk = np.linalg.norm(B_rk, axis = 0) # magnetic field magnitude
            b_rk = B_rk / B0_rk # unit vectors

            # Calculate magnetic field at the points in the ionosphere to which the grid maps:
            B_RI  = np.vstack(self.mainfield.get_B(mapped_grid.RI, mapped_grid.theta, mapped_grid.lon))
            B0_RI = np.linalg.norm(B_RI, axis = 0) # magnetic field magnitude
            sinI_RI = -B_RI[0] / B0_RI

            # find matrix that gets radial current at these coordinates:
            mapped_grid.construct_G(self.sha)

            # Calculate matrix that gives FAC from toroidal coefficients
            G_k = -mapped_grid.G * self.sha.n * (self.sha.n + 1) / self.RI / mu0 / sinI_RI.reshape((-1, 1)) # TODO: Handle singularity at equator (may be fine)

            # matrix that scales the FAC at RI to r_k and extracts the horizontal components:
            ratio = (B0_rk / B0_RI).reshape((1, -1))
            S_k = np.vstack((np.diag(b_rk[1]), np.diag(b_rk[2]))) * ratio

            # matrix that scales the terms by (R/r_k)**(n-1):
            A_k = np.diag((self.RI / r_k[i])**(self.sha.n - 1))

            # put it all together (crazy)
            shc_TB_to_shc_PFAC += Delta_k[i] * A_k.dot(jh_to_shc.dot(S_k.dot(G_k)))

        # return the matrix scaled by the term in front of the integral
        return(np.diag((self.sha.n + 1) / (2 * self.sha.n + 1)).dot(shc_TB_to_shc_PFAC) / self.RI)


    def _get_A5_and_c(self, _grid):
        """ Calculte A5 and c 
            

        """

        R, theta, phi = _grid.RI, _grid.theta, _grid.lon
        B = np.vstack(self.mainfield.get_B(R, theta, phi))
        B0 = np.linalg.norm(B, axis = 0)
        br, bt, bp = B / B0
        Br = B[0]
        d1, d2, _, _, _, _ = self.mainfield.basevectors(R, theta, phi)
        d1r, d1t, d1p = d1 # r theta phi components
        d2r, d2t, d2p = d2

        # The following lines of code are copied from output from sympy_matrix_algebra.py

        # constant that is to be multiplied by u in theta direction:
        c_ut_theta = Br*(bp*(bp*d1t - bt*d1p) + br*(br*d1t - bt*d1r))/(B0*br)
        c_ut_phi   = Br*(bp*(bp*d2t - bt*d2p) + br*(br*d2t - bt*d2r))/(B0*br)
        c_ut = np.hstack((c_ut_theta, c_ut_phi)).reshape((-1, 1)) # stack and convert to column vector

        # constant that is to be multiplied by u in phi direction:
        c_up_theta = Br*(br*(-bp*d1r + br*d1p) + bt*(-bp*d1t + bt*d1p))/(B0*br)
        c_up_phi   = Br*(br*(-bp*d2r + br*d2p) + bt*(-bp*d2t + bt*d2p))/(B0*br)
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

        # Get spherical harmonic matrices and multiply with the relevant geometry matrices
        Gph   = _grid.G_ph
        Gth   = _grid.G_th
        print('this is missing poloidal part of magnetic field...')
        G_DeltaB_th_V =  Gph / (self.sha.n + 1) / mu0
        G_DeltaB_ph_V =  Gth / (self.sha.n + 1) / mu0
        G_DeltaB_V    = np.vstack((G_DeltaB_th_V, G_DeltaB_ph_V))
        G_DeltaB_th_T = -Gth / (self.sha.n + 1) / mu0
        G_DeltaB_ph_T =  Gph / (self.sha.n + 1) / mu0
        G_DeltaB_T    = np.vstack((G_DeltaB_th_T, G_DeltaB_ph_T))

        A5_eP_V = a5eP.dot(G_DeltaB_V)
        A5_eH_V = a5eH.dot(G_DeltaB_V)
        A5_eP_T = a5eP.dot(G_DeltaB_T)
        A5_eH_T = a5eH.dot(G_DeltaB_T)

        return(c_ut, c_up, A5_eP_V, A5_eH_V, A5_eP_T, A5_eH_T)


    def get_GTB(self, _grid):
        """ Calculate matrix that maps the coefficients shc_TB to horizontal magnetic field above the ionosphere """
        GrxgradT = -_grid.Gdf * _grid.RI # matrix that gets -r x grad(T)
        GPFAC    = _grid.Gcf                      # matrix that calculates potential magnetic field of external source
        Gshield  = (_grid.Gcf / (self.sha.n + 1)) # matrix that calculates potential magnetic field of shielding current

        return(GrxgradT + (GPFAC + Gshield).dot(self.shc_TB_to_shc_PFAC))

    
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
            self.shc_VB = kwargs['VB']
            self.shc_VJ = self.RI / mu0 * (2 * self.sha.n + 1) / (self.sha.n + 1) * self.shc_VB
            self.shc_Br = 1 / self.sha.n * self.shc_VB
        elif key == 'TB':
            self.shc_TB = kwargs['TB']
            self.shc_TJ = -self.RI / mu0 * self.shc_TB
            self.shc_TJr = -self.sha.n * (self.sha.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TB_to_shc_PFAC.dot(self.shc_TB) 
        elif key == 'VJ':
            self.shc_VJ = kwargs['VJ']
            self.shc_VB = mu0 / self.RI * (self.sha.n + 1) / (2 * self.sha.n + 1) * self.shc_VJ
            self.shc_Br = 1 / self.sha.n * self.shc_VB
        elif key == 'TJ':
            self.shc_TJ = kwargs['TJ']
            self.shc_TB = -mu0 / self.RI * self.shc_TJ
            self.shc_TJr = -self.sha.n * (self.sha.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TB_to_shc_PFAC.dot(self.shc_TB) 
        elif key == 'Br':
            self.shc_Br = kwargs['Br']
            self.shc_VB = self.shc_Br / self.sha.n
            self.shc_VJ = -self.RI / mu0 * (2 * self.sha.n + 1) / (self.sha.n + 1) * self.shc_VB
        elif key == 'TJr':
            self.shc_TJr = kwargs['TJr']
            self.shc_TJ = -1 /(self.sha.n * (self.sha.n + 1)) * self.shc_TJr * self.RI**2
            self.shc_TB = -mu0 / self.RI * self.shc_TJ
            self.shc_PFAC = self.shc_TB_to_shc_PFAC.dot(self.shc_TB) 
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
        jr = -FAC * self.sinI 
        # Get the corresponding spherical harmonic coefficients
        TJr = np.linalg.lstsq(self.num_grid.GTG, self.num_grid.G.T.dot(jr), rcond = 1e-3)[0]
        # Propagate to the other coefficients (TB, TJ, PFAC):
        self.set_shc(TJr = TJr)

        if self.connect_hemispheres:
            print('connect_hemispheres is not fully implemented')

        print('Note: Check if rcond is really needed. It should not be necessary if the FAC is given sufficiently densely')
        print('Note to self: Remember to write a function that compares the AMPS SH coefficient to the ones derived here')


    def set_conductance(self, Hall, Pedersen):
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


    def update_shc_EW(self, grid):
        """ Update the coefficients for the induction electric field.

        """

        self.shc_EW = grid.vector_to_shc_df.dot(np.hstack(self.get_E(grid)))


    def update_shc_Phi(self, grid):
        """ Update the coefficients for the electric potential.

        """

        self.shc_Phi = grid.vector_to_shc_cf.dot(np.hstack(self.get_E(grid)))


    def evolve_Br(self, dt):
        """ Evolve ``Br`` in time.

        """

        #Eth, Eph = self.get_E(self.num_grid)
        #u1, u2, u3 = self.equations.sph_to_contravariant_cs(np.zeros_like(Eph), Eth, Eph)
        #curlEr = self.equations.curlr(u1, u2) 
        #Br = -self.GBr.dot(self.shc_VB) - dt * curlEr

        #self.set_shc(Br = self.GTG_inv.dot(self.num_grid.G.T.dot(-Br)))

        #GTE = self.Gcf.T.dot(np.hstack( self.get_E(self.num_grid)) )
        #self.shc_EW = self.GTGcf_inv.dot(GTE) # find coefficients for divergence-free / inductive E

        self.update_shc_EW(self.num_grid)
        new_shc_Br = self.shc_Br + self.sha.n * (self.sha.n + 1) * self.shc_EW * dt / self.RI**2
        self.set_shc(Br = new_shc_Br)


    def get_Br(self, grid, deg = False):
        """ Calculate ``Br``.

        """

        return(grid.G.dot(self.shc_Br))


    def get_JS(self, grid, deg = False):
        """ Calculate ionospheric sheet current.

        """

        Je_V =  grid.G_th.dot(self.shc_VJ) # r cross grad(VJ) eastward component
        Js_V = -grid.G_ph.dot(self.shc_VJ) # r cross grad(VJ) southward component
        Bs_T, Be_T = np.split(self.GTB.dot(self.shc_TB), 2, axis = 0)
        Js_T, Je_T = -Be_T/mu0, Bs_T/mu0


        #Je_T = -grid.G_ph.dot(self.shc_TJ) # -grad(VT) eastward component
        #Js_T = -grid.G_th.dot(self.shc_TJ) # -grad(VT) southward component

        Jth, Jph = Js_V + Js_T, Je_V + Je_T


        return(Jth, Jph)


    def get_Jr(self, grid, deg = False):
        """ Calculate radial current.

        """

        print('this must be fixed so that I can evaluate anywere')
        return grid.G.dot(self.shc_TJr)


    def get_equivalent_current_function(self, grid, deg = False):
        """ Calculate equivalent current function.

        """
        print('not implemented')


    def get_Phi(self, grid, deg = False):
        """ Calculate Phi.

        """

        print('this must be fixed so that Phi can be evaluated anywere')
        return grid.G.dot(self.shc_Phi) * 1e-3


    def get_W(self, grid, deg = False):
        """ Calculate the induction electric field scalar.

        """

        print('this must be fixed so that W can be evaluated anywere')
        return grid.G.dot(self.shc_EW) * 1e-3


    def get_E(self, grid, deg = False):
        """ Calculate electric field.

        """

        Jth, Jph = self.get_JS(grid)


        Eth = self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.br * Jph)
        Eph = self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.br * Jth)

        return(Eth, Eph)