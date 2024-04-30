import numpy as np
from pynamit.grid import grid

class state(object):
    """ State of the ionosphere.

    """

    def __init__(self, sha, mainfield, num_grid, mu0, RI, ignore_PNAF, FAC_integration_parameters, connect_hemispheres):
        """ Initialize the state of the ionosphere.
    
        """

        self.sha = sha
        self.mainfield = mainfield
        self.num_grid = num_grid

        self.mu0 = mu0
        self.RI = RI
        self.ignore_PNAF = ignore_PNAF

        self.connect_hemispheres = connect_hemispheres

        # get magnetic field unit vectors at CS grid:
        self.B = np.vstack(self.mainfield.get_B(self.RI, self.num_grid.theta, self.num_grid.lon))
        self.br, self.btheta, self.bphi = self.B / np.linalg.norm(self.B, axis = 0)
        self.sinI = -self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2) # sin(inclination)
        # construct the elements in the matrix in the electric field equation
        self.b00 = self.bphi**2 + self.br**2
        self.b01 = -self.btheta * self.bphi
        self.b10 = -self.btheta * self.bphi
        self.b11 = self.btheta**2 + self.br**2

        # Pre-calculate the matrix that maps from TJr_shc to coefficients for the poloidal magnetic field of FACs
        if self.mainfield.kind == 'radial' or self.ignore_PNAF: # no Poloidal field so get matrix of zeros
            self.shc_TJr_to_shc_PFAC = np.zeros((self.sha.Nshc, self.sha.Nshc))
        else: # Use the method by Engels and Olsen 1998, Eq. 13:
            r_k_steps = FAC_integration_parameters['steps']
            Delta_k = np.diff(r_k_steps)
            r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

            jh_to_shc = -self.num_grid.vector_to_shc_df * self.RI * mu0 # matrix to do SHA in Eq (7) in Engels and Olsen (inc. scaling)

            # initialize matrix that will map from self.TJr to coefficients for poloidal field:
            shc_TJr_to_shc_PFAC = np.zeros((self.sha.Nshc, self.sha.Nshc))
            for i in range(r_k.size): # TODO: it would be useful to use Dask for this loop to speed things up a little
                print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
                # map coordinates from r_k[i] to RI:
                theta_mapped, phi_mapped = self.mainfield.map_coords(self.num_grid.RI, r_k[i], self.num_grid.theta, self.num_grid.lon)
                mapped_grid = grid(self.RI, 90 - theta_mapped, phi_mapped)

                # Calculate magnetic field at grid points at r_k[i]:
                B_rk  = np.vstack(self.mainfield.get_B(r_k[i], self.num_grid.theta, self.num_grid.lon))
                B0_rk = np.linalg.norm(B_rk, axis = 0) # magnetic field magnitude
                b_rk = B_rk / B0_rk # unit vectors

                # Calculate magnetic field at the points in the ionosphere to which the grid maps:
                B_RI  = np.vstack(self.mainfield.get_B(mapped_grid.RI, mapped_grid.theta, mapped_grid.lon))
                B0_RI = np.linalg.norm(B_RI, axis = 0) # magnetic field magnitude
                sinI_RI = -B_RI[0] / B0_RI

                # find matrix that gets radial current at these coordinates:
                mapped_grid.construct_G(self.sha)

                # we need to scale this by -1/sin(inclination) to get the FAC:
                Q_k = -mapped_grid.G / sinI_RI.reshape((-1, 1)) # TODO: Handle singularity at equator (may be fine)

                # matrix that scales the FAC at RI to r_k and extracts the horizontal components:
                ratio = (B0_rk / B0_RI).reshape((1, -1))
                S_k = np.vstack((np.diag(b_rk[1]), np.diag(b_rk[2]))) * ratio

                # matrix that scales the terms by (R/r_k)**(n-1):
                A_k = np.diag((self.RI / r_k[i])**(self.sha.n - 1))

                # put it all together (crazy)
                shc_TJr_to_shc_PFAC += Delta_k[i] * A_k.dot(jh_to_shc.dot(S_k.dot(Q_k)))

            # finally scale the matrix by the term in front of the integral
            self.shc_TJr_to_shc_PFAC =  np.diag((self.sha.n + 1) / (2 * self.sha.n + 1)).dot(shc_TJr_to_shc_PFAC) / self.RI

            # make matrices that translate shc_PFAC to horizontal current density (assuming divergence-free shielding current)
            self.shc_PFAC_to_Jph = -  1 / (self.sha.n + 1) * self.num_grid.G_ph / mu0
            self.shc_PFAC_to_Jth =    1 / (self.sha.n + 1) * self.num_grid.G_th / mu0

        # Initialize the spherical harmonic coefficients
        self.set_shc(VB = np.zeros(sha.Nshc))
        self.set_shc(TB = np.zeros(sha.Nshc))

    
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
            self.shc_VJ = self.RI / self.mu0 * (2 * self.sha.n + 1) / (self.sha.n + 1) * self.shc_VB
            self.shc_Br = 1 / self.sha.n * self.shc_VB
        elif key == 'TB':
            self.shc_TB = kwargs['TB']
            self.shc_TJ = -self.RI / self.mu0 * self.shc_TB
            self.shc_TJr = -self.sha.n * (self.sha.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
        elif key == 'VJ':
            self.shc_VJ = kwargs['VJ']
            self.shc_VB = self.mu0 / self.RI * (self.sha.n + 1) / (2 * self.sha.n + 1) * self.shc_VJ
            self.shc_Br = 1 / self.sha.n * self.shc_VB
        elif key == 'TJ':
            self.shc_TJ = kwargs['TJ']
            self.shc_TB = -self.mu0 / self.RI * self.shc_TJ
            self.shc_TJr = -self.sha.n * (self.sha.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
        elif key == 'Br':
            self.shc_Br = kwargs['Br']
            self.shc_VB = self.shc_Br / self.sha.n
            self.shc_VJ = -self.RI / self.mu0 * (2 * self.sha.n + 1) / (self.sha.n + 1) * self.shc_VB
        elif key == 'TJr':
            self.shc_TJr = kwargs['TJr']
            self.shc_TJ = -1 /(self.sha.n * (self.sha.n + 1)) * self.shc_TJr * self.RI**2
            self.shc_TB = -self.mu0 / self.RI * self.shc_TJ
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
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
        Je_T = -grid.G_ph.dot(self.shc_TJ) # -grad(VT) eastward component
        Js_T = -grid.G_th.dot(self.shc_TJ) # -grad(VT) southward component

        Jth, Jph = Js_V + Js_T, Je_V + Je_T

        if not self.ignore_PNAF:
            Jth = Jth + self.shc_PFAC_to_Jth.dot(self.shc_PFAC)
            Jph = Jph + self.shc_PFAC_to_Jph.dot(self.shc_PFAC)

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